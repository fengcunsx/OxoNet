"""P3: 防泄漏切分 + 组织成 per-5mer 目录(训练用)。流式版(RAM有上限,稳60G)。
- oligo: per-batch 按 f5 切 90:5:5 (整个f5同侧,不泄漏)
- T2T:   按 part 切 90:5:5 (read不跨part,不泄漏), 补 label=0
- 每个中心5-mer一个npz; pos->8oxog_<split>/; neg=oligo+T2T合并shuffle->g_<split>/; g_oligo_<split>/(E1)
- native 不参与
流式: 逐文件读, 按ctx攒到内存上限(--ram-gb)就 flush 成分片, 最后逐ctx合并; RAM 只占上限, 不全攒。
schema: signal,label,read_id,kmers,mean,std,dwell,basecall_pos
用法(冒烟): python dataset/build_split.py --out /tmp/split_smoke --smoke 3
用法(全量): python dataset/build_split.py --out <disk>/split --ram-gb 8
"""
import argparse, glob, json, os, random, shutil, tempfile
from collections import defaultdict
import numpy as np

OLIGO_ROOT = '/home/bio/8oxog/data'   # 可用 --oligo-root 覆盖
T2T_ROOT = '/home/bio/8oxog/t2t'      # 可用 --t2t-root 覆盖
BATCHES = ['batch1', 'batch2', 'batch3', 'batch4']
# 三源相对结构不同(288/3439=<f5>/output_new/oxo; new=output/<f5>/output/oxo)
T2T_SUBPATS = {
    '288548394': '288548394/*/output_new/oxo/*.npz',
    '3439856925': '3439856925/*/output_new/oxo/*.npz',
    'new': 'new/output/*/output/oxo/*.npz',
}
KEYS = ['signal', 'label', 'read_id', 'kmers', 'mean', 'std', 'dwell', 'basecall_pos']
BYTES_PER_ROW = 1700  # 粗估: signal175*8 + 元数据


def c5(kmers):
    return np.array([s[1:6] for s in kmers])


def assign_splits_oligo(seed):
    rng = random.Random(seed); out = {}
    for b in BATCHES:
        f5s = [d for d in sorted(glob.glob(f'{OLIGO_ROOT}/{b}/{b}_work/fast5s/*')) if os.path.isdir(d)]
        rng.shuffle(f5s)
        n = len(f5s); nte = max(1, round(n * .05)); nva = max(1, round(n * .05))
        for i, d in enumerate(f5s):
            out[d] = 'test' if i < nte else ('valid' if i < nte + nva else 'train')
    return out


def assign_splits_t2t(seed):
    rng = random.Random(seed + 1); out = {}
    for s, sub in T2T_SUBPATS.items():
        parts = sorted(glob.glob(os.path.join(T2T_ROOT, sub))); rng.shuffle(parts)
        n = len(parts); nte = max(1, round(n * .05)); nva = max(1, round(n * .05))
        for i, p in enumerate(parts):
            out[p] = 'test' if i < nte else ('valid' if i < nte + nva else 'train')
    return out


def f32(x):
    """float64->float32(下游全 .float(),无损;临时分片/最终npz都减半)。"""
    return x.astype(np.float32) if x.dtype == np.float64 else x


def row_data(a, is_pos):
    """从一个npz取标准字段(补label/reshape basecall_pos)。signal/mean/std 降 float32。"""
    n = a['kmers'].shape[0]
    return {
        'signal': f32(a['signal']), 'kmers': a['kmers'], 'mean': f32(a['mean']), 'std': f32(a['std']),
        'dwell': a['dwell'], 'read_id': a['read_id'],
        'basecall_pos': a['basecall_pos'].reshape(-1, 1),
        'label': a['label'] if 'label' in a.files else np.full(n, 1 if is_pos else 0, dtype=np.int64),
    }


def build_source_streaming(paths_by_split, is_pos, out_root, out_prefix, ram_gb, smoke):
    """流式: 逐文件->按ctx攒->超RAM上限flush分片->逐ctx合并写最终per-ctx npz。"""
    ram_bytes = ram_gb * 1e9
    for split, paths in paths_by_split.items():
        if smoke:
            paths = paths[:smoke]
        tmp = tempfile.mkdtemp(prefix=f'bs_{out_prefix}_{split}_')
        buffers = defaultdict(lambda: defaultdict(list)); used = [0]; shard = [0]

        def flush():
            for ctx, d in buffers.items():
                if not d['label']:
                    continue
                cd = os.path.join(tmp, ctx); os.makedirs(cd, exist_ok=True)
                merged = {k: np.concatenate(d[k], axis=0) for k in KEYS}
                np.savez_compressed(os.path.join(cd, f'{shard[0]}.npz'), **merged)  # 压缩:临时占用~4x小,防满盘
            buffers.clear(); used[0] = 0; shard[0] += 1

        for p in paths:
            a = np.load(p)
            if a['kmers'].shape[0] == 0:
                continue
            data = row_data(a, is_pos); ctxs = c5(a['kmers'])
            for ctx in np.unique(ctxs):
                m = ctxs == ctx
                for k in KEYS:
                    buffers[str(ctx)][k].append(data[k][m])
                used[0] += int(m.sum()) * BYTES_PER_ROW
            if used[0] > ram_bytes:
                flush()
        flush()

        outdir = os.path.join(out_root, f'{out_prefix}_{split}'); os.makedirs(outdir, exist_ok=True)
        for ctx in sorted(os.listdir(tmp)):
            shards = sorted(glob.glob(os.path.join(tmp, ctx, '*.npz')))
            merged = {k: np.concatenate([np.load(s)[k] for s in shards], axis=0) for k in KEYS}
            np.savez_compressed(os.path.join(outdir, f'{ctx}.npz'), **merged)
        shutil.rmtree(tmp)
        print(f'  {out_prefix}_{split}: {len(os.listdir(outdir))} ctx')


def merge_shuffle_neg(oligo_dir, t2t_dir, out_dir, seed):
    """per-ctx 合并 oligo+T2T neg 并 shuffle -> out_dir (每ctx RAM小)。"""
    os.makedirs(out_dir, exist_ok=True)
    ctxs = set(os.path.basename(f)[:-4] for f in glob.glob(f'{oligo_dir}/*.npz'))
    ctxs |= set(os.path.basename(f)[:-4] for f in glob.glob(f'{t2t_dir}/*.npz'))
    for ctx in sorted(ctxs):
        parts = [np.load(os.path.join(d, f'{ctx}.npz')) for d in (oligo_dir, t2t_dir)
                 if os.path.isfile(os.path.join(d, f'{ctx}.npz'))]
        merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in KEYS}
        idx = np.arange(merged['label'].shape[0]); np.random.RandomState(seed).shuffle(idx)
        np.savez_compressed(os.path.join(out_dir, f'{ctx}.npz'), **{k: v[idx] for k, v in merged.items()})


def main():
    global OLIGO_ROOT, T2T_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ram-gb', type=float, default=8.0, help='累积内存上限(GB), 超了就flush分片')
    ap.add_argument('--smoke', type=int, default=0, help='>0: 每split只取前N个f5/part')
    ap.add_argument('--oligo-root', default=OLIGO_ROOT, help='oligo data 根(含 batchN/)')
    ap.add_argument('--t2t-root', default=T2T_ROOT, help='t2t 根(含 288548394/3439856925/new)')
    args = ap.parse_args()
    OLIGO_ROOT, T2T_ROOT = args.oligo_root, args.t2t_root

    oligo_split = assign_splits_oligo(args.seed)
    t2t_split = assign_splits_t2t(args.seed)
    print(f"oligo f5 train/valid/test: {sum(v=='train' for v in oligo_split.values())}/"
          f"{sum(v=='valid' for v in oligo_split.values())}/{sum(v=='test' for v in oligo_split.values())}")
    print(f"T2T part train/valid/test: {sum(v=='train' for v in t2t_split.values())}/"
          f"{sum(v=='valid' for v in t2t_split.values())}/{sum(v=='test' for v in t2t_split.values())}")

    # 持久化 f5/part -> split 清单(可复现 + esox 对齐; 相对根存, 换机也一致)
    os.makedirs(args.out, exist_ok=True)
    manifest = {
        'seed': args.seed, 'oligo_root': OLIGO_ROOT, 't2t_root': T2T_ROOT,
        'oligo': {os.path.relpath(k, OLIGO_ROOT): v for k, v in sorted(oligo_split.items())},
        't2t': {os.path.relpath(k, T2T_ROOT): v for k, v in sorted(t2t_split.items())},
    }
    with open(os.path.join(args.out, 'split_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f"split_manifest.json 写出 ({len(oligo_split)} oligo f5 + {len(t2t_split)} t2t part)")

    def by_split(assign, subpath=None):
        d = {'train': [], 'valid': [], 'test': []}
        for k, sp in assign.items():
            p = os.path.join(k, subpath) if subpath else k
            if os.path.isfile(p):
                d[sp].append(p)
        return d

    print("pos ..."); build_source_streaming(by_split(oligo_split, 'feature/oxo/8oxog/data.npz'),
                                              True, args.out, '8oxog', args.ram_gb, args.smoke)
    print("oligo neg ..."); build_source_streaming(by_split(oligo_split, 'feature/oxo/g/data.npz'),
                                                    False, args.out, 'g_oligo', args.ram_gb, args.smoke)
    print("T2T neg ..."); build_source_streaming(by_split(t2t_split), False, args.out, 'g_t2t', args.ram_gb, args.smoke)

    print("merge+shuffle neg (E2) ...")
    for sp in ('train', 'valid', 'test'):
        merge_shuffle_neg(os.path.join(args.out, f'g_oligo_{sp}'),
                          os.path.join(args.out, f'g_t2t_{sp}'),
                          os.path.join(args.out, f'g_{sp}'), args.seed)
    print("done ->", args.out)


if __name__ == '__main__':
    main()
