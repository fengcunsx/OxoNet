"""把 build_split 的 per-context 目录打包成"训练直载"格式(CPU机产, 传GPU机):
- pos_train.npz : 全部train pos, 按ctx排序拼接(与neg按ctx对齐) + ctx_counts.json(每ctx的pos数 l_ctx)
- neg_pack/<ctx>/<field>.npy : 每ctx的neg池, 未压缩(供mmap轮转), 字段: signal/mean/std/dwell/kmers_enc/label
    (kmers 存成编码后的 int8 数组, 省空间且免运行时编码)
- valid.npz / test_oligo.npz / test_t2t.npz : 评估用(压缩, 一次性载入), 保留 read_id/basecall_pos
GPU机训练时: pos_train 常驻RAM, neg_pack mmap, 每epoch按 l_ctx 轮转取切片(见 EpochPairDataset)。

用法: python dataset/pack_train.py --split-dir <build_split输出> --out <pack输出>
"""
import argparse, glob, json, os
import numpy as np

BASE = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'o': 2}
NEG_FIELDS = ['signal', 'mean', 'std', 'dwell', 'kmers_enc', 'label']


def enc_kmers(kmers):
    return np.stack([np.fromiter((BASE[c] for c in s), dtype=np.int8, count=len(s)) for s in kmers])


def f32(x):
    """float64 -> float32(训练本就.float(),无损;pack/传输减半)。int/str 原样。"""
    return x.astype(np.float32) if x.dtype == np.float64 else x


def ctx_list(d):
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{d}/*.npz'))


def pack_pos(pos_dir, out):
    ctxs = ctx_list(pos_dir)
    keys = ['signal', 'mean', 'std', 'dwell', 'label', 'read_id', 'basecall_pos']
    agg = {k: [] for k in keys}; agg['kmers_enc'] = []; counts = {}
    for ctx in ctxs:
        a = np.load(os.path.join(pos_dir, f'{ctx}.npz'))
        counts[ctx] = int(a['label'].shape[0])
        for k in keys:
            agg[k].append(a['basecall_pos'].reshape(-1, 1) if k == 'basecall_pos' else a[k])
        agg['kmers_enc'].append(enc_kmers(a['kmers']))
    merged = {k: f32(np.concatenate(v, axis=0)) for k, v in agg.items()}
    np.savez_compressed(out + '.npz', **merged)
    json.dump({'ctxs': ctxs, 'counts': counts}, open(out + '_ctx.json', 'w'))
    print(f"pos_train: {merged['label'].shape[0]} samples, {len(ctxs)} ctx -> {out}.npz")


def pack_neg(neg_dir, out_dir):
    """每ctx写未压缩.npy(供mmap)。"""
    os.makedirs(out_dir, exist_ok=True)
    for ctx in ctx_list(neg_dir):
        a = np.load(os.path.join(neg_dir, f'{ctx}.npz'))
        cd = os.path.join(out_dir, ctx); os.makedirs(cd, exist_ok=True)
        np.save(os.path.join(cd, 'signal.npy'), f32(a['signal']))  # float32: mmap减半(41G vs 79G)
        np.save(os.path.join(cd, 'mean.npy'), f32(a['mean']))
        np.save(os.path.join(cd, 'std.npy'), f32(a['std']))
        np.save(os.path.join(cd, 'dwell.npy'), a['dwell'])
        np.save(os.path.join(cd, 'kmers_enc.npy'), enc_kmers(a['kmers']))
        np.save(os.path.join(cd, 'label.npy'), a['label'])
    print(f"neg_pack: {len(ctx_list(neg_dir))} ctx -> {out_dir}")


def combine(pos_dir, neg_dirs, out, balance_seed=None):
    """评估集: pos + 各neg目录 合并成一个压缩npz(带read_id/basecall_pos)。
    balance_seed!=None 时: neg 下采样到 = pos 数(1:1平衡, 用于valid的每epoch监控, 又快又不失真)。"""
    keys = ['signal', 'mean', 'std', 'dwell', 'kmers', 'label', 'read_id', 'basecall_pos']

    def load_dir(d):
        agg = {k: [] for k in keys}
        for ctx in sorted(ctx_list(d)):
            a = np.load(os.path.join(d, f'{ctx}.npz'))
            for k in keys:
                agg[k].append(a['basecall_pos'].reshape(-1, 1) if k == 'basecall_pos' else a[k])
        return {k: np.concatenate(v, axis=0) for k, v in agg.items()} if agg[keys[0]] else None

    pos = load_dir(pos_dir)
    negs = [load_dir(d) for d in neg_dirs]
    negs = [n for n in negs if n is not None]
    neg = {k: np.concatenate([n[k] for n in negs], axis=0) for k in keys}
    if balance_seed is not None:
        npos = pos['label'].shape[0]; nn = neg['label'].shape[0]
        idx = np.arange(nn); np.random.RandomState(balance_seed).shuffle(idx)
        idx = idx[:npos]                      # neg 下采样到 pos 数
        neg = {k: v[idx] for k, v in neg.items()}
    merged = {k: f32(np.concatenate([pos[k], neg[k]], axis=0)) for k in keys}
    np.savez_compressed(out, **merged)
    u, c = np.unique(merged['label'], return_counts=True)
    print(f"{os.path.basename(out)}: {merged['label'].shape[0]} samples, label {dict(zip(u.tolist(), c.tolist()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--neg', default='g', help='训练用neg目录前缀: g(=oligo+T2T,E2) 或 g_oligo(E1)')
    ap.add_argument('--seed', type=int, default=42, help='valid平衡下采样seed')
    args = ap.parse_args()
    S, O = args.split_dir, args.out
    os.makedirs(O, exist_ok=True)

    pack_pos(f'{S}/8oxog_train', f'{O}/pos_train')
    pack_neg(f'{S}/{args.neg}_train', f'{O}/neg_pack')
    # valid.npz: 全量, 每epoch跑(画recall/spec曲线+选模型+选T*, 服务器上~10-15%开销可接受)
    combine(f'{S}/8oxog_valid', [f'{S}/{args.neg}_valid'], f'{O}/valid.npz')
    # test: 全量, 最终报告一次性(oligo-FPR 对比已发表 / 基因组-FPR 答M2)
    combine(f'{S}/8oxog_test', [f'{S}/g_oligo_test'], f'{O}/test_oligo.npz')
    combine(f'{S}/8oxog_test', [f'{S}/g_t2t_test'], f'{O}/test_t2t.npz')
    print('done ->', O)


if __name__ == '__main__':
    main()
