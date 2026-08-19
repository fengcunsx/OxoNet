"""把 OxoNet 的 pack 转成 NanoCon 原版能直接吃的 CSV(不改 NanoCon 任何代码)。

NanoCon 输入格式(src/utils.py::load_dataset, 首行当表头跳过):
  kmer7, "mean×7", "std×7", "dwell×7", label, read_id, basecall_pos

用法:
  # 训练集: 全部 train pos + 每个 5-mer context 按 ratio×pos 数抽 neg, 全局 shuffle
  python pack_to_nanocon_csv.py train --pack /path/pack --out out/train.csv --neg-ratio 4 --seed 42
  # 评估集(valid/test_oligo/test_t2t/native): 原样转, 可 shuffle
  python pack_to_nanocon_csv.py eval --npz /path/pack/valid.npz --out out/valid.csv --shuffle --seed 42

设计要点:
- pos:neg 默认 1:4 = NanoCon 原版 train.csv 的口径(2.83M pos/11.82M neg), 单个静态 CSV。
- neg 按 context 成比例抽(每 ctx 取 ratio×该 ctx 的 pos 数), 与 OxoNet 的 per-ctx 配对一致,
  避免模型靠 5-mer context 先验作弊。
- **必须全局 shuffle**: NanoCon 的 collate 用「前半 batch × 后半 batch」配对做对比损失,
  且 valid 的 AUPRC 是逐 batch 算完再平均(validation_epoch_end), valid loader shuffle=False
  → 落盘顺序若按类别聚集, 会出现 batch 内全阴/全阳, 指标失真。
- neg_pack 没有 read_id/basecall_pos 字段(pack 时未存), train.csv 的 neg 行这两列写空/-1;
  训练不用它们。评估集 npz 有, 会带上, 用于与 OxoNet/esox 逐位点对齐。
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

CODE = np.frombuffer(b'ACGT', dtype=np.uint8)  # kmers_enc: 0=A 1=C 2=G 3=T
NEED = ['mean', 'std', 'dwell']
# kmer, mean(7), std(7), dwell(7), label, read_id, pos
FMT = ('%s,"%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g",'
       '"%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g",'
       '"%d,%d,%d,%d,%d,%d,%d",%d,%s,%d')
HEADER = 'kmer,signal_means,signal_stds,signal_lens,label,read_id,pos'
CHUNK = 200_000


def decode_kmers(enc):
    """(N,7) int8 -> (N,) 'S7' -> list[str]"""
    return CODE[np.ascontiguousarray(enc, dtype=np.uint8)].view('S7').ravel().astype('U7')


def clean_kmers(kmers):
    """oligo 的中心可能是 'o'(8-oxoG); NanoCon 的 5-mer 词表只有 ACGT。"""
    return np.char.replace(kmers.astype('U7'), 'o', 'G')


def write_rows(fh, kmer, mean, std, dwell, label, read_id, pos):
    """分块格式化落盘, 避免一次性生成 1500 万条 Python 字符串。
    read_id 用 'S36' 存(比 U36 省 4 倍内存), 落盘前逐块 decode。"""
    n = len(kmer)
    as_bytes = read_id.dtype.kind == 'S'
    for i in range(0, n, CHUNK):
        j = min(i + CHUNK, n)
        rid = [x.decode() for x in read_id[i:j].tolist()] if as_bytes \
            else read_id[i:j].tolist()
        rows = zip(kmer[i:j].tolist(), mean[i:j].tolist(), std[i:j].tolist(),
                   dwell[i:j].tolist(), label[i:j].tolist(),
                   rid, pos[i:j].tolist())
        fh.write('\n'.join(FMT % (k, *m, *s, *d, l, r, p)
                           for k, m, s, d, l, r, p in rows))
        fh.write('\n')
        print(f'\r  写入 {j}/{n} ({100.0 * j / n:.1f}%)', end='', file=sys.stderr, flush=True)
    print('', file=sys.stderr)


def load_eval_npz(path):
    a = np.load(path)
    out = {k: a[k] for k in NEED}
    out['kmer'] = clean_kmers(a['kmers'])
    out['label'] = a['label'].astype(np.int64)
    n = out['label'].shape[0]
    out['read_id'] = (a['read_id'].astype('S36') if 'read_id' in a.files
                      else np.full(n, b'', dtype='S36'))
    bp = a['basecall_pos'] if 'basecall_pos' in a.files else np.full((n, 1), -1, np.int64)
    out['pos'] = np.asarray(bp).reshape(-1).astype(np.int64)
    return out


def neg_plan(neg_root, ctx_counts, ratio, rng):
    """决定每个 ctx 取哪些 neg 行。ratio<=0 表示全量。返回 [(ctx_dir, idx_or_None, n)]。"""
    plan = []
    ctxs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(neg_root, '*'))
                  if os.path.isdir(d))
    for ctx in ctxs:
        d = os.path.join(neg_root, ctx)
        avail = np.load(os.path.join(d, 'label.npy'), mmap_mode='r').shape[0]
        if ratio <= 0:                      # 全量: 不抽样, 顺序读(最快且无随机性)
            plan.append((d, None, avail))
            continue
        want = min(int(round(ratio * ctx_counts.get(ctx, 0))), avail)
        if want <= 0:
            continue
        idx = np.sort(rng.choice(avail, size=want, replace=False)) if want < avail else None
        plan.append((d, idx, want))
    return plan


def build_train(args):
    """内存友好: 预分配最终数组, 各源按随机目标位置 scatter 进去(避免 concat+permute 的两份全量拷贝)。
    全量 48.3M 行时峰值 ~8GB, 而非 ~28GB。"""
    pack = args.pack
    rng = np.random.default_rng(args.seed)

    print('载入 pos_train.npz ...', file=sys.stderr)
    a = np.load(os.path.join(pack, 'pos_train.npz'))
    pos = {f: a[f] for f in NEED}
    pos['kmer'] = decode_kmers(a['kmers_enc'])
    pos['label'] = a['label'].astype(np.int8)
    pos['read_id'] = a['read_id'].astype('S36')  # U36 会白占 4 倍内存
    pos['pos'] = np.asarray(a['basecall_pos']).reshape(-1).astype(np.int32)
    pos['dwell'] = pos['dwell'].astype(np.int32)
    npos = pos['label'].shape[0]
    if args.limit_pos:
        keep = np.sort(rng.choice(npos, size=min(args.limit_pos, npos), replace=False))
        pos = {k: v[keep] for k, v in pos.items()}
        npos = pos['label'].shape[0]
    # per-ctx 配额直接从 pos 的 7-mer 中心 5-mer 数出来(limit 模式也正确)
    ctx_of_pos = np.array([s[1:6] for s in pos['kmer'].tolist()], dtype='U5')
    ctxs, cnts = np.unique(ctx_of_pos, return_counts=True)
    counts = dict(zip(ctxs.tolist(), cnts.tolist()))
    print(f'pos = {npos}, {len(counts)} ctx', file=sys.stderr)

    plan = neg_plan(os.path.join(pack, 'neg_pack'), counts, args.neg_ratio, rng)
    nneg = sum(n for _, _, n in plan)
    ntot = npos + nneg
    mode = '全量' if args.neg_ratio <= 0 else f'ratio {args.neg_ratio:g}'
    print(f'neg = {nneg} ({mode}, pos:neg = 1:{nneg / max(1, npos):.2f}), 合计 {ntot} 行',
          file=sys.stderr)

    out = {
        'kmer': np.empty(ntot, dtype='U7'),
        'mean': np.empty((ntot, 7), dtype=np.float32),
        'std': np.empty((ntot, 7), dtype=np.float32),
        'dwell': np.empty((ntot, 7), dtype=np.int32),
        'label': np.empty(ntot, dtype=np.int8),
        'read_id': np.empty(ntot, dtype='S36'),
        'pos': np.empty(ntot, dtype=np.int32),
    }
    perm = rng.permutation(ntot)            # 全局 shuffle: 直接 scatter 到随机位置
    dst = perm[:npos]
    for k in out:
        out[k][dst] = pos[k]
    del pos, a
    off = npos
    for pi, (d, idx, n) in enumerate(plan):
        dst = perm[off:off + n]
        for f in NEED:
            src = np.load(os.path.join(d, f + '.npy'), mmap_mode='r')
            out[f][dst] = src[idx] if idx is not None else src[:]
        enc = np.load(os.path.join(d, 'kmers_enc.npy'), mmap_mode='r')
        out['kmer'][dst] = decode_kmers(enc[idx] if idx is not None else enc[:])
        out['label'][dst] = 0
        out['read_id'][dst] = b''
        out['pos'][dst] = -1
        off += n
        print(f'\r  neg 装载 {pi + 1}/{len(plan)} ctx, 累计 {off - npos}', end='',
              file=sys.stderr, flush=True)
    print('', file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    t0 = time.time()
    with open(args.out, 'w') as fh:
        fh.write(HEADER + '\n')
        write_rows(fh, out['kmer'], out['mean'], out['std'], out['dwell'],
                   out['label'], out['read_id'], out['pos'])
    print(f"{args.out}: {ntot} 行 (pos {npos}, neg {nneg}), "
          f"{os.path.getsize(args.out) / 2**30:.2f} GiB, {time.time() - t0:.0f}s", file=sys.stderr)


def build_eval(args):
    rng = np.random.default_rng(args.seed)
    d = load_eval_npz(args.npz)
    n = d['label'].shape[0]
    if args.limit:
        keep = np.sort(rng.choice(n, size=min(args.limit, n), replace=False))
        d = {k: v[keep] for k, v in d.items()}
        n = d['label'].shape[0]
    if args.shuffle:
        order = rng.permutation(n)
        d = {k: v[order] for k, v in d.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    t0 = time.time()
    with open(args.out, 'w') as fh:
        fh.write(HEADER + '\n')
        write_rows(fh, d['kmer'], d['mean'], d['std'], d['dwell'],
                   d['label'], d['read_id'], d['pos'])
    print(f"{args.out}: {n} 行 (pos {int(d['label'].sum())}), "
          f"{os.path.getsize(args.out) / 2**30:.2f} GiB, {time.time() - t0:.0f}s", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='pos_train.npz + neg_pack -> train.csv')
    t.add_argument('--pack', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--neg-ratio', type=float, default=0.0,
                   help='0(默认)=全量 neg(天然 1:15.2); >0 则每 ctx 取 ratio×pos 数')
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--limit-pos', type=int, default=0, help='冒烟测试: 只用这么多 pos')
    t.set_defaults(func=build_train)

    e = sub.add_parser('eval', help='valid/test_*/native npz -> csv')
    e.add_argument('--npz', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--shuffle', action='store_true')
    e.add_argument('--seed', type=int, default=42)
    e.add_argument('--limit', type=int, default=0)
    e.set_defaults(func=build_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
