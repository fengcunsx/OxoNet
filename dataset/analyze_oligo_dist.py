"""切分前的 oligo 分布体检：判断"按 f5 子目录切分"是否合理。
- 每 batch 的 f5 数、pos/neg 样本量、f5 大小分布
- 每个中心5-mer(=context)/7-mer 出现在多少个 f5 里(f5-spread) + 样本数
- 模拟 per-batch 90:5:5 的 f5 切分, 看每个 context 在 valid/test 是否被饿死
用法: python dataset/analyze_oligo_dist.py
"""
import glob, os, random
from collections import defaultdict
import numpy as np

BATCHES = ['batch1', 'batch2', 'batch3', 'batch4']
ROOT = '/home/bio/8oxog/data'


def f5_dirs(batch):
    base = f'{ROOT}/{batch}/{batch}_work/fast5s'
    return sorted(d for d in glob.glob(f'{base}/*') if os.path.isdir(d))


def load_kmers(f5, kind):  # kind: '8oxog'(pos) or 'g'(neg)
    p = f'{f5}/feature/oxo/{kind}/data.npz'
    if not os.path.isfile(p):
        return None
    return np.load(p)['kmers']  # (N,) U7


def c5(k7arr):  # 取中心5-mer
    return np.array([s[1:6] for s in k7arr])


def main():
    # 聚合
    per_batch = {}
    ctx_pos_f5 = defaultdict(set); ctx_neg_f5 = defaultdict(set)
    ctx_pos_n = defaultdict(int); ctx_neg_n = defaultdict(int)
    k7_pos = set(); k7_neg = set()
    # 每 f5 的 (batch, f5id) -> {context: pos_count}, 供切分模拟
    f5_ctx_pos = {}; f5_ctx_neg = {}

    for b in BATCHES:
        f5s = f5_dirs(b)
        pos_tot = neg_tot = 0
        sizes = []
        for f5 in f5s:
            fid = (b, os.path.basename(f5))
            kp = load_kmers(f5, '8oxog'); kn = load_kmers(f5, 'g')
            cp = defaultdict(int); cn = defaultdict(int)
            if kp is not None and len(kp):
                pos_tot += len(kp); sizes.append(len(kp))
                cc = c5(kp)
                for c in cc: cp[c] += 1
                for c in set(cc.tolist()): ctx_pos_f5[c].add(fid)
                for c, v in cp.items(): ctx_pos_n[c] += v
                k7_pos.update(kp.tolist())
            if kn is not None and len(kn):
                neg_tot += len(kn)
                cc = c5(kn)
                for c in cc: cn[c] += 1
                for c in set(cc.tolist()): ctx_neg_f5[c].add(fid)
                for c, v in cn.items(): ctx_neg_n[c] += v
                k7_neg.update(kn.tolist())
            f5_ctx_pos[fid] = cp; f5_ctx_neg[fid] = cn
        per_batch[b] = (len(f5s), pos_tot, neg_tot, sizes)

    print("=== 每 batch ===")
    for b, (nf5, pt, nt, sizes) in per_batch.items():
        sizes = sorted(sizes)
        med = sizes[len(sizes)//2] if sizes else 0
        print(f"  {b}: {nf5} f5 | pos={pt:,} neg={nt:,} | f5-posSize min/med/max = {min(sizes) if sizes else 0}/{med}/{max(sizes) if sizes else 0}")

    all_ctx = sorted(set(ctx_pos_n) | set(ctx_neg_n))
    print(f"\n=== 全局 ===")
    print(f"  中心5-mer context 数: pos={len(ctx_pos_n)} neg={len(ctx_neg_n)} 并集={len(all_ctx)}")
    print(f"  唯一7-mer: pos={len(k7_pos)} neg={len(k7_neg)} 并集={len(k7_pos|k7_neg)} | pos-only(仅阳)={len(k7_pos-k7_neg)}")

    # per-context f5-spread
    spreads_p = sorted(len(ctx_pos_f5[c]) for c in ctx_pos_n)
    spreads_n = sorted(len(ctx_neg_f5[c]) for c in ctx_neg_n)
    posn = sorted(ctx_pos_n.values()); negn = sorted(ctx_neg_n.values())
    def mmm(x): return f"{min(x)}/{x[len(x)//2]}/{max(x)}" if x else "-"
    print(f"\n=== per-context 覆盖(共{len(all_ctx)}个) ===")
    print(f"  pos: 每context样本数 min/med/max = {mmm(posn)}; 出现在几个f5 min/med/max = {mmm(spreads_p)}")
    print(f"  neg: 每context样本数 min/med/max = {mmm(negn)}; 出现在几个f5 min/med/max = {mmm(spreads_n)}")
    low_p = sum(1 for c in ctx_pos_n if len(ctx_pos_f5[c]) < 5)
    print(f"  ⚠️ pos 出现在 <5 个f5 的 context 数: {low_p} (这些按f5切易被饿死)")

    # 模拟 per-batch 90:5:5 f5 切分
    print(f"\n=== 模拟 per-batch 90:5:5 (按f5, seed=42) ===")
    random.seed(42)
    split = {}  # fid -> 'train'/'valid'/'test'
    for b in BATCHES:
        f5s = [(b, os.path.basename(d)) for d in f5_dirs(b)]
        random.shuffle(f5s)
        n = len(f5s); n_te = max(1, round(n*0.05)); n_va = max(1, round(n*0.05))
        for i, fid in enumerate(f5s):
            split[fid] = 'test' if i < n_te else ('valid' if i < n_te+n_va else 'train')
    # 每 split 每 context 计数
    for part in ('train', 'valid', 'test'):
        cp = defaultdict(int); cn = defaultdict(int)
        for fid, s in split.items():
            if s != part: continue
            for c, v in f5_ctx_pos.get(fid, {}).items(): cp[c] += v
            for c, v in f5_ctx_neg.get(fid, {}).items(): cn[c] += v
        missing_p = [c for c in all_ctx if cp[c] == 0]
        posn = sorted(cp[c] for c in all_ctx);
        print(f"  [{part}] context有pos: {sum(1 for c in all_ctx if cp[c]>0)}/{len(all_ctx)} | 每context pos min/med/max = {mmm([x for x in posn])} | pos全无的context数={len(missing_p)}")
        if missing_p and part in ('valid','test'):
            print(f"        被饿死(pos=0)的context: {missing_p[:15]}{'...' if len(missing_p)>15 else ''}")


if __name__ == '__main__':
    main()
