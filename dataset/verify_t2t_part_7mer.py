"""验证 T2T 按 part 名 90:5:5 切分下, 7-mer 分布是否合理:
- 每种7-mer在 train/valid/test 的计数
- 是否有"只出现在一个集合"的7-mer(train-only/valid-only/test-only)
- 缺席 valid/test 的数量; 比例(valid/test占比)
per-source 切 part(每源parts 90:5:5), seed=42。
"""
import glob, random
from collections import defaultdict
import numpy as np

PATS = {
    '288548394': '/home/bio/8oxog/t2t/288548394/*/output_new/oxo/*.npz',
    '3439856925': '/home/bio/8oxog/t2t/3439856925/*/output_new/oxo/*.npz',
    'new': '/home/bio/8oxog/t2t/new/output/*/output/oxo/*.npz',
}


def main():
    random.seed(42)
    part_split = {}   # path -> split
    for s, pat in PATS.items():
        parts = sorted(glob.glob(pat))
        random.shuffle(parts)
        n = len(parts); nte = max(1, round(n * 0.05)); nva = max(1, round(n * 0.05))
        for i, p in enumerate(parts):
            part_split[p] = 'test' if i < nte else ('valid' if i < nte + nva else 'train')
        print(f"{s}: {n} parts -> train {n-nte-nva} / valid {nva} / test {nte}")

    c = {'train': defaultdict(int), 'valid': defaultdict(int), 'test': defaultdict(int)}
    for p, sp in part_split.items():
        k7 = np.load(p)['kmers']
        for k in k7:
            c[sp][k] += 1

    allk = sorted(set(c['train']) | set(c['valid']) | set(c['test']))
    def has(s, k): return c[s][k] > 0
    in3 = sum(1 for k in allk if has('train', k) and has('valid', k) and has('test', k))
    tr_only = [k for k in allk if has('train', k) and not has('valid', k) and not has('test', k)]
    va_only = [k for k in allk if has('valid', k) and not has('train', k) and not has('test', k)]
    te_only = [k for k in allk if has('test', k) and not has('train', k) and not has('test', k) is False and not has('train', k)]
    te_only = [k for k in allk if has('test', k) and not has('train', k) and not has('valid', k)]
    miss_va = [k for k in allk if not has('valid', k)]
    miss_te = [k for k in allk if not has('test', k)]
    miss_tr = [k for k in allk if not has('train', k)]

    def mmm(xs): xs = sorted(xs); return f"{min(xs)}/{xs[len(xs)//2]}/{max(xs)}" if xs else "-"
    print(f"\n共 {len(allk)} 种 7-mer")
    print(f"  同时在 train+valid+test: {in3}/{len(allk)} ({in3/len(allk)*100:.1f}%)")
    print(f"  只在单一集合: train-only={len(tr_only)} valid-only={len(va_only)} test-only={len(te_only)}")
    print(f"  缺席: train缺={len(miss_tr)} valid缺={len(miss_va)} test缺={len(miss_te)}")
    print(f"  train 每种计数 min/med/max = {mmm([c['train'][k] for k in allk])}")
    print(f"  valid 每种计数 min/med/max = {mmm([c['valid'][k] for k in allk])}")
    print(f"  test  每种计数 min/med/max = {mmm([c['test'][k] for k in allk])}")
    tot = {k: c['train'][k] + c['valid'][k] + c['test'][k] for k in allk}
    fva = sorted(c['valid'][k] / tot[k] for k in allk)
    fte = sorted(c['test'][k] / tot[k] for k in allk)
    print(f"  valid占比 min/med/max = {fva[0]:.3f}/{fva[len(fva)//2]:.3f}/{fva[-1]:.3f} (目标~0.05)")
    print(f"  test 占比 min/med/max = {fte[0]:.3f}/{fte[len(fte)//2]:.3f}/{fte[-1]:.3f} (目标~0.05)")
    if va_only or te_only or tr_only:
        print(f"  ⚠️ 单一集合样例: train-only{tr_only[:5]} valid-only{va_only[:5]} test-only{te_only[:5]}")
    else:
        print("  ✓ 无任何7-mer只出现在单一集合")


if __name__ == '__main__':
    main()
