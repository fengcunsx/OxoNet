"""测: per-batch 90:5:5 按f5切分下, 5-mer 和 7-mer 在 train/valid/test 的覆盖与比例。
回答: f5切分能否保证每种7-mer同时出现在三个集? 比例是否一致? 5-mer呢?
"""
import glob, os, random
from collections import defaultdict
import numpy as np

BATCHES = ['batch1', 'batch2', 'batch3', 'batch4']
ROOT = '/home/bio/8oxog/data'


def f5_dirs(b):
    return sorted(d for d in glob.glob(f'{ROOT}/{b}/{b}_work/fast5s/*') if os.path.isdir(d))


def main():
    # 切分(与analyze_oligo_dist同法, seed=42)
    random.seed(42)
    split = {}
    for b in BATCHES:
        f5s = [os.path.basename(d) for d in f5_dirs(b)]
        random.shuffle(f5s)
        n = len(f5s); n_te = max(1, round(n*0.05)); n_va = max(1, round(n*0.05))
        for i, fid in enumerate(f5s):
            split[(b, fid)] = 'test' if i < n_te else ('valid' if i < n_te+n_va else 'train')

    # 逐f5聚合 pos 的 5-mer / 7-mer 计数到各split
    c5 = {s: defaultdict(int) for s in ('train', 'valid', 'test')}
    c7 = {s: defaultdict(int) for s in ('train', 'valid', 'test')}
    f5spread7 = defaultdict(set)
    for b in BATCHES:
        for d in f5_dirs(b):
            fid = (b, os.path.basename(d)); s = split[fid]
            p = f'{d}/feature/oxo/8oxog/data.npz'
            if not os.path.isfile(p): continue
            k7 = np.load(p)['kmers']
            if not len(k7): continue
            for kk in k7:
                c7[s][kk] += 1; c5[s][kk[1:6]] += 1
                f5spread7[kk].add(fid)

    all5 = sorted(set().union(*[set(c5[s]) for s in c5]))
    all7 = sorted(set().union(*[set(c7[s]) for s in c7]))
    def mmm(x): x=sorted(x); return f"{min(x)}/{x[len(x)//2]}/{max(x)}" if x else "-"

    print(f"=== 5-mer (共{len(all5)}种pos) ===")
    in3 = sum(1 for c in all5 if all(c5[s][c] > 0 for s in c5))
    print(f"  同时在train/valid/test的: {in3}/{len(all5)}")
    print(f"  valid每种计数 min/med/max = {mmm([c5['valid'][c] for c in all5])}")
    print(f"  test 每种计数 min/med/max = {mmm([c5['test'][c] for c in all5])}")
    # 比例(valid占比 该 ~0.05)
    fr_va = [c5['valid'][c]/max(1,(c5['train'][c]+c5['valid'][c]+c5['test'][c])) for c in all5]
    fr_te = [c5['test'][c]/max(1,(c5['train'][c]+c5['valid'][c]+c5['test'][c])) for c in all5]
    print(f"  每种 valid占比 min/med/max = {min(fr_va):.3f}/{sorted(fr_va)[len(fr_va)//2]:.3f}/{max(fr_va):.3f} (目标~0.05)")
    print(f"  每种 test 占比 min/med/max = {min(fr_te):.3f}/{sorted(fr_te)[len(fr_te)//2]:.3f}/{max(fr_te):.3f} (目标~0.05)")

    print(f"\n=== 7-mer (共{len(all7)}种pos) ===")
    in3 = sum(1 for c in all7 if all(c7[s][c] > 0 for s in c7))
    miss_va = sum(1 for c in all7 if c7['valid'][c] == 0)
    miss_te = sum(1 for c in all7 if c7['test'][c] == 0)
    print(f"  同时在train/valid/test的: {in3}/{len(all7)} ({in3/len(all7)*100:.0f}%)")
    print(f"  valid里缺席(=0)的7-mer: {miss_va}/{len(all7)} | test里缺席: {miss_te}/{len(all7)}")
    print(f"  每种7-mer摊在几个f5 min/med/max = {mmm([len(f5spread7[c]) for c in all7])}")
    print(f"  valid每种计数 min/med/max = {mmm([c7['valid'][c] for c in all7])}")
    print(f"  test 每种计数 min/med/max = {mmm([c7['test'][c] for c in all7])}")
    low_te = sum(1 for c in all7 if 0 < c7['test'][c] < 5)
    print(f"  test里计数<5的7-mer: {low_te}")
    fr_va = [c7['valid'][c]/max(1,(c7['train'][c]+c7['valid'][c]+c7['test'][c])) for c in all7]
    print(f"  每种7-mer valid占比 min/med/max = {min(fr_va):.3f}/{sorted(fr_va)[len(fr_va)//2]:.3f}/{max(fr_va):.3f} (目标~0.05)")


if __name__ == '__main__':
    main()
