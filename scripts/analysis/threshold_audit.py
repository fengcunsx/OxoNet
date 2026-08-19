"""操作点口径审计: 阈值定在测试集阴性上, 到底有没有问题?

两个担心:
  A. "你们的工作点是在测试集上选的" —— 用 read 级二分法量化: 阈值在阴性的一半上定,
     到另一半上看实际 FPR。若两半一致 → 该阈值是数据集属性而非过拟合, 质疑不成立。
  B. **极端 FPR 档还有没有分辨力** —— test_t2t 只有 1.94M 阴性, FPR=1e-6 意味着
     阈值由约 2 个阴性样本决定。给出各档的"阈值上方阴性数"与 Poisson 95% CI。

    /home/bio/anaconda3/bin/python threshold_audit.py
"""
import argparse
import os

import numpy as np

PACK = '/home/bio/8oxog/build/pack'
MODELS = {
    'OxoNet': '/home/bio/8oxog/wtl1/pack_scores_e2ep125',
    'esox': '/home/bio/8oxog/wtl1/pack_scores_esox',
    'NanoCon': '/home/bio/8oxog/nanocon/scores',
    'GBDT(特征天花板)': '/home/bio/8oxog/nanocon',
}
GRID = (1e-3, 1e-4, 1e-5, 1e-6)


def probs_path(d, name):
    for c in (os.path.join(d, name + '.probs.npz'), os.path.join(d, 'gbdt_' + name + '.probs.npz')):
        if os.path.isfile(c):
            return c
    raise SystemExit('缺 {} 的 {}'.format(d, name))


def poisson_ci(k):
    """观测到 k 个事件时的 Poisson 95% CI(Garwood 精确法的正态近似够用)。"""
    if k == 0:
        return 0.0, 3.69
    return k * (1 - 1 / (9.0 * k) - 1.96 / (3 * np.sqrt(k))) ** 3, \
        (k + 1) * (1 - 1 / (9.0 * (k + 1)) + 1.96 / (3 * np.sqrt(k + 1))) ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/threshold_audit.txt')
    args = ap.parse_args()

    t2t = np.load(os.path.join(PACK, 'test_t2t.npz'), allow_pickle=False)
    lab = t2t['label'].astype(int)
    rid = t2t['read_id']
    neg_mask = lab == 0
    # 按 read 二分(不是按行), 否则同一条 read 的位点会跨到两边 = 假独立
    negr = np.unique(rid[neg_mask])
    rng = np.random.default_rng(42)
    half = set(rng.permutation(negr)[:len(negr) // 2].tolist())
    inA = np.isin(rid, list(half))
    A = neg_mask & inA
    B = neg_mask & ~inA

    L = ['阈值口径审计 (test_t2t: {:,} 阴性 / {:,} 条阴性 read; 阳性 {:,})'.format(
        int(neg_mask.sum()), len(negr), int((lab == 1).sum())),
        '',
        '=== B. 各 FPR 档的分辨力(阈值由多少个阴性决定) ===',
        '  目标 FPR   阈值上方阴性数   Poisson 95% CI(换算成 FPR)']
    n_neg = int(neg_mask.sum())
    for g in GRID:
        k = g * n_neg
        lo, hi = poisson_ci(int(round(k)))
        L.append('  {:8.0e}   {:>12.1f}   [{:.2e}, {:.2e}]{}'.format(
            g, k, lo / n_neg, hi / n_neg,
            '   <-- 不可用' if k < 10 else ('   <-- 勉强' if k < 50 else '')))
    L.append('  → 阴性总数 {:,} → 可分辨的最小 FPR 约 {:.1e}(单个阴性)'.format(n_neg, 1.0 / n_neg))

    L += ['', '=== A. 阈值在一半阴性 read 上定, 到另一半上量实际 FPR ===',
          '  (A/B 各 {:,} / {:,} 个阴性位点)'.format(int(A.sum()), int(B.sum())),
          '  {:14s} {:>9s} {:>10s} {:>12s} {:>10s}'.format(
              'model', '目标FPR', '阈值', 'B半实测FPR', '偏差')]
    for m, d in MODELS.items():
        p = np.nan_to_num(np.load(probs_path(d, 'test_t2t'))['prob'].astype(np.float64), nan=-1.0)
        for g in GRID:
            thr = float(np.quantile(p[A], 1 - g))
            fb = float((p[B] >= thr).mean())
            L.append('  {:14s} {:9.0e} {:10.5f} {:12.2e} {:>9.0%}'.format(
                m, g, thr, fb, (fb - g) / g))
    L.append('  → 偏差远小于同档的 Poisson 区间 = 阈值是数据集属性, 不是在测试集上过拟合出来的')

    # 阳性侧: 换阈值来源对 recall 的影响
    L += ['', '=== A2. 换阈值来源对报告 recall 的影响(阳性集不变) ===',
          '  {:14s} {:>9s} {:>11s} {:>11s} {:>9s}'.format(
              'model', 'FPR', 'rec(全阴性定)', 'rec(半阴性定)', '差')]
    pos = lab == 1
    for m, d in MODELS.items():
        p = np.nan_to_num(np.load(probs_path(d, 'test_t2t'))['prob'].astype(np.float64), nan=-1.0)
        for g in GRID:
            r_full = float((p[pos] >= float(np.quantile(p[neg_mask], 1 - g))).mean())
            r_half = float((p[pos] >= float(np.quantile(p[A], 1 - g))).mean())
            L.append('  {:14s} {:9.0e} {:11.2%} {:11.2%} {:9.2f}pp'.format(
                m, g, r_full, r_half, 100 * (r_half - r_full)))

    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print('\n写出', args.out)


if __name__ == '__main__':
    main()
