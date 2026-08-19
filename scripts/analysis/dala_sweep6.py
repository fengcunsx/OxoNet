"""六样本(3L vs 3D)的 D-Ala 阈值稳健性扫描 —— dala_sweep.py 的多样本版。

为什么另起一个文件：`dala_sweep.py` 把样本对硬编码成 wtl1/wtd1
(`Ls, Ds = 'L (wtl1)', 'D (wtd1)'`)，且它只做**两个样本**的比较；
2026-08-19 wtl2 补齐后要做的是 **3L vs 3D 的样本级检验**，结构不同。
原文件逐字保留，旧结果仍可复现。

复用 `dala_sweep.py` 的 `hist_by_chrom/rate_at/rate_per_chrom/thr_for_rate`
（不重新实现），核心思路不变：**打分被截断到 4 位小数 -> 一遍扫描建
"染色体 × 10001 个分数桶"的计数矩阵，之后任意阈值都能解析求出**。
六个样本 × 三个模型 = 18 个矩阵，每个 ~2 MB，**缓存到磁盘**，
之后换阈值口径不必再读那 1800 个打分文件（一次扫描约 1 小时）。

三种口径与 dala_sweep.py 一致：
  ① 等基因组 FPR（1e-3/1e-4/1e-5，在 valid 的基因组阴性上定）—— 全文通用货币
  ② **校准到文献本底**：令 **L 组(三个样本合并)判正率 = 75/百万**，再看 D 组高多少
     —— 论文主结果用的就是这一档
  ③ esox@0.95 —— esox 论文自己的操作点，其余模型按同 FPR 折算

每档输出：六个样本各自的率、L/D 组均值、D−L、D/L、是否完全分离、
Welch t 检验 p、**精确置换检验 p**(C(6,3)=20 种分组，单侧)、
以及 L 组合并 vs D 组合并的**染色体级配对 t 检验**(n=23)。

    /home/bio/anaconda3/bin/python dala_sweep6.py
    /home/bio/anaconda3/bin/python dala_sweep6.py --rebuild   # 忽略缓存重扫
"""
import argparse
import itertools
import os

import numpy as np

from dala_effect import (PACK_SCORES, SAMPLES, COL, MAIN_CHROM, thr_at_fpr,
                         read_chrom_map, VALID_GENOMIC_MASK)
from dala_sweep import hist_by_chrom, rate_at, rate_per_chrom, thr_for_rate

LS = ['L (wtl1)', 'L (wtl2)', 'L (p53l1)']
DS = ['D (wtd1)', 'D (wtd2)', 'D (p53d1)']
CACHE = '/data/dala_hist'


def get_hist(model, sample, chrom_idx, rebuild=False):
    """(model, sample) -> 计数矩阵；缓存到 CACHE，避免重复扫 1800 个打分文件。"""
    key = '{}__{}.npz'.format(model, sample.replace(' ', '').replace('(', '_').replace(')', ''))
    path = os.path.join(CACHE, key)
    if os.path.isfile(path) and not rebuild:
        return np.load(path)['H']
    print('  扫 {} / {} ...'.format(model, sample), flush=True)
    cmap = read_chrom_map(SAMPLES[sample]['coords'])
    H = hist_by_chrom(SAMPLES[sample]['scores'][model], COL.get(model, 'prob'), cmap, chrom_idx)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, H=H)
    del cmap
    return H


def perm_p(a, b):
    """精确置换检验(单侧 D>L)：6 个数分成 3+3 共 20 种，均值差 >= 实测的比例。"""
    allv = np.concatenate([a, b])
    obs = b.mean() - a.mean()
    hit = tot = 0
    for idx in itertools.combinations(range(len(allv)), len(b)):
        g2 = allv[list(idx)]
        g1 = allv[[i for i in range(len(allv)) if i not in idx]]
        tot += 1
        hit += (g2.mean() - g1.mean() >= obs - 1e-12)
    return hit / tot, hit, tot


def block(H, title, thrs, chrom_idx, out):
    from scipy import stats
    out.append('=== {} ==='.format(title))
    out.append('{:9s} {:>8s} {:>26s} {:>26s} {:>8s} {:>8s} {:>7s} {:>6s} {:>8s} {:>9s}'.format(
        'model', '阈值', 'L 三样本', 'D 三样本', 'L均值', 'D均值', 'D/L', '分离', 'Welch p', '置换 p'))
    for m in PACK_SCORES:
        thr = thrs[m]
        a = np.array([1e6 * rate_at(H[(m, s)], thr)[1] / rate_at(H[(m, s)], thr)[0] for s in LS])
        b = np.array([1e6 * rate_at(H[(m, s)], thr)[1] / rate_at(H[(m, s)], thr)[0] for s in DS])
        _, pw = stats.ttest_ind(b, a, equal_var=False)
        pp, hit, tot = perm_p(a, b)
        out.append('{:9s} {:8.4f} {:>26s} {:>26s} {:8.1f} {:8.1f} {:6.3f}x {:>6s} {:8.3f} {:5.3f}({}/{})'.format(
            m, thr, '/'.join('%.1f' % x for x in a), '/'.join('%.1f' % x for x in b),
            a.mean(), b.mean(), b.mean() / a.mean(),
            '是' if b.min() > a.max() else '否', pw, pp, hit, tot))
        # 染色体级：L 三样本合并 vs D 三样本合并（组内先把计数相加，再算率）
        pl = rate_per_chrom(sum(H[(m, s)] for s in LS), thr, chrom_idx)
        pd_ = rate_per_chrom(sum(H[(m, s)] for s in DS), thr, chrom_idx)
        rows = [(1e6 * pl[k][0] / pl[k][1], 1e6 * pd_[k][0] / pd_[k][1])
                for k in MAIN_CHROM if k in pl and k in pd_ and pl[k][1] > 10000]
        x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
        _, pt = stats.ttest_rel(y, x)
        out.append('{:9s}   染色体级(L组合并 vs D组合并, n={}): 配对t p={:.1e}, {}/{} 条 D>L'.format(
            '', len(rows), pt, int((y > x).sum()), len(rows)))
    out.append('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fprs', nargs='+', type=float, default=[1e-3, 1e-4, 1e-5])
    ap.add_argument('--calib-rate', type=float, default=75e-6, help='校准口径: L 组目标判正率')
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--out', default='/data/dala_sweep_3v3.txt')
    args = ap.parse_args()

    chrom_idx = {c: i for i, c in enumerate(MAIN_CHROM)}
    H = {}
    for m in PACK_SCORES:
        for s in LS + DS:
            H[(m, s)] = get_hist(m, s, chrom_idx, args.rebuild)

    out = ['D-Ala 阈值稳健性扫描 · 3L vs 3D（六个 native 样本全齐）',
           'L = wtl1, wtl2, p53l1     D = wtd1, wtd2, p53d1',
           '每百万**被检测** G（中心 5-mer 属那 100 种 context，占基因组 G 的 41.2%）',
           '六个样本恒用同一阈值；置换 p = 精确单侧(C(6,3)=20 种分组)', '']

    for f in args.fprs:
        block(H, '① 等基因组 FPR = {:.0e}'.format(f),
              {m: thr_at_fpr(PACK_SCORES[m], f) for m in PACK_SCORES}, chrom_idx, out)

    # ② 校准：令 L 组(三样本合并)的判正率 = calib-rate
    block(H, '② 校准：L 组(三样本合并)判正率 = {:.0f}/百万（esox 报的基因组本底）'.format(
        args.calib_rate * 1e6),
        {m: thr_for_rate(sum(H[(m, s)] for s in LS), args.calib_rate) for m in PACK_SCORES},
        chrom_idx, out)

    # ③ esox 论文操作点
    e95 = 0.95
    a = np.load(os.path.join(PACK_SCORES['esox'], 'valid.probs.npz'))
    p = np.nan_to_num(a['prob'].astype(np.float64), nan=-1.0)
    f95 = float((p[np.load(VALID_GENOMIC_MASK)] >= e95).mean())
    out.append('（esox@0.95 对应的基因组 FPR = {:.2e}；下表其余模型按同一 FPR 折算）'.format(f95))
    block(H, '③ esox 论文操作点 score>0.95',
          {m: (e95 if m == 'esox' else thr_at_fpr(PACK_SCORES[m], f95)) for m in PACK_SCORES},
          chrom_idx, out)

    out += ['=== 边界 ===',
            '① **置换 p=0.050 是 3v3 的下限**(完全分离的单侧概率 = 1/C(6,3))，不是"刚好显著"；',
            '   样本数封顶：ENA PRJEB76712 只有 6 个 native run。',
            '② Welch p 受组内方差支配：三个 L 对照的绝对率相差可达 1.5 倍，是 flow cell 批次变异。',
            '③ 染色体级 p 是"效应在染色体间是否一致为正"，染色体间不独立，配对检验偏乐观；',
            '   它**不能**替代样本级 p。',
            '④ p53l1/p53d1 是 p53-/- 背景，按处理分组时基因型作为组内变异被吸收。']
    txt = '\n'.join(out)
    print(txt)
    with open(args.out, 'w') as fh:
        fh.write(txt + '\n')
    print('\n写出', args.out)


if __name__ == '__main__':
    main()
