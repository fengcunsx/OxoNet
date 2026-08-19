"""D-Ala 效应的阈值稳健性扫描 —— 效应是不是只在松工作点才有？

`dala_effect.py` 只给了 FPR=1e-4 一个点，而该点下 L 组绝对率 238/百万，比文献本底
(~70–80/百万) 高 3 倍，说明工作点偏松。本脚本在三种口径下复算：

  ① **等基因组 FPR 扫描**（1e-3/1e-4/1e-5，在 **valid 的基因组阴性**上定）—— 全文通用货币
     （1e-6 已弃用：test_t2t 仅 1.94M 阴性，该档由约 2 个阴性决定，见 METHOD_AUDIT.md）
  ② **校准到文献本底**：令 **L 对照组判正率 = 75/百万**，再看 D 组高多少
     （把"绝对率对不对"与"效应存不存在"分开，生物学解释力最强）
  ③ **esox@0.95** —— esox 论文自己的操作点，作锚点

实现要点：打分文件里的分数都截断到 4 位小数，所以**一遍扫描**即可建
"逐染色体 × 10001 个分数桶"的直方图（24×10001 个计数，2 MB），
之后**任意阈值都能解析求出**，不必为每个阈值重读 900 个文件。

    /home/bio/anaconda3/bin/python dala_sweep.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from dala_effect import (PACK_SCORES, SAMPLES, COL, MAIN_CHROM, uid64, thr_at_fpr,
                         read_chrom_map, VALID_GENOMIC_MASK)

NBIN = 10001                      # 分数 0.0000..1.0000, 步长 1e-4


def hist_by_chrom(score_dir, col, cmap, chrom_idx):
    """一遍扫描 -> (n_chrom+1, NBIN) 计数矩阵。最后一行 = 无坐标的位点。"""
    H = np.zeros((len(chrom_idx) + 1, NBIN), dtype=np.int64)
    for p in sorted(glob.glob(os.path.join(score_dir, '*.tsv.gz'))):
        d = pd.read_csv(p, sep='\t', usecols=['read_id', col])
        b = np.clip((d[col].to_numpy() * (NBIN - 1) + 0.5).astype(np.int32), 0, NBIN - 1)
        ci = np.fromiter((chrom_idx.get(cmap.get(u, ''), len(chrom_idx))
                          for u in uid64(d['read_id'].values)), np.int32, len(d))
        np.add.at(H, (ci, b), 1)
    return H


def rate_at(H, thr):
    """给定阈值 -> (总位点, 判正数)。thr 是分数, 转成桶下标后取后缀和。"""
    k = int(np.ceil(thr * (NBIN - 1) - 1e-9))
    k = max(0, min(NBIN, k))
    return int(H.sum()), int(H[:, k:].sum())


def rate_per_chrom(H, thr, chrom_idx):
    k = max(0, min(NBIN, int(np.ceil(thr * (NBIN - 1) - 1e-9))))
    out = {}
    for c, i in chrom_idx.items():
        n = int(H[i].sum())
        if n:
            out[c] = (int(H[i, k:].sum()), n)
    return out


def thr_for_rate(H, target):
    """反解：使总体判正率 = target 的阈值(取满足率<=target 的最小阈值)。"""
    tot = H.sum()
    suf = np.cumsum(H.sum(0)[::-1])[::-1]          # suf[k] = 分数>=k 的个数
    k = int(np.argmax(suf / tot <= target))
    return k / (NBIN - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fprs', nargs='+', type=float, default=[1e-3, 1e-4, 1e-5, 1e-6])
    ap.add_argument('--calib-rate', type=float, default=75e-6, help='校准口径: L 组目标判正率')
    ap.add_argument('--out', default='/home/bio/8oxog/wtd1/dala_sweep.txt')
    args = ap.parse_args()
    from scipy import stats

    chrom_idx = {c: i for i, c in enumerate(MAIN_CHROM)}
    H = {}
    for s, cfg in SAMPLES.items():
        cmap = read_chrom_map(cfg['coords'])
        print('{}: {} 条 read'.format(s, len(cmap)), flush=True)
        for m in PACK_SCORES:
            print('  扫 {} ...'.format(m), flush=True)
            H[(m, s)] = hist_by_chrom(cfg['scores'][m], COL.get(m, 'prob'), cmap, chrom_idx)
        del cmap

    Ls, Ds = 'L (wtl1)', 'D (wtd1)'
    L = ['D-Ala 效应 · 阈值稳健性扫描',
         '每百万**被检测** G（100 种 5-mer context 内的 G，占基因组 G 的 41.2%）',
         '两样本恒用同一阈值；染色体级配对 t 检验（n=23，常染色体+X）', '']

    def block(title, thrs):
        L.append('=== {} ==='.format(title))
        L.append('{:9s} {:>10s} {:>10s} {:>10s} {:>9s} {:>9s} {:>26s}'.format(
            'model', '阈值', 'L/百万', 'D/百万', 'D−L', 'D/L', '染色体检验'))
        for m in PACK_SCORES:
            thr = thrs[m]
            tl, cl = rate_at(H[(m, Ls)], thr)
            td, cd = rate_at(H[(m, Ds)], thr)
            rl, rd = 1e6 * cl / tl, 1e6 * cd / td
            pl = rate_per_chrom(H[(m, Ls)], thr, chrom_idx)
            pdd = rate_per_chrom(H[(m, Ds)], thr, chrom_idx)
            rows = [(k, 1e6 * pl[k][0] / pl[k][1], 1e6 * pdd[k][0] / pdd[k][1])
                    for k in MAIN_CHROM if k in pl and k in pdd and pl[k][1] > 10000]
            a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows])
            t, pt = stats.ttest_rel(b, a)
            L.append('{:9s} {:10.4f} {:10.1f} {:10.1f} {:+9.1f} {:8.3f}x  p={:.1e}, {}/{} 条 D>L'.format(
                m, thr, rl, rd, rd - rl, rd / max(rl, 1e-12), pt, int((b > a).sum()), len(rows)))
        L.append('')

    # ① 等基因组 FPR
    for f in args.fprs:
        block('① 等基因组 FPR = {:.0e}'.format(f),
              {m: thr_at_fpr(PACK_SCORES[m], f) for m in PACK_SCORES})

    # ② 校准到文献本底: 让 L 组的判正率 = calib-rate
    block('② 校准：L 对照组判正率 = {:.0f}/百万（esox 报的基因组本底）'.format(args.calib_rate * 1e6),
          {m: thr_for_rate(H[(m, Ls)], args.calib_rate) for m in PACK_SCORES})

    # ③ esox 论文操作点（其余模型用同 FPR 折算, 便于同表阅读）
    e95 = 0.95
    # esox@0.95 对应的基因组 FPR 也在 valid 基因组阴性上量, 与 thr_at_fpr 同口径
    a = np.load(os.path.join(PACK_SCORES['esox'], 'valid.probs.npz'))
    p = np.nan_to_num(a['prob'].astype(np.float64), nan=-1.0)
    f95 = float((p[np.load(VALID_GENOMIC_MASK)] >= e95).mean())
    L.append('（esox@0.95 对应的基因组 FPR = {:.2e}；下表其余模型按同一 FPR 折算）'.format(f95))
    block('③ esox 论文操作点 score>0.95',
          {m: (e95 if m == 'esox' else thr_at_fpr(PACK_SCORES[m], f95)) for m in PACK_SCORES})

    L += ['边界同 dala_effect.txt：每组仅 1 个 flow cell，处理与批次混杂；',
          'p 值是染色体级、非样本级；染色体间不独立，配对检验偏乐观。']
    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print('\n写出', args.out)


if __name__ == '__main__':
    main()
