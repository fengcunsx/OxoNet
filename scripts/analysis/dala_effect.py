"""D-Ala 效应：wtl1(L-Alanine 对照) vs wtd1(D-Alanine 诱导氧化压力)。

这是整个 resubmission 里**唯一方向已知的生物学阳性对照**（答 R2.M1）：
esox 论文报 D 组比 L 组高 **+12/百万 G（+16%, p=0.016）**，但那个 p 值是
**3 个 L 样本 vs 3 个 D 样本**的两侧 t 检验（Fig 4a 图注 "n = 3"）。
我们只有 1 对样本 → **给不出样本级 p 值**，退而做**染色体级配对检验**
（n=23；esox 自己 Fig 4c 也做染色体级 t 检验）。这个统计量的边界必须写清楚：
它检验的是"效应在染色体间是否一致地为正"，**分不开 flow cell 批次效应**。

阈值：**两个样本用同一个阈值**（等绝对基因组 FPR = 1e-4，在 **valid 的基因组阴性**上定），
否则率的差异无从解释。

每条 read 只有一个 primary 比对 → 用 **read_id → chrom** 映射（~52 万条）做染色体归属，
不必把 1.7 亿行坐标全读进内存。

    /home/bio/anaconda3/bin/python dala_effect.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

PACK_SCORES = {          # 定阈值用的 pack 打分（两样本共用）
    'OxoNet': '/home/bio/8oxog/wtl1/pack_scores_e2ep125',
    'esox': '/home/bio/8oxog/wtl1/pack_scores_esox',
    'NanoCon': '/home/bio/8oxog/nanocon/scores',
}
SAMPLES = {
    'L (wtl1)': {
        'coords': '/home/bio/8oxog/wtl1/coords/all_mapq.tsv.gz',
        'scores': {'OxoNet': '/home/bio/8oxog/wtl1/scores/oxonet_e2ep125',
                   'esox': '/home/bio/8oxog/wtl1/scores/esox',
                   'NanoCon': '/home/bio/8oxog/nanocon/scores/native'},
    },
    'D (wtd1)': {
        'coords': '/home/bio/8oxog/wtd1/coords/all_mapq.tsv.gz',
        'scores': {'OxoNet': '/home/bio/8oxog/wtd1/scores/oxonet_e2ep125',
                   'esox': '/home/bio/8oxog/wtd1/scores/esox',
                   'NanoCon': '/home/bio/8oxog/wtd1/scores/nanocon/native'},
    },
    # 第二张 D flow cell(FAT93061)。用 --d-sample 'D (wtd2)' 选它;
    # 默认仍是 wtd1,所以论文里已有的 wtd1 数字不受影响。
    'D (wtd2)': {
        'coords': '/home/bio/8oxog/wtd2/coords/all_mapq.tsv.gz',
        'scores': {'OxoNet': '/home/bio/8oxog/wtd2/scores/oxonet_e2ep125',
                   'esox': '/home/bio/8oxog/wtd2/scores/esox',
                   'NanoCon': '/home/bio/8oxog/wtd2/scores/nanocon/native'},
    },
    # 2026-08-15 新增的 p53-/- 一对(flow cell FAT92817 / FAT92828)。
    # 这两个是**同基因型的 L/D 配对**,不像 wtl1(WT) vs wtd1/wtd2(WT) 那样只有一个 L 对照;
    # 用 --l-sample 'L (p53l1)' --d-sample 'D (p53d1)' 取用。默认值未动,旧结果逐字不变。
    'L (p53l1)': {
        'coords': '/home/bio/8oxog/p53l1/coords/all_mapq.tsv.gz',
        'scores': {'OxoNet': '/home/bio/8oxog/p53l1/scores/oxonet_e2ep125',
                   'esox': '/home/bio/8oxog/p53l1/scores/esox',
                   'NanoCon': '/home/bio/8oxog/p53l1/scores/nanocon/native'},
    },
    'D (p53d1)': {
        'coords': '/home/bio/8oxog/p53d1/coords/all_mapq.tsv.gz',
        'scores': {'OxoNet': '/home/bio/8oxog/p53d1/scores/oxonet_e2ep125',
                   'esox': '/home/bio/8oxog/p53d1/scores/esox',
                   'NanoCon': '/home/bio/8oxog/p53d1/scores/nanocon/native'},
    },
    # 2026-08-19 新增: 第 6 个也是最后一个 native 样本(flow cell FAX39961, WT + L-Ala 重复 2)。
    # 补上它，样本级才从 2L vs 3D 变成 **3L vs 3D**：完全分离的精确单侧概率
    # 由 1/C(5,2)=0.10 降到 1/C(6,3)=0.05，才够得着常规显著性。默认值未动，旧结果逐字不变。
    'L (wtl2)': {
        'coords': '/home/bio/8oxog/wtl2/coords/all_mapq.tsv.gz',
        'scores': {'OxoNet': '/home/bio/8oxog/wtl2/scores/oxonet_e2ep125',
                   'esox': '/home/bio/8oxog/wtl2/scores/esox',
                   'NanoCon': '/home/bio/8oxog/wtl2/scores/nanocon/native'},
    },
}
COL = {'esox': 'oxog_score'}          # 其余 'prob'
MAIN_CHROM = ['chr' + c for c in [str(i) for i in range(1, 23)] + ['X']]


def uid64(s):
    return np.array([int(x[:8] + x[9:13] + x[14:18], 16) & 0x7FFFFFFFFFFFFFFF
                     for x in s], dtype=np.int64)


VALID_GENOMIC_MASK = '/home/bio/8oxog/build/pack/valid_genomic_neg_mask.npy'


def thr_at_fpr(pack_dir, fpr):
    """阈值定在 **valid 的基因组阴性**上(2026-08-03 口径, 见 METHOD_AUDIT.md):
    测试集只报结果, 不参与定工作点。掩码由 valid_neg_source.py 生成。
    dala_sweep.py 直接 import 本函数, 故两处口径一致。"""
    a = np.load(os.path.join(pack_dir, 'valid.probs.npz'))
    p = np.nan_to_num(a['prob'].astype(np.float64), nan=-1.0)
    return float(np.quantile(p[np.load(VALID_GENOMIC_MASK)], 1 - fpr))


def read_chrom_map(coords):
    """read_id(uid64) -> chrom。每条 read 只有一个 primary 比对，所以一对一。"""
    m = {}
    for ch in pd.read_csv(coords, sep='\t', usecols=['read_id', 'chrom'], chunksize=5_000_000):
        d = ch.drop_duplicates('read_id')
        for u, c in zip(uid64(d['read_id'].values), d['chrom'].values):
            m.setdefault(u, c)
    return m


def tally(score_dir, col, thr, cmap):
    """遍历该样本该模型的 150 个 tsv.gz，统计总体与逐染色体的 (判正数, 位点数)。"""
    tot = called = 0
    per = {}
    for p in sorted(glob.glob(os.path.join(score_dir, '*.tsv.gz'))):
        d = pd.read_csv(p, sep='\t', usecols=['read_id', col])
        u = uid64(d['read_id'].values)
        c = d[col].to_numpy() >= thr
        tot += len(c); called += int(c.sum())
        chrom = np.array([cmap.get(x, '') for x in u])
        for k in np.unique(chrom):
            if not k:
                continue
            s = chrom == k
            a, b = per.get(k, (0, 0))
            per[k] = (a + int(c[s].sum()), b + int(s.sum()))
    return tot, called, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--match-fpr', type=float, default=1e-4)
    ap.add_argument('--out', default='/home/bio/8oxog/wtd1/dala_effect.txt')
    ap.add_argument('--l-sample', default='L (wtl1)', choices=sorted(SAMPLES))
    ap.add_argument('--d-sample', default='D (wtd1)', choices=sorted(SAMPLES))
    args = ap.parse_args()
    from scipy import stats
    KL, KD = args.l_sample, args.d_sample

    L = ['D-Ala 效应：{}(L-Ala 对照) vs {}(D-Ala 诱导)'.format(KL, KD),
         '两个样本使用**同一阈值**（等绝对基因组 FPR = {:.0e}，在 **valid 的基因组阴性**上定）'.format(
             args.match_fpr),
         '率的分母 = 被检测的 G（中心 5-mer 属于那 100 种 context 者），不是全部基因组 G',
         '']

    cmaps = {}
    for s in (KL, KD):
        cfg = SAMPLES[s]
        print('读 {} 的 read->chrom 映射 ...'.format(s), flush=True)
        cmaps[s] = read_chrom_map(cfg['coords'])
        print('  {} 条 read'.format(len(cmaps[s])), flush=True)

    L.append('{:10s} {:>14s} {:>14s} {:>12s} {:>12s} {:>9s}'.format(
        'model', 'L 判正/百万', 'D 判正/百万', 'D-L', 'D/L', '染色体检验'))
    detail = {}
    for m in PACK_SCORES:
        thr = thr_at_fpr(PACK_SCORES[m], args.match_fpr)
        res = {}
        for s in (KL, KD):
            print('统计 {} / {} ...'.format(m, s), flush=True)
            res[s] = tally(SAMPLES[s]['scores'][m], COL.get(m, 'prob'), thr, cmaps[s])
        (tl, cl, pl), (td, cd, pd_) = res[KL], res[KD]
        rl, rd = 1e6 * cl / tl, 1e6 * cd / td

        # 染色体级配对：只用常染色体+X，且两边都要有足够位点
        rows = []
        for k in MAIN_CHROM:
            if k in pl and k in pd_ and pl[k][1] > 10000 and pd_[k][1] > 10000:
                rows.append((k, 1e6 * pl[k][0] / pl[k][1], 1e6 * pd_[k][0] / pd_[k][1]))
        a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows])
        t, pt = stats.ttest_rel(b, a)
        w, pw = stats.wilcoxon(b, a)
        L.append('{:10s} {:14.1f} {:14.1f} {:+12.1f} {:11.3f}x  n={} 配对t p={:.2g}'.format(
            m, rl, rd, rd - rl, rd / rl, len(rows), pt))
        L.append('{:10s} {:>14s} {:>14s} {:>12s} {:>12s}  Wilcoxon p={:.2g}, {}/{} 条染色体 D>L'.format(
            '', '', '', '', '', pw, int((b > a).sum()), len(rows)))
        L.append('{:10s}   阈值 {:.4f}；位点数 L={:,} D={:,}'.format('', thr, tl, td))
        detail[m] = rows

    L += ['', '=== 逐染色体（每百万被检测 G） ===']
    for m, rows in detail.items():
        L.append('--- {} ---'.format(m))
        L.append('  ' + ' '.join('{:>7s}'.format(r[0].replace('chr', '')) for r in rows))
        L.append('L ' + ' '.join('{:7.0f}'.format(r[1]) for r in rows))
        L.append('D ' + ' '.join('{:7.0f}'.format(r[2]) for r in rows))
        L.append('Δ ' + ' '.join('{:+7.0f}'.format(r[2] - r[1]) for r in rows))

    L += ['',
          '边界（必须写进论文）：',
          '  ① 每组只有 1 个 flow cell，**处理效应与批次效应完全混杂**，上面的 p 值是',
          '     染色体级的、不是样本级的，不能等同于 esox 的 p=0.016（那是 3 vs 3）',
          '  ② 染色体之间并非独立（共享文库制备、测序批次），配对检验偏乐观',
          '  ③ 率的分母是"被检测的 G"（100 种 5-mer context，占基因组 G 的 41.2%）']
    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print('\n写出', args.out)


if __name__ == '__main__':
    main()
