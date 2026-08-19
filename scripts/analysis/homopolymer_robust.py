"""Homopolymer 分层的稳健性复核 —— 回答"保留率是不是工作点造成的假象"。

`homopolymer_strat.py` 在各模型自己的 T*(valid 阴性 99.9% 特异性) 上比 recall,
但各模型在 T* 处的总体 recall 差一倍(OxoNet 70% vs esox 40%)。ROC 曲线上的位置
不同, 退化幅度本身就可能不同 → **保留率(4+/1) 必须在等灵敏度下复核**。

本脚本出四件事:
  1. 各 stratum 的 5-mer context 组成(4+ 是不是被单个 context 支配)
  2. 等灵敏度扫描: 把每个模型的阈值调到**总体 recall 相同**, 再比 4+/1 保留率
  3. esox 在其论文自己的操作点 0.95 上的保留率(不能只用我们选的阈值评它)
  4. 保留率的 bootstrap 95% CI(4+ 档只有 6,837 个阳性)

    /home/bio/anaconda3/bin/python homopolymer_robust.py
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
ESOX_PUBLISHED_T = 0.95      # esox 论文自报的操作点, 见 [[esox-operating-point]]


def probs_path(d, name):
    for cand in (os.path.join(d, name + '.probs.npz'),
                 os.path.join(d, 'gbdt_' + name + '.probs.npz')):
        if os.path.isfile(cand):
            return cand
    raise SystemExit('找不到 {} 的 {} 打分'.format(d, name))


def grun(kmers):
    """中心(index 3)所在的连续 G 段长度, 上限 7(7-mer 窗口内可见的部分)。"""
    out = np.empty(len(kmers), np.int8)
    for i, s in enumerate(kmers):
        s = s.replace('o', 'G')
        l = r = 3
        while l > 0 and s[l - 1] == 'G':
            l -= 1
        while r < 6 and s[r + 1] == 'G':
            r += 1
        out[i] = r - l + 1
    return out


def recall_at(p_pos, t):
    return float((p_pos >= t).mean())


def thr_for_recall(p_pos, target):
    """找使总体 recall = target 的阈值(分数从大到小的分位点)。"""
    return float(np.quantile(p_pos, 1.0 - target))


def boot_retention(p_pos, g_pos, t, n=2000, seed=0):
    """保留率 (4+ recall / 1 recall) 的 bootstrap CI, 按 stratum 内重采样。"""
    rng = np.random.default_rng(seed)
    a = (p_pos[g_pos == 1] >= t)
    b = (p_pos[g_pos == 4] >= t)
    ra = rng.binomial(len(a), a.mean(), n) / len(a)
    rb = rng.binomial(len(b), b.mean(), n) / len(b)
    r = rb / np.maximum(ra, 1e-12)
    return float(np.quantile(r, .025)), float(np.quantile(r, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='test_oligo', choices=['test_oligo', 'test_t2t'],
                    help='阳性两集共用, 只影响阴性/FPR; 保留率只用阳性故两者同')
    ap.add_argument('--spec', type=float, default=0.999)
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/homopolymer_robust.txt')
    args = ap.parse_args()

    src = np.load(os.path.join(PACK, args.set + '.npz'), allow_pickle=False)
    kmers = src['kmers'].astype('U7')
    lab = src['label'].astype(int)
    g = np.minimum(grun(kmers.tolist()), 4)
    pos = lab == 1
    g_pos = g[pos]
    L = []

    # ---------- 1. stratum 的 context 组成 ----------
    L.append('=== 1. 各 stratum 的 5-mer context 组成(阳性位点) ===')
    ctx = np.array([s[1:6] for s in kmers])
    for k in (1, 2, 3, 4):
        sel = (g == k) & pos
        u, c = np.unique(ctx[sel], return_counts=True)
        o = np.argsort(-c)
        top = ', '.join('{} {:.0%}'.format(u[i], c[i] / c.sum()) for i in o[:5])
        L.append('  G段={}: {:,} 个阳性, {} 种 context, 最大占比 {:.0%} | top5: {}'
                 .format('4+' if k == 4 else k, int(sel.sum()), len(u), c[o[0]] / c.sum(), top))

    # ---------- 载入各模型 ----------
    P, T = {}, {}
    for m, d in MODELS.items():
        v = np.load(probs_path(d, 'valid'))
        vp = np.nan_to_num(v['prob'].astype(np.float64), nan=-1.0)
        T[m] = float(np.quantile(vp[v['label'] == 0], args.spec))
        P[m] = np.nan_to_num(np.load(probs_path(d, args.set))['prob'].astype(np.float64),
                             nan=-1.0)[pos]

    def row(tag, m, t):
        p = P[m]
        rec = [(p[g_pos == k] >= t).mean() for k in (1, 2, 3, 4)]
        ret = rec[3] / max(rec[0], 1e-12)
        lo, hi = boot_retention(p, g_pos, t)
        return ('  {:22s} {:>8.2%} {:>8.2%} {:>8.2%} {:>8.2%} | 总体 {:>6.2%} | 保留 {:>5.0%} '
                '[{:.0%},{:.0%}]'.format(tag, *rec, (p >= t).mean(), ret, lo, hi))

    L.append('')
    L.append('=== 2. 各模型自己的 T*(valid 99.9% 特异性) —— 原表 ===')
    L.append('  {:22s} {:>8s} {:>8s} {:>8s} {:>8s}'.format('', 'G段=1', '2', '3', '4+'))
    for m in MODELS:
        L.append(row('{} @T*={:.4f}'.format(m, T[m]), m, T[m]))
    L.append(row('esox @0.95(其论文操作点)', 'esox', ESOX_PUBLISHED_T))

    # ---------- 3. 等灵敏度 ----------
    L.append('')
    L.append('=== 3. 等灵敏度复核: 把所有模型调到**相同总体 recall**, 再比保留率 ===')
    L.append('  (若保留率排序在每一档都不变, 则退化幅度不是工作点造成的)')
    for target in (0.25, 0.35, 0.4046, 0.50, 0.60, 0.70):
        L.append('  --- 总体 recall = {:.2%} ---'.format(target))
        for m in MODELS:
            t = thr_for_recall(P[m], target)
            L.append(row('{} @{:.4f}'.format(m, t), m, t))

    # ---------- 4. 等基因组 FPR(与论文其余部分同口径) ----------
    L.append('')
    L.append('=== 4. 等基因组 FPR 复核: 阈值定在 **valid 的基因组阴性**上, 比各 stratum 的绝对 recall ===')
    L.append('  (这是论文主口径 —— 稀有损伤检测里该对齐的是假阳而非灵敏度; 阈值只在 valid 上定,')
    L.append('   测试集只报结果, 见 METHOD_AUDIT.md。1e-6 档不列: 低于分辨力)')
    gmask = np.load(os.path.join(PACK, 'valid_genomic_neg_mask.npy'))
    for fpr in (1e-3, 1e-4, 1e-5):
        L.append('  --- 基因组 FPR = {:g} ---'.format(fpr))
        for m in MODELS:
            vneg = np.nan_to_num(np.load(probs_path(MODELS[m], 'valid'))['prob']
                                 .astype(np.float64), nan=-1.0)[gmask]
            t = float(np.quantile(vneg, 1.0 - fpr))
            L.append(row('{} @{:.4f}'.format(m, t), m, t))

    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print('\n写出', args.out)


if __name__ == '__main__':
    main()
