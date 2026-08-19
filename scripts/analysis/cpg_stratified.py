"""CpG 分层分析: 假阳是不是集中在 CpG 上下文, 以及这是不是 5mC 造成的。

背景: valid 上 ep125 最自信的那批假阳里 24% 落在含 CG 的 5-mer 上, 而阴性位点里
只有 10.9% 是 CG 上下文(富集 2.2x)。esox 论文独立观察到 "k-mers rich in cytosine
and guanine performing worse"(Supp Fig 17), 且排除了训练覆盖度的解释(Supp Fig 12)。

假说: T2T/native 是天然基因组 DNA, CpG 在体内甲基化(5mC), 信号偏离未修饰模型 ->
被判成 8-oxoG。oligo 是化学合成的, 没有甲基化。

朴素的全局对比会被 5-mer 组成混淆(人类基因组 CpG 被耗竭, oligo 文库是 110 个
5-mer 均衡设计), 所以这里**按 5-mer 配对比较** oligo vs T2T, 再看 CpG/非CpG 两组
的差异是否不同。这样 5-mer 组成被完全控制住。

    python cpg_stratified.py --pack-scores /home/bio/8oxog/wtl1/pack_scores_e2ep125
"""
import argparse
import os
import numpy as np

PACK = '/home/bio/8oxog/build/pack'


def central5(kmers):
    return np.array([s[1:6] for s in kmers])


def tstar(prob, label, spec):
    """valid 上达到目标 spec 的最小阈值"""
    neg = np.sort(prob[label == 0])[::-1]
    k = int(len(neg) * (1.0 - spec))
    return float(np.nextafter(float(neg[k]), 1.0))


def load(pack_scores, name):
    s = np.load(os.path.join(pack_scores, name + '.probs.npz'))
    src = np.load(os.path.join(PACK, name + '.npz'), allow_pickle=False)
    assert len(s['prob']) == len(src['label']), name
    assert (s['label'] == src['label'].astype(np.int8)).all(), name + ' label 不对齐'
    return s['prob'], src['label'].astype(int), central5(src['kmers'])


def main(args):
    vp = np.load(os.path.join(args.pack_scores, 'valid.probs.npz'))
    for spec in args.specs:
        thr = tstar(vp['prob'], vp['label'], spec)
        print('=' * 78)
        print('操作点 spec={:.4g}% on valid  ->  T* = {:.4f}'.format(spec * 100, thr))
        print('=' * 78)

        per = {}
        for name in ['test_oligo', 'test_t2t']:
            prob, label, c5 = load(args.pack_scores, name)
            nm = label == 0
            cpg = np.char.find(c5, 'CG') >= 0
            fp = (prob > thr) & nm
            print('\n{:<11} 阴性 n={:>9,}  CG上下文占 {:>5.1f}%   全局 FPR={:.4f}%'.format(
                name, int(nm.sum()), 100 * cpg[nm].mean(), 100 * fp.sum() / nm.sum()))
            print('   {:<8} CG上下文 FPR={:.4f}%   非CG FPR={:.4f}%   富集 {:.2f}x'.format(
                '', 100 * fp[nm & cpg].sum() / max((nm & cpg).sum(), 1),
                100 * fp[nm & ~cpg].sum() / max((nm & ~cpg).sum(), 1),
                (fp[nm & cpg].mean() / max(fp[nm & ~cpg].mean(), 1e-12))))
            d = {}
            for c in np.unique(c5[nm]):
                m = nm & (c5 == c)
                if m.sum() >= args.min_neg:
                    d[c] = (int(fp[m].sum()), int(m.sum()))
            per[name] = d

        # ---- 按 5-mer 配对: 完全控制住 5-mer 组成 ----
        common = sorted(set(per['test_oligo']) & set(per['test_t2t']))
        print('\n--- 按 5-mer 配对比较 (两边阴性都 >= {} 条的 {} 个 5-mer) ---'.format(
            args.min_neg, len(common)))
        rows = []
        for c in common:
            fo, no = per['test_oligo'][c]
            ft, nt = per['test_t2t'][c]
            rows.append((c, 'CG' in c, fo / no, ft / nt, no, nt, fo, ft))
        for grp, flag in [('含 CG', True), ('非 CG', False)]:
            g = [r for r in rows if r[1] == flag]
            if not g:
                continue
            fo = sum(r[6] for r in g); no = sum(r[4] for r in g)
            ft = sum(r[7] for r in g); nt = sum(r[5] for r in g)
            # 每个 5-mer 内部比, 再取中位数 -> 不被大 5-mer 主导
            ratios = [(r[3] + 1e-9) / (r[2] + 1e-9) for r in g if r[2] > 0 or r[3] > 0]
            print('  {}  n_5mer={:>3}   oligo FPR={:.4f}%  T2T FPR={:.4f}%   '
                  'T2T/oligo 合计={:.2f}x  逐5mer中位={:.2f}x'.format(
                      grp, len(g), 100 * fo / no, 100 * ft / nt,
                      (ft / nt) / max(fo / no, 1e-12),
                      float(np.median(ratios)) if ratios else float('nan')))
        print('\n  解读: 若 5mC 是主因 -> 含CG 组的 T2T/oligo 比值应显著 >1, 非CG 组应 ~1。')
        print('        若两组比值相近 -> 是 CpG 序列/信号本身的问题, 与甲基化无关。')

        if args.verbose:
            print('\n  逐 5-mer (T2T/oligo 比值降序, 前 15):')
            rows.sort(key=lambda r: -((r[3] + 1e-9) / (r[2] + 1e-9)))
            print('   5mer   CG   oligo_FPR   T2T_FPR    比值   n_oligo  n_t2t')
            for c, cg, a, b, no, nt, _, _ in rows[:15]:
                print('   {}  {:>3}   {:.5f}   {:.5f}   {:>6.2f}   {:>7,} {:>7,}'.format(
                    c, 'CG' if cg else '-', a, b, (b + 1e-9) / (a + 1e-9), no, nt))
        print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-scores', default='/home/bio/8oxog/wtl1/pack_scores_e2ep125')
    ap.add_argument('--specs', type=float, nargs='+', default=[0.999, 0.9999])
    ap.add_argument('--min-neg', type=int, default=500)
    ap.add_argument('--verbose', action='store_true')
    main(ap.parse_args())
