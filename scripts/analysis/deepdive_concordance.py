"""Deep-dive concordance (esox vs OxoNet) on native G sites -> markdown report + figures.

Four cuts to characterise WHERE/WHY the two models (dis)agree (global Spearman was only
~0.17, OxoNet over-calls 33-41%). Consistency check, NOT accuracy. All 100-context sites.

  (A) 5mer stratification   : per-motif means/pos-rates; do the models agree on WHICH
                              motifs are hot? (motif-level Spearman)
  (B) score joint density   : 2D histogram esox_score x oxonet_prob
  (C) OxoNet-decile -> esox : is OxoNet's ranking at all monotone in esox?
  (D) esox high-end locality : correlation & OxoNet response restricted to esox's own
                              confident calls (agreement at the confident end?)

    /home/bio/anaconda3/bin/python oxo/analysis/deepdive_concordance.py
Outputs: wt1/CONCORDANCE_DEEPDIVE.md  +  wt1/figs/*.png
"""
import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sp_sub(x, y, k=3_000_000, seed=0):
    """Spearman with subsample (rank-sort on 40M+ is slow)."""
    if len(x) > k:
        idx = np.random.default_rng(seed).choice(len(x), k, replace=False)
        x, y = x[idx], y[idx]
    return spearmanr(x, y)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', default='/home/bio/8oxog/wtl1/sites_scores.parquet')
    ap.add_argument('--figdir', default='/home/bio/8oxog/wtl1/figs')
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/CONCORDANCE_DEEPDIVE.md')
    ap.add_argument('--esox-thr', type=float, default=0.5)
    ap.add_argument('--oxonet-thr', type=float, default=0.665585)
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    df = pd.read_parquet(args.sites, columns=['oxog_score', 'oxonet_prob', '5mer'])
    e = df['oxog_score'].to_numpy(np.float64)
    o = df['oxonet_prob'].to_numpy(np.float64)
    n = len(e)
    et, ot = args.esox_thr, args.oxonet_thr
    md = ['# Native 一致性深挖 — esox vs OxoNet\n',
          '> consistency check (非 accuracy);全部 100-context native G 位点;'
          'esox 阈值 {:.3f}、OxoNet T*={:.6f}(oligo 99.9%-spec)。\n'.format(et, ot),
          '**位点数**: {:,}  (40 fast5)\n'.format(n)]

    g_sp = sp_sub(e, o); g_pr = pearsonr(e, o)[0]
    md.append('**全局**: Spearman={:.4f}  Pearson={:.4f}  |  '
              'esox+@{:.2f}={:.3%}  OxoNet+@T*={:.3%}\n'.format(
                  g_sp, g_pr, et, (e >= et).mean(), (o >= ot).mean()))

    # ---------- (A) 5mer stratification ----------
    df['e_pos'] = e >= et
    df['o_pos'] = o >= ot
    grp = df.groupby('5mer', observed=True)
    tab = grp.agg(n=('oxog_score', 'size'),
                  esox_mean=('oxog_score', 'mean'),
                  oxonet_mean=('oxonet_prob', 'mean'),
                  esox_posrate=('e_pos', 'mean'),
                  oxonet_posrate=('o_pos', 'mean')).reset_index()
    tab = tab[tab['n'] >= 200].sort_values('n', ascending=False)
    # motif-level agreement: do models rank motifs the same way?
    motif_sp = spearmanr(tab['esox_mean'], tab['oxonet_mean'])[0]
    motif_sp_pos = spearmanr(tab['esox_posrate'], tab['oxonet_posrate'])[0]
    md.append('\n## (A) 5mer 分层\n')
    md.append('- motif 数(n>=200): {}\n'.format(len(tab)))
    md.append('- **motif-level Spearman**: 均值 {:.4f} / 判正率 {:.4f} '
              '(两模型是否认同"哪些 motif 是热点")\n'.format(motif_sp, motif_sp_pos))
    top_e = tab.sort_values('esox_mean', ascending=False).head(10)
    top_o = tab.sort_values('oxonet_mean', ascending=False).head(10)
    md.append('\nesox 判 8-oxoG 最高的 10 个 motif:\n\n')
    md.append('| 5mer | n | esox_mean | esox_pos | oxonet_mean | oxonet_pos |\n|---|---|---|---|---|---|\n')
    for _, r in top_e.iterrows():
        md.append('| {} | {:,} | {:.4f} | {:.3%} | {:.4f} | {:.3%} |\n'.format(
            r['5mer'], int(r['n']), r['esox_mean'], r['esox_posrate'], r['oxonet_mean'], r['oxonet_posrate']))
    md.append('\nOxoNet 打分最高的 10 个 motif:\n\n')
    md.append('| 5mer | n | oxonet_mean | oxonet_pos | esox_mean | esox_pos |\n|---|---|---|---|---|---|\n')
    for _, r in top_o.iterrows():
        md.append('| {} | {:,} | {:.4f} | {:.3%} | {:.4f} | {:.3%} |\n'.format(
            r['5mer'], int(r['n']), r['oxonet_mean'], r['oxonet_posrate'], r['esox_mean'], r['esox_posrate']))

    # motif scatter
    plt.figure(figsize=(5.5, 5))
    plt.scatter(tab['esox_mean'], tab['oxonet_mean'],
                s=np.sqrt(tab['n']) / 3, alpha=0.5, edgecolors='none')
    plt.xlabel('esox mean score (per 5mer)'); plt.ylabel('OxoNet mean prob (per 5mer)')
    plt.title('motif-level: esox vs OxoNet (Spearman {:.3f})'.format(motif_sp))
    plt.tight_layout(); plt.savefig(os.path.join(args.figdir, 'A_motif_scatter.png'), dpi=110); plt.close()
    tab.sort_values('oxonet_mean', ascending=False).to_csv(
        os.path.join(args.figdir, 'A_motif_table.csv'), index=False)
    md.append('\n![motif scatter](figs/A_motif_scatter.png)\n')
    md.append('\n全表: `figs/A_motif_table.csv`\n')

    # ---------- (B) joint density ----------
    plt.figure(figsize=(6, 5))
    h = plt.hist2d(e, o, bins=[np.linspace(0, 1, 101), np.linspace(0, 1, 101)],
                   cmin=1, norm=matplotlib.colors.LogNorm())
    plt.colorbar(h[3], label='sites (log)')
    plt.axvline(et, color='r', lw=0.7, ls='--'); plt.axhline(ot, color='r', lw=0.7, ls='--')
    plt.xlabel('esox_score'); plt.ylabel('oxonet_prob'); plt.title('joint density')
    plt.tight_layout(); plt.savefig(os.path.join(args.figdir, 'B_joint_hist2d.png'), dpi=110); plt.close()
    md.append('\n## (B) score 联合密度\n')
    md.append('![joint](figs/B_joint_hist2d.png)\n')
    md.append('\n红虚线 = 各自阈值。左下=两者皆负,右上=两者皆正(共识)。\n')

    # ---------- (C) OxoNet decile -> esox ----------
    edges = np.linspace(0, 1, 11)
    obin = np.clip(np.digitize(o, edges[1:-1]), 0, 9)
    rows = []
    for b in range(10):
        m = obin == b
        rows.append((edges[b], edges[b + 1], int(m.sum()),
                     e[m].mean() if m.any() else np.nan,
                     (e[m] >= et).mean() if m.any() else np.nan))
    cdf = pd.DataFrame(rows, columns=['o_lo', 'o_hi', 'n', 'esox_mean', 'esox_posrate'])
    md.append('\n## (C) 按 OxoNet 分数分箱看 esox\n\n')
    md.append('| OxoNet 区间 | n | esox_mean | esox_pos率 |\n|---|---|---|---|\n')
    for _, r in cdf.iterrows():
        md.append('| [{:.1f},{:.1f}) | {:,} | {:.4f} | {:.3%} |\n'.format(
            r['o_lo'], r['o_hi'], int(r['n']), r['esox_mean'], r['esox_posrate']))
    plt.figure(figsize=(6, 4))
    xc = (cdf['o_lo'] + cdf['o_hi']) / 2
    plt.plot(xc, cdf['esox_posrate'], 'o-', label='esox pos-rate @{:.2f}'.format(et))
    plt.plot(xc, cdf['esox_mean'], 's--', label='esox mean score')
    plt.xlabel('OxoNet prob bin'); plt.ylabel('esox response'); plt.legend()
    plt.title('does higher OxoNet -> higher esox?')
    plt.tight_layout(); plt.savefig(os.path.join(args.figdir, 'C_oxonet_bin_esox.png'), dpi=110); plt.close()
    md.append('\n![C](figs/C_oxonet_bin_esox.png)\n')

    # ---------- (D) esox high-end locality ----------
    md.append('\n## (D) esox 高置信端的局部一致\n\n')
    md.append('| esox 分位 | 阈值 | 子集 n | 子集内 OxoNet+率 | 子集内 OxoNet 中位数 | 局部 Spearman |\n')
    md.append('|---|---|---|---|---|---|\n')
    for q in [0.0, 0.9, 0.99, 0.999]:
        thr = np.quantile(e, q) if q > 0 else -1
        m = e >= thr
        sub_sp = sp_sub(e[m], o[m]) if m.sum() > 50 else np.nan
        md.append('| top {:.1%} | esox>={:.4f} | {:,} | {:.3%} | {:.4f} | {:.4f} |\n'.format(
            1 - q, max(thr, 0), int(m.sum()), (o[m] >= ot).mean(), np.median(o[m]), sub_sp))
    # also: among esox positives, oxonet distribution
    ep = e >= et
    md.append('\nesox 判正位点({:,}) 的 OxoNet 分布: mean={:.4f} median={:.4f} '
              'OxoNet+率={:.3%}\n'.format(int(ep.sum()), o[ep].mean(), np.median(o[ep]), (o[ep] >= ot).mean()))

    md.append('\n---\n_由 `oxo/analysis/deepdive_concordance.py` 生成;重跑即复现。_\n')
    with open(args.out, 'w') as fh:
        fh.write(''.join(md))
    print('wrote', args.out)
    print('figs in', args.figdir)


if __name__ == '__main__':
    main()
