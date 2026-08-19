"""native 判正率随 f5 数量的收敛曲线 —— 回答"为什么只用 N 个 f5"。

审稿人若问"为什么不用全部数据", 一条平掉的曲线比任何论证都硬: 再加数据数字不变。
同时它也用来定 wtd1 该处理多少个 f5。

每个模型用自己的 T*(valid 阴性 99.9% 特异性), 逐个 f5 累加, 画累计判正率(每百万 G)。

    /home/bio/anaconda3/bin/python convergence_curve.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

MODELS = {                      # 名称 -> (native 打分目录, 分数列, pack 打分目录)
    'OxoNet': ('/home/bio/8oxog/wtl1/scores/oxonet_e2ep125', 'prob',
               '/home/bio/8oxog/wtl1/pack_scores_e2ep125'),
    'esox': ('/home/bio/8oxog/wtl1/scores/esox', 'oxog_score',
             '/home/bio/8oxog/wtl1/pack_scores_esox'),
    'NanoCon': ('/home/bio/8oxog/nanocon/scores/native', 'prob',
                '/home/bio/8oxog/nanocon/scores'),
}
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'   # dataviz 分类色 1/2/3
COLORS = {'OxoNet': BLUE, 'esox': ORANGE, 'NanoCon': GREEN}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2df'


def tstar(pack_dir, spec):
    p = os.path.join(pack_dir, 'valid.probs.npz')
    if not os.path.isfile(p):
        return None
    a = np.load(p)
    v = np.nan_to_num(a['prob'].astype(np.float64), nan=-1.0)
    return float(np.quantile(v[a['label'] == 0], spec))


def curve(d, col, thr):
    """逐 f5 累加 -> DataFrame(n_f5, sites, called, rate_per_M)"""
    rows = []
    tot = call = 0
    for i, p in enumerate(sorted(glob.glob(os.path.join(d, '*.tsv.gz')))):
        v = pd.read_csv(p, sep='\t', usecols=[col])[col].values
        tot += len(v)
        call += int((v >= thr).sum())
        rows.append((i + 1, tot, call, 1e6 * call / max(tot, 1)))
    return pd.DataFrame(rows, columns=['n_f5', 'sites', 'called', 'per_million_G'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=float, default=0.999)
    ap.add_argument('--out-dir', default='/home/bio/8oxog/wtl1')
    args = ap.parse_args()

    curves = {}
    for m, (d, col, pk) in MODELS.items():
        if not glob.glob(os.path.join(d, '*.tsv.gz')):
            print('跳过 {}: 无 native 打分'.format(m)); continue
        t = tstar(pk, args.spec)
        if t is None:
            print('跳过 {}: 无 valid 打分(定不了 T*)'.format(m)); continue
        c = curve(d, col, t)
        curves[m] = c
        last = c.iloc[-1]
        # 最后 20% 的 f5 里, 累计率的波动幅度 = "还在动吗"
        tail = c['per_million_G'].iloc[int(len(c) * 0.8):]
        print('{:8s} T*={:.4f}  {} 个 f5, {:,} 位点, {:.1f}/百万 G  (末段波动 ±{:.2f}/百万)'.format(
            m, t, int(last.n_f5), int(last.sites), last.per_million_G,
            (tail.max() - tail.min()) / 2))
        c.to_csv(os.path.join(args.out_dir, 'convergence_{}.csv'.format(m)), index=False)

    if not curves:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    fig.patch.set_facecolor('white')
    for m, c in curves.items():
        ax.plot(c.n_f5, c.per_million_G, color=COLORS.get(m, INK2), lw=1.8, label=m)
        ax.annotate(m, xy=(c.n_f5.iloc[-1], c.per_million_G.iloc[-1]), xytext=(4, 0),
                    textcoords='offset points', fontsize=8, color=COLORS.get(m, INK2),
                    va='center')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.8)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel('Number of fast5 files included', fontsize=8.5, color=INK)
    ax.set_ylabel('Cumulative calls per million G', fontsize=8.5, color=INK)
    ax.set_title('Native call rate converges well before the full dataset',
                 fontsize=9, color=INK, loc='left', pad=6)
    ax.set_yscale('log')
    ax.set_xlim(0, max(c.n_f5.iloc[-1] for c in curves.values()) * 1.12)
    for ext in ('pdf', 'png'):
        f = os.path.join(args.out_dir, 'convergence_curve.' + ext)
        fig.savefig(f, dpi=300, bbox_inches='tight', facecolor='white')
        print('写出', f)


if __name__ == '__main__':
    main()
