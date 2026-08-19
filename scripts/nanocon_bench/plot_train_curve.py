"""NanoCon 训练收敛曲线（supplementary 图）。

要传达的一件事：**它已经收敛**，不是我们提前停的。所以画的是逐 epoch 的
验证集 AUPRC（它自己的选模型指标）+ 验证集 loss，并把最后 5 个 epoch 的
波动量直接标在图上。

两个量尺度不同 → **两个 panel，不做双 y 轴**。每个 panel 单条线，标题即标识，
不需要图例。

    /home/bio/anaconda3/bin/python plot_train_curve.py \
        --csv /home/bio/8oxog/nanocon/train_curve.csv --out-dir /home/bio/8oxog/nanocon
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BLUE = '#2a78d6'      # 分类色 slot 1
ORANGE = '#eb6834'    # 分类色 slot 2  (validate_palette: 全部 PASS @ 白底)
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e3e2df'


def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='/home/bio/8oxog/nanocon/train_curve.csv')
    p.add_argument('--out-dir', default='/home/bio/8oxog/nanocon')
    p.add_argument('--plateau', type=int, default=5, help='标注最后这么多个 epoch 的波动')
    args = p.parse_args()

    d = pd.read_csv(args.csv)
    d['epoch'] = d['epoch'].ffill()
    v = d.dropna(subset=['avg_val_AUPRC']).groupby('epoch').last().reset_index()
    ep = v['epoch'].values
    ap = v['avg_val_AUPRC'].values
    lo = v['avg_val_loss'].values
    best = int(ep[ap.argmax()])
    tail = ap[-args.plateau:]
    d5 = tail.max() - tail.min()
    d10 = ap[-1] - ap[-10]

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True,
                             gridspec_kw={'height_ratios': [1.35, 1], 'hspace': 0.18})
    fig.patch.set_facecolor('white')

    # --- (a) 验证集 AUPRC = 它自己的选模型指标 ---
    ax = axes[0]
    style(ax)
    ax.axvspan(ep[-args.plateau], ep[-1], color=BLUE, alpha=0.07, lw=0, zorder=0)
    ax.plot(ep, ap, color=BLUE, lw=1.8, zorder=3)
    ax.plot(ep, ap, 'o', color=BLUE, ms=3.2, mec='white', mew=0.6, zorder=4)
    ax.plot([best], [ap.max()], 'o', color=BLUE, ms=7, mec='white', mew=1.2, zorder=5)
    ax.annotate('selected checkpoint\n(epoch {}, AUPRC {:.4f})'.format(best, ap.max()),
                xy=(best, ap.max()), xytext=(best - 9.5, ap.max() - 0.035),
                fontsize=7.5, color=INK2, ha='left',
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.7,
                                shrinkA=0, shrinkB=4))
    ax.annotate('plateau: range {:.4f}\nover final {} epochs'.format(d5, args.plateau),
                xy=(ep[-args.plateau] - 0.3, ap.min() + 0.02), fontsize=7.5,
                color=INK2, ha='right', va='bottom')
    ax.set_ylabel('Validation AUPRC', fontsize=8.5, color=INK)
    ax.set_title('(a)  NanoCon validation AUPRC (the authors\' model-selection metric)',
                 fontsize=9, color=INK, loc='left', pad=6)

    # --- (b) 验证集 loss ---
    ax = axes[1]
    style(ax)
    ax.axvspan(ep[-args.plateau], ep[-1], color=ORANGE, alpha=0.07, lw=0, zorder=0)
    ax.plot(ep, lo, color=ORANGE, lw=1.8, zorder=3)
    ax.plot(ep, lo, 'o', color=ORANGE, ms=3.2, mec='white', mew=0.6, zorder=4)
    ax.set_ylabel('Validation loss', fontsize=8.5, color=INK)
    ax.set_xlabel('Epoch', fontsize=8.5, color=INK)
    ax.set_title('(b)  Validation loss', fontsize=9, color=INK, loc='left', pad=6)
    ax.set_xlim(-0.8, ep[-1] + 0.8)

    fig.text(0.005, -0.005,
             'Trained on the identical dataset, split and validation set as OxoNet '
             '({:,} rows, batch 2048).  Final 10 epochs: ΔAUPRC = {:+.4f}.'.format(48344552, d10),
             fontsize=7, color=INK2, ha='left', va='top')

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        f = os.path.join(args.out_dir, 'nanocon_training_curve.' + ext)
        fig.savefig(f, dpi=300, bbox_inches='tight', facecolor='white')
        print('写出', f)
    print('best epoch {}  AUPRC {:.5f}   最后{}轮极差 {:.5f}   最后10轮变化 {:+.5f}'.format(
        best, ap.max(), args.plateau, d5, d10))


if __name__ == '__main__':
    main()
