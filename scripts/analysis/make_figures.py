"""生成论文配图（IEEE Access, access_r1.tex）。

配色与可读性规范：
  * 四方固定配色 = 蓝/橙/青/紫 (#2a78d6,#eb6834,#1baf7a,#4a3aa7)，
    该组合经 validate_palette.js 全项通过（all-pairs 色盲分离 ΔE 9.2 / 常色觉 16.3）。
  * **颜色跟实体走，不跟排名走** —— 每个方法在所有图里永远同一个颜色。
  * 黑白打印：颜色之外再叠**线型 + marker + 填充纹理**做二次编码，去色后仍可分辨。
  * **不用双轴**。两个量纲不同就拆成两个子图。
  * 网格/坐标轴退到背景，标签用文字色而非序列色。

    /home/bio/anaconda3/bin/python make_figures.py [--only roc,training,...]
"""
import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PACK = '/home/bio/8oxog/build/pack'
WT = '/home/bio/8oxog/wtl1'
OUT = '/home/bio/8oxog/paper/IEEE_Access/images/r1'
GMASK = os.path.join(PACK, 'valid_genomic_neg_mask.npy')

# 方法 -> (打分目录, 颜色, 线型, marker, 纹理)
METHODS = [
    ('OxoNet',      '/home/bio/8oxog/wtl1/pack_scores_e2ep125', '#2a78d6', '-',   'o', ''),
    ('esox',        '/home/bio/8oxog/wtl1/pack_scores_esox',    '#eb6834', '--',  's', '///'),
    ('NanoCon',     '/home/bio/8oxog/nanocon/scores',          '#1baf7a', '-.',  '^', '...'),
    ('GBDT control', '/home/bio/8oxog/nanocon',                '#4a3aa7', (0, (1, 1)), 'D', 'xxx'),
]
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d8d7d2'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.labelcolor': INK, 'text.color': INK,
    'grid.color': GRID, 'grid.linewidth': 0.5,
    'legend.frameon': False, 'figure.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.dpi': 300, 'savefig.facecolor': 'none',
    # IEEE 排版流程通常拒收 Type 3 字体; 42 = TrueType 嵌入
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
W1, W2 = 3.45, 7.16          # IEEE 单栏 / 双栏宽度(inch)


def probs_path(d, name):
    for c in (os.path.join(d, name + '.probs.npz'), os.path.join(d, 'gbdt_' + name + '.probs.npz')):
        if os.path.isfile(c):
            return c
    raise SystemExit('缺 {} 的 {}'.format(d, name))


def load(d, name):
    a = np.load(probs_path(d, name))
    return np.nan_to_num(a['prob'].astype(np.float64), nan=-1.0), a['label'].astype(int)


def style(ax, grid_axis='both'):
    ax.grid(True, axis=grid_axis, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, name + '.' + ext))
    plt.close(fig)
    print('  ->', os.path.join(OUT, name + '.pdf'))


# --------------------------------------------------------------------------
def fig_roc_pr():
    """ROC(对数横轴, 稀有事件真正关心的区域) + PR, 基因组阴性背景。"""
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
    for name, d, c, ls, mk, _ in METHODS:
        p, l = load(d, 'test_t2t')
        fpr, tpr, _ = roc_curve(l, p)
        axes[0].plot(np.maximum(fpr, 1e-7), tpr, color=c, ls=ls, lw=1.4, label=name)
        pr, rc, _ = precision_recall_curve(l, p)
        axes[1].plot(rc, pr, color=c, ls=ls, lw=1.4, label=name)
    a = axes[0]
    a.set_xscale('log'); a.set_xlim(1e-5, 1); a.set_ylim(0, 1)
    a.set_xlabel('False-positive rate on genomic negatives (log scale)')
    a.set_ylabel('Recall')
    for g in (1e-4,):        # 论文主工作点
        a.axvline(g, color=INK2, lw=0.6, ls=':', zorder=0)
        # 上方是图例、中段是曲线; 贴 x 轴的窄带是唯一空白, 水平排版
        a.text(g * 1.3, 0.015, 'main operating point', color=INK2, fontsize=6.5, va='bottom')
    a.set_title('(a) ROC, genomic negative background', loc='left')
    style(a)
    b = axes[1]
    b.set_xlim(0, 1); b.set_ylim(0, 1.02)
    b.set_xlabel('Recall'); b.set_ylabel('Precision')
    b.set_title('(b) Precision-recall', loc='left')
    style(b)
    a.legend(loc='upper left', handlelength=2.6)
    save(fig, 'fig_roc_pr')


# --------------------------------------------------------------------------
def fig_training():
    """收敛曲线。

    两处刻意的设计, 改之前先读:

    1. **横轴是样本呈现数, 不是 epoch。** OxoNet 一个 epoch 是 5.96M(1:1 平衡采样,
       2,978,201 阳性 x2), NanoCon 一个 epoch 是全部 48,344,552 行 —— 相差 8.1 倍。
       画在 "Epoch" 轴上会让人以为 OxoNet 训了 5 倍长, 而事实是 NanoCon 总共多看了
       1.62 倍样本(1.45e9 vs 8.94e8)。那正好和 IV-G 的公平性论证相反, 等于递刀给审稿人。
    2. **(b) 与 (d) 故意画不同的指标, 且绝不共用纵轴。**
       原先 (b) 画 recall@99.9%spec、(d) 画 recall@0.5 却都只标 "recall",
       会被横向误比(0.66 vs 0.56), 而正文的真实差距是 55.9 vs 28.1。
       但**换成两边都画 AUROC 同样是错的**: NanoCon 的 avg_val_AUROC 来自
       lighting.py:314 `torch.stack([x['val_AUROC'] for x in outputs]).mean()`,
       是**逐 batch 算 AUROC 再平均**, 而且 validation_step 吃的是成对数据
       (seq1/seq2), batch 内类别配对平衡; OxoNet 的 auroc 则是全部 2,936,701 个
       不平衡验证位点上的**全局** AUROC。两者不是同一个量。
       NanoCon 是 save_top_k=1, 没有逐 epoch 权重也没有逐 epoch 概率, 无法补算。
       所以 (b)=OxoNet 全局 AUROC, (d)=NanoCon 自己记的 batch 平均 AUPRC,
       连指标名都不同, 纵轴范围也不共用。操作点比较一律交给 Table 2/7。

    另: OxoNet 的 `test_info.tsv` 里的 "test" 其实是 **valid**
    (train.py:248 `test_dataset = valid_dataset`), 所以 (a) 标 validation 是对的。
    """
    import csv
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    OX_PER_EP = 5_956_402          # 2,978,201 阳性 x 2 (1:1 平衡采样)
    NC_PER_EP = 48_344_552         # 全部训练行
    OXC, NCC = '#2a78d6', '#1baf7a'

    def band(ax, x, ys, color, label, ls='-'):
        """mean 线 + min-max 带。ys = 若干条等长曲线。"""
        A = np.vstack(ys)
        m = A.mean(0)
        if len(ys) > 1:
            ax.fill_between(x, A.min(0), A.max(0), color=color, alpha=0.18, lw=0)
        ax.plot(x, m, color=color, ls=ls, lw=1.4, label=label)
        return m

    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.4))

    # ---------- OxoNet: 三个 seed 的 loss(全程 150 轮都有) ----------
    ox_dirs = ['/home/bio/8oxog/726train1',
               '/data/seeds_and_ablation/train_curves/seed0',
               '/data/seeds_and_ablation/train_curves/seed3407']
    trs = [np.loadtxt(os.path.join(d, 'train_info.tsv')) for d in ox_dirs]
    tes = [np.loadtxt(os.path.join(d, 'test_info.tsv')) for d in ox_dirs]
    n = min(len(t) for t in trs + tes)
    x_ox = (np.arange(n) + 1) * OX_PER_EP / 1e8

    a = axes[0][0]
    band(a, x_ox, [t[:n, 6] for t in trs], INK2, 'training', ls='--')
    band(a, x_ox, [t[:n, 6] for t in tes], OXC, 'validation')
    a.set_xlabel('Sample presentations ($\\times 10^{8}$)')
    a.set_ylabel('Focal loss')
    a.set_title('(a) OxoNet loss', loc='left')
    a.axvline(46 * OX_PER_EP / 1e8, color=INK2, lw=0.6, ls=':', zorder=0)
    a.text(47 * OX_PER_EP / 1e8, a.get_ylim()[1] * 0.90, 'phase 2', color=INK2, fontsize=6.5)
    a.legend(); style(a)

    # ---------- OxoNet AUROC: seed42 全程; 另两个 seed 只存了 ep100-149 ----------
    b = axes[0][1]
    i42 = np.genfromtxt('/home/bio/8oxog/726train1/info999.tsv', names=True, delimiter='\t')
    e42, a42 = i42['epoch'].astype(int), i42['auroc']
    others = []
    for f in ('/data/seeds_and_ablation/_csv/info999Seed0.csv',
              '/data/seeds_and_ablation/_csv/info999Seed3407.csv'):
        r = list(csv.DictReader(open(f)))
        others.append((np.array([int(float(x['epoch'])) for x in r]),
                       np.array([float(x['auroc']) for x in r])))
    lo, hi = max(o[0].min() for o in others), min(o[0].max() for o in others)
    sel = (e42 >= lo) & (e42 <= hi)
    stack = np.vstack([a42[sel]] + [o[1][(o[0] >= lo) & (o[0] <= hi)] for o in others])
    b.fill_between((np.arange(lo, hi + 1) + 1) * OX_PER_EP / 1e8, stack.min(0), stack.max(0),
                   color=OXC, alpha=0.18, lw=0)
    b.plot((e42 + 1) * OX_PER_EP / 1e8, a42, color=OXC, lw=1.4, label='seed 42 (full run)')
    b.axvline(126 * OX_PER_EP / 1e8, color=INK2, lw=0.8, zorder=0)
    b.text(125 * OX_PER_EP / 1e8, 0.905, 'selected epoch', color=INK2, fontsize=6.5,
           rotation=90, ha='right', va='bottom')
    b.set_ylim(0.88, 1.0)
    b.set_xlabel('Sample presentations ($\\times 10^{8}$)')
    b.set_ylabel('Validation AUROC')
    b.set_title('(b) OxoNet validation AUROC', loc='left')
    b.text(0.03, 0.94, 'global, full validation set', transform=b.transAxes,
           fontsize=6.5, color=INK2, va='top')
    from matplotlib.patches import Patch
    hb, lb = b.get_legend_handles_labels()
    hb.append(Patch(facecolor=OXC, alpha=0.18, edgecolor='none'))
    lb.append('range over 3 seeds\n(final 50 epochs only)')
    b.legend(hb, lb, loc='lower right', fontsize=6.8, handlelength=1.8)
    style(b)

    # ---------- NanoCon: 三个 seed 的 tfevents ----------
    runs = ['/data/nanocon_seeds/logs/version_0',
            '/data/nanocon_seeds/logs/version_0_seed0',
            '/data/nanocon_seeds/logs/version_0_seed_3407']
    def scal(d, tag):
        ea = EventAccumulator(d); ea.Reload()
        return np.array([e.value for e in ea.Scalars(tag)])
    losses = [scal(d, 'avg_val_loss') for d in runs]
    auprcs = [scal(d, 'avg_val_AUPRC') for d in runs]   # 注意: batch 平均, 非全局
    m = min(len(v) for v in losses + auprcs)
    x_nc = (np.arange(m) + 1) * NC_PER_EP / 1e8

    c = axes[1][0]
    band(c, x_nc, [v[:m] for v in losses], NCC, 'validation')
    c.set_xlabel('Sample presentations ($\\times 10^{8}$)')
    c.set_ylabel('NanoCon composite loss')
    c.set_title('(c) NanoCon loss', loc='left')
    c.legend(); style(c)

    d_ = axes[1][1]
    band(d_, x_nc, [v[:m] for v in auprcs], NCC, 'mean of 3 seeds')
    d_.set_xlabel('Sample presentations ($\\times 10^{8}$)')
    d_.set_ylabel('Validation AUPRC')
    d_.set_title("(d) NanoCon validation AUPRC", loc='left')
    _lo, _hi = d_.get_ylim()                    # 给顶部注记留出空间, 别压到曲线
    d_.set_ylim(_lo, _hi + 0.22 * (_hi - _lo))
    d_.text(0.03, 0.96, "batch-averaged, as logged by NanoCon\n(not comparable with (b))",
            transform=d_.transAxes, fontsize=6.5, color=INK2, va='top')
    hd, ld = d_.get_legend_handles_labels()
    hd.append(Patch(facecolor=NCC, alpha=0.18, edgecolor='none'))
    ld.append('range over 3 seeds')
    d_.legend(hd, ld, loc='lower right', fontsize=6.8, handlelength=1.8)
    style(d_)
    # 四个面板共用横轴范围: NanoCon 的曲线明显更长, "谁看的数据多" 一眼可见
    xmax = max(x_ox[-1], x_nc[-1]) * 1.02
    for ax in (a, b, c, d_):
        ax.set_xlim(0, xmax)
    # 两条训练量的终点各标一下
    for ax, xe, txt, col in ((a, x_ox[-1], '8.94', OXC), (c, x_nc[-1], '14.50', NCC)):
        ax.axvline(xe, color=col, lw=0.7, ls=':', zorder=0)
        ax.text(xe, ax.get_ylim()[1], txt, color=col, fontsize=6.5,
                ha='right', va='top')

    fig.tight_layout()
    save(fig, 'fig_training')


# --------------------------------------------------------------------------
def _grun(kmers):
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


def fig_homopolymer():
    """等基因组 FPR=1e-4 下, 按中心 G 所在同聚物长度分层的 recall。"""
    gm = np.load(GMASK)
    src = np.load(os.path.join(PACK, 'test_oligo.npz'), allow_pickle=False)
    lab = src['label'].astype(int)
    g = np.minimum(_grun(src['kmers'].astype('U7').tolist()), 4)[lab == 1]
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    w = 0.2
    for i, (name, d, c, _, _, hatch) in enumerate(METHODS):
        vp, _ = load(d, 'valid')
        t = float(np.quantile(vp[gm], 1 - 1e-4))
        p, l = load(d, 'test_oligo')
        pos = p[l == 1]
        vals = [(pos[g == k] >= t).mean() * 100 for k in (1, 2, 3, 4)]
        ax.bar(np.arange(4) + (i - 1.5) * w, vals, w * 0.92, color=c, hatch=hatch,
               edgecolor='white', linewidth=0.6, label=name)
    ax.set_xticks(range(4)); ax.set_xticklabels(['1', '2', '3', r'$\geq$4'])
    ax.set_xlabel('Guanine run length')
    ax.set_ylabel('Recall (%)')
    ax.set_ylim(0, 74)                      # 留出图例空间, 不压柱子
    ax.set_title(r'Recall by homopolymer length at $\mathrm{FPR}=10^{-4}$', loc='left')
    ax.legend(ncol=2, loc='upper center', columnspacing=1.0, handlelength=1.6)
    style(ax, 'y')
    fig.tight_layout()
    save(fig, 'fig_homopolymer')


# --------------------------------------------------------------------------
def fig_native_ppv():
    """(a) native 等灵敏度下的每百万判正数(对数); (b) PPV 随真实丰度的变化。"""
    gm = np.load(GMASK)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))

    # (a) 数据来自 four_way_report.txt 的 native 段(等灵敏度 38.42%)
    # 阈值在 valid 阳性上定(目标 recall 41.56% = esox@T* 在 valid 上); 见 four_way_report.txt
    native = [('OxoNet', 27.9), ('esox', 2286.1), ('NanoCon', 1125.2)]
    cols = {n: c for n, _, c, _, _, _ in METHODS}
    hats = {n: h for n, _, _, _, _, h in METHODS}
    a = axes[0]
    for i, (n, v) in enumerate(native):
        a.bar(i, v, 0.6, color=cols[n], hatch=hats[n], edgecolor='white', linewidth=0.6)
        a.text(i, v * 1.15, '{:,.1f}'.format(v), ha='center', fontsize=7, color=INK)
    a.axhline(75, color=INK2, lw=0.8, ls='--')
    a.set_yscale('log'); a.set_ylim(8, 9000)
    # 虚线上唯一的空白在左端(OxoNet 柱只到 27.9); 右边两根柱子都穿过 y=75
    a.text(-0.45, 82, '75/M scenario', color=INK2, fontsize=6.5,
           ha='left', va='bottom')
    a.set_xticks(range(3)); a.set_xticklabels([n for n, _ in native])
    a.set_ylabel('Calls per million\ninterrogated guanines')  # 分母 = 100 种 context 内的 G
    a.set_title('(a) Native DNA at matched sensitivity', loc='left')
    style(a, 'y')

    # (b) PPV(π) = π·r / (π·r + (1-π)·f), 各方法在 FPR=1e-4 的 (r,f)
    b = axes[1]
    pi = np.logspace(-6.3, -3.3, 200)
    # **用名义 f = 1e-4, 不用各方法在 test 上的实测 f。**
    # 阈值定在 valid 基因组阴性上再搬到 test, 实测 FPR 会漂: 实测/名义 = 0.95(NanoCon)
    # ~1.27(esox)。若各用各的实测 f, 这张图就不再是"等 FPR"比较(全文口径), 而且
    # esox 会因阈值迁移多跑 27% 的 FPR 白丢 3.2pp —— 方向恰好对 OxoNet 有利, 不能这么画。
    # 名义口径与标题、Table 2 完全一致, 且是保守方向。实测值在图注里交代。
    F_NOM = 1e-4
    for name, d, c, ls, _, _ in METHODS:
        vp, _ = load(d, 'valid')
        t = float(np.quantile(vp[gm], 1 - F_NOM))
        po, lo = load(d, 'test_oligo')
        pt, lt = load(d, 'test_t2t')
        r = float((po[lo == 1] >= t).mean())
        f_real = float((pt[lt == 0] >= t).mean())
        print('  [ppv] {:14} recall={:.4f}  realized FPR={:.3e} ({:.2f}x nominal)'.format(
            name, r, f_real, f_real / F_NOM))
        b.plot(pi * 1e6, 100 * pi * r / (pi * r + (1 - pi) * F_NOM),
               color=c, ls=ls, lw=1.4, label=name)
    b.axvline(75, color=INK2, lw=0.8, ls='--')
    b.text(80, 4, '75/M\nscenario', color=INK2, fontsize=7)
    b.set_xscale('log')
    b.set_xlabel('Assumed lesion prevalence\n(per million interrogated guanines)')
    b.set_ylabel('Positive predictive value (%)')
    b.set_title(r'(b) PPV vs. abundance at $\mathrm{FPR}=10^{-4}$', loc='left')
    b.legend(loc='upper left'); style(b)
    fig.tight_layout()
    save(fig, 'fig_native_ppv')


# --------------------------------------------------------------------------
def _parse_blocks(path):
    """把 genome_dist.txt / gene_region.txt 按 '=== 模型 ===' 切块。"""
    txt = open(path, encoding='utf-8').read()
    out, cur = {}, None
    for line in txt.splitlines():
        m = re.match(r'^=== (.+?) ===$', line.strip())
        if m:
            cur = m.group(1).strip(); out[cur] = []
        elif cur:
            out[cur].append(line)
    return out


def fig_genome():
    """(a) GC 分箱的 5-mer 标准化比(剔除 CpG 前后); (b) 基因区标准化比。"""
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
    name_map = {'OxoNet': 'OxoNet', 'esox': 'esox', 'NanoCon': 'NanoCon'}

    def gc_ratios(path):
        got = {}
        for mod, lines in _parse_blocks(path).items():
            vals, inblk = [], False
            for ln in lines:
                if 'GC 含量' in ln:
                    inblk = True; continue
                if inblk and ln.strip().startswith('--'):
                    break
                m = re.search(r'\(([\d.]+), ([\d.]+)\]\s+[\d,]+\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)x', ln)
                if m:
                    vals.append(float(m.group(3)))
            if vals:
                got[mod] = vals
        return got

    with_cpg = gc_ratios(os.path.join(WT, 'genome_dist.txt'))
    no_cpg = gc_ratios(os.path.join(WT, 'genome_dist_nocpg.txt'))
    xs = np.arange(7)
    labels = ['<35', '35-40', '40-45', '45-50', '50-55', '55-60', '>60']
    a = axes[0]
    for name, _, c, ls, mk, _ in METHODS:
        if name not in no_cpg:
            continue
        a.plot(xs, no_cpg[name], color=c, ls=ls, lw=1.4, marker=mk, ms=3.5, label=name + ' (non-CpG)')
        a.plot(xs, with_cpg[name], color=c, ls=ls, lw=0.9, alpha=0.4, marker=mk, ms=2.5)
    a.axhline(1.0, color=INK2, lw=0.6, ls=':', zorder=0)
    a.set_xticks(xs); a.set_xticklabels(labels, rotation=30)
    a.set_xlabel('GC content of the 1 kb bin (%)')
    a.set_ylabel('Standardized ratio (obs./exp.)')
    a.set_title('(a) GC dependence', loc='left')
    # 6 条线只有 3 个图例条目会看不懂: 给淡色那组一个专门的图例条目
    from matplotlib.lines import Line2D
    h, l = a.get_legend_handles_labels()
    h.append(Line2D([], [], color=INK2, lw=0.9, alpha=0.4, marker='o', ms=2.5))
    l.append('same three, CpG sites included')
    a.legend(h, l, loc='upper left', fontsize=7.5, handlelength=2.4)
    style(a)

    regions, vals = ['5\'UTR', '3\'UTR', 'exon', 'intron', 'intergenic'], {}
    for mod, lines in _parse_blocks(os.path.join(WT, 'gene_region.txt')).items():
        d = {}
        for ln in lines:
            m = re.match(r"\s*(\S+)\s+[\d,]+\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)x", ln)
            if m:
                d[m.group(1)] = float(m.group(2))
        if d:
            vals[mod] = d
    b = axes[1]
    w = 0.26
    for i, (name, _, c, _, _, hatch) in enumerate(METHODS):
        if name not in vals:
            continue
        v = np.array([vals[name].get(r, np.nan) for r in
                      ["5'UTR", "3'UTR", 'exon', 'intron', 'intergenic']])
        # 柱子锚定在基线 1.0 上(方向编码正负), 而不是从 0 截断
        b.bar(np.arange(5) + (i - 1) * w, v - 1.0, w * 0.9, bottom=1.0, color=c, hatch=hatch,
              edgecolor='white', linewidth=0.6, label=name)
    b.axhline(1.0, color=INK2, lw=0.8, zorder=2)
    b.set_xticks(range(5)); b.set_xticklabels(regions, rotation=30)
    b.set_ylabel('Standardized ratio (baseline 1.0)'); b.set_ylim(0.85, 1.36)
    b.set_title('(b) Gene regions', loc='left')
    b.legend(ncol=1, loc='upper right'); style(b, 'y')
    fig.tight_layout()
    save(fig, 'fig_genome')


# --------------------------------------------------------------------------
def fig_dala():
    """(a) 逐染色体 L vs D(OxoNet); (b) 三方效应量随工作点收紧的变化。"""
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))

    # (a) 六个样本级点(不是挑一对 run 的 23 条染色体) —— 避免 pseudo-replication 观感
    a = axes[0]
    L = {'wtl1': 245.7, 'wtl2': 190.3, 'p53l1': 286.6}
    D = {'wtd1': 303.9, 'wtd2': 366.3, 'p53d1': 317.9}
    for k, (lab, vals, x) in enumerate([('L-Ala', L, 0), ('D-Ala', D, 1)]):
        for name, v in vals.items():
            wt = not name.startswith('p53')
            a.scatter(x + (np.random.default_rng(abs(hash(name)) % 999).uniform(-.07, .07)), v,
                      s=46, marker='o' if wt else '^',
                      color='#2a78d6' if wt else '#eb6834',
                      edgecolor='white', linewidth=0.6, zorder=3)
    # 不连配对线: 公开元数据只给出 L-1/L-2/D-1/D-2 这样的编号, 没有证据表明它们是
    # 同一培养物/批次的一一配对。基因型是设计上平衡的, 配对不是。
    a.plot([0, 1], [np.mean(list(L.values())), np.mean(list(D.values()))],
           color=INK, lw=1.8, marker='_', ms=16, zorder=4)
    a.set_xlim(-0.4, 1.4); a.set_xticks([0, 1]); a.set_xticklabels(['L-Ala', 'D-Ala'])
    a.set_ylabel('OxoNet calls per million G')
    a.set_title('(a) Six samples, three per condition', loc='left')
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], ls='none', marker='o', color='#2a78d6', ms=6, label='wild type'),
                      Line2D([], [], ls='none', marker='^', color='#eb6834', ms=6, label='$p53^{-/-}$'),
                      Line2D([], [], color=INK, lw=1.8, label='group mean')],
             loc='upper left', fontsize=7)
    style(a, 'y')

    b = axes[1]
    # 3L vs 3D 的比值(六个样本), 不是 wtl1-vs-wtd1 单对 —— 必须与正文一致
    sweep = open('/data/dala_sweep_3v3.txt', encoding='utf-8').read()
    ops = [('1e-03', r'$10^{-3}$'), ('1e-04', r'$10^{-4}$'), ('1e-05', r'$10^{-5}$')]
    got = {n: [] for n, *_ in METHODS[:3]}
    for key, _ in ops:
        sec = re.search(r'=== ① 等基因组 FPR = ' + key + r' ===\n(.*?)(?=\n=== |\Z)', sweep, re.S)
        for ln in sec.group(1).splitlines():
            m = re.search(r'^\s*(\S+)\s.*?([\d.]+)x\s', ln)
            if m and m.group(1) in got:
                got[m.group(1)].append(float(m.group(2)))
    x = np.arange(3)
    for name, _, c, ls, mk, _ in METHODS[:3]:
        if len(got.get(name, [])) == 3:
            b.plot(x, got[name], color=c, ls=ls, lw=1.4, marker=mk, ms=4, label=name)
    b.axhline(1.0, color=INK2, lw=0.6, ls=':', zorder=0)
    b.set_xticks(x); b.set_xticklabels([lbl for _, lbl in ops])
    b.set_xlabel('Operating point (genomic FPR)')
    b.set_ylabel('D-Ala / L-Ala call-rate ratio')
    b.set_title('(b) Effect size vs. stringency', loc='left')
    b.legend(loc='upper left'); style(b)
    fig.tight_layout()
    save(fig, 'fig_dala')


# --------------------------------------------------------------------------
def fig_per5mer():
    """逐 5-mer **配对**比较: 同一 context 上 OxoNet - 基线。

    早先版本让各方法各自升序排列, 同一横坐标并不对应同一 context, 因此只能比较
    分布、不能主张"逐 context 更优"。这里改成配对差值。"""
    gm = np.load(GMASK)
    src = np.load(os.path.join(PACK, 'test_oligo.npz'), allow_pickle=False)
    lab = src['label'].astype(int)
    ctx = np.array([s[1:6] for s in src['kmers'].astype('U7')])[lab == 1]
    uc = np.unique(ctx)
    rec = {}
    for name, d, c, ls, mk, _ in METHODS:
        vp, _ = load(d, 'valid')
        t = float(np.quantile(vp[gm], 1 - 1e-4))
        p, l = load(d, 'test_oligo')
        pos = p[l == 1]
        rec[name] = np.array([(pos[ctx == u] >= t).mean() for u in uc]) * 100

    # 双栏宽: 单栏塞 300 个点会让标记互相压死, 图例文字也没处放
    fig, ax = plt.subplots(figsize=(W2, 2.8))
    base = rec['OxoNet']
    order = np.argsort(base)                       # 固定共同排序: 按 OxoNet 难度
    rng = np.random.default_rng(0)
    for name, d, c, ls, mk, _ in METHODS[1:]:
        diff = (base - rec[name])[order]
        win = int((diff > 0).sum())
        bs = [np.median(rng.choice(diff, len(diff))) for _ in range(2000)]
        # 图例只放方法名; 胜场/中位数/CI 交给图注, 否则文字块必然压在散点上
        ax.plot(np.arange(len(diff)), diff, color=c, ls='none', marker=mk, ms=3.4,
                alpha=0.8, label=name)
        print('  [per5mer] {:14} wins {}/{}, median +{:.1f} pp [{:.1f}, {:.1f}]'.format(
            name, win, len(diff), np.median(diff),
            np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel('5-mer context (common ordering: by OxoNet recall)')
    ax.set_ylabel('OxoNet $-$ baseline recall (pp)')
    ax.set_title(r'Paired per-5-mer difference at $\mathrm{FPR}=10^{-4}$', loc='left')
    leg = ax.legend(loc='upper left', fontsize=7.5, ncol=3, frameon=True,
                    framealpha=0.92, edgecolor='none', handletextpad=0.3,
                    columnspacing=1.2)
    leg.get_frame().set_facecolor('white')
    _lo, _hi = ax.get_ylim()                     # 顶部留白给图例, 别压散点
    ax.set_ylim(min(-2, _lo), _hi + 0.14 * (_hi - _lo))
    style(ax)
    fig.tight_layout()
    save(fig, 'fig_per5mer')


ALL = {'roc': fig_roc_pr, 'training': fig_training, 'homopolymer': fig_homopolymer,
       'native': fig_native_ppv, 'genome': fig_genome, 'dala': fig_dala,
       'per5mer': fig_per5mer}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    todo = a.only.split(',') if a.only else list(ALL)
    for k in todo:
        print('==', k)
        ALL[k]()
