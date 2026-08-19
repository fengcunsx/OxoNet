"""来源捷径诊断：模型学到的是"8-oxo-dG"还是"这条 read 来自 oligo 还是人基因组"？

审稿压力测试提出的 P0 风险：训练时阳性全部来自合成 oligo、阴性大部分来自人基因组，
标签与数据来源高度共线。若模型部分地在做 source classification，
所有低 FPR 结论都会被质疑。

本脚本只用**未修饰位点**（label=0），在 valid 上比较 oligo 阴性 vs 基因组阴性：

  ① 表征层面能不能分开？—— 用 GBDT 在 mean/std/dwell 上训 source classifier，
     给出"来源信息的上限"。这是数据本身的性质，与我们的模型无关。
  ② **OxoNet 的分数**能不能分开来源？—— 若 ① 高而 ② 接近 0.5，
     说明来源信息虽然存在，但模型没有拿它做决策。这是关键证据。
  ③ 三类样本的分数分布（合成阴性 / 基因组阴性 / 合成阳性）。
  ④ 判别力对称性：AUROC(合成阳性 vs 合成阴性) 与 AUROC(合成阳性 vs 基因组阴性)。
     若模型走捷径，后者应显著高于前者（因为来源本身就能把两类分开）。

    /home/bio/anaconda3/bin/python source_shortcut.py
"""
import argparse
import os

import numpy as np

PACK = '/home/bio/8oxog/build/pack'
GMASK = os.path.join(PACK, 'valid_genomic_neg_mask.npy')
MODELS = {
    'OxoNet': '/home/bio/8oxog/wtl1/pack_scores_e2ep125',
    'esox': '/home/bio/8oxog/wtl1/pack_scores_esox',
    'NanoCon': '/home/bio/8oxog/nanocon/scores',
}


def probs_path(d, name):
    for c in (os.path.join(d, name + '.probs.npz'), os.path.join(d, 'gbdt_' + name + '.probs.npz')):
        if os.path.isfile(c):
            return c
    raise SystemExit('缺 ' + name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200000, help='每类抽样数(GBDT 用)')
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/source_shortcut.txt')
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier

    v = np.load(os.path.join(PACK, 'valid.npz'), allow_pickle=False)
    lab = v['label'].astype(int)
    gm = np.load(GMASK)
    neg = lab == 0
    ol = neg & (~gm)                     # 合成阴性
    ge = gm                              # 基因组阴性(掩码本身已含 neg 条件)
    L = ['来源捷径诊断（只用未修饰位点，valid 集）',
         '  合成阴性 {:,} 个 / 基因组阴性 {:,} 个 / 合成阳性 {:,} 个'.format(
             int(ol.sum()), int(ge.sum()), int((lab == 1).sum())), '']

    # ---------- ① 表征层面的来源可分性上限 ----------
    rng = np.random.default_rng(0)
    io = rng.choice(np.where(ol)[0], min(args.n, int(ol.sum())), replace=False)
    ig = rng.choice(np.where(ge)[0], min(args.n, int(ge.sum())), replace=False)
    idx = np.concatenate([io, ig])
    y = np.concatenate([np.zeros(len(io)), np.ones(len(ig))])
    X = np.hstack([v['mean'][idx], v['std'][idx], v['dwell'][idx]]).astype(np.float32)
    cut = rng.permutation(len(idx))
    tr, te = cut[:len(cut) // 2], cut[len(cut) // 2:]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr])
    auc_feat = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
    L.append('=== ① 来源信息的上限（GBDT 在 mean/std/dwell 上分辨 oligo vs 基因组）===')
    L.append('  source AUROC = {:.4f}   ({} 训 / {} 测)'.format(auc_feat, len(tr), len(te)))
    L.append('  → 这是**数据本身**的可分性，与我们的模型无关；越高说明捷径越唾手可得')

    # ---------- ② 各模型分数是否携带来源信息 ----------
    L.append('')
    L.append('=== ② 各模型的**分数**能否分辨来源（同一批未修饰位点）===')
    L.append('  {:10s} {:>14s} {:>28s}'.format('model', 'source AUROC', '合成阴性/基因组阴性 中位分数'))
    for m, d in MODELS.items():
        p = np.nan_to_num(np.load(probs_path(d, 'valid'))['prob'].astype(np.float64), nan=-1.0)
        a = roc_auc_score(np.concatenate([np.zeros(int(ol.sum())), np.ones(int(ge.sum()))]),
                          np.concatenate([p[ol], p[ge]]))
        L.append('  {:10s} {:14.4f} {:>28s}'.format(
            m, a, '{:.4f} / {:.4f}'.format(np.median(p[ol]), np.median(p[ge]))))
    L.append('  → 若 ① 高而此处接近 0.5，说明来源可分但模型**没有拿它做决策**')

    # ---------- ③ 判别力对称性 ----------
    L.append('')
    L.append('=== ③ 判别力对称性（走捷径的话，用基因组阴性当负例会"虚高"）===')
    L.append('  {:10s} {:>22s} {:>22s} {:>10s}'.format(
        'model', 'AUROC 阳性vs合成阴性', 'AUROC 阳性vs基因组阴性', '差'))
    for m, d in MODELS.items():
        p = np.nan_to_num(np.load(probs_path(d, 'valid'))['prob'].astype(np.float64), nan=-1.0)
        pos = p[lab == 1]
        a1 = roc_auc_score(np.concatenate([np.ones(len(pos)), np.zeros(int(ol.sum()))]),
                           np.concatenate([pos, p[ol]]))
        a2 = roc_auc_score(np.concatenate([np.ones(len(pos)), np.zeros(int(ge.sum()))]),
                           np.concatenate([pos, p[ge]]))
        L.append('  {:10s} {:22.4f} {:22.4f} {:+10.4f}'.format(m, a1, a2, a2 - a1))
    L.append('  → 差值接近 0 = 换成基因组阴性不会让模型"变好"，即没有来源加成')

    # ---------- ④ 分数分布 ----------
    L.append('')
    L.append('=== ④ OxoNet 分数分布（分位数）===')
    p = np.nan_to_num(np.load(probs_path(MODELS['OxoNet'], 'valid'))['prob'].astype(np.float64),
                      nan=-1.0)
    qs = [50, 90, 99, 99.9, 99.99]
    L.append('  {:14s} {}'.format('', ' '.join('{:>9s}'.format('p' + str(q)) for q in qs)))
    for tag, sel in (('合成阴性', ol), ('基因组阴性', ge), ('合成阳性', lab == 1)):
        L.append('  {:14s} {}'.format(
            tag, ' '.join('{:9.4f}'.format(np.percentile(p[sel], q)) for q in qs)))

    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print('\n写出', args.out)


if __name__ == '__main__':
    main()
