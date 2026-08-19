"""NanoCon vs OxoNet(E2 ep125) 在**完全相同位点**上的对比报告。

两边打的都是 `8oxog/build/pack` 的同一批 npz(同 read/同 basecall_pos/同顺序)，
所以 probs 数组逐行对齐，不需要 join。native 侧两边各自输出 <base>.tsv.gz，
按 (read_id, basecall_pos) join 做一致性分析。

阈值口径与 OxoNet 一样：T* = valid 阴性上达到目标特异性(默认 99.9%)的分数。
每个模型用**自己**的 T*(分数尺度不可比，工作点才可比)。

    /home/bio/anaconda3/bin/python compare_report.py \
        --nanocon /home/bio/8oxog/nanocon/scores \
        --oxonet  /home/bio/8oxog/wt1/pack_scores_e2ep125
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

SETS = ['valid', 'test_oligo', 'test_t2t']


def load(d, name):
    a = np.load(os.path.join(d, name + '.probs.npz'))
    return a['prob'].astype(np.float64), a['label'].astype(np.int8)


def auroc_auprc(prob, label):
    from sklearn.metrics import roc_auc_score, average_precision_score
    m = label >= 0
    return roc_auc_score(label[m], prob[m]), average_precision_score(label[m], prob[m])


def tstar(prob, label, spec):
    """valid 阴性分布的 spec 分位数 = 达到该特异性的最小阈值。"""
    return float(np.quantile(prob[label == 0], spec))


def rates(prob, label, thr):
    pos, neg = label == 1, label == 0
    rec = float((prob[pos] >= thr).mean()) if pos.any() else float('nan')
    fpr = float((prob[neg] >= thr).mean()) if neg.any() else float('nan')
    return rec, fpr


def recall_at_fpr(prob, label, target_fpr):
    """在负样本上定 FPR=target 的阈值，报此时的 recall(同一测试集内定阈值)。"""
    neg = prob[label == 0]
    thr = float(np.quantile(neg, 1.0 - target_fpr))
    return float((prob[label == 1] >= thr).mean()), thr


def native_rate(d, thr, col='prob', only=None):
    """native(全基因组 G，绝大多数应为阴)在阈值上的判正率。"""
    tot = called = 0
    per = {}
    for p in sorted(glob.glob(os.path.join(d, '*.tsv.gz'))):
        if only and os.path.basename(p) not in only:
            continue
        v = pd.read_csv(p, sep='\t', usecols=[col])[col].values
        c = int((v >= thr).sum())
        per[os.path.basename(p).replace('.tsv.gz', '')] = (len(v), c)
        tot += len(v)
        called += c
    return tot, called, per


def native_concordance(d1, d2, thr1, thr2, limit=0):
    """两个模型在同一 native read/位点上的判正一致性 + 富集倍数。"""
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d1, '*.tsv.gz')))
    files = [f for f in files if os.path.isfile(os.path.join(d2, f))]
    if limit:
        files = files[:limit]
    n = both = c1 = c2 = 0
    for f in files:
        a = pd.read_csv(os.path.join(d1, f), sep='\t')
        b = pd.read_csv(os.path.join(d2, f), sep='\t')
        m = a.merge(b, on=['read_id', 'basecall_pos'], suffixes=('_1', '_2'))
        p1, p2 = m['prob_1'].values >= thr1, m['prob_2'].values >= thr2
        n += len(m)
        c1 += int(p1.sum())
        c2 += int(p2.sum())
        both += int((p1 & p2).sum())
    return n, c1, c2, both, len(files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nanocon', default='/home/bio/8oxog/nanocon/scores')
    p.add_argument('--oxonet', default='/home/bio/8oxog/wt1/pack_scores_e2ep125')
    p.add_argument('--nanocon-native', default='/home/bio/8oxog/nanocon/scores/native')
    p.add_argument('--oxonet-native', default='/home/bio/8oxog/wt1/scores/oxonet_e2ep125')
    p.add_argument('--spec', type=float, default=0.999)
    p.add_argument('--esox-native', default='/home/bio/8oxog/wt1/scores/esox')
    p.add_argument('--esox-thr', type=float, default=0.95)   # esox 论文口径的操作点
    p.add_argument('--skip-native', action='store_true')
    args = p.parse_args()

    models = {'NanoCon': args.nanocon, 'OxoNet-E2ep125': args.oxonet}
    probs = {m: {s: load(d, s) for s in SETS} for m, d in models.items()}
    T = {m: tstar(*probs[m]['valid'], args.spec) for m in models}

    print('=' * 78)
    print('位点集合完全相同(pack 同一批 npz, 逐行对齐)。'
          ' T* = valid 阴性 {:.3%} 特异性分位数'.format(args.spec))
    for m in models:
        print('  T*[{}] = {:.6f}'.format(m, T[m]))
    print()

    print('--- 判别力(全局, 不是逐 batch 平均) ---')
    print('{:16s} {:12s} {:>9s} {:>9s}'.format('set', 'model', 'AUROC', 'AUPRC'))
    for s in SETS:
        for m in models:
            au, ap = auroc_auprc(*probs[m][s])
            print('{:16s} {:12s} {:9.5f} {:9.5f}'.format(s, m, au, ap))
    print()

    print('--- 各自 T* 下的 recall / FPR ---')
    print('{:16s} {:12s} {:>12s} {:>12s} {:>12s} {:>12s}'.format(
        'set', 'model', 'recall@0.5', 'FPR@0.5', 'recall@T*', 'FPR@T*'))
    for s in SETS:
        for m in models:
            pr, lb = probs[m][s]
            r5, f5 = rates(pr, lb, 0.5)
            rt, ft = rates(pr, lb, T[m])
            print('{:16s} {:12s} {:11.4%} {:11.4%} {:11.4%} {:11.4%}'.format(s, m, r5, f5, rt, ft))
    print()

    print('--- 同一特异性下的 recall(阈值在该测试集负样本上定, 最公平的头对头) ---')
    grid = [1e-2, 1e-3, 1e-4]
    print('{:16s} {:12s} {}'.format('set', 'model',
                                    ' '.join('{:>13s}'.format('rec@FPR=' + str(g)) for g in grid)))
    for s in ['test_oligo', 'test_t2t']:
        for m in models:
            pr, lb = probs[m][s]
            cells = ' '.join('{:12.4%} '.format(recall_at_fpr(pr, lb, g)[0]) for g in grid)
            print('{:16s} {:12s} {}'.format(s, m, cells))
    print()

    if args.skip_native:
        return
    print('--- native(wt1 真实基因组 DNA, 同一批 read) ---')
    nat = {'NanoCon': args.nanocon_native, 'OxoNet-E2ep125': args.oxonet_native}
    for m, d in nat.items():
        if not glob.glob(os.path.join(d, '*.tsv.gz')):
            print('  {}: 无打分文件, 跳过'.format(m))
            continue
        tot, called, _ = native_rate(d, T[m])
        print('  {:14s} {:>12,} 个 G 位点, T* 下判正 {:>10,} = {:.4%}'.format(
            m, tot, called, called / max(tot, 1)))
    if args.esox_native and glob.glob(os.path.join(args.esox_native, '*.tsv.gz')):
        only = {os.path.basename(p) for p in glob.glob(os.path.join(nat['NanoCon'], '*.tsv.gz'))}
        tot, called, _ = native_rate(args.esox_native, args.esox_thr, 'oxog_score', only)
        print('  {:14s} {:>12,} 个 G 位点, score>{:g} 判正 {:>10,} = {:.4%} (参考: esox 自己的'
              '操作点, 灵敏度未在同一 test 上标定)'.format('esox', tot, args.esox_thr, called,
                                                          called / max(tot, 1)))
    # T* 是"等特异性"工作点, 但两个模型在该点的灵敏度差一倍, 直接比 native 判正率不公平。
    # 这里把阈值挪到**同一召回**上(召回定义在同一批 test_oligo 阳性上)再比 native。
    print('  -- 等灵敏度(同一批 test 阳性上 recall 相同)下的 native 判正率 --')
    for tgt_m in models:
        pr_t, lb_t = probs[tgt_m]['test_oligo']
        tgt_rec = float((pr_t[lb_t == 1] >= T[tgt_m]).mean())
        print('     目标 recall = {:.2%} (= {} 在自己 T* 上的召回)'.format(tgt_rec, tgt_m))
        for m in models:
            pr, lb = probs[m]['test_oligo']
            thr = float(np.quantile(pr[lb == 1], 1.0 - tgt_rec))
            tot, called, _ = native_rate(nat[m], thr)
            print('       {:14s} thr={:.6f}  native 判正 {:>9,}/{:,} = {:.4%}'.format(
                m, thr, called, tot, called / max(tot, 1)))
    if all(glob.glob(os.path.join(d, '*.tsv.gz')) for d in nat.values()):
        n, c1, c2, both, nf = native_concordance(nat['NanoCon'], nat['OxoNet-E2ep125'],
                                                 T['NanoCon'], T['OxoNet-E2ep125'])
        exp = c1 / n * c2 / n * n
        print('  join {} 个 base / {:,} 位点: NanoCon 判正 {:,}, OxoNet 判正 {:,}, '
              '共同 {:,} (随机期望 {:.0f}, 富集 {:.1f}x)'.format(
                  nf, n, c1, c2, both, exp, both / max(exp, 1e-9)))


if __name__ == '__main__':
    main()
