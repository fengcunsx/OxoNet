"""四方对比：OxoNet / esox / NanoCon / GBDT(特征天花板)，逐位点相同的集合。

答 R2.M3(benchmark 太窄)。四行的分工：
  OxoNet  我们的方法(175 点原始电流)
  esox    已发表、任务专用、同样吃原始信号 —— 最强的对照
  NanoCon 审稿人点名的近期通用 caller(21 个摘要统计量)
  GBDT    与 NanoCon 完全相同输入的梯度提升树 = **该输入表征的上界**

分数尺度不可比，所以一律比**工作点**。**阈值一律在 valid 的基因组阴性上确定**
(2026-08-03 审计后定稿, 见 `8oxog/METHOD_AUDIT.md`): 测试集只用来报结果, 消除
"操作点选在测试集上"的质疑。valid 的基因组阴性靠"每 read 位点数 > 100"分离
(混入率 0.13%/漏检 1.42%, 见 valid_neg_source.py), 共 2,090,100 个位点。
  ① 各自 T*(valid **基因组**阴性 99.9% 特异性) 下的 recall/FPR
  ② 同一 FPR 网格下的 recall(阈值同样在 valid 基因组阴性上定) —— 最公平的头对头
     FPR=1e-6 不列: test_t2t 仅 1.94M 阴性, 该档由约 2 个阴性决定(见 threshold_audit.py)
  ③ esox 另给一行它论文自己的操作点(score > 0.95)
  ④ native: 等灵敏度下的每百万 G 判正数

    /home/bio/anaconda3/bin/python four_way_report.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

PACK = '/home/bio/8oxog/build/pack'
VALID_GENOMIC_MASK = os.path.join(PACK, 'valid_genomic_neg_mask.npy')
FPR_GRID = (1e-2, 1e-3, 1e-4, 1e-5)
SETS = ['valid', 'test_oligo', 'test_t2t']
# 模型 -> (pack 打分目录, native 打分目录 或 None)
MODELS = {
    'OxoNet': ('/home/bio/8oxog/wtl1/pack_scores_e2ep125',
               '/home/bio/8oxog/wtl1/scores/oxonet_e2ep125'),
    'esox': ('/home/bio/8oxog/wtl1/pack_scores_esox',
             '/home/bio/8oxog/wtl1/scores/esox'),
    'NanoCon': ('/home/bio/8oxog/nanocon/scores',
                '/home/bio/8oxog/nanocon/scores/native'),
    'GBDT(天花板)': ('/home/bio/8oxog/nanocon', None),
}
NATIVE_COL = {'esox': 'oxog_score'}     # 其余是 'prob'


def probs(d, name):
    for c in (os.path.join(d, name + '.probs.npz'), os.path.join(d, 'gbdt_' + name + '.probs.npz')):
        if os.path.isfile(c):
            a = np.load(c)
            return a['prob'].astype(np.float64), a['label'].astype(np.int8)
    return None, None


def native_rate(d, thr, col, only=None):
    tot = called = 0
    for p in sorted(glob.glob(os.path.join(d, '*.tsv.gz'))):
        if only is not None and os.path.basename(p) not in only:
            continue
        v = pd.read_csv(p, sep='\t', usecols=[col])[col].values
        tot += len(v)
        called += int((v >= thr).sum())
    return tot, called


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=float, default=0.999)
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/four_way_report.txt')
    ap.add_argument('--skip-native', action='store_true')
    args = ap.parse_args()
    global GMASK
    GMASK = np.load(VALID_GENOMIC_MASK)
    from sklearn.metrics import roc_auc_score, average_precision_score

    P, T = {}, {}
    for m, (d, _) in MODELS.items():
        P[m] = {s: probs(d, s) for s in SETS}
        vp, vl = P[m]['valid']
        if vp is None:
            print('!! {} 缺 valid 打分, 跳过该模型'.format(m))
            continue
        n = int(np.isnan(vp).sum())
        if n:
            print('!! {} valid 有 {} 个 NaN(未匹配), 按 -1 处理'.format(m, n))
            vp = np.nan_to_num(vp, nan=-1.0)
            P[m]['valid'] = (vp, vl)
        T[m] = float(np.quantile(vp[GMASK], args.spec))
    live = [m for m in MODELS if m in T]

    L = ['四方对比(逐位点相同集合; T* @ valid **基因组**阴性 {:.1%} 特异性; '
         '阈值一律在 valid 上定, 测试集只报结果)'.format(args.spec),
         '  valid 基因组阴性 = {:,} 个位点'.format(int(GMASK.sum()))]
    L += ['  T*[{}] = {:.6f}'.format(m, T[m]) for m in live]

    L += ['', '--- 判别力(全局) ---',
          '{:14s} {:12s} {:>9s} {:>9s}'.format('set', 'model', 'AUROC', 'AUPRC')]
    for s in SETS:
        for m in live:
            p, l = P[m][s]
            if p is None:
                continue
            p = np.nan_to_num(p, nan=-1.0)
            L.append('{:14s} {:12s} {:9.5f} {:9.5f}'.format(
                s, m, roc_auc_score(l, p), average_precision_score(l, p)))

    L += ['', '--- 同一 FPR 下的 recall(阈值一律在 valid 上定, 两种阴性背景分开报) ---',
          '  合成背景 = valid 的 oligo 阴性(与文献/esox 原论文口径可比)',
          '  基因组背景 = valid 的基因组阴性(**论文主口径**: 真实全基因组分析面对的是它)',
          '{:14s} {:12s} {:>13s} {:>13s} {:>13s} {:>13s}'.format(
              '阴性背景', 'model', 'rec@FPR1e-2', 'rec@FPR1e-3', 'rec@FPR1e-4', 'rec@FPR1e-5')]
    vlab = P[live[0]]['valid'][1]
    OMASK = (vlab == 0) & (~GMASK)          # valid 里的 oligo 阴性
    for tag, mask in (('合成(oligo)', OMASK), ('基因组(T2T)', GMASK)):
        n_ = int(mask.sum())
        warn = ' '.join('FPR={:.0e}→{:.0f}个事件{}'.format(g, g * n_, '⚠' if g * n_ < 10 else '')
                        for g in FPR_GRID if g * n_ < 50)
        L.append('  [{} 阴性 {:,} 个; 定阈值时 {}]'.format(tag, n_, warn or '各档事件数均 >=50'))
        for m in live:
            p_, l_ = P[m]['test_oligo']
            if p_ is None:
                continue
            p_ = np.nan_to_num(p_, nan=-1.0)
            vp = np.nan_to_num(P[m]['valid'][0], nan=-1.0)[mask]
            cells = ['{:12.2%} '.format((p_[l_ == 1] >= float(np.quantile(vp, 1 - g))).mean())
                     for g in FPR_GRID]
            L.append('{:14s} {:12s} {}'.format('', m, ' '.join(cells)))

    L += ['', '--- 各自 T* 下的 recall / FPR (valid 行的 FPR 是**全部** valid 阴性, 含 oligo) ---',
          '{:14s} {:12s} {:>11s} {:>11s}'.format('set', 'model', 'recall@T*', 'FPR@T*')]
    for s in SETS:
        for m in live:
            p, l = P[m][s]
            if p is None:
                continue
            p = np.nan_to_num(p, nan=-1.0)
            L.append('{:14s} {:12s} {:11.4%} {:11.4%}'.format(
                s, m, (p[l == 1] >= T[m]).mean(), (p[l == 0] >= T[m]).mean()))
    if 'esox' in live:      # esox 论文自己的操作点
        for s in SETS:
            p, l = P['esox'][s]
            p = np.nan_to_num(p, nan=-1.0)
            L.append('{:14s} {:12s} {:11.4%} {:11.4%}  <- esox@0.95(其论文口径)'.format(
                s, 'esox', (p[l == 1] >= 0.95).mean(), (p[l == 0] >= 0.95).mean()))

    if not args.skip_native:
        L += ['', '--- native(wtl1, 同一批 read): 等灵敏度下的每百万 G 判正数 ---']
        nat = {m: d for m, (_, d) in MODELS.items() if d and glob.glob(os.path.join(d, '*.tsv.gz'))}
        common = None                      # 只统计所有模型都打过分的 base
        for m, d in nat.items():
            b = {os.path.basename(p) for p in glob.glob(os.path.join(d, '*.tsv.gz'))}
            common = b if common is None else (common & b)
        L.append('  共同 base 数: {}'.format(len(common or [])))
        # ⚠️ 目标 recall 与各模型阈值**一律在 valid 阳性上确定**(2026-08-03 修正)。
        # 此前用的是 test_oligo 阳性, 与 §Evaluation 里"阈值从不在测试集上拟合"自相矛盾。
        L.append('  (目标 recall 与阈值均在 **valid 阳性**上定; 测试集只报结果)')
        for tgt in live:                   # 以每个模型自己 T* 在 valid 上的召回为目标
            p, l = P[tgt]['valid']
            if p is None:
                continue
            rec = float((np.nan_to_num(p, nan=-1.0)[l == 1] >= T[tgt]).mean())
            L.append('  目标 recall = {:.2%} (= {} @T*, 在 valid 上)'.format(rec, tgt))
            for m in nat:
                pm, lm = P[m]['valid']
                if pm is None:
                    continue
                thr = float(np.quantile(np.nan_to_num(pm, nan=-1.0)[lm == 1], 1 - rec))
                tot, called = native_rate(nat[m], thr, NATIVE_COL.get(m, 'prob'), common)
                L.append('    {:12s} thr={:.4f}  {:>9,}/{:>12,} = {:8.4%} = {:7.1f}/百万 G'.format(
                    m, thr, called, tot, called / max(tot, 1), 1e6 * called / max(tot, 1)))

    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print('\n写出', args.out)


if __name__ == '__main__':
    main()
