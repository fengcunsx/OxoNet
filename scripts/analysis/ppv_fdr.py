"""真实丰度下的 PPV / FDR —— 答 R2.M2。

审稿人的话："even a small false positive rate can produce thousands of erroneous
calls across the human genome. At one modification per million guanines, high
recall alone may not lead to reliable biological findings."

口径（2026-08-03 审计后定稿，见 `8oxog/METHOD_AUDIT.md`）：
  1. **工作点(阈值)一律在 valid 的基因组阴性上确定**，测试集只用来报结果 ——
     消除"操作点选在测试集上"的质疑。valid 的基因组阴性靠"每 read 位点数 > 100"分离
     （混入率 0.13% / 漏检 1.42%，见 `valid_neg_source.py`），共 2,090,100 个位点。
  2. recall 取 **test_oligo 阳性**（有 ground truth）；FPR 在 **test_t2t 基因组阴性**上实测
     —— 合成 DNA 不是真实阴性背景。
  3. PPV = π·recall / [π·recall + (1−π)·FPR]，FDR = 1 − PPV。
  4. **FPR=1e-6 档已删除**：test_t2t 仅 1.94M 阴性，该档阈值由约 2 个阴性决定
     （半分裂偏差 112–218%）。1e-5 档仅约 19 个事件，必须带 CI。
  5. **报告的 FPR 是上界**：基因组阴性集来自真实人类 DNA，本身含约 75/百万的真实
     8-oxo-dG，故 FPR 收到 ~1e-4 以下时，计入的"假阳"里混有真损伤。这个下限与样本量无关。

    /home/bio/anaconda3/bin/python ppv_fdr.py
"""
import argparse
import os

import numpy as np

PACK = '/home/bio/8oxog/build/pack'
VALID_GENOMIC_MASK = os.path.join(PACK, 'valid_genomic_neg_mask.npy')
MODELS = {
    'OxoNet': '/home/bio/8oxog/wtl1/pack_scores_e2ep125',
    'esox': '/home/bio/8oxog/wtl1/pack_scores_esox',
    'NanoCon': '/home/bio/8oxog/nanocon/scores',
    'GBDT(天花板)': '/home/bio/8oxog/nanocon',
}
FPR_GRID = (1e-3, 1e-4, 1e-5)          # 1e-6 已删: 低于分辨力
TRUE_BACKGROUND = 7.5e-5               # esox 实测基因组本底 ~75/百万


def load(d, name):
    for c in (os.path.join(d, name + '.probs.npz'), os.path.join(d, 'gbdt_' + name + '.probs.npz')):
        if os.path.isfile(c):
            a = np.load(c)
            return np.nan_to_num(a['prob'].astype(np.float64), nan=-1.0), a['label'].astype(np.int8)
    return None, None


def ppv(pi, rec, fpr):
    den = pi * rec + (1 - pi) * fpr
    return pi * rec / den if den > 0 else float('nan')


def poisson_ci(k):
    """观测 k 个事件的 Poisson 95% CI(Garwood)。"""
    if k <= 0:
        return 0.0, 3.69
    lo = k * (1 - 1 / (9.0 * k) - 1.96 / (3 * np.sqrt(k))) ** 3
    hi = (k + 1) * (1 - 1 / (9.0 * (k + 1)) + 1.96 / (3 * np.sqrt(k + 1))) ** 3
    return max(lo, 0.0), hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/ppv_fdr.txt')
    ap.add_argument('--target-ppv', type=float, default=0.5)
    ap.add_argument('--target-prev', type=float, default=TRUE_BACKGROUND)
    args = ap.parse_args()

    gmask = np.load(VALID_GENOMIC_MASK)
    n_valid_g = int(gmask.sum())

    L = ['真实丰度下的 PPV / FDR（答 R2.M2）',
         '  阈值来源: valid 的基因组阴性 {:,} 个位点(分辨力 {:.2e})'.format(n_valid_g, 1.0 / n_valid_g),
         '  recall: test_oligo 阳性 | FPR: test_t2t 基因组阴性上**实测**(带 Poisson 95% CI)',
         '  ⚠ FPR 为上界: 基因组阴性含约 75/百万真实 8-oxo-dG',
         '']
    header = ('{:12s} {:>11s} {:>8s} {:>8s} {:>10s} {:>21s} {:>9s} {:>9s} {:>9s}'.format(
        'model', '工作点', 'thr', 'recall', '实测FPR', 'FPR 95% CI', 'PPV@1', 'PPV@75', 'PPV@100'))
    L += [header, '  (PPV@N = 本底 N/百万 G 时的单分子 PPV)']

    for m, d in MODELS.items():
        po, lo = load(d, 'test_oligo')
        pt, lt = load(d, 'test_t2t')
        vp, vl = load(d, 'valid')
        if po is None or pt is None or vp is None:
            L.append('{:12s} (缺打分, 跳过)'.format(m))
            continue
        pos, neg = po[lo == 1], pt[lt == 0]
        vneg_g = vp[gmask]                      # ← 阈值只在这上面定

        # 注意 T*(99.9% 特异性) 与 FPR=1e-3 是同一个点, 故只按 FPR 网格列一次
        pts = []
        for g in FPR_GRID:
            tag = 'FPR={:.0e}{}'.format(g, '=T*' if g == 1e-3 else '')
            pts.append((tag, float(np.quantile(vneg_g, 1 - g))))
        if m == 'esox':
            pts.append(('论文@0.95', 0.95))

        for tag, thr in pts:
            r = float((pos >= thr).mean())
            k = int((neg >= thr).sum())         # 实测假阳事件数
            f = k / len(neg)
            klo, khi = poisson_ci(k)
            flo, fhi = klo / len(neg), khi / len(neg)
            flag = '' if k >= 50 else (' [n={}]'.format(k) if k >= 10 else ' [n={} 不可用]'.format(k))
            L.append('{:12s} {:>11s} {:8.4f} {:8.2%} {:10.5%} [{:8.2e},{:8.2e}] {:9.4%} {:9.2%} {:9.2%}{}'
                     .format(m, tag, thr, r, f, flo, fhi,
                             ppv(1e-6, r, f), ppv(7.5e-5, r, f), ppv(1e-4, r, f), flag))

        # 要达到 target-ppv @ target-prev 需要多严的 FPR / 还剩多少 recall
        pi, tp = args.target_prev, args.target_ppv
        need_fpr = pi * (1 - tp) / (tp * (1 - pi))
        if need_fpr < 1.0 / n_valid_g:
            note = 'FPR 需 <{:.2e}, 低于 valid 基因组阴性的分辨力(1/{:,})'.format(need_fpr, n_valid_g)
        else:
            thr = float(np.quantile(vneg_g, 1 - need_fpr))
            k = int((neg >= thr).sum())
            note = ('FPR 需 <{:.2e} → thr {:.4f}, 此时 recall {:.2%} '
                    '(测试集实测 {} 个假阳)'.format(
                        need_fpr, thr, float((pos >= thr).mean()), k))
        L.append('{:12s}   要在 {:.0f}/百万 本底下拿到 PPV {:.0%}: {}'.format(
            m, pi * 1e6, tp, note))
        L.append('')

    L += ['注 1: 审稿人假设的 1/百万 本底下, 所有方法单分子 PPV 都 <20% —— 这一点他说得对;',
          '      但 esox 实测的真实本底是 ~75/百万, PPV@75 才是现实工作条件。',
          '注 2: 本数据集覆盖度 ~0.16x(151 f5)/~1x(全量 959 f5), 位点级 k-of-n 聚合需 >=5x,',
          '      即每样本约 5,000 个 f5 = 5 张 flow cell → 该数据集不可行, 上表均为**单分子** PPV。',
          '注 3: FPR=1e-6 档已删除(阈值由约 2 个阴性决定); 事件数 <50 的行已标注 [n=..]。']
    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print('写出', args.out)


if __name__ == '__main__':
    main()
