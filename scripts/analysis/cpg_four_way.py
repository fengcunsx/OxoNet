"""CpG 分层，四方版 —— 合成数据训练的共性失效模式？

已知（ep125 单模型）：基因组假阳高度集中在 CpG（native 富集 17.6x），而**无甲基化的
合成 oligo 上反而耗竭**。若 esox / NanoCon / GBDT 也如此，这就不是"我们模型的毛病"，
而是**用无甲基化合成 DNA 训练的共性失效模式** —— 分量完全不同（答 M1 + M2）。

CpG 定义同 `cpg_stratified.py`：中心 5-mer（7-mer 的 [1:6]）含 'CG'。

比较必须在**同一工作点**上。这里给两组：各自 T*，以及**等灵敏度**（统一到某个 recall），
后者才能横向比"谁的假阳更集中在 CpG"。

native 没有标签，报判正率；kmer 从 oxo 特征 npz 取，按 (read_id, basecall_pos) 核验对齐。

    /home/bio/anaconda3/bin/python cpg_four_way.py --native-bases 30
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

PACK = '/home/bio/8oxog/build/pack'
WT = '/home/bio/8oxog/wtl1'
MODELS = {
    'OxoNet': ('/home/bio/8oxog/wtl1/pack_scores_e2ep125',
               '/home/bio/8oxog/wtl1/scores/oxonet_e2ep125', 'prob'),
    'esox': ('/home/bio/8oxog/wtl1/pack_scores_esox',
             '/home/bio/8oxog/wtl1/scores/esox', 'oxog_score'),
    'NanoCon': ('/home/bio/8oxog/nanocon/scores',
                '/home/bio/8oxog/nanocon/scores/native', 'prob'),
    'GBDT(天花板)': ('/home/bio/8oxog/nanocon', None, None),
}


def load(d, name):
    for c in (os.path.join(d, name + '.probs.npz'), os.path.join(d, 'gbdt_' + name + '.probs.npz')):
        if os.path.isfile(c):
            a = np.load(c)
            return np.nan_to_num(a['prob'].astype(np.float64), nan=-1.0), a['label'].astype(np.int8)
    return None, None


def is_cpg(kmers):
    """中心 5-mer 含 CG"""
    return np.array(['CG' in s[1:6].replace('o', 'G') for s in kmers])


def native_kmers(base):
    """按 shard 排序拼接, 与打分文件同序(run_e2_all / score_nanocon 都是这个顺序)"""
    ks, rid, bp = [], [], []
    for p in sorted(glob.glob(os.path.join(WT, 'out', base, 'oxo', base + '.w*.part*.npz'))):
        a = np.load(p, allow_pickle=False)
        if a['kmers'].shape[0] == 0:
            continue
        ks.append(a['kmers'].astype('U7'))
        rid.append(np.asarray(a['read_id']).reshape(-1))
        bp.append(np.asarray(a['basecall_pos']).reshape(-1))
    return np.concatenate(ks), np.concatenate(rid), np.concatenate(bp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=float, default=0.999)
    ap.add_argument('--match-recall', type=float, default=0.3685,
                    help='等灵敏度比较用的 recall(默认 = esox 自己 T* 的召回)')
    ap.add_argument('--match-fpr', type=float, default=1e-4,
                    help='等【绝对 FPR】比较用的基因组假阳率(在 **valid 基因组阴性**上定阈值)。'
                         '富集倍数跨方法可比的唯一正确口径 —— 等 recall 时各方绝对 FPR 差很多, '
                         '低 FPR 的一方富集倍数会被系统性抬高。')
    ap.add_argument('--native-bases', type=int, default=30)
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/cpg_four_way.txt')
    args = ap.parse_args()

    P, T, Tm, Tf = {}, {}, {}, {}
    for m, (d, _, _) in MODELS.items():
        P[m] = {s: load(d, s) for s in ('valid', 'test_oligo', 'test_t2t')}
        vp, vl = P[m]['valid']
        if vp is None:
            continue
        gm = np.load('/home/bio/8oxog/build/pack/valid_genomic_neg_mask.npy')
        T[m] = float(np.quantile(vp[gm], args.spec))                     # T* 也用基因组阴性
        po, lo = P[m]['test_oligo']
        Tm[m] = float(np.quantile(po[lo == 1], 1 - args.match_recall))   # 等灵敏度阈值
        Tf[m] = float(np.quantile(vp[gm], 1 - args.match_fpr))           # 等绝对 FPR(valid 上定)
    live = [m for m in MODELS if m in T]

    L = ['CpG 分层（CpG = 中心 5-mer 含 CG）',
         '  阈值一律在 **valid 基因组阴性**上定(见 METHOD_AUDIT.md); T* = {:.1%} 特异性;'
         ' 等灵敏度 = test_oligo 阳性上 recall {:.2%}; 等FPR = {:.0e}'.format(
             args.spec, args.match_recall, args.match_fpr)]

    for name in ('test_oligo', 'test_t2t'):
        src = np.load(os.path.join(PACK, name + '.npz'), allow_pickle=False)
        cg = is_cpg(src['kmers'].astype('U7').tolist())
        lab = src['label'].astype(int)
        neg = lab == 0
        L += ['', '=== {} (阴性位点中 CpG 占 {:.1%}) ==='.format(name, cg[neg].mean()),
              '{:12s} {:>10s} {:>11s} {:>11s} {:>9s} {:>12s}'.format(
                  'model', '工作点', 'CpG FPR', '非CpG FPR', '富集', 'CpG占假阳')]
        for m in live:
            p, _ = P[m][name]
            for tag, thr in (('T*', T[m]), ('等灵敏度', Tm[m]), ('等FPR', Tf[m])):
                call = p >= thr
                a = call[neg & cg].mean()
                b = call[neg & ~cg].mean()
                share = (call & neg & cg).sum() / max((call & neg).sum(), 1)
                L.append('{:12s} {:>10s} {:10.4%} {:10.4%} {:8.1f}x {:11.1%}'.format(
                    m, tag, a, b, a / max(b, 1e-12), share))

    # ---- native ----
    bases = sorted(os.path.basename(p).replace('.tsv.gz', '')
                   for p in glob.glob(os.path.join(WT, 'scores', 'oxonet_e2ep125', '*.tsv.gz')))
    bases = bases[:args.native_bases]
    L += ['', '=== native wtl1 ({} 个 f5) ==='.format(len(bases)),
          '{:12s} {:>10s} {:>11s} {:>11s} {:>9s} {:>12s}'.format(
              'model', '工作点', 'CpG 判正率', '非CpG 判正率', '富集', 'CpG占判正')]
    kcache = {}
    for m in live:
        _, nd, col = MODELS[m]
        if nd is None:
            continue
        agg = {k: np.zeros(4) for k in ('等灵敏度', '等FPR')}   # cpg_call, cpg_n, non_call, non_n
        for b in bases:
            f = os.path.join(nd, b + '.tsv.gz')
            if not os.path.isfile(f):
                continue
            if b not in kcache:
                kcache[b] = native_kmers(b)
            km, rid, bp = kcache[b]
            df = pd.read_csv(f, sep='\t')
            if len(df) == len(km) and df['read_id'].iloc[0] == rid[0] and \
                    int(df['basecall_pos'].iloc[-1]) == int(bp[-1]):
                cg = is_cpg(km.tolist())                      # 同序快路径
            else:                                             # 退回 join
                mm = pd.DataFrame({'read_id': rid, 'basecall_pos': bp,
                                   'cg': is_cpg(km.tolist())})
                df = df.merge(mm, on=['read_id', 'basecall_pos'], how='left')
                cg = df['cg'].fillna(False).to_numpy(bool)
            v = df[col].to_numpy()
            for k, thr in (('等灵敏度', Tm[m]), ('等FPR', Tf[m])):
                call = v >= thr
                agg[k] += [np.sum(call & cg), cg.sum(), np.sum(call & ~cg), (~cg).sum()]
        for k in ('等灵敏度', '等FPR'):
            g = agg[k]
            a, b_ = g[0] / max(g[1], 1), g[2] / max(g[3], 1)
            L.append('{:12s} {:>10s} {:10.4%} {:10.4%} {:8.1f}x {:11.1%}'.format(
                m, k, a, b_, a / max(b_, 1e-12), g[0] / max(g[0] + g[2], 1)))
    L.append('  (native CpG 位点占比 {:.1%})'.format(
        float(np.mean([is_cpg(kcache[b][0].tolist()).mean() for b in kcache])) if kcache else 0))

    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print('\n写出', args.out)


if __name__ == '__main__':
    main()
