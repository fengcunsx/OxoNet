"""按比对质量(MAPQ)分层 native 判正率 —— 答 R2.M1 的 "mapping ambiguity"。

审稿人担心真实样本里的 mapping ambiguity 会让方法失灵。MAPQ 是 minimap2 给的
比对唯一性度量：MAPQ 0 = 该 read 在基因组上有多处同样好的位置(重复区/低复杂度)，
MAPQ 60 = 唯一比对。若判正率在低 MAPQ 上暴涨，就说明假阳被 mapping ambiguity 驱动。

join 方式：read_id(UUID) 取前 16 个 hex → int64，与 basecall_pos 组成两列整数键，
避免在 1.7 亿行上做字符串 join（内存扛不住）。

阈值口径：**等绝对基因组 FPR**(在 test_t2t 阴性上定)，跨方法唯一可比的口径。

    /home/bio/anaconda3/bin/python mapq_strat.py --bases 30
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

WT = '/home/bio/8oxog/wtl1'
COORDS = os.path.join(WT, 'coords', 'all_mapq.tsv.gz')
MODELS = {
    'OxoNet': ('/home/bio/8oxog/wtl1/pack_scores_e2ep125',
               '/home/bio/8oxog/wtl1/scores/oxonet_e2ep125', 'prob'),
    'esox': ('/home/bio/8oxog/wtl1/pack_scores_esox',
             '/home/bio/8oxog/wtl1/scores/esox', 'oxog_score'),
    'NanoCon': ('/home/bio/8oxog/nanocon/scores',
                '/home/bio/8oxog/nanocon/scores/native', 'prob'),
}
BINS = [-1, 0, 4, 29, 59, 60]          # 0 / 1-4 / 5-29 / 30-59 / 60
LBL = ['MAPQ 0', '1-4', '5-29', '30-59', '60(唯一)']


def uid64(s):
    """UUID 前 16 个 hex -> int64(可能为负, 无所谓, 只当键用)"""
    return np.array([int(x[:8] + x[9:13] + x[14:18], 16) & 0x7FFFFFFFFFFFFFFF
                     for x in s], dtype=np.int64)


def thr_at_fpr(pack_dir, fpr):
    a = np.load(os.path.join(pack_dir, 'test_t2t.probs.npz'))
    p = np.nan_to_num(a['prob'].astype(np.float64), nan=-1.0)
    return float(np.quantile(p[a['label'] == 0], 1 - fpr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bases', type=int, default=30)
    ap.add_argument('--match-fpr', type=float, default=1e-4)
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/mapq_strat.txt')
    args = ap.parse_args()

    bases = sorted(os.path.basename(p).replace('.tsv.gz', '')
                   for p in glob.glob(os.path.join(WT, 'scores', 'oxonet_e2ep125', '*.tsv.gz')))
    bases = bases[:args.bases]
    print('用 {} 个 f5'.format(len(bases)), flush=True)

    # 各模型的分数, 按 (uid, pos) 索引
    frames = {}
    for m, (pk, nd, col) in MODELS.items():
        thr = thr_at_fpr(pk, args.match_fpr)
        parts = []
        for b in bases:
            f = os.path.join(nd, b + '.tsv.gz')
            if os.path.isfile(f):
                d = pd.read_csv(f, sep='\t', usecols=['read_id', 'basecall_pos', col])
                parts.append(pd.DataFrame({'uid': uid64(d['read_id'].values),
                                           'pos': d['basecall_pos'].values.astype(np.int64),
                                           'call': (d[col].values >= thr)}))
        frames[m] = pd.concat(parts, ignore_index=True)
        print('{}: thr={:.4f}, {:,} 位点, 判正 {:,}'.format(
            m, thr, len(frames[m]), int(frames[m]['call'].sum())), flush=True)

    keys = frames[list(frames)[0]][['uid', 'pos']]
    want = set(keys['uid'].unique().tolist())
    print('涉及 {:,} 条 read, 开始扫 coords ...'.format(len(want)), flush=True)

    cs = []
    for ch in pd.read_csv(COORDS, sep='\t', usecols=['read_id', 'basecall_pos', 'mapq'],
                          chunksize=5_000_000):
        u = uid64(ch['read_id'].values)
        keep = np.fromiter((x in want for x in u), bool, len(u))
        if keep.any():
            cs.append(pd.DataFrame({'uid': u[keep],
                                    'pos': ch['basecall_pos'].values[keep].astype(np.int64),
                                    'mapq': ch['mapq'].values[keep].astype(np.int16)}))
    co = pd.concat(cs, ignore_index=True).drop_duplicates(['uid', 'pos'])
    print('coords 命中 {:,} 行'.format(len(co)), flush=True)

    L = ['按 MAPQ 分层的 native 判正率（等绝对基因组 FPR = {:.0e}，{} 个 f5）'.format(
        args.match_fpr, len(bases)),
        '{:10s} {:>12s} {:>10s} {:>10s} {:>10s} {:>12s}'.format(
            'model', *LBL)]
    n_line = None
    for m, df in frames.items():
        j = df.merge(co, on=['uid', 'pos'], how='inner')
        g = pd.cut(j['mapq'], BINS, labels=LBL)
        rate = j.groupby(g, observed=False)['call'].mean()
        cnt = j.groupby(g, observed=False)['call'].size()
        L.append('{:10s} {}'.format(m, ' '.join('{:11.4%}'.format(rate.get(k, np.nan))
                                                for k in LBL)))
        n_line = '{:10s} {}'.format('位点数', ' '.join('{:>11,}'.format(int(cnt.get(k, 0)))
                                                       for k in LBL))
    L.append(n_line)
    L.append('')
    L.append('注: MAPQ 0 = 多处等好比对(重复区/低复杂度); 60 = 唯一比对。')
    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print('写出', args.out)


if __name__ == '__main__':
    main()
