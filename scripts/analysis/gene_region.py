"""基因区分档：5'UTR / 3'UTR / exon / intron / intergenic —— 补齐 esox Fig 4d 的口径。

esox Methods 原文：
  "8-oxo-dG counts on 5'-UTR, 3'-UTR, exon and intron regions were only considered if the
   molecule was found on the **same DNA strand** as the annotated region; in intergenic,
   satellite, centromeres and complex repetitive regions counts were considered regardless
   of the DNA strand. 5'UTR was annotated as the upstream 5000 base pairs before the start
   of all annotated coding sequences. 3'UTR was annotated as the downstream 5000 base pairs
   of all annotated coding sequences. Intergenic regions were considered as all intervals
   that had no annotation."

归一化同 `genome_dist.py`：判正率(按 coverage 归一) + 按 5-mer 组成的间接标准化。
重叠时的优先级（我们的选择，需在方法里写明）：5'UTR > 3'UTR > exon > intron > intergenic。

    /home/bio/anaconda3/bin/python gene_region.py --bases 20
"""
import argparse
import collections
import glob
import os

import numpy as np
import pandas as pd

from genome_dist import uid64, thr_at_fpr, std_rate, MODELS, WT, COORDS, ANNOT

GFF = os.path.join(ANNOT, 'chm13.draft_v2.0.gene_annotation.gff3')
FLANK = 5000


def parse_gff(path):
    """只取 exon / CDS / gene 三类。返回 {chrom: {kind: (starts, ends, strands)}}"""
    d = collections.defaultdict(lambda: collections.defaultdict(lambda: ([], [], [])))
    # 该 gff3 直接标注了 intron(1,204,286 条), 用它比从 gene-exon 推导更忠实
    keep = {'exon', 'CDS', 'intron'}
    n = 0
    with open(path) as f:
        for line in f:
            if line[0] == '#':
                continue
            fl = line.split('\t')
            if len(fl) < 8 or fl[2] not in keep:
                continue
            c, kind, s, e, st = fl[0], fl[2], int(fl[3]) - 1, int(fl[4]), fl[6]
            d[c][kind][0].append(s); d[c][kind][1].append(e); d[c][kind][2].append(st)
            n += 1
            if n % 5_000_000 == 0:
                print('  已读 {:,} 条特征'.format(n), flush=True)
    out = {}
    for c, kinds in d.items():
        out[c] = {}
        for k, (s, e, st) in kinds.items():
            s = np.array(s); e = np.array(e); st = np.array(st)
            o = np.argsort(s)
            out[c][k] = (s[o], e[o], st[o])
    print('gff3: {:,} 条 exon/CDS/intron 特征, {} 条染色体'.format(n, len(out)), flush=True)
    return out


def in_intervals(gpos, strand, s, e, st, shift_lo=0, shift_hi=0):
    """区间(可整体平移成上/下游窗口)命中且同链 -> bool。区间可重叠, 用累计计数法。"""
    lo = s + shift_lo
    hi = e + shift_hi
    res = np.zeros(len(gpos), bool)
    for want in ('+', '-'):
        m = strand == want
        if not m.any():
            continue
        k = st == want
        if not k.any():
            continue
        a = np.sort(lo[k]); b = np.sort(hi[k])
        # 命中数 = 起点<=x 的个数 − 终点<=x 的个数
        cnt = np.searchsorted(a, gpos[m], 'right') - np.searchsorted(b, gpos[m], 'right')
        res[m] = cnt > 0
    return res


def classify(chrom, gpos, strand, gff):
    """返回类别数组。优先级 5'UTR > 3'UTR > exon > intron > intergenic"""
    lab = np.full(len(gpos), 'intergenic', dtype=object)
    for c in np.unique(chrom):
        if c not in gff:
            continue
        m = chrom == c
        g, sd = gpos[m], strand[m]
        cur = np.full(m.sum(), 'intergenic', dtype=object)
        kinds = gff[c]
        if 'intron' in kinds:
            s, e, st = kinds['intron']
            cur[in_intervals(g, sd, s, e, st)] = 'intron'
        if 'exon' in kinds:
            s, e, st = kinds['exon']
            cur[in_intervals(g, sd, s, e, st)] = 'exon'
        if 'CDS' in kinds:
            s, e, st = kinds['CDS']
            # 5'UTR = CDS 起点上游 5kb（按链向）; 3'UTR = 终点下游 5kb
            up5 = in_intervals(g, sd, s, s, st, -FLANK, 0) & (st[:1].size > 0)
            dn3 = in_intervals(g, sd, e, e, st, 0, FLANK)
            # 负链方向相反
            plus = sd == '+'
            u = np.where(plus, up5, dn3)
            v = np.where(plus, dn3, up5)
            cur[u] = "5'UTR"
            cur[v & (cur != "5'UTR")] = "3'UTR"
        lab[m] = cur
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bases', type=int, default=20)
    ap.add_argument('--match-fpr', type=float, default=1e-4)
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/gene_region.txt')
    args = ap.parse_args()

    bases = sorted(os.path.basename(p).replace('.tsv.gz', '')
                   for p in glob.glob(os.path.join(WT, 'scores', 'oxonet_e2ep125', '*.tsv.gz')))
    bases = bases[:args.bases]

    parts = []
    for b in bases:
        ks, rid, bp = [], [], []
        for p in sorted(glob.glob(os.path.join(WT, 'out', b, 'oxo', b + '.w*.part*.npz'))):
            a = np.load(p, allow_pickle=False)
            if a['kmers'].shape[0] == 0:
                continue
            ks.append(a['kmers'].astype('U7')); rid.append(np.asarray(a['read_id']).reshape(-1))
            bp.append(np.asarray(a['basecall_pos']).reshape(-1))
        km = np.concatenate(ks)
        parts.append(pd.DataFrame({'uid': uid64(np.concatenate(rid)),
                                   'pos': np.concatenate(bp).astype(np.int64),
                                   'k5': np.array([s[1:6].replace('o', 'G') for s in km])}))
    sites = pd.concat(parts, ignore_index=True)
    for m, (pk, nd, col) in MODELS.items():
        thr = thr_at_fpr(pk, args.match_fpr)
        v = np.concatenate([pd.read_csv(os.path.join(nd, b + '.tsv.gz'), sep='\t',
                                        usecols=[col])[col].values for b in bases])
        sites[m] = v >= thr
    print('位点 {:,}'.format(len(sites)), flush=True)

    want = set(sites['uid'].unique().tolist())
    cs = []
    for ch in pd.read_csv(COORDS, sep='\t',
                          usecols=['read_id', 'basecall_pos', 'chrom', 'gpos', 'strand'],
                          chunksize=5_000_000):
        u = uid64(ch['read_id'].values)
        keep = np.fromiter((x in want for x in u), bool, len(u))
        if keep.any():
            cs.append(pd.DataFrame({'uid': u[keep],
                                    'pos': ch['basecall_pos'].values[keep].astype(np.int64),
                                    'chrom': ch['chrom'].values[keep],
                                    'gpos': ch['gpos'].values[keep].astype(np.int64),
                                    'strand': ch['strand'].values[keep]}))
    co = pd.concat(cs, ignore_index=True).drop_duplicates(['uid', 'pos'])
    d = sites.merge(co, on=['uid', 'pos'], how='inner')
    print('有坐标 {:,}'.format(len(d)), flush=True)

    gff = parse_gff(GFF)
    d['region'] = classify(d['chrom'].values, d['gpos'].values, d['strand'].values, gff)

    L = ['基因区分档（{} 个 f5, {:,} 个位点; 等绝对基因组 FPR = {:.0e}; 同链才计入, '
         '优先级 5\'UTR > 3\'UTR > exon > intron > intergenic）'.format(
             len(bases), len(d), args.match_fpr)]
    for m in MODELS:
        gr = d.groupby('k5')[m].mean().to_dict()
        L += ['', '=== {} ==='.format(m),
              '{:>12s} {:>12s} {:>11s} {:>11s} {:>9s}'.format(
                  'region', '位点数', '判正率', '期望(5mer)', '标准化比')]
        for k, g in d.groupby('region', observed=True):
            o, e, r = std_rate(g[m].values, g['k5'].values, gr)
            L.append('{:>12s} {:>12,} {:10.4%} {:10.4%} {:8.2f}x'.format(str(k), len(g), o, e, r))
    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    d[['chrom', 'gpos', 'strand', 'region', 'k5'] + list(MODELS)].to_parquet(
        os.path.join(WT, 'gene_region_sites.parquet'))
    print('写出', args.out)


if __name__ == '__main__':
    main()
