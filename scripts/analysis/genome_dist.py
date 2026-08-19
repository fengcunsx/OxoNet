"""基因组分布：GC 相关 + 重复区/卫星/着丝粒富集 —— 答 R2.M2，对标 esox Fig 4b/4d。

**照 esox Methods 的口径**（原文核对过）：
  参考基因组 CHM13v2 + minimap2（我们本来就是）；计数用 score > 阈值；
  "counts were normalized per coverage **as well as per 5-mer relative abundance per region**"。
  → 两重归一化在这里的实现：
    ① per coverage = 用**判正率**(判正数/该区域观测到的 G 数)，覆盖度自动约掉；
    ② per 5-mer = **间接标准化**：期望率 = Σ_5mer (该区域的 5-mer 计数 × 该 5-mer 的全局判正率) / 该区域总数，
       报 观测/期望。不做这步，GC 富集会被 5-mer 组成混淆（与 CpG 分析必须按 5-mer 配对同理）。

区域定义：RepeatMasker(按 repeat class) / censat(卫星与着丝粒)/ 其余=非重复。
esox 对 intergenic/satellite/centromere/repeat 是**不分链**统计的，这里同样不分链。
（5'UTR/3'UTR/exon/intron 需要 gene_annotation.gff3，下完再补。）

    /home/bio/anaconda3/bin/python genome_dist.py --bases 20
"""
import argparse
import collections
import glob
import gzip
import os

import numpy as np
import pandas as pd

WT = '/home/bio/8oxog/wtl1'
REF = '/home/bio/8oxog/chm13v2/chm13v2.0.fa.gz'
ANNOT = '/home/bio/8oxog/chm13v2/annot'
COORDS = os.path.join(WT, 'coords', 'all_mapq.tsv.gz')
MODELS = {
    'OxoNet': ('/home/bio/8oxog/wtl1/pack_scores_e2ep125',
               '/home/bio/8oxog/wtl1/scores/oxonet_e2ep125', 'prob'),
    'esox': ('/home/bio/8oxog/wtl1/pack_scores_esox',
             '/home/bio/8oxog/wtl1/scores/esox', 'oxog_score'),
    'NanoCon': ('/home/bio/8oxog/nanocon/scores',
                '/home/bio/8oxog/nanocon/scores/native', 'prob'),
}
BIN = 1000


def uid64(s):
    return np.array([int(x[:8] + x[9:13] + x[14:18], 16) & 0x7FFFFFFFFFFFFFFF
                     for x in s], dtype=np.int64)


VALID_GENOMIC_MASK = '/home/bio/8oxog/build/pack/valid_genomic_neg_mask.npy'


def thr_at_fpr(pack_dir, fpr):
    """阈值定在 **valid 的基因组阴性**上(2026-08-03 口径, 见 METHOD_AUDIT.md):
    测试集只报结果, 不参与定工作点。掩码由 valid_neg_source.py 生成。"""
    a = np.load(os.path.join(pack_dir, 'valid.probs.npz'))
    p = np.nan_to_num(a['prob'].astype(np.float64), nan=-1.0)
    return float(np.quantile(p[np.load(VALID_GENOMIC_MASK)], 1 - fpr))


def gc_by_bin(path=REF):
    """{chrom: float32 array of GC fraction per 1kb bin}，逐条染色体流式算，不留全序列。"""
    out, cur, buf = {}, None, []
    def flush():
        if cur is None:
            return
        s = np.frombuffer(''.join(buf).encode(), dtype=np.uint8)
        n = len(s) // BIN * BIN
        s = s[:n].reshape(-1, BIN)
        gc = ((s == ord('G')) | (s == ord('C')) | (s == ord('g')) | (s == ord('c'))).sum(1)
        out[cur] = (gc / BIN).astype(np.float32)
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line[0] == '>':
                flush(); buf = []
                cur = line[1:].split()[0]
            else:
                buf.append(line.strip())
    flush()
    return out


def load_bed(path, name_col=3, chrom_prefix_ok=True):
    """{chrom: (starts, ends, names)} 已按 start 排序"""
    d = collections.defaultdict(lambda: ([], [], []))
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        if len(f) <= name_col or not f[1].isdigit():
            continue
        d[f[0]][0].append(int(f[1])); d[f[0]][1].append(int(f[2])); d[f[0]][2].append(f[name_col])
    out = {}
    for c, (s, e, n) in d.items():
        s = np.array(s); e = np.array(e); n = np.array(n)
        o = np.argsort(s)
        out[c] = (s[o], e[o], n[o])
    return out


def annotate(chrom, gpos, bed):
    """每个位点落在哪个区间(取第一个覆盖它的); 返回 name 数组, 未覆盖为 ''"""
    res = np.full(len(gpos), '', dtype=object)
    for c in np.unique(chrom):
        if c not in bed:
            continue
        m = chrom == c
        s, e, n = bed[c]
        idx = np.searchsorted(s, gpos[m], side='right') - 1
        ok = (idx >= 0) & (e[np.clip(idx, 0, len(e) - 1)] > gpos[m])
        v = np.full(m.sum(), '', dtype=object)
        v[ok] = n[idx[ok]]
        res[m] = v
    return res


def std_rate(call, kmer5, glob_rate):
    """间接标准化: 观测率 / 由 5-mer 组成算出的期望率"""
    obs = call.mean()
    exp = np.mean([glob_rate.get(k, np.nan) for k in kmer5])
    return obs, exp, (obs / exp if exp and not np.isnan(exp) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bases', type=int, default=20)
    ap.add_argument('--match-fpr', type=float, default=1e-4)
    ap.add_argument('--exclude-cpg', action='store_true',
                    help='剔除中心 5-mer 含 CG 的位点。用来验证 GC 富集不是 CpG(5mC)造成的 —— '
                         'CpG 判正率高且在高 GC 区占比大, 5-mer 标准化会把高 GC 区的期望值抬高、'
                         '掩盖残余富集; 剔掉后才看得到真实的 GC 效应。')
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/genome_dist.txt')
    args = ap.parse_args()

    bases = sorted(os.path.basename(p).replace('.tsv.gz', '')
                   for p in glob.glob(os.path.join(WT, 'scores', 'oxonet_e2ep125', '*.tsv.gz')))
    bases = bases[:args.bases]
    print('用 {} 个 f5'.format(len(bases)), flush=True)

    # 位点表: uid,pos,5mer + 每个模型的 call
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
    print('位点 {:,}'.format(len(sites)), flush=True)

    for m, (pk, nd, col) in MODELS.items():
        thr = thr_at_fpr(pk, args.match_fpr)
        v = []
        for b in bases:
            f = os.path.join(nd, b + '.tsv.gz')
            v.append(pd.read_csv(f, sep='\t', usecols=[col])[col].values if os.path.isfile(f)
                     else np.array([]))
        v = np.concatenate(v)
        assert len(v) == len(sites), '{}: 行数不匹配 {} vs {}'.format(m, len(v), len(sites))
        sites[m] = v >= thr
        print('{}: thr={:.4f} 判正 {:,}'.format(m, thr, int(sites[m].sum())), flush=True)

    # CpG 剔除必须放在**打分之后**: 上面的 v 是按原始行序拼的, 先过滤会错位
    if args.exclude_cpg:
        n0 = len(sites)
        sites = sites[~sites['k5'].str.contains('CG')].reset_index(drop=True)
        print('剔除 CpG: {:,} -> {:,} ({:.1%} 被剔除)'.format(
            n0, len(sites), 1 - len(sites) / n0), flush=True)

    # 坐标
    want = set(sites['uid'].unique().tolist())
    cs = []
    for ch in pd.read_csv(COORDS, sep='\t', usecols=['read_id', 'basecall_pos', 'chrom', 'gpos'],
                          chunksize=5_000_000):
        u = uid64(ch['read_id'].values)
        keep = np.fromiter((x in want for x in u), bool, len(u))
        if keep.any():
            cs.append(pd.DataFrame({'uid': u[keep], 'pos': ch['basecall_pos'].values[keep].astype(np.int64),
                                    'chrom': ch['chrom'].values[keep],
                                    'gpos': ch['gpos'].values[keep].astype(np.int64)}))
    co = pd.concat(cs, ignore_index=True).drop_duplicates(['uid', 'pos'])
    d = sites.merge(co, on=['uid', 'pos'], how='inner')
    print('有坐标的位点 {:,} ({:.1%})'.format(len(d), len(d) / len(sites)), flush=True)

    print('算 1kb bin 的 GC ...', flush=True)
    gcmap = gc_by_bin()
    gc = np.full(len(d), np.nan, np.float32)
    ch_arr = d['chrom'].values; gp = d['gpos'].values
    for c in np.unique(ch_arr):
        if c not in gcmap:
            continue
        m = ch_arr == c
        b = (gp[m] - 1) // BIN
        arr = gcmap[c]
        ok = b < len(arr)
        v = np.full(m.sum(), np.nan, np.float32)
        v[ok] = arr[b[ok]]
        gc[m] = v
    d['gc'] = gc

    print('注释重复区/卫星 ...', flush=True)
    rm = load_bed(os.path.join(ANNOT, 'chm13v2.0_RepeatMasker_4.1.2p1.2022Apr14.bed'), name_col=6)
    ct = load_bed(os.path.join(ANNOT, 'chm13v2.0_censat_v2.1.bed'), name_col=3)
    d['repeat'] = annotate(ch_arr, gp, rm)
    d['censat'] = annotate(ch_arr, gp, ct)

    L = ['基因组分布（{} 个 f5, {:,} 个有坐标的 G; 等绝对基因组 FPR = {:.0e}）'.format(
        len(bases), len(d), args.match_fpr),
        '归一化：判正率已按 coverage 归一；"标准化比"= 观测率 / 由 5-mer 组成算出的期望率（照 esox 口径）']

    for m in MODELS:
        gr = d.groupby('k5')[m].mean().to_dict()          # 每个 5-mer 的全局判正率
        L += ['', '=== {} ==='.format(m)]
        L.append('-- GC 含量（1kb bin）--')
        L.append('{:>12s} {:>10s} {:>11s} {:>11s} {:>9s}'.format(
            'GC bin', '位点数', '判正率', '期望(5mer)', '标准化比'))
        qs = pd.cut(d['gc'], [0, .35, .40, .45, .50, .55, .60, 1.0])
        for k, g in d.groupby(qs, observed=True):
            o, e, r = std_rate(g[m].values, g['k5'].values, gr)
            L.append('{:>12s} {:>10,} {:10.4%} {:10.4%} {:8.2f}x'.format(str(k), len(g), o, e, r))
        L.append('-- 重复序列类别（RepeatMasker）--')
        L.append('{:>12s} {:>10s} {:>11s} {:>11s} {:>9s}'.format(
            'class', '位点数', '判正率', '期望(5mer)', '标准化比'))
        rep = d['repeat'].replace('', '非重复')
        for k, g in d.groupby(rep, observed=True):
            if len(g) < 5000:
                continue
            o, e, r = std_rate(g[m].values, g['k5'].values, gr)
            L.append('{:>12s} {:>10,} {:10.4%} {:10.4%} {:8.2f}x'.format(str(k)[:12], len(g), o, e, r))
        L.append('-- 着丝粒/卫星（censat）--')
        cs2 = d['censat'].str.replace(r'[_(].*', '', regex=True).replace('', '非着丝粒')
        for k, g in d.groupby(cs2, observed=True):
            if len(g) < 5000:
                continue
            o, e, r = std_rate(g[m].values, g['k5'].values, gr)
            L.append('{:>12s} {:>10,} {:10.4%} {:10.4%} {:8.2f}x'.format(str(k)[:12], len(g), o, e, r))

    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    d[['chrom', 'gpos', 'gc', 'repeat', 'censat', 'k5'] + list(MODELS)].to_parquet(
        os.path.join(WT, 'genome_dist_sites.parquet'))
    print('写出', args.out)


if __name__ == '__main__':
    main()
