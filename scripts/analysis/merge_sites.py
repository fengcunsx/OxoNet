"""Stage D — merge per-base esox_score + oxonet_prob into a unified site table.

Joins on (read_id, basecall_pos). Genome coords (Stage B all.tsv.gz) are NOT joined
here: Stage F score-Spearman / overlap is coordinate-free, so we keep this light and
add coords only for Stage G (GC-rho). Only the bases that have an OxoNet score are
included (OxoNet was run on a 40-base subset; esox on all 150).

    /home/bio/anaconda3/bin/python oxo/analysis/merge_sites.py
Output: /home/bio/8oxog/wtl1/sites_scores.parquet
    columns: read_id, basecall_pos, 5mer, oxog_score, oxonet_prob
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--esox-dir', default='/home/bio/8oxog/wtl1/scores/esox')
    ap.add_argument('--oxonet-dir', default='/home/bio/8oxog/wtl1/scores/oxonet')
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/sites_scores.parquet')
    args = ap.parse_args()

    # drive off OxoNet subset (fewer bases); require matching esox tsv
    oxo_bases = sorted(os.path.basename(p)[:-7]
                       for p in glob.glob(os.path.join(args.oxonet_dir, '*.tsv.gz')))
    print('OxoNet bases: {}'.format(len(oxo_bases)), flush=True)

    frames = []
    tot = matched = 0
    for i, base in enumerate(oxo_bases):
        ep = os.path.join(args.esox_dir, base + '.tsv.gz')
        op = os.path.join(args.oxonet_dir, base + '.tsv.gz')
        if not os.path.isfile(ep):
            print('  !! no esox for', base); continue
        e = pd.read_csv(ep, sep='\t', dtype={'read_id': str, 'basecall_pos': np.int32,
                                             'oxog_score': np.float32, '5mer': str})
        o = pd.read_csv(op, sep='\t', dtype={'read_id': str, 'basecall_pos': np.int32,
                                             'prob': np.float32})
        m = e.merge(o, on=['read_id', 'basecall_pos'], how='inner')
        m = m.rename(columns={'prob': 'oxonet_prob'})
        frames.append(m)
        tot += len(e); matched += len(m)
        if (i + 1) % 5 == 0 or i + 1 == len(oxo_bases):
            print('  {}/{} base={} esox_rows={} matched={}'.format(
                i + 1, len(oxo_bases), base, len(e), len(m)), flush=True)

    df = pd.concat(frames, ignore_index=True)
    print('TOTAL sites: {}  (esox_rows={}, match_rate={:.4f})'.format(
        len(df), tot, matched / tot if tot else 0), flush=True)
    df.to_parquet(args.out, index=False)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
