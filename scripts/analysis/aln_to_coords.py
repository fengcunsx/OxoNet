"""Offline per-base mapping: join compact alignments (from sam_to_aln.py) with the
candidate G positions (esox npz basecall_pos) and CIGAR-map each to genome coords.

Low memory: alignments dict (~read count) held once; positions loaded one base at a
time. No pipe, no minimap here.

    python aln_to_coords.py --aln wt1/aln.tsv --out-root wt1/out --out wt1/coords/all.tsv

Output tsv: read_id  basecall_pos  chrom  gpos(1-based)  strand
basecall_pos indexes the BONITO read; reverse-strand handled via qlen-1-p.
Positions in insertions/soft-clips (no reference base) are dropped.
"""
import os
import sys
import argparse
import glob
import bisect
import gzip

import numpy as np

_CONSUME = {
    'M': (True, True), '=': (True, True), 'X': (True, True),
    'I': (True, False), 'S': (True, False),
    'D': (False, True), 'N': (False, True),
    'H': (False, False), 'P': (False, False),
}


def parse_cigar(cig):
    out, num = [], ''
    for c in cig:
        if c.isdigit():
            num += c
        else:
            out.append((int(num), c)); num = ''
    return out


def query_len(cigar):
    return sum(l for l, op in cigar if _CONSUME[op][0])


def load_aln(path):
    """read_id -> (flag, chrom, pos1based, cigar_str)."""
    aln = {}
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            rid, flag, chrom, pos, cigar = line.rstrip('\n').split('\t')
            aln[rid] = (int(flag), chrom, int(pos), cigar)
    return aln


def base_positions(out_root, base):
    """read_id -> sorted unique np.array of basecall_pos for one base."""
    from collections import defaultdict
    pos = defaultdict(list)
    for npz in glob.glob(os.path.join(out_root, base, 'esox', base + '.w*.part*.npz')):
        a = np.load(npz)
        for rid, p in zip(a['read_id'], a['basecall_pos']):
            pos[str(rid)].append(int(p))
    return {rid: np.array(sorted(set(ps))) for rid, ps in pos.items()}


def map_read(ps, flag, ref_start0, cigar, out, rid, chrom):
    reverse = bool(flag & 0x10)
    qlen = query_len(cigar)
    qs = (qlen - 1 - ps) if reverse else ps
    order = np.argsort(qs)
    qs_sorted = qs[order]; p_sorted = ps[order]
    strand = '-' if reverse else '+'
    qpos = 0; rpos = ref_start0
    n = 0
    for l, op in cigar:
        cq, cr = _CONSUME[op]
        if cq and cr:
            lo = bisect.bisect_left(qs_sorted, qpos)
            hi = bisect.bisect_left(qs_sorted, qpos + l)
            for k in range(lo, hi):
                g = rpos + (int(qs_sorted[k]) - qpos) + 1
                out.write('{}\t{}\t{}\t{}\t{}\n'.format(rid, int(p_sorted[k]), chrom, g, strand))
                n += 1
        if cq:
            qpos += l
        if cr:
            rpos += l
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aln', required=True, help='alignments tsv from sam_to_aln.py')
    ap.add_argument('--out-root', default='/home/bio/8oxog/wtl1/out')
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/coords/all.tsv.gz',
                    help='gzip if endswith .gz')
    args = ap.parse_args()

    aln = load_aln(args.aln)
    sys.stderr.write('alignments: {} reads\n'.format(len(aln)))

    bases = sorted(d for d in os.listdir(args.out_root)
                   if os.path.isdir(os.path.join(args.out_root, d)) and d != '_tmp')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tot_sites = tot_reads = 0
    opener = (lambda p: gzip.open(p, 'wt')) if args.out.endswith('.gz') else (lambda p: open(p, 'w'))
    with opener(args.out) as out:
        out.write('read_id\tbasecall_pos\tchrom\tgpos\tstrand\n')
        for base in bases:
            want = base_positions(args.out_root, base)
            for rid, ps in want.items():
                a = aln.get(rid)
                if a is None or len(ps) == 0:
                    continue
                flag, chrom, pos1, cigar = a
                tot_sites += map_read(ps, flag, pos1 - 1, parse_cigar(cigar), out, rid, chrom)
                tot_reads += 1
            sys.stderr.write('  {}: cumulative reads={} sites={}\n'.format(base, tot_reads, tot_sites))
    sys.stderr.write('done: {} reads, {} sites -> {}\n'.format(tot_reads, tot_sites, args.out))


if __name__ == '__main__':
    main()
