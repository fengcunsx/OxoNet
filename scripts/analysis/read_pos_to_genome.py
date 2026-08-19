"""Map per-read candidate G positions (basecall_pos, an index into the BONITO read
sequence) to genome coordinates on CHM13v2.0, by streaming minimap2 SAM from stdin.

No BAM, no .mmi kept. Usage (one base or many; read_ids are globally unique):

    cat <base>/<base>.bonito.fastq | minimap2 -ax map-ont chm13v2.0.fa.gz - \
        | python read_pos_to_genome.py --out-root /home/bio/8oxog/wtl1/out --base <base> \
        > coords_<base>.tsv

Positions to map are loaded from the esox npz shards (read_id, basecall_pos only).
Output tsv columns: read_id  basecall_pos  chrom  gpos(1-based)  strand
Only PRIMARY, mapped alignments are used; positions in insertions/soft-clips (no
reference base) are dropped.
"""
import os
import sys
import argparse
import glob
import bisect
from collections import defaultdict

import numpy as np

# CIGAR op consumes: (query, ref)
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


def load_positions(out_root, bases):
    """read_id -> sorted np.array of basecall_pos (across the given bases' esox shards)."""
    pos = defaultdict(list)
    for base in bases:
        for npz in glob.glob(os.path.join(out_root, base, 'esox', base + '.w*.part*.npz')):
            a = np.load(npz)
            for rid, p in zip(a['read_id'], a['basecall_pos']):
                pos[str(rid)].append(int(p))
    return {rid: np.array(sorted(set(ps))) for rid, ps in pos.items()}


def query_len(cigar):
    return sum(l for l, op in cigar if _CONSUME[op][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', default='/home/bio/8oxog/wtl1/out')
    ap.add_argument('--base', default=None, help='single base (smoke)')
    ap.add_argument('--bases-file', default=None, help='file with one base per line (full run)')
    args = ap.parse_args()

    if args.base:
        bases = [args.base]
    elif args.bases_file:
        bases = [l.strip() for l in open(args.bases_file) if l.strip()]
    else:
        bases = sorted(d for d in os.listdir(args.out_root)
                       if os.path.isdir(os.path.join(args.out_root, d)) and d != '_tmp')

    want = load_positions(args.out_root, bases)   # read_id -> sorted positions (original read coords)
    sys.stderr.write('loaded positions for {} reads\n'.format(len(want)))

    out = sys.stdout
    out.write('read_id\tbasecall_pos\tchrom\tgpos\tstrand\tmapq\n')
    n_reads = n_sites = 0

    for line in sys.stdin:
        if line[0] == '@':
            continue
        f = line.rstrip('\n').split('\t')
        if len(f) < 6:
            continue
        rid = f[0]
        flag = int(f[1])
        if flag & 0x4 or flag & 0x100 or flag & 0x800:   # unmapped / secondary / supplementary
            continue
        ps = want.get(rid)
        if ps is None or len(ps) == 0:
            continue
        chrom = f[2]
        ref_start = int(f[3]) - 1        # 0-based
        mapq = int(f[4])   # 比对质量: 答 M1 的 'mapping ambiguity'; 0 = 多处等好比对
        cigar = parse_cigar(f[5])
        reverse = bool(flag & 0x10)
        qlen = query_len(cigar)

        # original read position p -> SAM query position q (SAM SEQ is revcomp when reverse)
        # map q -> ref via CIGAR M/=/X blocks; use bisect over needed q's.
        qs = (qlen - 1 - ps) if reverse else ps          # np.array
        order = np.argsort(qs)
        qs_sorted = qs[order]
        p_sorted = ps[order]
        strand = '-' if reverse else '+'

        qpos = 0
        rpos = ref_start
        n_reads += 1
        for l, op in cigar:
            cq, cr = _CONSUME[op]
            if cq and cr:  # aligned block: query [qpos, qpos+l) -> ref [rpos, rpos+l)
                lo = bisect.bisect_left(qs_sorted, qpos)
                hi = bisect.bisect_left(qs_sorted, qpos + l)
                for k in range(lo, hi):
                    q = qs_sorted[k]
                    g = rpos + (q - qpos) + 1            # 1-based genome
                    out.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        rid, int(p_sorted[k]), chrom, g, strand, mapq))
                    n_sites += 1
            if cq:
                qpos += l
            if cr:
                rpos += l

    sys.stderr.write('mapped {} reads, {} sites\n'.format(n_reads, n_sites))


if __name__ == '__main__':
    main()
