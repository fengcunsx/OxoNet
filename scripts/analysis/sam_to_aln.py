"""Streaming SAM filter: stdin minimap2 SAM -> compact primary-alignment table.

Kept intentionally trivial so it drains stdin immediately (no big preload), which
prevents minimap2 from blocking on a full pipe / being OOM-killed while a heavy
reader preloads. Downstream `aln_to_coords.py` does the per-base CIGAR mapping offline.

Output tsv (stdout): read_id  flag  chrom  pos(1-based leftmost)  cigar
Only PRIMARY, mapped records (drop unmapped 0x4 / secondary 0x100 / supplementary 0x800).
"""
import sys

def main():
    out = sys.stdout
    out.write('read_id\tflag\tchrom\tpos\tcigar\n')
    n = 0
    for line in sys.stdin:
        if line[0] == '@':
            continue
        f = line.split('\t', 6)
        if len(f) < 6:
            continue
        flag = int(f[1])
        if flag & 0x4 or flag & 0x100 or flag & 0x800:
            continue
        out.write('{}\t{}\t{}\t{}\t{}\n'.format(f[0], flag, f[2], f[3], f[5]))
        n += 1
    sys.stderr.write('wrote {} primary alignments\n'.format(n))


if __name__ == '__main__':
    main()
