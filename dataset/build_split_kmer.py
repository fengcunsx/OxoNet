"""P3 for the wide-kmer study: organise the k-mer re-extraction into the SAME
train/valid/test split the 7-mer pack uses, then bucket by center-5mer.

This is a NEW script. It does NOT modify `build_split.py`, which stays the oracle
for the 7-mer pack the paper reports.

Two things differ from `build_split.py`, both required for k != 7:

1. `c5()` took `s[1:6]` -- the center 5-mer of a 7-mer. At k=13 the center is at
   index 6, so the slice must follow k. Left as-is, every row would be bucketed
   under the wrong context.

2. The split is INHERITED, never re-derived. `build_split.py` recomputes the
   assignment by globbing directories and shuffling with a seed; re-running that
   against the k-mer output could hand a read a different split, which would make
   the two widths incomparable. Here we read `split_manifest.json` and, more
   importantly, resolve it down to READ IDS:

     - oligo: split is per fast5 sub-dir, and a read never leaves its fast5, so
       the sub-dir basename -> split mapping is already read-exact.
     - t2t: the manifest is keyed on npz SHARD paths, and shard boundaries are
       "every 250 successfully processed reads". A handful of reads yield rows at
       one width but not the other (7 lost / 8 gained across 39 runs), which can
       shift every later read in that worker by one slot and silently move a read
       across a shard -- and neighbouring shards can be in DIFFERENT splits. So
       we map shard -> split -> its read ids from the ARCHIVE, and then assign
       k-mer rows by read id.

Per the user's call, the read set is pinned to the 7-mer one: rows whose read is
not in the 7-mer archive are DROPPED (~0.03% of t2t reads), so both widths are
trained and evaluated on exactly the same reads.

Output layout is identical to build_split.py:
    8oxog_<split>/<ctx>.npz     pos
    g_oligo_<split>/<ctx>.npz   oligo neg
    g_t2t_<split>/<ctx>.npz     T2T neg
    g_<split>/<ctx>.npz         oligo+T2T neg merged and shuffled (E2)

Usage:
  python dataset/build_split_kmer.py --kmer 13 --out /data/k13/split --ram-gb 8
  python dataset/build_split_kmer.py --kmer 13 --out /tmp/smoke --smoke 3
"""
import argparse
import glob
import json
import os
import shutil
import tempfile
from collections import defaultdict

import numpy as np

MANIFEST = '/home/bio/8oxog/build/split_manifest.json'
OLIGO_K = '/data/k13/oligo'          # batch<N>/fast5s/<name>/feature/oxo/{8oxog,g}/data.npz
T2T_K = '/data/k13/t2t'              # <name>/oxo/*.npz
T2T_ARCHIVE = '/home/bio/8oxog/data/t2t_all/t2t'   # where the 7-mer shards live
# the manifest's t2t keys are relative to its own t2t_root and keep this shape
T2T_ARC_GLOB = {
    '288548394': '288548394/*/output_new/oxo/*.npz',
    '3439856925': '3439856925/*/output_new/oxo/*.npz',
    'new': 'new/output/*/output/oxo/*.npz',
}
KEYS = ['signal', 'label', 'read_id', 'kmers', 'mean', 'std', 'dwell', 'basecall_pos']
BYTES_PER_ROW_7 = 1700               # build_split.py's estimate at k=7


def c5(kmers, k):
    """Center 5-mer of a k-mer. k=7 -> s[1:6] (matches build_split.py exactly)."""
    lo = k // 2 - 2
    return np.array([s[lo:lo + 5] for s in kmers])


def t2t_run_name(shard_relpath):
    """`288548394/<run>/output_new/oxo/<run>.wN.partNNNN.npz` -> `<run>`
    `new/output/<run>/output/oxo/<run>.wN.partNNNN.npz`       -> `<run>`"""
    return os.path.basename(shard_relpath).split('.w')[0]


def build_read_split(manifest, smoke=0):
    """read_id -> split, resolved from the 7-mer archive.

    Returns (oligo_name2split, t2t_read2split, t2t_allowed_by_run, oligo_allowed_by_name).
    The `allowed` sets pin the read set to the 7-mer extraction.
    """
    m = json.load(open(manifest))

    # ---- oligo: sub-dir basename -> split (already read-exact) ----------------
    oligo_name2split = {os.path.basename(k): v for k, v in m['oligo'].items()}

    # ---- t2t: shard -> split, then shard's ARCHIVE read ids -> split ----------
    t2t_read2split = {}
    shards = sorted(m['t2t'].items())
    if smoke:
        shards = shards[:smoke * 5]
    for rel, split in shards:
        p = os.path.join(T2T_ARCHIVE, rel)
        if not os.path.isfile(p):
            continue
        for rid in np.load(p, allow_pickle=True)['read_id'].tolist():
            t2t_read2split[rid] = split
    return oligo_name2split, t2t_read2split


def oligo_allowed_reads(subdir):
    """Read ids the 7-mer extraction kept for this fast5 (feature_k7/), or None
    if the archive is missing (then nothing is dropped for this sub-dir)."""
    s = set()
    found = False
    for b in ('8oxog', 'g'):
        p = os.path.join(subdir, 'feature_k7', 'oxo', b, 'data.npz')
        if os.path.isfile(p):
            found = True
            s |= set(np.load(p, allow_pickle=True)['read_id'].tolist())
    return s if found else None


_ALLOWED_CACHE = {}


def allowed_arr(allowed):
    """Sorted ndarray view of a read-id set, memoised: np.isin rebuilds+sorts its
    second argument on every call, and each set is reused across many shards."""
    key = id(allowed)
    hit = _ALLOWED_CACHE.get(key)
    if hit is None or hit[0] is not allowed:
        hit = (allowed, np.array(sorted(allowed), dtype='<U36'))
        _ALLOWED_CACHE[key] = hit
    return hit[1]


def row_data(a, is_pos, keep_mask):
    """Standard fields for the kept rows (fills label when the npz has none)."""
    out = {}
    for k in ('signal', 'mean', 'std'):
        v = a[k][keep_mask]
        out[k] = v.astype(np.float32) if v.dtype == np.float64 else v
    out['dwell'] = a['dwell'][keep_mask]
    out['kmers'] = a['kmers'][keep_mask]
    out['read_id'] = a['read_id'][keep_mask]
    out['basecall_pos'] = a['basecall_pos'].reshape(-1, 1)[keep_mask]
    if 'label' in a.files:
        out['label'] = a['label'][keep_mask]
    else:
        out['label'] = np.full(int(keep_mask.sum()), 1 if is_pos else 0, dtype=np.int64)
    return out


def stream_build(tasks, is_pos, out_root, out_prefix, ram_gb, k, bytes_per_row, tmp_root):
    """tasks: {split: [(npz_path, allowed_read_ids, universe_or_None), ...]}
    Buffers per center-5mer, flushes shards when RAM cap is hit, then merges.

    `tmp_root` defaults to --out, NOT to $TMPDIR: the intermediate shards are as
    large as the final output (tens of GB) and /tmp is on the small root
    partition here. Letting tempfile pick filled / and killed the run.
    """
    ram_bytes = ram_gb * 1e9
    for split, items in tasks.items():
        if not items:
            continue
        tmp = tempfile.mkdtemp(prefix='bsk_{}_{}_'.format(out_prefix, split), dir=tmp_root)
        buffers = defaultdict(lambda: defaultdict(list))
        used = [0]
        shard = [0]
        dropped = [0]
        other = [0]
        kept = [0]

        def flush():
            for ctx, d in buffers.items():
                if not d['label']:
                    continue
                cd = os.path.join(tmp, ctx)
                os.makedirs(cd, exist_ok=True)
                merged = {kk: np.concatenate(d[kk], axis=0) for kk in KEYS}
                np.savez_compressed(os.path.join(cd, '{}.npz'.format(shard[0])), **merged)
            buffers.clear()
            used[0] = 0
            shard[0] += 1

        for p, allowed, universe in items:
            a = np.load(p, allow_pickle=True)
            if a['kmers'].shape[0] == 0:
                continue
            rid = a['read_id']
            if allowed is None:
                mask = np.ones(rid.shape[0], dtype=bool)
            else:
                # np.isin, not a Python `in` per row: t2t shards run to ~10^5 rows
                # and every shard is visited once per split.
                mask = np.isin(rid, allowed_arr(allowed))
            kept[0] += int(mask.sum())
            if universe is None:
                dropped[0] += int((~mask).sum())
            else:
                # separate "belongs to another split" from "this read does not
                # exist in the 7-mer extraction at all" -- only the latter is loss
                outside = ~np.isin(rid, allowed_arr(universe))
                dropped[0] += int(outside.sum())
                other[0] += int(((~mask) & ~outside).sum())
            if not mask.any():
                continue
            data = row_data(a, is_pos, mask)
            ctxs = c5(data['kmers'], k)
            for ctx in np.unique(ctxs):
                cm = ctxs == ctx
                for kk in KEYS:
                    buffers[str(ctx)][kk].append(data[kk][cm])
                used[0] += int(cm.sum()) * bytes_per_row
            if used[0] > ram_bytes:
                flush()
        flush()

        outdir = os.path.join(out_root, '{}_{}'.format(out_prefix, split))
        os.makedirs(outdir, exist_ok=True)
        for ctx in sorted(os.listdir(tmp)):
            parts = sorted(glob.glob(os.path.join(tmp, ctx, '*.npz')))
            merged = {kk: np.concatenate([np.load(s)[kk] for s in parts], axis=0) for kk in KEYS}
            np.savez_compressed(os.path.join(outdir, '{}.npz'.format(ctx)), **merged)
        shutil.rmtree(tmp)
        print('  {}_{}: {} ctx | kept {} rows | other-split {} | DROPPED {} '
              '(read absent from the 7-mer extraction)'
              .format(out_prefix, split, len(os.listdir(outdir)), kept[0],
                      other[0], dropped[0]))


def merge_shuffle_neg(oligo_dir, t2t_dir, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    ctxs = set(os.path.basename(f)[:-4] for f in glob.glob(oligo_dir + '/*.npz'))
    ctxs |= set(os.path.basename(f)[:-4] for f in glob.glob(t2t_dir + '/*.npz'))
    for ctx in sorted(ctxs):
        parts = [np.load(os.path.join(d, ctx + '.npz')) for d in (oligo_dir, t2t_dir)
                 if os.path.isfile(os.path.join(d, ctx + '.npz'))]
        merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in KEYS}
        idx = np.arange(merged['label'].shape[0])
        np.random.RandomState(seed).shuffle(idx)
        np.savez_compressed(os.path.join(out_dir, ctx + '.npz'),
                            **{k: v[idx] for k, v in merged.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output root (put it on /data, not /)')
    ap.add_argument('--kmer', type=int, default=13)
    ap.add_argument('--seed', type=int, default=42, help='only used to shuffle the merged negatives')
    ap.add_argument('--ram-gb', type=float, default=8.0)
    ap.add_argument('--smoke', type=int, default=0, help='>0: only the first N inputs per split')
    ap.add_argument('--tmp-root', default=None,
                    help='where the intermediate per-ctx shards go; defaults to --out. '
                         'Never let this land on a small partition -- the temporaries '
                         'are as big as the final output.')
    ap.add_argument('--manifest', default=MANIFEST)
    ap.add_argument('--oligo-root', default=OLIGO_K)
    ap.add_argument('--t2t-root', default=T2T_K)
    args = ap.parse_args()

    k = args.kmer
    # signal width scales with k, so the RAM accounting must too
    bytes_per_row = int(BYTES_PER_ROW_7 * (k * 25) / 175.0)
    os.makedirs(args.out, exist_ok=True)
    tmp_root = args.tmp_root or args.out
    os.makedirs(tmp_root, exist_ok=True)
    st = os.statvfs(tmp_root)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    print('tmp_root={} (free {:.0f} GB)'.format(tmp_root, free_gb))
    if free_gb < 60:
        raise SystemExit('tmp_root only has {:.0f} GB free; the intermediates need '
                         '~50-70 GB at k={}. Point --tmp-root at a bigger disk.'
                         .format(free_gb, k))

    print('resolving split -> read ids from the 7-mer archive ...')
    oligo_name2split, t2t_read2split = build_read_split(args.manifest, args.smoke)
    print('  oligo sub-dirs {} | t2t reads {}'.format(len(oligo_name2split), len(t2t_read2split)))

    # ---- oligo tasks: one (pos npz, neg npz) per sub-dir ----------------------
    pos_tasks = {'train': [], 'valid': [], 'test': []}
    oneg_tasks = {'train': [], 'valid': [], 'test': []}
    n_unmapped = 0
    for d in sorted(glob.glob(os.path.join(args.oligo_root, 'batch*/fast5s/*/'))):
        name = os.path.basename(d.rstrip('/'))
        split = oligo_name2split.get(name)
        if split is None:
            n_unmapped += 1
            continue
        allowed = oligo_allowed_reads(d)
        for sub, tasks in (('8oxog', pos_tasks), ('g', oneg_tasks)):
            p = os.path.join(d, 'feature', 'oxo', sub, 'data.npz')
            if os.path.isfile(p):
                tasks[split].append((p, allowed, allowed))
    if n_unmapped:
        print('  WARNING: {} oligo sub-dirs absent from the manifest, skipped'.format(n_unmapped))

    # ---- t2t tasks: shards, filtered by inherited read ids --------------------
    tneg_tasks = {'train': [], 'valid': [], 'test': []}
    by_split_reads = {'train': set(), 'valid': set(), 'test': set()}
    for rid, sp in t2t_read2split.items():
        by_split_reads[sp].add(rid)
    all_t2t_reads = set(t2t_read2split)
    for p in sorted(glob.glob(os.path.join(args.t2t_root, '*/oxo/*.npz'))):
        # a shard can hold reads of only one split? NO -- assign per split, and
        # each pass keeps just that split's reads. Cheap: 3 passes over metadata.
        for sp in ('train', 'valid', 'test'):
            tneg_tasks[sp].append((p, by_split_reads[sp], all_t2t_reads))

    if args.smoke:
        for t in (pos_tasks, oneg_tasks, tneg_tasks):
            for sp in t:
                t[sp] = t[sp][:args.smoke]

    print('oligo pos ...')
    stream_build(pos_tasks, True, args.out, '8oxog', args.ram_gb, k, bytes_per_row, tmp_root)
    print('oligo neg ...')
    stream_build(oneg_tasks, False, args.out, 'g_oligo', args.ram_gb, k, bytes_per_row, tmp_root)
    print('T2T neg ...')
    stream_build(tneg_tasks, False, args.out, 'g_t2t', args.ram_gb, k, bytes_per_row, tmp_root)

    print('merge+shuffle neg (E2) ...')
    for sp in ('train', 'valid', 'test'):
        merge_shuffle_neg(os.path.join(args.out, 'g_oligo_' + sp),
                          os.path.join(args.out, 'g_t2t_' + sp),
                          os.path.join(args.out, 'g_' + sp), args.seed)

    with open(os.path.join(args.out, 'BUILD_INFO.json'), 'w') as f:
        json.dump({'kmer': k, 'manifest': args.manifest, 'seed': args.seed,
                   'oligo_root': args.oligo_root, 't2t_root': args.t2t_root,
                   'read_set': 'pinned to the 7-mer archive (extras dropped)'}, f, indent=1)
    print('done ->', args.out)


if __name__ == '__main__':
    main()
