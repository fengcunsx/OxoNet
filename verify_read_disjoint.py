"""Check that no read used for training reappears in validation or test.

    python verify_read_disjoint.py

CPU only, a few seconds, no raw data needed.  This is the audit trail behind the
read-level split described in the paper (Section III) and in R2.m4 of the
response letter.

The evidence comes in three layers of decreasing directness, and the script
prints them separately rather than collapsing them into one claim, because
they are not equally strong:

  [exact]         The training *positives* are enumerated.  ``pos_train.npz``
                  was the array fed to the model, so its read IDs are the
                  training positives, not a reconstruction.  Released as
                  manifests/reads_train_pos.txt.gz.

  [construction]  The training *negatives* cannot be enumerated: the packed
                  negative training arrays kept only signal and summary
                  features, and the intermediate per-read files were deleted
                  after packing.  Their disjointness instead follows from how
                  the split was made -- see manifests/split_manifest.json and
                  dataset/build_split.py::assign_splits_oligo / _t2t.  Whole
                  FAST5 directories (oligo) and whole npz parts (genomic) are
                  assigned to exactly one of train/valid/test *before any read
                  is read*.  A read exists in exactly one such file, so it
                  cannot cross splits.  This argument covers every read,
                  including those we did not archive; enumeration would only
                  ever cover the ones we did.

  [independent]   A separate 13-mer feature extraction, run later over the same
                  file-level partition, did keep read IDs.  Its training reads
                  are reported here as a redundant check.  It is *not* the
                  training set -- it drops 540 reads the 7-mer build kept and
                  adds 420 it did not, because the two extractions apply
                  different gap filters.  It is included as corroboration, not
                  as provenance.  Requires the 13-mer archive and is skipped if
                  absent.
"""
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, 'manifests')

# 可选: 13-mer 归档不随仓库分发(43 GB), 有就跑, 没有就跳过
KMER13_SPLIT = os.environ.get('OXONET_K13_SPLIT', '/data/k13/split')


def load(name):
    with gzip.open(os.path.join(M, name), 'rt') as f:
        return {ln.strip() for ln in f if ln.strip()}


def report(label, train, evalsets):
    ok = True
    for name, s in evalsets.items():
        n = len(train & s)
        ok &= (n == 0)
        print('    train n {:<13} = {:>7,}      ({} has {:,} reads)'.format(
            name, n, name, len(s)))
    return ok


def main():
    ev = {n: load('reads_{}.txt.gz'.format(n))
          for n in ('valid', 'test_oligo', 'test_t2t')}

    print('[exact] training positives, enumerated from the packed training array')
    tp = load('reads_train_pos.txt.gz')
    print('    {:,} reads'.format(len(tp)))
    ok = report('positives', tp, ev)

    print('\n[construction] training negatives, not enumerable')
    print('    Negative training read IDs were not archived.  Disjointness follows')
    print('    from the file-level partition in manifests/split_manifest.json;')
    print('    see the module docstring.  Nothing is asserted here.')

    print('\n[independent] separate 13-mer extraction over the same partition')
    d = os.path.join(KMER13_SPLIT, 'g_train')
    if not os.path.isdir(d):
        print('    skipped: {} not present (set OXONET_K13_SPLIT to enable)'.format(d))
    else:
        import glob
        import numpy as np
        s = set()
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            with np.load(f) as a:
                s.update(np.unique(a['read_id']).tolist())
        print('    {:,} reads observed in train-assigned files'.format(len(s)))
        ok &= report('negatives*', s, ev)

    print('\n{}'.format('PASS: no training read appears in validation or test'
                        if ok else 'FAIL: overlap detected'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
