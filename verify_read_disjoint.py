"""Check that no read used for training reappears in validation or test.

    python verify_read_disjoint.py

CPU only, a few seconds, no raw data needed.  This is the audit trail behind the
read-level split described in the paper (Section III) and in R2.m4 of the
response letter.

The evidence comes in four layers of decreasing directness, and the script
prints them separately rather than collapsing them into one claim, because
they are not equally strong.  In particular, the final line is deliberately not
a single global PASS: no check here establishes disjointness for the training
negatives, and the output says so.

  [manifest]      The released split assignment is well formed: seed 42, every
                  entry labelled train/valid/test, and the group counts the
                  paper reports (oligo 846 = 762/42/42, genomic 315 =
                  283/16/16).  This validates the manifest.  It does not by
                  itself show that a read never spans two groups; that is a
                  property of the preprocessing, stated below.

  [exhaustive]    Every read basecalled in every one of the 846 oligonucleotide
                  groups, taken from the per-group basecall FASTQ rather than
                  from the packed arrays: 2,683,085 train, 146,932 validation,
                  148,225 test.  This covers reads whether or not they entered
                  training, so it does not depend on which fields the packing
                  step happened to retain.  All three sets are pairwise
                  disjoint.  Released as manifests/oligo_all_reads_*.txt.gz.

  [exact]         The training *positives* as fed to the model
                  (manifests/reads_train_pos.txt.gz).  A subset of the above,
                  checked separately because it is the set the model actually
                  saw.

  [construction]  Genomic training negatives are the one set not enumerable
                  here: the packed negative arrays kept only signal and summary
                  features, and the source trees were deleted after packing.
                  Their disjointness follows from the assignment itself -- see
                  manifests/split_manifest.json and dataset/build_split.py.
                  Whole FASTQ groups and whole npz parts go to exactly one
                  subset *before any read is opened*, and a read belongs to
                  exactly one group, so it cannot cross subsets.

  [independent]   A later 13-mer extraction over the same partition retained
                  read IDs and also shows no overlap.  It is corroboration, not
                  provenance: it is a different extraction from the one used for
                  training.  Requires the 13-mer archive; skipped if absent.
"""
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, 'manifests')

# 可选: 13-mer 归档不随仓库分发(43 GB), 有就跑, 没有就跳过
KMER13_SPLIT = os.environ.get('OXONET_K13_SPLIT', '/data/k13/split')


def check_manifest():
    """Validate the released split assignment itself.

    This is *manifest validation*, not a read-identifier check: it confirms that
    the released assignment is well formed and matches what the paper describes.
    That a read never spans two groups is a property of the preprocessing (each
    read is written to exactly one fast5 sub-directory and one extraction part),
    not something these files can demonstrate on their own.
    """
    import json
    with open(os.path.join(M, 'split_manifest.json')) as f:
        man = json.load(f)
    ok = (man.get('seed') == 42)
    print('    seed {} {}'.format(man.get('seed'), '(expected 42)' if ok else '<- EXPECTED 42'))
    for key, expect in (('oligo', {'train': 762, 'valid': 42, 'test': 42}),
                        ('t2t', {'train': 283, 'valid': 16, 'test': 16})):
        got = {}
        for v in man[key].values():
            got[v] = got.get(v, 0) + 1
        bad = set(got) - {'train', 'valid', 'test'}
        ok &= (not bad) and got == expect
        print('    {:<6} {:>4} groups  train/valid/test = {}/{}/{}{}'.format(
            key, len(man[key]), got.get('train', 0), got.get('valid', 0), got.get('test', 0),
            '' if got == expect else '   <- expected {}'.format(expect)))
        if bad:
            print('      unexpected split labels: {}'.format(sorted(bad)))
    return ok


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

    print('[manifest] split assignment, validated')
    man_ok = check_manifest()

    print('\n[exhaustive] every read basecalled in the 846 oligonucleotide groups')
    import glob
    otr = set()
    for f in sorted(glob.glob(os.path.join(M, 'oligo_all_reads_train_*.txt.gz'))):
        otr |= load(os.path.basename(f))
    oev = {sp: load('oligo_all_reads_{}.txt.gz'.format(sp)) for sp in ('valid', 'test')}
    print('    train {:,} | valid {:,} | test {:,}'.format(
        len(otr), len(oev['valid']), len(oev['test'])))
    all_ok = report('oligo', otr, oev)
    all_ok &= (len(oev['valid'] & oev['test']) == 0)
    print('    valid n test          = {:>7,}'.format(len(oev['valid'] & oev['test'])))
    all_ok &= report('oligo', otr, ev)

    print('\n[exact] training positives, as fed to the model')
    tp = load('reads_train_pos.txt.gz')
    print('    {:,} reads, {:,} of them outside the exhaustive train set'.format(
        len(tp), len(tp - otr)))
    pos_ok = report('positives', tp, ev) and not (tp - otr)

    print('\n[construction] genomic training negatives, not enumerable')
    print('    Their read IDs were not archived.  Disjointness follows from the')
    print('    assignment in manifests/split_manifest.json; nothing is asserted here.')

    print('\n[independent] separate 13-mer extraction over the same partition')
    k13_ok = None
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
        k13_ok = report('negatives*', s, ev)

    print()
    print('{} [manifest]:     released split assignment is well formed (seed 42, '
          'expected group counts).'.format('PASS' if man_ok else 'FAIL'))
    print('{} [exhaustive]:   the 2,683,085 oligonucleotide training reads are disjoint from '
          'validation and test.'.format('PASS' if all_ok else 'FAIL'))
    print('{} [exact]:        no training-positive read appears in validation '
          'or test.'.format('PASS' if pos_ok else 'FAIL'))
    print('DOCUMENTED [construction]: genomic training-negative disjointness follows from the')
    print('           file and part assignment; a direct training-negative identifier')
    print('           intersection is unavailable and is not claimed.')
    print('{} [independent]: 13-mer corroboration{}'.format(
        'PASS' if k13_ok else ('SKIPPED' if k13_ok is None else 'FAIL'),
        ' archive not present.' if k13_ok is None else
        ' extraction shows no overlap.' if k13_ok else ' extraction shows overlap.'))
    bad = (not man_ok) or (not pos_ok) or (not all_ok) or (k13_ok is False)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
