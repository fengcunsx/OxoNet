"""Stage F (6.A) — model concordance between esox and OxoNet on native G sites.

This is a CONSISTENCY check, NOT an accuracy claim (native has no site-level truth).
Red lines (LOCKED): only 100-context sites (already filtered upstream) / concordance
not validation / no ranking, no assumed direction / no leakage.

Reads sites_scores.parquet (read_id, basecall_pos, 5mer, oxog_score, oxonet_prob).

Reports (threshold-free first):
  1. Spearman & Pearson (esox_score vs oxonet_prob) — the primary, threshold-free evidence.
  2. Per-model positive rate + overlap / Jaccard at thr=0.5 (both models).
     (99.9%-spec T* needs the oligo labeled test set -> Stage E, deferred.)
  3. Consensus positives (both call +) as a high-confidence proxy -> each model's
     recall relative to the consensus set.

    /home/bio/anaconda3/bin/python oxo/analysis/concordance.py
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def overlap_stats(e_pos, o_pos):
    inter = int(np.sum(e_pos & o_pos))
    union = int(np.sum(e_pos | o_pos))
    ne, no = int(e_pos.sum()), int(o_pos.sum())
    jac = inter / union if union else float('nan')
    return dict(esox_pos=ne, oxonet_pos=no, both=inter, union=union, jaccard=jac,
                esox_recall_vs_consensus=inter / ne if ne else float('nan'),
                oxonet_recall_vs_consensus=inter / no if no else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', default='/home/bio/8oxog/wtl1/sites_scores.parquet')
    ap.add_argument('--thresholds', default='0.5', help='SYMMETRIC comma list (same thr both models)')
    ap.add_argument('--esox-thr', type=float, default=0.5, help='esox operating point')
    ap.add_argument('--oxonet-thrs', default='', help='per-model: esox@--esox-thr vs each oxonet thr (comma list); e.g. OxoNet 99.9%-spec T*=0.665585')
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/concordance.txt')
    ap.add_argument('--spearman-sample', type=int, default=5_000_000,
                    help='subsample for Spearman (rank sort on 47M is slow); 0=all')
    args = ap.parse_args()

    df = pd.read_parquet(args.sites, columns=['oxog_score', 'oxonet_prob'])
    e = df['oxog_score'].to_numpy(np.float64)
    o = df['oxonet_prob'].to_numpy(np.float64)
    n = len(e)

    lines = []
    def emit(s=''):
        print(s, flush=True); lines.append(s)

    emit('=== Stage F / 6.A model concordance (esox vs OxoNet) ===')
    emit('sites: {}  (native G, 100-context, 40 fast5)'.format(n))
    emit('esox_score:   mean={:.4f} median={:.4f} p99={:.4f}'.format(
        e.mean(), np.median(e), np.percentile(e, 99)))
    emit('oxonet_prob:  mean={:.4f} median={:.4f} p99={:.4f}'.format(
        o.mean(), np.median(o), np.percentile(o, 99)))
    emit('')

    # 1. threshold-free correlation
    emit('--- (1) threshold-free correlation ---')
    pr, _ = pearsonr(e, o)
    emit('Pearson  (all {} pts): {:.4f}'.format(n, pr))
    if args.spearman_sample and n > args.spearman_sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, args.spearman_sample, replace=False)
        sr, _ = spearmanr(e[idx], o[idx])
        emit('Spearman (subsample {}): {:.4f}'.format(args.spearman_sample, sr))
    else:
        sr, _ = spearmanr(e, o)
        emit('Spearman (all {} pts): {:.4f}'.format(n, sr))
    emit('')

    # 2/3. overlap + consensus at each threshold
    emit('--- (2/3) overlap / Jaccard / consensus @ thresholds ---')
    for thr in [float(t) for t in args.thresholds.split(',')]:
        st = overlap_stats(e >= thr, o >= thr)
        emit('thr={:.3f}: esox+={} ({:.3%})  oxonet+={} ({:.3%})'.format(
            thr, st['esox_pos'], st['esox_pos'] / n, st['oxonet_pos'], st['oxonet_pos'] / n))
        emit('          both={}  union={}  Jaccard={:.4f}'.format(
            st['both'], st['union'], st['jaccard']))
        emit('          consensus recall: esox={:.3%}  oxonet={:.3%}'.format(
            st['esox_recall_vs_consensus'], st['oxonet_recall_vs_consensus']))
    emit('')

    if args.oxonet_thrs:
        emit('--- (2/3b) PER-MODEL operating points: esox@{:.4f} vs OxoNet@T* ---'.format(args.esox_thr))
        e_pos = e >= args.esox_thr
        for ot in [float(t) for t in args.oxonet_thrs.split(',')]:
            st = overlap_stats(e_pos, o >= ot)
            emit('esox@{:.3f} / oxonet@{:.6f}: esox+={} ({:.3%})  oxonet+={} ({:.3%})'.format(
                args.esox_thr, ot, st['esox_pos'], st['esox_pos'] / n, st['oxonet_pos'], st['oxonet_pos'] / n))
            emit('          both={}  union={}  Jaccard={:.4f}'.format(
                st['both'], st['union'], st['jaccard']))
            emit('          consensus recall: esox={:.3%}  oxonet={:.3%}'.format(
                st['esox_recall_vs_consensus'], st['oxonet_recall_vs_consensus']))
        emit('')
    emit('NOTE: OxoNet trained on balanced data -> prob centered ~0.5, so thr=0.5 is not'
         ' calibrated; 99.9%-spec T* (Stage E, needs oligo test set) is the fair operating'
         ' point. Spearman above is threshold-free and is the primary concordance evidence.')

    with open(args.out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
