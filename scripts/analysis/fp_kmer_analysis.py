"""Do OxoNet's FALSE POSITIVES concentrate on k-mers that are distribution-shifted /
under-covered in the oligo TRAINING data?

T2T is ~all-negative genomic G -> every OxoNet positive call on T2T is a false positive.
For each k-mer context we compute:
  - t2t_fpr  : fraction of T2T sites of that k-mer called positive @T* (= per-kmer FPR)
  - oligo_ppos, oligo_n : that k-mer's positive-fraction and sample count in the oligo
                          training/test data (merged_old.npz, which HAS labels)
Then test two hypotheses for the over-calling:
  (H1 context/label shift): t2t_fpr high where oligo_ppos high (model memorized context->pos)
  (H2 coverage gap):        t2t_fpr high where oligo_n low / kmer ABSENT in oligo (extrapolation)

k = 7 (signal sensing window) and 5 (model's Seq-Net input) both reported.

    /home/bio/anaconda3/bin/python oxo/analysis/fp_kmer_analysis.py --sample 500000
Outputs: wt1/fp_kmer.txt + wt1/figs/FP_kmer_*.png + wt1/fp_kmer_table_{5,7}.csv
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oxo_predict import load_model, encode_kmers


def score_t2t(npz_glob, model_file, device, batch_size, sample, cache):
    if cache and os.path.isfile(cache):
        d = np.load(cache, allow_pickle=True)
        return d['km'].astype('<U7'), d['prob']
    files = sorted(glob.glob(npz_glob))
    assert files, 'no T2T npz'
    cap = sample * 3
    sig, km, dw, acc = [], [], [], 0
    for f in files:
        if acc >= cap:
            break
        a = np.load(f)
        if a['signal'].shape[0] == 0:
            continue
        sig.append(a['signal']); km.append(np.asarray(a['kmers']).astype('<U7')); dw.append(a['dwell'])
        acc += a['signal'].shape[0]
    sig = np.concatenate(sig); km = np.concatenate(km); dw = np.concatenate(dw)
    n = sig.shape[0]
    if n > sample:
        idx = np.random.default_rng(0).choice(n, sample, replace=False)
        sig, km, dw = sig[idx], km[idx], dw[idx]; n = sample
    model = load_model(model_file, device)
    signal = torch.from_numpy(sig.astype(np.float32)).unsqueeze(-1)
    kmer5 = torch.from_numpy(encode_kmers(km)[:, 1:6])
    sig_l = torch.from_numpy(dw.astype(np.float32).sum(1))
    prob = np.empty(n, dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        with torch.no_grad():
            _, p = model(signal[s:e].to(device), kmer5[s:e].to(device), sig_l[s:e].to(device))
        prob[s:e] = p.squeeze(-1).cpu().numpy()
    if cache:
        np.savez_compressed(cache, km=km, prob=prob)
    return km, prob


def per_kmer(km7, values, k, is_label=False):
    """aggregate values by central k-mer (k in {5,7}). km7 is '<U7'."""
    off = (7 - k) // 2
    kk = np.array([s[off:off + k] for s in km7])
    df = pd.DataFrame({'kmer': kk, 'v': values})
    g = df.groupby('kmer')['v'].agg(['size', 'mean'])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t2t-glob', default='/home/bio/8oxog/t2t/288548394/*/output_new/oxo/*.npz')
    ap.add_argument('--oligo', default='/home/bio/8oxog/data/merged_old.npz')
    ap.add_argument('--model-file', default='/home/bio/8oxog/oxo_old_check/model_145.pth')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--sample', type=int, default=500000)
    ap.add_argument('--tstar', type=float, default=0.6656)
    ap.add_argument('--min-n', type=int, default=30, help='min sites per kmer in both sets')
    ap.add_argument('--figdir', default='/home/bio/8oxog/wtl1/figs')
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/fp_kmer.txt')
    ap.add_argument('--cache', default='/home/bio/8oxog/wtl1/t2t_scored.npz')
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    km7, prob = score_t2t(args.t2t_glob, args.model_file, args.device,
                          args.batch_size, args.sample, args.cache)
    fp = (prob >= args.tstar).astype(float)
    o = np.load(args.oligo)
    okm7 = np.asarray(o['kmers']).astype('<U7'); olab = np.asarray(o['label']).reshape(-1).astype(float)

    lines = []
    def emit(s=''):
        print(s, flush=True); lines.append(s)
    emit('=== OxoNet false-positive k-mers vs oligo training distribution ===')
    emit('T2T (all-negative): {} sites scored, overall FPR@{}={:.3%}'.format(
        len(prob), args.tstar, fp.mean()))
    emit('')

    for k in (7, 5):
        t = per_kmer(km7, fp, k).rename(columns={'size': 't2t_n', 'mean': 't2t_fpr'})
        og = per_kmer(okm7, olab, k).rename(columns={'size': 'oligo_n', 'mean': 'oligo_ppos'})
        j = t.join(og, how='left')                       # left: keep all T2T kmers
        j['in_oligo'] = j['oligo_n'].notna()
        j.to_csv(os.path.join(args.figdir, '..', 'fp_kmer_table_{}.csv'.format(k)))
        tot = len(j)
        jn = j[j['t2t_n'] >= args.min_n].copy()          # enough T2T sites for a stable FPR
        emit('--- {}-mer ({} distinct in T2T; {} with n>={}) ---'.format(k, tot, len(jn), args.min_n))

        def cat(row):
            if not row['in_oligo']:
                return '0_absent_in_oligo'
            p = row['oligo_ppos']
            if p >= 0.99: return '1_pos_only(>=.99)'
            if p >= 0.6:  return '2_pos_heavy(.6-.99)'
            if p >= 0.2:  return '3_mid(.2-.6)'
            return '4_neg_heavy(<.2)'
        jn['cat'] = jn.apply(cat, axis=1)
        emit('  T2T false-positive rate by oligo-training category:')
        emit('    {:<22} {:>6} {:>10} {:>9}'.format('category', 'kmers', 't2t_sites', 'FPR'))
        for c, sub in jn.groupby('cat'):
            emit('    {:<22} {:>6} {:>10} {:>8.2%}'.format(
                c, len(sub), int(sub['t2t_n'].sum()),
                np.average(sub['t2t_fpr'], weights=sub['t2t_n'])))

        present = jn[jn['in_oligo'] & (jn['oligo_n'] >= args.min_n)]
        if len(present) > 5:
            rho_p = spearmanr(present['oligo_ppos'], present['t2t_fpr'])[0]
            rho_n = spearmanr(present['oligo_n'], present['t2t_fpr'])[0]
            emit('  Spearman( oligo_ppos , t2t_fpr ) = {:.3f}   [H1 context/label shift]'.format(rho_p))
            emit('  Spearman( oligo_n    , t2t_fpr ) = {:.3f}   [H2 coverage: neg if - ]'.format(rho_n))
            if k == 7:
                plt.figure(figsize=(5.5, 4.5))
                plt.scatter(present['oligo_ppos'], present['t2t_fpr'],
                            s=np.sqrt(present['t2t_n']), alpha=.5, edgecolors='none')
                plt.xlabel('oligo P(pos) per 7-mer (training)')
                plt.ylabel('T2T false-positive rate per 7-mer')
                plt.title('FP concentrate where oligo is pos-biased? rho={:.3f}'.format(rho_p))
                plt.tight_layout(); plt.savefig(os.path.join(args.figdir, 'FP_kmer_ppos7.png'), dpi=110); plt.close()
        # top FP 7-mers and their oligo status
        if k == 7:
            top = j[j['t2t_n'] >= args.min_n].sort_values('t2t_fpr', ascending=False).head(12)
            emit('  top-FPR 7-mers (n>={}):'.format(args.min_n))
            emit('    {:<8} {:>7} {:>8} | {:>8} {:>9}'.format('7mer', 't2t_n', 't2t_fpr', 'oligo_n', 'oligo_ppos'))
            for kmer, r in top.iterrows():
                emit('    {:<8} {:>7} {:>7.1%} | {:>8} {:>9}'.format(
                    kmer, int(r['t2t_n']), r['t2t_fpr'],
                    '-' if not r['in_oligo'] else int(r['oligo_n']),
                    '-' if not r['in_oligo'] else '{:.3f}'.format(r['oligo_ppos'])))
        emit('')

    with open(args.out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
