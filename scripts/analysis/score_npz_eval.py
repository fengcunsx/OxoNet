"""Score any oxo-format npz with OxoNet and report positive rate @0.5 and @T*.

Purpose: 3-way control ladder to separate "inference bug" / "processing shift" /
"genomic OOD" for OxoNet's native over-calling.
  (1) oligo test set merged_old.npz  -> template-anchored, IN-DISTRIBUTION, LABELED
      (should reproduce paper: ~77% recall, ~0.1-0.4% FP on label=0). Validates inference.
  (2) T2T genomic-negative G          -> self-anchored (basecall_unified), all-negative.
  (3) native wt1                      -> self-anchored, genomic + real background.

If npz has 'label', also reports recall / specificity / FPR. Reuses oxo_predict's
model + kmer encoding so the forward is byte-identical to the native run.

    /home/bio/anaconda3/bin/python oxo/analysis/score_npz_eval.py --npz <file|glob> [--sample N]
"""
import os
import sys
import glob
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oxo_predict import load_model, encode_kmers


def collect(npz_arg, sample, seed=0):
    files = sorted(glob.glob(npz_arg)) if any(c in npz_arg for c in '*?[') else [npz_arg]
    assert files, 'no npz matched {}'.format(npz_arg)
    # early-stop cap so we never load 32M T2T sites into RAM before subsampling
    cap = sample * 3 if sample else float('inf')
    sig, km, dw, lab = [], [], [], []
    have_lab = None
    acc = 0
    nf_read = 0
    for f in files:
        if acc >= cap:
            break
        a = np.load(f)
        if a['signal'].shape[0] == 0:
            continue
        acc += a['signal'].shape[0]; nf_read += 1
        sig.append(a['signal']); km.append(np.asarray(a['kmers']).astype('<U7'))
        dw.append(a['dwell'])
        hl = 'label' in a.files
        have_lab = hl if have_lab is None else (have_lab and hl)
        if hl:
            lab.append(np.asarray(a['label']).reshape(-1))
    sig = np.concatenate(sig); km = np.concatenate(km); dw = np.concatenate(dw)
    lab = np.concatenate(lab) if have_lab else None
    n = sig.shape[0]
    if sample and n > sample:
        idx = np.random.default_rng(seed).choice(n, sample, replace=False)
        sig, km, dw = sig[idx], km[idx], dw[idx]
        if lab is not None:
            lab = lab[idx]
    return sig, km, dw, lab, nf_read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='single file or glob')
    ap.add_argument('--model-file', default='/home/bio/8oxog/oxo_old_check/model_145.pth')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--sample', type=int, default=200000, help='0=all')
    ap.add_argument('--tstar', type=float, default=0.6656)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    sig, km, dw, lab, nf = collect(args.npz, args.sample)
    n = sig.shape[0]
    dev = args.device
    model = load_model(args.model_file, dev)

    signal = torch.from_numpy(sig.astype(np.float32)).unsqueeze(-1)
    kmer5 = torch.from_numpy(encode_kmers(km)[:, 1:6])
    sig_l = torch.from_numpy(dw.astype(np.float32).sum(1))
    probs = np.empty(n, dtype=np.float32)
    for s in range(0, n, args.batch_size):
        e = min(s + args.batch_size, n)
        with torch.no_grad():
            _, pred = model(signal[s:e].to(dev), kmer5[s:e].to(dev), sig_l[s:e].to(dev))
        probs[s:e] = pred.squeeze(-1).cpu().numpy()

    print('=== {} ({} files, {} sites scored) ==='.format(args.tag or args.npz, nf, n))
    print('prob: mean={:.4f} median={:.4f} p99={:.4f}'.format(
        probs.mean(), np.median(probs), np.percentile(probs, 99)))
    for thr in (0.5, args.tstar):
        pos = (probs >= thr).mean()
        line = '  @{:.4f}: positive rate = {:.3%}'.format(thr, pos)
        if lab is not None:
            L = lab.astype(bool); P = probs >= thr
            tp = int((P & L).sum()); fp = int((P & ~L).sum())
            fn = int((~P & L).sum()); tn = int((~P & ~L).sum())
            rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); fpr = fp / max(fp + tn, 1)
            line += '  | recall={:.3%} spec={:.3%} FPR={:.3%}'.format(rec, spec, fpr)
        print(line)
    if lab is not None:
        print('  (labeled: {} pos / {} neg)'.format(int(lab.sum()), int((lab == 0).sum())))


if __name__ == '__main__':
    main()
