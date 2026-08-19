"""Score the packed train/valid/test npz (build/pack) with an OxoNet checkpoint.

Two modes:

  score   one pack npz (valid.npz / test_oligo.npz / test_t2t.npz) -> probs.npz
          (keys: prob float32[N], label int8[N]) -- small, ~12 MB per 3M sites.
  report  pick T* at a target specificity on the valid probs and apply it to the
          other score files (recall / specificity / FPR at 0.5 and at T*).

The pack npz layout comes from dataset/pack_train.py: signal[N,175] f32,
kmers[N] '<U7', dwell[N,7] f32, label[N], read_id, basecall_pos. Tensor
construction mirrors dataset.dataset.TestDataset (signal -> (N,175,1), full
7-mer, sig_l = dwell.sum(1)); nothing in the OxoNet project is imported except
model.model.DetectModel, read-only.

    /home/bio/anaconda3/bin/python oxo/analysis/score_pack.py score \
        --npz /home/bio/8oxog/build/pack/valid.npz --out .../valid.probs.npz \
        --model-file /home/bio/8oxog/726train1/last.ckpt \
        --oxonet-dir /home/bio/bio_seq/100_retrain --seq-l 7 --pos-mode rope
"""
import argparse
import os
import time

import numpy as np
import torch

from oxo_predict import encode_kmers, load_model


def score(args):
    """Score every --npz in ONE process.

    Deliberately one process for all sets: each CUDA context teardown is a
    power-state transition, and it was exactly such a transition that tripped
    the Xid 62 PMU halt (see gpu_health.py). Fewer processes, fewer chances.
    """
    model = load_model(args.model_file, args.device, args.oxonet_dir, args.seq_l, args.pos_mode)
    for npz_path, out_path in zip(args.npz, args.out):
        if os.path.isfile(out_path):
            print('skip (exists):', out_path, flush=True)
            continue
        score_one(npz_path, out_path, model, args)


def score_one(npz_path, out_path, model, args):
    t0 = time.time()
    a = np.load(npz_path, allow_pickle=False)
    signal = a['signal'].astype(np.float32, copy=False)
    n = signal.shape[0]
    km = encode_kmers(a['kmers'])
    if args.seq_l < 7:
        cut = 7 - args.seq_l
        km = np.ascontiguousarray(km[:, cut // 2: 7 - (cut - cut // 2)])
    sig_l = a['dwell'].astype(np.float32, copy=False).sum(1)
    label = a['label'].astype(np.int8) if 'label' in a.files else np.full(n, -1, np.int8)
    print('{}: n={} loaded in {:.0f}s'.format(os.path.basename(npz_path), n, time.time() - t0),
          flush=True)

    probs = np.empty(n, dtype=np.float32)
    bs = args.batch_size
    for s in range(0, n, bs):
        e = min(s + bs, n)
        sig = torch.from_numpy(signal[s:e]).unsqueeze(-1).to(args.device)
        kmr = torch.from_numpy(km[s:e]).to(args.device)
        sl = torch.from_numpy(sig_l[s:e]).to(args.device)
        with torch.no_grad():
            _, pred = model(sig, kmr, sl)
        probs[s:e] = pred.squeeze(-1).float().cpu().numpy()
        if (s // bs) % 200 == 0:
            el = time.time() - t0
            print('  {:.1%}  {:.0f}s elapsed  ({:.0f} sites/s)'.format(e / n, el, e / max(el, 1e-9)),
                  flush=True)
    np.savez(out_path, prob=probs, label=label)
    print('wrote {}  ({} sites, {:.0f}s total, {:.0f} sites/s)'.format(
        out_path, n, time.time() - t0, n / max(time.time() - t0, 1e-9)), flush=True)


def _rates(prob, label, thr):
    pos, neg = label == 1, label == 0
    recall = float((prob[pos] >= thr).mean()) if pos.any() else float('nan')
    fpr = float((prob[neg] >= thr).mean()) if neg.any() else float('nan')
    return recall, fpr


def report(args):
    v = np.load(args.valid)
    vp, vl = v['prob'], v['label']
    neg = vp[vl == 0]
    # T* = smallest threshold whose specificity on valid negatives >= target
    tstar = float(np.quantile(neg, args.spec))
    print('valid: {} pos / {} neg'.format(int((vl == 1).sum()), len(neg)))
    print('T* @ {:.3%} specificity on valid negatives = {:.6f}'.format(args.spec, tstar))
    r, f = _rates(vp, vl, tstar)
    print('  valid    recall@T*={:.4%}  FPR@T*={:.4%}'.format(r, f))
    r, f = _rates(vp, vl, 0.5)
    print('  valid    recall@0.5={:.4%}  FPR@0.5={:.4%}'.format(r, f))
    for path in args.others:
        d = np.load(path)
        p, l = d['prob'], d['label']
        name = os.path.basename(path)
        print('{}: n={} ({} pos / {} neg)'.format(name, len(p), int((l == 1).sum()),
                                                  int((l == 0).sum())))
        for thr, tag in ((0.5, '0.5'), (tstar, 'T*')):
            r, f = _rates(p, l, thr)
            print('  {:<8} recall@{}={:.4%}  FPR@{}={:.4%}'.format(name, tag, r, tag, f))
    with open(args.tstar_out, 'w') as fh:
        fh.write('{:.6f}\n'.format(tstar))
    print('T* written to', args.tstar_out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('score')
    s.add_argument('--npz', required=True, nargs='+', help='one or more pack npz (one process)')
    s.add_argument('--out', required=True, nargs='+', help='matching output paths')
    s.add_argument('--model-file', default='/home/bio/8oxog/726train1/last.ckpt')
    s.add_argument('--oxonet-dir', default='/home/bio/bio_seq/100_retrain')
    s.add_argument('--seq-l', type=int, default=7)
    s.add_argument('--pos-mode', default='rope')
    s.add_argument('--device', default='cuda:0')
    s.add_argument('--batch-size', type=int, default=512)
    s.set_defaults(func=score)

    r = sub.add_parser('report')
    r.add_argument('--valid', required=True, help='valid probs.npz (T* is picked here)')
    r.add_argument('--others', nargs='*', default=[])
    r.add_argument('--spec', type=float, default=0.999)
    r.add_argument('--tstar-out', default='tstar.txt')
    r.set_defaults(func=report)

    args = ap.parse_args()
    args.func(args)
