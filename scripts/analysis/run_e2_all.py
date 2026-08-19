"""Whole E2 evaluation in ONE process -- pack test sets, T*, then all native bases.

Why one process: this laptop's GPU hits Xid 62 (PMU halt) on CUDA *context
teardown* -- three for three, always within seconds of a CUDA process exiting,
never during sustained load (valid ran 341s inside one process untouched).
After the halt the GPU is stuck at ~1/30 speed until reboot. So the whole job
runs in a single context and only exits once everything is on disk; a crash at
that point costs nothing.

A loader thread prefetches the next npz shard while the GPU works on the
current one, which is most of what the 4-way parallelism used to buy.

    /home/bio/anaconda3/bin/python run_e2_all.py            # full run
    /home/bio/anaconda3/bin/python run_e2_all.py --skip-pack --limit 4
"""
import argparse
import glob
import os
import queue
import threading
import time

import numpy as np
import pandas as pd
import torch

from oxo_predict import encode_kmers, load_model
from score_pack import _rates

PACK = '/home/bio/8oxog/build/pack'
WT = '/home/bio/8oxog/wtl1'


def log(msg):
    print('[{}] {}'.format(time.strftime('%H:%M:%S'), msg), flush=True)


def run_batches(model, signal, km, sig_l, device, bs, tag, t0):
    """Score arrays already in RAM; returns probs. Prints throughput as it goes."""
    n = signal.shape[0]
    probs = np.empty(n, dtype=np.float32)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        sig = torch.from_numpy(signal[s:e]).unsqueeze(-1).to(device)
        kmr = torch.from_numpy(km[s:e]).to(device)
        sl = torch.from_numpy(sig_l[s:e]).to(device)
        with torch.no_grad():
            _, pred = model(sig, kmr, sl)
        probs[s:e] = pred.squeeze(-1).float().cpu().numpy()
        if (s // bs) % 400 == 0 and s:
            el = time.time() - t0
            log('  {} {:.1%}  {:.0f} sites/s'.format(tag, e / n, e / max(el, 1e-9)))
    return probs


def prep(a, seq_l):
    """npz -> (signal f32, kmer int64, sig_l f32) exactly as TestDataset does."""
    signal = a['signal'].astype(np.float32, copy=False)
    km = encode_kmers(a['kmers'])
    if seq_l < 7:
        cut = 7 - seq_l
        km = np.ascontiguousarray(km[:, cut // 2: 7 - (cut - cut // 2)])
    sig_l = a['dwell'].astype(np.float32, copy=False).sum(1)
    return signal, km, sig_l


def do_pack(model, args, ps):
    for name in ('valid', 'test_t2t', 'test_oligo'):
        out = os.path.join(ps, name + '.probs.npz')
        if os.path.isfile(out):
            log('pack {}: skip (exists)'.format(name))
            continue
        t0 = time.time()
        a = np.load(os.path.join(PACK, name + '.npz'), allow_pickle=False)
        signal, km, sig_l = prep(a, args.seq_l)
        label = a['label'].astype(np.int8) if 'label' in a.files else np.full(len(km), -1, np.int8)
        log('pack {}: n={} loaded {:.0f}s'.format(name, len(km), time.time() - t0))
        probs = run_batches(model, signal, km, sig_l, args.device, args.batch_size, name, t0)
        np.savez(out, prob=probs, label=label)
        log('pack {}: done {:.0f}s ({:.0f} sites/s)'.format(
            name, time.time() - t0, len(km) / max(time.time() - t0, 1e-9)))


def pick_tstar(ps, spec):
    v = np.load(os.path.join(ps, 'valid.probs.npz'))
    vp, vl = v['prob'], v['label']
    tstar = float(np.quantile(vp[vl == 0], spec))
    lines = ['T* @ {:.3%} spec on valid negatives = {:.6f}'.format(spec, tstar)]
    for name in ('valid', 'test_oligo', 'test_t2t'):
        p = os.path.join(ps, name + '.probs.npz')
        if not os.path.isfile(p):
            continue
        d = np.load(p)
        for thr, tag in ((0.5, '0.5'), (tstar, 'T*')):
            r, f = _rates(d['prob'], d['label'], thr)
            lines.append('  {:<11} recall@{:<3}={:8.4%}   FPR@{:<3}={:8.4%}'.format(
                name, tag, r, tag, f))
    report = '\n'.join(lines)
    print(report, flush=True)
    with open(os.path.join(ps, 'tstar.txt'), 'w') as fh:
        fh.write('{:.6f}\n'.format(tstar))
    with open(os.path.join(WT, 'logs_' + os.path.basename(ps).split('_', 2)[-1], 'report.txt'),
              'w') as fh:
        fh.write(report + '\n')
    return tstar


def loader(paths, q):
    """Prefetch thread: decompressing a shard is ~40% of per-shard wall time."""
    for p in paths:
        q.put((p, np.load(p)))
    q.put(None)


def do_native(model, args, out_dir):
    # 默认沿用 E0 时代的 40 个 base 子集; 要打全部已提特征的 f5 就传 --bases-file
    # (清单文件一行一个 base 名, 见 wt1/all_bases.txt)。
    with open(args.bases_file or os.path.join(WT, 'e0_bases.txt')) as f:
        bases = sorted(l.strip() for l in f if l.strip())
    if args.limit:
        bases = bases[:args.limit]
    todo = [b for b in bases if not os.path.isfile(os.path.join(out_dir, b + '.tsv.gz'))]
    log('native: {}/{} bases to do'.format(len(todo), len(bases)))

    for bi, base in enumerate(todo):
        t0 = time.time()
        shards = sorted(glob.glob(os.path.join(WT, 'out', base, 'oxo', base + '.w*.part*.npz')))
        q = queue.Queue(maxsize=2)
        threading.Thread(target=loader, args=(shards, q), daemon=True).start()
        frames, total = [], 0
        while True:
            item = q.get()
            if item is None:
                break
            _, a = item
            if a['signal'].shape[0] == 0:
                continue
            signal, km, sig_l = prep(a, args.seq_l)
            probs = run_batches(model, signal, km, sig_l, args.device, args.batch_size,
                                base[-6:], t0)
            total += len(probs)
            frames.append(pd.DataFrame({
                'read_id': np.asarray(a['read_id']).reshape(-1),
                'basecall_pos': np.asarray(a['basecall_pos']).reshape(-1),
                'prob': np.trunc(probs * 1e4) / 1e4,
            }))
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(os.path.join(out_dir, base + '.tsv.gz'), sep='\t', index=False,
                  compression='gzip')
        el = time.time() - t0
        log('native [{}/{}] {}: {} sites, {:.0f}s ({:.0f} sites/s)'.format(
            bi + 1, len(todo), base, total, el, total / max(el, 1e-9)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/home/bio/8oxog/726train1/last.ckpt')
    ap.add_argument('--oxonet-dir', default='/home/bio/bio_seq/100_retrain')
    ap.add_argument('--seq-l', type=int, default=7)
    ap.add_argument('--pos-mode', default='rope')
    ap.add_argument('--tag', default='e2ep72')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--spec', type=float, default=0.999)
    ap.add_argument('--limit', type=int, default=0, help='>0: only first N native bases')
    ap.add_argument('--skip-pack', action='store_true')
    ap.add_argument('--bases-file', default='', help='native base 清单(一行一个); 缺省=e0_bases.txt 的 40 个')
    ap.add_argument('--wt-root', default='', help='native 样本根目录(含 out/); 缺省 wt1。'
                                                 '打 wtd1 时传 /home/bio/8oxog/wtd1')
    args = ap.parse_args()
    if args.wt_root:
        WT = args.wt_root

    ps = os.path.join(WT, 'pack_scores_' + args.tag)
    out_dir = os.path.join(WT, 'scores', 'oxonet_' + args.tag)
    for d in (ps, out_dir, os.path.join(WT, 'logs_' + args.tag)):
        os.makedirs(d, exist_ok=True)

    t_start = time.time()
    model = load_model(args.ckpt, args.device, args.oxonet_dir, args.seq_l, args.pos_mode)
    if not args.skip_pack:
        do_pack(model, args, ps)
        pick_tstar(ps, args.spec)
    do_native(model, args, out_dir)
    log('ALL DONE in {:.0f} min'.format((time.time() - t_start) / 60))
