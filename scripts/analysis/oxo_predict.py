"""OxoNet (DetectModel) scoring for native per-base oxo npz.

Slim predictor: the repo's script/prediction.py loads ONE combined test.npz + needs
labels + dumps embeddings/mean/std/dwell — infeasible for 176M native sites (signal
is 175xfloat64 ~= 246GB combined). Here we run PER BASE and emit only read_id /
basecall_pos / prob (gzip).

Input = the oxo npz shards from basecall_unified_mp (keys: signal[N,175], kmers[N]'<U7',
dwell[N,7], read_id, basecall_pos). Model uses only signal + center-5 kmer + sig_l
(mean/std/dwell are loaded by their dataset but NOT fed to the model — see train.py
parse_data). Tensor construction mirrors dataset.TestDataset exactly.

    cd /home/bio/bio_seq/100_42 && python <this> --model .../model_145.pth [--base X]
"""
import os
import sys
import argparse
import glob
import gzip

import numpy as np
import pandas as pd
import torch

# import DetectModel from the canonical OxoNet project. 100_42 = the published
# model_145 (seq_l=5); 100_retrain = the E1/E2 retrain (seq_l=7, RoPE). Read-only
# import -- neither project is modified by this script.
# BASE_DICT is hardcoded (copy of dataset.dataset.BASE_DICT) to avoid pulling the
# dataset->generate_paire import chain, which may break under the base env.
OXONET = os.environ.get('OXONET_DIR', '/home/bio/bio_seq/100_42')
BASE_DICT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'o': 2}

# ord-LUT for fast 7-mer encoding (default -> G's code 2, matches BASE_DICT['o']/'G')
_LUT = np.full(128, BASE_DICT['G'], dtype=np.int64)
for ch, v in BASE_DICT.items():
    _LUT[ord(ch)] = v


def encode_kmers(kmers):
    """(N,) '<U7' -> (N,7) int64 via code points."""
    cp = np.ascontiguousarray(kmers).astype('<U7').view(np.uint32).reshape(len(kmers), 7)
    return _LUT[cp]


def load_model(model_file, device, oxonet_dir, seq_l, pos_mode):
    """Import DetectModel from oxonet_dir and load weights.

    Accepts all three checkpoint layouts seen so far: bare state_dict,
    {'model_state_dict': ...} (published model_145) and {'model': ..., 'optimizer': ...}
    (train.py's last.ckpt).
    """
    sys.path.insert(0, oxonet_dir)
    from model.model import DetectModel
    kw = {}
    if pos_mode:                       # 100_42's DetectModel has no pos_mode arg
        kw['pos_mode'] = pos_mode
    model = DetectModel(sig_blocks=4, sig_l=175, seq_l=seq_l, **kw).to(device)
    ck = torch.load(model_file, map_location=device)
    if isinstance(ck, dict) and 'model' in ck:
        print('[ckpt] {} epoch={} global_step={}'.format(
            model_file, ck.get('epoch'), ck.get('global_step')), flush=True)
        sd = ck['model']
    elif isinstance(ck, dict) and 'model_state_dict' in ck:
        sd = ck['model_state_dict']
    else:
        sd = ck
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def predict_base(base_dir, base, model, device, batch_size, out_path, seq_l):
    shards = sorted(glob.glob(os.path.join(base_dir, 'oxo', base + '.w*.part*.npz')))
    assert shards, 'no oxo npz shards in {}'.format(os.path.join(base_dir, 'oxo'))
    frames = []
    for si, npz in enumerate(shards):
        a = np.load(npz)
        n = a['signal'].shape[0]
        if n == 0:
            continue
        print('  shard {}/{} n={}'.format(si + 1, len(shards), n), flush=True)
        signal = torch.from_numpy(a['signal'].astype(np.float32)).unsqueeze(-1)  # (n,175,1)
        km = encode_kmers(a['kmers'])                                             # (n,7)
        if seq_l < 7:                     # seq_l=5 -> center 5-mer (mirrors train.py parse_data)
            cut = 7 - seq_l
            km = km[:, cut // 2: 7 - (cut - cut // 2)]
        kmer5 = torch.from_numpy(np.ascontiguousarray(km))                        # (n,seq_l)
        sig_l = torch.from_numpy(a['dwell'].astype(np.float32).sum(1))            # (n,)
        probs = np.empty(n, dtype=np.float32)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            with torch.no_grad():
                _, pred = model(signal[s:e].to(device), kmer5[s:e].to(device), sig_l[s:e].to(device))
            probs[s:e] = pred.squeeze(-1).cpu().numpy()
        frames.append(pd.DataFrame({
            'read_id': np.asarray(a['read_id']).reshape(-1),
            'basecall_pos': np.asarray(a['basecall_pos']).reshape(-1),  # oxo npz stores (N,1)
            'prob': np.asarray(np.trunc(probs * 1e4) / 1e4).reshape(-1),
        }))
    df = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=['read_id', 'basecall_pos', 'prob'])
    df.to_csv(out_path, sep='\t', index=False, compression='gzip')
    return len(df)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', default='/home/bio/8oxog/wtl1/out')
    ap.add_argument('--out-dir', default='/home/bio/8oxog/wtl1/scores/oxonet')
    ap.add_argument('--model-file', default='/home/bio/8oxog/oxo_old_check/model_145.pth')
    ap.add_argument('--oxonet-dir', default=OXONET,
                    help='project dir to import model.model from (100_42 | 100_retrain)')
    ap.add_argument('--seq-l', type=int, default=5, help='5 = published model_145, 7 = retrain')
    ap.add_argument('--pos-mode', default=None, help="retrain only: 'rope' | 'none' | 'learnable'")
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--base', default=None)
    ap.add_argument('--bases-file', default=None, help='file with one base name per line')
    ap.add_argument('--limit', type=int, default=0, help='>0: only first N bases (sorted)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.model_file, args.device, args.oxonet_dir, args.seq_l, args.pos_mode)

    if args.base:
        bases = [args.base]
    elif args.bases_file:
        with open(args.bases_file) as f:
            bases = sorted(l.strip() for l in f if l.strip())
        bases = bases[args.shard::args.num_shards]
    else:
        bases = sorted(d for d in os.listdir(args.out_root)
                       if os.path.isdir(os.path.join(args.out_root, d)) and d != '_tmp')
        if args.limit > 0:
            bases = bases[:args.limit]
        bases = bases[args.shard::args.num_shards]

    for base in bases:
        out_path = os.path.join(args.out_dir, base + '.tsv.gz')
        if os.path.isfile(out_path):
            print('skip (exists):', base); continue
        n = predict_base(os.path.join(args.out_root, base), base, model, args.device,
                         args.batch_size, out_path, args.seq_l)
        print('oxonet {}: {} sites -> {}'.format(base, n, out_path))
