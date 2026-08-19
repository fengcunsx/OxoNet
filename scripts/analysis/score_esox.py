"""esox (Remora) scoring for native per-base output dirs.

Reuses the exact scoring internals of scripts/modcall.py (RemoraModel + remora.pt +
RemoraDataset.get_data + softmax class-1), but adapts to the native layout produced
by basecall_unified_mp.py:
    <out>/<base>/esox/<base>.wN.partNNNN.npz      (esox features: x,e1,s1,p1,read_id,basecall_pos)
    <out>/<base>/<base>.bonito.fastq              (the sequence basecall_pos indexes into)

For each <base>: load the bonito fastq once, score every esox shard, emit ONE tsv
    <scores-dir>/<base>.tsv   columns: read_id  basecall_pos  oxog_score  5mer

Model is loaded once and reused across all bases.
"""
import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from esox.models.remora import RemoraModel
from esox.dataclasses import RemoraDataset
from esox.fast_io import read_fast


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def load_model(model_file, device):
    model = RemoraModel(device=device, dataset=None, size=256, dropout=0,
                        use_raw=True, use_seq_1=True, use_expected_signal_1=True)
    ck = torch.load(model_file, map_location=device)
    model.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)
    return model.to(device)


def score_base(base_dir, base, model, rds, device, batch_size, out_tsv):
    fastq = os.path.join(base_dir, base + '.bonito.fastq')
    assert os.path.isfile(fastq), 'missing bonito fastq: {}'.format(fastq)
    basecalls = read_fast(fastq)

    shards = sorted(glob.glob(os.path.join(base_dir, 'esox', base + '.w*.part*.npz')))
    assert shards, 'no esox npz shards in {}'.format(os.path.join(base_dir, 'esox'))

    frames = []
    for npz_file in shards:
        arr = np.load(npz_file)
        read_ids = arr['read_id']
        positions = arr['basecall_pos']
        n = arr['x'].shape[0]
        if n == 0:
            continue
        data = {k: arr[k] for k in arr.keys() if k not in ('read_id', 'basecall_pos')}

        scores = np.zeros((n,), dtype=float)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            batch = default_collate([rds.get_data(data, i) for i in range(s, e)])
            with torch.no_grad():
                scores[s:e] = torch.nn.functional.softmax(
                    model.predict_step(batch), -1)[:, 1].cpu().numpy()
        scores = trunc(scores, decs=4)

        mer5 = []
        for rid, bp in zip(read_ids, positions):
            seq = basecalls[rid][0]
            bp = int(bp)
            assert 0 <= bp < len(seq) and seq[bp] == 'G', \
                'pos {} not G in read {} (len {})'.format(bp, rid, len(seq))
            mer5.append(seq[bp - 2:bp + 3])

        frames.append(pd.DataFrame({
            'read_id': read_ids,
            'basecall_pos': positions,
            'oxog_score': scores,
            '5mer': np.array(mer5),
        }))

    df = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=['read_id', 'basecall_pos', 'oxog_score', '5mer'])
    df.to_csv(out_tsv, sep='\t', index=False,
              compression='gzip' if out_tsv.endswith('.gz') else None)
    return len(df)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', default='/home/bio/8oxog/wtl1/out', help='dir with per-base subdirs')
    ap.add_argument('--scores-dir', default='/home/bio/8oxog/wtl1/scores/esox')
    ap.add_argument('--model-file', default='/home/bio/esox/esox-main/static/models/remora.pt')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--base', default=None, help='only this base (smoke test); else all')
    ap.add_argument('--shard', type=int, default=0, help='this process handles bases[shard::num-shards]')
    ap.add_argument('--num-shards', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.scores_dir, exist_ok=True)
    device = args.device
    model = load_model(args.model_file, device)
    rds = RemoraDataset(data_dir='.', non_masked_bases=3, seed=0,
                        primary_name='primary', secondary_name=None, inference_mode=True)

    if args.base:
        bases = [args.base]
    else:
        bases = sorted(d for d in os.listdir(args.out_root)
                       if os.path.isdir(os.path.join(args.out_root, d)) and d != '_tmp')
        bases = bases[args.shard::args.num_shards]

    for base in bases:
        out_tsv = os.path.join(args.scores_dir, base + '.tsv.gz')
        if os.path.isfile(out_tsv):
            print('skip (exists):', base); continue
        base_dir = os.path.join(args.out_root, base)
        n = score_base(base_dir, base, model, rds, device, args.batch_size, out_tsv)
        print('scored {}: {} sites -> {}'.format(base, n, out_tsv))
