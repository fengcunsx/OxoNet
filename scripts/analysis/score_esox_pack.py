"""用 esox(Remora) 给**我们自己的 pack 测试集**打分 —— 让三方在逐位点相同的集合上可比。

为什么需要这个：`score_esox.py` 是给 native 的目录布局写的(每个 base 一个
`esox/` 子目录 + bonito fastq)，而我们的 valid/test_oligo/test_t2t 来自 pack，
源头分散在 oligo 的 `feature/esox/{8oxog,g}/data.npz` 和 t2t 的 `output_new/esox/*.npz`。
本脚本按 `split_manifest.json` 找到属于该 split 的 esox 特征文件，打完分后**按
(read_id, basecall_pos) join 回 pack 的行序**，输出与 OxoNet/NanoCon 完全对齐的
`<set>.probs.npz`(prob + label)。

两种 esox 特征 schema(实测)：
  oligo : x1(f64), e1(f64), s1(i64)          ← 无 p1，且 raw 叫 x1
  t2t   : x(f32),  e1(f32), s1(i8), p1(i8)
模型(use_raw/use_seq_1/use_expected_signal_1)只吃 x/e1/s1；p1 虽被
`RemoraDataset.get_data` 读取并 mask，但没有对应的 seq 层(use_phredq_1 默认 False)，
所以 oligo 侧补零 p1 是安全的。

    /home/bio/anaconda3/envs/esox/bin/python score_esox_pack.py --sets valid test_oligo test_t2t
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from esox.models.remora import RemoraModel
from esox.dataclasses import RemoraDataset

PACK = '/home/bio/8oxog/build/pack'
MANIFEST = '/home/bio/8oxog/build/split_manifest.json'


def log(m):
    print('[{}] {}'.format(time.strftime('%H:%M:%S'), m), flush=True)


def load_model(model_file, device):
    model = RemoraModel(device=device, dataset=None, size=256, dropout=0,
                        use_raw=True, use_seq_1=True, use_expected_signal_1=True)
    ck = torch.load(model_file, map_location=device)
    model.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)
    return model.to(device).eval()


def esox_files(split):
    """该 split 用到的 esox 特征文件清单。"""
    m = json.load(open(MANIFEST))
    out = []
    for rel, s in m['oligo'].items():
        if s != split:
            continue
        for cls in ('8oxog', 'g'):
            p = os.path.join(m['oligo_root'], rel, 'feature', 'esox', cls, 'data.npz')
            if os.path.isfile(p):
                out.append(p)
    for rel, s in m['t2t'].items():
        if s != split:
            continue
        p = os.path.join(m['t2t_root'], rel).replace('/oxo/', '/esox/')
        if os.path.isfile(p):
            out.append(p)
    return sorted(out)


def normalize(a):
    """两种 schema 归一到模型要的 dict(x/e1/s1/p1)，dtype 对齐 native 侧。"""
    x = a['x'] if 'x' in a.files else a['x1']
    d = {'x': np.asarray(x, dtype=np.float32),
         'e1': np.asarray(a['e1'], dtype=np.float32),
         's1': np.asarray(a['s1'], dtype=np.int8)}
    d['p1'] = (np.asarray(a['p1'], dtype=np.int8) if 'p1' in a.files
               else np.zeros_like(d['s1']))          # 模型不吃 p1，补零仅为过 get_data
    return d


@torch.no_grad()
def score_file(path, model, rds, device, bs):
    a = np.load(path, allow_pickle=False)
    n = a['read_id'].shape[0]
    if n == 0:
        return None
    data = normalize(a)
    probs = np.empty(n, dtype=np.float32)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        batch = default_collate([rds.get_data(data, i) for i in range(s, e)])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        probs[s:e] = torch.softmax(model.predict_step(batch), -1)[:, 1].cpu().numpy()
    return pd.DataFrame({
        'read_id': np.asarray(a['read_id']).reshape(-1),
        'basecall_pos': np.asarray(a['basecall_pos']).reshape(-1).astype(np.int64),
        'esox': probs,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', default=['valid', 'test_oligo', 'test_t2t'])
    ap.add_argument('--out-dir', default='/home/bio/8oxog/wtl1/pack_scores_esox')
    ap.add_argument('--model-file', default='/home/bio/esox/esox-main/static/models/remora.pt')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.model_file, args.device)
    rds = RemoraDataset(data_dir='.', non_masked_bases=3, seed=0,
                        primary_name='primary', secondary_name=None, inference_mode=True)

    # pack 里 valid/test_* 的来源 split 名: valid<-valid, test_*<-test
    split_of = {'valid': 'valid', 'test_oligo': 'test', 'test_t2t': 'test'}
    cache = {}
    for name in args.sets:
        out = os.path.join(args.out_dir, name + '.probs.npz')
        if os.path.isfile(out):
            log('skip(已存在) ' + out)
            continue
        split = split_of[name]
        if split not in cache:
            files = esox_files(split)
            log('{} split: {} 个 esox 特征文件'.format(split, len(files)))
            frames = []
            t0 = time.time()
            for i, p in enumerate(files):
                df = score_file(p, model, rds, args.device, args.batch_size)
                if df is not None:
                    frames.append(df)
                if (i + 1) % 20 == 0 or i + 1 == len(files):
                    done = sum(len(f) for f in frames)
                    log('  {}/{} 文件, {:,} 位点, {:.0f} sites/s'.format(
                        i + 1, len(files), done, done / max(time.time() - t0, 1e-9)))
            cache[split] = pd.concat(frames, ignore_index=True)

        src = np.load(os.path.join(PACK, name + '.npz'), allow_pickle=False)
        tgt = pd.DataFrame({
            'read_id': np.asarray(src['read_id']).reshape(-1),
            'basecall_pos': np.asarray(src['basecall_pos']).reshape(-1).astype(np.int64),
        })
        merged = tgt.merge(cache[split], on=['read_id', 'basecall_pos'], how='left')
        assert len(merged) == len(tgt), '{}: join 后行数变了({} -> {}), 说明源里有重复键'.format(
            name, len(tgt), len(merged))
        miss = int(merged['esox'].isna().sum())
        log('{}: {:,} 行, 未匹配 {:,} ({:.3%})'.format(name, len(tgt), miss, miss / len(tgt)))
        np.savez(out, prob=merged['esox'].to_numpy(np.float32),
                 label=src['label'].astype(np.int8))
        log('写出 ' + out)
    log('全部完成')


if __name__ == '__main__':
    main()
