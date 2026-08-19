"""只测 esox RemoraModel 的 forward 吞吐, 用于跨 conda 环境对拍。

背景: bench_vs_esox.py 在 base(torch 2.1.2) 里同时 import esox 和 OxoNet。
但 esox 自己的环境是 torch 1.13.1, 版本差两代。若 esox 在自己环境里明显更快/更慢,
那么"在 base 里测出的 esox 数字"就不能代表它的真实性能, 三方对比会失真。
本脚本在任一环境下都能跑, 测法与 bench_vs_esox.py 完全一致(fp32/eval/no_grad/同 BS)。

    <env>/bin/python oxo/analysis/bench_esox_only.py
"""
import glob
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0, '/home/bio/esox/esox-main')

DEV = 'cuda:0'
BS = 1024
SHARD = '/home/bio/8oxog/build/pack/test_oligo.npz'


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def timed(fn, w=10, n=40):
    for _ in range(w):
        fn()
    sync()
    t0 = time.time()
    for _ in range(n):
        fn()
    sync()
    return (time.time() - t0) / n


def main():
    from esox.models.remora import RemoraModel
    from esox.dataclasses import RemoraDataset
    m = RemoraModel(device=DEV, dataset=None, size=256, dropout=0,
                    use_raw=True, use_seq_1=True, use_expected_signal_1=True).to(DEV).eval()
    ck = torch.load('/home/bio/esox/esox-main/static/models/remora.pt', map_location=DEV)
    m.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)
    npar = sum(p.numel() for p in m.parameters())

    # 与 bench_vs_esox.py 逐字相同的取数方式(数据在 NVMe 固态的根分区上)
    rds = RemoraDataset(data_dir='.', non_masked_bases=3, seed=0,
                        primary_name='primary', secondary_name=None, inference_mode=True)
    shard = sorted(glob.glob('/home/bio/8oxog/wtl1/out/*/esox/*.w*.part*.npz'))[0]
    arr = np.load(shard)
    data = {k: arr[k] for k in arr.keys() if k not in ('read_id', 'basecall_pos')}
    n = arr['x'].shape[0]
    idx = list(range(min(BS, n)))
    batch = default_collate([rds.get_data(data, i) for i in idx])
    batch = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}

    fwd = timed(lambda: m.predict_step(batch))
    print(f"env torch {torch.__version__} | {torch.cuda.get_device_name(0)} | bs={BS} fp32/eval/no_grad")
    print(f"esox RemoraModel : {npar/1e6:.3f}M params | forward {BS/fwd:,.0f} sites/s")


if __name__ == '__main__':
    main()
