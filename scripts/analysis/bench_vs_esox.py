"""同环境(base)、同测法对比 OxoNet(DetectModel) vs esox(RemoraModel) 推理吞吐。
fp32 / eval / no_grad / 同 GPU。报告: 纯 forward(位点/s) 和 端到端(get_data+forward)。
用法: PYTHONPATH=/home/bio/esox/esox-main:/home/bio/bio_seq/100_retrain python oxo/analysis/bench_vs_esox.py
"""
import os, sys, time, glob
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0, '/home/bio/esox/esox-main')
sys.path.insert(0, '/home/bio/bio_seq/100_retrain')

DEV = 'cuda:0'
BS = 1024   # 与 NanoCon 对齐(它 batch_first=False, 4096 会 OOM), 三方同 batch 才可比


def sync(): torch.cuda.synchronize()


@torch.no_grad()
def timed(fn, n=30, w=10):
    for _ in range(w): fn()
    sync(); s = time.perf_counter()
    for _ in range(n): fn()
    sync(); return (time.perf_counter() - s) / n


# ---------- esox ----------
def bench_esox():
    from esox.models.remora import RemoraModel
    from esox.dataclasses import RemoraDataset
    m = RemoraModel(device=DEV, dataset=None, size=256, dropout=0,
                    use_raw=True, use_seq_1=True, use_expected_signal_1=True).to(DEV).eval()
    ck = torch.load('/home/bio/esox/esox-main/static/models/remora.pt', map_location=DEV)
    m.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)
    rds = RemoraDataset(data_dir='.', non_masked_bases=3, seed=0,
                        primary_name='primary', secondary_name=None, inference_mode=True)
    shard = sorted(glob.glob('/home/bio/8oxog/wtl1/out/*/esox/*.w*.part*.npz'))[0]
    arr = np.load(shard)
    data = {k: arr[k] for k in arr.keys() if k not in ('read_id', 'basecall_pos')}
    n = arr['x'].shape[0]
    idx = list(range(min(BS, n)))
    batch = default_collate([rds.get_data(data, i) for i in idx])
    batch = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
    npar = sum(p.numel() for p in m.parameters())

    fwd = timed(lambda: m.predict_step(batch))
    # 端到端: 每次重建 batch (get_data) + forward
    def e2e():
        b = default_collate([rds.get_data(data, i) for i in idx])
        b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
        m.predict_step(b)
    ee = timed(e2e, n=20, w=5)
    return npar, len(idx) / fwd, len(idx) / ee


# ---------- ours ----------
def bench_ours():
    from model.model import DetectModel
    # seq_l 必须 = 7: 训练用的是 DetectModel(..., seq_l=7, pos_mode='rope')
    # (100_retrain/script/train.py:121)。seq_l=5 是**从未训练过**的更小架构
    # (2.164M vs 真实 2.229M), 用它测吞吐等于测了个不存在的模型。
    m = DetectModel(dim=128, sig_blocks=4, sig_l=175, seq_l=7, pos_mode='rope').to(DEV).eval()
    sig = torch.randn(BS, 175, 1, device=DEV)
    seq = torch.randint(0, 4, (BS, 7), device=DEV)   # 与 seq_l=7 匹配
    sl = torch.full((BS,), 175, dtype=torch.long, device=DEV)
    npar = sum(p.numel() for p in m.parameters())
    fwd = timed(lambda: m(sig, seq, sl))
    return npar, BS / fwd


if __name__ == '__main__':
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  bs={BS}  fp32/eval/no_grad\n")
    ep, ef, ee = bench_esox()
    op, of = bench_ours()
    print(f"esox  RemoraModel : {ep/1e6:.3f}M params | forward {ef:,.0f} sites/s | 端到端(get_data+fwd) {ee:,.0f} sites/s")
    print(f"ours  DetectModel : {op/1e6:.3f}M params | forward {of:,.0f} sites/s")
    print(f"\nforward 速度比 (ours/esox): {of/ef:.2f}x  ({'我们快' if of>ef else 'esox快'})")
