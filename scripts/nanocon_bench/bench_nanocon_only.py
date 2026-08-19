"""只测 NanoCon 的 forward 吞吐, 测法与 oxo/analysis/bench_vs_esox.py 完全一致
(fp32 / eval / no_grad / 同 batch / warmup 后 cuda.synchronize 计时)。

公平推理路径 = `model.get_logits(seq, nano)`, 与 score_nanocon.py:90 打分时用的**同一条**。
**不要**用 test_step: 那里每个 pair 做 4 次前向(repre1/repre2 + get_logits x2), 是为对比
损失服务的, 会白给 2x 惩罚。

batch 固定 1024: NanoCon 的 TransformerEncoderLayer 是 batch_first=False, 输入 (B,7,128)
被当作 (L=B, N=7), 注意力矩阵是 B*B, 4096 会 OOM。三方统一 1024 才可比。

数据取自 build/pack(根分区 = NVMe 固态)。计时循环内没有磁盘 IO: batch 建一次反复喂。

    /home/bio/anaconda3/envs/NanoCon/bin/python bench_nanocon_only.py
"""
import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/home/bio/bio_seq/nanocon_bench')
import score_nanocon as sn          # 复用它的 tokenize / nano_of / load_vocab, 保证逐字一致

DEV = 'cuda:0'
BS = 1024
PACK = '/home/bio/8oxog/build/pack/test_oligo.npz'


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
    sys.path.insert(0, sn.NANOCON_SRC)
    from pytorch_lighting_model.NanoCon import model_encoder
    ckpt = sorted(glob.glob('/home/bio/8oxog/nanocon/*.ckpt'))[0]
    model = model_encoder.load_from_checkpoint(ckpt, map_location='cpu').to(DEV).eval()
    npar = sum(p.numel() for p in model.parameters())

    a = np.load(PACK, allow_pickle=False)
    lookup = sn.load_vocab()
    tok = sn.tokenize(a['kmers'][:BS], lookup)
    nano = sn.nano_of({k: a[k][:BS] for k in ('mean', 'std', 'dwell')})
    x = torch.from_numpy(tok).to(DEV)
    nd = torch.from_numpy(nano).to(DEV)

    fwd = timed(lambda: model.get_logits(x, nd))
    print(f"env torch {torch.__version__} | {torch.cuda.get_device_name(0)} | bs={BS} fp32/eval/no_grad")
    print(f"NanoCon : {npar/1e6:.3f}M params | forward {BS/fwd:,.0f} sites/s")


if __name__ == '__main__':
    main()
