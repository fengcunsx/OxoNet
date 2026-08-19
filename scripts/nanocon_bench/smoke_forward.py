"""CPU 冒烟: 用 NanoCon 原版的 loader/collate/模型跑几个 batch, 验证我们造的 CSV 能直接喂进去。
(不改 NanoCon 代码; 它的 Trainer 在 CPU 上 devices=[] 会报错, 所以这里手动跑 step 的等价计算。)

用法: python smoke_forward.py --csv train.csv [--batch-size 512] [--steps 3]
"""
import argparse
import sys

SRC = '/home/bio/bio_seq/NanoCon-main/src'
sys.path.insert(0, SRC)

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import build_kmer_lookup, load_dataset, make_data, MyDataSet
from train import collate
from pytorch_lighting_model.NanoCon import model_encoder, ContrastiveLoss

p = argparse.ArgumentParser()
p.add_argument('--csv', required=True)
p.add_argument('--batch-size', type=int, default=512)
p.add_argument('--steps', type=int, default=3)
p.add_argument('--limit', type=int, default=20000)
a = p.parse_args()

lk = build_kmer_lookup(f'{SRC}/../pretrain/DNAbert_5mer/vocab.txt')
d = load_dataset(a.csv, kmer_lookup=lk)
seq, nano, lab, tok, rid, pos = make_data(d)
seq, nano, lab, tok = seq[:a.limit], nano[:a.limit], lab[:a.limit], tok[:a.limit]
ds = MyDataSet(seq, nano, lab, tok)
dl = DataLoader(ds, a.batch_size, True, collate_fn=collate)

model = model_encoder()
model.train()
crit, con = nn.CrossEntropyLoss(), ContrastiveLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
print(f'params = {sum(p.numel() for p in model.parameters()):,}')

for i, batch in enumerate(dl):
    if i >= a.steps:
        break
    t1, t2, n1, n2, label, l1, l2, *_ = batch
    r1, r2 = model(t1, n1), model(t2, n2)
    g1, g2 = model.get_logits(t1, n1), model.get_logits(t2, n2)
    loss = 10 * con(r1, r2, label) + crit(g1, l1) + crit(g2, l2)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f'step {i}: repre={tuple(r1.shape)} logits={tuple(g1.shape)} '
          f'pair_pos_frac={label.float().mean():.3f} loss={loss.item():.4f}')

print('OK: CSV → NanoCon 原版 loader/collate/模型 全链路通')
