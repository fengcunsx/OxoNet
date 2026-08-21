"""最小验收: 实例化 7-mer 模型、严格加载已发布权重、跑一个 batch 的 forward。

    python smoke_test.py
不需要 GPU、不需要原始数据。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model.model import DetectModel

CKPT = os.path.join(HERE, 'weights', 'oxonet_seed42_ep125.pth')


def main():
    m = DetectModel(dim=128, sig_blocks=4, sig_l=175, seq_l=7, pos_mode='rope').eval()
    sd = torch.load(CKPT, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    m.load_state_dict(sd)                       # strict: 结构必须与权重完全一致
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == 2_229_321, n
    print('loaded {} trainable parameters (expected 2,229,321)'.format(format(n, ',')))

    b = 4
    sig = torch.randn(b, 175, 1)
    seq = torch.randint(0, 4, (b, 7))
    sig_l = torch.full((b,), 175, dtype=torch.long)
    with torch.no_grad():
        _, prob = m(sig, seq, sig_l)
    assert prob.shape[0] == b and float(prob.min()) >= 0 and float(prob.max()) <= 1
    print('forward OK, output shape {}, range [{:.3f}, {:.3f}]'.format(
        tuple(prob.shape), float(prob.min()), float(prob.max())))
    print('\nsmoke test passed')


if __name__ == '__main__':
    main()
