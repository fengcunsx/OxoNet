"""Release acceptance test.  CPU only, no raw data needed.

    python smoke_test.py

Three checks, in increasing strength:

1. the published architecture loads the published checkpoint *strictly* and has
   the parameter count reported in the paper;
2. a forward pass produces a probability;
3. the same thing through ``model.loader.build_model`` -- the code path the
   scripts in ``scripts/`` actually use.  Check 3 exists because an earlier
   release passed checks 1-2 while every script in ``scripts/`` was broken:
   the model was never moved to the device and the sequence input was silently
   trimmed to a 5-mer.  Testing the entry point, not just the class, is what
   catches that.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model.model import DetectModel
from model.loader import KMER, build_model, pick_device

CKPT = os.path.join(HERE, 'weights', 'oxonet_seed42_ep125.pth')
N_PARAMS = 2_229_321


def batch(b, device=None):
    return (torch.randn(b, 175, 1, device=device),
            torch.randint(0, 4, (b, KMER), device=device),
            torch.full((b,), 175, dtype=torch.long, device=device))


def main():
    assert KMER == 7, KMER

    # 1 -- architecture and checkpoint agree exactly
    m = DetectModel(dim=128, sig_blocks=4, sig_l=175, seq_l=KMER, pos_mode='rope').eval()
    sd = torch.load(CKPT, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    m.load_state_dict(sd)                       # strict
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == N_PARAMS, n
    print('[1/3] loaded {} trainable parameters (expected {})'.format(
        format(n, ','), format(N_PARAMS, ',')))

    # 2 -- forward pass
    with torch.no_grad():
        _, prob = m(*batch(4))
    assert prob.shape[0] == 4 and 0.0 <= float(prob.min()) and float(prob.max()) <= 1.0
    print('[2/3] forward OK, shape {}, range [{:.3f}, {:.3f}]'.format(
        tuple(prob.shape), float(prob.min()), float(prob.max())))

    # 3 -- the entry point used by scripts/, on CPU
    dev = pick_device(prefer_cuda=False)
    assert str(dev) == 'cpu', dev
    m2, dev2 = build_model(CKPT, prefer_cuda=False)
    assert next(m2.parameters()).device.type == 'cpu'      # was never moved before
    assert not m2.training
    with torch.no_grad():
        _, prob2 = m2(*batch(4, dev2))
    assert prob2.shape == (4, 1), prob2.shape
    print('[3/3] build_model() on {}: strict load, eval mode, {}-mer input, forward OK'.format(
        dev2, KMER))

    print('\nsmoke test passed')


if __name__ == '__main__':
    main()
