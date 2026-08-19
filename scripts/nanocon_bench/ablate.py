"""NanoCon 输入消融：它到底在用哪一路信息？

它的输入只有 (7,3) 的 mean/std/dwell + 7 个 5-mer token——**没有原始信号**
（OxoNet/esox 吃 175 点电流）。把每一路单独敲掉，看判别力掉多少，就知道
0.93 的 AUROC 是靠什么撑起来的、以及"少了原始信号"值多少分。

敲法沿用它原版 `utils.load_dataset` 的 mask 语义（mask=0 均值置零 / 1 标准差 / 2 dwell），
序列那一路它原版是 'C'*15（会把 token 数从 7 变成 11，改了几何），我们改成
'CC'+'CCCCCCC'+'CC' —— 同样是常量序列但保持 7 个位置，只用于消融、不进 benchmark。

    /home/bio/anaconda3/envs/NanoCon/bin/python ablate.py --ckpt X.ckpt
"""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_nanocon import load_vocab, tokenize, nano_of, load_model, score_arrays, log


def metrics(prob, label):
    au = roc_auc_score(label, prob)
    ap = average_precision_score(label, prob)
    thr = np.quantile(prob[label == 0], 0.999)
    rec = float((prob[label == 1] >= thr).mean())
    return au, ap, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--npz', default='/home/bio/8oxog/build/pack/test_oligo.npz')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=1024)
    args = p.parse_args()

    lookup = load_vocab()
    model = load_model(args.ckpt, args.device)
    a = np.load(args.npz, allow_pickle=False)
    label = a['label'].astype(np.int8)
    tok = tokenize(a['kmers'], lookup)
    nano = nano_of(a)
    const_tok = tokenize(np.array(['CCCCCCC'] * 1), lookup).repeat(len(tok), 0)

    variants = {
        '全部输入':        (tok, nano),
        '去掉 mean':       (tok, nano.copy()),
        '去掉 std':        (tok, nano.copy()),
        '去掉 dwell':      (tok, nano.copy()),
        '去掉序列(常量)':  (const_tok, nano),
        '只留序列':        (tok, np.zeros_like(nano)),
        '只留 dwell':      (const_tok, nano.copy()),
    }
    variants['去掉 mean'][1][:, :, 0] = 0
    variants['去掉 std'][1][:, :, 1] = 0
    variants['去掉 dwell'][1][:, :, 2] = 0
    variants['只留 dwell'][1][:, :, :2] = 0

    print('{:18s} {:>9s} {:>9s} {:>14s}'.format('变体', 'AUROC', 'AUPRC', 'recall@FPR1e-3'))
    for name, (t, n) in variants.items():
        prob = score_arrays(model, np.ascontiguousarray(t), np.ascontiguousarray(n),
                            args.device, args.batch_size, name)
        au, ap, rec = metrics(prob, label)
        print('{:18s} {:9.5f} {:9.5f} {:13.2%}'.format(name, au, ap, rec), flush=True)


if __name__ == '__main__':
    main()
