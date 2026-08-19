"""用训练好的 NanoCon ckpt 给 pack 的 test 集 / native 打分——**一个进程跑完所有集合**。

为什么不走 CSV + 它的 test.py：
  1) 位点集合完全一样才有可比性，而 pack 的 npz 就是 OxoNet 打分用的同一份文件
     （同 read、同 basecall_pos、同顺序），直接吃 npz = 零对齐风险；
  2) 它的 load_dataset 是纯 Python 解析，1.05 KB/行 RSS，native 那 11.4M 位点要 12 GB
     内存 + 12 GB 磁盘 CSV，本机只有 31 GB RAM；
  3) 它的 test_step 每个 pair 做 4 次前向（repre1/repre2 + get_logits×2）是为了算对比损失，
     推理只需要 `cls2(forward(x))` 这一条路径，白给它 ~2× 的速度惩罚不公平。
  4) 本机 GPU 有 Xid 62 坑（CUDA context teardown 触发 PMU 挂起），进程越少越好。

模型/权重/前处理**完全用它原版**：seq = 'CC' + 7mer + 'CC' → 7 个 5-mer token
（`utils.load_dataset`），nano = (7,3) 的 mean/std/dwell（`np.column_stack`，不做归一化），
打分 = `F.softmax(model.get_logits(...), dim=1)[:, 1]`（`NanoCon.test_step`）。
`--verify` 会在 CPU 上拿它自己的 load_dataset/make_data 跑同一批行做逐位点比对。

    python score_nanocon.py --ckpt X.ckpt --out-dir OUT --pack /home/bio/8oxog/build/pack
    python score_nanocon.py --ckpt X.ckpt --out-dir OUT --native-bases 10
"""
import argparse
import glob
import gzip
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

NANOCON_SRC = '/home/bio/bio_seq/NanoCon-main/src'
VOCAB = '/home/bio/bio_seq/NanoCon-main/pretrain/DNAbert_5mer/vocab.txt'
PACK = '/home/bio/8oxog/build/pack'
WT_OUT = '/home/bio/8oxog/wt1/out'


def log(msg):
    print('[{}] {}'.format(time.strftime('%H:%M:%S'), msg), flush=True)


def load_vocab():
    lookup = {}
    with open(VOCAB, encoding='utf-8') as f:
        for i, line in enumerate(f):
            lookup[line.strip()] = i
    return lookup


def tokenize(kmers, lookup):
    """(N,) '<U7' -> (N,7) int64，与 utils.load_dataset 逐字一致。

    先 unique（实际只有 ~1600 种 7-mer）再查表，避免 N 次 Python 循环。
    'o'(8-oxoG) 在 CSV 侧就被换成 'G'（DNABERT 5-mer 词表只有 ACGT），这里同样处理。
    """
    k = np.char.replace(kmers.astype('U7'), 'o', 'G')
    uniq, inv = np.unique(k, return_inverse=True)
    tok_u = np.empty((len(uniq), 7), dtype=np.int64)
    for j, s in enumerate(uniq.tolist()):
        seq = 'C' * 2 + s + 'C' * 2          # 11 字符
        tok_u[j] = [lookup[seq[i:i + 5]] for i in range(7)]
    return tok_u[inv]


def nano_of(a):
    """(N,7,3) float32：mean, std, dwell —— 顺序同 load_dataset 的 column_stack。"""
    return np.stack([a['mean'].astype(np.float32),
                     a['std'].astype(np.float32),
                     a['dwell'].astype(np.float32)], axis=2)


def load_model(ckpt, device):
    sys.path.insert(0, NANOCON_SRC)
    from pytorch_lighting_model.NanoCon import model_encoder
    model = model_encoder.load_from_checkpoint(ckpt, map_location='cpu')
    model.eval().to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log('ckpt {}  可训练参数 {:,}'.format(os.path.basename(ckpt), n_train))
    return model


@torch.no_grad()
def score_arrays(model, tok, nano, device, bs, tag):
    n = tok.shape[0]
    out = np.empty(n, dtype=np.float32)
    t0 = time.time()
    for s in range(0, n, bs):
        e = min(s + bs, n)
        x = torch.from_numpy(tok[s:e]).to(device)
        nd = torch.from_numpy(nano[s:e]).to(device)
        logits = model.get_logits(x, nd)
        out[s:e] = F.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
        if (s // bs) % 200 == 0 and s:
            el = time.time() - t0
            log('  {} {:.1%}  {:.0f} sites/s'.format(tag, e / n, e / max(el, 1e-9)))
    el = time.time() - t0
    log('  {} 完成 {} 位点 {:.0f}s ({:.0f} sites/s)'.format(tag, n, el, n / max(el, 1e-9)))
    return out


def score_npz(model, path, lookup, args, tag):
    a = np.load(path, allow_pickle=False)
    tok = tokenize(a['kmers'], lookup)
    nano = nano_of(a)
    label = a['label'].astype(np.int8) if 'label' in a.files else np.full(tok.shape[0], -1, np.int8)
    prob = score_arrays(model, tok, nano, args.device, args.batch_size, tag)
    return prob, label, a


def run_pack(model, lookup, args):
    for name in args.sets:
        out = os.path.join(args.out_dir, name + '.probs.npz')
        if os.path.isfile(out) and not args.force:
            log('skip(已存在) ' + out)
            continue
        prob, label, _ = score_npz(model, os.path.join(args.pack, name + '.npz'),
                                   lookup, args, name)
        np.savez(out, prob=prob, label=label)
        log('写出 ' + out)


def run_native(model, lookup, args):
    """native 每个 base 目录一个 .tsv.gz(read_id, basecall_pos, prob)——
    与 oxo/analysis 里 OxoNet native 打分的输出格式完全一致，可直接对拍。"""
    nd = os.path.join(args.out_dir, 'native')
    os.makedirs(nd, exist_ok=True)
    if args.native_like:      # 只跑 OxoNet 已打过分的那些 base（同 read 同位点）
        bases = sorted(os.path.basename(p).replace('.tsv.gz', '')
                       for p in glob.glob(os.path.join(args.native_like, '*.tsv.gz')))
    else:
        bases = sorted(os.path.basename(p) for p in glob.glob(os.path.join(WT_OUT, '*')))
    if args.native_bases:
        bases = bases[:args.native_bases]
    log('native: {} 个 base'.format(len(bases)))
    for bi, base in enumerate(bases):
        out = os.path.join(nd, base + '.tsv.gz')
        if os.path.isfile(out) and not args.force:
            log('skip(已存在) ' + out)
            continue
        parts = sorted(glob.glob(os.path.join(WT_OUT, base, 'oxo', '*.npz')))
        t0, n_tot = time.time(), 0
        with gzip.open(out + '.tmp', 'wt') as fh:
            fh.write('read_id\tbasecall_pos\tprob\n')
            for p in parts:
                a = np.load(p, allow_pickle=False)
                if a['kmers'].shape[0] == 0:
                    continue
                tok = tokenize(a['kmers'], lookup)
                prob = score_arrays(model, tok, nano_of(a), args.device,
                                    args.batch_size, base)
                rid = a['read_id']
                bp = np.asarray(a['basecall_pos']).reshape(-1)
                fh.write('\n'.join('{}\t{}\t{:.4f}'.format(r, int(b), float(q))
                                   for r, b, q in zip(rid.tolist(), bp.tolist(), prob.tolist())))
                fh.write('\n')
                n_tot += len(prob)
        os.replace(out + '.tmp', out)
        log('native [{}/{}] {}: {} 位点 {:.0f}s'.format(bi + 1, len(bases), base, n_tot,
                                                       time.time() - t0))


def verify(model, lookup, args):
    """CPU 上用它原版的 load_dataset/make_data 复算同一批行，逐位点比对。

    先用 pack_to_nanocon_csv.py 把 --verify-n 行落成 CSV（和训练数据同一条产线），
    再比 token_ids / nano / 最终 prob。差异只应来自 CSV 的 %.6g 格式化。
    """
    import subprocess
    sys.path.insert(0, NANOCON_SRC)
    from utils import load_dataset, make_data
    csv_path = os.path.join(args.out_dir, 'verify.csv')
    npz = os.path.join(args.pack, args.verify_set + '.npz')
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 'pack_to_nanocon_csv.py'),
                    'eval', '--npz', npz, '--out', csv_path,
                    '--limit', str(args.verify_n), '--seed', '7'], check=True)
    dl = load_dataset(csv_path, kmer_lookup=lookup)
    seq, nano_t, label_t, tok_t, rids, poss = make_data(dl)
    # 我们这条路径：同样的行(--limit 用 seed 7 的 rng.choice+sort 选出来)
    a = np.load(npz, allow_pickle=False)
    rng = np.random.default_rng(7)
    keep = np.sort(rng.choice(a['label'].shape[0],
                              size=min(args.verify_n, a['label'].shape[0]), replace=False))
    tok = tokenize(a['kmers'][keep], lookup)
    nano = nano_of({k: a[k][keep] for k in ('mean', 'std', 'dwell')})
    assert (tok == tok_t.numpy()).all(), 'token_ids 不一致'
    d = np.abs(nano - nano_t.numpy()).max()
    log('token_ids 完全一致; nano 最大差 {:.3g}(=CSV %.6g 舍入)'.format(d))
    cpu = model.to('cpu')
    p_ours = score_arrays(cpu, tok, nano, 'cpu', 1024, 'verify-ours')
    p_theirs = score_arrays(cpu, tok_t.numpy(), nano_t.numpy(), 'cpu', 1024, 'verify-theirs')
    log('prob 最大差 {:.3g}, 相关 {:.8f}'.format(np.abs(p_ours - p_theirs).max(),
                                                np.corrcoef(p_ours, p_theirs)[0, 1]))
    model.to(args.device)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--pack', default=PACK)
    p.add_argument('--sets', nargs='*', default=['valid', 'test_oligo', 'test_t2t'])
    p.add_argument('--native-bases', type=int, default=0, help='>0 则跑 native，取前 N 个 base')
    p.add_argument('--native-like', default='', help='只跑该目录下已有 .tsv.gz 的那些 base')
    p.add_argument('--wt-out', default='', help='native 特征根目录(<root>/<base>/oxo/*.npz); 缺省 wt1/out')
    # 1024 = 它训练/测试时每次前向的实际样本数(argparse 默认 batch 2048, collate 对半分)。
    # 它的 TransformerEncoderLayer 是 batch_first=False 而输入是 (B,7,128) → 注意力其实
    # 跨的是 batch 维；实测分数对 batch 组成近乎不敏感(≤1e-4)，但仍固定成它的原生宽度。
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--device', default='cuda')
    p.add_argument('--force', action='store_true')
    p.add_argument('--verify', action='store_true', help='先做 CPU 逐位点等价性验证')
    p.add_argument('--verify-n', type=int, default=4000)
    p.add_argument('--verify-set', default='test_oligo')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    global WT_OUT
    if args.wt_out:
        WT_OUT = args.wt_out
    lookup = load_vocab()
    model = load_model(args.ckpt, 'cpu' if args.verify else args.device)
    if args.verify:
        verify(model, lookup, args)
    if args.sets:
        run_pack(model, lookup, args)
    if args.native_bases or args.native_like:
        run_native(model, lookup, args)
    log('全部完成')


if __name__ == '__main__':
    main()
