"""同特征对照：21 维摘要统计量到底能撑到多高？

动机（答"NanoCon 是不是被你们训坏了"）：NanoCon 的输入 = 7 个碱基 + 每位置
mean/std/dwell（21 个标量），**没有原始信号**；OxoNet/esox 吃 175 点电流。
如果我们用**完全相同的 21 维特征 + 同一份数据/切分/测试位点**训一个与 NanoCon
毫无关系的模型（梯度提升树），它也停在 NanoCon 附近，那就说明天花板在**特征**，
不在我们对 NanoCon 的处理；OxoNet 高出来的部分 = 原始信号的价值。

这个实验可以两头切：若 GBDT 明显高过 NanoCon，说明 NanoCon 没吃透自己的输入，
那才是我们要担心的事。所以它是个真检验，不是走过场。

    /home/bio/anaconda3/bin/python feature_ceiling.py --out /home/bio/8oxog/nanocon/feature_ceiling.txt
"""
import argparse
import glob
import json
import os
import time

import numpy as np

BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'o': 2}   # 'o'(8-oxoG) 与 NanoCon 侧一样当 G


def log(m):
    print('[{}] {}'.format(time.strftime('%H:%M:%S'), m), flush=True)


def feats(kmers, mean, std, dwell):
    """(N,28): 7 个碱基(类别) + mean7 + std7 + dwell7 —— 与 NanoCon 的输入等价。"""
    n = len(kmers)
    b = np.empty((n, 7), dtype=np.float32)
    uniq, inv = np.unique(kmers.astype('U7'), return_inverse=True)
    tab = np.array([[BASES[c] for c in s] for s in uniq.tolist()], dtype=np.float32)
    b[:] = tab[inv]
    return np.hstack([b, mean.astype(np.float32), std.astype(np.float32),
                      dwell.astype(np.float32)])


def load_eval(path):
    a = np.load(path, allow_pickle=False)
    return feats(a['kmers'], a['mean'], a['std'], a['dwell']), a['label'].astype(np.int8)


def load_train(pack, n_pos, neg_ratio, seed, shuffle=True):
    rng = np.random.default_rng(seed)
    a = np.load(os.path.join(pack, 'pos_train.npz'))
    npos = a['label'].shape[0]
    idx = np.sort(rng.choice(npos, size=min(n_pos, npos), replace=False))
    CODE = np.frombuffer(b'ACGT', dtype=np.uint8)
    km = CODE[np.ascontiguousarray(a['kmers_enc'][idx], dtype=np.uint8)].view('S7').ravel().astype('U7')
    Xp = feats(km, a['mean'][idx], a['std'][idx], a['dwell'][idx])
    log('pos {}'.format(len(Xp)))

    want = int(len(Xp) * neg_ratio)
    ctxs = sorted(d for d in glob.glob(os.path.join(pack, 'neg_pack', '*')) if os.path.isdir(d))
    avail = [np.load(os.path.join(d, 'label.npy'), mmap_mode='r').shape[0] for d in ctxs]
    tot = sum(avail)
    parts = []
    for d, av in zip(ctxs, avail):
        k = min(av, int(round(want * av / tot)))     # 按 ctx 成比例，与 NanoCon 侧同口径
        if k <= 0:
            continue
        sel = np.sort(rng.choice(av, size=k, replace=False))
        enc = np.load(os.path.join(d, 'kmers_enc.npy'), mmap_mode='r')[sel]
        km = CODE[np.ascontiguousarray(enc, dtype=np.uint8)].view('S7').ravel().astype('U7')
        parts.append(feats(km, np.load(os.path.join(d, 'mean.npy'), mmap_mode='r')[sel],
                           np.load(os.path.join(d, 'std.npy'), mmap_mode='r')[sel],
                           np.load(os.path.join(d, 'dwell.npy'), mmap_mode='r')[sel]))
    Xn = np.vstack(parts)
    log('neg {}'.format(len(Xn)))
    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(len(Xp), np.int8), np.zeros(len(Xn), np.int8)])
    if shuffle:      # 树不需要打乱(sklearn 的 validation_fraction 本身随机切),
        p = rng.permutation(len(y))   # 全量负样本时这一步会多占一份 X 的内存
        X, y = X[p], y[p]
    return X, y


def report(name, prob, label, out):
    from sklearn.metrics import roc_auc_score, average_precision_score
    au = roc_auc_score(label, prob)
    ap = average_precision_score(label, prob)
    line = '{:12s} AUROC={:.5f} AUPRC={:.5f}'.format(name, au, ap)
    for g in (1e-2, 1e-3, 1e-4):
        thr = np.quantile(prob[label == 0], 1 - g)
        line += '  rec@FPR{:g}={:.2%}'.format(g, (prob[label == 1] >= thr).mean())
    print(line, flush=True)
    out.append(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pack', default='/home/bio/8oxog/build/pack')
    p.add_argument('--out', default='/home/bio/8oxog/nanocon/feature_ceiling.txt')
    p.add_argument('--n-pos', type=int, default=1_500_000)
    p.add_argument('--neg-ratio', type=float, default=4.0)
    p.add_argument('--max-iter', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-shuffle', action='store_true', help='全量负样本时省一份 X 的内存')
    p.add_argument('--feature-set', choices=['all', 'bases', 'stats'], default='all',
                   help='all=7碱基+21统计量; bases=只有序列; stats=只有 mean/std/dwell。'
                        '用来拆"序列窗口"与"信号统计量"各贡献多少 —— 若 bases 贡献极小, '
                        '把窗口从 7-mer 加宽到 13-mer 也补不上与 OxoNet 的差距。')
    args = p.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    COLS = {'all': slice(0, 28), 'bases': slice(0, 7), 'stats': slice(7, 28)}[args.feature_set]
    X, y = load_train(args.pack, args.n_pos, args.neg_ratio, args.seed, not args.no_shuffle)
    X = np.ascontiguousarray(X[:, COLS])
    log('train {} 行 x {} 特征, 正类 {:.2%}'.format(*X.shape, y.mean()))

    clf = HistGradientBoostingClassifier(
        max_iter=args.max_iter, learning_rate=0.1, max_leaf_nodes=63,
        categorical_features=list(range(7)) if args.feature_set != 'stats' else [],
        early_stopping=True,
        validation_fraction=0.05, n_iter_no_change=20, random_state=args.seed, verbose=1)
    t0 = time.time()
    clf.fit(X, y)
    log('训练完成 {:.0f}s, 用了 {} 轮'.format(time.time() - t0, clf.n_iter_))
    del X, y

    out = ['GBDT 特征子集={} (同 pack/同切分/同测试位点)'.format(args.feature_set),
           '训练: {} pos x 1:{:g} neg, {} 轮'.format(args.n_pos, args.neg_ratio, clf.n_iter_)]
    for s in ['valid', 'test_oligo', 'test_t2t']:
        Xe, ye = load_eval(os.path.join(args.pack, s + '.npz'))
        prob = clf.predict_proba(np.ascontiguousarray(Xe[:, COLS]))[:, 1]
        np.savez(os.path.join(os.path.dirname(args.out),
                              'gbdt_{}_{}.probs.npz'.format(args.feature_set, s)),
                 prob=prob.astype(np.float32), label=ye)
        report(s, prob, ye, out)
        del Xe, ye, prob
    with open(args.out, 'w') as f:
        f.write('\n'.join(out) + '\n')
    log('写出 ' + args.out)


if __name__ == '__main__':
    main()
