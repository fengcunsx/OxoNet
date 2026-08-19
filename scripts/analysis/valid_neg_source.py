"""从 valid 的阴性里分离出"基因组来源"(T2T)那一部分, 并验证这个判据准不准。

动机: 工作点(阈值)应当只在 valid 上确定, 测试集只用来报结果 —— 否则会被质疑
"操作点是在测试集上选的"。但 valid 的阴性是 oligo + T2T 混合, 而我们要的是**纯基因组** FPR。
pack 里没有来源字段, T2T 源文件本地已归档删除 → 用"每条 read 的位点数"分离:
基因组 read 长、每条贡献几百个 G; oligo read 短、每条只有个位数。

判据在**已知答案的两个测试集上验证**(test_oligo 阴性全是 oligo, test_t2t 阴性全是基因组),
给出位点级的误判率, 再套用到 valid。

    /home/bio/anaconda3/bin/python valid_neg_source.py
"""
import argparse
import os

import numpy as np

PACK = '/home/bio/8oxog/build/pack'
OUT_MASK = os.path.join(PACK, 'valid_genomic_neg_mask.npy')


def sites_per_read(read_id):
    u, inv, c = np.unique(read_id, return_inverse=True, return_counts=True)
    return c[inv]          # 每一行所属 read 的总位点数


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cut', type=int, default=100, help='每 read 位点数 > cut 判为基因组')
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/valid_neg_source.txt')
    args = ap.parse_args()
    L = ['valid 阴性来源分离 (判据: 每条 read 的位点数 > {})'.format(args.cut)]

    # ---- 1. 在已知答案的测试集上验证判据 ----
    L.append('')
    L.append('=== 1. 判据验证(测试集上有ground truth) ===')
    err = {}
    for name, truth in (('test_oligo', 'oligo'), ('test_t2t', '基因组')):
        d = np.load(os.path.join(PACK, name + '.npz'), allow_pickle=False)
        neg = d['label'].astype(int) == 0
        spr = sites_per_read(d['read_id'][neg])
        pred_genomic = spr > args.cut
        if truth == 'oligo':      # 全是 oligo, 被判成基因组的都是错的
            e = float(pred_genomic.mean())
            L.append('  {:11s}(全 oligo 阴性 {:>10,} 个位点): 误判为基因组 {:>9,} 个 = {:.4%}'
                     .format(name, int(neg.sum()), int(pred_genomic.sum()), e))
        else:                     # 全是基因组, 没被判出来的是漏的
            e = float((~pred_genomic).mean())
            L.append('  {:11s}(全基因组阴性 {:>10,} 个位点): 漏判(算成 oligo) {:>9,} 个 = {:.4%}'
                     .format(name, int(neg.sum()), int((~pred_genomic).sum()), e))
        err[name] = e
    L.append('  → 位点级: 混入率 {:.4%}(oligo 被当基因组), 漏检率 {:.4%}(基因组被当 oligo)'
             .format(err['test_oligo'], err['test_t2t']))
    L.append('  混入率决定阈值会不会被污染(要极低); 漏检率只影响样本量(可接受)')

    # ---- 2. 套用到 valid ----
    v = np.load(os.path.join(PACK, 'valid.npz'), allow_pickle=False)
    vlab = v['label'].astype(int)
    vneg = vlab == 0
    spr = np.zeros(len(vlab), np.int64)
    spr[vneg] = sites_per_read(v['read_id'][vneg])
    genomic = vneg & (spr > args.cut)
    L.append('')
    L.append('=== 2. 套用到 valid ===')
    L.append('  valid 总行数 {:,}: 阳性 {:,} / 阴性 {:,}'.format(
        len(vlab), int((vlab == 1).sum()), int(vneg.sum())))
    L.append('  判为**基因组阴性**: {:,} 个位点 ({:.1%} 的阴性), 来自 {:,} 条 read'.format(
        int(genomic.sum()), genomic.sum() / vneg.sum(), len(np.unique(v['read_id'][genomic]))))
    L.append('  (build_split 时 neg 里 T2T 约占 75% → 与 {:.1%} 吻合, 互为旁证)'
             .format(genomic.sum() / vneg.sum()))
    L.append('  可分辨的最小 FPR = 1/{:,} = {:.2e}'.format(
        int(genomic.sum()), 1.0 / int(genomic.sum())))
    np.save(OUT_MASK, genomic)
    L.append('  掩码已存: {}'.format(OUT_MASK))

    # ---- 3. 换阈值来源: valid 基因组阴性 vs test_t2t 阴性 ----
    MODELS = {
        'OxoNet': '/home/bio/8oxog/wtl1/pack_scores_e2ep125',
        'esox': '/home/bio/8oxog/wtl1/pack_scores_esox',
        'NanoCon': '/home/bio/8oxog/nanocon/scores',
        'GBDT(特征天花板)': '/home/bio/8oxog/nanocon',
    }

    def pp(d, n):
        for c in (os.path.join(d, n + '.probs.npz'), os.path.join(d, 'gbdt_' + n + '.probs.npz')):
            if os.path.isfile(c):
                return c
        raise SystemExit('缺 ' + n)

    t = np.load(os.path.join(PACK, 'test_t2t.npz'), allow_pickle=False)
    tlab = t['label'].astype(int)
    L.append('')
    L.append('=== 3. 阈值来源对比: valid 基因组阴性(新) vs test_t2t 阴性(旧) ===')
    L.append('  {:14s} {:>7s} {:>9s} {:>9s} {:>10s} {:>10s} {:>9s}'.format(
        'model', 'FPR', 'thr(valid)', 'thr(test)', 'rec(valid定)', 'rec(test定)', '差'))
    for m, d in MODELS.items():
        pv = np.nan_to_num(np.load(pp(d, 'valid'))['prob'].astype(np.float64), nan=-1.0)
        pt = np.nan_to_num(np.load(pp(d, 'test_t2t'))['prob'].astype(np.float64), nan=-1.0)
        vneg_p, tneg_p, tpos_p = pv[genomic], pt[tlab == 0], pt[tlab == 1]
        for g in (1e-3, 1e-4, 1e-5):
            tv = float(np.quantile(vneg_p, 1 - g))
            tt = float(np.quantile(tneg_p, 1 - g))
            rv = float((tpos_p >= tv).mean())
            rt = float((tpos_p >= tt).mean())
            L.append('  {:14s} {:7.0e} {:9.5f} {:9.5f} {:10.2%} {:10.2%} {:8.2f}pp'
                     .format(m, g, tv, tt, rv, rt, 100 * (rv - rt)))
    L.append('  → 差值小 = 两种定法等价, 换成 valid 纯属防守(消除"操作点选在测试集上"的质疑)')

    txt = '\n'.join(L)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print('\n写出', args.out)


if __name__ == '__main__':
    main()
