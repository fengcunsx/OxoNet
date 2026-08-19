"""Homopolymer 分层: 中心 G 落在多长的连续 G 段里, 各方法表现如何。

直接答 R2.M1 举的例子 —— "a guanine homopolymer within a repetitive element may
behave quite differently under native genomic sequencing conditions than in an
isolated oligo" —— 以及 R1.2 (homopolymer 误差段是亮点, 加一句未来改进)。

在 **test 集上做**(有标签 → 能出 recall 和 FPR); native 侧没有 ground truth,
只能报判正率, 由 native_strat.py 负责。

每个模型用**自己的 T\***(valid 阴性上 99.9% 特异性), 因为分数尺度不可比、工作点才可比。

    /home/bio/anaconda3/bin/python homopolymer_strat.py
"""
import argparse
import os

import numpy as np

PACK = '/home/bio/8oxog/build/pack'
MODELS = {
    'OxoNet': '/home/bio/8oxog/wtl1/pack_scores_e2ep125',
    'esox': '/home/bio/8oxog/wtl1/pack_scores_esox',
    'NanoCon': '/home/bio/8oxog/nanocon/scores',
    'GBDT(特征天花板)': '/home/bio/8oxog/nanocon',      # gbdt_all_<set>.probs.npz
}


def probs_path(d, name):
    # gbdt_<set>.probs.npz = 最强那版天花板模型(全 pos, 1:8, 986 轮早停)
    for cand in (os.path.join(d, name + '.probs.npz'),
                 os.path.join(d, 'gbdt_' + name + '.probs.npz')):
        if os.path.isfile(cand):
            return cand
    raise SystemExit('找不到 {} 的 {} 打分'.format(d, name))


def grun(kmers):
    """中心(index 3)所在的连续 G 段长度。'o'(8-oxoG) 算 G。"""
    out = np.empty(len(kmers), np.int8)
    for i, s in enumerate(kmers):
        s = s.replace('o', 'G')
        l = r = 3
        while l > 0 and s[l - 1] == 'G':
            l -= 1
        while r < 6 and s[r + 1] == 'G':
            r += 1
        out[i] = r - l + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=float, default=0.999)
    ap.add_argument('--out', default='/home/bio/8oxog/wtl1/homopolymer_strat.txt')
    args = ap.parse_args()

    lines = ['Homopolymer 分层 (中心 G 所在连续 G 段长度; 每个模型用自己 T* @ valid 阴性 {:.1%} 特异性)'
             .format(args.spec)]

    # 每个模型的 T*
    T = {}
    for m, d in MODELS.items():
        v = np.load(probs_path(d, 'valid'))
        vp = np.nan_to_num(v['prob'].astype(np.float64), nan=-1.0)
        T[m] = float(np.quantile(vp[v['label'] == 0], args.spec))
        lines.append('  T*[{}] = {:.6f}'.format(m, T[m]))

    for name in ('test_oligo', 'test_t2t'):
        src = np.load(os.path.join(PACK, name + '.npz'), allow_pickle=False)
        g = grun(src['kmers'].astype('U7').tolist())
        lab = src['label'].astype(int)
        g = np.minimum(g, 4)                     # 4 = "4 及以上"
        lines.append('')
        lines.append('=== {} ==='.format(name))
        lines.append('{:18s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
            'G 段长度', '1', '2', '3', '4+'))
        lines.append('{:18s} {:>10,} {:>10,} {:>10,} {:>10,}'.format(
            '阳性位点数', *[int(((g == k) & (lab == 1)).sum()) for k in (1, 2, 3, 4)]))
        lines.append('{:18s} {:>10,} {:>10,} {:>10,} {:>10,}'.format(
            '阴性位点数', *[int(((g == k) & (lab == 0)).sum()) for k in (1, 2, 3, 4)]))
        for m, d in MODELS.items():
            p = np.nan_to_num(np.load(probs_path(d, name))['prob'].astype(np.float64), nan=-1.0)
            call = p >= T[m]
            rec = ['{:9.2%}'.format(call[(g == k) & (lab == 1)].mean()) for k in (1, 2, 3, 4)]
            fpr = ['{:9.3%}'.format(call[(g == k) & (lab == 0)].mean()) for k in (1, 2, 3, 4)]
            r1 = call[(g == 1) & (lab == 1)].mean()
            r4 = call[(g == 4) & (lab == 1)].mean()
            lines.append('{:18s} {} <- recall  (4+ / 1 = {:.0%} 保留)'.format(
                m, ' '.join(rec), r4 / max(r1, 1e-12)))
            lines.append('{:18s} {} <- FPR'.format('', ' '.join(fpr)))

    txt = '\n'.join(lines)
    print(txt)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    print('\n写出', args.out)


if __name__ == '__main__':
    main()
