"""从已发布的 predictions + manifests 重算论文中的表 —— 不需要 GPU、不需要权重、不需要原始数据。

    python reproduce_tables.py

复现范围: Table 8(复现性, 3 seeds) 与 Table 7(消融) 的 recall 列。
Table 2-4 需要 esox / NanoCon 的预测, 见 README 中各基线的说明。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TH = os.path.join(HERE, 'manifests', 'valid_thresholds.json')
PRED = os.path.join(HERE, 'predictions')
SITES = os.path.join(HERE, 'manifests', 'sites_test_t2t.csv.gz')


def labels():
    """从发布的位点清单读 label, 与 predictions 逐行对齐。"""
    import gzip
    with gzip.open(SITES, 'rt') as f:
        next(f)
        return np.fromiter((int(l.rsplit(',', 1)[1]) for l in f), dtype=np.int8)


def main():
    if not os.path.exists(TH):
        sys.exit('missing ' + TH)
    th = json.load(open(TH))
    lab = labels()
    pos = lab == 1
    print('test set: {:,} sites ({:,} positive)\n'.format(len(lab), pos.sum()))

    arms = [('full_seed42_ep125', 'Table 7 full model'),
            ('full_seed42_ep149', 'Table 8 seed 42'),
            ('seed0_ep149', 'Table 8 seed 0'),
            ('seed3407_ep149', 'Table 8 seed 3407'),
            ('ab_conv_ep149', 'Table 7 -multi-scale'),
            ('ab_mha_ep149', 'Table 7 -attention'),
            ('ab_deform_ep149', 'Table 7 -deformable'),
            ('ab_allsig_ep149', 'Table 7 -all three'),
            ('ab_seq_ep149', 'Table 7 -Seq-Net')]

    print('{:22} {:>10} {:>10} {:>10}   {}'.format('arm', '1e-3', '1e-4', '1e-5', 'note'))
    for key, note in arms:
        f = os.path.join(PRED, 'probs_{}_test_t2t.npy'.format(key))
        if not (os.path.exists(f) and key in th):
            print('{:22} {:>34}   {}'.format(key, '(not released)', note))
            continue
        p = np.load(f)[pos]
        row = ['{:9.2f}%'.format(100 * (p >= th[key]['T@' + k]).mean())
               for k in ('1e-03', '1e-04', '1e-05')]
        print('{:22} {} {} {}   {}'.format(key, *row, note))


if __name__ == '__main__':
    main()
