"""把 esox 特征按 **oxo pack 的行序** 抽出来，做成自足的 esox pack。

为什么要它：`build/pack` 里只有 oxo 侧特征（signal175/mean/std/dwell/kmers），
**没有 esox 的 100 点窗口特征**（x/e1/s1），后者只散落在 `data/batch*/feature/esox/`
和 `t2t/*/esox/`。做完这个 pack，`build/` 就对四个方法都自足了 —— 原始目录归档到
网盘后，任何分析（含重新给 esox 打分）都不必再取回来。

**切分绝对不重算**：行序直接取自 `build/pack/<set>.npz` 的 (read_id, basecall_pos)，
源文件按同一个 `split_manifest.json` 挑选，最后左连接回 pack 的行序，并做三条硬断言：
行数一致、未匹配为 0、(read_id, basecall_pos) 逐元素相等。
（`score_esox_pack.py` 昨晚已实测三个集合未匹配率均为 0.000%。）

**范围**：只做 valid / test_oligo / test_t2t（评测三件套）。**不做 train** ——
train 有 4834 万行，esox 特征要 ~40 GB，而我们不训练 esox 模型，train 侧没有用途。

    /home/bio/anaconda3/bin/python build_esox_pack.py
"""
import argparse
import os

import numpy as np
import pandas as pd

from score_esox_pack import esox_files, normalize, PACK

OUT = '/home/bio/8oxog/build/pack_esox'
SPLIT_OF = {'valid': 'valid', 'test_oligo': 'test', 'test_t2t': 'test'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', default=['valid', 'test_oligo', 'test_t2t'])
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cache = {}
    for name in args.sets:
        out = os.path.join(args.out, name + '.npz')
        if os.path.isfile(out):
            print('skip(已存在)', out, flush=True)
            continue

        split = SPLIT_OF[name]
        if split not in cache:
            files = esox_files(split)
            print('{} split: {} 个 esox 特征文件, 载入中...'.format(split, len(files)), flush=True)
            rid, bp, xs, es, ss = [], [], [], [], []
            for i, p in enumerate(files):
                a = np.load(p, allow_pickle=False)
                if a['read_id'].shape[0] == 0:
                    continue
                d = normalize(a)          # 统一两种 schema -> x/e1/s1/p1
                rid.append(np.asarray(a['read_id']).reshape(-1).astype('U36'))
                bp.append(np.asarray(a['basecall_pos']).reshape(-1).astype(np.int64))
                xs.append(d['x']); es.append(d['e1']); ss.append(d['s1'])
                if (i + 1) % 20 == 0 or i + 1 == len(files):
                    print('  {}/{} 文件, {:,} 行'.format(
                        i + 1, len(files), sum(len(r) for r in rid)), flush=True)
            cache[split] = (np.concatenate(rid), np.concatenate(bp),
                            np.concatenate(xs), np.concatenate(es), np.concatenate(ss))

        s_rid, s_bp, s_x, s_e, s_s = cache[split]
        src = np.load(os.path.join(PACK, name + '.npz'), allow_pickle=False)
        t_rid = np.asarray(src['read_id']).reshape(-1).astype('U36')
        t_bp = np.asarray(src['basecall_pos']).reshape(-1).astype(np.int64)

        # 左连接: 把源行号映射到 pack 的行序
        left = pd.DataFrame({'read_id': t_rid, 'basecall_pos': t_bp,
                             'row': np.arange(len(t_rid))})
        right = pd.DataFrame({'read_id': s_rid, 'basecall_pos': s_bp,
                              'srow': np.arange(len(s_rid))})
        m = left.merge(right, on=['read_id', 'basecall_pos'], how='left')

        assert len(m) == len(left), '{}: join 后行数变了 ({} -> {}) —— 源里有重复键'.format(
            name, len(left), len(m))
        miss = int(m['srow'].isna().sum())
        assert miss == 0, '{}: {} 行未匹配, 不能出 pack'.format(name, miss)
        idx = m['srow'].to_numpy(np.int64)
        assert (s_rid[idx] == t_rid).all() and (s_bp[idx] == t_bp).all(), \
            '{}: 逐元素校验失败'.format(name)

        np.savez_compressed(out, x=s_x[idx], e1=s_e[idx], s1=s_s[idx],
                            read_id=t_rid, basecall_pos=t_bp,
                            label=src['label'].astype(np.int8))
        print('{}: {:,} 行, 未匹配 0, 逐元素校验通过 -> {} ({:.2f} GB)'.format(
            name, len(t_rid), out, os.path.getsize(out) / 2**30), flush=True)

    with open(os.path.join(args.out, 'README.md'), 'w') as f:
        f.write("""# esox pack

`build/pack` 的 esox 侧配套：x / e1 / s1（100 点窗口），**行序与 `build/pack/<set>.npz` 完全一致**
（同 read_id、同 basecall_pos、同顺序，建包时逐元素断言过）。

- 覆盖 **valid / test_oligo / test_t2t**；**没有 train**（train 4834 万行、esox 特征约 40 GB，
  而我们不训练 esox 模型）。
- **没有 p1**（phredq）：`RemoraModel(use_phredq_1=False)` 不消费它；`RemoraDataset.get_data`
  会读 `p1`，用 `np.zeros_like(s1)` 补即可（与我们打分时的做法一致）。
- oligo 侧源文件的 raw 字段叫 `x1`、t2t 侧叫 `x`，建包时已统一成 `x`；dtype 统一为
  float32/float32/int8。
- 用途：原始 `data/batch*` 与 `t2t/` 归档到网盘后，仍可在本地重新给 esox 打分、
  重做任何评测分析。

生成脚本：`oxo/analysis/build_esox_pack.py`
""")
    print('全部完成 ->', args.out)


if __name__ == '__main__':
    main()
