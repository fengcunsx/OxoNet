"""验证 T2T part 切分的两个前提(只读):
1) 一条 read 是否只出现在一个 part(=按part切不泄漏)?
2) 同名 part 的 oxo 与 esox 的 (read_id, basecall_pos) 是否逐行一致(=esox/oxonet valid/test站点自动相同)?
"""
import glob, os
from collections import defaultdict
import numpy as np

T2T = '/home/bio/8oxog/t2t'


def f5_groups():
    g = defaultdict(list)  # (source,f5) -> [oxo npz paths]
    for f in glob.glob(f'{T2T}/*/**/oxo/*.npz', recursive=True):
        source = f.split('/t2t/')[1].split('/')[0]
        parts = f.split('/'); idx = parts.index('oxo'); f5 = parts[idx - 2]
        g[(source, f5)].append(f)
    return g


def main():
    groups = f5_groups()
    print(f"T2T f5 组数: {len(groups)}; 总 oxo part: {sum(len(v) for v in groups.values())}")

    # 1) read 是否跨 part (在每个 f5 内检查)
    worst = 0; worst_where = None; checked_f5 = 0
    for (s, f5), paths in groups.items():
        read2parts = defaultdict(set)
        for p in paths:
            rid = np.load(p)['read_id']
            pn = os.path.basename(p)
            for r in np.unique(rid):
                read2parts[r].add(pn)
        mx = max((len(v) for v in read2parts.values()), default=0)
        if mx > worst: worst, worst_where = mx, (s, f5)
        checked_f5 += 1
    print(f"\n[1] read跨part检查(全{checked_f5}个f5): 单条read出现的最多part数 = {worst} "
          f"{'✓ 无read跨part(按part切安全)' if worst<=1 else '⚠️有read跨part!'} @ {worst_where}")

    # 2) oxo vs esox 同名 part 的 (read_id, basecall_pos) 一致性 (抽查几个f5的所有part)
    print("\n[2] oxo↔esox 同名part (read_id,basecall_pos) 一致性抽查:")
    n_ok = n_bad = 0; sample = 0
    for (s, f5), paths in list(groups.items())[:5]:
        for oxo_p in paths:
            esox_p = oxo_p.replace('/oxo/', '/esox/')
            if not os.path.isfile(esox_p):
                print(f"  缺 esox 对应: {os.path.basename(oxo_p)}"); n_bad += 1; continue
            ao = np.load(oxo_p); ae = np.load(esox_p)
            # basecall_pos 形状 oxo=(N,1)/esox=(N,), 用 ravel 比内容; 站点集合一致才算 same
            same = (np.array_equal(ao['read_id'], ae['read_id'])
                    and np.array_equal(ao['basecall_pos'].ravel(), ae['basecall_pos'].ravel()))
            n_ok += same; n_bad += (not same); sample += 1
    print(f"  抽查 {sample} 个同名part对: 一致={n_ok}, 不一致={n_bad} "
          f"{'✓ esox/oxonet站点自动相同' if n_bad==0 else '⚠️不一致!'}")


if __name__ == '__main__':
    main()
