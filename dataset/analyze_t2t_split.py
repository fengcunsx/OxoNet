"""实测 T2T 切分可行性(只读,不写盘):
- T2T 规模(每源 f5/样本/read)、唯一 5-mer/7-mer
- 覆盖 oligo 的 1600 pos-7mer 和 877 pos-only 吗
- 三种切分下每种7-mer是否同时在 train/valid/test: (a)f5 per源90:5:5 (b)每源挑1f5做valid+1做test (c)read级90:5:5
"""
import glob, os, random, hashlib
from collections import defaultdict
import numpy as np

ROOT = '/home/bio/8oxog/data'
BATCHES = ['batch1', 'batch2', 'batch3', 'batch4']
T2T = '/home/bio/8oxog/t2t'


def oligo_7mers():
    pos = set(); neg = set()
    for b in BATCHES:
        for d in glob.glob(f'{ROOT}/{b}/{b}_work/fast5s/*'):
            for kind, S in (('8oxog', pos), ('g', neg)):
                p = f'{d}/feature/oxo/{kind}/data.npz'
                if os.path.isfile(p):
                    S.update(np.load(p)['kmers'].tolist())
    return pos, neg


def t2t_oxo_by_f5():
    """返回 {(source,f5): [npz paths]}"""
    groups = defaultdict(list)
    for f in glob.glob(f'{T2T}/*/**/oxo/*.npz', recursive=True):
        source = f.split('/t2t/')[1].split('/')[0]
        parts = f.split('/')
        idx = parts.index('oxo')          # f5 = oxo 前两层(output*/oxo 之前)
        f5 = parts[idx - 2]
        groups[(source, f5)].append(f)
    return groups


def split_id(rid, n=20):
    h = int(hashlib.md5(rid.encode()).hexdigest(), 16) % n
    return 'test' if h == 0 else ('valid' if h == 1 else 'train')  # ~90/5/5


def main():
    print("加载 oligo 7-mer 集...")
    opos, oneg = oligo_7mers()
    pos_only = opos - oneg
    print(f"  oligo pos-7mer={len(opos)} neg-7mer={len(oneg)} pos-only={len(pos_only)}")

    groups = t2t_oxo_by_f5()
    by_source = defaultdict(list)
    for (s, f5) in groups: by_source[s].append(f5)
    print(f"\nT2T f5 数: " + ", ".join(f"{s}={len(v)}" for s, v in by_source.items()))

    # 聚合: 每f5的7-mer计数; read级split的7-mer计数
    f5_k7 = {}          # (s,f5)->{k7:count}
    t2t_all7 = set()
    read_split_c7 = {'train': defaultdict(int), 'valid': defaultdict(int), 'test': defaultdict(int)}
    tot = 0; reads = set()
    for key, paths in groups.items():
        cc = defaultdict(int)
        for p in paths:
            a = np.load(p)
            k7 = a['kmers']; rid = a['read_id']
            tot += len(k7)
            for k in k7: cc[k] += 1; t2t_all7.add(k)
            for k, r in zip(k7.tolist(), rid.tolist()):
                read_split_c7[split_id(r)][k] += 1
        f5_k7[key] = cc
    print(f"T2T 总样本={tot:,} 唯一7-mer={len(t2t_all7)}")
    print(f"覆盖 oligo 1600 pos-7mer: {len(t2t_all7 & opos)}/{len(opos)} | 覆盖 877 pos-only: {len(t2t_all7 & pos_only)}/{len(pos_only)}")

    def report(name, c7):
        allk = sorted(set().union(*[set(c7[s]) for s in c7]))
        in3 = sum(1 for k in allk if all(c7[s][k] > 0 for s in c7))
        mv = sum(1 for k in allk if c7['valid'][k] == 0); mt = sum(1 for k in allk if c7['test'][k] == 0)
        te = sorted(c7['test'][k] for k in allk)
        med = te[len(te)//2] if te else 0
        print(f"  [{name}] 7-mer总={len(allk)} 三集都在={in3}({in3/max(1,len(allk))*100:.0f}%) valid缺={mv} test缺={mt} | test每种计数 min/med={min(te) if te else 0}/{med}")

    # (a) f5 per源 90:5:5
    random.seed(42); a_c7 = {'train': defaultdict(int), 'valid': defaultdict(int), 'test': defaultdict(int)}
    for s, f5s in by_source.items():
        f5s = f5s[:]; random.shuffle(f5s)
        n = len(f5s); nte = max(1, round(n*0.05)); nva = max(1, round(n*0.05))
        for i, f5 in enumerate(f5s):
            part = 'test' if i < nte else ('valid' if i < nte+nva else 'train')
            for k, v in f5_k7[(s, f5)].items(): a_c7[part][k] += v
    # (b) 每源挑1 f5 valid +1 test
    random.seed(42); b_c7 = {'train': defaultdict(int), 'valid': defaultdict(int), 'test': defaultdict(int)}
    for s, f5s in by_source.items():
        f5s = f5s[:]; random.shuffle(f5s)
        for i, f5 in enumerate(f5s):
            part = 'test' if i == 0 else ('valid' if i == 1 else 'train')
            for k, v in f5_k7[(s, f5)].items(): b_c7[part][k] += v

    print("\n=== 三种切分下 7-mer 覆盖 ===")
    report("a: f5 per源90:5:5", a_c7)
    report("b: 每源挑1f5 valid/test", b_c7)
    report("c: read级90:5:5", read_split_c7)


if __name__ == '__main__':
    main()
