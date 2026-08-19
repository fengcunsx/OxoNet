"""
按 epoch 扫 checkpoint, 在 valid.npz 上算**操作点指标**(不是 @0.5), 用来选最佳 epoch。

train_info/test_info.tsv 里记的是 threshold=0.5 的 recall/spec/f1, 但论文的操作点是
T* = "valid 上 spec 达标时的最小阈值"(见 findBestCheckPoint.find_best_threshold)。
两者排序不一样, 所以选 epoch 必须重新在 valid 上打分。

用法(autodl 或本地都一样, 只要有 pack/valid.npz 和 model_<e>.pth):
    cd <项目根>
    PYTHONPATH=. python script/sweep_epochs.py \
        --ckpt-dir out/E2 --valid-npz ../pack/valid.npz \
        --out out/E2/sweep_valid.csv --batch-size 1024

    # 先粗扫(每5个epoch), 再对候选区间细扫:
    PYTHONPATH=. python script/sweep_epochs.py ... --start 0 --end 149 --stride 5
    PYTHONPATH=. python script/sweep_epochs.py ... --epochs 128-149        # 细扫, 追加进同一个 csv

一次进程内跑完所有 ckpt(不要每个 epoch 起一个进程: 本地 4060 上 CUDA 进程反复退出
会触发 Xid 62 / PMU 挂起, 速度塌 28x)。CSV 是逐行 flush 的, 中断可用 --skip-done 续跑。
"""
import argparse
import csv
import os
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from dataset.dataset import TestDataset, collate_fn
from model.model import DetectModel


def load_state_dict(path, device):
    """兼容三种存法: 纯 state_dict / {'model':...}(last.ckpt) / {'model_state_dict':...}(旧脚本)"""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        for key in ('model', 'model_state_dict', 'state_dict'):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def parse_epochs(spec, start, end, stride):
    """--epochs '0,5,128-149' 优先; 否则用 --start/--end/--stride"""
    if not spec:
        return list(range(start, end + 1, stride))
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


@torch.no_grad()
def score(model, loader, device, amp=False):
    """跑一遍 valid, 返回 prob 向量(顺序 = dataset 顺序, shuffle=False)"""
    probs = []
    pbar = tqdm.tqdm(total=len(loader), desc='  scoring', leave=False)
    for batch in loader:
        signal = batch['signals'].to(device, non_blocking=True)
        kmer = batch['kmers'].to(device, non_blocking=True)
        sig_l = batch['sig_l'].to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            _, pred = model(signal, kmer, sig_l)
        probs.append(pred.squeeze(-1).float().cpu().numpy())
        pbar.update(1)
    pbar.close()
    return np.concatenate(probs)


def metrics_at_spec(probs, labels, spec_targets):
    """
    对每个 spec 目标, 求达标所需的最小阈值 T* 和该点的 recall。
    做法: 把 neg 分数排序, spec=s 对应允许 FP = floor(n_neg*(1-s)),
    T* = 第 (FP+1) 大的 neg 分数(再往上抬一点点), recall = pos 中 >= T* 的比例。
    比 findBestCheckPoint 的逐样本扫描等价但快得多(纯 numpy, 无 python 循环)。
    """
    pos = np.sort(probs[labels == 1])[::-1]
    neg = np.sort(probs[labels == 0])[::-1]
    n_pos, n_neg = len(pos), len(neg)
    out = {}
    for s in spec_targets:
        n_fp = int(np.floor(n_neg * (1.0 - s)))       # 允许的假阳数
        # 第 n_fp 大的 neg 分数就是"再高一档就超标"的边界; 阈值取它的上方
        thr = float(neg[n_fp]) if n_fp < n_neg else 0.0
        thr = np.nextafter(thr, 1.0)                   # 严格大于, 保证 FP <= n_fp
        tp = int(np.searchsorted(-pos, -thr, side='left'))   # pos 中 >= thr 的个数
        fp = int(np.searchsorted(-neg, -thr, side='left'))
        out[f'T@{s}'] = thr
        out[f'recall@{s}'] = tp / n_pos
        out[f'spec@{s}'] = (n_neg - fp) / n_neg
        out[f'fp@{s}'] = fp
    return out


def auroc(probs, labels):
    """Mann-Whitney U, 无需 sklearn"""
    order = np.argsort(probs)
    ranks = np.empty(len(probs), dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1)
    # 处理并列: 同分给平均秩
    p_sorted = probs[order]
    i = 0
    while i < len(p_sorted):
        j = i
        while j + 1 < len(p_sorted) and p_sorted[j + 1] == p_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int((labels == 1).sum())
    n_neg = len(labels) - n_pos
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    spec_targets = [float(s) for s in args.spec_targets.split(',')]

    print(f'[load] {args.valid_npz}')
    ds = TestDataset(args.valid_npz)
    labels = np.asarray(ds.label).astype(int)

    # 可选: 下采样 neg 省时间(全部 pos 保留)。粗扫用, 细扫建议全量。
    subset = None
    if args.neg_frac < 1.0:
        rng = np.random.default_rng(args.subsample_seed)
        neg_idx = np.flatnonzero(labels == 0)
        keep_neg = rng.choice(neg_idx, size=int(len(neg_idx) * args.neg_frac), replace=False)
        subset = np.sort(np.concatenate([np.flatnonzero(labels == 1), keep_neg]))
        ds = torch.utils.data.Subset(ds, subset.tolist())
        labels = labels[subset]
        print(f'[subsample] neg_frac={args.neg_frac} -> n={len(labels)} '
              f'(pos={int((labels==1).sum())}, neg={int((labels==0).sum())})')
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    print(f'[valid] n={len(labels)} pos={n_pos} neg={n_neg}')
    for s in spec_targets:
        print(f'  spec={s}: 允许 FP={int(n_neg*(1-s))}  |  recall 的统计误差 ~{np.sqrt(0.7*0.3/n_pos):.4f}')

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)

    model = DetectModel(sig_blocks=4, sig_l=175, seq_l=7, pos_mode='rope').to(device)
    model.eval()

    epochs = parse_epochs(args.epochs, args.start, args.end, args.stride)
    done = set()
    if args.skip_done and os.path.exists(args.out):
        with open(args.out) as f:
            done = {int(r['epoch']) for r in csv.DictReader(f)}
        print(f'[resume] 已有 {len(done)} 个 epoch, 跳过')
    epochs = [e for e in epochs if e not in done]
    print(f'[sweep] {len(epochs)} 个 checkpoint: {epochs[:5]}{"..." if len(epochs) > 5 else ""}')

    if args.prob_dir:
        os.makedirs(args.prob_dir, exist_ok=True)

    fieldnames = None
    fout = open(args.out, 'a' if done else 'w', newline='')
    writer = None

    for e in epochs:
        ckpt = os.path.join(args.ckpt_dir, args.ckpt_pattern.format(e))
        if not os.path.exists(ckpt):
            print(f'[skip] 缺文件 {ckpt}')
            continue
        model.load_state_dict(load_state_dict(ckpt, device), strict=True)
        probs = score(model, loader, device, amp=args.amp)

        row = {'epoch': e}
        row.update(metrics_at_spec(probs, labels, spec_targets))
        row['auroc'] = auroc(probs, labels)
        # 也记 @0.5, 好和 test_info.tsv 对得上(自检: 应该几乎一致)
        pred = probs >= 0.5
        tp = int((pred & (labels == 1)).sum()); fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum()); tn = int((~pred & (labels == 0)).sum())
        row['recall@0.5'] = tp / max(tp + fn, 1)
        row['spec@0.5'] = tn / max(tn + fp, 1)
        row['f1@0.5'] = 2 * tp / max(2 * tp + fp + fn, 1)

        if writer is None:
            fieldnames = list(row.keys())
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            if not done:
                writer.writeheader()
        writer.writerow(row)
        fout.flush()

        if args.prob_dir:
            np.save(os.path.join(args.prob_dir, f'probs_{e}.npy'), probs.astype(np.float32))

        main_s = spec_targets[0]
        print(f"ep{e:3d}  T*={row[f'T@{main_s}']:.4f}  recall@{main_s}={row[f'recall@{main_s}']:.4f}  "
              f"auroc={row['auroc']:.5f}  f1@0.5={row['f1@0.5']:.4f}")

    fout.close()
    # 同时出一份 tsv(和 train_info/test_info 一个格式, 方便直接并排看)
    tsv_path = os.path.splitext(args.out)[0] + '.tsv'
    with open(args.out) as fi, open(tsv_path, 'w', newline='') as ft:
        rd = csv.reader(fi)
        csv.writer(ft, delimiter='\t').writerows(rd)
    print(f'\n[done] -> {args.out}\n         -> {tsv_path}')
    report(args.out, spec_targets[0], args.window)


def report(csv_path, spec, window):
    """按 recall@spec 排名; 同时给滑窗均值排名(单点 argmax 会 overfit valid)"""
    rows = sorted((({k: float(v) for k, v in r.items()}) for r in csv.DictReader(open(csv_path))),
                  key=lambda r: r['epoch'])
    if not rows:
        return
    key = f'recall@{spec}'
    print(f'\n=== top10 单点 {key} ===')
    for r in sorted(rows, key=lambda r: -r[key])[:10]:
        print(f"  ep{int(r['epoch']):3d}  {key}={r[key]:.4f}  T*={r[f'T@{spec}']:.4f}  auroc={r['auroc']:.5f}")
    if len(rows) >= window:
        print(f'\n=== top10 {window}-epoch 滑窗均值(更稳, 推荐用它选) ===')
        wins = []
        for i in range(len(rows) - window + 1):
            w = rows[i:i + window]
            wins.append((np.mean([r[key] for r in w]), w[window // 2]['epoch'],
                         int(w[0]['epoch']), int(w[-1]['epoch'])))
        for m, c, a, b in sorted(wins, reverse=True)[:10]:
            print(f'  ep{a}-{b} 均值={m:.4f}  → 取中心 ep{int(c)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', type=str, default='/root/autodl-fs/726/models/oxonetBaseline', help='存 model_<e>.pth 的目录')
    p.add_argument('--ckpt-pattern', type=str, default='model_{}.pth')
    p.add_argument('--valid-npz', type=str, default='/root/autodl-tmp/data/pack/valid.npz')
    p.add_argument('--out', type=str, default='/root/autodl-fs/726/models/info999.csv', help='结果 csv')
    p.add_argument('--prob-dir', type=str, default='/root/autodl-fs/726/models/valid999',
                   help='存每个 epoch 的 prob(每个 ~12MB); 存了以后换指标不用重跑')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=149)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--epochs', type=str, default=None, help="如 '0,5,128-149', 优先于 start/end/stride")
    p.add_argument('--spec-targets', type=str, default='0.999,0.9999',
                   help='第一个是主选择指标')
    p.add_argument('--neg-frac', type=float, default=1.0, help='<1 则下采样 neg 提速(粗扫用)')
    p.add_argument('--subsample-seed', type=int, default=42)
    p.add_argument('--window', type=int, default=5, help='滑窗均值宽度')
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--amp', action='store_true', help='bf16 推理提速(选 epoch 够用; 最终打分建议 fp32)')
    p.add_argument('--skip-done', action='store_true', help='续跑: 跳过 csv 里已有的 epoch')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    try:
        if args.report_only:
            report(args.out, float(args.spec_targets.split(',')[0]), args.window)
        else:
            main(args)
    finally:
        os.system("shutdown now")