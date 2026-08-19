import argparse
import math
import os
import shutil
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import tqdm
from dataset.dataset import (PairDataset, MultiSimSampler, collate_fn, get_dataset,
                             get_test_dataset, get_dataset_autodl, EpochPairDataset, TestDataset)
from model.loss import FocalLoss
from model.model import DetectModel
from pytorch_metric_learning import losses, miners, samplers
from torch import nn
import numpy as np
import random

# 定义颜色常量
BLUE = '\033[94m'
GREEN = '\033[92m'


def parse_data(batch: dict, device, kmers=7):
    signal = batch['signals'].to(device)
    kmer = batch['kmers'].to(device)
    kmer_len = kmer[0].shape[0] - kmers
    kmer = kmer[:, kmer_len // 2: -kmer_len // 2] if kmer_len > 0 else kmer  # 7-mer: 不切
    mean = batch['mean'].to(device)
    std = batch['std'].to(device)
    dwell = batch['dwell'].to(device)
    label = batch['labels'].to(device)
    sig_l = batch['sig_l'].to(device)
    return signal, kmer, mean, std, dwell, sig_l, label


def calculate_confusion(pred: torch.Tensor, label: torch.Tensor, threshold: float = 0.5):
    """
    返回 TP, TN, FP, FN (batch 内)
    """
    label_float = label.float()
    pred_bin = (pred >= threshold).float()

    TP = ((pred_bin == 1) & (label_float == 1)).sum().item()
    TN = ((pred_bin == 0) & (label_float == 0)).sum().item()
    FP = ((pred_bin == 1) & (label_float == 0)).sum().item()
    FN = ((pred_bin == 0) & (label_float == 1)).sum().item()

    return TP, TN, FP, FN


def compute_metrics_from_confusion(TP, TN, FP, FN):
    """
    由全局的 TP, TN, FP, FN 计算分类指标
    """
    accuracy = (TP + TN) / max((TP + TN + FP + FN), 1)
    precision = TP / max((TP + FP), 1)
    recall = TP / max((TP + FN), 1)  # sensitivity
    specificity = TN / max((TN + FP), 1)
    f1 = 2 * precision * recall / max((precision + recall), 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        # "TP": TP,
        # "TN": TN,
        # "FP": FP,
        # "FN": FN,
    }


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_lambda(current_step, warmup_steps, phase1_end_step, total_steps,
                  peak_lr=1e-3, phase2_peak_lr=3e-4, min_lr=1e-5):
    """
    分阶段 LR schedule:
    - Phase 1 [0, phase1_end_step]: warmup → cosine decay 到 phase2_peak_lr
    - Phase 2 [phase1_end_step, total_steps]: 短暂 re-warmup → cosine decay 到 min_lr
    """
    # Phase 1
    if current_step < phase1_end_step:
        if current_step < warmup_steps:
            return current_step / warmup_steps  # 线性 warmup 0 → 1
        else:
            progress = (current_step - warmup_steps) / (phase1_end_step - warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            end_factor = phase2_peak_lr / peak_lr
            return end_factor + (1 - end_factor) * cosine_factor
    # Phase 2
    else:
        steps_into_phase2 = current_step - phase1_end_step
        phase2_total_steps = total_steps - phase1_end_step
        phase2_warmup_steps = phase2_total_steps // 20  # Phase 2 的 5% 用于 re-warmup

        if steps_into_phase2 < phase2_warmup_steps:
            return phase2_peak_lr / peak_lr  # 保持 phase2_peak_lr
        else:
            progress = (steps_into_phase2 - phase2_warmup_steps) / (phase2_total_steps - phase2_warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            min_factor = min_lr / peak_lr
            phase2_factor = phase2_peak_lr / peak_lr
            return min_factor + (phase2_factor - min_factor) * cosine_factor


def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "")
    model = DetectModel(sig_blocks=4, sig_l=175, seq_l=7, pos_mode='rope').to(device)
    os.makedirs(args.save_dir, exist_ok=True)
    # class_loss = nn.BCELoss()
    class_loss = FocalLoss(alpha=0.01, gamma=2.5, reduction='sum')
    miner = miners.MultiSimilarityMiner(epsilon=0.1)  # ϵ 参数
    feature_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)  # α, β, λ (λ 在代码中叫 base)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)

    # pos 常驻RAM, neg mmap; 每epoch set_epoch重采(不预生成150文件)
    dataset = EpochPairDataset(args.pos_train, args.ctx_json, args.neg_pack)
    valid_dataset = TestDataset(args.valid_npz)   # 每epoch评估用(载入一次)

    # scheduler 在循环外建好(步数只取决于 数据量/batch_size/epochs) -> resume 时能原样恢复进度
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    warmup_steps = 5 * steps_per_epoch          # warmup_epochs = 5
    phase1_end_step = 45 * steps_per_epoch      # phase1_epochs = 45 (与 MultiSim 关闭点一致)
    total_steps = args.epochs * steps_per_epoch
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: get_lr_lambda(
        step, warmup_steps, phase1_end_step, total_steps
    ))

    global_step = 0
    start_epoch = args.resume_epoch
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:      # 完整 checkpoint: 权重+优化器+LR进度
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            global_step = ckpt['global_step']
            start_epoch = ckpt['epoch'] + 1
            if ckpt.get('epochs') != args.epochs:
                print(f"[resume] ⚠ checkpoint 的 --epochs={ckpt.get('epochs')} 与本次 {args.epochs} 不同, "
                      f"cosine 总长变了 -> LR 曲线会与原计划不一致")
            print(f"[resume] 从 {args.resume} 续训: epoch {start_epoch}, global_step {global_step}, "
                  f"lr {scheduler.get_last_lr()[0]:.3e}")
        else:                                               # 旧格式: 只有 state_dict
            model.load_state_dict(ckpt)
            print(f"[resume] {args.resume} 只含权重, 优化器/LR 从头开始 "
                  f"(要正确续训请用 last.ckpt; --resume_epoch={start_epoch})")

    # train and test
    for epoch in range(start_epoch, args.epochs):

        # train
        model.train()
        dataset.set_epoch(epoch)
        sampler = MultiSimSampler(len(dataset) // 2, n_sample=args.batch_size, shuffle=True, seed=args.seed)
        sampler.set_epoch(epoch)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                sampler=sampler,
                                collate_fn=collate_fn, num_workers=args.num_workers)

        TP_total = TN_total = FP_total = FN_total = 0
        train_loss_total = 0
        pbar = tqdm.tqdm(total=len(dataloader), desc="Train Epoch {}".format(epoch))
        for batch in dataloader:
            optimizer.zero_grad()
            signal, kmer, _, _, _, sig_l, label = parse_data(batch, device)
            feature, pred = model(signal, kmer, sig_l)

            label_float = label.float()  # 用于 BCE

            # 挖掘样本
            if epoch < 45:
                label_long = label.long()
                hard_pairs = miner(feature, label_long)
                # 损失
                loss = class_loss(pred.squeeze(-1), label_float) + 0.1 * feature_loss(feature, label_long, hard_pairs)
            else:
                loss = class_loss(pred.squeeze(-1), label_float)

            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            train_loss_total += loss.item()  #
            TP, TN, FP, FN = calculate_confusion(pred.squeeze(-1), label, args.threshold)
            TP_total += TP
            TN_total += TN
            FP_total += FP
            FN_total += FN

            pbar.set_postfix({'loss': round(loss.item(), 4)})
            pbar.update(1)
            if args.max_steps and (pbar.n >= args.max_steps):
                print(f"\n[debug] max_steps={args.max_steps} 到,提前结束本 epoch 训练")
                break

        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_{}.pth'.format(epoch)))
        # 完整断点(权重+AdamW动量+LR进度), 原子写: 崩在写一半也不会毁掉上一个可用断点
        ckpt_path = os.path.join(args.save_dir, 'last.ckpt')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'global_step': global_step,
                    'epoch': epoch, 'epochs': args.epochs}, ckpt_path + '.tmp')
        os.replace(ckpt_path + '.tmp', ckpt_path)
        metrics = compute_metrics_from_confusion(TP_total, TN_total, FP_total, FN_total)
        pbar.close()
        train_loss_avg = train_loss_total / max(pbar.n, 1)  # ← 取平均(debug早停也对)
        print(BLUE + 'Train Epoch {} ==> \t'
                     'accuracy: {:.4f}, \t'
                     'precision: {:.4f}, \t'
                     'recall: {:.4f}, \t'
                     'specificity: {:.4f}, \t'
                     'f1: {:.4f},\t loss{:.4f}'.format(
            epoch,
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['specificity'],
            metrics['f1'],
            train_loss_avg
        ))
        train_info = os.path.join(args.save_dir, 'train_info.tsv')
        with open(train_info, 'a') as f:
            f.write('\t'.join([str(e) for e in [epoch,
                                                metrics['accuracy'],
                                                metrics['precision'],
                                                metrics['recall'],
                                                metrics['specificity'],
                                                metrics['f1'],
                                                train_loss_avg]]) + '\n')
            # scheduler.step()
        # test
        model.eval()
        with torch.no_grad():
            test_dataset = valid_dataset
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=collate_fn, num_workers=args.num_workers)
            pbar = tqdm.tqdm(total=len(test_dataloader), desc="Test Epoch {}".format(epoch))

            TP_total = TN_total = FP_total = FN_total = 0
            test_loss_total = 0
            for batch in test_dataloader:
                signal, kmer, _, _, _, sig_l, label = parse_data(batch, device)
                _, pred = model(signal, kmer, sig_l)

                label_float = label.float()  # 用于 BCE

                # 损失
                loss = class_loss(pred.squeeze(-1), label_float)
                test_loss_total += loss.item()  # ← 加这行
                TP, TN, FP, FN = calculate_confusion(pred.squeeze(-1), label, args.threshold)
                TP_total += TP
                TN_total += TN
                FP_total += FP
                FN_total += FN

                pbar.set_postfix({'loss': round(loss.item(), 4)})
                pbar.update(1)
                if args.max_steps and (pbar.n >= args.max_steps):
                    print(f"\n[debug] max_steps={args.max_steps} 到,提前结束本 epoch valid")
                    break
            test_loss_avg = test_loss_total / max(pbar.n, 1)  # ← 取平均(debug早停也对)
            metrics = compute_metrics_from_confusion(TP_total, TN_total, FP_total, FN_total)
            pbar.close()
            print(GREEN + 'Test Epoch {} ==> \t'
                          'accuracy: {:.4f}, \t'
                          'precision: {:.4f}, \t'
                          'recall: {:.4f}, \t'
                          'specificity: {:.4f}, \t'
                          'f1: {:.4f}, \tloss{:.4f}'.format(
                epoch,
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['specificity'],
                metrics['f1'], test_loss_avg
            ))
            test_info = os.path.join(args.save_dir, 'test_info.tsv')
            with open(test_info, 'a') as f:
                f.write('\t'.join([str(e) for e in [epoch,
                                                    metrics['accuracy'],
                                                    metrics['precision'],
                                                    metrics['recall'],
                                                    metrics['specificity'],
                                                    metrics['f1'],
                                                    test_loss_avg]]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='断点续训: 给 <save_dir>/last.ckpt 可恢复权重+优化器+LR进度; '
                             '给 model_<e>.pth 只恢复权重(LR从头)')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='起始 epoch; 用 last.ckpt 续训时自动推导, 无需手动给')
    parser.add_argument('--pos-train', type=str, default='/root/autodl-tmp/data/pack/pos_train.npz', help='pack_train 的 pos_train.npz')
    parser.add_argument('--ctx-json', type=str, default='/root/autodl-tmp/data/pack/pos_train_ctx.json', help='pos_train_ctx.json')
    parser.add_argument('--neg-pack', type=str, default='/root/autodl-tmp/data/pack/neg_pack', help='pack_train 的 neg_pack/ 目录(mmap)')
    parser.add_argument('--valid-npz', type=str, default='/root/autodl-tmp/data/pack/valid.npz', help='pack_train 的 valid.npz')
    parser.add_argument('--batch-size', type=int, default=1024*4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=0,
                        help='调试用: >0 时每个 train/valid epoch 只跑这么多 step(看流程,非正式训练)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='/root/autodl-fs/726/models/oxonetBaseline')
    args = parser.parse_args()
    start = time.time()
    train(args)
    print('Training took {} seconds'.format(time.time() - start))
    os.system("shutdown now")