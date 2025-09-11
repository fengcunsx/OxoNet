import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from dataset.dataset import PairDataset, MultiSimSampler, collate_fn, get_dataset
from model.model import DetectModel
from pytorch_metric_learning import losses, miners, samplers
from torch import nn

# 定义颜色常量
BLUE = '\033[94m'
GREEN = '\033[92m'

def parse_data(batch: dict, device):
    signal = batch['signals'].to(device)
    kmer = batch['kmers'].to(device)
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


def train(args: argparse.Namespace):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "")
    model = DetectModel().to(device)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
    os.makedirs(args.save_dir, exist_ok=True)
    class_loss = nn.BCELoss()
    miner = miners.MultiSimilarityMiner(epsilon=0.1)  # ϵ 参数
    feature_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)  # α, β, λ (λ 在代码中叫 base)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # train and test
    for epoch in range(args.epochs):

        # train
        model.train()
        dataset = get_dataset(args.pos_dir, args.neg_dir, args.work_dir, epoch, kmer=5, n_threads=8, generate=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                sampler=MultiSimSampler(len(dataset) // 2, n_sample=args.batch_size, shuffle=True),
                                collate_fn=collate_fn, num_workers=args.num_workers)
        TP_total = TN_total = FP_total = FN_total = 0
        pbar = tqdm.tqdm(total=len(dataloader), desc="Train Epoch {}".format(epoch))
        for batch in dataloader:
            optimizer.zero_grad()
            signal, kmer, mean, std, dwell, sig_l, label = parse_data(batch, device)
            feature, pred = model(signal, mean, std, dwell, kmer, sig_l)

            label_float = label.float()  # 用于 BCE
            label_long = label.long()

            # 挖掘样本
            hard_pairs = miner(feature, label_long)

            # 损失
            loss = class_loss(pred.squeeze(-1), label_float) + feature_loss(feature, label_long, hard_pairs)

            loss.backward()
            optimizer.step()
            TP, TN, FP, FN = calculate_confusion(pred.squeeze(-1), label, args.threshold)
            TP_total += TP
            TN_total += TN
            FP_total += FP
            FN_total += FN

            pbar.set_postfix({'loss': round(loss.item(), 4)})
            pbar.update(1)

        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_{}.pth'.format(epoch)))
        metrics = compute_metrics_from_confusion(TP_total, TN_total, FP_total, FN_total)
        pbar.close()
        print(BLUE+'Train Epoch {} ==> \t'
              'accuracy: {:.4f}, \t'
              'precision: {:.4f}, \t'
              'recall: {:.4f}, \t'
              'specificity: {:.4f}, \t'
              'f1: {:.4f}'.format(
            epoch,
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['specificity'],
            metrics['f1']
        ))

        # test
        model.eval()
        test_dataset = get_dataset(args.test_pos_dir, args.test_neg_dir, args.test_work_dir, epoch, kmer=5, n_threads=8,
                                   generate=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     sampler=MultiSimSampler(len(test_dataset) // 2, n_sample=args.batch_size,
                                                             shuffle=True),
                                     collate_fn=collate_fn, num_workers=args.num_workers)
        pbar = tqdm.tqdm(total=len(test_dataloader),desc="Test Epoch {}".format(epoch))

        TP_total = TN_total = FP_total = FN_total = 0
        for batch in test_dataloader:
            signal, kmer, mean, std, dwell, sig_l, label = parse_data(batch, device)
            _, pred = model(signal, mean, std, dwell, kmer, sig_l)

            label_float = label.float()  # 用于 BCE

            # 损失
            loss = class_loss(pred.squeeze(-1), label_float)

            TP, TN, FP, FN = calculate_confusion(pred.squeeze(-1), label, args.threshold)
            TP_total += TP
            TN_total += TN
            FP_total += FP
            FN_total += FN

            pbar.set_postfix({'loss': round(loss.item(), 4)})
            pbar.update(1)

        metrics = compute_metrics_from_confusion(TP_total, TN_total, FP_total, FN_total)
        pbar.close()
        print(GREEN+'Test Epoch {} ==> \t'
              'accuracy: {:.4f}, \t'
              'precision: {:.4f}, \t'
              'recall: {:.4f}, \t'
              'specificity: {:.4f}, \t'
              'f1: {:.4f}'.format(
            epoch,
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['specificity'],
            metrics['f1']
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pos-dir', type=str, default='../exp/split/8oxog_train')
    parser.add_argument('--neg-dir', type=str, default='../exp/split/g_train')
    parser.add_argument('--work-dir', type=str, default='../exp/workspace')
    parser.add_argument('--test-pos-dir', type=str, default='../exp/split/8oxog_test')
    parser.add_argument('--test-neg-dir', type=str, default='../exp/split/g_test')
    parser.add_argument('--test-work-dir', type=str, default='../exp/workspace_test')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--save-dir', type=str, default='../model_save')
    args = parser.parse_args()

    train(args)
