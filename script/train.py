import argparse
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import tqdm
from dataset.dataset import PairDataset, MultiSimSampler, collate_fn, get_dataset, get_test_dataset
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控 specificity，越大越好
        factor=0.5,  # 每次 lr 减半
        patience=3,  # 连续 3 个 epoch 无提升就降 lr
        threshold=1e-4,  # 容忍的小幅波动
        min_lr=1e-7,  # lr 下限
        verbose=True  # 打印每次调整
    )


    # train and test
    for epoch in range(args.epochs):

        # train
        model.train()
        epoch += args.resume_epoch
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
            loss = class_loss(pred.squeeze(-1), label_float) + 0.1 * feature_loss(feature, label_long, hard_pairs)

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
        print(BLUE + 'Train Epoch {} ==> \t'
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
        # scheduler.step()
        # test
        model.eval()
        # test_dataset = get_dataset(args.test_pos_dir, args.test_neg_dir, args.test_work_dir, epoch, kmer=5, n_threads=8,
        #                            generate=True)
        test_dataset = get_test_dataset(args.test_pos_dir, args.test_neg_dir, args.test_work_dir,
                                        generate=True if epoch == 0 else False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                     collate_fn=collate_fn, num_workers=args.num_workers)
        pbar = tqdm.tqdm(total=len(test_dataloader), desc="Test Epoch {}".format(epoch))

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
        print(GREEN + 'Test Epoch {} ==> \t'
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
        scheduler.step(metrics['specificity'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='../model_save_lr_schedule_feta/model_59.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--resume_epoch', type=int, default=60)
    parser.add_argument('--pos-dir', type=str, default='/home/bio/8oxog/data/feature/8oxog_train')
    parser.add_argument('--neg-dir', type=str, default='/home/bio/8oxog/data/feature/g_train')
    parser.add_argument('--work-dir', type=str, default='/home/bio/8oxog/data/feature/workspace')
    parser.add_argument('--test-pos-dir', type=str, default='/home/bio/8oxog/data/feature/8oxog_test')
    parser.add_argument('--test-neg-dir', type=str, default='/home/bio/8oxog/data/feature/g_test')
    parser.add_argument('--test-work-dir', type=str, default='/home/bio/8oxog/data/feature/test_workspace')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--save-dir', type=str, default='../model_save_lr_schedule_feta')
    args = parser.parse_args()

    train(args)

'''
baseline
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Train Epoch 0: 100%|██████████| 6195/6195 [04:42<00:00, 21.90it/s, loss=2.26]
Train Epoch 0 ==> 	accuracy: 0.8794, 	precision: 0.8873, 	recall: 0.8693, 	specificity: 0.8896, 	f1: 0.8782
Test Epoch 0: 100%|██████████| 1715/1715 [00:35<00:00, 48.84it/s, loss=0.446]
Test Epoch 0 ==> 	accuracy: 0.8780, 	precision: 0.6438, 	recall: 0.8785, 	specificity: 0.8779, 	f1: 0.7431
Train Epoch 1: 100%|██████████| 6195/6195 [05:54<00:00, 17.46it/s, loss=2.46]
Train Epoch 1 ==> 	accuracy: 0.8979, 	precision: 0.9075, 	recall: 0.8862, 	specificity: 0.9096, 	f1: 0.8967
Test Epoch 1: 100%|██████████| 1715/1715 [00:37<00:00, 46.01it/s, loss=0.253]
Test Epoch 1 ==> 	accuracy: 0.8990, 	precision: 0.6997, 	recall: 0.8700, 	specificity: 0.9062, 	f1: 0.7757
Train Epoch 2: 100%|██████████| 6195/6195 [06:05<00:00, 16.97it/s, loss=2.38]
Train Epoch 2 ==> 	accuracy: 0.9064, 	precision: 0.9132, 	recall: 0.8982, 	specificity: 0.9146, 	f1: 0.9056
Test Epoch 2: 100%|██████████| 1715/1715 [00:36<00:00, 46.92it/s, loss=0.16]
Test Epoch 2 ==> 	accuracy: 0.8978, 	precision: 0.6887, 	recall: 0.8957, 	specificity: 0.8983, 	f1: 0.7787
Train Epoch 3: 100%|██████████| 6195/6195 [06:00<00:00, 17.18it/s, loss=2.35]
Train Epoch 3 ==> 	accuracy: 0.9113, 	precision: 0.9167, 	recall: 0.9049, 	specificity: 0.9178, 	f1: 0.9107
Test Epoch 3: 100%|██████████| 1715/1715 [00:39<00:00, 43.36it/s, loss=0.334]
Test Epoch 3 ==> 	accuracy: 0.8924, 	precision: 0.6704, 	recall: 0.9130, 	specificity: 0.8872, 	f1: 0.7731
Train Epoch 4: 100%|██████████| 6195/6195 [06:00<00:00, 17.18it/s, loss=2.25]
Train Epoch 4 ==> 	accuracy: 0.9176, 	precision: 0.9231, 	recall: 0.9110, 	specificity: 0.9241, 	f1: 0.9170
Test Epoch 4: 100%|██████████| 1715/1715 [00:38<00:00, 44.45it/s, loss=0.224]
Test Epoch 4 ==> 	accuracy: 0.8982, 	precision: 0.6837, 	recall: 0.9174, 	specificity: 0.8934, 	f1: 0.7835
Train Epoch 5: 100%|██████████| 6195/6195 [06:07<00:00, 16.86it/s, loss=2.53]
Train Epoch 5 ==> 	accuracy: 0.9176, 	precision: 0.9230, 	recall: 0.9112, 	specificity: 0.9240, 	f1: 0.9171
Test Epoch 5: 100%|██████████| 1715/1715 [00:39<00:00, 43.70it/s, loss=0.228]
Test Epoch 5 ==> 	accuracy: 0.9081, 	precision: 0.7137, 	recall: 0.9056, 	specificity: 0.9087, 	f1: 0.7982
Train Epoch 6: 100%|██████████| 6195/6195 [06:08<00:00, 16.80it/s, loss=2.44]
Train Epoch 6 ==> 	accuracy: 0.9175, 	precision: 0.9228, 	recall: 0.9112, 	specificity: 0.9237, 	f1: 0.9169
Test Epoch 6: 100%|██████████| 1715/1715 [00:42<00:00, 40.70it/s, loss=0.196]
Test Epoch 6 ==> 	accuracy: 0.9108, 	precision: 0.7246, 	recall: 0.8965, 	specificity: 0.9144, 	f1: 0.8014
Train Epoch 7: 100%|██████████| 6195/6195 [06:01<00:00, 17.15it/s, loss=2.4]
Train Epoch 7 ==> 	accuracy: 0.9245, 	precision: 0.9278, 	recall: 0.9207, 	specificity: 0.9284, 	f1: 0.9242
Test Epoch 7: 100%|██████████| 1715/1715 [00:42<00:00, 40.13it/s, loss=0.179]
Test Epoch 7 ==> 	accuracy: 0.9197, 	precision: 0.7510, 	recall: 0.8979, 	specificity: 0.9252, 	f1: 0.8179
Train Epoch 8: 100%|██████████| 6195/6195 [06:09<00:00, 16.77it/s, loss=2.39]
Train Epoch 8 ==> 	accuracy: 0.9188, 	precision: 0.9250, 	recall: 0.9116, 	specificity: 0.9261, 	f1: 0.9182
Test Epoch 8: 100%|██████████| 1715/1715 [00:41<00:00, 41.41it/s, loss=0.144]
Test Epoch 8 ==> 	accuracy: 0.9123, 	precision: 0.7241, 	recall: 0.9095, 	specificity: 0.9130, 	f1: 0.8063
Train Epoch 9: 100%|██████████| 6195/6195 [06:12<00:00, 16.62it/s, loss=2.32]
Train Epoch 9 ==> 	accuracy: 0.9233, 	precision: 0.9279, 	recall: 0.9179, 	specificity: 0.9287, 	f1: 0.9229
Test Epoch 9: 100%|██████████| 1715/1715 [00:48<00:00, 35.68it/s, loss=0.174]
Test Epoch 9 ==> 	accuracy: 0.9152, 	precision: 0.7386, 	recall: 0.8940, 	specificity: 0.9205, 	f1: 0.8089
Train Epoch 10: 100%|██████████| 6195/6195 [05:59<00:00, 17.21it/s, loss=2.03]
Train Epoch 10 ==> 	accuracy: 0.9255, 	precision: 0.9285, 	recall: 0.9220, 	specificity: 0.9290, 	f1: 0.9252
Test Epoch 10: 100%|██████████| 1715/1715 [00:43<00:00, 39.49it/s, loss=0.188]
Test Epoch 10 ==> 	accuracy: 0.9080, 	precision: 0.7112, 	recall: 0.9118, 	specificity: 0.9070, 	f1: 0.7991
Train Epoch 11: 100%|██████████| 6195/6195 [06:12<00:00, 16.63it/s, loss=2.1]
Train Epoch 11 ==> 	accuracy: 0.9256, 	precision: 0.9301, 	recall: 0.9204, 	specificity: 0.9308, 	f1: 0.9252
Test Epoch 11: 100%|██████████| 1715/1715 [00:40<00:00, 42.07it/s, loss=0.176]
Test Epoch 11 ==> 	accuracy: 0.9144, 	precision: 0.7356, 	recall: 0.8953, 	specificity: 0.9192, 	f1: 0.8076
Train Epoch 12: 100%|██████████| 6195/6195 [06:16<00:00, 16.47it/s, loss=2.06]
Train Epoch 12 ==> 	accuracy: 0.9253, 	precision: 0.9292, 	recall: 0.9208, 	specificity: 0.9299, 	f1: 0.9250
Test Epoch 12: 100%|██████████| 1715/1715 [00:41<00:00, 41.09it/s, loss=0.233]
Test Epoch 12 ==> 	accuracy: 0.9199, 	precision: 0.7450, 	recall: 0.9139, 	specificity: 0.9214, 	f1: 0.8208
Train Epoch 13: 100%|██████████| 6195/6195 [06:14<00:00, 16.52it/s, loss=2.06]
Train Epoch 13 ==> 	accuracy: 0.9269, 	precision: 0.9308, 	recall: 0.9224, 	specificity: 0.9314, 	f1: 0.9266
Test Epoch 13: 100%|██████████| 1715/1715 [00:42<00:00, 40.25it/s, loss=0.373]
Test Epoch 13 ==> 	accuracy: 0.9031, 	precision: 0.6957, 	recall: 0.9198, 	specificity: 0.8989, 	f1: 0.7922
Train Epoch 14: 100%|██████████| 6195/6195 [06:02<00:00, 17.08it/s, loss=1.89]
Train Epoch 14 ==> 	accuracy: 0.9328, 	precision: 0.9345, 	recall: 0.9309, 	specificity: 0.9348, 	f1: 0.9327
Test Epoch 14: 100%|██████████| 1715/1715 [00:43<00:00, 39.82it/s, loss=0.121]
Test Epoch 14 ==> 	accuracy: 0.9122, 	precision: 0.7204, 	recall: 0.9194, 	specificity: 0.9104, 	f1: 0.8079
Train Epoch 15: 100%|██████████| 6195/6195 [06:10<00:00, 16.72it/s, loss=2.18]
Train Epoch 15 ==> 	accuracy: 0.9261, 	precision: 0.9311, 	recall: 0.9204, 	specificity: 0.9319, 	f1: 0.9257
Test Epoch 15: 100%|██████████| 1715/1715 [00:44<00:00, 38.27it/s, loss=0.177]
Test Epoch 15 ==> 	accuracy: 0.9139, 	precision: 0.7288, 	recall: 0.9094, 	specificity: 0.9150, 	f1: 0.8092
Train Epoch 16: 100%|██████████| 6195/6195 [06:15<00:00, 16.50it/s, loss=1.96]
Train Epoch 16 ==> 	accuracy: 0.9297, 	precision: 0.9328, 	recall: 0.9262, 	specificity: 0.9332, 	f1: 0.9294
Test Epoch 16: 100%|██████████| 1715/1715 [00:40<00:00, 42.81it/s, loss=0.176]
Test Epoch 16 ==> 	accuracy: 0.9242, 	precision: 0.7611, 	recall: 0.9074, 	specificity: 0.9284, 	f1: 0.8278
Train Epoch 17: 100%|██████████| 6195/6195 [06:10<00:00, 16.70it/s, loss=1.94]
Train Epoch 17 ==> 	accuracy: 0.9333, 	precision: 0.9362, 	recall: 0.9299, 	specificity: 0.9366, 	f1: 0.9330
Test Epoch 17: 100%|██████████| 1715/1715 [00:43<00:00, 39.79it/s, loss=0.239]
Test Epoch 17 ==> 	accuracy: 0.9158, 	precision: 0.7304, 	recall: 0.9199, 	specificity: 0.9147, 	f1: 0.8143
Train Epoch 18: 100%|██████████| 6195/6195 [06:03<00:00, 17.05it/s, loss=2.11]
Train Epoch 18 ==> 	accuracy: 0.9302, 	precision: 0.9340, 	recall: 0.9259, 	specificity: 0.9346, 	f1: 0.9299
Test Epoch 18: 100%|██████████| 1715/1715 [00:42<00:00, 40.70it/s, loss=0.124]
Test Epoch 18 ==> 	accuracy: 0.9188, 	precision: 0.7436, 	recall: 0.9090, 	specificity: 0.9213, 	f1: 0.8180
Train Epoch 19: 100%|██████████| 6195/6195 [06:16<00:00, 16.47it/s, loss=2.15]
Train Epoch 19 ==> 	accuracy: 0.9301, 	precision: 0.9328, 	recall: 0.9270, 	specificity: 0.9332, 	f1: 0.9299
Test Epoch 19: 100%|██████████| 1715/1715 [00:40<00:00, 42.85it/s, loss=0.209]
Test Epoch 19 ==> 	accuracy: 0.9268, 	precision: 0.7694, 	recall: 0.9072, 	specificity: 0.9317, 	f1: 0.8326
Train Epoch 20: 100%|██████████| 6195/6195 [06:08<00:00, 16.83it/s, loss=2.04]
Train Epoch 20 ==> 	accuracy: 0.9328, 	precision: 0.9359, 	recall: 0.9292, 	specificity: 0.9363, 	f1: 0.9325
Test Epoch 20: 100%|██████████| 1715/1715 [00:44<00:00, 38.79it/s, loss=0.241]
Test Epoch 20 ==> 	accuracy: 0.9179, 	precision: 0.7379, 	recall: 0.9168, 	specificity: 0.9182, 	f1: 0.8177
Train Epoch 21: 100%|██████████| 6195/6195 [06:22<00:00, 16.20it/s, loss=2.15]
Train Epoch 21 ==> 	accuracy: 0.9346, 	precision: 0.9364, 	recall: 0.9325, 	specificity: 0.9367, 	f1: 0.9344
Test Epoch 21: 100%|██████████| 1715/1715 [00:41<00:00, 40.87it/s, loss=0.182]
Test Epoch 21 ==> 	accuracy: 0.9236, 	precision: 0.7590, 	recall: 0.9073, 	specificity: 0.9276, 	f1: 0.8266
Train Epoch 22: 100%|██████████| 6195/6195 [06:21<00:00, 16.25it/s, loss=1.96]
Train Epoch 22 ==> 	accuracy: 0.9301, 	precision: 0.9339, 	recall: 0.9258, 	specificity: 0.9345, 	f1: 0.9298
Test Epoch 22: 100%|██████████| 1715/1715 [00:40<00:00, 42.63it/s, loss=0.149]
Test Epoch 22 ==> 	accuracy: 0.9103, 	precision: 0.7194, 	recall: 0.9066, 	specificity: 0.9112, 	f1: 0.8022
Train Epoch 23: 100%|██████████| 6195/6195 [06:15<00:00, 16.48it/s, loss=2.06]
Train Epoch 23 ==> 	accuracy: 0.9359, 	precision: 0.9370, 	recall: 0.9345, 	specificity: 0.9372, 	f1: 0.9358
Test Epoch 23: 100%|██████████| 1715/1715 [00:41<00:00, 41.25it/s, loss=0.131]
Test Epoch 23 ==> 	accuracy: 0.9043, 	precision: 0.7001, 	recall: 0.9158, 	specificity: 0.9014, 	f1: 0.7936
Train Epoch 24: 100%|██████████| 6195/6195 [06:08<00:00, 16.81it/s, loss=1.96]
Train Epoch 24 ==> 	accuracy: 0.9344, 	precision: 0.9373, 	recall: 0.9312, 	specificity: 0.9377, 	f1: 0.9342
Test Epoch 24: 100%|██████████| 1715/1715 [00:41<00:00, 41.71it/s, loss=0.153]
Test Epoch 24 ==> 	accuracy: 0.9278, 	precision: 0.7694, 	recall: 0.9142, 	specificity: 0.9312, 	f1: 0.8356
Train Epoch 25: 100%|██████████| 6195/6195 [06:25<00:00, 16.06it/s, loss=2.22]
Train Epoch 25 ==> 	accuracy: 0.9352, 	precision: 0.9377, 	recall: 0.9322, 	specificity: 0.9381, 	f1: 0.9350
Test Epoch 25: 100%|██████████| 1715/1715 [00:44<00:00, 38.65it/s, loss=0.112]
Test Epoch 25 ==> 	accuracy: 0.9255, 	precision: 0.7620, 	recall: 0.9147, 	specificity: 0.9283, 	f1: 0.8314
Train Epoch 26: 100%|██████████| 6195/6195 [06:13<00:00, 16.57it/s, loss=1.96]
Train Epoch 26 ==> 	accuracy: 0.9336, 	precision: 0.9355, 	recall: 0.9315, 	specificity: 0.9358, 	f1: 0.9335
Test Epoch 26: 100%|██████████| 1715/1715 [00:44<00:00, 38.52it/s, loss=0.239]
Test Epoch 26 ==> 	accuracy: 0.9274, 	precision: 0.7735, 	recall: 0.9025, 	specificity: 0.9336, 	f1: 0.8330
Train Epoch 27: 100%|██████████| 6195/6195 [06:12<00:00, 16.62it/s, loss=1.93]
Train Epoch 27 ==> 	accuracy: 0.9400, 	precision: 0.9413, 	recall: 0.9386, 	specificity: 0.9414, 	f1: 0.9399
Test Epoch 27: 100%|██████████| 1715/1715 [00:43<00:00, 39.73it/s, loss=0.167]
Test Epoch 27 ==> 	accuracy: 0.9198, 	precision: 0.7423, 	recall: 0.9202, 	specificity: 0.9197, 	f1: 0.8217
Train Epoch 28: 100%|██████████| 6195/6195 [06:09<00:00, 16.75it/s, loss=1.89]
Train Epoch 28 ==> 	accuracy: 0.9346, 	precision: 0.9375, 	recall: 0.9313, 	specificity: 0.9379, 	f1: 0.9344
Test Epoch 28: 100%|██████████| 1715/1715 [00:40<00:00, 42.74it/s, loss=0.154]
Test Epoch 28 ==> 	accuracy: 0.9184, 	precision: 0.7382, 	recall: 0.9202, 	specificity: 0.9180, 	f1: 0.8192
Train Epoch 29: 100%|██████████| 6195/6195 [06:10<00:00, 16.73it/s, loss=2.1]
Train Epoch 29 ==> 	accuracy: 0.9358, 	precision: 0.9372, 	recall: 0.9342, 	specificity: 0.9374, 	f1: 0.9357
Test Epoch 29: 100%|██████████| 1715/1715 [00:44<00:00, 38.78it/s, loss=0.17]
Test Epoch 29 ==> 	accuracy: 0.9239, 	precision: 0.7567, 	recall: 0.9152, 	specificity: 0.9261, 	f1: 0.8284

Process finished with exit code 0
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Train Epoch 30: 100%|██████████| 6195/6195 [05:35<00:00, 18.49it/s, loss=2.04]
Train Epoch 30 ==> 	accuracy: 0.9364, 	precision: 0.9387, 	recall: 0.9338, 	specificity: 0.9390, 	f1: 0.9362
Test Epoch 30: 100%|██████████| 1715/1715 [00:41<00:00, 41.41it/s, loss=0.152]
Test Epoch 30 ==> 	accuracy: 0.9157, 	precision: 0.7341, 	recall: 0.9092, 	specificity: 0.9173, 	f1: 0.8123
Train Epoch 31: 100%|██████████| 6195/6195 [06:12<00:00, 16.63it/s, loss=1.86]
Train Epoch 31 ==> 	accuracy: 0.9387, 	precision: 0.9391, 	recall: 0.9383, 	specificity: 0.9391, 	f1: 0.9387
Test Epoch 31: 100%|██████████| 1715/1715 [00:39<00:00, 43.19it/s, loss=0.227]
Test Epoch 31 ==> 	accuracy: 0.9236, 	precision: 0.7555, 	recall: 0.9161, 	specificity: 0.9255, 	f1: 0.8281
Train Epoch 32: 100%|██████████| 6195/6195 [06:20<00:00, 16.30it/s, loss=1.99]
Train Epoch 32 ==> 	accuracy: 0.9350, 	precision: 0.9384, 	recall: 0.9311, 	specificity: 0.9389, 	f1: 0.9347
Test Epoch 32: 100%|██████████| 1715/1715 [00:41<00:00, 41.34it/s, loss=0.134]
Test Epoch 32 ==> 	accuracy: 0.9319, 	precision: 0.7906, 	recall: 0.8985, 	specificity: 0.9402, 	f1: 0.8411
Train Epoch 33: 100%|██████████| 6195/6195 [06:22<00:00, 16.20it/s, loss=2.03]
Train Epoch 33 ==> 	accuracy: 0.9380, 	precision: 0.9379, 	recall: 0.9381, 	specificity: 0.9379, 	f1: 0.9380
Test Epoch 33: 100%|██████████| 1715/1715 [00:45<00:00, 37.64it/s, loss=0.217]
Test Epoch 33 ==> 	accuracy: 0.9260, 	precision: 0.7655, 	recall: 0.9100, 	specificity: 0.9300, 	f1: 0.8315
Train Epoch 34: 100%|██████████| 6195/6195 [06:22<00:00, 16.19it/s, loss=2.1]
Train Epoch 34 ==> 	accuracy: 0.9388, 	precision: 0.9409, 	recall: 0.9364, 	specificity: 0.9412, 	f1: 0.9386
Test Epoch 34: 100%|██████████| 1715/1715 [00:46<00:00, 37.22it/s, loss=0.129]
Test Epoch 34 ==> 	accuracy: 0.9230, 	precision: 0.7540, 	recall: 0.9149, 	specificity: 0.9250, 	f1: 0.8267
Train Epoch 35: 100%|██████████| 6195/6195 [06:20<00:00, 16.30it/s, loss=2.01]
Train Epoch 35 ==> 	accuracy: 0.9373, 	precision: 0.9390, 	recall: 0.9353, 	specificity: 0.9393, 	f1: 0.9371
Test Epoch 35: 100%|██████████| 1715/1715 [00:42<00:00, 39.89it/s, loss=0.181]
Test Epoch 35 ==> 	accuracy: 0.9149, 	precision: 0.7299, 	recall: 0.9147, 	specificity: 0.9150, 	f1: 0.8119
Train Epoch 36: 100%|██████████| 6195/6195 [06:26<00:00, 16.01it/s, loss=2.07]
Train Epoch 36 ==> 	accuracy: 0.9402, 	precision: 0.9409, 	recall: 0.9394, 	specificity: 0.9410, 	f1: 0.9402
Test Epoch 36: 100%|██████████| 1715/1715 [00:41<00:00, 41.24it/s, loss=0.133]
Test Epoch 36 ==> 	accuracy: 0.9234, 	precision: 0.7527, 	recall: 0.9213, 	specificity: 0.9240, 	f1: 0.8285
Train Epoch 37: 100%|██████████| 6195/6195 [06:24<00:00, 16.13it/s, loss=2.34]
Train Epoch 37 ==> 	accuracy: 0.9382, 	precision: 0.9402, 	recall: 0.9359, 	specificity: 0.9405, 	f1: 0.9380
Test Epoch 37: 100%|██████████| 1715/1715 [00:41<00:00, 41.82it/s, loss=0.195]
Test Epoch 37 ==> 	accuracy: 0.9310, 	precision: 0.7855, 	recall: 0.9025, 	specificity: 0.9381, 	f1: 0.8400
Train Epoch 38: 100%|██████████| 6195/6195 [06:25<00:00, 16.08it/s, loss=1.86]
Train Epoch 38 ==> 	accuracy: 0.9411, 	precision: 0.9417, 	recall: 0.9403, 	specificity: 0.9418, 	f1: 0.9410
Test Epoch 38: 100%|██████████| 1715/1715 [00:44<00:00, 38.50it/s, loss=0.147]
Test Epoch 38 ==> 	accuracy: 0.9275, 	precision: 0.7656, 	recall: 0.9210, 	specificity: 0.9292, 	f1: 0.8361
Train Epoch 39: 100%|██████████| 6195/6195 [06:22<00:00, 16.21it/s, loss=1.84]
Train Epoch 39 ==> 	accuracy: 0.9377, 	precision: 0.9397, 	recall: 0.9356, 	specificity: 0.9399, 	f1: 0.9376
Test Epoch 39: 100%|██████████| 1715/1715 [00:42<00:00, 40.35it/s, loss=0.219]
Test Epoch 39 ==> 	accuracy: 0.9313, 	precision: 0.7847, 	recall: 0.9063, 	specificity: 0.9375, 	f1: 0.8411

Process finished with exit code 0
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Train Epoch 40: 100%|██████████| 6195/6195 [06:06<00:00, 16.88it/s, loss=1.93]
Train Epoch 40 ==> 	accuracy: 0.9453, 	precision: 0.9440, 	recall: 0.9467, 	specificity: 0.9439, 	f1: 0.9454
Test Epoch 40: 100%|██████████| 1715/1715 [00:47<00:00, 36.21it/s, loss=0.176]
Test Epoch 40 ==> 	accuracy: 0.9290, 	precision: 0.7743, 	recall: 0.9121, 	specificity: 0.9332, 	f1: 0.8376
Train Epoch 41: 100%|██████████| 6195/6195 [06:22<00:00, 16.19it/s, loss=1.92]
Train Epoch 41 ==> 	accuracy: 0.9362, 	precision: 0.9387, 	recall: 0.9335, 	specificity: 0.9390, 	f1: 0.9361
Test Epoch 41: 100%|██████████| 1715/1715 [00:41<00:00, 41.03it/s, loss=0.224]
Test Epoch 41 ==> 	accuracy: 0.9273, 	precision: 0.7678, 	recall: 0.9140, 	specificity: 0.9306, 	f1: 0.8346
Train Epoch 42: 100%|██████████| 6195/6195 [06:10<00:00, 16.73it/s, loss=2.19]
Train Epoch 42 ==> 	accuracy: 0.9426, 	precision: 0.9426, 	recall: 0.9426, 	specificity: 0.9426, 	f1: 0.9426
Test Epoch 42: 100%|██████████| 1715/1715 [00:42<00:00, 40.58it/s, loss=0.227]
Test Epoch 42 ==> 	accuracy: 0.9306, 	precision: 0.7817, 	recall: 0.9079, 	specificity: 0.9363, 	f1: 0.8401
Train Epoch 43: 100%|██████████| 6195/6195 [06:06<00:00, 16.89it/s, loss=2.08]
Train Epoch 43 ==> 	accuracy: 0.9414, 	precision: 0.9415, 	recall: 0.9411, 	specificity: 0.9416, 	f1: 0.9413
Test Epoch 43: 100%|██████████| 1715/1715 [00:40<00:00, 42.02it/s, loss=0.192]
Test Epoch 43 ==> 	accuracy: 0.9269, 	precision: 0.7666, 	recall: 0.9142, 	specificity: 0.9301, 	f1: 0.8340
Train Epoch 44: 100%|██████████| 6195/6195 [06:09<00:00, 16.76it/s, loss=2.25]
Train Epoch 44 ==> 	accuracy: 0.9395, 	precision: 0.9400, 	recall: 0.9390, 	specificity: 0.9401, 	f1: 0.9395
Test Epoch 44: 100%|██████████| 1715/1715 [00:37<00:00, 45.44it/s, loss=0.208]
Test Epoch 44 ==> 	accuracy: 0.9338, 	precision: 0.7943, 	recall: 0.9045, 	specificity: 0.9412, 	f1: 0.8458
Train Epoch 45: 100%|██████████| 6195/6195 [06:08<00:00, 16.79it/s, loss=1.92]
Train Epoch 45 ==> 	accuracy: 0.9411, 	precision: 0.9422, 	recall: 0.9399, 	specificity: 0.9424, 	f1: 0.9411
Test Epoch 45: 100%|██████████| 1715/1715 [00:36<00:00, 46.58it/s, loss=0.159]
Test Epoch 45 ==> 	accuracy: 0.9322, 	precision: 0.7854, 	recall: 0.9115, 	specificity: 0.9374, 	f1: 0.8438
Train Epoch 46: 100%|██████████| 6195/6195 [06:05<00:00, 16.93it/s, loss=1.94]
Train Epoch 46 ==> 	accuracy: 0.9415, 	precision: 0.9417, 	recall: 0.9414, 	specificity: 0.9417, 	f1: 0.9415
Test Epoch 46: 100%|██████████| 1715/1715 [00:39<00:00, 43.20it/s, loss=0.254]
Test Epoch 46 ==> 	accuracy: 0.9202, 	precision: 0.7425, 	recall: 0.9220, 	specificity: 0.9197, 	f1: 0.8226
Train Epoch 47: 100%|██████████| 6195/6195 [05:44<00:00, 17.97it/s, loss=1.88]
Train Epoch 47 ==> 	accuracy: 0.9445, 	precision: 0.9442, 	recall: 0.9448, 	specificity: 0.9442, 	f1: 0.9445
Test Epoch 47: 100%|██████████| 1715/1715 [00:37<00:00, 46.29it/s, loss=0.143]
Test Epoch 47 ==> 	accuracy: 0.9229, 	precision: 0.7526, 	recall: 0.9179, 	specificity: 0.9242, 	f1: 0.8271
Train Epoch 48: 100%|██████████| 6195/6195 [05:44<00:00, 17.97it/s, loss=1.94]
Train Epoch 48 ==> 	accuracy: 0.9385, 	precision: 0.9395, 	recall: 0.9373, 	specificity: 0.9396, 	f1: 0.9384
Test Epoch 48: 100%|██████████| 1715/1715 [00:41<00:00, 41.23it/s, loss=0.241]
Test Epoch 48 ==> 	accuracy: 0.9239, 	precision: 0.7602, 	recall: 0.9072, 	specificity: 0.9281, 	f1: 0.8272
Train Epoch 49: 100%|██████████| 6195/6195 [05:55<00:00, 17.42it/s, loss=1.9]
Train Epoch 49 ==> 	accuracy: 0.9432, 	precision: 0.9436, 	recall: 0.9427, 	specificity: 0.9437, 	f1: 0.9431
Test Epoch 49: 100%|██████████| 1715/1715 [00:43<00:00, 39.26it/s, loss=0.177]
Test Epoch 49 ==> 	accuracy: 0.9259, 	precision: 0.7637, 	recall: 0.9135, 	specificity: 0.9290, 	f1: 0.8319

Process finished with exit code 0

/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Train Epoch 40: 100%|██████████| 6195/6195 [05:09<00:00, 19.99it/s, loss=2.21]
Train Epoch 40 ==> 	accuracy: 0.9478, 	precision: 0.9457, 	recall: 0.9501, 	specificity: 0.9455, 	f1: 0.9479
Test Epoch 40: 100%|██████████| 1715/1715 [00:35<00:00, 48.21it/s, loss=0.188]
Test Epoch 40 ==> 	accuracy: 0.9237, 	precision: 0.7501, 	recall: 0.9300, 	specificity: 0.9222, 	f1: 0.8304
Train Epoch 41: 100%|██████████| 6195/6195 [06:15<00:00, 16.48it/s, loss=1.99]
Train Epoch 41 ==> 	accuracy: 0.9395, 	precision: 0.9410, 	recall: 0.9378, 	specificity: 0.9412, 	f1: 0.9394
Test Epoch 41: 100%|██████████| 1715/1715 [00:40<00:00, 42.22it/s, loss=0.112]
Test Epoch 41 ==> 	accuracy: 0.9327, 	precision: 0.7912, 	recall: 0.9032, 	specificity: 0.9401, 	f1: 0.8435
Train Epoch 42: 100%|██████████| 6195/6195 [06:15<00:00, 16.48it/s, loss=1.79]
Train Epoch 42 ==> 	accuracy: 0.9463, 	precision: 0.9460, 	recall: 0.9465, 	specificity: 0.9460, 	f1: 0.9463
Test Epoch 42: 100%|██████████| 1715/1715 [00:47<00:00, 35.93it/s, loss=0.152]
Test Epoch 42 ==> 	accuracy: 0.9250, 	precision: 0.7579, 	recall: 0.9204, 	specificity: 0.9262, 	f1: 0.8313
Train Epoch 43: 100%|██████████| 6195/6195 [06:20<00:00, 16.30it/s, loss=1.91]
Train Epoch 43 ==> 	accuracy: 0.9460, 	precision: 0.9452, 	recall: 0.9468, 	specificity: 0.9451, 	f1: 0.9460
Test Epoch 43: 100%|██████████| 1715/1715 [00:49<00:00, 34.89it/s, loss=0.101]
Test Epoch 43 ==> 	accuracy: 0.9346, 	precision: 0.7947, 	recall: 0.9092, 	specificity: 0.9410, 	f1: 0.8481
Train Epoch 44: 100%|██████████| 6195/6195 [06:45<00:00, 15.29it/s, loss=1.94]
Train Epoch 44 ==> 	accuracy: 0.9444, 	precision: 0.9434, 	recall: 0.9456, 	specificity: 0.9433, 	f1: 0.9445
Test Epoch 44: 100%|██████████| 1715/1715 [00:45<00:00, 38.07it/s, loss=0.253]
Test Epoch 44 ==> 	accuracy: 0.9312, 	precision: 0.7821, 	recall: 0.9115, 	specificity: 0.9362, 	f1: 0.8418
Train Epoch 45: 100%|██████████| 6195/6195 [06:35<00:00, 15.68it/s, loss=1.8]
Train Epoch 45 ==> 	accuracy: 0.9467, 	precision: 0.9461, 	recall: 0.9473, 	specificity: 0.9460, 	f1: 0.9467
Test Epoch 45: 100%|██████████| 1715/1715 [00:40<00:00, 42.11it/s, loss=0.107]
Test Epoch 45 ==> 	accuracy: 0.9316, 	precision: 0.7823, 	recall: 0.9132, 	specificity: 0.9362, 	f1: 0.8427
Train Epoch 46: 100%|██████████| 6195/6195 [06:20<00:00, 16.28it/s, loss=2.06]
Train Epoch 46 ==> 	accuracy: 0.9472, 	precision: 0.9460, 	recall: 0.9485, 	specificity: 0.9459, 	f1: 0.9472
Test Epoch 46: 100%|██████████| 1715/1715 [00:46<00:00, 36.96it/s, loss=0.175]
Test Epoch 46 ==> 	accuracy: 0.9294, 	precision: 0.7750, 	recall: 0.9132, 	specificity: 0.9334, 	f1: 0.8385
Train Epoch 47: 100%|██████████| 6195/6195 [06:14<00:00, 16.52it/s, loss=1.8]
Train Epoch 47 ==> 	accuracy: 0.9504, 	precision: 0.9489, 	recall: 0.9521, 	specificity: 0.9487, 	f1: 0.9505
Test Epoch 47: 100%|██████████| 1715/1715 [00:46<00:00, 36.61it/s, loss=0.167]
Test Epoch 47 ==> 	accuracy: 0.9290, 	precision: 0.7762, 	recall: 0.9082, 	specificity: 0.9342, 	f1: 0.8370
Train Epoch 48: 100%|██████████| 6195/6195 [06:13<00:00, 16.58it/s, loss=1.52]
Train Epoch 48 ==> 	accuracy: 0.9455, 	precision: 0.9447, 	recall: 0.9465, 	specificity: 0.9446, 	f1: 0.9456
Test Epoch 48: 100%|██████████| 1715/1715 [00:40<00:00, 42.43it/s, loss=0.111]
Test Epoch 48 ==> 	accuracy: 0.9320, 	precision: 0.7846, 	recall: 0.9112, 	specificity: 0.9372, 	f1: 0.8432
Train Epoch 49: 100%|██████████| 6195/6195 [06:20<00:00, 16.29it/s, loss=2.23]
Train Epoch 49 ==> 	accuracy: 0.9499, 	precision: 0.9484, 	recall: 0.9517, 	specificity: 0.9482, 	f1: 0.9500
Test Epoch 49: 100%|██████████| 1715/1715 [00:44<00:00, 38.30it/s, loss=0.0834]
Test Epoch 49 ==> 	accuracy: 0.9342, 	precision: 0.7916, 	recall: 0.9122, 	specificity: 0.9397, 	f1: 0.8476
Train Epoch 50: 100%|██████████| 6195/6195 [06:18<00:00, 16.37it/s, loss=1.73]
Train Epoch 50 ==> 	accuracy: 0.9510, 	precision: 0.9481, 	recall: 0.9543, 	specificity: 0.9478, 	f1: 0.9512
Test Epoch 50: 100%|██████████| 1715/1715 [00:46<00:00, 36.97it/s, loss=0.105]
Test Epoch 50 ==> 	accuracy: 0.9405, 	precision: 0.8133, 	recall: 0.9132, 	specificity: 0.9473, 	f1: 0.8604
Train Epoch 51: 100%|██████████| 6195/6195 [06:16<00:00, 16.47it/s, loss=1.97]
Train Epoch 51 ==> 	accuracy: 0.9488, 	precision: 0.9480, 	recall: 0.9496, 	specificity: 0.9479, 	f1: 0.9488
Test Epoch 51: 100%|██████████| 1715/1715 [00:41<00:00, 41.50it/s, loss=0.0898]
Test Epoch 51 ==> 	accuracy: 0.9460, 	precision: 0.8387, 	recall: 0.9050, 	specificity: 0.9563, 	f1: 0.8706
Train Epoch 52: 100%|██████████| 6195/6195 [06:30<00:00, 15.88it/s, loss=1.87]
Train Epoch 52 ==> 	accuracy: 0.9492, 	precision: 0.9469, 	recall: 0.9518, 	specificity: 0.9467, 	f1: 0.9494
Test Epoch 52: 100%|██████████| 1715/1715 [00:41<00:00, 41.60it/s, loss=0.175]
Test Epoch 52 ==> 	accuracy: 0.9460, 	precision: 0.8392, 	recall: 0.9042, 	specificity: 0.9565, 	f1: 0.8705
Train Epoch 53: 100%|██████████| 6195/6195 [06:18<00:00, 16.35it/s, loss=1.61]
Train Epoch 53 ==> 	accuracy: 0.9519, 	precision: 0.9493, 	recall: 0.9547, 	specificity: 0.9490, 	f1: 0.9520
Test Epoch 53: 100%|██████████| 1715/1715 [00:44<00:00, 38.67it/s, loss=0.117]
Test Epoch 53 ==> 	accuracy: 0.9449, 	precision: 0.8310, 	recall: 0.9110, 	specificity: 0.9535, 	f1: 0.8691
Train Epoch 54: 100%|██████████| 6195/6195 [06:18<00:00, 16.38it/s, loss=1.79]
Train Epoch 54 ==> 	accuracy: 0.9512, 	precision: 0.9491, 	recall: 0.9536, 	specificity: 0.9489, 	f1: 0.9513
Test Epoch 54: 100%|██████████| 1715/1715 [00:40<00:00, 42.15it/s, loss=0.131]
Test Epoch 54 ==> 	accuracy: 0.9443, 	precision: 0.8269, 	recall: 0.9138, 	specificity: 0.9520, 	f1: 0.8682
Train Epoch 55: 100%|██████████| 6195/6195 [06:24<00:00, 16.12it/s, loss=1.84]
Train Epoch 55 ==> 	accuracy: 0.9493, 	precision: 0.9479, 	recall: 0.9508, 	specificity: 0.9478, 	f1: 0.9493
Test Epoch 55: 100%|██████████| 1715/1715 [00:41<00:00, 41.20it/s, loss=0.0991]
Test Epoch 55 ==> 	accuracy: 0.9490, 	precision: 0.8493, 	recall: 0.9069, 	specificity: 0.9596, 	f1: 0.8772
Train Epoch 56: 100%|██████████| 6195/6195 [06:19<00:00, 16.34it/s, loss=1.6]
Train Epoch 56 ==> 	accuracy: 0.9517, 	precision: 0.9491, 	recall: 0.9547, 	specificity: 0.9488, 	f1: 0.9519
Test Epoch 56: 100%|██████████| 1715/1715 [00:41<00:00, 41.72it/s, loss=0.122]
Test Epoch 56 ==> 	accuracy: 0.9464, 	precision: 0.8350, 	recall: 0.9137, 	specificity: 0.9547, 	f1: 0.8726
Train Epoch 57: 100%|██████████| 6195/6195 [06:20<00:00, 16.28it/s, loss=1.65]
Train Epoch 57 ==> 	accuracy: 0.9517, 	precision: 0.9489, 	recall: 0.9550, 	specificity: 0.9485, 	f1: 0.9519
Test Epoch 57: 100%|██████████| 1715/1715 [00:42<00:00, 40.60it/s, loss=0.154]
Test Epoch 57 ==> 	accuracy: 0.9466, 	precision: 0.8375, 	recall: 0.9111, 	specificity: 0.9556, 	f1: 0.8727
Train Epoch 58: 100%|██████████| 6195/6195 [06:17<00:00, 16.39it/s, loss=1.62]
Train Epoch 58 ==> 	accuracy: 0.9510, 	precision: 0.9497, 	recall: 0.9524, 	specificity: 0.9496, 	f1: 0.9510
Test Epoch 58: 100%|██████████| 1715/1715 [00:45<00:00, 38.09it/s, loss=0.0626]
Test Epoch 58 ==> 	accuracy: 0.9487, 	precision: 0.8485, 	recall: 0.9066, 	specificity: 0.9593, 	f1: 0.8766
Train Epoch 59: 100%|██████████| 6195/6195 [06:24<00:00, 16.10it/s, loss=1.85]
Train Epoch 59 ==> 	accuracy: 0.9512, 	precision: 0.9478, 	recall: 0.9549, 	specificity: 0.9474, 	f1: 0.9513
Test Epoch 59: 100%|██████████| 1715/1715 [00:46<00:00, 37.18it/s, loss=0.157]
Test Epoch 59 ==> 	accuracy: 0.9468, 	precision: 0.8399, 	recall: 0.9084, 	specificity: 0.9565, 	f1: 0.8728

Process finished with exit code 0

'''

'''
feat med

/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Train Epoch 0: 100%|██████████| 6195/6195 [06:14<00:00, 16.54it/s, loss=0.77]
Train Epoch 0 ==> 	accuracy: 0.8776, 	precision: 0.8855, 	recall: 0.8673, 	specificity: 0.8879, 	f1: 0.8763
Test Epoch 0: 100%|██████████| 1715/1715 [00:41<00:00, 41.01it/s, loss=0.28]
Test Epoch 0 ==> 	accuracy: 0.8849, 	precision: 0.6648, 	recall: 0.8611, 	specificity: 0.8909, 	f1: 0.7503
Train Epoch 1: 100%|██████████| 6195/6195 [06:16<00:00, 16.45it/s, loss=0.67]
Train Epoch 1 ==> 	accuracy: 0.8988, 	precision: 0.9074, 	recall: 0.8882, 	specificity: 0.9094, 	f1: 0.8977
Test Epoch 1: 100%|██████████| 1715/1715 [00:45<00:00, 38.11it/s, loss=0.286]
Test Epoch 1 ==> 	accuracy: 0.8960, 	precision: 0.6888, 	recall: 0.8791, 	specificity: 0.9002, 	f1: 0.7724
Train Epoch 2: 100%|██████████| 6195/6195 [06:19<00:00, 16.32it/s, loss=0.585]
Train Epoch 2 ==> 	accuracy: 0.9080, 	precision: 0.9136, 	recall: 0.9013, 	specificity: 0.9147, 	f1: 0.9074
Test Epoch 2: 100%|██████████| 1715/1715 [00:45<00:00, 37.53it/s, loss=0.0799]
Test Epoch 2 ==> 	accuracy: 0.8974, 	precision: 0.6907, 	recall: 0.8856, 	specificity: 0.9004, 	f1: 0.7761
Train Epoch 3: 100%|██████████| 6195/6195 [06:17<00:00, 16.41it/s, loss=0.614]
Train Epoch 3 ==> 	accuracy: 0.9140, 	precision: 0.9187, 	recall: 0.9084, 	specificity: 0.9196, 	f1: 0.9135
Test Epoch 3: 100%|██████████| 1715/1715 [00:46<00:00, 36.62it/s, loss=0.23]
Test Epoch 3 ==> 	accuracy: 0.8963, 	precision: 0.6827, 	recall: 0.9030, 	specificity: 0.8946, 	f1: 0.7775
Train Epoch 4: 100%|██████████| 6195/6195 [06:19<00:00, 16.34it/s, loss=0.569]
Train Epoch 4 ==> 	accuracy: 0.9206, 	precision: 0.9253, 	recall: 0.9151, 	specificity: 0.9262, 	f1: 0.9202
Test Epoch 4: 100%|██████████| 1715/1715 [00:43<00:00, 39.00it/s, loss=0.262]
Test Epoch 4 ==> 	accuracy: 0.8984, 	precision: 0.6841, 	recall: 0.9173, 	specificity: 0.8936, 	f1: 0.7837
Train Epoch 5: 100%|██████████| 6195/6195 [06:21<00:00, 16.23it/s, loss=0.694]
Train Epoch 5 ==> 	accuracy: 0.9208, 	precision: 0.9253, 	recall: 0.9155, 	specificity: 0.9261, 	f1: 0.9204
Test Epoch 5: 100%|██████████| 1715/1715 [00:43<00:00, 39.04it/s, loss=0.205]
Test Epoch 5 ==> 	accuracy: 0.9066, 	precision: 0.7131, 	recall: 0.8947, 	specificity: 0.9096, 	f1: 0.7936
Train Epoch 6: 100%|██████████| 6195/6195 [06:16<00:00, 16.48it/s, loss=0.607]
Train Epoch 6 ==> 	accuracy: 0.9214, 	precision: 0.9257, 	recall: 0.9163, 	specificity: 0.9264, 	f1: 0.9210
Test Epoch 6: 100%|██████████| 1715/1715 [00:43<00:00, 39.49it/s, loss=0.31]
Test Epoch 6 ==> 	accuracy: 0.9064, 	precision: 0.7101, 	recall: 0.9022, 	specificity: 0.9075, 	f1: 0.7947
Train Epoch 7: 100%|██████████| 6195/6195 [06:10<00:00, 16.71it/s, loss=0.577]
Train Epoch 7 ==> 	accuracy: 0.9282, 	precision: 0.9310, 	recall: 0.9249, 	specificity: 0.9314, 	f1: 0.9279
Test Epoch 7: 100%|██████████| 1715/1715 [00:43<00:00, 39.20it/s, loss=0.163]
Test Epoch 7 ==> 	accuracy: 0.9181, 	precision: 0.7420, 	recall: 0.9076, 	specificity: 0.9207, 	f1: 0.8165
Train Epoch 8: 100%|██████████| 6195/6195 [06:17<00:00, 16.39it/s, loss=0.629]
Train Epoch 8 ==> 	accuracy: 0.9229, 	precision: 0.9279, 	recall: 0.9171, 	specificity: 0.9287, 	f1: 0.9225
Test Epoch 8: 100%|██████████| 1715/1715 [00:41<00:00, 40.92it/s, loss=0.188]
Test Epoch 8 ==> 	accuracy: 0.9091, 	precision: 0.7164, 	recall: 0.9054, 	specificity: 0.9100, 	f1: 0.7999
Train Epoch 9: 100%|██████████| 6195/6195 [06:10<00:00, 16.74it/s, loss=0.609]
Train Epoch 9 ==> 	accuracy: 0.9273, 	precision: 0.9312, 	recall: 0.9226, 	specificity: 0.9319, 	f1: 0.9269
Test Epoch 9: 100%|██████████| 1715/1715 [00:43<00:00, 39.10it/s, loss=0.268]
Test Epoch 9 ==> 	accuracy: 0.9156, 	precision: 0.7365, 	recall: 0.9022, 	specificity: 0.9189, 	f1: 0.8110
Train Epoch 10: 100%|██████████| 6195/6195 [06:23<00:00, 16.15it/s, loss=0.558]
Train Epoch 10 ==> 	accuracy: 0.9296, 	precision: 0.9322, 	recall: 0.9265, 	specificity: 0.9326, 	f1: 0.9294
Test Epoch 10: 100%|██████████| 1715/1715 [00:44<00:00, 38.52it/s, loss=0.154]
Test Epoch 10 ==> 	accuracy: 0.8993, 	precision: 0.6863, 	recall: 0.9183, 	specificity: 0.8945, 	f1: 0.7855
Train Epoch 11: 100%|██████████| 6195/6195 [06:26<00:00, 16.01it/s, loss=0.617]
Train Epoch 11 ==> 	accuracy: 0.9300, 	precision: 0.9335, 	recall: 0.9261, 	specificity: 0.9340, 	f1: 0.9298
Test Epoch 11: 100%|██████████| 1715/1715 [00:40<00:00, 42.22it/s, loss=0.184]
Epoch 00012: reducing learning rate of group 0 to 5.0000e-05.
Test Epoch 11 ==> 	accuracy: 0.8992, 	precision: 0.6876, 	recall: 0.9121, 	specificity: 0.8959, 	f1: 0.7841
Train Epoch 12: 100%|██████████| 6195/6195 [06:09<00:00, 16.78it/s, loss=0.588]
Train Epoch 12 ==> 	accuracy: 0.9320, 	precision: 0.9349, 	recall: 0.9286, 	specificity: 0.9353, 	f1: 0.9317
Test Epoch 12: 100%|██████████| 1715/1715 [00:44<00:00, 38.86it/s, loss=0.165]
Test Epoch 12 ==> 	accuracy: 0.9274, 	precision: 0.7728, 	recall: 0.9045, 	specificity: 0.9332, 	f1: 0.8335
Train Epoch 13: 100%|██████████| 6195/6195 [06:19<00:00, 16.34it/s, loss=0.548]
Train Epoch 13 ==> 	accuracy: 0.9338, 	precision: 0.9368, 	recall: 0.9303, 	specificity: 0.9372, 	f1: 0.9335
Test Epoch 13: 100%|██████████| 1715/1715 [00:42<00:00, 40.11it/s, loss=0.206]
Test Epoch 13 ==> 	accuracy: 0.9192, 	precision: 0.7438, 	recall: 0.9112, 	specificity: 0.9212, 	f1: 0.8190
Train Epoch 14: 100%|██████████| 6195/6195 [06:07<00:00, 16.86it/s, loss=0.575]
Train Epoch 14 ==> 	accuracy: 0.9397, 	precision: 0.9409, 	recall: 0.9384, 	specificity: 0.9410, 	f1: 0.9396
Test Epoch 14: 100%|██████████| 1715/1715 [00:43<00:00, 39.14it/s, loss=0.182]
Test Epoch 14 ==> 	accuracy: 0.9214, 	precision: 0.7494, 	recall: 0.9141, 	specificity: 0.9232, 	f1: 0.8236
Train Epoch 15: 100%|██████████| 6195/6195 [06:07<00:00, 16.86it/s, loss=0.553]
Train Epoch 15 ==> 	accuracy: 0.9333, 	precision: 0.9372, 	recall: 0.9290, 	specificity: 0.9377, 	f1: 0.9331
Test Epoch 15: 100%|██████████| 1715/1715 [00:45<00:00, 37.50it/s, loss=0.187]
Test Epoch 15 ==> 	accuracy: 0.9203, 	precision: 0.7520, 	recall: 0.8997, 	specificity: 0.9255, 	f1: 0.8192
Train Epoch 16: 100%|██████████| 6195/6195 [06:15<00:00, 16.50it/s, loss=0.587]
Train Epoch 16 ==> 	accuracy: 0.9365, 	precision: 0.9386, 	recall: 0.9341, 	specificity: 0.9389, 	f1: 0.9363
Test Epoch 16: 100%|██████████| 1715/1715 [00:41<00:00, 41.74it/s, loss=0.176]
Test Epoch 16 ==> 	accuracy: 0.9286, 	precision: 0.7742, 	recall: 0.9094, 	specificity: 0.9334, 	f1: 0.8364
Train Epoch 17: 100%|██████████| 6195/6195 [06:24<00:00, 16.12it/s, loss=0.631]
Train Epoch 17 ==> 	accuracy: 0.9399, 	precision: 0.9413, 	recall: 0.9382, 	specificity: 0.9415, 	f1: 0.9398
Test Epoch 17: 100%|██████████| 1715/1715 [00:41<00:00, 41.66it/s, loss=0.232]
Test Epoch 17 ==> 	accuracy: 0.9255, 	precision: 0.7618, 	recall: 0.9151, 	specificity: 0.9281, 	f1: 0.8315
Train Epoch 18: 100%|██████████| 6195/6195 [06:12<00:00, 16.61it/s, loss=0.543]
Train Epoch 18 ==> 	accuracy: 0.9368, 	precision: 0.9396, 	recall: 0.9337, 	specificity: 0.9399, 	f1: 0.9366
Test Epoch 18: 100%|██████████| 1715/1715 [00:43<00:00, 39.38it/s, loss=0.189]
Test Epoch 18 ==> 	accuracy: 0.9288, 	precision: 0.7747, 	recall: 0.9098, 	specificity: 0.9336, 	f1: 0.8369
Train Epoch 19: 100%|██████████| 6195/6195 [06:12<00:00, 16.64it/s, loss=0.617]
Train Epoch 19 ==> 	accuracy: 0.9372, 	precision: 0.9389, 	recall: 0.9352, 	specificity: 0.9391, 	f1: 0.9371
Test Epoch 19: 100%|██████████| 1715/1715 [00:44<00:00, 38.68it/s, loss=0.253]
Test Epoch 19 ==> 	accuracy: 0.9298, 	precision: 0.7814, 	recall: 0.9030, 	specificity: 0.9365, 	f1: 0.8378
Train Epoch 20: 100%|██████████| 6195/6195 [06:22<00:00, 16.20it/s, loss=0.468]
Train Epoch 20 ==> 	accuracy: 0.9394, 	precision: 0.9414, 	recall: 0.9371, 	specificity: 0.9416, 	f1: 0.9392
Test Epoch 20: 100%|██████████| 1715/1715 [00:41<00:00, 41.27it/s, loss=0.146]
Test Epoch 20 ==> 	accuracy: 0.9235, 	precision: 0.7572, 	recall: 0.9111, 	specificity: 0.9266, 	f1: 0.8271
Train Epoch 21: 100%|██████████| 6195/6195 [06:17<00:00, 16.40it/s, loss=0.541]
Train Epoch 21 ==> 	accuracy: 0.9411, 	precision: 0.9419, 	recall: 0.9402, 	specificity: 0.9420, 	f1: 0.9410
Test Epoch 21: 100%|██████████| 1715/1715 [00:44<00:00, 38.41it/s, loss=0.121]
Test Epoch 21 ==> 	accuracy: 0.9251, 	precision: 0.7637, 	recall: 0.9080, 	specificity: 0.9294, 	f1: 0.8296
Train Epoch 22: 100%|██████████| 6195/6195 [06:11<00:00, 16.68it/s, loss=0.535]
Train Epoch 22 ==> 	accuracy: 0.9369, 	precision: 0.9395, 	recall: 0.9339, 	specificity: 0.9399, 	f1: 0.9367
Test Epoch 22: 100%|██████████| 1715/1715 [00:43<00:00, 39.17it/s, loss=0.251]
Test Epoch 22 ==> 	accuracy: 0.9217, 	precision: 0.7542, 	recall: 0.9047, 	specificity: 0.9259, 	f1: 0.8226
Train Epoch 23: 100%|██████████| 6195/6195 [06:07<00:00, 16.87it/s, loss=0.504]
Train Epoch 23 ==> 	accuracy: 0.9426, 	precision: 0.9426, 	recall: 0.9426, 	specificity: 0.9426, 	f1: 0.9426
Test Epoch 23: 100%|██████████| 1715/1715 [00:44<00:00, 38.21it/s, loss=0.202]
Epoch 00024: reducing learning rate of group 0 to 2.5000e-05.
Test Epoch 23 ==> 	accuracy: 0.9164, 	precision: 0.7357, 	recall: 0.9111, 	specificity: 0.9178, 	f1: 0.8140
Train Epoch 24: 100%|██████████| 6195/6195 [06:23<00:00, 16.15it/s, loss=0.526]
Train Epoch 24 ==> 	accuracy: 0.9423, 	precision: 0.9435, 	recall: 0.9408, 	specificity: 0.9437, 	f1: 0.9422
Test Epoch 24: 100%|██████████| 1715/1715 [00:42<00:00, 40.36it/s, loss=0.17]
Test Epoch 24 ==> 	accuracy: 0.9294, 	precision: 0.7738, 	recall: 0.9160, 	specificity: 0.9327, 	f1: 0.8389
Train Epoch 25: 100%|██████████| 6195/6195 [06:12<00:00, 16.62it/s, loss=0.467]
Train Epoch 25 ==> 	accuracy: 0.9436, 	precision: 0.9447, 	recall: 0.9423, 	specificity: 0.9449, 	f1: 0.9435
Test Epoch 25: 100%|██████████| 1715/1715 [00:42<00:00, 40.21it/s, loss=0.144]
Test Epoch 25 ==> 	accuracy: 0.9265, 	precision: 0.7649, 	recall: 0.9154, 	specificity: 0.9293, 	f1: 0.8334
Train Epoch 26: 100%|██████████| 6195/6195 [06:11<00:00, 16.67it/s, loss=0.617]
Train Epoch 26 ==> 	accuracy: 0.9421, 	precision: 0.9425, 	recall: 0.9415, 	specificity: 0.9426, 	f1: 0.9420
Test Epoch 26: 100%|██████████| 1715/1715 [00:39<00:00, 43.05it/s, loss=0.272]
Test Epoch 26 ==> 	accuracy: 0.9286, 	precision: 0.7744, 	recall: 0.9088, 	specificity: 0.9335, 	f1: 0.8363
Train Epoch 27: 100%|██████████| 6195/6195 [06:13<00:00, 16.59it/s, loss=0.513]
Train Epoch 27 ==> 	accuracy: 0.9480, 	precision: 0.9481, 	recall: 0.9478, 	specificity: 0.9482, 	f1: 0.9480
Test Epoch 27: 100%|██████████| 1715/1715 [00:45<00:00, 37.42it/s, loss=0.13]
Epoch 00028: reducing learning rate of group 0 to 1.2500e-05.
Test Epoch 27 ==> 	accuracy: 0.9262, 	precision: 0.7634, 	recall: 0.9166, 	specificity: 0.9286, 	f1: 0.8330
Train Epoch 28: 100%|██████████| 6195/6195 [06:07<00:00, 16.84it/s, loss=0.486]
Train Epoch 28 ==> 	accuracy: 0.9436, 	precision: 0.9451, 	recall: 0.9419, 	specificity: 0.9453, 	f1: 0.9435
Test Epoch 28: 100%|██████████| 1715/1715 [00:40<00:00, 41.91it/s, loss=0.157]
Test Epoch 28 ==> 	accuracy: 0.9288, 	precision: 0.7730, 	recall: 0.9133, 	specificity: 0.9326, 	f1: 0.8373
Train Epoch 29: 100%|██████████| 6195/6195 [06:22<00:00, 16.21it/s, loss=0.591]
Train Epoch 29 ==> 	accuracy: 0.9447, 	precision: 0.9448, 	recall: 0.9447, 	specificity: 0.9448, 	f1: 0.9447
Test Epoch 29: 100%|██████████| 1715/1715 [00:41<00:00, 41.10it/s, loss=0.205]
Test Epoch 29 ==> 	accuracy: 0.9314, 	precision: 0.7834, 	recall: 0.9098, 	specificity: 0.9368, 	f1: 0.8419
Train Epoch 30: 100%|██████████| 6195/6195 [06:13<00:00, 16.60it/s, loss=0.437]
Train Epoch 30 ==> 	accuracy: 0.9452, 	precision: 0.9463, 	recall: 0.9439, 	specificity: 0.9464, 	f1: 0.9451
Test Epoch 30: 100%|██████████| 1715/1715 [00:42<00:00, 40.48it/s, loss=0.175]
Test Epoch 30 ==> 	accuracy: 0.9277, 	precision: 0.7711, 	recall: 0.9103, 	specificity: 0.9321, 	f1: 0.8349
Train Epoch 31: 100%|██████████| 6195/6195 [06:12<00:00, 16.62it/s, loss=0.591]
Train Epoch 31 ==> 	accuracy: 0.9469, 	precision: 0.9466, 	recall: 0.9473, 	specificity: 0.9465, 	f1: 0.9470
Test Epoch 31: 100%|██████████| 1715/1715 [00:43<00:00, 38.99it/s, loss=0.22]
Test Epoch 31 ==> 	accuracy: 0.9305, 	precision: 0.7795, 	recall: 0.9119, 	specificity: 0.9352, 	f1: 0.8405
Train Epoch 32: 100%|██████████| 6195/6195 [06:09<00:00, 16.76it/s, loss=0.455]
Train Epoch 32 ==> 	accuracy: 0.9434, 	precision: 0.9453, 	recall: 0.9412, 	specificity: 0.9456, 	f1: 0.9432
Test Epoch 32: 100%|██████████| 1715/1715 [00:45<00:00, 37.58it/s, loss=0.195]
Test Epoch 32 ==> 	accuracy: 0.9333, 	precision: 0.7917, 	recall: 0.9062, 	specificity: 0.9401, 	f1: 0.8451
Train Epoch 33: 100%|██████████| 6195/6195 [06:09<00:00, 16.75it/s, loss=0.498]
Train Epoch 33 ==> 	accuracy: 0.9462, 	precision: 0.9454, 	recall: 0.9471, 	specificity: 0.9453, 	f1: 0.9463
Test Epoch 33: 100%|██████████| 1715/1715 [00:43<00:00, 39.36it/s, loss=0.228]
Test Epoch 33 ==> 	accuracy: 0.9316, 	precision: 0.7845, 	recall: 0.9091, 	specificity: 0.9373, 	f1: 0.8422
Train Epoch 34: 100%|██████████| 6195/6195 [06:19<00:00, 16.33it/s, loss=0.474]
Train Epoch 34 ==> 	accuracy: 0.9470, 	precision: 0.9479, 	recall: 0.9460, 	specificity: 0.9480, 	f1: 0.9469
Test Epoch 34: 100%|██████████| 1715/1715 [00:45<00:00, 37.69it/s, loss=0.118]
Test Epoch 34 ==> 	accuracy: 0.9286, 	precision: 0.7736, 	recall: 0.9113, 	specificity: 0.9330, 	f1: 0.8368
Train Epoch 35: 100%|██████████| 6195/6195 [06:14<00:00, 16.56it/s, loss=0.458]
Train Epoch 35 ==> 	accuracy: 0.9453, 	precision: 0.9456, 	recall: 0.9449, 	specificity: 0.9457, 	f1: 0.9453
Test Epoch 35: 100%|██████████| 1715/1715 [00:42<00:00, 39.89it/s, loss=0.187]
Test Epoch 35 ==> 	accuracy: 0.9285, 	precision: 0.7740, 	recall: 0.9097, 	specificity: 0.9333, 	f1: 0.8364
Train Epoch 36: 100%|██████████| 6195/6195 [06:15<00:00, 16.49it/s, loss=0.438]
Train Epoch 36 ==> 	accuracy: 0.9485, 	precision: 0.9484, 	recall: 0.9485, 	specificity: 0.9484, 	f1: 0.9485
Test Epoch 36: 100%|██████████| 1715/1715 [00:47<00:00, 36.35it/s, loss=0.225]
Epoch 00037: reducing learning rate of group 0 to 6.2500e-06.
Test Epoch 36 ==> 	accuracy: 0.9312, 	precision: 0.7818, 	recall: 0.9116, 	specificity: 0.9361, 	f1: 0.8417
Train Epoch 37: 100%|██████████| 6195/6195 [06:16<00:00, 16.45it/s, loss=0.468]
Train Epoch 37 ==> 	accuracy: 0.9459, 	precision: 0.9465, 	recall: 0.9453, 	specificity: 0.9466, 	f1: 0.9459
Test Epoch 37: 100%|██████████| 1715/1715 [00:44<00:00, 38.94it/s, loss=0.201]
Test Epoch 37 ==> 	accuracy: 0.9322, 	precision: 0.7868, 	recall: 0.9088, 	specificity: 0.9381, 	f1: 0.8434
Train Epoch 38: 100%|██████████| 6195/6195 [06:20<00:00, 16.27it/s, loss=0.375]
Train Epoch 38 ==> 	accuracy: 0.9492, 	precision: 0.9487, 	recall: 0.9496, 	specificity: 0.9487, 	f1: 0.9492
Test Epoch 38: 100%|██████████| 1715/1715 [00:43<00:00, 39.18it/s, loss=0.0594]
Test Epoch 38 ==> 	accuracy: 0.9291, 	precision: 0.7748, 	recall: 0.9118, 	specificity: 0.9334, 	f1: 0.8377
Train Epoch 39: 100%|██████████| 6195/6195 [06:16<00:00, 16.47it/s, loss=0.597]
Train Epoch 39 ==> 	accuracy: 0.9459, 	precision: 0.9470, 	recall: 0.9447, 	specificity: 0.9471, 	f1: 0.9458
Test Epoch 39: 100%|██████████| 1715/1715 [00:41<00:00, 41.21it/s, loss=0.163]
Test Epoch 39 ==> 	accuracy: 0.9322, 	precision: 0.7864, 	recall: 0.9095, 	specificity: 0.9379, 	f1: 0.8435

Process finished with exit code 0

/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Train Epoch 40: 100%|██████████| 6195/6195 [04:47<00:00, 21.53it/s, loss=0.501]
Train Epoch 40 ==> 	accuracy: 0.9451, 	precision: 0.9450, 	recall: 0.9454, 	specificity: 0.9449, 	f1: 0.9452
Test Epoch 40: 100%|██████████| 1715/1715 [00:34<00:00, 50.09it/s, loss=0.156]
Test Epoch 40 ==> 	accuracy: 0.9274, 	precision: 0.7698, 	recall: 0.9103, 	specificity: 0.9316, 	f1: 0.8342
Train Epoch 41: 100%|██████████| 6195/6195 [05:50<00:00, 17.66it/s, loss=0.56]
Train Epoch 41 ==> 	accuracy: 0.9355, 	precision: 0.9389, 	recall: 0.9317, 	specificity: 0.9394, 	f1: 0.9353
Test Epoch 41: 100%|██████████| 1715/1715 [00:40<00:00, 41.89it/s, loss=0.292]
Test Epoch 41 ==> 	accuracy: 0.9245, 	precision: 0.7616, 	recall: 0.9085, 	specificity: 0.9286, 	f1: 0.8286
Train Epoch 42: 100%|██████████| 6195/6195 [06:00<00:00, 17.20it/s, loss=0.573]
Train Epoch 42 ==> 	accuracy: 0.9417, 	precision: 0.9425, 	recall: 0.9407, 	specificity: 0.9426, 	f1: 0.9416
Test Epoch 42: 100%|██████████| 1715/1715 [00:40<00:00, 42.12it/s, loss=0.143]
Test Epoch 42 ==> 	accuracy: 0.9225, 	precision: 0.7536, 	recall: 0.9123, 	specificity: 0.9251, 	f1: 0.8254
Train Epoch 43: 100%|██████████| 6195/6195 [05:59<00:00, 17.25it/s, loss=0.513]
Train Epoch 43 ==> 	accuracy: 0.9407, 	precision: 0.9413, 	recall: 0.9399, 	specificity: 0.9414, 	f1: 0.9406
Test Epoch 43: 100%|██████████| 1715/1715 [00:43<00:00, 39.75it/s, loss=0.144]
Test Epoch 43 ==> 	accuracy: 0.9272, 	precision: 0.7679, 	recall: 0.9133, 	specificity: 0.9306, 	f1: 0.8343
Train Epoch 44: 100%|██████████| 6195/6195 [06:08<00:00, 16.82it/s, loss=0.545]
Train Epoch 44 ==> 	accuracy: 0.9392, 	precision: 0.9403, 	recall: 0.9379, 	specificity: 0.9405, 	f1: 0.9391
Test Epoch 44: 100%|██████████| 1715/1715 [00:39<00:00, 42.93it/s, loss=0.256]
Epoch 00005: reducing learning rate of group 0 to 5.0000e-05.
Test Epoch 44 ==> 	accuracy: 0.9218, 	precision: 0.7526, 	recall: 0.9098, 	specificity: 0.9249, 	f1: 0.8237
Train Epoch 45: 100%|██████████| 6195/6195 [06:08<00:00, 16.82it/s, loss=0.51]
Train Epoch 45 ==> 	accuracy: 0.9440, 	precision: 0.9451, 	recall: 0.9427, 	specificity: 0.9452, 	f1: 0.9439
Test Epoch 45: 100%|██████████| 1715/1715 [00:42<00:00, 40.69it/s, loss=0.146]
Test Epoch 45 ==> 	accuracy: 0.9214, 	precision: 0.7499, 	recall: 0.9132, 	specificity: 0.9235, 	f1: 0.8235
Train Epoch 46: 100%|██████████| 6195/6195 [06:06<00:00, 16.92it/s, loss=0.523]
Train Epoch 46 ==> 	accuracy: 0.9456, 	precision: 0.9457, 	recall: 0.9455, 	specificity: 0.9457, 	f1: 0.9456
Test Epoch 46: 100%|██████████| 1715/1715 [00:42<00:00, 40.33it/s, loss=0.197]
Test Epoch 46 ==> 	accuracy: 0.9124, 	precision: 0.7212, 	recall: 0.9186, 	specificity: 0.9108, 	f1: 0.8080
Train Epoch 47: 100%|██████████| 6195/6195 [06:11<00:00, 16.70it/s, loss=0.432]
Train Epoch 47 ==> 	accuracy: 0.9483, 	precision: 0.9486, 	recall: 0.9481, 	specificity: 0.9486, 	f1: 0.9483
Test Epoch 47: 100%|██████████| 1715/1715 [00:43<00:00, 39.04it/s, loss=0.174]
Test Epoch 47 ==> 	accuracy: 0.9213, 	precision: 0.7505, 	recall: 0.9104, 	specificity: 0.9240, 	f1: 0.8228
Train Epoch 48: 100%|██████████| 6195/6195 [06:09<00:00, 16.75it/s, loss=0.45]
Train Epoch 48 ==> 	accuracy: 0.9432, 	precision: 0.9440, 	recall: 0.9423, 	specificity: 0.9441, 	f1: 0.9431
Test Epoch 48: 100%|██████████| 1715/1715 [00:43<00:00, 39.52it/s, loss=0.165]
Epoch 00009: reducing learning rate of group 0 to 2.5000e-05.
Test Epoch 48 ==> 	accuracy: 0.9266, 	precision: 0.7692, 	recall: 0.9062, 	specificity: 0.9317, 	f1: 0.8321
Train Epoch 49: 100%|██████████| 6195/6195 [06:07<00:00, 16.85it/s, loss=0.41]
Train Epoch 49 ==> 	accuracy: 0.9496, 	precision: 0.9496, 	recall: 0.9496, 	specificity: 0.9496, 	f1: 0.9496
Test Epoch 49: 100%|██████████| 1715/1715 [00:44<00:00, 38.51it/s, loss=0.142]
Test Epoch 49 ==> 	accuracy: 0.9323, 	precision: 0.7873, 	recall: 0.9083, 	specificity: 0.9384, 	f1: 0.8435
Train Epoch 50: 100%|██████████| 6195/6195 [06:07<00:00, 16.86it/s, loss=0.592]
Train Epoch 50 ==> 	accuracy: 0.9510, 	precision: 0.9500, 	recall: 0.9522, 	specificity: 0.9498, 	f1: 0.9511
Test Epoch 50: 100%|██████████| 1715/1715 [00:44<00:00, 38.39it/s, loss=0.205]
Test Epoch 50 ==> 	accuracy: 0.9345, 	precision: 0.7921, 	recall: 0.9136, 	specificity: 0.9398, 	f1: 0.8485
Train Epoch 51: 100%|██████████| 6195/6195 [06:06<00:00, 16.92it/s, loss=0.509]
Train Epoch 51 ==> 	accuracy: 0.9484, 	precision: 0.9492, 	recall: 0.9476, 	specificity: 0.9493, 	f1: 0.9484
Test Epoch 51: 100%|██████████| 1715/1715 [00:40<00:00, 42.63it/s, loss=0.167]
Test Epoch 51 ==> 	accuracy: 0.9417, 	precision: 0.8202, 	recall: 0.9090, 	specificity: 0.9500, 	f1: 0.8623
Train Epoch 52: 100%|██████████| 6195/6195 [06:08<00:00, 16.81it/s, loss=0.577]
Train Epoch 52 ==> 	accuracy: 0.9490, 	precision: 0.9483, 	recall: 0.9497, 	specificity: 0.9482, 	f1: 0.9490
Test Epoch 52: 100%|██████████| 1715/1715 [00:43<00:00, 39.54it/s, loss=0.14]
Test Epoch 52 ==> 	accuracy: 0.9453, 	precision: 0.8362, 	recall: 0.9050, 	specificity: 0.9555, 	f1: 0.8692
Train Epoch 53: 100%|██████████| 6195/6195 [06:16<00:00, 16.44it/s, loss=0.448]
Train Epoch 53 ==> 	accuracy: 0.9514, 	precision: 0.9509, 	recall: 0.9519, 	specificity: 0.9509, 	f1: 0.9514
Test Epoch 53: 100%|██████████| 1715/1715 [00:39<00:00, 43.10it/s, loss=0.216]
Test Epoch 53 ==> 	accuracy: 0.9456, 	precision: 0.8361, 	recall: 0.9070, 	specificity: 0.9553, 	f1: 0.8701
Train Epoch 54: 100%|██████████| 6195/6195 [06:14<00:00, 16.52it/s, loss=0.435]
Train Epoch 54 ==> 	accuracy: 0.9507, 	precision: 0.9506, 	recall: 0.9508, 	specificity: 0.9506, 	f1: 0.9507
Test Epoch 54: 100%|██████████| 1715/1715 [00:43<00:00, 39.11it/s, loss=0.103]
Test Epoch 54 ==> 	accuracy: 0.9451, 	precision: 0.8330, 	recall: 0.9090, 	specificity: 0.9542, 	f1: 0.8693
Train Epoch 55: 100%|██████████| 6195/6195 [06:16<00:00, 16.47it/s, loss=0.439]
Train Epoch 55 ==> 	accuracy: 0.9488, 	precision: 0.9485, 	recall: 0.9492, 	specificity: 0.9484, 	f1: 0.9489
Test Epoch 55: 100%|██████████| 1715/1715 [00:42<00:00, 40.49it/s, loss=0.0889]
Test Epoch 55 ==> 	accuracy: 0.9465, 	precision: 0.8407, 	recall: 0.9050, 	specificity: 0.9569, 	f1: 0.8716
Train Epoch 56: 100%|██████████| 6195/6195 [06:11<00:00, 16.69it/s, loss=0.427]
Train Epoch 56 ==> 	accuracy: 0.9514, 	precision: 0.9507, 	recall: 0.9522, 	specificity: 0.9506, 	f1: 0.9515
Test Epoch 56: 100%|██████████| 1715/1715 [00:40<00:00, 42.70it/s, loss=0.145]
Test Epoch 56 ==> 	accuracy: 0.9442, 	precision: 0.8285, 	recall: 0.9108, 	specificity: 0.9527, 	f1: 0.8677
Train Epoch 57: 100%|██████████| 6195/6195 [06:13<00:00, 16.57it/s, loss=0.41]
Train Epoch 57 ==> 	accuracy: 0.9509, 	precision: 0.9500, 	recall: 0.9519, 	specificity: 0.9499, 	f1: 0.9509
Test Epoch 57: 100%|██████████| 1715/1715 [00:42<00:00, 40.43it/s, loss=0.247]
Test Epoch 57 ==> 	accuracy: 0.9412, 	precision: 0.8199, 	recall: 0.9061, 	specificity: 0.9500, 	f1: 0.8608
Train Epoch 58: 100%|██████████| 6195/6195 [06:16<00:00, 16.45it/s, loss=0.563]
Train Epoch 58 ==> 	accuracy: 0.9504, 	precision: 0.9507, 	recall: 0.9501, 	specificity: 0.9507, 	f1: 0.9504
Test Epoch 58: 100%|██████████| 1715/1715 [00:45<00:00, 37.51it/s, loss=0.194]
Test Epoch 58 ==> 	accuracy: 0.9417, 	precision: 0.8239, 	recall: 0.9024, 	specificity: 0.9515, 	f1: 0.8613
Train Epoch 59: 100%|██████████| 6195/6195 [06:23<00:00, 16.14it/s, loss=0.382]
Train Epoch 59 ==> 	accuracy: 0.9507, 	precision: 0.9495, 	recall: 0.9520, 	specificity: 0.9494, 	f1: 0.9508
Test Epoch 59: 100%|██████████| 1715/1715 [00:41<00:00, 41.19it/s, loss=0.0674]
Epoch 00020: reducing learning rate of group 0 to 1.2500e-05.
Test Epoch 59 ==> 	accuracy: 0.9423, 	precision: 0.8244, 	recall: 0.9058, 	specificity: 0.9515, 	f1: 0.8632

Process finished with exit code 0
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Train Epoch 60: 100%|██████████| 6195/6195 [05:51<00:00, 17.62it/s, loss=0.277]
Train Epoch 60 ==> 	accuracy: 0.9547, 	precision: 0.9543, 	recall: 0.9551, 	specificity: 0.9543, 	f1: 0.9547
Test Epoch 60: 100%|██████████| 1715/1715 [00:43<00:00, 39.08it/s, loss=0.0769]
Test Epoch 60 ==> 	accuracy: 0.9461, 	precision: 0.8337, 	recall: 0.9139, 	specificity: 0.9542, 	f1: 0.8720
Train Epoch 61: 100%|██████████| 6195/6195 [06:17<00:00, 16.42it/s, loss=0.251]
Train Epoch 61 ==> 	accuracy: 0.9514, 	precision: 0.9506, 	recall: 0.9523, 	specificity: 0.9505, 	f1: 0.9514
Test Epoch 61: 100%|██████████| 1715/1715 [00:46<00:00, 36.93it/s, loss=0.114]
Test Epoch 61 ==> 	accuracy: 0.9463, 	precision: 0.8402, 	recall: 0.9046, 	specificity: 0.9568, 	f1: 0.8712
Train Epoch 62: 100%|██████████| 6195/6195 [06:22<00:00, 16.20it/s, loss=0.295]
Train Epoch 62 ==> 	accuracy: 0.9522, 	precision: 0.9522, 	recall: 0.9523, 	specificity: 0.9522, 	f1: 0.9522
Test Epoch 62: 100%|██████████| 1715/1715 [00:45<00:00, 37.76it/s, loss=0.152]
Test Epoch 62 ==> 	accuracy: 0.9474, 	precision: 0.8446, 	recall: 0.9042, 	specificity: 0.9582, 	f1: 0.8734
Train Epoch 63: 100%|██████████| 6195/6195 [06:24<00:00, 16.10it/s, loss=0.27]
Train Epoch 63 ==> 	accuracy: 0.9537, 	precision: 0.9524, 	recall: 0.9551, 	specificity: 0.9523, 	f1: 0.9538
Test Epoch 63: 100%|██████████| 1715/1715 [00:46<00:00, 37.14it/s, loss=0.0902]
Test Epoch 63 ==> 	accuracy: 0.9439, 	precision: 0.8300, 	recall: 0.9062, 	specificity: 0.9534, 	f1: 0.8664
Train Epoch 64: 100%|██████████| 6195/6195 [06:23<00:00, 16.17it/s, loss=0.283]
Train Epoch 64 ==> 	accuracy: 0.9542, 	precision: 0.9537, 	recall: 0.9547, 	specificity: 0.9536, 	f1: 0.9542
Test Epoch 64: 100%|██████████| 1715/1715 [00:44<00:00, 38.32it/s, loss=0.15]
Test Epoch 64 ==> 	accuracy: 0.9438, 	precision: 0.8288, 	recall: 0.9076, 	specificity: 0.9529, 	f1: 0.8664
Train Epoch 65: 100%|██████████| 6195/6195 [06:30<00:00, 15.88it/s, loss=0.264]
Train Epoch 65 ==> 	accuracy: 0.9531, 	precision: 0.9528, 	recall: 0.9534, 	specificity: 0.9527, 	f1: 0.9531
Test Epoch 65: 100%|██████████| 1715/1715 [00:45<00:00, 37.33it/s, loss=0.126]
Test Epoch 65 ==> 	accuracy: 0.9470, 	precision: 0.8438, 	recall: 0.9034, 	specificity: 0.9580, 	f1: 0.8726
Train Epoch 66: 100%|██████████| 6195/6195 [06:30<00:00, 15.86it/s, loss=0.316]
Train Epoch 66 ==> 	accuracy: 0.9541, 	precision: 0.9525, 	recall: 0.9559, 	specificity: 0.9523, 	f1: 0.9542
Test Epoch 66: 100%|██████████| 1715/1715 [00:43<00:00, 39.57it/s, loss=0.181]
Test Epoch 66 ==> 	accuracy: 0.9479, 	precision: 0.8462, 	recall: 0.9052, 	specificity: 0.9587, 	f1: 0.8747
Train Epoch 67: 100%|██████████| 6195/6195 [06:29<00:00, 15.89it/s, loss=0.347]
Train Epoch 67 ==> 	accuracy: 0.9556, 	precision: 0.9539, 	recall: 0.9575, 	specificity: 0.9537, 	f1: 0.9557
Test Epoch 67: 100%|██████████| 1715/1715 [00:45<00:00, 37.29it/s, loss=0.149]
Test Epoch 67 ==> 	accuracy: 0.9449, 	precision: 0.8321, 	recall: 0.9093, 	specificity: 0.9539, 	f1: 0.8690
Train Epoch 68: 100%|██████████| 6195/6195 [06:25<00:00, 16.07it/s, loss=0.355]
Train Epoch 68 ==> 	accuracy: 0.9510, 	precision: 0.9506, 	recall: 0.9514, 	specificity: 0.9506, 	f1: 0.9510
Test Epoch 68: 100%|██████████| 1715/1715 [00:42<00:00, 40.68it/s, loss=0.187]
Test Epoch 68 ==> 	accuracy: 0.9495, 	precision: 0.8537, 	recall: 0.9032, 	specificity: 0.9611, 	f1: 0.8777
Train Epoch 69: 100%|██████████| 6195/6195 [06:32<00:00, 15.77it/s, loss=0.236]
Train Epoch 69 ==> 	accuracy: 0.9562, 	precision: 0.9546, 	recall: 0.9580, 	specificity: 0.9544, 	f1: 0.9563
Test Epoch 69: 100%|██████████| 1715/1715 [00:43<00:00, 39.71it/s, loss=0.0973]
Test Epoch 69 ==> 	accuracy: 0.9440, 	precision: 0.8290, 	recall: 0.9083, 	specificity: 0.9530, 	f1: 0.8669
Train Epoch 70: 100%|██████████| 6195/6195 [06:30<00:00, 15.85it/s, loss=0.314]
Train Epoch 70 ==> 	accuracy: 0.9545, 	precision: 0.9537, 	recall: 0.9554, 	specificity: 0.9536, 	f1: 0.9545
Test Epoch 70: 100%|██████████| 1715/1715 [00:41<00:00, 40.96it/s, loss=0.158]
Test Epoch 70 ==> 	accuracy: 0.9449, 	precision: 0.8333, 	recall: 0.9071, 	specificity: 0.9544, 	f1: 0.8687
Train Epoch 71: 100%|██████████| 6195/6195 [06:28<00:00, 15.95it/s, loss=0.429]
Train Epoch 71 ==> 	accuracy: 0.9546, 	precision: 0.9532, 	recall: 0.9561, 	specificity: 0.9530, 	f1: 0.9546
Test Epoch 71: 100%|██████████| 1715/1715 [00:41<00:00, 41.63it/s, loss=0.132]
Test Epoch 71 ==> 	accuracy: 0.9415, 	precision: 0.8208, 	recall: 0.9068, 	specificity: 0.9503, 	f1: 0.8617
Train Epoch 72: 100%|██████████| 6195/6195 [06:28<00:00, 15.97it/s, loss=0.316]
Train Epoch 72 ==> 	accuracy: 0.9526, 	precision: 0.9520, 	recall: 0.9532, 	specificity: 0.9520, 	f1: 0.9526
Test Epoch 72: 100%|██████████| 1715/1715 [00:40<00:00, 42.57it/s, loss=0.101]
Test Epoch 72 ==> 	accuracy: 0.9500, 	precision: 0.8584, 	recall: 0.8992, 	specificity: 0.9627, 	f1: 0.8783
Train Epoch 73: 100%|██████████| 6195/6195 [06:28<00:00, 15.95it/s, loss=0.343]
Train Epoch 73 ==> 	accuracy: 0.9581, 	precision: 0.9558, 	recall: 0.9607, 	specificity: 0.9556, 	f1: 0.9582
Test Epoch 73: 100%|██████████| 1715/1715 [00:42<00:00, 40.59it/s, loss=0.171]
Test Epoch 73 ==> 	accuracy: 0.9472, 	precision: 0.8406, 	recall: 0.9092, 	specificity: 0.9567, 	f1: 0.8736
Train Epoch 74: 100%|██████████| 6195/6195 [06:25<00:00, 16.06it/s, loss=0.308]
Train Epoch 74 ==> 	accuracy: 0.9531, 	precision: 0.9518, 	recall: 0.9545, 	specificity: 0.9517, 	f1: 0.9532
Test Epoch 74: 100%|██████████| 1715/1715 [00:46<00:00, 37.25it/s, loss=0.181]
Test Epoch 74 ==> 	accuracy: 0.9470, 	precision: 0.8445, 	recall: 0.9022, 	specificity: 0.9583, 	f1: 0.8724
Train Epoch 75: 100%|██████████| 6195/6195 [06:26<00:00, 16.05it/s, loss=0.239]
Train Epoch 75 ==> 	accuracy: 0.9553, 	precision: 0.9545, 	recall: 0.9561, 	specificity: 0.9544, 	f1: 0.9553
Test Epoch 75: 100%|██████████| 1715/1715 [00:45<00:00, 37.57it/s, loss=0.179]
Test Epoch 75 ==> 	accuracy: 0.9466, 	precision: 0.8433, 	recall: 0.9013, 	specificity: 0.9579, 	f1: 0.8713
Train Epoch 76: 100%|██████████| 6195/6195 [06:26<00:00, 16.02it/s, loss=0.224]
Train Epoch 76 ==> 	accuracy: 0.9577, 	precision: 0.9562, 	recall: 0.9592, 	specificity: 0.9561, 	f1: 0.9577
Test Epoch 76: 100%|██████████| 1715/1715 [00:46<00:00, 36.82it/s, loss=0.0862]
Test Epoch 76 ==> 	accuracy: 0.9458, 	precision: 0.8381, 	recall: 0.9048, 	specificity: 0.9561, 	f1: 0.8701
Epoch 00017: reducing learning rate of group 0 to 1.0000e-05.
Train Epoch 77: 100%|██████████| 6195/6195 [06:26<00:00, 16.02it/s, loss=0.358]
Train Epoch 77 ==> 	accuracy: 0.9565, 	precision: 0.9554, 	recall: 0.9577, 	specificity: 0.9553, 	f1: 0.9566
Test Epoch 77: 100%|██████████| 1715/1715 [00:41<00:00, 41.41it/s, loss=0.176]
Test Epoch 77 ==> 	accuracy: 0.9456, 	precision: 0.8386, 	recall: 0.9029, 	specificity: 0.9564, 	f1: 0.8696
Train Epoch 78: 100%|██████████| 6195/6195 [06:37<00:00, 15.58it/s, loss=0.273]
Train Epoch 78 ==> 	accuracy: 0.9570, 	precision: 0.9551, 	recall: 0.9592, 	specificity: 0.9549, 	f1: 0.9571
Test Epoch 78: 100%|██████████| 1715/1715 [00:41<00:00, 41.46it/s, loss=0.277]
Test Epoch 78 ==> 	accuracy: 0.9394, 	precision: 0.8137, 	recall: 0.9054, 	specificity: 0.9479, 	f1: 0.8571
Train Epoch 79: 100%|██████████| 6195/6195 [06:29<00:00, 15.91it/s, loss=0.28]
Train Epoch 79 ==> 	accuracy: 0.9562, 	precision: 0.9550, 	recall: 0.9574, 	specificity: 0.9549, 	f1: 0.9562
Test Epoch 79: 100%|██████████| 1715/1715 [00:44<00:00, 38.12it/s, loss=0.131]
Test Epoch 79 ==> 	accuracy: 0.9371, 	precision: 0.8073, 	recall: 0.9017, 	specificity: 0.9459, 	f1: 0.8519

Process finished with exit code 0

'''