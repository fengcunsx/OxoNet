import argparse
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import tqdm
from dataset.dataset import PairDataset, MultiSimSampler, collate_fn, get_dataset, get_test_dataset
from model.loss import FocalLoss
from model.model import DetectModel
from pytorch_metric_learning import losses, miners, samplers
from torch import nn

# 定义颜色常量
BLUE = '\033[94m'
GREEN = '\033[92m'


def parse_data(batch: dict, device, kmers=5):
    signal = batch['signals'].to(device)
    kmer = batch['kmers'].to(device)
    kmer_len = kmer[0].shape[0] - kmers
    kmer = kmer[:, kmer_len // 2: -kmer_len // 2]
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
    model = DetectModel(sig_blocks=4, sig_l=175, seq_l=5).to(device)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
    os.makedirs(args.save_dir, exist_ok=True)
    # class_loss = nn.BCELoss()
    class_loss = FocalLoss(alpha=0.01, gamma=2.5, reduction='sum')
    miner = miners.MultiSimilarityMiner(epsilon=0.1)  # ϵ 参数
    feature_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)  # α, β, λ (λ 在代码中叫 base)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9, verbose=True)

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
            signal, kmer, _, _, _, sig_l, label = parse_data(batch, device)
            feature, pred = model(signal, kmer, sig_l)

            label_float = label.float()  # 用于 BCE

            # 挖掘样本
            if epoch < 40:
                label_long = label.long()
                hard_pairs = miner(feature, label_long)
                # 损失
                loss = class_loss(pred.squeeze(-1), label_float) + 0.1 * feature_loss(feature, label_long, hard_pairs)
            else:
                loss = class_loss(pred.squeeze(-1), label_float)

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
            signal, kmer, _, _, _, sig_l, label = parse_data(batch, device)
            _, pred = model(signal, kmer, sig_l)

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
        # scheduler.step(metrics['specificity'])
        if epoch >= 40:
            scheduler.step()
        # scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--pos-dir', type=str, default='/home/bio/8oxog/data/7mer_feature/8oxog_train')
    parser.add_argument('--neg-dir', type=str, default='/home/bio/8oxog/data/7mer_feature/g_train')
    parser.add_argument('--work-dir', type=str, default='/home/bio/8oxog/data/7mer_feature/workspace')
    parser.add_argument('--test-pos-dir', type=str, default='/home/bio/8oxog/data/7mer_feature/8oxog_test')
    parser.add_argument('--test-neg-dir', type=str, default='/home/bio/8oxog/data/7mer_feature/g_test')
    parser.add_argument('--test-work-dir', type=str, default='/home/bio/8oxog/data/7mer_feature/test_workspace')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--save-dir', type=str, default='../model_save_sigBlock4_focalWithMs_deformable_7mer_ab_seq'
                        )
    args = parser.parse_args()

    train(args)

'''
whole model
'../model_save_sigBlock4_focalWithMs_deformable'
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6195/6195 [08:51<00:00, 11.66it/s, loss=0.59]
Train Epoch 0 ==> 	accuracy: 0.6106, 	precision: 0.9942, 	recall: 0.2225, 	specificity: 0.9987, 	f1: 0.3636
Test Epoch 0: 100%|██████████| 1715/1715 [00:55<00:00, 31.02it/s, loss=0.686]
Test Epoch 0 ==> 	accuracy: 0.8917, 	precision: 0.9729, 	recall: 0.4737, 	specificity: 0.9967, 	f1: 0.6371
Train Epoch 1: 100%|██████████| 6195/6195 [09:05<00:00, 11.35it/s, loss=0.588]
Train Epoch 1 ==> 	accuracy: 0.7087, 	precision: 0.9967, 	recall: 0.4188, 	specificity: 0.9986, 	f1: 0.5898
Test Epoch 1: 100%|██████████| 1715/1715 [00:59<00:00, 28.99it/s, loss=0.296]
Test Epoch 1 ==> 	accuracy: 0.9053, 	precision: 0.9812, 	recall: 0.5389, 	specificity: 0.9974, 	f1: 0.6957
Train Epoch 2: 100%|██████████| 6195/6195 [09:03<00:00, 11.39it/s, loss=0.69]
Train Epoch 2 ==> 	accuracy: 0.7362, 	precision: 0.9972, 	recall: 0.4738, 	specificity: 0.9987, 	f1: 0.6423
Test Epoch 2: 100%|██████████| 1715/1715 [01:07<00:00, 25.38it/s, loss=0.725]
Test Epoch 2 ==> 	accuracy: 0.8850, 	precision: 0.9815, 	recall: 0.4353, 	specificity: 0.9979, 	f1: 0.6031
Train Epoch 3: 100%|██████████| 6195/6195 [09:01<00:00, 11.43it/s, loss=0.517]
Train Epoch 3 ==> 	accuracy: 0.7465, 	precision: 0.9975, 	recall: 0.4943, 	specificity: 0.9987, 	f1: 0.6610
Test Epoch 3: 100%|██████████| 1715/1715 [00:56<00:00, 30.25it/s, loss=0.759]
Test Epoch 3 ==> 	accuracy: 0.9109, 	precision: 0.9760, 	recall: 0.5703, 	specificity: 0.9965, 	f1: 0.7199
Train Epoch 4: 100%|██████████| 6195/6195 [09:01<00:00, 11.44it/s, loss=0.525]
Train Epoch 4 ==> 	accuracy: 0.7616, 	precision: 0.9978, 	recall: 0.5243, 	specificity: 0.9988, 	f1: 0.6874
Test Epoch 4: 100%|██████████| 1715/1715 [00:55<00:00, 30.83it/s, loss=0.353]
Test Epoch 4 ==> 	accuracy: 0.9178, 	precision: 0.9712, 	recall: 0.6087, 	specificity: 0.9955, 	f1: 0.7483
Train Epoch 5: 100%|██████████| 6195/6195 [08:51<00:00, 11.65it/s, loss=0.769]
Train Epoch 5 ==> 	accuracy: 0.7659, 	precision: 0.9979, 	recall: 0.5330, 	specificity: 0.9989, 	f1: 0.6948
Test Epoch 5: 100%|██████████| 1715/1715 [00:56<00:00, 30.25it/s, loss=0.266]
Test Epoch 5 ==> 	accuracy: 0.9168, 	precision: 0.9616, 	recall: 0.6100, 	specificity: 0.9939, 	f1: 0.7465
Train Epoch 6: 100%|██████████| 6195/6195 [08:53<00:00, 11.60it/s, loss=0.583]
Train Epoch 6 ==> 	accuracy: 0.7696, 	precision: 0.9981, 	recall: 0.5402, 	specificity: 0.9990, 	f1: 0.7010
Test Epoch 6: 100%|██████████| 1715/1715 [00:56<00:00, 30.17it/s, loss=0.549]
Test Epoch 6 ==> 	accuracy: 0.9133, 	precision: 0.9816, 	recall: 0.5789, 	specificity: 0.9973, 	f1: 0.7283
Train Epoch 7: 100%|██████████| 6195/6195 [08:52<00:00, 11.64it/s, loss=0.549]
Train Epoch 7 ==> 	accuracy: 0.7733, 	precision: 0.9981, 	recall: 0.5477, 	specificity: 0.9990, 	f1: 0.7073
Test Epoch 7: 100%|██████████| 1715/1715 [00:57<00:00, 30.02it/s, loss=0.556]
Test Epoch 7 ==> 	accuracy: 0.9132, 	precision: 0.9707, 	recall: 0.5854, 	specificity: 0.9956, 	f1: 0.7304
Train Epoch 8: 100%|██████████| 6195/6195 [08:58<00:00, 11.51it/s, loss=0.535]
Train Epoch 8 ==> 	accuracy: 0.7748, 	precision: 0.9982, 	recall: 0.5506, 	specificity: 0.9990, 	f1: 0.7098
Test Epoch 8: 100%|██████████| 1715/1715 [01:02<00:00, 27.49it/s, loss=0.461]
Test Epoch 8 ==> 	accuracy: 0.9178, 	precision: 0.9699, 	recall: 0.6097, 	specificity: 0.9952, 	f1: 0.7487
Train Epoch 9: 100%|██████████| 6195/6195 [08:50<00:00, 11.68it/s, loss=0.623]
Train Epoch 9 ==> 	accuracy: 0.7801, 	precision: 0.9983, 	recall: 0.5612, 	specificity: 0.9990, 	f1: 0.7185
Test Epoch 9: 100%|██████████| 1715/1715 [00:58<00:00, 29.54it/s, loss=1.02]
Test Epoch 9 ==> 	accuracy: 0.9178, 	precision: 0.9685, 	recall: 0.6106, 	specificity: 0.9950, 	f1: 0.7490
Train Epoch 10: 100%|██████████| 6195/6195 [08:51<00:00, 11.66it/s, loss=0.78]
Train Epoch 10 ==> 	accuracy: 0.7840, 	precision: 0.9984, 	recall: 0.5688, 	specificity: 0.9991, 	f1: 0.7247
Test Epoch 10: 100%|██████████| 1715/1715 [00:54<00:00, 31.72it/s, loss=0.214]
Test Epoch 10 ==> 	accuracy: 0.9140, 	precision: 0.9798, 	recall: 0.5835, 	specificity: 0.9970, 	f1: 0.7314
Train Epoch 11: 100%|██████████| 6195/6195 [08:53<00:00, 11.60it/s, loss=0.455]
Train Epoch 11 ==> 	accuracy: 0.7855, 	precision: 0.9985, 	recall: 0.5718, 	specificity: 0.9991, 	f1: 0.7272
Test Epoch 11: 100%|██████████| 1715/1715 [01:02<00:00, 27.36it/s, loss=0.303]
Test Epoch 11 ==> 	accuracy: 0.9267, 	precision: 0.9692, 	recall: 0.6555, 	specificity: 0.9948, 	f1: 0.7821
Train Epoch 12: 100%|██████████| 6195/6195 [08:56<00:00, 11.54it/s, loss=0.446]
Train Epoch 12 ==> 	accuracy: 0.7878, 	precision: 0.9985, 	recall: 0.5764, 	specificity: 0.9991, 	f1: 0.7309
Test Epoch 12: 100%|██████████| 1715/1715 [01:03<00:00, 27.02it/s, loss=0.721]
Test Epoch 12 ==> 	accuracy: 0.9216, 	precision: 0.9777, 	recall: 0.6237, 	specificity: 0.9964, 	f1: 0.7616
Train Epoch 13: 100%|██████████| 6195/6195 [08:47<00:00, 11.74it/s, loss=0.416]
Train Epoch 13 ==> 	accuracy: 0.7911, 	precision: 0.9985, 	recall: 0.5830, 	specificity: 0.9991, 	f1: 0.7362
Test Epoch 13: 100%|██████████| 1715/1715 [00:58<00:00, 29.50it/s, loss=0.285]
Test Epoch 13 ==> 	accuracy: 0.9175, 	precision: 0.9563, 	recall: 0.6171, 	specificity: 0.9929, 	f1: 0.7501
Train Epoch 14: 100%|██████████| 6195/6195 [08:57<00:00, 11.52it/s, loss=0.593]
Train Epoch 14 ==> 	accuracy: 0.7962, 	precision: 0.9985, 	recall: 0.5932, 	specificity: 0.9991, 	f1: 0.7443
Test Epoch 14: 100%|██████████| 1715/1715 [00:54<00:00, 31.73it/s, loss=0.583]
Test Epoch 14 ==> 	accuracy: 0.9241, 	precision: 0.9534, 	recall: 0.6539, 	specificity: 0.9920, 	f1: 0.7758
Train Epoch 15: 100%|██████████| 6195/6195 [08:47<00:00, 11.74it/s, loss=0.527]
Train Epoch 15 ==> 	accuracy: 0.7939, 	precision: 0.9987, 	recall: 0.5886, 	specificity: 0.9992, 	f1: 0.7407
Test Epoch 15: 100%|██████████| 1715/1715 [01:01<00:00, 27.81it/s, loss=0.224]
Test Epoch 15 ==> 	accuracy: 0.9237, 	precision: 0.9736, 	recall: 0.6374, 	specificity: 0.9957, 	f1: 0.7704
Train Epoch 16: 100%|██████████| 6195/6195 [08:50<00:00, 11.68it/s, loss=0.544]
Train Epoch 16 ==> 	accuracy: 0.7950, 	precision: 0.9987, 	recall: 0.5908, 	specificity: 0.9992, 	f1: 0.7424
Test Epoch 16: 100%|██████████| 1715/1715 [01:01<00:00, 27.71it/s, loss=0.272]
Test Epoch 16 ==> 	accuracy: 0.9223, 	precision: 0.9825, 	recall: 0.6240, 	specificity: 0.9972, 	f1: 0.7632
Train Epoch 17: 100%|██████████| 6195/6195 [08:52<00:00, 11.62it/s, loss=0.451]
Train Epoch 17 ==> 	accuracy: 0.7987, 	precision: 0.9988, 	recall: 0.5981, 	specificity: 0.9993, 	f1: 0.7481
Test Epoch 17: 100%|██████████| 1715/1715 [00:56<00:00, 30.23it/s, loss=0.32]
Test Epoch 17 ==> 	accuracy: 0.9261, 	precision: 0.9753, 	recall: 0.6482, 	specificity: 0.9959, 	f1: 0.7788
Train Epoch 18: 100%|██████████| 6195/6195 [09:00<00:00, 11.47it/s, loss=0.462]
Train Epoch 18 ==> 	accuracy: 0.8000, 	precision: 0.9988, 	recall: 0.6007, 	specificity: 0.9993, 	f1: 0.7502
Test Epoch 18: 100%|██████████| 1715/1715 [00:56<00:00, 30.31it/s, loss=0.437]
Test Epoch 18 ==> 	accuracy: 0.9216, 	precision: 0.9692, 	recall: 0.6292, 	specificity: 0.9950, 	f1: 0.7631
Train Epoch 19: 100%|██████████| 6195/6195 [08:57<00:00, 11.53it/s, loss=0.468]
Train Epoch 19 ==> 	accuracy: 0.7989, 	precision: 0.9989, 	recall: 0.5985, 	specificity: 0.9993, 	f1: 0.7485
Test Epoch 19: 100%|██████████| 1715/1715 [00:56<00:00, 30.43it/s, loss=0.511]
Test Epoch 19 ==> 	accuracy: 0.9200, 	precision: 0.9770, 	recall: 0.6162, 	specificity: 0.9963, 	f1: 0.7557
Train Epoch 20: 100%|██████████| 6195/6195 [08:47<00:00, 11.74it/s, loss=0.561]
Train Epoch 20 ==> 	accuracy: 0.7975, 	precision: 0.9989, 	recall: 0.5957, 	specificity: 0.9994, 	f1: 0.7463
Test Epoch 20: 100%|██████████| 1715/1715 [01:04<00:00, 26.42it/s, loss=0.364]
Test Epoch 20 ==> 	accuracy: 0.9242, 	precision: 0.9836, 	recall: 0.6329, 	specificity: 0.9973, 	f1: 0.7702
Train Epoch 21: 100%|██████████| 6195/6195 [08:48<00:00, 11.71it/s, loss=0.463]
Train Epoch 21 ==> 	accuracy: 0.8055, 	precision: 0.9990, 	recall: 0.6116, 	specificity: 0.9994, 	f1: 0.7587
Test Epoch 21: 100%|██████████| 1715/1715 [00:56<00:00, 30.48it/s, loss=0.281]
Test Epoch 21 ==> 	accuracy: 0.9268, 	precision: 0.9782, 	recall: 0.6500, 	specificity: 0.9964, 	f1: 0.7810
Train Epoch 22: 100%|██████████| 6195/6195 [08:44<00:00, 11.80it/s, loss=0.464]
Train Epoch 22 ==> 	accuracy: 0.8035, 	precision: 0.9990, 	recall: 0.6076, 	specificity: 0.9994, 	f1: 0.7556
Test Epoch 22: 100%|██████████| 1715/1715 [00:53<00:00, 32.01it/s, loss=0.432]
Test Epoch 22 ==> 	accuracy: 0.9214, 	precision: 0.9842, 	recall: 0.6186, 	specificity: 0.9975, 	f1: 0.7597
Train Epoch 23: 100%|██████████| 6195/6195 [08:40<00:00, 11.90it/s, loss=0.589]
Train Epoch 23 ==> 	accuracy: 0.8057, 	precision: 0.9990, 	recall: 0.6120, 	specificity: 0.9994, 	f1: 0.7590
Test Epoch 23: 100%|██████████| 1715/1715 [01:02<00:00, 27.42it/s, loss=0.965]
Test Epoch 23 ==> 	accuracy: 0.9266, 	precision: 0.9519, 	recall: 0.6680, 	specificity: 0.9915, 	f1: 0.7851
Train Epoch 24: 100%|██████████| 6195/6195 [08:38<00:00, 11.95it/s, loss=0.428]
Train Epoch 24 ==> 	accuracy: 0.8057, 	precision: 0.9989, 	recall: 0.6121, 	specificity: 0.9994, 	f1: 0.7591
Test Epoch 24: 100%|██████████| 1715/1715 [01:00<00:00, 28.36it/s, loss=0.583]
Test Epoch 24 ==> 	accuracy: 0.9273, 	precision: 0.9770, 	recall: 0.6533, 	specificity: 0.9961, 	f1: 0.7830
Train Epoch 25: 100%|██████████| 6195/6195 [08:32<00:00, 12.10it/s, loss=0.519]
Train Epoch 25 ==> 	accuracy: 0.8077, 	precision: 0.9991, 	recall: 0.6159, 	specificity: 0.9994, 	f1: 0.7621
Test Epoch 25: 100%|██████████| 1715/1715 [00:55<00:00, 31.03it/s, loss=0.246]
Test Epoch 25 ==> 	accuracy: 0.9220, 	precision: 0.9746, 	recall: 0.6278, 	specificity: 0.9959, 	f1: 0.7637
Train Epoch 26: 100%|██████████| 6195/6195 [08:39<00:00, 11.93it/s, loss=0.442]
Train Epoch 26 ==> 	accuracy: 0.8077, 	precision: 0.9990, 	recall: 0.6161, 	specificity: 0.9994, 	f1: 0.7621
Test Epoch 26: 100%|██████████| 1715/1715 [00:59<00:00, 28.63it/s, loss=0.425]
Test Epoch 26 ==> 	accuracy: 0.9317, 	precision: 0.9760, 	recall: 0.6763, 	specificity: 0.9958, 	f1: 0.7990
Train Epoch 27: 100%|██████████| 6195/6195 [08:35<00:00, 12.01it/s, loss=0.531]
Train Epoch 27 ==> 	accuracy: 0.8118, 	precision: 0.9990, 	recall: 0.6243, 	specificity: 0.9994, 	f1: 0.7684
Test Epoch 27: 100%|██████████| 1715/1715 [00:55<00:00, 30.69it/s, loss=0.866]
Test Epoch 27 ==> 	accuracy: 0.9265, 	precision: 0.9761, 	recall: 0.6496, 	specificity: 0.9960, 	f1: 0.7801
Train Epoch 28: 100%|██████████| 6195/6195 [08:29<00:00, 12.15it/s, loss=0.43]
Train Epoch 28 ==> 	accuracy: 0.8126, 	precision: 0.9991, 	recall: 0.6258, 	specificity: 0.9994, 	f1: 0.7696
Test Epoch 28: 100%|██████████| 1715/1715 [00:58<00:00, 29.37it/s, loss=0.25]
Test Epoch 28 ==> 	accuracy: 0.9257, 	precision: 0.9770, 	recall: 0.6448, 	specificity: 0.9962, 	f1: 0.7769
Train Epoch 29: 100%|██████████| 6195/6195 [08:23<00:00, 12.30it/s, loss=0.48]
Train Epoch 29 ==> 	accuracy: 0.8119, 	precision: 0.9991, 	recall: 0.6243, 	specificity: 0.9994, 	f1: 0.7684
Test Epoch 29: 100%|██████████| 1715/1715 [00:59<00:00, 28.64it/s, loss=0.575]
Test Epoch 29 ==> 	accuracy: 0.9314, 	precision: 0.9595, 	recall: 0.6873, 	specificity: 0.9927, 	f1: 0.8009
Train Epoch 30: 100%|██████████| 6195/6195 [08:25<00:00, 12.25it/s, loss=0.462]
Train Epoch 30 ==> 	accuracy: 0.8134, 	precision: 0.9991, 	recall: 0.6273, 	specificity: 0.9994, 	f1: 0.7707
Test Epoch 30: 100%|██████████| 1715/1715 [00:56<00:00, 30.30it/s, loss=1.63]
Test Epoch 30 ==> 	accuracy: 0.9261, 	precision: 0.9723, 	recall: 0.6502, 	specificity: 0.9954, 	f1: 0.7793
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 31: 100%|██████████| 6195/6195 [08:48<00:00, 11.72it/s, loss=0.496]
Train Epoch 31 ==> 	accuracy: 0.8143, 	precision: 0.9992, 	recall: 0.6291, 	specificity: 0.9995, 	f1: 0.7721
Test Epoch 31: 100%|██████████| 1715/1715 [00:59<00:00, 28.78it/s, loss=0.318]
Test Epoch 31 ==> 	accuracy: 0.9305, 	precision: 0.9728, 	recall: 0.6725, 	specificity: 0.9953, 	f1: 0.7953
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 32: 100%|██████████| 6195/6195 [08:28<00:00, 12.19it/s, loss=0.526]
Train Epoch 32 ==> 	accuracy: 0.8126, 	precision: 0.9992, 	recall: 0.6258, 	specificity: 0.9995, 	f1: 0.7696
Test Epoch 32: 100%|██████████| 1715/1715 [01:00<00:00, 28.16it/s, loss=0.299]
Test Epoch 32 ==> 	accuracy: 0.9323, 	precision: 0.9837, 	recall: 0.6741, 	specificity: 0.9972, 	f1: 0.8000
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 33: 100%|██████████| 6195/6195 [08:36<00:00, 11.99it/s, loss=0.504]
Train Epoch 33 ==> 	accuracy: 0.8136, 	precision: 0.9991, 	recall: 0.6277, 	specificity: 0.9995, 	f1: 0.7710
Test Epoch 33: 100%|██████████| 1715/1715 [00:57<00:00, 29.86it/s, loss=0.291]
Test Epoch 33 ==> 	accuracy: 0.9319, 	precision: 0.9740, 	recall: 0.6790, 	specificity: 0.9955, 	f1: 0.8002
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 34: 100%|██████████| 6195/6195 [08:14<00:00, 12.52it/s, loss=0.473]
Train Epoch 34 ==> 	accuracy: 0.8199, 	precision: 0.9992, 	recall: 0.6403, 	specificity: 0.9995, 	f1: 0.7805
Test Epoch 34: 100%|██████████| 1715/1715 [00:58<00:00, 29.08it/s, loss=0.252]
Test Epoch 34 ==> 	accuracy: 0.9308, 	precision: 0.9827, 	recall: 0.6670, 	specificity: 0.9971, 	f1: 0.7947
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 35: 100%|██████████| 6195/6195 [08:31<00:00, 12.11it/s, loss=0.424]
Train Epoch 35 ==> 	accuracy: 0.8234, 	precision: 0.9992, 	recall: 0.6473, 	specificity: 0.9995, 	f1: 0.7856
Test Epoch 35: 100%|██████████| 1715/1715 [00:56<00:00, 30.59it/s, loss=1.98]
Test Epoch 35 ==> 	accuracy: 0.9333, 	precision: 0.9776, 	recall: 0.6836, 	specificity: 0.9961, 	f1: 0.8045
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 36: 100%|██████████| 6195/6195 [08:17<00:00, 12.44it/s, loss=0.386]
Train Epoch 36 ==> 	accuracy: 0.8234, 	precision: 0.9992, 	recall: 0.6473, 	specificity: 0.9995, 	f1: 0.7857
Test Epoch 36: 100%|██████████| 1715/1715 [00:58<00:00, 29.25it/s, loss=2.09]
Test Epoch 36 ==> 	accuracy: 0.9320, 	precision: 0.9579, 	recall: 0.6917, 	specificity: 0.9924, 	f1: 0.8033
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 37: 100%|██████████| 6195/6195 [08:16<00:00, 12.49it/s, loss=0.722]
Train Epoch 37 ==> 	accuracy: 0.8219, 	precision: 0.9993, 	recall: 0.6443, 	specificity: 0.9995, 	f1: 0.7835
Test Epoch 37: 100%|██████████| 1715/1715 [00:56<00:00, 30.38it/s, loss=1.56]
Test Epoch 37 ==> 	accuracy: 0.9328, 	precision: 0.9790, 	recall: 0.6797, 	specificity: 0.9963, 	f1: 0.8023
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 38: 100%|██████████| 6195/6195 [08:25<00:00, 12.25it/s, loss=0.461]
Train Epoch 38 ==> 	accuracy: 0.8278, 	precision: 0.9993, 	recall: 0.6562, 	specificity: 0.9995, 	f1: 0.7922
Test Epoch 38: 100%|██████████| 1715/1715 [00:58<00:00, 29.51it/s, loss=0.544]
Test Epoch 38 ==> 	accuracy: 0.9339, 	precision: 0.9718, 	recall: 0.6910, 	specificity: 0.9950, 	f1: 0.8077
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 39: 100%|██████████| 6195/6195 [08:27<00:00, 12.21it/s, loss=0.486]
Train Epoch 39 ==> 	accuracy: 0.8251, 	precision: 0.9993, 	recall: 0.6507, 	specificity: 0.9996, 	f1: 0.7882
Test Epoch 39: 100%|██████████| 1715/1715 [00:58<00:00, 29.48it/s, loss=0.379]
Test Epoch 39 ==> 	accuracy: 0.9345, 	precision: 0.9716, 	recall: 0.6940, 	specificity: 0.9949, 	f1: 0.8097
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 40: 100%|██████████| 6195/6195 [08:24<00:00, 12.28it/s, loss=0.455]
Train Epoch 40 ==> 	accuracy: 0.8302, 	precision: 0.9993, 	recall: 0.6610, 	specificity: 0.9995, 	f1: 0.7956
Test Epoch 40: 100%|██████████| 1715/1715 [00:57<00:00, 29.92it/s, loss=0.179]
Test Epoch 40 ==> 	accuracy: 0.9314, 	precision: 0.9781, 	recall: 0.6732, 	specificity: 0.9962, 	f1: 0.7975
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 41: 100%|██████████| 6195/6195 [08:36<00:00, 11.99it/s, loss=0.418]
Train Epoch 41 ==> 	accuracy: 0.8273, 	precision: 0.9993, 	recall: 0.6551, 	specificity: 0.9996, 	f1: 0.7914
Test Epoch 41: 100%|██████████| 1715/1715 [00:58<00:00, 29.23it/s, loss=0.474]
Test Epoch 41 ==> 	accuracy: 0.9333, 	precision: 0.9805, 	recall: 0.6811, 	specificity: 0.9966, 	f1: 0.8038
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 42: 100%|██████████| 6195/6195 [08:31<00:00, 12.11it/s, loss=0.377]
Train Epoch 42 ==> 	accuracy: 0.8326, 	precision: 0.9993, 	recall: 0.6657, 	specificity: 0.9996, 	f1: 0.7991
Test Epoch 42: 100%|██████████| 1715/1715 [00:59<00:00, 28.96it/s, loss=2.58]
Test Epoch 42 ==> 	accuracy: 0.9358, 	precision: 0.9748, 	recall: 0.6980, 	specificity: 0.9955, 	f1: 0.8135
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 43: 100%|██████████| 6195/6195 [08:20<00:00, 12.38it/s, loss=0.517]
Train Epoch 43 ==> 	accuracy: 0.8312, 	precision: 0.9994, 	recall: 0.6629, 	specificity: 0.9996, 	f1: 0.7971
Test Epoch 43: 100%|██████████| 1715/1715 [00:57<00:00, 29.89it/s, loss=0.574]
Test Epoch 43 ==> 	accuracy: 0.9360, 	precision: 0.9743, 	recall: 0.6995, 	specificity: 0.9954, 	f1: 0.8143
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 44: 100%|██████████| 6195/6195 [08:21<00:00, 12.35it/s, loss=0.544]
Train Epoch 44 ==> 	accuracy: 0.8306, 	precision: 0.9994, 	recall: 0.6617, 	specificity: 0.9996, 	f1: 0.7962
Test Epoch 44: 100%|██████████| 1715/1715 [01:00<00:00, 28.55it/s, loss=0.999]
Test Epoch 44 ==> 	accuracy: 0.9370, 	precision: 0.9754, 	recall: 0.7038, 	specificity: 0.9955, 	f1: 0.8176
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 45: 100%|██████████| 6195/6195 [08:28<00:00, 12.18it/s, loss=0.478]
Train Epoch 45 ==> 	accuracy: 0.8324, 	precision: 0.9993, 	recall: 0.6652, 	specificity: 0.9995, 	f1: 0.7988
Test Epoch 45: 100%|██████████| 1715/1715 [00:56<00:00, 30.39it/s, loss=0.339]
Test Epoch 45 ==> 	accuracy: 0.9346, 	precision: 0.9736, 	recall: 0.6929, 	specificity: 0.9953, 	f1: 0.8096
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 46: 100%|██████████| 6195/6195 [08:29<00:00, 12.15it/s, loss=0.45]
Train Epoch 46 ==> 	accuracy: 0.8361, 	precision: 0.9994, 	recall: 0.6727, 	specificity: 0.9996, 	f1: 0.8041
Test Epoch 46: 100%|██████████| 1715/1715 [00:56<00:00, 30.09it/s, loss=0.345]
Test Epoch 46 ==> 	accuracy: 0.9345, 	precision: 0.9707, 	recall: 0.6949, 	specificity: 0.9947, 	f1: 0.8100
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 47: 100%|██████████| 6195/6195 [08:26<00:00, 12.23it/s, loss=0.4]
Train Epoch 47 ==> 	accuracy: 0.8373, 	precision: 0.9994, 	recall: 0.6750, 	specificity: 0.9996, 	f1: 0.8058
Test Epoch 47: 100%|██████████| 1715/1715 [00:54<00:00, 31.66it/s, loss=0.301]
Test Epoch 47 ==> 	accuracy: 0.9379, 	precision: 0.9691, 	recall: 0.7133, 	specificity: 0.9943, 	f1: 0.8217
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 48: 100%|██████████| 6195/6195 [08:20<00:00, 12.38it/s, loss=0.407]
Train Epoch 48 ==> 	accuracy: 0.8363, 	precision: 0.9994, 	recall: 0.6731, 	specificity: 0.9996, 	f1: 0.8044
Test Epoch 48: 100%|██████████| 1715/1715 [00:55<00:00, 31.04it/s, loss=0.237]
Test Epoch 48 ==> 	accuracy: 0.9375, 	precision: 0.9742, 	recall: 0.7075, 	specificity: 0.9953, 	f1: 0.8197
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 49: 100%|██████████| 6195/6195 [08:25<00:00, 12.26it/s, loss=0.46]
Train Epoch 49 ==> 	accuracy: 0.8402, 	precision: 0.9995, 	recall: 0.6807, 	specificity: 0.9996, 	f1: 0.8098
Test Epoch 49: 100%|██████████| 1715/1715 [00:58<00:00, 29.44it/s, loss=0.321]
Test Epoch 49 ==> 	accuracy: 0.9369, 	precision: 0.9716, 	recall: 0.7062, 	specificity: 0.9948, 	f1: 0.8179
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 50: 100%|██████████| 6195/6195 [08:33<00:00, 12.06it/s, loss=0.451]
Train Epoch 50 ==> 	accuracy: 0.8393, 	precision: 0.9994, 	recall: 0.6790, 	specificity: 0.9996, 	f1: 0.8086
Test Epoch 50: 100%|██████████| 1715/1715 [00:56<00:00, 30.47it/s, loss=0.409]
Test Epoch 50 ==> 	accuracy: 0.9406, 	precision: 0.9818, 	recall: 0.7177, 	specificity: 0.9967, 	f1: 0.8292
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 51: 100%|██████████| 6195/6195 [08:23<00:00, 12.29it/s, loss=0.466]
Train Epoch 51 ==> 	accuracy: 0.8412, 	precision: 0.9994, 	recall: 0.6829, 	specificity: 0.9996, 	f1: 0.8114
Test Epoch 51: 100%|██████████| 1715/1715 [00:58<00:00, 29.12it/s, loss=0.302]
Test Epoch 51 ==> 	accuracy: 0.9363, 	precision: 0.9862, 	recall: 0.6925, 	specificity: 0.9976, 	f1: 0.8137
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 52: 100%|██████████| 6195/6195 [08:22<00:00, 12.33it/s, loss=0.448]
Train Epoch 52 ==> 	accuracy: 0.8416, 	precision: 0.9995, 	recall: 0.6835, 	specificity: 0.9996, 	f1: 0.8118
Test Epoch 52: 100%|██████████| 1715/1715 [01:00<00:00, 28.52it/s, loss=0.196]
Test Epoch 52 ==> 	accuracy: 0.9393, 	precision: 0.9842, 	recall: 0.7092, 	specificity: 0.9971, 	f1: 0.8244
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 53: 100%|██████████| 6195/6195 [08:33<00:00, 12.07it/s, loss=0.398]
Train Epoch 53 ==> 	accuracy: 0.8421, 	precision: 0.9995, 	recall: 0.6846, 	specificity: 0.9996, 	f1: 0.8126
Test Epoch 53: 100%|██████████| 1715/1715 [00:57<00:00, 29.74it/s, loss=0.138]
Test Epoch 53 ==> 	accuracy: 0.9407, 	precision: 0.9863, 	recall: 0.7146, 	specificity: 0.9975, 	f1: 0.8288
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 54: 100%|██████████| 6195/6195 [08:23<00:00, 12.31it/s, loss=0.37]
Train Epoch 54 ==> 	accuracy: 0.8439, 	precision: 0.9995, 	recall: 0.6881, 	specificity: 0.9997, 	f1: 0.8151
Test Epoch 54: 100%|██████████| 1715/1715 [00:56<00:00, 30.35it/s, loss=0.703]
Test Epoch 54 ==> 	accuracy: 0.9408, 	precision: 0.9835, 	recall: 0.7172, 	specificity: 0.9970, 	f1: 0.8295
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 55: 100%|██████████| 6195/6195 [08:27<00:00, 12.21it/s, loss=0.51]
Train Epoch 55 ==> 	accuracy: 0.8441, 	precision: 0.9995, 	recall: 0.6886, 	specificity: 0.9997, 	f1: 0.8154
Test Epoch 55: 100%|██████████| 1715/1715 [00:54<00:00, 31.47it/s, loss=0.199]
Test Epoch 55 ==> 	accuracy: 0.9420, 	precision: 0.9842, 	recall: 0.7225, 	specificity: 0.9971, 	f1: 0.8333
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 56: 100%|██████████| 6195/6195 [08:22<00:00, 12.34it/s, loss=0.553]
Train Epoch 56 ==> 	accuracy: 0.8471, 	precision: 0.9995, 	recall: 0.6944, 	specificity: 0.9997, 	f1: 0.8195
Test Epoch 56: 100%|██████████| 1715/1715 [00:55<00:00, 31.04it/s, loss=2.1]
Test Epoch 56 ==> 	accuracy: 0.9402, 	precision: 0.9705, 	recall: 0.7239, 	specificity: 0.9945, 	f1: 0.8293
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 57: 100%|██████████| 6195/6195 [08:42<00:00, 11.87it/s, loss=0.48]
Train Epoch 57 ==> 	accuracy: 0.8460, 	precision: 0.9995, 	recall: 0.6924, 	specificity: 0.9997, 	f1: 0.8181
Test Epoch 57: 100%|██████████| 1715/1715 [00:59<00:00, 28.92it/s, loss=0.138]
Test Epoch 57 ==> 	accuracy: 0.9377, 	precision: 0.9868, 	recall: 0.6992, 	specificity: 0.9976, 	f1: 0.8184
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 58: 100%|██████████| 6195/6195 [08:28<00:00, 12.19it/s, loss=0.371]
Train Epoch 58 ==> 	accuracy: 0.8475, 	precision: 0.9995, 	recall: 0.6953, 	specificity: 0.9997, 	f1: 0.8201
Test Epoch 58: 100%|██████████| 1715/1715 [00:54<00:00, 31.74it/s, loss=0.165]
Test Epoch 58 ==> 	accuracy: 0.9419, 	precision: 0.9835, 	recall: 0.7230, 	specificity: 0.9970, 	f1: 0.8333
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 59: 100%|██████████| 6195/6195 [08:22<00:00, 12.32it/s, loss=0.521]
Train Epoch 59 ==> 	accuracy: 0.8486, 	precision: 0.9995, 	recall: 0.6974, 	specificity: 0.9997, 	f1: 0.8216
Test Epoch 59: 100%|██████████| 1715/1715 [00:56<00:00, 30.29it/s, loss=0.198]
Test Epoch 59 ==> 	accuracy: 0.9419, 	precision: 0.9769, 	recall: 0.7278, 	specificity: 0.9957, 	f1: 0.8342
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 60: 100%|██████████| 6195/6195 [08:19<00:00, 12.40it/s, loss=0.284]
Train Epoch 60 ==> 	accuracy: 0.8506, 	precision: 0.9995, 	recall: 0.7016, 	specificity: 0.9997, 	f1: 0.8245
Test Epoch 60: 100%|██████████| 1715/1715 [00:57<00:00, 29.84it/s, loss=0.281]
Test Epoch 60 ==> 	accuracy: 0.9428, 	precision: 0.9771, 	recall: 0.7323, 	specificity: 0.9957, 	f1: 0.8372
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 61: 100%|██████████| 6195/6195 [08:19<00:00, 12.40it/s, loss=0.397]
Train Epoch 61 ==> 	accuracy: 0.8485, 	precision: 0.9995, 	recall: 0.6973, 	specificity: 0.9997, 	f1: 0.8215
Test Epoch 61: 100%|██████████| 1715/1715 [00:59<00:00, 28.69it/s, loss=0.15]
Test Epoch 61 ==> 	accuracy: 0.9420, 	precision: 0.9820, 	recall: 0.7244, 	specificity: 0.9967, 	f1: 0.8337
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 62: 100%|██████████| 6195/6195 [08:29<00:00, 12.16it/s, loss=0.445]
Train Epoch 62 ==> 	accuracy: 0.8503, 	precision: 0.9996, 	recall: 0.7009, 	specificity: 0.9997, 	f1: 0.8240
Test Epoch 62: 100%|██████████| 1715/1715 [01:00<00:00, 28.36it/s, loss=0.179]
Test Epoch 62 ==> 	accuracy: 0.9428, 	precision: 0.9779, 	recall: 0.7314, 	specificity: 0.9958, 	f1: 0.8369
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 63: 100%|██████████| 6195/6195 [08:33<00:00, 12.07it/s, loss=0.537]
Train Epoch 63 ==> 	accuracy: 0.8532, 	precision: 0.9995, 	recall: 0.7066, 	specificity: 0.9997, 	f1: 0.8279
Test Epoch 63: 100%|██████████| 1715/1715 [00:55<00:00, 30.99it/s, loss=0.127]
Test Epoch 63 ==> 	accuracy: 0.9435, 	precision: 0.9788, 	recall: 0.7345, 	specificity: 0.9960, 	f1: 0.8392
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 64: 100%|██████████| 6195/6195 [08:22<00:00, 12.33it/s, loss=0.457]
Train Epoch 64 ==> 	accuracy: 0.8540, 	precision: 0.9996, 	recall: 0.7084, 	specificity: 0.9997, 	f1: 0.8292
Test Epoch 64: 100%|██████████| 1715/1715 [00:57<00:00, 29.90it/s, loss=0.368]
Test Epoch 64 ==> 	accuracy: 0.9426, 	precision: 0.9790, 	recall: 0.7295, 	specificity: 0.9961, 	f1: 0.8360
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 65: 100%|██████████| 6195/6195 [08:19<00:00, 12.41it/s, loss=0.426]
Train Epoch 65 ==> 	accuracy: 0.8531, 	precision: 0.9996, 	recall: 0.7066, 	specificity: 0.9997, 	f1: 0.8279
Test Epoch 65: 100%|██████████| 1715/1715 [01:01<00:00, 27.97it/s, loss=0.135]
Test Epoch 65 ==> 	accuracy: 0.9429, 	precision: 0.9817, 	recall: 0.7291, 	specificity: 0.9966, 	f1: 0.8368
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 66: 100%|██████████| 6195/6195 [08:18<00:00, 12.43it/s, loss=0.393]
Train Epoch 66 ==> 	accuracy: 0.8538, 	precision: 0.9995, 	recall: 0.7079, 	specificity: 0.9997, 	f1: 0.8288
Test Epoch 66: 100%|██████████| 1715/1715 [00:55<00:00, 30.73it/s, loss=0.323]
Test Epoch 66 ==> 	accuracy: 0.9433, 	precision: 0.9832, 	recall: 0.7299, 	specificity: 0.9969, 	f1: 0.8378
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 67: 100%|██████████| 6195/6195 [08:30<00:00, 12.13it/s, loss=0.403]
Train Epoch 67 ==> 	accuracy: 0.8556, 	precision: 0.9996, 	recall: 0.7115, 	specificity: 0.9997, 	f1: 0.8313
Test Epoch 67: 100%|██████████| 1715/1715 [00:58<00:00, 29.18it/s, loss=0.273]
Test Epoch 67 ==> 	accuracy: 0.9450, 	precision: 0.9770, 	recall: 0.7437, 	specificity: 0.9956, 	f1: 0.8445
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 68: 100%|██████████| 6195/6195 [08:33<00:00, 12.07it/s, loss=0.464]
Train Epoch 68 ==> 	accuracy: 0.8540, 	precision: 0.9996, 	recall: 0.7083, 	specificity: 0.9997, 	f1: 0.8291
Test Epoch 68: 100%|██████████| 1715/1715 [00:52<00:00, 32.86it/s, loss=0.26]
Test Epoch 68 ==> 	accuracy: 0.9404, 	precision: 0.9837, 	recall: 0.7151, 	specificity: 0.9970, 	f1: 0.8282
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 69: 100%|██████████| 6195/6195 [08:31<00:00, 12.12it/s, loss=0.397]
Train Epoch 69 ==> 	accuracy: 0.8564, 	precision: 0.9996, 	recall: 0.7132, 	specificity: 0.9997, 	f1: 0.8324
Test Epoch 69: 100%|██████████| 1715/1715 [00:58<00:00, 29.39it/s, loss=0.261]
Test Epoch 69 ==> 	accuracy: 0.9430, 	precision: 0.9720, 	recall: 0.7374, 	specificity: 0.9947, 	f1: 0.8386
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 70: 100%|██████████| 6195/6195 [08:22<00:00, 12.33it/s, loss=0.406]
Train Epoch 70 ==> 	accuracy: 0.8570, 	precision: 0.9996, 	recall: 0.7143, 	specificity: 0.9997, 	f1: 0.8332
Test Epoch 70: 100%|██████████| 1715/1715 [00:56<00:00, 30.34it/s, loss=0.354]
Test Epoch 70 ==> 	accuracy: 0.9443, 	precision: 0.9799, 	recall: 0.7377, 	specificity: 0.9962, 	f1: 0.8417
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 71: 100%|██████████| 6195/6195 [08:26<00:00, 12.23it/s, loss=0.374]
Train Epoch 71 ==> 	accuracy: 0.8580, 	precision: 0.9996, 	recall: 0.7164, 	specificity: 0.9997, 	f1: 0.8346
Test Epoch 71: 100%|██████████| 1715/1715 [01:00<00:00, 28.55it/s, loss=0.281]
Test Epoch 71 ==> 	accuracy: 0.9447, 	precision: 0.9805, 	recall: 0.7393, 	specificity: 0.9963, 	f1: 0.8430
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 72: 100%|██████████| 6195/6195 [08:34<00:00, 12.05it/s, loss=0.355]
Train Epoch 72 ==> 	accuracy: 0.8571, 	precision: 0.9996, 	recall: 0.7144, 	specificity: 0.9997, 	f1: 0.8333
Test Epoch 72: 100%|██████████| 1715/1715 [00:53<00:00, 31.94it/s, loss=0.168]
Test Epoch 72 ==> 	accuracy: 0.9434, 	precision: 0.9806, 	recall: 0.7328, 	specificity: 0.9964, 	f1: 0.8388
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 73: 100%|██████████| 6195/6195 [08:39<00:00, 11.92it/s, loss=0.669]
Train Epoch 73 ==> 	accuracy: 0.8605, 	precision: 0.9996, 	recall: 0.7213, 	specificity: 0.9997, 	f1: 0.8380
Test Epoch 73: 100%|██████████| 1715/1715 [00:55<00:00, 30.90it/s, loss=2.01]
Test Epoch 73 ==> 	accuracy: 0.9474, 	precision: 0.9758, 	recall: 0.7569, 	specificity: 0.9953, 	f1: 0.8525
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 74: 100%|██████████| 6195/6195 [08:41<00:00, 11.89it/s, loss=0.625]
Train Epoch 74 ==> 	accuracy: 0.8591, 	precision: 0.9996, 	recall: 0.7186, 	specificity: 0.9997, 	f1: 0.8361
Test Epoch 74: 100%|██████████| 1715/1715 [00:55<00:00, 30.66it/s, loss=0.24]
Test Epoch 74 ==> 	accuracy: 0.9434, 	precision: 0.9803, 	recall: 0.7329, 	specificity: 0.9963, 	f1: 0.8387
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 75: 100%|██████████| 6195/6195 [08:41<00:00, 11.87it/s, loss=0.466]
Train Epoch 75 ==> 	accuracy: 0.8598, 	precision: 0.9996, 	recall: 0.7199, 	specificity: 0.9997, 	f1: 0.8370
Test Epoch 75: 100%|██████████| 1715/1715 [01:03<00:00, 26.82it/s, loss=0.254]
Test Epoch 75 ==> 	accuracy: 0.9456, 	precision: 0.9784, 	recall: 0.7453, 	specificity: 0.9959, 	f1: 0.8461
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 76: 100%|██████████| 6195/6195 [08:48<00:00, 11.72it/s, loss=0.384]
Train Epoch 76 ==> 	accuracy: 0.8621, 	precision: 0.9996, 	recall: 0.7245, 	specificity: 0.9997, 	f1: 0.8401
Test Epoch 76: 100%|██████████| 1715/1715 [00:55<00:00, 30.90it/s, loss=1.31]
Test Epoch 76 ==> 	accuracy: 0.9438, 	precision: 0.9789, 	recall: 0.7360, 	specificity: 0.9960, 	f1: 0.8403
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 77: 100%|██████████| 6195/6195 [08:34<00:00, 12.05it/s, loss=0.404]
Train Epoch 77 ==> 	accuracy: 0.8623, 	precision: 0.9996, 	recall: 0.7250, 	specificity: 0.9997, 	f1: 0.8404
Test Epoch 77: 100%|██████████| 1715/1715 [00:57<00:00, 30.03it/s, loss=0.586]
Test Epoch 77 ==> 	accuracy: 0.9462, 	precision: 0.9775, 	recall: 0.7492, 	specificity: 0.9957, 	f1: 0.8482
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 78: 100%|██████████| 6195/6195 [08:43<00:00, 11.84it/s, loss=0.449]
Train Epoch 78 ==> 	accuracy: 0.8614, 	precision: 0.9996, 	recall: 0.7231, 	specificity: 0.9997, 	f1: 0.8391
Test Epoch 78: 100%|██████████| 1715/1715 [00:57<00:00, 29.91it/s, loss=0.262]
Test Epoch 78 ==> 	accuracy: 0.9450, 	precision: 0.9744, 	recall: 0.7457, 	specificity: 0.9951, 	f1: 0.8449
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 79: 100%|██████████| 6195/6195 [08:47<00:00, 11.75it/s, loss=0.46]
Train Epoch 79 ==> 	accuracy: 0.8637, 	precision: 0.9996, 	recall: 0.7277, 	specificity: 0.9997, 	f1: 0.8422
Test Epoch 79: 100%|██████████| 1715/1715 [00:59<00:00, 28.91it/s, loss=0.576]
Test Epoch 79 ==> 	accuracy: 0.9452, 	precision: 0.9608, 	recall: 0.7581, 	specificity: 0.9922, 	f1: 0.8475
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 80: 100%|██████████| 6195/6195 [08:38<00:00, 11.95it/s, loss=0.349]
Train Epoch 80 ==> 	accuracy: 0.8640, 	precision: 0.9996, 	recall: 0.7283, 	specificity: 0.9997, 	f1: 0.8426
Test Epoch 80: 100%|██████████| 1715/1715 [00:55<00:00, 30.71it/s, loss=0.342]
Test Epoch 80 ==> 	accuracy: 0.9462, 	precision: 0.9677, 	recall: 0.7571, 	specificity: 0.9936, 	f1: 0.8495
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 81: 100%|██████████| 6195/6195 [08:41<00:00, 11.87it/s, loss=0.464]
Train Epoch 81 ==> 	accuracy: 0.8621, 	precision: 0.9996, 	recall: 0.7244, 	specificity: 0.9997, 	f1: 0.8401
Test Epoch 81: 100%|██████████| 1715/1715 [01:03<00:00, 26.87it/s, loss=0.349]
Test Epoch 81 ==> 	accuracy: 0.9450, 	precision: 0.9692, 	recall: 0.7497, 	specificity: 0.9940, 	f1: 0.8454
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 82: 100%|██████████| 6195/6195 [08:38<00:00, 11.94it/s, loss=0.382]
Train Epoch 82 ==> 	accuracy: 0.8636, 	precision: 0.9996, 	recall: 0.7274, 	specificity: 0.9997, 	f1: 0.8421
Test Epoch 82: 100%|██████████| 1715/1715 [00:55<00:00, 30.99it/s, loss=0.641]
Test Epoch 82 ==> 	accuracy: 0.9460, 	precision: 0.9736, 	recall: 0.7511, 	specificity: 0.9949, 	f1: 0.8480
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 83: 100%|██████████| 6195/6195 [08:40<00:00, 11.91it/s, loss=0.438]
Train Epoch 83 ==> 	accuracy: 0.8645, 	precision: 0.9996, 	recall: 0.7292, 	specificity: 0.9997, 	f1: 0.8433
Test Epoch 83: 100%|██████████| 1715/1715 [00:57<00:00, 29.88it/s, loss=0.216]
Test Epoch 83 ==> 	accuracy: 0.9470, 	precision: 0.9732, 	recall: 0.7568, 	specificity: 0.9948, 	f1: 0.8515
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 84: 100%|██████████| 6195/6195 [08:47<00:00, 11.75it/s, loss=0.434]
Train Epoch 84 ==> 	accuracy: 0.8665, 	precision: 0.9996, 	recall: 0.7334, 	specificity: 0.9997, 	f1: 0.8460
Test Epoch 84: 100%|██████████| 1715/1715 [00:55<00:00, 30.77it/s, loss=0.426]
Test Epoch 84 ==> 	accuracy: 0.9467, 	precision: 0.9745, 	recall: 0.7542, 	specificity: 0.9950, 	f1: 0.8503
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 85: 100%|██████████| 6195/6195 [08:55<00:00, 11.56it/s, loss=0.358]
Train Epoch 85 ==> 	accuracy: 0.8650, 	precision: 0.9997, 	recall: 0.7303, 	specificity: 0.9998, 	f1: 0.8440
Test Epoch 85: 100%|██████████| 1715/1715 [00:57<00:00, 30.08it/s, loss=0.167]
Test Epoch 85 ==> 	accuracy: 0.9457, 	precision: 0.9739, 	recall: 0.7494, 	specificity: 0.9950, 	f1: 0.8471
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 86: 100%|██████████| 6195/6195 [08:50<00:00, 11.68it/s, loss=0.509]
Train Epoch 86 ==> 	accuracy: 0.8651, 	precision: 0.9996, 	recall: 0.7305, 	specificity: 0.9997, 	f1: 0.8441
Test Epoch 86: 100%|██████████| 1715/1715 [00:56<00:00, 30.16it/s, loss=0.701]
Test Epoch 86 ==> 	accuracy: 0.9457, 	precision: 0.9651, 	recall: 0.7570, 	specificity: 0.9931, 	f1: 0.8485
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 87: 100%|██████████| 6195/6195 [08:58<00:00, 11.49it/s, loss=0.42]
Train Epoch 87 ==> 	accuracy: 0.8670, 	precision: 0.9996, 	recall: 0.7343, 	specificity: 0.9997, 	f1: 0.8467
Test Epoch 87: 100%|██████████| 1715/1715 [00:59<00:00, 28.60it/s, loss=0.157]
Test Epoch 87 ==> 	accuracy: 0.9465, 	precision: 0.9675, 	recall: 0.7593, 	specificity: 0.9936, 	f1: 0.8508
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 88: 100%|██████████| 6195/6195 [08:52<00:00, 11.63it/s, loss=0.513]
Train Epoch 88 ==> 	accuracy: 0.8679, 	precision: 0.9997, 	recall: 0.7361, 	specificity: 0.9998, 	f1: 0.8479
Test Epoch 88: 100%|██████████| 1715/1715 [00:58<00:00, 29.29it/s, loss=0.684]
Test Epoch 88 ==> 	accuracy: 0.9464, 	precision: 0.9721, 	recall: 0.7547, 	specificity: 0.9946, 	f1: 0.8497
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 89: 100%|██████████| 6195/6195 [09:03<00:00, 11.39it/s, loss=0.369]
Train Epoch 89 ==> 	accuracy: 0.8683, 	precision: 0.9996, 	recall: 0.7368, 	specificity: 0.9997, 	f1: 0.8484
Test Epoch 89: 100%|██████████| 1715/1715 [00:56<00:00, 30.46it/s, loss=2.43]
Test Epoch 89 ==> 	accuracy: 0.9457, 	precision: 0.9590, 	recall: 0.7621, 	specificity: 0.9918, 	f1: 0.8493
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 90: 100%|██████████| 6195/6195 [09:00<00:00, 11.45it/s, loss=0.288]
Train Epoch 90 ==> 	accuracy: 0.8692, 	precision: 0.9996, 	recall: 0.7388, 	specificity: 0.9997, 	f1: 0.8496
Test Epoch 90: 100%|██████████| 1715/1715 [00:59<00:00, 28.91it/s, loss=2.58]
Test Epoch 90 ==> 	accuracy: 0.9473, 	precision: 0.9644, 	recall: 0.7658, 	specificity: 0.9929, 	f1: 0.8537
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 91: 100%|██████████| 6195/6195 [08:58<00:00, 11.50it/s, loss=0.413]
Train Epoch 91 ==> 	accuracy: 0.8684, 	precision: 0.9996, 	recall: 0.7371, 	specificity: 0.9997, 	f1: 0.8485
Test Epoch 91: 100%|██████████| 1715/1715 [00:57<00:00, 29.77it/s, loss=1.02]
Test Epoch 91 ==> 	accuracy: 0.9464, 	precision: 0.9728, 	recall: 0.7541, 	specificity: 0.9947, 	f1: 0.8496
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 92: 100%|██████████| 6195/6195 [09:06<00:00, 11.33it/s, loss=0.395]
Train Epoch 92 ==> 	accuracy: 0.8684, 	precision: 0.9996, 	recall: 0.7372, 	specificity: 0.9997, 	f1: 0.8486
Test Epoch 92: 100%|██████████| 1715/1715 [00:56<00:00, 30.52it/s, loss=1.15]
Test Epoch 92 ==> 	accuracy: 0.9463, 	precision: 0.9742, 	recall: 0.7522, 	specificity: 0.9950, 	f1: 0.8490
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 93: 100%|██████████| 6195/6195 [08:53<00:00, 11.60it/s, loss=0.401]
Train Epoch 93 ==> 	accuracy: 0.8689, 	precision: 0.9997, 	recall: 0.7380, 	specificity: 0.9998, 	f1: 0.8491
Test Epoch 93: 100%|██████████| 1715/1715 [00:56<00:00, 30.56it/s, loss=0.102]
Test Epoch 93 ==> 	accuracy: 0.9471, 	precision: 0.9772, 	recall: 0.7540, 	specificity: 0.9956, 	f1: 0.8512
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 94: 100%|██████████| 6195/6195 [08:50<00:00, 11.68it/s, loss=0.403]
Train Epoch 94 ==> 	accuracy: 0.8678, 	precision: 0.9997, 	recall: 0.7358, 	specificity: 0.9998, 	f1: 0.8477
Test Epoch 94: 100%|██████████| 1715/1715 [00:57<00:00, 29.78it/s, loss=0.139]
Test Epoch 94 ==> 	accuracy: 0.9470, 	precision: 0.9695, 	recall: 0.7598, 	specificity: 0.9940, 	f1: 0.8519
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 95: 100%|██████████| 6195/6195 [08:54<00:00, 11.58it/s, loss=0.413]
Train Epoch 95 ==> 	accuracy: 0.8687, 	precision: 0.9996, 	recall: 0.7377, 	specificity: 0.9997, 	f1: 0.8489
Test Epoch 95: 100%|██████████| 1715/1715 [00:55<00:00, 30.71it/s, loss=1.91]
Test Epoch 95 ==> 	accuracy: 0.9488, 	precision: 0.9703, 	recall: 0.7684, 	specificity: 0.9941, 	f1: 0.8576
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 96: 100%|██████████| 6195/6195 [08:49<00:00, 11.70it/s, loss=0.427]
Train Epoch 96 ==> 	accuracy: 0.8704, 	precision: 0.9996, 	recall: 0.7411, 	specificity: 0.9997, 	f1: 0.8512
Test Epoch 96: 100%|██████████| 1715/1715 [01:06<00:00, 25.97it/s, loss=0.18]
Test Epoch 96 ==> 	accuracy: 0.9440, 	precision: 0.9739, 	recall: 0.7411, 	specificity: 0.9950, 	f1: 0.8417
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 97: 100%|██████████| 6195/6195 [08:57<00:00, 11.52it/s, loss=0.389]
Train Epoch 97 ==> 	accuracy: 0.8695, 	precision: 0.9997, 	recall: 0.7392, 	specificity: 0.9997, 	f1: 0.8499
Test Epoch 97: 100%|██████████| 1715/1715 [00:55<00:00, 31.01it/s, loss=0.587]
Test Epoch 97 ==> 	accuracy: 0.9470, 	precision: 0.9726, 	recall: 0.7575, 	specificity: 0.9946, 	f1: 0.8517
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 98: 100%|██████████| 6195/6195 [08:51<00:00, 11.66it/s, loss=0.402]
Train Epoch 98 ==> 	accuracy: 0.8715, 	precision: 0.9996, 	recall: 0.7433, 	specificity: 0.9997, 	f1: 0.8526
Test Epoch 98: 100%|██████████| 1715/1715 [00:56<00:00, 30.43it/s, loss=0.416]
Test Epoch 98 ==> 	accuracy: 0.9461, 	precision: 0.9677, 	recall: 0.7567, 	specificity: 0.9936, 	f1: 0.8493
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 99: 100%|██████████| 6195/6195 [08:50<00:00, 11.67it/s, loss=0.527]
Train Epoch 99 ==> 	accuracy: 0.8712, 	precision: 0.9997, 	recall: 0.7427, 	specificity: 0.9997, 	f1: 0.8522
Test Epoch 99: 100%|██████████| 1715/1715 [00:53<00:00, 32.04it/s, loss=0.872]
Test Epoch 99 ==> 	accuracy: 0.9477, 	precision: 0.9681, 	recall: 0.7648, 	specificity: 0.9937, 	f1: 0.8545
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 100: 100%|██████████| 6195/6195 [08:43<00:00, 11.83it/s, loss=0.393]
Train Epoch 100 ==> 	accuracy: 0.8725, 	precision: 0.9997, 	recall: 0.7452, 	specificity: 0.9997, 	f1: 0.8539
Test Epoch 100: 100%|██████████| 1715/1715 [01:03<00:00, 27.15it/s, loss=0.134]
Test Epoch 100 ==> 	accuracy: 0.9482, 	precision: 0.9687, 	recall: 0.7667, 	specificity: 0.9938, 	f1: 0.8560
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 101: 100%|██████████| 6195/6195 [08:36<00:00, 11.99it/s, loss=0.445]
Train Epoch 101 ==> 	accuracy: 0.8714, 	precision: 0.9996, 	recall: 0.7430, 	specificity: 0.9997, 	f1: 0.8524
Test Epoch 101: 100%|██████████| 1715/1715 [01:00<00:00, 28.15it/s, loss=0.298]
Test Epoch 101 ==> 	accuracy: 0.9475, 	precision: 0.9622, 	recall: 0.7689, 	specificity: 0.9924, 	f1: 0.8548
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 102: 100%|██████████| 6195/6195 [08:55<00:00, 11.57it/s, loss=0.329]
Train Epoch 102 ==> 	accuracy: 0.8737, 	precision: 0.9996, 	recall: 0.7476, 	specificity: 0.9997, 	f1: 0.8555
Test Epoch 102: 100%|██████████| 1715/1715 [00:56<00:00, 30.51it/s, loss=0.245]
Test Epoch 102 ==> 	accuracy: 0.9479, 	precision: 0.9609, 	recall: 0.7718, 	specificity: 0.9921, 	f1: 0.8560
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 103: 100%|██████████| 6195/6195 [08:34<00:00, 12.04it/s, loss=0.321]
Train Epoch 103 ==> 	accuracy: 0.8723, 	precision: 0.9997, 	recall: 0.7447, 	specificity: 0.9998, 	f1: 0.8536
Test Epoch 103: 100%|██████████| 1715/1715 [00:53<00:00, 31.84it/s, loss=0.368]
Test Epoch 103 ==> 	accuracy: 0.9472, 	precision: 0.9652, 	recall: 0.7646, 	specificity: 0.9931, 	f1: 0.8533
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 104: 100%|██████████| 6195/6195 [08:39<00:00, 11.92it/s, loss=0.331]
Train Epoch 104 ==> 	accuracy: 0.8711, 	precision: 0.9997, 	recall: 0.7424, 	specificity: 0.9998, 	f1: 0.8520
Test Epoch 104: 100%|██████████| 1715/1715 [00:57<00:00, 30.05it/s, loss=0.185]
Test Epoch 104 ==> 	accuracy: 0.9476, 	precision: 0.9749, 	recall: 0.7583, 	specificity: 0.9951, 	f1: 0.8531
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 105: 100%|██████████| 6195/6195 [08:46<00:00, 11.76it/s, loss=0.33]
Train Epoch 105 ==> 	accuracy: 0.8732, 	precision: 0.9997, 	recall: 0.7465, 	specificity: 0.9998, 	f1: 0.8548
Test Epoch 105: 100%|██████████| 1715/1715 [01:03<00:00, 27.22it/s, loss=0.269]
Test Epoch 105 ==> 	accuracy: 0.9485, 	precision: 0.9656, 	recall: 0.7710, 	specificity: 0.9931, 	f1: 0.8574
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 106: 100%|██████████| 6195/6195 [08:35<00:00, 12.03it/s, loss=0.311]
Train Epoch 106 ==> 	accuracy: 0.8745, 	precision: 0.9997, 	recall: 0.7493, 	specificity: 0.9997, 	f1: 0.8565
Test Epoch 106: 100%|██████████| 1715/1715 [01:01<00:00, 27.71it/s, loss=2.35]
Test Epoch 106 ==> 	accuracy: 0.9481, 	precision: 0.9651, 	recall: 0.7694, 	specificity: 0.9930, 	f1: 0.8562
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 107: 100%|██████████| 6195/6195 [08:42<00:00, 11.85it/s, loss=0.341]
Train Epoch 107 ==> 	accuracy: 0.8740, 	precision: 0.9997, 	recall: 0.7482, 	specificity: 0.9997, 	f1: 0.8558
Test Epoch 107: 100%|██████████| 1715/1715 [00:56<00:00, 30.50it/s, loss=0.189]
Test Epoch 107 ==> 	accuracy: 0.9473, 	precision: 0.9662, 	recall: 0.7643, 	specificity: 0.9933, 	f1: 0.8535
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 108: 100%|██████████| 6195/6195 [08:30<00:00, 12.13it/s, loss=0.402]
Train Epoch 108 ==> 	accuracy: 0.8723, 	precision: 0.9997, 	recall: 0.7449, 	specificity: 0.9998, 	f1: 0.8537
Test Epoch 108: 100%|██████████| 1715/1715 [00:53<00:00, 31.96it/s, loss=0.366]
Test Epoch 108 ==> 	accuracy: 0.9470, 	precision: 0.9678, 	recall: 0.7614, 	specificity: 0.9936, 	f1: 0.8523
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 109: 100%|██████████| 6195/6195 [08:40<00:00, 11.91it/s, loss=0.501]
Train Epoch 109 ==> 	accuracy: 0.8769, 	precision: 0.9997, 	recall: 0.7540, 	specificity: 0.9997, 	f1: 0.8596
Test Epoch 109: 100%|██████████| 1715/1715 [01:04<00:00, 26.71it/s, loss=0.698]
Test Epoch 109 ==> 	accuracy: 0.9483, 	precision: 0.9701, 	recall: 0.7662, 	specificity: 0.9941, 	f1: 0.8562
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 110: 100%|██████████| 6195/6195 [08:40<00:00, 11.91it/s, loss=0.501]
Train Epoch 110 ==> 	accuracy: 0.8745, 	precision: 0.9997, 	recall: 0.7493, 	specificity: 0.9998, 	f1: 0.8566
Test Epoch 110: 100%|██████████| 1715/1715 [01:04<00:00, 26.73it/s, loss=0.659]
Test Epoch 110 ==> 	accuracy: 0.9489, 	precision: 0.9685, 	recall: 0.7706, 	specificity: 0.9937, 	f1: 0.8583
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 111: 100%|██████████| 6195/6195 [08:46<00:00, 11.78it/s, loss=0.664]
Train Epoch 111 ==> 	accuracy: 0.8763, 	precision: 0.9997, 	recall: 0.7529, 	specificity: 0.9998, 	f1: 0.8589
Test Epoch 111: 100%|██████████| 1715/1715 [00:59<00:00, 28.72it/s, loss=1.04]
Test Epoch 111 ==> 	accuracy: 0.9491, 	precision: 0.9667, 	recall: 0.7730, 	specificity: 0.9933, 	f1: 0.8590
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 112: 100%|██████████| 6195/6195 [08:45<00:00, 11.78it/s, loss=0.384]
Train Epoch 112 ==> 	accuracy: 0.8751, 	precision: 0.9996, 	recall: 0.7505, 	specificity: 0.9997, 	f1: 0.8573
Test Epoch 112: 100%|██████████| 1715/1715 [00:55<00:00, 31.00it/s, loss=1.5]
Test Epoch 112 ==> 	accuracy: 0.9487, 	precision: 0.9654, 	recall: 0.7720, 	specificity: 0.9931, 	f1: 0.8580
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 113: 100%|██████████| 6195/6195 [08:41<00:00, 11.88it/s, loss=0.379]
Train Epoch 113 ==> 	accuracy: 0.8766, 	precision: 0.9997, 	recall: 0.7535, 	specificity: 0.9997, 	f1: 0.8593
Test Epoch 113: 100%|██████████| 1715/1715 [00:57<00:00, 29.69it/s, loss=0.335]
Test Epoch 113 ==> 	accuracy: 0.9495, 	precision: 0.9664, 	recall: 0.7752, 	specificity: 0.9932, 	f1: 0.8603
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 114: 100%|██████████| 6195/6195 [08:49<00:00, 11.70it/s, loss=0.544]
Train Epoch 114 ==> 	accuracy: 0.8748, 	precision: 0.9996, 	recall: 0.7498, 	specificity: 0.9997, 	f1: 0.8569
Test Epoch 114: 100%|██████████| 1715/1715 [00:55<00:00, 30.83it/s, loss=0.305]
Test Epoch 114 ==> 	accuracy: 0.9477, 	precision: 0.9708, 	recall: 0.7626, 	specificity: 0.9942, 	f1: 0.8542
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 115: 100%|██████████| 6195/6195 [08:50<00:00, 11.68it/s, loss=0.416]
Train Epoch 115 ==> 	accuracy: 0.8760, 	precision: 0.9997, 	recall: 0.7523, 	specificity: 0.9998, 	f1: 0.8585
Test Epoch 115: 100%|██████████| 1715/1715 [00:58<00:00, 29.20it/s, loss=0.258]
Test Epoch 115 ==> 	accuracy: 0.9465, 	precision: 0.9702, 	recall: 0.7567, 	specificity: 0.9942, 	f1: 0.8503
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 116: 100%|██████████| 6195/6195 [08:42<00:00, 11.86it/s, loss=0.343]
Train Epoch 116 ==> 	accuracy: 0.8754, 	precision: 0.9997, 	recall: 0.7511, 	specificity: 0.9998, 	f1: 0.8578
Test Epoch 116: 100%|██████████| 1715/1715 [00:55<00:00, 31.16it/s, loss=0.281]
Test Epoch 116 ==> 	accuracy: 0.9500, 	precision: 0.9738, 	recall: 0.7717, 	specificity: 0.9948, 	f1: 0.8610
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 117: 100%|██████████| 6195/6195 [08:51<00:00, 11.65it/s, loss=0.367]
Train Epoch 117 ==> 	accuracy: 0.8750, 	precision: 0.9997, 	recall: 0.7502, 	specificity: 0.9998, 	f1: 0.8571
Test Epoch 117: 100%|██████████| 1715/1715 [00:55<00:00, 30.99it/s, loss=1.71]
Test Epoch 117 ==> 	accuracy: 0.9496, 	precision: 0.9664, 	recall: 0.7758, 	specificity: 0.9932, 	f1: 0.8606
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 118: 100%|██████████| 6195/6195 [08:51<00:00, 11.66it/s, loss=0.465]
Train Epoch 118 ==> 	accuracy: 0.8757, 	precision: 0.9997, 	recall: 0.7516, 	specificity: 0.9998, 	f1: 0.8581
Test Epoch 118: 100%|██████████| 1715/1715 [00:54<00:00, 31.74it/s, loss=0.116]
Test Epoch 118 ==> 	accuracy: 0.9487, 	precision: 0.9667, 	recall: 0.7711, 	specificity: 0.9933, 	f1: 0.8579
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 119: 100%|██████████| 6195/6195 [08:34<00:00, 12.04it/s, loss=0.336]
Train Epoch 119 ==> 	accuracy: 0.8756, 	precision: 0.9997, 	recall: 0.7513, 	specificity: 0.9998, 	f1: 0.8579
Test Epoch 119: 100%|██████████| 1715/1715 [01:03<00:00, 27.14it/s, loss=0.121]
Test Epoch 119 ==> 	accuracy: 0.9492, 	precision: 0.9675, 	recall: 0.7727, 	specificity: 0.9935, 	f1: 0.8592
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 120: 100%|██████████| 6195/6195 [08:47<00:00, 11.75it/s, loss=0.369]
Train Epoch 120 ==> 	accuracy: 0.8757, 	precision: 0.9997, 	recall: 0.7517, 	specificity: 0.9997, 	f1: 0.8582
Test Epoch 120: 100%|██████████| 1715/1715 [00:54<00:00, 31.38it/s, loss=1.58]
Test Epoch 120 ==> 	accuracy: 0.9481, 	precision: 0.9688, 	recall: 0.7660, 	specificity: 0.9938, 	f1: 0.8556
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 121: 100%|██████████| 6195/6195 [08:43<00:00, 11.83it/s, loss=0.374]
Train Epoch 121 ==> 	accuracy: 0.8756, 	precision: 0.9997, 	recall: 0.7514, 	specificity: 0.9998, 	f1: 0.8580
Test Epoch 121: 100%|██████████| 1715/1715 [00:54<00:00, 31.36it/s, loss=0.435]
Test Epoch 121 ==> 	accuracy: 0.9491, 	precision: 0.9682, 	recall: 0.7717, 	specificity: 0.9936, 	f1: 0.8589
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 122: 100%|██████████| 6195/6195 [08:50<00:00, 11.67it/s, loss=0.398]
Train Epoch 122 ==> 	accuracy: 0.8793, 	precision: 0.9997, 	recall: 0.7588, 	specificity: 0.9998, 	f1: 0.8628
Test Epoch 122: 100%|██████████| 1715/1715 [01:00<00:00, 28.14it/s, loss=0.191]
Test Epoch 122 ==> 	accuracy: 0.9496, 	precision: 0.9635, 	recall: 0.7787, 	specificity: 0.9926, 	f1: 0.8613
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 123: 100%|██████████| 6195/6195 [08:35<00:00, 12.02it/s, loss=0.354]
Train Epoch 123 ==> 	accuracy: 0.8770, 	precision: 0.9997, 	recall: 0.7542, 	specificity: 0.9998, 	f1: 0.8598
Test Epoch 123: 100%|██████████| 1715/1715 [01:01<00:00, 27.79it/s, loss=0.218]
Test Epoch 123 ==> 	accuracy: 0.9494, 	precision: 0.9674, 	recall: 0.7741, 	specificity: 0.9934, 	f1: 0.8600
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 124: 100%|██████████| 6195/6195 [08:34<00:00, 12.05it/s, loss=0.48]
Train Epoch 124 ==> 	accuracy: 0.8777, 	precision: 0.9997, 	recall: 0.7555, 	specificity: 0.9998, 	f1: 0.8606
Test Epoch 124: 100%|██████████| 1715/1715 [00:54<00:00, 31.21it/s, loss=0.965]
Test Epoch 124 ==> 	accuracy: 0.9490, 	precision: 0.9604, 	recall: 0.7779, 	specificity: 0.9919, 	f1: 0.8596
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 125: 100%|██████████| 6195/6195 [08:37<00:00, 11.97it/s, loss=0.483]
Train Epoch 125 ==> 	accuracy: 0.8771, 	precision: 0.9997, 	recall: 0.7545, 	specificity: 0.9998, 	f1: 0.8600
Test Epoch 125: 100%|██████████| 1715/1715 [00:53<00:00, 32.26it/s, loss=0.552]
Test Epoch 125 ==> 	accuracy: 0.9491, 	precision: 0.9661, 	recall: 0.7738, 	specificity: 0.9932, 	f1: 0.8593
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 126: 100%|██████████| 6195/6195 [08:29<00:00, 12.16it/s, loss=0.394]
Train Epoch 126 ==> 	accuracy: 0.8792, 	precision: 0.9997, 	recall: 0.7587, 	specificity: 0.9997, 	f1: 0.8627
Test Epoch 126: 100%|██████████| 1715/1715 [01:01<00:00, 28.02it/s, loss=0.387]
Test Epoch 126 ==> 	accuracy: 0.9504, 	precision: 0.9637, 	recall: 0.7824, 	specificity: 0.9926, 	f1: 0.8636
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 127: 100%|██████████| 6195/6195 [08:33<00:00, 12.07it/s, loss=0.409]
Train Epoch 127 ==> 	accuracy: 0.8770, 	precision: 0.9997, 	recall: 0.7542, 	specificity: 0.9998, 	f1: 0.8598
Test Epoch 127: 100%|██████████| 1715/1715 [00:56<00:00, 30.48it/s, loss=0.402]
Test Epoch 127 ==> 	accuracy: 0.9499, 	precision: 0.9713, 	recall: 0.7731, 	specificity: 0.9943, 	f1: 0.8609
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 128: 100%|██████████| 6195/6195 [08:39<00:00, 11.92it/s, loss=0.351]
Train Epoch 128 ==> 	accuracy: 0.8761, 	precision: 0.9997, 	recall: 0.7524, 	specificity: 0.9998, 	f1: 0.8586
Test Epoch 128: 100%|██████████| 1715/1715 [00:54<00:00, 31.30it/s, loss=1.94]
Test Epoch 128 ==> 	accuracy: 0.9516, 	precision: 0.9754, 	recall: 0.7784, 	specificity: 0.9951, 	f1: 0.8658
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 129: 100%|██████████| 6195/6195 [08:38<00:00, 11.96it/s, loss=0.34]
Train Epoch 129 ==> 	accuracy: 0.8774, 	precision: 0.9997, 	recall: 0.7550, 	specificity: 0.9998, 	f1: 0.8603
Test Epoch 129: 100%|██████████| 1715/1715 [01:01<00:00, 27.82it/s, loss=0.139]
Test Epoch 129 ==> 	accuracy: 0.9497, 	precision: 0.9776, 	recall: 0.7671, 	specificity: 0.9956, 	f1: 0.8597
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 130: 100%|██████████| 6195/6195 [08:36<00:00, 11.99it/s, loss=0.336]
Train Epoch 130 ==> 	accuracy: 0.8768, 	precision: 0.9997, 	recall: 0.7537, 	specificity: 0.9998, 	f1: 0.8595
Test Epoch 130: 100%|██████████| 1715/1715 [00:57<00:00, 29.87it/s, loss=0.612]
Test Epoch 130 ==> 	accuracy: 0.9510, 	precision: 0.9772, 	recall: 0.7740, 	specificity: 0.9955, 	f1: 0.8638
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 131: 100%|██████████| 6195/6195 [08:40<00:00, 11.91it/s, loss=1.01]
Train Epoch 131 ==> 	accuracy: 0.8770, 	precision: 0.9997, 	recall: 0.7543, 	specificity: 0.9998, 	f1: 0.8598
Test Epoch 131: 100%|██████████| 1715/1715 [00:59<00:00, 28.61it/s, loss=0.0824]
Test Epoch 131 ==> 	accuracy: 0.9513, 	precision: 0.9758, 	recall: 0.7764, 	specificity: 0.9952, 	f1: 0.8648
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 132: 100%|██████████| 6195/6195 [08:27<00:00, 12.22it/s, loss=0.325]
Train Epoch 132 ==> 	accuracy: 0.8771, 	precision: 0.9997, 	recall: 0.7545, 	specificity: 0.9998, 	f1: 0.8599
Test Epoch 132: 100%|██████████| 1715/1715 [00:58<00:00, 29.15it/s, loss=1.06]
Test Epoch 132 ==> 	accuracy: 0.9515, 	precision: 0.9755, 	recall: 0.7782, 	specificity: 0.9951, 	f1: 0.8657
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 133: 100%|██████████| 6195/6195 [08:35<00:00, 12.03it/s, loss=0.45]
Train Epoch 133 ==> 	accuracy: 0.8798, 	precision: 0.9997, 	recall: 0.7599, 	specificity: 0.9998, 	f1: 0.8634
Test Epoch 133: 100%|██████████| 1715/1715 [00:54<00:00, 31.27it/s, loss=0.154]
Test Epoch 133 ==> 	accuracy: 0.9513, 	precision: 0.9764, 	recall: 0.7760, 	specificity: 0.9953, 	f1: 0.8647
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 134: 100%|██████████| 6195/6195 [08:34<00:00, 12.03it/s, loss=0.369]
Train Epoch 134 ==> 	accuracy: 0.8781, 	precision: 0.9997, 	recall: 0.7565, 	specificity: 0.9998, 	f1: 0.8613
Test Epoch 134: 100%|██████████| 1715/1715 [01:01<00:00, 27.83it/s, loss=0.543]
Test Epoch 134 ==> 	accuracy: 0.9498, 	precision: 0.9766, 	recall: 0.7681, 	specificity: 0.9954, 	f1: 0.8599
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 135: 100%|██████████| 6195/6195 [08:34<00:00, 12.05it/s, loss=0.305]
Train Epoch 135 ==> 	accuracy: 0.8808, 	precision: 0.9997, 	recall: 0.7619, 	specificity: 0.9998, 	f1: 0.8647
Test Epoch 135: 100%|██████████| 1715/1715 [00:59<00:00, 28.87it/s, loss=0.493]
Test Epoch 135 ==> 	accuracy: 0.9521, 	precision: 0.9700, 	recall: 0.7860, 	specificity: 0.9939, 	f1: 0.8683
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 136: 100%|██████████| 6195/6195 [08:41<00:00, 11.89it/s, loss=0.312]
Train Epoch 136 ==> 	accuracy: 0.8801, 	precision: 0.9997, 	recall: 0.7603, 	specificity: 0.9998, 	f1: 0.8637
Test Epoch 136: 100%|██████████| 1715/1715 [00:54<00:00, 31.23it/s, loss=0.18]
Test Epoch 136 ==> 	accuracy: 0.9517, 	precision: 0.9727, 	recall: 0.7815, 	specificity: 0.9945, 	f1: 0.8667
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 137: 100%|██████████| 6195/6195 [08:50<00:00, 11.67it/s, loss=0.363]
Train Epoch 137 ==> 	accuracy: 0.8802, 	precision: 0.9997, 	recall: 0.7607, 	specificity: 0.9998, 	f1: 0.8640
Test Epoch 137: 100%|██████████| 1715/1715 [00:59<00:00, 29.00it/s, loss=0.192]
Test Epoch 137 ==> 	accuracy: 0.9516, 	precision: 0.9730, 	recall: 0.7808, 	specificity: 0.9946, 	f1: 0.8664
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 138: 100%|██████████| 6195/6195 [08:40<00:00, 11.91it/s, loss=0.391]
Train Epoch 138 ==> 	accuracy: 0.8789, 	precision: 0.9997, 	recall: 0.7580, 	specificity: 0.9998, 	f1: 0.8622
Test Epoch 138: 100%|██████████| 1715/1715 [00:54<00:00, 31.39it/s, loss=0.662]
Test Epoch 138 ==> 	accuracy: 0.9510, 	precision: 0.9728, 	recall: 0.7779, 	specificity: 0.9945, 	f1: 0.8645
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 139: 100%|██████████| 6195/6195 [08:42<00:00, 11.85it/s, loss=0.382]
Train Epoch 139 ==> 	accuracy: 0.8815, 	precision: 0.9997, 	recall: 0.7633, 	specificity: 0.9998, 	f1: 0.8657
Test Epoch 139: 100%|██████████| 1715/1715 [00:57<00:00, 29.84it/s, loss=1.74]
Test Epoch 139 ==> 	accuracy: 0.9510, 	precision: 0.9731, 	recall: 0.7774, 	specificity: 0.9946, 	f1: 0.8643
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 140: 100%|██████████| 6195/6195 [08:42<00:00, 11.86it/s, loss=0.459]
Train Epoch 140 ==> 	accuracy: 0.8786, 	precision: 0.9997, 	recall: 0.7574, 	specificity: 0.9998, 	f1: 0.8618
Test Epoch 140: 100%|██████████| 1715/1715 [01:02<00:00, 27.59it/s, loss=0.131]
Test Epoch 140 ==> 	accuracy: 0.9518, 	precision: 0.9715, 	recall: 0.7830, 	specificity: 0.9942, 	f1: 0.8672
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 141: 100%|██████████| 6195/6195 [08:38<00:00, 11.95it/s, loss=0.389]
Train Epoch 141 ==> 	accuracy: 0.8780, 	precision: 0.9997, 	recall: 0.7563, 	specificity: 0.9998, 	f1: 0.8611
Test Epoch 141: 100%|██████████| 1715/1715 [00:57<00:00, 29.83it/s, loss=0.219]
Test Epoch 141 ==> 	accuracy: 0.9514, 	precision: 0.9744, 	recall: 0.7785, 	specificity: 0.9949, 	f1: 0.8655
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 142: 100%|██████████| 6195/6195 [08:41<00:00, 11.88it/s, loss=0.4]
Train Epoch 142 ==> 	accuracy: 0.8801, 	precision: 0.9997, 	recall: 0.7604, 	specificity: 0.9998, 	f1: 0.8638
Test Epoch 142: 100%|██████████| 1715/1715 [00:55<00:00, 30.94it/s, loss=0.142]
Test Epoch 142 ==> 	accuracy: 0.9512, 	precision: 0.9747, 	recall: 0.7768, 	specificity: 0.9949, 	f1: 0.8646
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 143: 100%|██████████| 6195/6195 [08:49<00:00, 11.71it/s, loss=0.321]
Train Epoch 143 ==> 	accuracy: 0.8787, 	precision: 0.9997, 	recall: 0.7576, 	specificity: 0.9998, 	f1: 0.8620
Test Epoch 143: 100%|██████████| 1715/1715 [00:56<00:00, 30.59it/s, loss=4.86]
Test Epoch 143 ==> 	accuracy: 0.9504, 	precision: 0.9769, 	recall: 0.7714, 	specificity: 0.9954, 	f1: 0.8621
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 144: 100%|██████████| 6195/6195 [08:47<00:00, 11.73it/s, loss=0.365]
Train Epoch 144 ==> 	accuracy: 0.8802, 	precision: 0.9997, 	recall: 0.7606, 	specificity: 0.9998, 	f1: 0.8639
Test Epoch 144: 100%|██████████| 1715/1715 [00:56<00:00, 30.58it/s, loss=0.293]
Test Epoch 144 ==> 	accuracy: 0.9516, 	precision: 0.9734, 	recall: 0.7801, 	specificity: 0.9946, 	f1: 0.8661
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 145: 100%|██████████| 6195/6195 [08:51<00:00, 11.66it/s, loss=0.384]
Train Epoch 145 ==> 	accuracy: 0.8795, 	precision: 0.9997, 	recall: 0.7591, 	specificity: 0.9998, 	f1: 0.8630
Test Epoch 145: 100%|██████████| 1715/1715 [01:03<00:00, 26.99it/s, loss=2.51]
Test Epoch 145 ==> 	accuracy: 0.9517, 	precision: 0.9711, 	recall: 0.7826, 	specificity: 0.9942, 	f1: 0.8667
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 146: 100%|██████████| 6195/6195 [08:42<00:00, 11.85it/s, loss=0.369]
Train Epoch 146 ==> 	accuracy: 0.8808, 	precision: 0.9997, 	recall: 0.7619, 	specificity: 0.9998, 	f1: 0.8647
Test Epoch 146: 100%|██████████| 1715/1715 [00:53<00:00, 32.13it/s, loss=0.261]
Test Epoch 146 ==> 	accuracy: 0.9517, 	precision: 0.9729, 	recall: 0.7814, 	specificity: 0.9945, 	f1: 0.8667
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 147: 100%|██████████| 6195/6195 [08:44<00:00, 11.82it/s, loss=0.35]
Train Epoch 147 ==> 	accuracy: 0.8809, 	precision: 0.9997, 	recall: 0.7620, 	specificity: 0.9998, 	f1: 0.8648
Test Epoch 147: 100%|██████████| 1715/1715 [00:55<00:00, 30.72it/s, loss=0.163]
Test Epoch 147 ==> 	accuracy: 0.9517, 	precision: 0.9748, 	recall: 0.7794, 	specificity: 0.9949, 	f1: 0.8662
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 148: 100%|██████████| 6195/6195 [08:40<00:00, 11.91it/s, loss=0.419]
Train Epoch 148 ==> 	accuracy: 0.8817, 	precision: 0.9997, 	recall: 0.7636, 	specificity: 0.9998, 	f1: 0.8659
Test Epoch 148: 100%|██████████| 1715/1715 [00:55<00:00, 31.15it/s, loss=0.141]
Test Epoch 148 ==> 	accuracy: 0.9519, 	precision: 0.9734, 	recall: 0.7818, 	specificity: 0.9946, 	f1: 0.8671
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 149: 100%|██████████| 6195/6195 [08:43<00:00, 11.82it/s, loss=0.324]
Train Epoch 149 ==> 	accuracy: 0.8813, 	precision: 0.9997, 	recall: 0.7629, 	specificity: 0.9998, 	f1: 0.8654
Test Epoch 149: 100%|██████████| 1715/1715 [00:53<00:00, 32.09it/s, loss=1.03]
Test Epoch 149 ==> 	accuracy: 0.9510, 	precision: 0.9741, 	recall: 0.7767, 	specificity: 0.9948, 	f1: 0.8643
Adjusting learning rate of group 0 to 4.2391e-06.

Process finished with exit code 0

'''

'''
ab seq
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6195/6195 [05:38<00:00, 18.28it/s, loss=0.781]
Train Epoch 0 ==> 	accuracy: 0.5764, 	precision: 0.9935, 	recall: 0.1538, 	specificity: 0.9990, 	f1: 0.2663
Test Epoch 0: 100%|██████████| 1715/1715 [00:40<00:00, 42.77it/s, loss=0.572]
Test Epoch 0 ==> 	accuracy: 0.8643, 	precision: 0.9844, 	recall: 0.3291, 	specificity: 0.9987, 	f1: 0.4933
Train Epoch 1: 100%|██████████| 6195/6195 [07:02<00:00, 14.68it/s, loss=0.642]
Train Epoch 1 ==> 	accuracy: 0.6427, 	precision: 0.9964, 	recall: 0.2865, 	specificity: 0.9990, 	f1: 0.4451
Test Epoch 1: 100%|██████████| 1715/1715 [00:48<00:00, 35.29it/s, loss=0.299]
Test Epoch 1 ==> 	accuracy: 0.8693, 	precision: 0.9908, 	recall: 0.3520, 	specificity: 0.9992, 	f1: 0.5195
Train Epoch 2: 100%|██████████| 6195/6195 [07:05<00:00, 14.57it/s, loss=0.691]
Train Epoch 2 ==> 	accuracy: 0.6737, 	precision: 0.9969, 	recall: 0.3485, 	specificity: 0.9989, 	f1: 0.5165
Test Epoch 2: 100%|██████████| 1715/1715 [00:48<00:00, 35.38it/s, loss=0.324]
Test Epoch 2 ==> 	accuracy: 0.8869, 	precision: 0.9849, 	recall: 0.4434, 	specificity: 0.9983, 	f1: 0.6115
Train Epoch 3: 100%|██████████| 6195/6195 [07:00<00:00, 14.72it/s, loss=0.586]
Train Epoch 3 ==> 	accuracy: 0.6836, 	precision: 0.9970, 	recall: 0.3684, 	specificity: 0.9989, 	f1: 0.5380
Test Epoch 3: 100%|██████████| 1715/1715 [00:49<00:00, 34.31it/s, loss=0.316]
Test Epoch 3 ==> 	accuracy: 0.8724, 	precision: 0.9756, 	recall: 0.3736, 	specificity: 0.9976, 	f1: 0.5403
Train Epoch 4: 100%|██████████| 6195/6195 [07:07<00:00, 14.48it/s, loss=0.606]
Train Epoch 4 ==> 	accuracy: 0.6954, 	precision: 0.9972, 	recall: 0.3918, 	specificity: 0.9989, 	f1: 0.5626
Test Epoch 4: 100%|██████████| 1715/1715 [00:52<00:00, 32.95it/s, loss=0.258]
Test Epoch 4 ==> 	accuracy: 0.8881, 	precision: 0.9890, 	recall: 0.4476, 	specificity: 0.9987, 	f1: 0.6163
Train Epoch 5: 100%|██████████| 6195/6195 [07:12<00:00, 14.33it/s, loss=0.593]
Train Epoch 5 ==> 	accuracy: 0.7012, 	precision: 0.9975, 	recall: 0.4035, 	specificity: 0.9990, 	f1: 0.5746
Test Epoch 5: 100%|██████████| 1715/1715 [00:51<00:00, 33.59it/s, loss=0.452]
Test Epoch 5 ==> 	accuracy: 0.8838, 	precision: 0.9899, 	recall: 0.4254, 	specificity: 0.9989, 	f1: 0.5951
Train Epoch 6: 100%|██████████| 6195/6195 [07:09<00:00, 14.42it/s, loss=1.17]
Train Epoch 6 ==> 	accuracy: 0.7071, 	precision: 0.9977, 	recall: 0.4152, 	specificity: 0.9990, 	f1: 0.5864
Test Epoch 6: 100%|██████████| 1715/1715 [00:52<00:00, 32.63it/s, loss=0.257]
Test Epoch 6 ==> 	accuracy: 0.8883, 	precision: 0.9912, 	recall: 0.4474, 	specificity: 0.9990, 	f1: 0.6165
Train Epoch 7: 100%|██████████| 6195/6195 [07:18<00:00, 14.13it/s, loss=1.39]
Train Epoch 7 ==> 	accuracy: 0.7127, 	precision: 0.9976, 	recall: 0.4265, 	specificity: 0.9990, 	f1: 0.5975
Test Epoch 7: 100%|██████████| 1715/1715 [00:52<00:00, 32.45it/s, loss=0.567]
Test Epoch 7 ==> 	accuracy: 0.8934, 	precision: 0.9916, 	recall: 0.4729, 	specificity: 0.9990, 	f1: 0.6404
Train Epoch 8: 100%|██████████| 6195/6195 [07:10<00:00, 14.40it/s, loss=0.611]
Train Epoch 8 ==> 	accuracy: 0.7097, 	precision: 0.9979, 	recall: 0.4204, 	specificity: 0.9991, 	f1: 0.5916
Test Epoch 8: 100%|██████████| 1715/1715 [00:52<00:00, 32.39it/s, loss=0.271]
Test Epoch 8 ==> 	accuracy: 0.8904, 	precision: 0.9910, 	recall: 0.4583, 	specificity: 0.9990, 	f1: 0.6267
Train Epoch 9: 100%|██████████| 6195/6195 [07:14<00:00, 14.25it/s, loss=0.701]
Train Epoch 9 ==> 	accuracy: 0.7164, 	precision: 0.9979, 	recall: 0.4337, 	specificity: 0.9991, 	f1: 0.6046
Test Epoch 9: 100%|██████████| 1715/1715 [00:47<00:00, 36.08it/s, loss=0.199]
Test Epoch 9 ==> 	accuracy: 0.8889, 	precision: 0.9924, 	recall: 0.4499, 	specificity: 0.9991, 	f1: 0.6191
Train Epoch 10: 100%|██████████| 6195/6195 [07:13<00:00, 14.30it/s, loss=0.56]
Train Epoch 10 ==> 	accuracy: 0.7180, 	precision: 0.9980, 	recall: 0.4370, 	specificity: 0.9991, 	f1: 0.6078
Test Epoch 10: 100%|██████████| 1715/1715 [00:46<00:00, 37.10it/s, loss=0.274]
Test Epoch 10 ==> 	accuracy: 0.8986, 	precision: 0.9888, 	recall: 0.5003, 	specificity: 0.9986, 	f1: 0.6645
Train Epoch 11: 100%|██████████| 6195/6195 [07:15<00:00, 14.22it/s, loss=0.661]
Train Epoch 11 ==> 	accuracy: 0.7231, 	precision: 0.9981, 	recall: 0.4470, 	specificity: 0.9991, 	f1: 0.6174
Test Epoch 11: 100%|██████████| 1715/1715 [00:47<00:00, 35.99it/s, loss=0.332]
Test Epoch 11 ==> 	accuracy: 0.8912, 	precision: 0.9908, 	recall: 0.4624, 	specificity: 0.9989, 	f1: 0.6305
Train Epoch 12: 100%|██████████| 6195/6195 [07:10<00:00, 14.40it/s, loss=0.66]
Train Epoch 12 ==> 	accuracy: 0.7252, 	precision: 0.9982, 	recall: 0.4513, 	specificity: 0.9992, 	f1: 0.6216
Test Epoch 12: 100%|██████████| 1715/1715 [00:51<00:00, 33.45it/s, loss=0.357]
Test Epoch 12 ==> 	accuracy: 0.8919, 	precision: 0.9925, 	recall: 0.4652, 	specificity: 0.9991, 	f1: 0.6335
Train Epoch 13: 100%|██████████| 6195/6195 [07:13<00:00, 14.29it/s, loss=0.607]
Train Epoch 13 ==> 	accuracy: 0.7263, 	precision: 0.9982, 	recall: 0.4535, 	specificity: 0.9992, 	f1: 0.6236
Test Epoch 13: 100%|██████████| 1715/1715 [00:48<00:00, 35.58it/s, loss=0.813]
Test Epoch 13 ==> 	accuracy: 0.8923, 	precision: 0.9549, 	recall: 0.4863, 	specificity: 0.9942, 	f1: 0.6444
Train Epoch 14: 100%|██████████| 6195/6195 [07:16<00:00, 14.20it/s, loss=0.583]
Train Epoch 14 ==> 	accuracy: 0.7337, 	precision: 0.9981, 	recall: 0.4684, 	specificity: 0.9991, 	f1: 0.6376
Test Epoch 14: 100%|██████████| 1715/1715 [00:49<00:00, 34.67it/s, loss=0.809]
Test Epoch 14 ==> 	accuracy: 0.9032, 	precision: 0.9848, 	recall: 0.5258, 	specificity: 0.9980, 	f1: 0.6856
Train Epoch 15: 100%|██████████| 6195/6195 [07:13<00:00, 14.28it/s, loss=0.594]
Train Epoch 15 ==> 	accuracy: 0.7294, 	precision: 0.9983, 	recall: 0.4595, 	specificity: 0.9992, 	f1: 0.6294
Test Epoch 15: 100%|██████████| 1715/1715 [00:48<00:00, 35.02it/s, loss=0.221]
Test Epoch 15 ==> 	accuracy: 0.8969, 	precision: 0.9917, 	recall: 0.4904, 	specificity: 0.9990, 	f1: 0.6562
Train Epoch 16: 100%|██████████| 6195/6195 [07:20<00:00, 14.05it/s, loss=0.535]
Train Epoch 16 ==> 	accuracy: 0.7313, 	precision: 0.9984, 	recall: 0.4634, 	specificity: 0.9992, 	f1: 0.6330
Test Epoch 16: 100%|██████████| 1715/1715 [00:47<00:00, 35.92it/s, loss=0.24]
Test Epoch 16 ==> 	accuracy: 0.8954, 	precision: 0.9917, 	recall: 0.4830, 	specificity: 0.9990, 	f1: 0.6496
Train Epoch 17: 100%|██████████| 6195/6195 [07:22<00:00, 14.00it/s, loss=0.663]
Train Epoch 17 ==> 	accuracy: 0.7361, 	precision: 0.9984, 	recall: 0.4729, 	specificity: 0.9992, 	f1: 0.6418
Test Epoch 17: 100%|██████████| 1715/1715 [00:52<00:00, 32.97it/s, loss=0.296]
Test Epoch 17 ==> 	accuracy: 0.8982, 	precision: 0.9900, 	recall: 0.4977, 	specificity: 0.9987, 	f1: 0.6624
Train Epoch 18: 100%|██████████| 6195/6195 [07:16<00:00, 14.20it/s, loss=0.535]
Train Epoch 18 ==> 	accuracy: 0.7384, 	precision: 0.9985, 	recall: 0.4775, 	specificity: 0.9993, 	f1: 0.6461
Test Epoch 18: 100%|██████████| 1715/1715 [00:48<00:00, 35.27it/s, loss=0.267]
Test Epoch 18 ==> 	accuracy: 0.8990, 	precision: 0.9900, 	recall: 0.5022, 	specificity: 0.9987, 	f1: 0.6664
Train Epoch 19: 100%|██████████| 6195/6195 [07:23<00:00, 13.97it/s, loss=0.549]
Train Epoch 19 ==> 	accuracy: 0.7378, 	precision: 0.9986, 	recall: 0.4762, 	specificity: 0.9993, 	f1: 0.6449
Test Epoch 19: 100%|██████████| 1715/1715 [00:48<00:00, 35.35it/s, loss=0.371]
Test Epoch 19 ==> 	accuracy: 0.9024, 	precision: 0.9895, 	recall: 0.5194, 	specificity: 0.9986, 	f1: 0.6812
Train Epoch 20: 100%|██████████| 6195/6195 [07:22<00:00, 13.99it/s, loss=0.635]
Train Epoch 20 ==> 	accuracy: 0.7386, 	precision: 0.9986, 	recall: 0.4779, 	specificity: 0.9993, 	f1: 0.6464
Test Epoch 20: 100%|██████████| 1715/1715 [00:47<00:00, 35.89it/s, loss=0.202]
Test Epoch 20 ==> 	accuracy: 0.9057, 	precision: 0.9870, 	recall: 0.5374, 	specificity: 0.9982, 	f1: 0.6959
Train Epoch 21: 100%|██████████| 6195/6195 [07:21<00:00, 14.02it/s, loss=0.634]
Train Epoch 21 ==> 	accuracy: 0.7456, 	precision: 0.9986, 	recall: 0.4919, 	specificity: 0.9993, 	f1: 0.6592
Test Epoch 21: 100%|██████████| 1715/1715 [00:49<00:00, 34.65it/s, loss=0.418]
Test Epoch 21 ==> 	accuracy: 0.9072, 	precision: 0.9863, 	recall: 0.5453, 	specificity: 0.9981, 	f1: 0.7023
Train Epoch 22: 100%|██████████| 6195/6195 [07:14<00:00, 14.27it/s, loss=0.624]
Train Epoch 22 ==> 	accuracy: 0.7402, 	precision: 0.9986, 	recall: 0.4810, 	specificity: 0.9993, 	f1: 0.6493
Test Epoch 22: 100%|██████████| 1715/1715 [00:49<00:00, 34.91it/s, loss=0.348]
Test Epoch 22 ==> 	accuracy: 0.8985, 	precision: 0.9915, 	recall: 0.4988, 	specificity: 0.9989, 	f1: 0.6637
Train Epoch 23: 100%|██████████| 6195/6195 [07:21<00:00, 14.03it/s, loss=0.555]
Train Epoch 23 ==> 	accuracy: 0.7468, 	precision: 0.9986, 	recall: 0.4944, 	specificity: 0.9993, 	f1: 0.6613
Test Epoch 23: 100%|██████████| 1715/1715 [00:54<00:00, 31.54it/s, loss=0.303]
Test Epoch 23 ==> 	accuracy: 0.9037, 	precision: 0.9822, 	recall: 0.5300, 	specificity: 0.9976, 	f1: 0.6885
Train Epoch 24: 100%|██████████| 6195/6195 [07:21<00:00, 14.04it/s, loss=0.562]
Train Epoch 24 ==> 	accuracy: 0.7435, 	precision: 0.9987, 	recall: 0.4876, 	specificity: 0.9994, 	f1: 0.6553
Test Epoch 24: 100%|██████████| 1715/1715 [00:50<00:00, 34.07it/s, loss=1.06]
Test Epoch 24 ==> 	accuracy: 0.9014, 	precision: 0.9901, 	recall: 0.5138, 	specificity: 0.9987, 	f1: 0.6765
Train Epoch 25: 100%|██████████| 6195/6195 [07:40<00:00, 13.44it/s, loss=0.643]
Train Epoch 25 ==> 	accuracy: 0.7494, 	precision: 0.9987, 	recall: 0.4994, 	specificity: 0.9993, 	f1: 0.6658
Test Epoch 25: 100%|██████████| 1715/1715 [00:50<00:00, 34.16it/s, loss=0.317]
Test Epoch 25 ==> 	accuracy: 0.9071, 	precision: 0.9873, 	recall: 0.5443, 	specificity: 0.9982, 	f1: 0.7018
Train Epoch 26: 100%|██████████| 6195/6195 [07:14<00:00, 14.27it/s, loss=0.533]
Train Epoch 26 ==> 	accuracy: 0.7477, 	precision: 0.9987, 	recall: 0.4961, 	specificity: 0.9994, 	f1: 0.6629
Test Epoch 26: 100%|██████████| 1715/1715 [00:45<00:00, 37.49it/s, loss=0.409]
Test Epoch 26 ==> 	accuracy: 0.9087, 	precision: 0.9852, 	recall: 0.5536, 	specificity: 0.9979, 	f1: 0.7089
Train Epoch 27: 100%|██████████| 6195/6195 [07:38<00:00, 13.51it/s, loss=0.531]
Train Epoch 27 ==> 	accuracy: 0.7517, 	precision: 0.9987, 	recall: 0.5040, 	specificity: 0.9994, 	f1: 0.6699
Test Epoch 27: 100%|██████████| 1715/1715 [00:50<00:00, 33.80it/s, loss=0.29]
Test Epoch 27 ==> 	accuracy: 0.9048, 	precision: 0.9863, 	recall: 0.5330, 	specificity: 0.9981, 	f1: 0.6920
Train Epoch 28: 100%|██████████| 6195/6195 [07:21<00:00, 14.04it/s, loss=0.491]
Train Epoch 28 ==> 	accuracy: 0.7506, 	precision: 0.9988, 	recall: 0.5018, 	specificity: 0.9994, 	f1: 0.6680
Test Epoch 28: 100%|██████████| 1715/1715 [00:53<00:00, 32.08it/s, loss=0.482]
Test Epoch 28 ==> 	accuracy: 0.9022, 	precision: 0.9880, 	recall: 0.5190, 	specificity: 0.9984, 	f1: 0.6805
Train Epoch 29: 100%|██████████| 6195/6195 [07:11<00:00, 14.35it/s, loss=0.663]
Train Epoch 29 ==> 	accuracy: 0.7507, 	precision: 0.9988, 	recall: 0.5020, 	specificity: 0.9994, 	f1: 0.6682
Test Epoch 29: 100%|██████████| 1715/1715 [00:52<00:00, 32.69it/s, loss=0.385]
Test Epoch 29 ==> 	accuracy: 0.9044, 	precision: 0.9878, 	recall: 0.5301, 	specificity: 0.9984, 	f1: 0.6899
Train Epoch 30: 100%|██████████| 6195/6195 [07:23<00:00, 13.96it/s, loss=0.49]
Train Epoch 30 ==> 	accuracy: 0.7507, 	precision: 0.9988, 	recall: 0.5020, 	specificity: 0.9994, 	f1: 0.6682
Test Epoch 30: 100%|██████████| 1715/1715 [00:45<00:00, 37.82it/s, loss=0.268]
Test Epoch 30 ==> 	accuracy: 0.8947, 	precision: 0.9927, 	recall: 0.4789, 	specificity: 0.9991, 	f1: 0.6461
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 31: 100%|██████████| 6195/6195 [07:14<00:00, 14.25it/s, loss=0.516]
Train Epoch 31 ==> 	accuracy: 0.7560, 	precision: 0.9989, 	recall: 0.5126, 	specificity: 0.9994, 	f1: 0.6775
Test Epoch 31: 100%|██████████| 1715/1715 [00:46<00:00, 36.50it/s, loss=0.64]
Test Epoch 31 ==> 	accuracy: 0.9005, 	precision: 0.9896, 	recall: 0.5098, 	specificity: 0.9987, 	f1: 0.6730
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 32: 100%|██████████| 6195/6195 [07:16<00:00, 14.19it/s, loss=0.68]
Train Epoch 32 ==> 	accuracy: 0.7517, 	precision: 0.9990, 	recall: 0.5040, 	specificity: 0.9995, 	f1: 0.6700
Test Epoch 32: 100%|██████████| 1715/1715 [00:51<00:00, 33.36it/s, loss=0.457]
Test Epoch 32 ==> 	accuracy: 0.9044, 	precision: 0.9881, 	recall: 0.5302, 	specificity: 0.9984, 	f1: 0.6901
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 33: 100%|██████████| 6195/6195 [07:24<00:00, 13.93it/s, loss=0.535]
Train Epoch 33 ==> 	accuracy: 0.7561, 	precision: 0.9989, 	recall: 0.5128, 	specificity: 0.9994, 	f1: 0.6777
Test Epoch 33: 100%|██████████| 1715/1715 [00:48<00:00, 35.63it/s, loss=1.2]
Test Epoch 33 ==> 	accuracy: 0.9072, 	precision: 0.9874, 	recall: 0.5448, 	specificity: 0.9982, 	f1: 0.7022
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 34: 100%|██████████| 6195/6195 [07:23<00:00, 13.97it/s, loss=0.548]
Train Epoch 34 ==> 	accuracy: 0.7586, 	precision: 0.9989, 	recall: 0.5177, 	specificity: 0.9995, 	f1: 0.6820
Test Epoch 34: 100%|██████████| 1715/1715 [00:51<00:00, 33.48it/s, loss=0.26]
Test Epoch 34 ==> 	accuracy: 0.9092, 	precision: 0.9870, 	recall: 0.5552, 	specificity: 0.9982, 	f1: 0.7106
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 35: 100%|██████████| 6195/6195 [07:17<00:00, 14.15it/s, loss=0.56]
Train Epoch 35 ==> 	accuracy: 0.7608, 	precision: 0.9990, 	recall: 0.5222, 	specificity: 0.9995, 	f1: 0.6859
Test Epoch 35: 100%|██████████| 1715/1715 [00:47<00:00, 36.06it/s, loss=0.303]
Test Epoch 35 ==> 	accuracy: 0.9051, 	precision: 0.9887, 	recall: 0.5333, 	specificity: 0.9985, 	f1: 0.6929
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 36: 100%|██████████| 6195/6195 [07:21<00:00, 14.03it/s, loss=0.599]
Train Epoch 36 ==> 	accuracy: 0.7635, 	precision: 0.9990, 	recall: 0.5276, 	specificity: 0.9995, 	f1: 0.6905
Test Epoch 36: 100%|██████████| 1715/1715 [00:49<00:00, 34.79it/s, loss=0.291]
Test Epoch 36 ==> 	accuracy: 0.9092, 	precision: 0.9710, 	recall: 0.5645, 	specificity: 0.9958, 	f1: 0.7139
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 37: 100%|██████████| 6195/6195 [07:22<00:00, 14.00it/s, loss=0.545]
Train Epoch 37 ==> 	accuracy: 0.7605, 	precision: 0.9990, 	recall: 0.5214, 	specificity: 0.9995, 	f1: 0.6852
Test Epoch 37: 100%|██████████| 1715/1715 [00:48<00:00, 35.14it/s, loss=0.883]
Test Epoch 37 ==> 	accuracy: 0.9090, 	precision: 0.9881, 	recall: 0.5532, 	specificity: 0.9983, 	f1: 0.7093
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 38: 100%|██████████| 6195/6195 [07:27<00:00, 13.85it/s, loss=0.521]
Train Epoch 38 ==> 	accuracy: 0.7685, 	precision: 0.9991, 	recall: 0.5374, 	specificity: 0.9995, 	f1: 0.6989
Test Epoch 38: 100%|██████████| 1715/1715 [00:52<00:00, 32.71it/s, loss=0.635]
Test Epoch 38 ==> 	accuracy: 0.9089, 	precision: 0.9859, 	recall: 0.5542, 	specificity: 0.9980, 	f1: 0.7096
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 39: 100%|██████████| 6195/6195 [07:22<00:00, 13.99it/s, loss=0.631]
Train Epoch 39 ==> 	accuracy: 0.7663, 	precision: 0.9991, 	recall: 0.5331, 	specificity: 0.9995, 	f1: 0.6952
Test Epoch 39: 100%|██████████| 1715/1715 [00:54<00:00, 31.65it/s, loss=1.23]
Test Epoch 39 ==> 	accuracy: 0.9096, 	precision: 0.9856, 	recall: 0.5579, 	specificity: 0.9980, 	f1: 0.7125
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 40: 100%|██████████| 6195/6195 [07:24<00:00, 13.95it/s, loss=0.538]
Train Epoch 40 ==> 	accuracy: 0.7718, 	precision: 0.9991, 	recall: 0.5441, 	specificity: 0.9995, 	f1: 0.7045
Test Epoch 40: 100%|██████████| 1715/1715 [00:49<00:00, 34.66it/s, loss=0.383]
Test Epoch 40 ==> 	accuracy: 0.9105, 	precision: 0.9839, 	recall: 0.5637, 	specificity: 0.9977, 	f1: 0.7167
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 41: 100%|██████████| 6195/6195 [07:18<00:00, 14.12it/s, loss=0.712]
Train Epoch 41 ==> 	accuracy: 0.7667, 	precision: 0.9991, 	recall: 0.5340, 	specificity: 0.9995, 	f1: 0.6960
Test Epoch 41: 100%|██████████| 1715/1715 [00:52<00:00, 32.91it/s, loss=0.356]
Test Epoch 41 ==> 	accuracy: 0.9103, 	precision: 0.9864, 	recall: 0.5610, 	specificity: 0.9981, 	f1: 0.7152
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 42: 100%|██████████| 6195/6195 [07:42<00:00, 13.39it/s, loss=0.529]
Train Epoch 42 ==> 	accuracy: 0.7760, 	precision: 0.9991, 	recall: 0.5525, 	specificity: 0.9995, 	f1: 0.7115
Test Epoch 42: 100%|██████████| 1715/1715 [00:47<00:00, 36.28it/s, loss=0.257]
Test Epoch 42 ==> 	accuracy: 0.9096, 	precision: 0.9880, 	recall: 0.5566, 	specificity: 0.9983, 	f1: 0.7121
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 43: 100%|██████████| 6195/6195 [07:14<00:00, 14.25it/s, loss=0.693]
Train Epoch 43 ==> 	accuracy: 0.7739, 	precision: 0.9992, 	recall: 0.5483, 	specificity: 0.9996, 	f1: 0.7081
Test Epoch 43: 100%|██████████| 1715/1715 [00:47<00:00, 36.18it/s, loss=0.273]
Test Epoch 43 ==> 	accuracy: 0.9165, 	precision: 0.9807, 	recall: 0.5958, 	specificity: 0.9970, 	f1: 0.7412
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 44: 100%|██████████| 6195/6195 [07:17<00:00, 14.15it/s, loss=0.446]
Train Epoch 44 ==> 	accuracy: 0.7735, 	precision: 0.9991, 	recall: 0.5474, 	specificity: 0.9995, 	f1: 0.7073
Test Epoch 44: 100%|██████████| 1715/1715 [00:51<00:00, 33.30it/s, loss=0.289]
Test Epoch 44 ==> 	accuracy: 0.9141, 	precision: 0.9838, 	recall: 0.5820, 	specificity: 0.9976, 	f1: 0.7313
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 45: 100%|██████████| 6195/6195 [07:24<00:00, 13.95it/s, loss=0.479]
Train Epoch 45 ==> 	accuracy: 0.7730, 	precision: 0.9991, 	recall: 0.5465, 	specificity: 0.9995, 	f1: 0.7065
Test Epoch 45: 100%|██████████| 1715/1715 [00:50<00:00, 33.90it/s, loss=0.382]
Test Epoch 45 ==> 	accuracy: 0.9130, 	precision: 0.9851, 	recall: 0.5754, 	specificity: 0.9978, 	f1: 0.7265
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 46: 100%|██████████| 6195/6195 [07:17<00:00, 14.17it/s, loss=0.537]
Train Epoch 46 ==> 	accuracy: 0.7787, 	precision: 0.9991, 	recall: 0.5580, 	specificity: 0.9995, 	f1: 0.7161
Test Epoch 46: 100%|██████████| 1715/1715 [00:53<00:00, 32.21it/s, loss=1.25]
Test Epoch 46 ==> 	accuracy: 0.9100, 	precision: 0.9684, 	recall: 0.5702, 	specificity: 0.9953, 	f1: 0.7177
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 47: 100%|██████████| 6195/6195 [07:14<00:00, 14.27it/s, loss=0.559]
Train Epoch 47 ==> 	accuracy: 0.7794, 	precision: 0.9992, 	recall: 0.5593, 	specificity: 0.9995, 	f1: 0.7172
Test Epoch 47: 100%|██████████| 1715/1715 [00:49<00:00, 34.57it/s, loss=0.224]
Test Epoch 47 ==> 	accuracy: 0.9140, 	precision: 0.9851, 	recall: 0.5806, 	specificity: 0.9978, 	f1: 0.7306
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 48: 100%|██████████| 6195/6195 [07:18<00:00, 14.13it/s, loss=0.519]
Train Epoch 48 ==> 	accuracy: 0.7786, 	precision: 0.9992, 	recall: 0.5577, 	specificity: 0.9996, 	f1: 0.7159
Test Epoch 48: 100%|██████████| 1715/1715 [00:48<00:00, 35.15it/s, loss=0.381]
Test Epoch 48 ==> 	accuracy: 0.9162, 	precision: 0.9826, 	recall: 0.5932, 	specificity: 0.9974, 	f1: 0.7398
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 49: 100%|██████████| 6195/6195 [07:24<00:00, 13.94it/s, loss=0.582]
Train Epoch 49 ==> 	accuracy: 0.7792, 	precision: 0.9992, 	recall: 0.5588, 	specificity: 0.9995, 	f1: 0.7168
Test Epoch 49: 100%|██████████| 1715/1715 [00:54<00:00, 31.32it/s, loss=1.81]
Test Epoch 49 ==> 	accuracy: 0.9154, 	precision: 0.9843, 	recall: 0.5879, 	specificity: 0.9976, 	f1: 0.7361
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 50: 100%|██████████| 6195/6195 [07:18<00:00, 14.14it/s, loss=0.488]
Train Epoch 50 ==> 	accuracy: 0.7833, 	precision: 0.9993, 	recall: 0.5670, 	specificity: 0.9996, 	f1: 0.7235
Test Epoch 50: 100%|██████████| 1715/1715 [00:48<00:00, 35.73it/s, loss=0.296]
Test Epoch 50 ==> 	accuracy: 0.9175, 	precision: 0.9815, 	recall: 0.6001, 	specificity: 0.9972, 	f1: 0.7448
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 51: 100%|██████████| 6195/6195 [07:25<00:00, 13.90it/s, loss=0.62]
Train Epoch 51 ==> 	accuracy: 0.7845, 	precision: 0.9992, 	recall: 0.5695, 	specificity: 0.9996, 	f1: 0.7255
Test Epoch 51: 100%|██████████| 1715/1715 [00:50<00:00, 34.08it/s, loss=0.257]
Test Epoch 51 ==> 	accuracy: 0.9176, 	precision: 0.9810, 	recall: 0.6012, 	specificity: 0.9971, 	f1: 0.7455
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 52: 100%|██████████| 6195/6195 [07:31<00:00, 13.71it/s, loss=0.525]
Train Epoch 52 ==> 	accuracy: 0.7857, 	precision: 0.9993, 	recall: 0.5717, 	specificity: 0.9996, 	f1: 0.7273
Test Epoch 52: 100%|██████████| 1715/1715 [00:52<00:00, 32.67it/s, loss=0.766]
Test Epoch 52 ==> 	accuracy: 0.9198, 	precision: 0.9797, 	recall: 0.6134, 	specificity: 0.9968, 	f1: 0.7544
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 53: 100%|██████████| 6195/6195 [07:28<00:00, 13.81it/s, loss=0.507]
Train Epoch 53 ==> 	accuracy: 0.7849, 	precision: 0.9992, 	recall: 0.5703, 	specificity: 0.9995, 	f1: 0.7261
Test Epoch 53: 100%|██████████| 1715/1715 [00:54<00:00, 31.41it/s, loss=0.38]
Test Epoch 53 ==> 	accuracy: 0.9181, 	precision: 0.9821, 	recall: 0.6032, 	specificity: 0.9972, 	f1: 0.7473
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 54: 100%|██████████| 6195/6195 [07:23<00:00, 13.97it/s, loss=0.489]
Train Epoch 54 ==> 	accuracy: 0.7872, 	precision: 0.9993, 	recall: 0.5749, 	specificity: 0.9996, 	f1: 0.7299
Test Epoch 54: 100%|██████████| 1715/1715 [00:50<00:00, 34.11it/s, loss=0.218]
Test Epoch 54 ==> 	accuracy: 0.9164, 	precision: 0.9830, 	recall: 0.5940, 	specificity: 0.9974, 	f1: 0.7405
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 55: 100%|██████████| 6195/6195 [07:24<00:00, 13.94it/s, loss=0.532]
Train Epoch 55 ==> 	accuracy: 0.7886, 	precision: 0.9993, 	recall: 0.5777, 	specificity: 0.9996, 	f1: 0.7321
Test Epoch 55: 100%|██████████| 1715/1715 [00:48<00:00, 35.49it/s, loss=0.242]
Test Epoch 55 ==> 	accuracy: 0.9195, 	precision: 0.9793, 	recall: 0.6119, 	specificity: 0.9968, 	f1: 0.7532
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 56: 100%|██████████| 6195/6195 [07:12<00:00, 14.33it/s, loss=0.642]
Train Epoch 56 ==> 	accuracy: 0.7898, 	precision: 0.9993, 	recall: 0.5800, 	specificity: 0.9996, 	f1: 0.7340
Test Epoch 56: 100%|██████████| 1715/1715 [00:52<00:00, 32.98it/s, loss=0.306]
Test Epoch 56 ==> 	accuracy: 0.9200, 	precision: 0.9649, 	recall: 0.6243, 	specificity: 0.9943, 	f1: 0.7581
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 57: 100%|██████████| 6195/6195 [07:08<00:00, 14.46it/s, loss=0.752]
Train Epoch 57 ==> 	accuracy: 0.7880, 	precision: 0.9993, 	recall: 0.5765, 	specificity: 0.9996, 	f1: 0.7312
Test Epoch 57: 100%|██████████| 1715/1715 [00:52<00:00, 32.64it/s, loss=0.448]
Test Epoch 57 ==> 	accuracy: 0.9210, 	precision: 0.9782, 	recall: 0.6203, 	specificity: 0.9965, 	f1: 0.7592
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 58: 100%|██████████| 6195/6195 [07:22<00:00, 14.01it/s, loss=0.506]
Train Epoch 58 ==> 	accuracy: 0.7899, 	precision: 0.9993, 	recall: 0.5801, 	specificity: 0.9996, 	f1: 0.7341
Test Epoch 58: 100%|██████████| 1715/1715 [00:50<00:00, 33.84it/s, loss=1.61]
Test Epoch 58 ==> 	accuracy: 0.9216, 	precision: 0.9772, 	recall: 0.6239, 	specificity: 0.9963, 	f1: 0.7616
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 59: 100%|██████████| 6195/6195 [07:17<00:00, 14.14it/s, loss=0.475]
Train Epoch 59 ==> 	accuracy: 0.7927, 	precision: 0.9993, 	recall: 0.5858, 	specificity: 0.9996, 	f1: 0.7387
Test Epoch 59: 100%|██████████| 1715/1715 [00:47<00:00, 36.02it/s, loss=0.554]
Test Epoch 59 ==> 	accuracy: 0.9204, 	precision: 0.9794, 	recall: 0.6165, 	specificity: 0.9967, 	f1: 0.7567
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 60: 100%|██████████| 6195/6195 [07:10<00:00, 14.40it/s, loss=0.495]
Train Epoch 60 ==> 	accuracy: 0.7933, 	precision: 0.9993, 	recall: 0.5869, 	specificity: 0.9996, 	f1: 0.7395
Test Epoch 60: 100%|██████████| 1715/1715 [00:50<00:00, 33.94it/s, loss=0.259]
Test Epoch 60 ==> 	accuracy: 0.9203, 	precision: 0.9772, 	recall: 0.6172, 	specificity: 0.9964, 	f1: 0.7566
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 61: 100%|██████████| 6195/6195 [07:36<00:00, 13.58it/s, loss=0.456]
Train Epoch 61 ==> 	accuracy: 0.7947, 	precision: 0.9994, 	recall: 0.5899, 	specificity: 0.9996, 	f1: 0.7419
Test Epoch 61: 100%|██████████| 1715/1715 [00:47<00:00, 35.91it/s, loss=0.792]
Test Epoch 61 ==> 	accuracy: 0.9216, 	precision: 0.9770, 	recall: 0.6243, 	specificity: 0.9963, 	f1: 0.7618
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 62: 100%|██████████| 6195/6195 [07:18<00:00, 14.13it/s, loss=0.521]
Train Epoch 62 ==> 	accuracy: 0.7933, 	precision: 0.9993, 	recall: 0.5870, 	specificity: 0.9996, 	f1: 0.7396
Test Epoch 62: 100%|██████████| 1715/1715 [00:45<00:00, 37.58it/s, loss=0.199]
Test Epoch 62 ==> 	accuracy: 0.9210, 	precision: 0.9777, 	recall: 0.6208, 	specificity: 0.9964, 	f1: 0.7594
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 63: 100%|██████████| 6195/6195 [07:20<00:00, 14.06it/s, loss=0.524]
Train Epoch 63 ==> 	accuracy: 0.7965, 	precision: 0.9993, 	recall: 0.5934, 	specificity: 0.9996, 	f1: 0.7446
Test Epoch 63: 100%|██████████| 1715/1715 [00:52<00:00, 32.42it/s, loss=0.309]
Test Epoch 63 ==> 	accuracy: 0.9199, 	precision: 0.9795, 	recall: 0.6140, 	specificity: 0.9968, 	f1: 0.7549
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 64: 100%|██████████| 6195/6195 [07:15<00:00, 14.22it/s, loss=0.421]
Train Epoch 64 ==> 	accuracy: 0.7973, 	precision: 0.9994, 	recall: 0.5949, 	specificity: 0.9996, 	f1: 0.7458
Test Epoch 64: 100%|██████████| 1715/1715 [00:51<00:00, 33.07it/s, loss=0.36]
Test Epoch 64 ==> 	accuracy: 0.9228, 	precision: 0.9737, 	recall: 0.6325, 	specificity: 0.9957, 	f1: 0.7668
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 65: 100%|██████████| 6195/6195 [07:17<00:00, 14.15it/s, loss=0.57]
Train Epoch 65 ==> 	accuracy: 0.7974, 	precision: 0.9994, 	recall: 0.5953, 	specificity: 0.9996, 	f1: 0.7461
Test Epoch 65: 100%|██████████| 1715/1715 [00:52<00:00, 32.93it/s, loss=0.508]
Test Epoch 65 ==> 	accuracy: 0.9189, 	precision: 0.9813, 	recall: 0.6078, 	specificity: 0.9971, 	f1: 0.7506
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 66: 100%|██████████| 6195/6195 [07:17<00:00, 14.16it/s, loss=0.508]
Train Epoch 66 ==> 	accuracy: 0.7983, 	precision: 0.9994, 	recall: 0.5969, 	specificity: 0.9996, 	f1: 0.7474
Test Epoch 66: 100%|██████████| 1715/1715 [00:51<00:00, 33.30it/s, loss=0.477]
Test Epoch 66 ==> 	accuracy: 0.9222, 	precision: 0.9765, 	recall: 0.6274, 	specificity: 0.9962, 	f1: 0.7640
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 67: 100%|██████████| 6195/6195 [07:33<00:00, 13.67it/s, loss=0.456]
Train Epoch 67 ==> 	accuracy: 0.8028, 	precision: 0.9994, 	recall: 0.6059, 	specificity: 0.9996, 	f1: 0.7544
Test Epoch 67: 100%|██████████| 1715/1715 [00:50<00:00, 34.21it/s, loss=0.189]
Test Epoch 67 ==> 	accuracy: 0.9228, 	precision: 0.9760, 	recall: 0.6311, 	specificity: 0.9961, 	f1: 0.7665
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 68: 100%|██████████| 6195/6195 [07:16<00:00, 14.21it/s, loss=0.516]
Train Epoch 68 ==> 	accuracy: 0.7974, 	precision: 0.9994, 	recall: 0.5952, 	specificity: 0.9996, 	f1: 0.7460
Test Epoch 68: 100%|██████████| 1715/1715 [00:53<00:00, 32.35it/s, loss=0.221]
Test Epoch 68 ==> 	accuracy: 0.9236, 	precision: 0.9739, 	recall: 0.6367, 	specificity: 0.9957, 	f1: 0.7700
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 69: 100%|██████████| 6195/6195 [07:11<00:00, 14.34it/s, loss=0.425]
Train Epoch 69 ==> 	accuracy: 0.8019, 	precision: 0.9994, 	recall: 0.6042, 	specificity: 0.9996, 	f1: 0.7531
Test Epoch 69: 100%|██████████| 1715/1715 [00:53<00:00, 32.04it/s, loss=0.456]
Test Epoch 69 ==> 	accuracy: 0.9228, 	precision: 0.9626, 	recall: 0.6401, 	specificity: 0.9937, 	f1: 0.7689
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 70: 100%|██████████| 6195/6195 [07:20<00:00, 14.07it/s, loss=0.685]
Train Epoch 70 ==> 	accuracy: 0.7989, 	precision: 0.9994, 	recall: 0.5983, 	specificity: 0.9996, 	f1: 0.7485
Test Epoch 70: 100%|██████████| 1715/1715 [00:51<00:00, 33.01it/s, loss=0.459]
Test Epoch 70 ==> 	accuracy: 0.9222, 	precision: 0.9763, 	recall: 0.6276, 	specificity: 0.9962, 	f1: 0.7640
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 71: 100%|██████████| 6195/6195 [07:13<00:00, 14.29it/s, loss=0.472]
Train Epoch 71 ==> 	accuracy: 0.8025, 	precision: 0.9994, 	recall: 0.6054, 	specificity: 0.9996, 	f1: 0.7540
Test Epoch 71: 100%|██████████| 1715/1715 [00:53<00:00, 32.06it/s, loss=0.283]
Test Epoch 71 ==> 	accuracy: 0.9245, 	precision: 0.9724, 	recall: 0.6423, 	specificity: 0.9954, 	f1: 0.7736
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 72: 100%|██████████| 6195/6195 [07:23<00:00, 13.98it/s, loss=0.539]
Train Epoch 72 ==> 	accuracy: 0.8016, 	precision: 0.9994, 	recall: 0.6035, 	specificity: 0.9996, 	f1: 0.7526
Test Epoch 72: 100%|██████████| 1715/1715 [00:50<00:00, 33.88it/s, loss=4.27]
Test Epoch 72 ==> 	accuracy: 0.9241, 	precision: 0.9740, 	recall: 0.6390, 	specificity: 0.9957, 	f1: 0.7717
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 73: 100%|██████████| 6195/6195 [07:00<00:00, 14.73it/s, loss=0.462]
Train Epoch 73 ==> 	accuracy: 0.8063, 	precision: 0.9994, 	recall: 0.6130, 	specificity: 0.9996, 	f1: 0.7599
Test Epoch 73: 100%|██████████| 1715/1715 [00:52<00:00, 32.39it/s, loss=0.307]
Test Epoch 73 ==> 	accuracy: 0.9204, 	precision: 0.9790, 	recall: 0.6165, 	specificity: 0.9967, 	f1: 0.7566
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 74: 100%|██████████| 6195/6195 [07:28<00:00, 13.80it/s, loss=0.539]
Train Epoch 74 ==> 	accuracy: 0.8024, 	precision: 0.9994, 	recall: 0.6052, 	specificity: 0.9996, 	f1: 0.7539
Test Epoch 74: 100%|██████████| 1715/1715 [00:46<00:00, 36.97it/s, loss=0.439]
Test Epoch 74 ==> 	accuracy: 0.9230, 	precision: 0.9764, 	recall: 0.6316, 	specificity: 0.9962, 	f1: 0.7671
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 75: 100%|██████████| 6195/6195 [07:16<00:00, 14.19it/s, loss=0.491]
Train Epoch 75 ==> 	accuracy: 0.8059, 	precision: 0.9994, 	recall: 0.6122, 	specificity: 0.9996, 	f1: 0.7593
Test Epoch 75: 100%|██████████| 1715/1715 [00:46<00:00, 36.99it/s, loss=1.65]
Test Epoch 75 ==> 	accuracy: 0.9218, 	precision: 0.9782, 	recall: 0.6243, 	specificity: 0.9965, 	f1: 0.7622
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 76: 100%|██████████| 6195/6195 [07:12<00:00, 14.33it/s, loss=0.573]
Train Epoch 76 ==> 	accuracy: 0.8088, 	precision: 0.9995, 	recall: 0.6180, 	specificity: 0.9997, 	f1: 0.7638
Test Epoch 76: 100%|██████████| 1715/1715 [00:50<00:00, 33.98it/s, loss=0.276]
Test Epoch 76 ==> 	accuracy: 0.9233, 	precision: 0.9754, 	recall: 0.6340, 	specificity: 0.9960, 	f1: 0.7685
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 77: 100%|██████████| 6195/6195 [07:14<00:00, 14.25it/s, loss=0.461]
Train Epoch 77 ==> 	accuracy: 0.8084, 	precision: 0.9995, 	recall: 0.6172, 	specificity: 0.9997, 	f1: 0.7631
Test Epoch 77: 100%|██████████| 1715/1715 [00:44<00:00, 38.69it/s, loss=0.234]
Test Epoch 77 ==> 	accuracy: 0.9247, 	precision: 0.9734, 	recall: 0.6425, 	specificity: 0.9956, 	f1: 0.7741
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 78: 100%|██████████| 6195/6195 [07:15<00:00, 14.23it/s, loss=0.442]
Train Epoch 78 ==> 	accuracy: 0.8074, 	precision: 0.9994, 	recall: 0.6151, 	specificity: 0.9996, 	f1: 0.7615
Test Epoch 78: 100%|██████████| 1715/1715 [00:48<00:00, 35.13it/s, loss=1.07]
Test Epoch 78 ==> 	accuracy: 0.9253, 	precision: 0.9733, 	recall: 0.6454, 	specificity: 0.9956, 	f1: 0.7762
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 79: 100%|██████████| 6195/6195 [07:12<00:00, 14.33it/s, loss=0.518]
Train Epoch 79 ==> 	accuracy: 0.8081, 	precision: 0.9994, 	recall: 0.6165, 	specificity: 0.9996, 	f1: 0.7626
Test Epoch 79: 100%|██████████| 1715/1715 [00:45<00:00, 37.97it/s, loss=0.55]
Test Epoch 79 ==> 	accuracy: 0.9235, 	precision: 0.9649, 	recall: 0.6424, 	specificity: 0.9941, 	f1: 0.7713
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 80: 100%|██████████| 6195/6195 [07:11<00:00, 14.34it/s, loss=0.544]
Train Epoch 80 ==> 	accuracy: 0.8078, 	precision: 0.9994, 	recall: 0.6159, 	specificity: 0.9996, 	f1: 0.7621
Test Epoch 80: 100%|██████████| 1715/1715 [00:48<00:00, 35.39it/s, loss=0.279]
Test Epoch 80 ==> 	accuracy: 0.9271, 	precision: 0.9696, 	recall: 0.6573, 	specificity: 0.9948, 	f1: 0.7835
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 81: 100%|██████████| 6195/6195 [07:22<00:00, 14.01it/s, loss=0.454]
Train Epoch 81 ==> 	accuracy: 0.8067, 	precision: 0.9994, 	recall: 0.6138, 	specificity: 0.9996, 	f1: 0.7605
Test Epoch 81: 100%|██████████| 1715/1715 [00:46<00:00, 36.87it/s, loss=0.214]
Test Epoch 81 ==> 	accuracy: 0.9264, 	precision: 0.9710, 	recall: 0.6529, 	specificity: 0.9951, 	f1: 0.7808
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 82: 100%|██████████| 6195/6195 [07:17<00:00, 14.16it/s, loss=0.79]
Train Epoch 82 ==> 	accuracy: 0.8096, 	precision: 0.9994, 	recall: 0.6196, 	specificity: 0.9997, 	f1: 0.7650
Test Epoch 82: 100%|██████████| 1715/1715 [00:43<00:00, 39.12it/s, loss=1.18]
Test Epoch 82 ==> 	accuracy: 0.9232, 	precision: 0.9750, 	recall: 0.6337, 	specificity: 0.9959, 	f1: 0.7682
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 83: 100%|██████████| 6195/6195 [07:15<00:00, 14.23it/s, loss=0.567]
Train Epoch 83 ==> 	accuracy: 0.8097, 	precision: 0.9995, 	recall: 0.6196, 	specificity: 0.9997, 	f1: 0.7650
Test Epoch 83: 100%|██████████| 1715/1715 [00:48<00:00, 35.20it/s, loss=0.312]
Test Epoch 83 ==> 	accuracy: 0.9251, 	precision: 0.9715, 	recall: 0.6460, 	specificity: 0.9952, 	f1: 0.7760
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 84: 100%|██████████| 6195/6195 [07:21<00:00, 14.02it/s, loss=0.457]
Train Epoch 84 ==> 	accuracy: 0.8125, 	precision: 0.9994, 	recall: 0.6253, 	specificity: 0.9996, 	f1: 0.7693
Test Epoch 84: 100%|██████████| 1715/1715 [00:45<00:00, 38.08it/s, loss=2.53]
Test Epoch 84 ==> 	accuracy: 0.9242, 	precision: 0.9768, 	recall: 0.6375, 	specificity: 0.9962, 	f1: 0.7715
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 85: 100%|██████████| 6195/6195 [07:16<00:00, 14.19it/s, loss=0.494]
Train Epoch 85 ==> 	accuracy: 0.8127, 	precision: 0.9995, 	recall: 0.6258, 	specificity: 0.9997, 	f1: 0.7697
Test Epoch 85: 100%|██████████| 1715/1715 [00:46<00:00, 37.06it/s, loss=0.294]
Test Epoch 85 ==> 	accuracy: 0.9253, 	precision: 0.9737, 	recall: 0.6451, 	specificity: 0.9956, 	f1: 0.7761
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 86: 100%|██████████| 6195/6195 [07:10<00:00, 14.38it/s, loss=0.441]
Train Epoch 86 ==> 	accuracy: 0.8144, 	precision: 0.9995, 	recall: 0.6291, 	specificity: 0.9997, 	f1: 0.7722
Test Epoch 86: 100%|██████████| 1715/1715 [00:47<00:00, 36.11it/s, loss=1.43]
Test Epoch 86 ==> 	accuracy: 0.9274, 	precision: 0.9688, 	recall: 0.6595, 	specificity: 0.9947, 	f1: 0.7848
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 87: 100%|██████████| 6195/6195 [07:14<00:00, 14.27it/s, loss=0.489]
Train Epoch 87 ==> 	accuracy: 0.8121, 	precision: 0.9994, 	recall: 0.6246, 	specificity: 0.9996, 	f1: 0.7688
Test Epoch 87: 100%|██████████| 1715/1715 [00:48<00:00, 35.63it/s, loss=0.402]
Test Epoch 87 ==> 	accuracy: 0.9273, 	precision: 0.9699, 	recall: 0.6581, 	specificity: 0.9949, 	f1: 0.7842
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 88: 100%|██████████| 6195/6195 [07:07<00:00, 14.48it/s, loss=0.474]
Train Epoch 88 ==> 	accuracy: 0.8160, 	precision: 0.9995, 	recall: 0.6323, 	specificity: 0.9997, 	f1: 0.7746
Test Epoch 88: 100%|██████████| 1715/1715 [00:46<00:00, 36.90it/s, loss=0.84]
Test Epoch 88 ==> 	accuracy: 0.9289, 	precision: 0.9657, 	recall: 0.6696, 	specificity: 0.9940, 	f1: 0.7909
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 89: 100%|██████████| 6195/6195 [07:12<00:00, 14.32it/s, loss=0.424]
Train Epoch 89 ==> 	accuracy: 0.8159, 	precision: 0.9995, 	recall: 0.6321, 	specificity: 0.9997, 	f1: 0.7745
Test Epoch 89: 100%|██████████| 1715/1715 [00:48<00:00, 35.24it/s, loss=0.284]
Test Epoch 89 ==> 	accuracy: 0.9258, 	precision: 0.9637, 	recall: 0.6549, 	specificity: 0.9938, 	f1: 0.7798
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 90: 100%|██████████| 6195/6195 [07:08<00:00, 14.47it/s, loss=0.454]
Train Epoch 90 ==> 	accuracy: 0.8163, 	precision: 0.9995, 	recall: 0.6330, 	specificity: 0.9997, 	f1: 0.7751
Test Epoch 90: 100%|██████████| 1715/1715 [00:47<00:00, 36.18it/s, loss=0.3]
Test Epoch 90 ==> 	accuracy: 0.9254, 	precision: 0.9732, 	recall: 0.6464, 	specificity: 0.9955, 	f1: 0.7768
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 91: 100%|██████████| 6195/6195 [07:09<00:00, 14.41it/s, loss=0.568]
Train Epoch 91 ==> 	accuracy: 0.8141, 	precision: 0.9995, 	recall: 0.6285, 	specificity: 0.9997, 	f1: 0.7717
Test Epoch 91: 100%|██████████| 1715/1715 [00:47<00:00, 35.82it/s, loss=0.222]
Test Epoch 91 ==> 	accuracy: 0.9266, 	precision: 0.9696, 	recall: 0.6548, 	specificity: 0.9948, 	f1: 0.7817
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 92: 100%|██████████| 6195/6195 [07:08<00:00, 14.44it/s, loss=0.495]
Train Epoch 92 ==> 	accuracy: 0.8143, 	precision: 0.9994, 	recall: 0.6289, 	specificity: 0.9996, 	f1: 0.7720
Test Epoch 92: 100%|██████████| 1715/1715 [00:42<00:00, 40.55it/s, loss=0.23]
Test Epoch 92 ==> 	accuracy: 0.9283, 	precision: 0.9677, 	recall: 0.6649, 	specificity: 0.9944, 	f1: 0.7882
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 93: 100%|██████████| 6195/6195 [07:01<00:00, 14.71it/s, loss=0.45]
Train Epoch 93 ==> 	accuracy: 0.8163, 	precision: 0.9995, 	recall: 0.6328, 	specificity: 0.9997, 	f1: 0.7750
Test Epoch 93: 100%|██████████| 1715/1715 [00:44<00:00, 38.60it/s, loss=3.02]
Test Epoch 93 ==> 	accuracy: 0.9247, 	precision: 0.9748, 	recall: 0.6417, 	specificity: 0.9958, 	f1: 0.7739
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 94: 100%|██████████| 6195/6195 [06:58<00:00, 14.82it/s, loss=0.478]
Train Epoch 94 ==> 	accuracy: 0.8139, 	precision: 0.9995, 	recall: 0.6282, 	specificity: 0.9997, 	f1: 0.7715
Test Epoch 94: 100%|██████████| 1715/1715 [00:45<00:00, 37.70it/s, loss=0.761]
Test Epoch 94 ==> 	accuracy: 0.9274, 	precision: 0.9700, 	recall: 0.6590, 	specificity: 0.9949, 	f1: 0.7848
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 95: 100%|██████████| 6195/6195 [07:09<00:00, 14.42it/s, loss=0.412]
Train Epoch 95 ==> 	accuracy: 0.8149, 	precision: 0.9995, 	recall: 0.6301, 	specificity: 0.9997, 	f1: 0.7729
Test Epoch 95: 100%|██████████| 1715/1715 [00:44<00:00, 38.35it/s, loss=0.411]
Test Epoch 95 ==> 	accuracy: 0.9274, 	precision: 0.9702, 	recall: 0.6586, 	specificity: 0.9949, 	f1: 0.7846
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 96: 100%|██████████| 6195/6195 [07:02<00:00, 14.66it/s, loss=0.721]
Train Epoch 96 ==> 	accuracy: 0.8160, 	precision: 0.9995, 	recall: 0.6323, 	specificity: 0.9997, 	f1: 0.7746
Test Epoch 96: 100%|██████████| 1715/1715 [00:44<00:00, 38.13it/s, loss=0.535]
Test Epoch 96 ==> 	accuracy: 0.9288, 	precision: 0.9671, 	recall: 0.6682, 	specificity: 0.9943, 	f1: 0.7903
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 97: 100%|██████████| 6195/6195 [07:12<00:00, 14.31it/s, loss=0.484]
Train Epoch 97 ==> 	accuracy: 0.8175, 	precision: 0.9995, 	recall: 0.6354, 	specificity: 0.9997, 	f1: 0.7769
Test Epoch 97: 100%|██████████| 1715/1715 [00:49<00:00, 34.33it/s, loss=0.39]
Test Epoch 97 ==> 	accuracy: 0.9279, 	precision: 0.9701, 	recall: 0.6610, 	specificity: 0.9949, 	f1: 0.7863
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 98: 100%|██████████| 6195/6195 [07:13<00:00, 14.30it/s, loss=0.451]
Train Epoch 98 ==> 	accuracy: 0.8185, 	precision: 0.9995, 	recall: 0.6373, 	specificity: 0.9997, 	f1: 0.7783
Test Epoch 98: 100%|██████████| 1715/1715 [00:46<00:00, 36.66it/s, loss=2.29]
Test Epoch 98 ==> 	accuracy: 0.9293, 	precision: 0.9663, 	recall: 0.6712, 	specificity: 0.9941, 	f1: 0.7921
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 99: 100%|██████████| 6195/6195 [07:16<00:00, 14.20it/s, loss=0.45]
Train Epoch 99 ==> 	accuracy: 0.8178, 	precision: 0.9995, 	recall: 0.6360, 	specificity: 0.9997, 	f1: 0.7773
Test Epoch 99: 100%|██████████| 1715/1715 [00:49<00:00, 34.83it/s, loss=0.336]
Test Epoch 99 ==> 	accuracy: 0.9291, 	precision: 0.9672, 	recall: 0.6693, 	specificity: 0.9943, 	f1: 0.7912
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 100: 100%|██████████| 6195/6195 [07:10<00:00, 14.39it/s, loss=0.435]
Train Epoch 100 ==> 	accuracy: 0.8204, 	precision: 0.9995, 	recall: 0.6411, 	specificity: 0.9997, 	f1: 0.7812
Test Epoch 100: 100%|██████████| 1715/1715 [00:43<00:00, 39.81it/s, loss=0.45]
Test Epoch 100 ==> 	accuracy: 0.9281, 	precision: 0.9683, 	recall: 0.6637, 	specificity: 0.9945, 	f1: 0.7876
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 101: 100%|██████████| 6195/6195 [07:09<00:00, 14.42it/s, loss=0.429]
Train Epoch 101 ==> 	accuracy: 0.8192, 	precision: 0.9996, 	recall: 0.6386, 	specificity: 0.9997, 	f1: 0.7793
Test Epoch 101: 100%|██████████| 1715/1715 [00:47<00:00, 36.46it/s, loss=0.174]
Test Epoch 101 ==> 	accuracy: 0.9277, 	precision: 0.9701, 	recall: 0.6604, 	specificity: 0.9949, 	f1: 0.7858
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 102: 100%|██████████| 6195/6195 [07:12<00:00, 14.32it/s, loss=0.544]
Train Epoch 102 ==> 	accuracy: 0.8231, 	precision: 0.9995, 	recall: 0.6465, 	specificity: 0.9997, 	f1: 0.7851
Test Epoch 102: 100%|██████████| 1715/1715 [00:47<00:00, 36.09it/s, loss=0.745]
Test Epoch 102 ==> 	accuracy: 0.9288, 	precision: 0.9589, 	recall: 0.6743, 	specificity: 0.9927, 	f1: 0.7918
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 103: 100%|██████████| 6195/6195 [07:04<00:00, 14.60it/s, loss=0.512]
Train Epoch 103 ==> 	accuracy: 0.8194, 	precision: 0.9995, 	recall: 0.6390, 	specificity: 0.9997, 	f1: 0.7796
Test Epoch 103: 100%|██████████| 1715/1715 [00:46<00:00, 36.92it/s, loss=0.265]
Test Epoch 103 ==> 	accuracy: 0.9295, 	precision: 0.9654, 	recall: 0.6730, 	specificity: 0.9939, 	f1: 0.7931
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 104: 100%|██████████| 6195/6195 [07:19<00:00, 14.10it/s, loss=0.511]
Train Epoch 104 ==> 	accuracy: 0.8184, 	precision: 0.9995, 	recall: 0.6371, 	specificity: 0.9997, 	f1: 0.7782
Test Epoch 104: 100%|██████████| 1715/1715 [00:44<00:00, 38.62it/s, loss=2.15]
Test Epoch 104 ==> 	accuracy: 0.9304, 	precision: 0.9625, 	recall: 0.6797, 	specificity: 0.9933, 	f1: 0.7968
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 105: 100%|██████████| 6195/6195 [07:18<00:00, 14.12it/s, loss=0.447]
Train Epoch 105 ==> 	accuracy: 0.8188, 	precision: 0.9995, 	recall: 0.6380, 	specificity: 0.9997, 	f1: 0.7788
Test Epoch 105: 100%|██████████| 1715/1715 [00:48<00:00, 35.42it/s, loss=0.346]
Test Epoch 105 ==> 	accuracy: 0.9296, 	precision: 0.9664, 	recall: 0.6725, 	specificity: 0.9941, 	f1: 0.7931
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 106: 100%|██████████| 6195/6195 [07:16<00:00, 14.21it/s, loss=0.477]
Train Epoch 106 ==> 	accuracy: 0.8215, 	precision: 0.9995, 	recall: 0.6433, 	specificity: 0.9997, 	f1: 0.7828
Test Epoch 106: 100%|██████████| 1715/1715 [00:48<00:00, 35.19it/s, loss=0.871]
Test Epoch 106 ==> 	accuracy: 0.9292, 	precision: 0.9658, 	recall: 0.6711, 	specificity: 0.9940, 	f1: 0.7919
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 107: 100%|██████████| 6195/6195 [07:07<00:00, 14.48it/s, loss=0.441]
Train Epoch 107 ==> 	accuracy: 0.8204, 	precision: 0.9995, 	recall: 0.6411, 	specificity: 0.9997, 	f1: 0.7811
Test Epoch 107: 100%|██████████| 1715/1715 [00:45<00:00, 37.93it/s, loss=0.265]
Test Epoch 107 ==> 	accuracy: 0.9297, 	precision: 0.9645, 	recall: 0.6747, 	specificity: 0.9938, 	f1: 0.7940
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 108: 100%|██████████| 6195/6195 [07:07<00:00, 14.48it/s, loss=0.489]
Train Epoch 108 ==> 	accuracy: 0.8188, 	precision: 0.9995, 	recall: 0.6379, 	specificity: 0.9997, 	f1: 0.7788
Test Epoch 108: 100%|██████████| 1715/1715 [00:44<00:00, 38.39it/s, loss=0.383]
Test Epoch 108 ==> 	accuracy: 0.9291, 	precision: 0.9663, 	recall: 0.6700, 	specificity: 0.9941, 	f1: 0.7913
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 109: 100%|██████████| 6195/6195 [07:03<00:00, 14.64it/s, loss=0.391]
Train Epoch 109 ==> 	accuracy: 0.8244, 	precision: 0.9995, 	recall: 0.6492, 	specificity: 0.9997, 	f1: 0.7871
Test Epoch 109: 100%|██████████| 1715/1715 [00:50<00:00, 34.05it/s, loss=0.309]
Test Epoch 109 ==> 	accuracy: 0.9302, 	precision: 0.9636, 	recall: 0.6780, 	specificity: 0.9936, 	f1: 0.7960
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 110: 100%|██████████| 6195/6195 [07:07<00:00, 14.50it/s, loss=0.373]
Train Epoch 110 ==> 	accuracy: 0.8218, 	precision: 0.9995, 	recall: 0.6438, 	specificity: 0.9997, 	f1: 0.7832
Test Epoch 110: 100%|██████████| 1715/1715 [00:47<00:00, 36.48it/s, loss=0.718]
Test Epoch 110 ==> 	accuracy: 0.9305, 	precision: 0.9623, 	recall: 0.6804, 	specificity: 0.9933, 	f1: 0.7971
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 111: 100%|██████████| 6195/6195 [07:02<00:00, 14.67it/s, loss=0.64]
Train Epoch 111 ==> 	accuracy: 0.8230, 	precision: 0.9995, 	recall: 0.6463, 	specificity: 0.9997, 	f1: 0.7850
Test Epoch 111: 100%|██████████| 1715/1715 [00:45<00:00, 37.84it/s, loss=0.187]
Test Epoch 111 ==> 	accuracy: 0.9277, 	precision: 0.9702, 	recall: 0.6601, 	specificity: 0.9949, 	f1: 0.7856
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 112: 100%|██████████| 6195/6195 [07:07<00:00, 14.49it/s, loss=0.446]
Train Epoch 112 ==> 	accuracy: 0.8207, 	precision: 0.9995, 	recall: 0.6416, 	specificity: 0.9997, 	f1: 0.7816
Test Epoch 112: 100%|██████████| 1715/1715 [00:44<00:00, 38.94it/s, loss=2.96]
Test Epoch 112 ==> 	accuracy: 0.9300, 	precision: 0.9594, 	recall: 0.6803, 	specificity: 0.9928, 	f1: 0.7961
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 113: 100%|██████████| 6195/6195 [07:05<00:00, 14.57it/s, loss=0.49]
Train Epoch 113 ==> 	accuracy: 0.8240, 	precision: 0.9995, 	recall: 0.6483, 	specificity: 0.9997, 	f1: 0.7865
Test Epoch 113: 100%|██████████| 1715/1715 [00:46<00:00, 36.59it/s, loss=0.994]
Test Epoch 113 ==> 	accuracy: 0.9314, 	precision: 0.9593, 	recall: 0.6875, 	specificity: 0.9927, 	f1: 0.8009
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 114: 100%|██████████| 6195/6195 [07:06<00:00, 14.53it/s, loss=0.429]
Train Epoch 114 ==> 	accuracy: 0.8218, 	precision: 0.9995, 	recall: 0.6439, 	specificity: 0.9997, 	f1: 0.7832
Test Epoch 114: 100%|██████████| 1715/1715 [00:44<00:00, 38.13it/s, loss=0.449]
Test Epoch 114 ==> 	accuracy: 0.9299, 	precision: 0.9640, 	recall: 0.6763, 	specificity: 0.9936, 	f1: 0.7949
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 115: 100%|██████████| 6195/6195 [07:10<00:00, 14.40it/s, loss=0.449]
Train Epoch 115 ==> 	accuracy: 0.8249, 	precision: 0.9995, 	recall: 0.6500, 	specificity: 0.9997, 	f1: 0.7877
Test Epoch 115: 100%|██████████| 1715/1715 [00:44<00:00, 38.83it/s, loss=0.245]
Test Epoch 115 ==> 	accuracy: 0.9300, 	precision: 0.9646, 	recall: 0.6763, 	specificity: 0.9938, 	f1: 0.7951
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 116: 100%|██████████| 6195/6195 [07:06<00:00, 14.52it/s, loss=0.476]
Train Epoch 116 ==> 	accuracy: 0.8230, 	precision: 0.9995, 	recall: 0.6464, 	specificity: 0.9997, 	f1: 0.7850
Test Epoch 116: 100%|██████████| 1715/1715 [00:43<00:00, 39.02it/s, loss=2.15]
Test Epoch 116 ==> 	accuracy: 0.9300, 	precision: 0.9651, 	recall: 0.6758, 	specificity: 0.9939, 	f1: 0.7950
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 117: 100%|██████████| 6195/6195 [07:09<00:00, 14.42it/s, loss=0.499]
Train Epoch 117 ==> 	accuracy: 0.8224, 	precision: 0.9995, 	recall: 0.6451, 	specificity: 0.9997, 	f1: 0.7841
Test Epoch 117: 100%|██████████| 1715/1715 [00:44<00:00, 38.14it/s, loss=0.403]
Test Epoch 117 ==> 	accuracy: 0.9299, 	precision: 0.9642, 	recall: 0.6759, 	specificity: 0.9937, 	f1: 0.7947
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 118: 100%|██████████| 6195/6195 [07:12<00:00, 14.33it/s, loss=0.457]
Train Epoch 118 ==> 	accuracy: 0.8238, 	precision: 0.9995, 	recall: 0.6480, 	specificity: 0.9997, 	f1: 0.7862
Test Epoch 118: 100%|██████████| 1715/1715 [00:45<00:00, 37.47it/s, loss=0.461]
Test Epoch 118 ==> 	accuracy: 0.9303, 	precision: 0.9637, 	recall: 0.6786, 	specificity: 0.9936, 	f1: 0.7964
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 119: 100%|██████████| 6195/6195 [07:15<00:00, 14.22it/s, loss=0.39]
Train Epoch 119 ==> 	accuracy: 0.8235, 	precision: 0.9995, 	recall: 0.6472, 	specificity: 0.9997, 	f1: 0.7857
Test Epoch 119: 100%|██████████| 1715/1715 [00:49<00:00, 34.85it/s, loss=1.41]
Test Epoch 119 ==> 	accuracy: 0.9300, 	precision: 0.9643, 	recall: 0.6762, 	specificity: 0.9937, 	f1: 0.7949
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 120: 100%|██████████| 6195/6195 [07:11<00:00, 14.36it/s, loss=0.467]
Train Epoch 120 ==> 	accuracy: 0.8241, 	precision: 0.9996, 	recall: 0.6486, 	specificity: 0.9997, 	f1: 0.7867
Test Epoch 120: 100%|██████████| 1715/1715 [00:42<00:00, 40.08it/s, loss=3.4]
Test Epoch 120 ==> 	accuracy: 0.9300, 	precision: 0.9640, 	recall: 0.6765, 	specificity: 0.9937, 	f1: 0.7951
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 121: 100%|██████████| 6195/6195 [07:04<00:00, 14.60it/s, loss=0.517]
Train Epoch 121 ==> 	accuracy: 0.8243, 	precision: 0.9995, 	recall: 0.6490, 	specificity: 0.9997, 	f1: 0.7870
Test Epoch 121: 100%|██████████| 1715/1715 [00:42<00:00, 40.37it/s, loss=0.377]
Test Epoch 121 ==> 	accuracy: 0.9314, 	precision: 0.9614, 	recall: 0.6857, 	specificity: 0.9931, 	f1: 0.8004
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 122: 100%|██████████| 6195/6195 [06:57<00:00, 14.85it/s, loss=0.881]
Train Epoch 122 ==> 	accuracy: 0.8284, 	precision: 0.9995, 	recall: 0.6571, 	specificity: 0.9997, 	f1: 0.7929
Test Epoch 122: 100%|██████████| 1715/1715 [00:53<00:00, 31.76it/s, loss=0.588]
Test Epoch 122 ==> 	accuracy: 0.9314, 	precision: 0.9563, 	recall: 0.6901, 	specificity: 0.9921, 	f1: 0.8016
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 123: 100%|██████████| 6195/6195 [07:18<00:00, 14.12it/s, loss=0.531]
Train Epoch 123 ==> 	accuracy: 0.8261, 	precision: 0.9995, 	recall: 0.6525, 	specificity: 0.9997, 	f1: 0.7896
Test Epoch 123: 100%|██████████| 1715/1715 [00:46<00:00, 37.02it/s, loss=0.221]
Test Epoch 123 ==> 	accuracy: 0.9303, 	precision: 0.9651, 	recall: 0.6774, 	specificity: 0.9939, 	f1: 0.7961
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 124: 100%|██████████| 6195/6195 [07:12<00:00, 14.33it/s, loss=0.448]
Train Epoch 124 ==> 	accuracy: 0.8260, 	precision: 0.9995, 	recall: 0.6523, 	specificity: 0.9997, 	f1: 0.7895
Test Epoch 124: 100%|██████████| 1715/1715 [00:48<00:00, 35.56it/s, loss=0.493]
Test Epoch 124 ==> 	accuracy: 0.9309, 	precision: 0.9619, 	recall: 0.6826, 	specificity: 0.9932, 	f1: 0.7985
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 125: 100%|██████████| 6195/6195 [07:12<00:00, 14.33it/s, loss=0.503]
Train Epoch 125 ==> 	accuracy: 0.8272, 	precision: 0.9996, 	recall: 0.6547, 	specificity: 0.9997, 	f1: 0.7911
Test Epoch 125: 100%|██████████| 1715/1715 [00:46<00:00, 37.25it/s, loss=0.936]
Test Epoch 125 ==> 	accuracy: 0.9314, 	precision: 0.9593, 	recall: 0.6872, 	specificity: 0.9927, 	f1: 0.8008
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 126: 100%|██████████| 6195/6195 [07:21<00:00, 14.04it/s, loss=0.495]
Train Epoch 126 ==> 	accuracy: 0.8295, 	precision: 0.9995, 	recall: 0.6594, 	specificity: 0.9997, 	f1: 0.7946
Test Epoch 126: 100%|██████████| 1715/1715 [00:47<00:00, 35.77it/s, loss=3.23]
Test Epoch 126 ==> 	accuracy: 0.9330, 	precision: 0.9541, 	recall: 0.6999, 	specificity: 0.9915, 	f1: 0.8075
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 127: 100%|██████████| 6195/6195 [07:17<00:00, 14.16it/s, loss=0.397]
Train Epoch 127 ==> 	accuracy: 0.8243, 	precision: 0.9995, 	recall: 0.6489, 	specificity: 0.9997, 	f1: 0.7870
Test Epoch 127: 100%|██████████| 1715/1715 [00:50<00:00, 33.81it/s, loss=0.176]
Test Epoch 127 ==> 	accuracy: 0.9303, 	precision: 0.9634, 	recall: 0.6785, 	specificity: 0.9935, 	f1: 0.7962
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 128: 100%|██████████| 6195/6195 [07:14<00:00, 14.24it/s, loss=0.413]
Train Epoch 128 ==> 	accuracy: 0.8243, 	precision: 0.9995, 	recall: 0.6489, 	specificity: 0.9997, 	f1: 0.7869
Test Epoch 128: 100%|██████████| 1715/1715 [00:46<00:00, 36.84it/s, loss=1.82]
Test Epoch 128 ==> 	accuracy: 0.9297, 	precision: 0.9664, 	recall: 0.6734, 	specificity: 0.9941, 	f1: 0.7937
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 129: 100%|██████████| 6195/6195 [07:06<00:00, 14.52it/s, loss=0.453]
Train Epoch 129 ==> 	accuracy: 0.8228, 	precision: 0.9995, 	recall: 0.6459, 	specificity: 0.9997, 	f1: 0.7847
Test Epoch 129: 100%|██████████| 1715/1715 [00:43<00:00, 39.43it/s, loss=2.42]
Test Epoch 129 ==> 	accuracy: 0.9307, 	precision: 0.9643, 	recall: 0.6799, 	specificity: 0.9937, 	f1: 0.7975
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 130: 100%|██████████| 6195/6195 [07:15<00:00, 14.21it/s, loss=0.488]
Train Epoch 130 ==> 	accuracy: 0.8238, 	precision: 0.9995, 	recall: 0.6480, 	specificity: 0.9997, 	f1: 0.7862
Test Epoch 130: 100%|██████████| 1715/1715 [00:46<00:00, 37.23it/s, loss=0.301]
Test Epoch 130 ==> 	accuracy: 0.9314, 	precision: 0.9624, 	recall: 0.6852, 	specificity: 0.9933, 	f1: 0.8005
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 131: 100%|██████████| 6195/6195 [07:11<00:00, 14.37it/s, loss=0.492]
Train Epoch 131 ==> 	accuracy: 0.8253, 	precision: 0.9995, 	recall: 0.6510, 	specificity: 0.9997, 	f1: 0.7884
Test Epoch 131: 100%|██████████| 1715/1715 [00:49<00:00, 34.44it/s, loss=0.232]
Test Epoch 131 ==> 	accuracy: 0.9306, 	precision: 0.9637, 	recall: 0.6800, 	specificity: 0.9936, 	f1: 0.7974
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 132: 100%|██████████| 6195/6195 [07:06<00:00, 14.53it/s, loss=0.511]
Train Epoch 132 ==> 	accuracy: 0.8261, 	precision: 0.9995, 	recall: 0.6524, 	specificity: 0.9997, 	f1: 0.7895
Test Epoch 132: 100%|██████████| 1715/1715 [00:49<00:00, 34.71it/s, loss=1.8]
Test Epoch 132 ==> 	accuracy: 0.9321, 	precision: 0.9604, 	recall: 0.6900, 	specificity: 0.9929, 	f1: 0.8030
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 133: 100%|██████████| 6195/6195 [07:19<00:00, 14.11it/s, loss=0.433]
Train Epoch 133 ==> 	accuracy: 0.8271, 	precision: 0.9995, 	recall: 0.6545, 	specificity: 0.9997, 	f1: 0.7910
Test Epoch 133: 100%|██████████| 1715/1715 [00:46<00:00, 36.60it/s, loss=2.44]
Test Epoch 133 ==> 	accuracy: 0.9316, 	precision: 0.9610, 	recall: 0.6873, 	specificity: 0.9930, 	f1: 0.8014
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 134: 100%|██████████| 6195/6195 [07:09<00:00, 14.41it/s, loss=0.493]
Train Epoch 134 ==> 	accuracy: 0.8271, 	precision: 0.9996, 	recall: 0.6544, 	specificity: 0.9997, 	f1: 0.7910
Test Epoch 134: 100%|██████████| 1715/1715 [00:45<00:00, 37.57it/s, loss=3.39]
Test Epoch 134 ==> 	accuracy: 0.9313, 	precision: 0.9620, 	recall: 0.6849, 	specificity: 0.9932, 	f1: 0.8002
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 135: 100%|██████████| 6195/6195 [07:02<00:00, 14.65it/s, loss=0.484]
Train Epoch 135 ==> 	accuracy: 0.8300, 	precision: 0.9996, 	recall: 0.6602, 	specificity: 0.9997, 	f1: 0.7952
Test Epoch 135: 100%|██████████| 1715/1715 [00:49<00:00, 34.73it/s, loss=2.87]
Test Epoch 135 ==> 	accuracy: 0.9313, 	precision: 0.9590, 	recall: 0.6870, 	specificity: 0.9926, 	f1: 0.8006
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 136: 100%|██████████| 6195/6195 [07:06<00:00, 14.54it/s, loss=0.491]
Train Epoch 136 ==> 	accuracy: 0.8274, 	precision: 0.9995, 	recall: 0.6552, 	specificity: 0.9997, 	f1: 0.7915
Test Epoch 136: 100%|██████████| 1715/1715 [00:45<00:00, 37.42it/s, loss=0.264]
Test Epoch 136 ==> 	accuracy: 0.9305, 	precision: 0.9643, 	recall: 0.6789, 	specificity: 0.9937, 	f1: 0.7968
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 137: 100%|██████████| 6195/6195 [07:07<00:00, 14.47it/s, loss=0.541]
Train Epoch 137 ==> 	accuracy: 0.8281, 	precision: 0.9995, 	recall: 0.6566, 	specificity: 0.9997, 	f1: 0.7926
Test Epoch 137: 100%|██████████| 1715/1715 [00:43<00:00, 39.08it/s, loss=0.482]
Test Epoch 137 ==> 	accuracy: 0.9314, 	precision: 0.9607, 	recall: 0.6865, 	specificity: 0.9929, 	f1: 0.8008
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 138: 100%|██████████| 6195/6195 [07:06<00:00, 14.52it/s, loss=0.552]
Train Epoch 138 ==> 	accuracy: 0.8284, 	precision: 0.9995, 	recall: 0.6571, 	specificity: 0.9997, 	f1: 0.7929
Test Epoch 138: 100%|██████████| 1715/1715 [00:44<00:00, 38.27it/s, loss=1.08]
Test Epoch 138 ==> 	accuracy: 0.9317, 	precision: 0.9598, 	recall: 0.6888, 	specificity: 0.9928, 	f1: 0.8020
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 139: 100%|██████████| 6195/6195 [07:08<00:00, 14.45it/s, loss=0.485]
Train Epoch 139 ==> 	accuracy: 0.8301, 	precision: 0.9995, 	recall: 0.6604, 	specificity: 0.9997, 	f1: 0.7953
Test Epoch 139: 100%|██████████| 1715/1715 [00:45<00:00, 37.81it/s, loss=0.434]
Test Epoch 139 ==> 	accuracy: 0.9317, 	precision: 0.9585, 	recall: 0.6897, 	specificity: 0.9925, 	f1: 0.8021
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 140: 100%|██████████| 6195/6195 [06:54<00:00, 14.95it/s, loss=0.494]
Train Epoch 140 ==> 	accuracy: 0.8273, 	precision: 0.9995, 	recall: 0.6550, 	specificity: 0.9997, 	f1: 0.7914
Test Epoch 140: 100%|██████████| 1715/1715 [00:49<00:00, 34.34it/s, loss=2.6]
Test Epoch 140 ==> 	accuracy: 0.9312, 	precision: 0.9610, 	recall: 0.6852, 	specificity: 0.9930, 	f1: 0.8000
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 141: 100%|██████████| 6195/6195 [07:04<00:00, 14.60it/s, loss=0.477]
Train Epoch 141 ==> 	accuracy: 0.8264, 	precision: 0.9995, 	recall: 0.6530, 	specificity: 0.9997, 	f1: 0.7900
Test Epoch 141: 100%|██████████| 1715/1715 [00:48<00:00, 35.72it/s, loss=0.425]
Test Epoch 141 ==> 	accuracy: 0.9315, 	precision: 0.9612, 	recall: 0.6867, 	specificity: 0.9930, 	f1: 0.8011
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 142: 100%|██████████| 6195/6195 [07:00<00:00, 14.73it/s, loss=0.454]
Train Epoch 142 ==> 	accuracy: 0.8284, 	precision: 0.9995, 	recall: 0.6570, 	specificity: 0.9997, 	f1: 0.7929
Test Epoch 142: 100%|██████████| 1715/1715 [00:46<00:00, 36.78it/s, loss=2.82]
Test Epoch 142 ==> 	accuracy: 0.9301, 	precision: 0.9654, 	recall: 0.6760, 	specificity: 0.9939, 	f1: 0.7952
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 143: 100%|██████████| 6195/6195 [07:05<00:00, 14.57it/s, loss=0.494]
Train Epoch 143 ==> 	accuracy: 0.8264, 	precision: 0.9996, 	recall: 0.6532, 	specificity: 0.9997, 	f1: 0.7901
Test Epoch 143: 100%|██████████| 1715/1715 [00:44<00:00, 38.94it/s, loss=1.37]
Test Epoch 143 ==> 	accuracy: 0.9313, 	precision: 0.9627, 	recall: 0.6843, 	specificity: 0.9933, 	f1: 0.8000
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 144: 100%|██████████| 6195/6195 [07:03<00:00, 14.63it/s, loss=0.538]
Train Epoch 144 ==> 	accuracy: 0.8266, 	precision: 0.9995, 	recall: 0.6535, 	specificity: 0.9997, 	f1: 0.7903
Test Epoch 144: 100%|██████████| 1715/1715 [00:46<00:00, 36.57it/s, loss=0.245]
Test Epoch 144 ==> 	accuracy: 0.9308, 	precision: 0.9643, 	recall: 0.6804, 	specificity: 0.9937, 	f1: 0.7978
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 145: 100%|██████████| 6195/6195 [06:59<00:00, 14.76it/s, loss=0.497]
Train Epoch 145 ==> 	accuracy: 0.8269, 	precision: 0.9996, 	recall: 0.6541, 	specificity: 0.9997, 	f1: 0.7908
Test Epoch 145: 100%|██████████| 1715/1715 [00:43<00:00, 39.42it/s, loss=0.872]
Test Epoch 145 ==> 	accuracy: 0.9315, 	precision: 0.9606, 	recall: 0.6870, 	specificity: 0.9929, 	f1: 0.8011
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 146: 100%|██████████| 6195/6195 [07:06<00:00, 14.51it/s, loss=0.549]
Train Epoch 146 ==> 	accuracy: 0.8276, 	precision: 0.9995, 	recall: 0.6554, 	specificity: 0.9997, 	f1: 0.7917
Test Epoch 146: 100%|██████████| 1715/1715 [00:48<00:00, 35.68it/s, loss=3.4]
Test Epoch 146 ==> 	accuracy: 0.9311, 	precision: 0.9626, 	recall: 0.6836, 	specificity: 0.9933, 	f1: 0.7995
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 147: 100%|██████████| 6195/6195 [07:02<00:00, 14.65it/s, loss=0.389]
Train Epoch 147 ==> 	accuracy: 0.8291, 	precision: 0.9995, 	recall: 0.6585, 	specificity: 0.9997, 	f1: 0.7940
Test Epoch 147: 100%|██████████| 1715/1715 [00:45<00:00, 37.51it/s, loss=0.585]
Test Epoch 147 ==> 	accuracy: 0.9322, 	precision: 0.9593, 	recall: 0.6918, 	specificity: 0.9926, 	f1: 0.8039
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 148: 100%|██████████| 6195/6195 [06:54<00:00, 14.94it/s, loss=0.449]
Train Epoch 148 ==> 	accuracy: 0.8293, 	precision: 0.9995, 	recall: 0.6589, 	specificity: 0.9997, 	f1: 0.7942
Test Epoch 148: 100%|██████████| 1715/1715 [00:43<00:00, 39.62it/s, loss=0.356]
Test Epoch 148 ==> 	accuracy: 0.9322, 	precision: 0.9597, 	recall: 0.6911, 	specificity: 0.9927, 	f1: 0.8035
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 149: 100%|██████████| 6195/6195 [06:52<00:00, 15.03it/s, loss=0.396]
Train Epoch 149 ==> 	accuracy: 0.8300, 	precision: 0.9996, 	recall: 0.6603, 	specificity: 0.9997, 	f1: 0.7952
Test Epoch 149: 100%|██████████| 1715/1715 [00:45<00:00, 38.04it/s, loss=0.406]
Test Epoch 149 ==> 	accuracy: 0.9318, 	precision: 0.9587, 	recall: 0.6902, 	specificity: 0.9925, 	f1: 0.8026
Adjusting learning rate of group 0 to 4.2391e-06.

Process finished with exit code 0

'''

'''
'../model_save_sigBlock4_focalWithMs_deformable_block6'

/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6195/6195 [08:05<00:00, 12.75it/s, loss=0.49]
Train Epoch 0 ==> 	accuracy: 0.6095, 	precision: 0.9948, 	recall: 0.2201, 	specificity: 0.9988, 	f1: 0.3605
Test Epoch 0: 100%|██████████| 1715/1715 [00:56<00:00, 30.20it/s, loss=0.549]
Test Epoch 0 ==> 	accuracy: 0.8965, 	precision: 0.9396, 	recall: 0.5177, 	specificity: 0.9916, 	f1: 0.6675
Train Epoch 1: 100%|██████████| 6195/6195 [09:55<00:00, 10.41it/s, loss=0.678]
Train Epoch 1 ==> 	accuracy: 0.7059, 	precision: 0.9963, 	recall: 0.4134, 	specificity: 0.9985, 	f1: 0.5843
Test Epoch 1: 100%|██████████| 1715/1715 [01:02<00:00, 27.47it/s, loss=0.435]
Test Epoch 1 ==> 	accuracy: 0.8943, 	precision: 0.9688, 	recall: 0.4892, 	specificity: 0.9960, 	f1: 0.6501
Train Epoch 2: 100%|██████████| 6195/6195 [10:14<00:00, 10.08it/s, loss=0.635]
Train Epoch 2 ==> 	accuracy: 0.7374, 	precision: 0.9971, 	recall: 0.4761, 	specificity: 0.9986, 	f1: 0.6445
Test Epoch 2: 100%|██████████| 1715/1715 [01:02<00:00, 27.26it/s, loss=0.301]
Test Epoch 2 ==> 	accuracy: 0.9104, 	precision: 0.9483, 	recall: 0.5855, 	specificity: 0.9920, 	f1: 0.7240
Train Epoch 3: 100%|██████████| 6195/6195 [10:31<00:00,  9.82it/s, loss=0.743]
Train Epoch 3 ==> 	accuracy: 0.7459, 	precision: 0.9973, 	recall: 0.4931, 	specificity: 0.9986, 	f1: 0.6599
Test Epoch 3: 100%|██████████| 1715/1715 [01:04<00:00, 26.42it/s, loss=0.502]
Test Epoch 3 ==> 	accuracy: 0.9085, 	precision: 0.9413, 	recall: 0.5805, 	specificity: 0.9909, 	f1: 0.7181
Train Epoch 4: 100%|██████████| 6195/6195 [10:43<00:00,  9.63it/s, loss=0.502]
Train Epoch 4 ==> 	accuracy: 0.7572, 	precision: 0.9976, 	recall: 0.5157, 	specificity: 0.9987, 	f1: 0.6799
Test Epoch 4: 100%|██████████| 1715/1715 [01:03<00:00, 26.87it/s, loss=0.672]
Test Epoch 4 ==> 	accuracy: 0.9123, 	precision: 0.9612, 	recall: 0.5868, 	specificity: 0.9941, 	f1: 0.7287
Train Epoch 5: 100%|██████████| 6195/6195 [10:52<00:00,  9.49it/s, loss=0.578]
Train Epoch 5 ==> 	accuracy: 0.7624, 	precision: 0.9978, 	recall: 0.5259, 	specificity: 0.9988, 	f1: 0.6888
Test Epoch 5: 100%|██████████| 1715/1715 [01:06<00:00, 25.98it/s, loss=0.452]
Test Epoch 5 ==> 	accuracy: 0.9088, 	precision: 0.9589, 	recall: 0.5704, 	specificity: 0.9939, 	f1: 0.7153
Train Epoch 6: 100%|██████████| 6195/6195 [10:51<00:00,  9.50it/s, loss=0.594]
Train Epoch 6 ==> 	accuracy: 0.7649, 	precision: 0.9980, 	recall: 0.5308, 	specificity: 0.9989, 	f1: 0.6930
Test Epoch 6: 100%|██████████| 1715/1715 [01:11<00:00, 24.02it/s, loss=0.764]
Test Epoch 6 ==> 	accuracy: 0.9101, 	precision: 0.9507, 	recall: 0.5825, 	specificity: 0.9924, 	f1: 0.7224
Train Epoch 7: 100%|██████████| 6195/6195 [10:58<00:00,  9.40it/s, loss=0.496]
Train Epoch 7 ==> 	accuracy: 0.7717, 	precision: 0.9979, 	recall: 0.5446, 	specificity: 0.9989, 	f1: 0.7046
Test Epoch 7: 100%|██████████| 1715/1715 [01:05<00:00, 26.00it/s, loss=0.333]
Test Epoch 7 ==> 	accuracy: 0.9160, 	precision: 0.9776, 	recall: 0.5952, 	specificity: 0.9966, 	f1: 0.7399
Train Epoch 8: 100%|██████████| 6195/6195 [10:53<00:00,  9.48it/s, loss=0.499]
Train Epoch 8 ==> 	accuracy: 0.7737, 	precision: 0.9981, 	recall: 0.5484, 	specificity: 0.9990, 	f1: 0.7078
Test Epoch 8: 100%|██████████| 1715/1715 [01:09<00:00, 24.61it/s, loss=0.242]
Test Epoch 8 ==> 	accuracy: 0.9145, 	precision: 0.9572, 	recall: 0.6009, 	specificity: 0.9933, 	f1: 0.7383
Train Epoch 9: 100%|██████████| 6195/6195 [10:59<00:00,  9.40it/s, loss=0.471]
Train Epoch 9 ==> 	accuracy: 0.7771, 	precision: 0.9982, 	recall: 0.5552, 	specificity: 0.9990, 	f1: 0.7135
Test Epoch 9: 100%|██████████| 1715/1715 [01:06<00:00, 25.89it/s, loss=1.18]
Test Epoch 9 ==> 	accuracy: 0.9156, 	precision: 0.9783, 	recall: 0.5926, 	specificity: 0.9967, 	f1: 0.7381
Train Epoch 10: 100%|██████████| 6195/6195 [10:57<00:00,  9.42it/s, loss=0.466]
Train Epoch 10 ==> 	accuracy: 0.7811, 	precision: 0.9983, 	recall: 0.5631, 	specificity: 0.9991, 	f1: 0.7200
Test Epoch 10: 100%|██████████| 1715/1715 [01:10<00:00, 24.35it/s, loss=0.731]
Test Epoch 10 ==> 	accuracy: 0.9223, 	precision: 0.9596, 	recall: 0.6398, 	specificity: 0.9932, 	f1: 0.7677
Train Epoch 11: 100%|██████████| 6195/6195 [10:48<00:00,  9.55it/s, loss=0.508]
Train Epoch 11 ==> 	accuracy: 0.7848, 	precision: 0.9984, 	recall: 0.5706, 	specificity: 0.9991, 	f1: 0.7262
Test Epoch 11: 100%|██████████| 1715/1715 [01:04<00:00, 26.79it/s, loss=0.408]
Test Epoch 11 ==> 	accuracy: 0.9197, 	precision: 0.9785, 	recall: 0.6133, 	specificity: 0.9966, 	f1: 0.7540
Train Epoch 12: 100%|██████████| 6195/6195 [11:02<00:00,  9.35it/s, loss=0.629]
Train Epoch 12 ==> 	accuracy: 0.7863, 	precision: 0.9985, 	recall: 0.5734, 	specificity: 0.9992, 	f1: 0.7285
Test Epoch 12: 100%|██████████| 1715/1715 [01:07<00:00, 25.26it/s, loss=0.428]
Test Epoch 12 ==> 	accuracy: 0.9201, 	precision: 0.9784, 	recall: 0.6154, 	specificity: 0.9966, 	f1: 0.7556
Train Epoch 13: 100%|██████████| 6195/6195 [10:37<00:00,  9.72it/s, loss=0.504]
Train Epoch 13 ==> 	accuracy: 0.7882, 	precision: 0.9985, 	recall: 0.5772, 	specificity: 0.9991, 	f1: 0.7315
Test Epoch 13: 100%|██████████| 1715/1715 [01:08<00:00, 24.95it/s, loss=0.279]
Test Epoch 13 ==> 	accuracy: 0.9197, 	precision: 0.9725, 	recall: 0.6173, 	specificity: 0.9956, 	f1: 0.7552
Train Epoch 14: 100%|██████████| 6195/6195 [10:42<00:00,  9.65it/s, loss=0.431]
Train Epoch 14 ==> 	accuracy: 0.7946, 	precision: 0.9985, 	recall: 0.5900, 	specificity: 0.9991, 	f1: 0.7417
Test Epoch 14: 100%|██████████| 1715/1715 [01:07<00:00, 25.42it/s, loss=0.647]
Test Epoch 14 ==> 	accuracy: 0.9238, 	precision: 0.9745, 	recall: 0.6369, 	specificity: 0.9958, 	f1: 0.7703
Train Epoch 15: 100%|██████████| 6195/6195 [10:44<00:00,  9.61it/s, loss=0.586]
Train Epoch 15 ==> 	accuracy: 0.7913, 	precision: 0.9986, 	recall: 0.5833, 	specificity: 0.9992, 	f1: 0.7365
Test Epoch 15: 100%|██████████| 1715/1715 [01:05<00:00, 26.10it/s, loss=0.368]
Test Epoch 15 ==> 	accuracy: 0.9213, 	precision: 0.9773, 	recall: 0.6227, 	specificity: 0.9964, 	f1: 0.7607
Train Epoch 16: 100%|██████████| 6195/6195 [10:34<00:00,  9.77it/s, loss=0.473]
Train Epoch 16 ==> 	accuracy: 0.7929, 	precision: 0.9986, 	recall: 0.5867, 	specificity: 0.9992, 	f1: 0.7391
Test Epoch 16: 100%|██████████| 1715/1715 [01:03<00:00, 26.93it/s, loss=0.579]
Test Epoch 16 ==> 	accuracy: 0.9247, 	precision: 0.9676, 	recall: 0.6467, 	specificity: 0.9946, 	f1: 0.7752
Train Epoch 17: 100%|██████████| 6195/6195 [10:37<00:00,  9.72it/s, loss=0.485]
Train Epoch 17 ==> 	accuracy: 0.7959, 	precision: 0.9987, 	recall: 0.5926, 	specificity: 0.9992, 	f1: 0.7438
Test Epoch 17: 100%|██████████| 1715/1715 [01:07<00:00, 25.42it/s, loss=0.425]
Test Epoch 17 ==> 	accuracy: 0.9235, 	precision: 0.9771, 	recall: 0.6340, 	specificity: 0.9963, 	f1: 0.7690
Train Epoch 18: 100%|██████████| 6195/6195 [10:47<00:00,  9.56it/s, loss=0.484]
Train Epoch 18 ==> 	accuracy: 0.7964, 	precision: 0.9988, 	recall: 0.5936, 	specificity: 0.9993, 	f1: 0.7447
Test Epoch 18: 100%|██████████| 1715/1715 [01:04<00:00, 26.55it/s, loss=0.613]
Test Epoch 18 ==> 	accuracy: 0.9210, 	precision: 0.9624, 	recall: 0.6312, 	specificity: 0.9938, 	f1: 0.7624
Train Epoch 19: 100%|██████████| 6195/6195 [10:26<00:00,  9.89it/s, loss=0.437]
Train Epoch 19 ==> 	accuracy: 0.7976, 	precision: 0.9988, 	recall: 0.5960, 	specificity: 0.9993, 	f1: 0.7465
Test Epoch 19: 100%|██████████| 1715/1715 [01:04<00:00, 26.63it/s, loss=0.734]
Test Epoch 19 ==> 	accuracy: 0.9251, 	precision: 0.9693, 	recall: 0.6474, 	specificity: 0.9948, 	f1: 0.7763
Train Epoch 20: 100%|██████████| 6195/6195 [10:38<00:00,  9.70it/s, loss=0.488]
Train Epoch 20 ==> 	accuracy: 0.7946, 	precision: 0.9989, 	recall: 0.5898, 	specificity: 0.9993, 	f1: 0.7417
Test Epoch 20: 100%|██████████| 1715/1715 [01:08<00:00, 24.87it/s, loss=0.423]
Test Epoch 20 ==> 	accuracy: 0.9220, 	precision: 0.9722, 	recall: 0.6297, 	specificity: 0.9955, 	f1: 0.7644
Train Epoch 21: 100%|██████████| 6195/6195 [11:18<00:00,  9.13it/s, loss=0.497]
Train Epoch 21 ==> 	accuracy: 0.8035, 	precision: 0.9989, 	recall: 0.6078, 	specificity: 0.9993, 	f1: 0.7557
Test Epoch 21: 100%|██████████| 1715/1715 [01:13<00:00, 23.42it/s, loss=0.734]
Test Epoch 21 ==> 	accuracy: 0.9253, 	precision: 0.9827, 	recall: 0.6393, 	specificity: 0.9972, 	f1: 0.7747
Train Epoch 22: 100%|██████████| 6195/6195 [11:30<00:00,  8.97it/s, loss=0.453]
Train Epoch 22 ==> 	accuracy: 0.8001, 	precision: 0.9989, 	recall: 0.6009, 	specificity: 0.9994, 	f1: 0.7504
Test Epoch 22: 100%|██████████| 1715/1715 [01:13<00:00, 23.21it/s, loss=0.554]
Test Epoch 22 ==> 	accuracy: 0.9193, 	precision: 0.9847, 	recall: 0.6073, 	specificity: 0.9976, 	f1: 0.7513
Train Epoch 23: 100%|██████████| 6195/6195 [11:28<00:00,  8.99it/s, loss=0.519]
Train Epoch 23 ==> 	accuracy: 0.8049, 	precision: 0.9990, 	recall: 0.6104, 	specificity: 0.9994, 	f1: 0.7578
Test Epoch 23: 100%|██████████| 1715/1715 [01:14<00:00, 22.94it/s, loss=0.537]
Test Epoch 23 ==> 	accuracy: 0.9269, 	precision: 0.9537, 	recall: 0.6686, 	specificity: 0.9918, 	f1: 0.7861
Train Epoch 24: 100%|██████████| 6195/6195 [11:22<00:00,  9.07it/s, loss=0.559]
Train Epoch 24 ==> 	accuracy: 0.8043, 	precision: 0.9989, 	recall: 0.6092, 	specificity: 0.9994, 	f1: 0.7569
Test Epoch 24: 100%|██████████| 1715/1715 [01:11<00:00, 23.89it/s, loss=0.286]
Test Epoch 24 ==> 	accuracy: 0.9263, 	precision: 0.9806, 	recall: 0.6459, 	specificity: 0.9968, 	f1: 0.7788
Train Epoch 25: 100%|██████████| 6195/6195 [11:22<00:00,  9.08it/s, loss=0.515]
Train Epoch 25 ==> 	accuracy: 0.8081, 	precision: 0.9990, 	recall: 0.6168, 	specificity: 0.9994, 	f1: 0.7627
Test Epoch 25: 100%|██████████| 1715/1715 [01:10<00:00, 24.40it/s, loss=0.207]
Test Epoch 25 ==> 	accuracy: 0.9268, 	precision: 0.9700, 	recall: 0.6559, 	specificity: 0.9949, 	f1: 0.7826
Train Epoch 26: 100%|██████████| 6195/6195 [11:25<00:00,  9.04it/s, loss=0.492]
Train Epoch 26 ==> 	accuracy: 0.8076, 	precision: 0.9990, 	recall: 0.6158, 	specificity: 0.9994, 	f1: 0.7619
Test Epoch 26: 100%|██████████| 1715/1715 [01:13<00:00, 23.25it/s, loss=0.214]
Test Epoch 26 ==> 	accuracy: 0.9273, 	precision: 0.9820, 	recall: 0.6500, 	specificity: 0.9970, 	f1: 0.7822
Train Epoch 27: 100%|██████████| 6195/6195 [11:22<00:00,  9.08it/s, loss=0.433]
Train Epoch 27 ==> 	accuracy: 0.8119, 	precision: 0.9990, 	recall: 0.6244, 	specificity: 0.9994, 	f1: 0.7685
Test Epoch 27: 100%|██████████| 1715/1715 [01:10<00:00, 24.30it/s, loss=0.869]
Test Epoch 27 ==> 	accuracy: 0.9263, 	precision: 0.9751, 	recall: 0.6496, 	specificity: 0.9958, 	f1: 0.7797
Train Epoch 28: 100%|██████████| 6195/6195 [11:31<00:00,  8.96it/s, loss=0.437]
Train Epoch 28 ==> 	accuracy: 0.8092, 	precision: 0.9991, 	recall: 0.6190, 	specificity: 0.9994, 	f1: 0.7644
Test Epoch 28: 100%|██████████| 1715/1715 [01:14<00:00, 22.93it/s, loss=0.323]
Test Epoch 28 ==> 	accuracy: 0.9272, 	precision: 0.9656, 	recall: 0.6609, 	specificity: 0.9941, 	f1: 0.7847
Train Epoch 29: 100%|██████████| 6195/6195 [11:26<00:00,  9.02it/s, loss=0.47]
Train Epoch 29 ==> 	accuracy: 0.8116, 	precision: 0.9991, 	recall: 0.6239, 	specificity: 0.9994, 	f1: 0.7681
Test Epoch 29: 100%|██████████| 1715/1715 [01:13<00:00, 23.46it/s, loss=0.45]
Test Epoch 29 ==> 	accuracy: 0.9286, 	precision: 0.9657, 	recall: 0.6679, 	specificity: 0.9940, 	f1: 0.7897
Train Epoch 30: 100%|██████████| 6195/6195 [11:23<00:00,  9.07it/s, loss=0.49]
Train Epoch 30 ==> 	accuracy: 0.8119, 	precision: 0.9990, 	recall: 0.6245, 	specificity: 0.9994, 	f1: 0.7686
Test Epoch 30: 100%|██████████| 1715/1715 [01:12<00:00, 23.69it/s, loss=0.641]
Test Epoch 30 ==> 	accuracy: 0.9278, 	precision: 0.9647, 	recall: 0.6647, 	specificity: 0.9939, 	f1: 0.7871
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 31: 100%|██████████| 6195/6195 [11:25<00:00,  9.04it/s, loss=0.398]
Train Epoch 31 ==> 	accuracy: 0.8140, 	precision: 0.9991, 	recall: 0.6285, 	specificity: 0.9995, 	f1: 0.7716
Test Epoch 31: 100%|██████████| 1715/1715 [01:17<00:00, 22.04it/s, loss=0.273]
Test Epoch 31 ==> 	accuracy: 0.9298, 	precision: 0.9742, 	recall: 0.6681, 	specificity: 0.9956, 	f1: 0.7926
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 32: 100%|██████████| 6195/6195 [11:18<00:00,  9.13it/s, loss=0.467]
Train Epoch 32 ==> 	accuracy: 0.8132, 	precision: 0.9991, 	recall: 0.6269, 	specificity: 0.9995, 	f1: 0.7704
Test Epoch 32: 100%|██████████| 1715/1715 [01:18<00:00, 21.78it/s, loss=0.17]
Test Epoch 32 ==> 	accuracy: 0.9330, 	precision: 0.9755, 	recall: 0.6832, 	specificity: 0.9957, 	f1: 0.8036
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 33: 100%|██████████| 6195/6195 [11:49<00:00,  8.73it/s, loss=0.491]
Train Epoch 33 ==> 	accuracy: 0.8141, 	precision: 0.9991, 	recall: 0.6288, 	specificity: 0.9994, 	f1: 0.7718
Test Epoch 33: 100%|██████████| 1715/1715 [01:18<00:00, 21.90it/s, loss=0.455]
Test Epoch 33 ==> 	accuracy: 0.9283, 	precision: 0.9845, 	recall: 0.6532, 	specificity: 0.9974, 	f1: 0.7853
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 34: 100%|██████████| 6195/6195 [11:16<00:00,  9.16it/s, loss=0.551]
Train Epoch 34 ==> 	accuracy: 0.8164, 	precision: 0.9992, 	recall: 0.6334, 	specificity: 0.9995, 	f1: 0.7753
Test Epoch 34: 100%|██████████| 1715/1715 [01:18<00:00, 21.82it/s, loss=0.429]
Test Epoch 34 ==> 	accuracy: 0.9297, 	precision: 0.9787, 	recall: 0.6645, 	specificity: 0.9964, 	f1: 0.7915
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 35: 100%|██████████| 6195/6195 [11:16<00:00,  9.16it/s, loss=0.498]
Train Epoch 35 ==> 	accuracy: 0.8184, 	precision: 0.9992, 	recall: 0.6373, 	specificity: 0.9995, 	f1: 0.7783
Test Epoch 35: 100%|██████████| 1715/1715 [01:08<00:00, 24.86it/s, loss=0.311]
Test Epoch 35 ==> 	accuracy: 0.9305, 	precision: 0.9801, 	recall: 0.6675, 	specificity: 0.9966, 	f1: 0.7941
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 36: 100%|██████████| 6195/6195 [11:09<00:00,  9.25it/s, loss=0.409]
Train Epoch 36 ==> 	accuracy: 0.8211, 	precision: 0.9992, 	recall: 0.6428, 	specificity: 0.9995, 	f1: 0.7823
Test Epoch 36: 100%|██████████| 1715/1715 [01:13<00:00, 23.21it/s, loss=1.73]
Test Epoch 36 ==> 	accuracy: 0.9313, 	precision: 0.9540, 	recall: 0.6911, 	specificity: 0.9916, 	f1: 0.8016
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 37: 100%|██████████| 6195/6195 [11:18<00:00,  9.14it/s, loss=0.448]
Train Epoch 37 ==> 	accuracy: 0.8187, 	precision: 0.9992, 	recall: 0.6379, 	specificity: 0.9995, 	f1: 0.7787
Test Epoch 37: 100%|██████████| 1715/1715 [01:13<00:00, 23.35it/s, loss=0.547]
Test Epoch 37 ==> 	accuracy: 0.9290, 	precision: 0.9780, 	recall: 0.6613, 	specificity: 0.9963, 	f1: 0.7890
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 38: 100%|██████████| 6195/6195 [11:24<00:00,  9.05it/s, loss=0.467]
Train Epoch 38 ==> 	accuracy: 0.8245, 	precision: 0.9993, 	recall: 0.6494, 	specificity: 0.9995, 	f1: 0.7872
Test Epoch 38: 100%|██████████| 1715/1715 [01:11<00:00, 23.85it/s, loss=0.275]
Test Epoch 38 ==> 	accuracy: 0.9358, 	precision: 0.9690, 	recall: 0.7028, 	specificity: 0.9943, 	f1: 0.8147
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 39: 100%|██████████| 6195/6195 [11:15<00:00,  9.17it/s, loss=0.426]
Train Epoch 39 ==> 	accuracy: 0.8234, 	precision: 0.9993, 	recall: 0.6472, 	specificity: 0.9995, 	f1: 0.7856
Test Epoch 39: 100%|██████████| 1715/1715 [01:11<00:00, 24.10it/s, loss=0.238]
Test Epoch 39 ==> 	accuracy: 0.9299, 	precision: 0.9623, 	recall: 0.6776, 	specificity: 0.9933, 	f1: 0.7952
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 40: 100%|██████████| 6195/6195 [11:23<00:00,  9.06it/s, loss=0.39]
Train Epoch 40 ==> 	accuracy: 0.8285, 	precision: 0.9993, 	recall: 0.6575, 	specificity: 0.9996, 	f1: 0.7931
Test Epoch 40: 100%|██████████| 1715/1715 [01:11<00:00, 24.10it/s, loss=0.365]
Test Epoch 40 ==> 	accuracy: 0.9346, 	precision: 0.9770, 	recall: 0.6905, 	specificity: 0.9959, 	f1: 0.8091
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 41: 100%|██████████| 6195/6195 [11:14<00:00,  9.18it/s, loss=0.452]
Train Epoch 41 ==> 	accuracy: 0.8267, 	precision: 0.9993, 	recall: 0.6538, 	specificity: 0.9996, 	f1: 0.7904
Test Epoch 41: 100%|██████████| 1715/1715 [01:15<00:00, 22.75it/s, loss=0.661]
Test Epoch 41 ==> 	accuracy: 0.9303, 	precision: 0.9732, 	recall: 0.6714, 	specificity: 0.9954, 	f1: 0.7946
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 42: 100%|██████████| 6195/6195 [11:14<00:00,  9.18it/s, loss=0.52]
Train Epoch 42 ==> 	accuracy: 0.8315, 	precision: 0.9993, 	recall: 0.6634, 	specificity: 0.9995, 	f1: 0.7974
Test Epoch 42: 100%|██████████| 1715/1715 [01:12<00:00, 23.63it/s, loss=0.315]
Test Epoch 42 ==> 	accuracy: 0.9362, 	precision: 0.9735, 	recall: 0.7015, 	specificity: 0.9952, 	f1: 0.8154
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 43: 100%|██████████| 6195/6195 [11:21<00:00,  9.09it/s, loss=0.396]
Train Epoch 43 ==> 	accuracy: 0.8303, 	precision: 0.9994, 	recall: 0.6611, 	specificity: 0.9996, 	f1: 0.7957
Test Epoch 43: 100%|██████████| 1715/1715 [01:11<00:00, 23.97it/s, loss=0.224]
Test Epoch 43 ==> 	accuracy: 0.9360, 	precision: 0.9757, 	recall: 0.6985, 	specificity: 0.9956, 	f1: 0.8142
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 44: 100%|██████████| 6195/6195 [11:16<00:00,  9.16it/s, loss=0.494]
Train Epoch 44 ==> 	accuracy: 0.8296, 	precision: 0.9993, 	recall: 0.6597, 	specificity: 0.9996, 	f1: 0.7947
Test Epoch 44: 100%|██████████| 1715/1715 [01:15<00:00, 22.78it/s, loss=0.432]
Test Epoch 44 ==> 	accuracy: 0.9341, 	precision: 0.9794, 	recall: 0.6861, 	specificity: 0.9964, 	f1: 0.8069
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 45: 100%|██████████| 6195/6195 [11:31<00:00,  8.97it/s, loss=0.393]
Train Epoch 45 ==> 	accuracy: 0.8324, 	precision: 0.9994, 	recall: 0.6652, 	specificity: 0.9996, 	f1: 0.7987
Test Epoch 45: 100%|██████████| 1715/1715 [01:15<00:00, 22.69it/s, loss=0.662]
Test Epoch 45 ==> 	accuracy: 0.9346, 	precision: 0.9770, 	recall: 0.6904, 	specificity: 0.9959, 	f1: 0.8091
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 46: 100%|██████████| 6195/6195 [11:03<00:00,  9.33it/s, loss=0.486]
Train Epoch 46 ==> 	accuracy: 0.8351, 	precision: 0.9994, 	recall: 0.6707, 	specificity: 0.9996, 	f1: 0.8027
Test Epoch 46: 100%|██████████| 1715/1715 [01:16<00:00, 22.47it/s, loss=0.35]
Test Epoch 46 ==> 	accuracy: 0.9348, 	precision: 0.9619, 	recall: 0.7032, 	specificity: 0.9930, 	f1: 0.8125
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 47: 100%|██████████| 6195/6195 [11:27<00:00,  9.01it/s, loss=0.377]
Train Epoch 47 ==> 	accuracy: 0.8351, 	precision: 0.9994, 	recall: 0.6707, 	specificity: 0.9996, 	f1: 0.8027
Test Epoch 47: 100%|██████████| 1715/1715 [01:15<00:00, 22.77it/s, loss=0.222]
Test Epoch 47 ==> 	accuracy: 0.9376, 	precision: 0.9638, 	recall: 0.7160, 	specificity: 0.9932, 	f1: 0.8216
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 48: 100%|██████████| 6195/6195 [11:19<00:00,  9.11it/s, loss=0.444]
Train Epoch 48 ==> 	accuracy: 0.8333, 	precision: 0.9994, 	recall: 0.6670, 	specificity: 0.9996, 	f1: 0.8000
Test Epoch 48: 100%|██████████| 1715/1715 [01:17<00:00, 22.27it/s, loss=0.409]
Test Epoch 48 ==> 	accuracy: 0.9366, 	precision: 0.9714, 	recall: 0.7048, 	specificity: 0.9948, 	f1: 0.8169
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 49: 100%|██████████| 6195/6195 [11:05<00:00,  9.31it/s, loss=0.506]
Train Epoch 49 ==> 	accuracy: 0.8375, 	precision: 0.9994, 	recall: 0.6755, 	specificity: 0.9996, 	f1: 0.8061
Test Epoch 49: 100%|██████████| 1715/1715 [01:13<00:00, 23.20it/s, loss=0.516]
Test Epoch 49 ==> 	accuracy: 0.9313, 	precision: 0.9716, 	recall: 0.6779, 	specificity: 0.9950, 	f1: 0.7986
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 50: 100%|██████████| 6195/6195 [11:12<00:00,  9.21it/s, loss=0.447]
Train Epoch 50 ==> 	accuracy: 0.8390, 	precision: 0.9994, 	recall: 0.6784, 	specificity: 0.9996, 	f1: 0.8082
Test Epoch 50: 100%|██████████| 1715/1715 [01:15<00:00, 22.79it/s, loss=0.742]
Test Epoch 50 ==> 	accuracy: 0.9403, 	precision: 0.9796, 	recall: 0.7174, 	specificity: 0.9962, 	f1: 0.8282
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 51: 100%|██████████| 6195/6195 [11:23<00:00,  9.07it/s, loss=0.4]
Train Epoch 51 ==> 	accuracy: 0.8405, 	precision: 0.9994, 	recall: 0.6814, 	specificity: 0.9996, 	f1: 0.8103
Test Epoch 51: 100%|██████████| 1715/1715 [01:14<00:00, 23.00it/s, loss=0.511]
Test Epoch 51 ==> 	accuracy: 0.9407, 	precision: 0.9836, 	recall: 0.7164, 	specificity: 0.9970, 	f1: 0.8290
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 52: 100%|██████████| 6195/6195 [11:13<00:00,  9.20it/s, loss=0.406]
Train Epoch 52 ==> 	accuracy: 0.8403, 	precision: 0.9995, 	recall: 0.6810, 	specificity: 0.9996, 	f1: 0.8101
Test Epoch 52: 100%|██████████| 1715/1715 [01:17<00:00, 22.17it/s, loss=0.346]
Test Epoch 52 ==> 	accuracy: 0.9391, 	precision: 0.9850, 	recall: 0.7073, 	specificity: 0.9973, 	f1: 0.8234
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 53: 100%|██████████| 6195/6195 [11:12<00:00,  9.22it/s, loss=0.475]
Train Epoch 53 ==> 	accuracy: 0.8400, 	precision: 0.9995, 	recall: 0.6804, 	specificity: 0.9996, 	f1: 0.8096
Test Epoch 53: 100%|██████████| 1715/1715 [01:11<00:00, 24.13it/s, loss=0.565]
Test Epoch 53 ==> 	accuracy: 0.9393, 	precision: 0.9801, 	recall: 0.7119, 	specificity: 0.9964, 	f1: 0.8247
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 54: 100%|██████████| 6195/6195 [11:21<00:00,  9.09it/s, loss=0.39]
Train Epoch 54 ==> 	accuracy: 0.8435, 	precision: 0.9995, 	recall: 0.6873, 	specificity: 0.9996, 	f1: 0.8145
Test Epoch 54: 100%|██████████| 1715/1715 [01:16<00:00, 22.30it/s, loss=0.406]
Test Epoch 54 ==> 	accuracy: 0.9420, 	precision: 0.9781, 	recall: 0.7274, 	specificity: 0.9959, 	f1: 0.8344
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 55: 100%|██████████| 6195/6195 [11:10<00:00,  9.24it/s, loss=0.502]
Train Epoch 55 ==> 	accuracy: 0.8432, 	precision: 0.9995, 	recall: 0.6867, 	specificity: 0.9996, 	f1: 0.8141
Test Epoch 55: 100%|██████████| 1715/1715 [01:19<00:00, 21.66it/s, loss=0.24]
Test Epoch 55 ==> 	accuracy: 0.9421, 	precision: 0.9785, 	recall: 0.7275, 	specificity: 0.9960, 	f1: 0.8345
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 56: 100%|██████████| 6195/6195 [11:22<00:00,  9.08it/s, loss=0.408]
Train Epoch 56 ==> 	accuracy: 0.8458, 	precision: 0.9995, 	recall: 0.6919, 	specificity: 0.9996, 	f1: 0.8178
Test Epoch 56: 100%|██████████| 1715/1715 [01:16<00:00, 22.49it/s, loss=0.599]
Test Epoch 56 ==> 	accuracy: 0.9405, 	precision: 0.9731, 	recall: 0.7236, 	specificity: 0.9950, 	f1: 0.8300
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 57: 100%|██████████| 6195/6195 [11:09<00:00,  9.26it/s, loss=0.396]
Train Epoch 57 ==> 	accuracy: 0.8443, 	precision: 0.9995, 	recall: 0.6890, 	specificity: 0.9997, 	f1: 0.8157
Test Epoch 57: 100%|██████████| 1715/1715 [01:15<00:00, 22.63it/s, loss=0.159]
Test Epoch 57 ==> 	accuracy: 0.9411, 	precision: 0.9802, 	recall: 0.7214, 	specificity: 0.9963, 	f1: 0.8311
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 58: 100%|██████████| 6195/6195 [11:05<00:00,  9.31it/s, loss=0.371]
Train Epoch 58 ==> 	accuracy: 0.8458, 	precision: 0.9995, 	recall: 0.6919, 	specificity: 0.9997, 	f1: 0.8177
Test Epoch 58: 100%|██████████| 1715/1715 [01:17<00:00, 22.10it/s, loss=0.284]
Test Epoch 58 ==> 	accuracy: 0.9402, 	precision: 0.9785, 	recall: 0.7180, 	specificity: 0.9960, 	f1: 0.8282
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 59: 100%|██████████| 6195/6195 [11:19<00:00,  9.11it/s, loss=0.46]
Train Epoch 59 ==> 	accuracy: 0.8471, 	precision: 0.9995, 	recall: 0.6945, 	specificity: 0.9997, 	f1: 0.8196
Test Epoch 59: 100%|██████████| 1715/1715 [01:14<00:00, 22.94it/s, loss=0.235]
Test Epoch 59 ==> 	accuracy: 0.9418, 	precision: 0.9793, 	recall: 0.7255, 	specificity: 0.9961, 	f1: 0.8335
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 60: 100%|██████████| 6195/6195 [11:14<00:00,  9.19it/s, loss=0.419]
Train Epoch 60 ==> 	accuracy: 0.8498, 	precision: 0.9996, 	recall: 0.6999, 	specificity: 0.9997, 	f1: 0.8233
Test Epoch 60: 100%|██████████| 1715/1715 [01:10<00:00, 24.18it/s, loss=0.169]
Test Epoch 60 ==> 	accuracy: 0.9407, 	precision: 0.9818, 	recall: 0.7181, 	specificity: 0.9967, 	f1: 0.8295
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 61: 100%|██████████| 6195/6195 [11:12<00:00,  9.22it/s, loss=0.503]
Train Epoch 61 ==> 	accuracy: 0.8492, 	precision: 0.9995, 	recall: 0.6986, 	specificity: 0.9997, 	f1: 0.8224
Test Epoch 61: 100%|██████████| 1715/1715 [01:10<00:00, 24.25it/s, loss=0.301]
Test Epoch 61 ==> 	accuracy: 0.9426, 	precision: 0.9818, 	recall: 0.7275, 	specificity: 0.9966, 	f1: 0.8357
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 62: 100%|██████████| 6195/6195 [11:28<00:00,  9.00it/s, loss=0.408]
Train Epoch 62 ==> 	accuracy: 0.8506, 	precision: 0.9995, 	recall: 0.7016, 	specificity: 0.9997, 	f1: 0.8245
Test Epoch 62: 100%|██████████| 1715/1715 [01:12<00:00, 23.69it/s, loss=0.253]
Test Epoch 62 ==> 	accuracy: 0.9406, 	precision: 0.9792, 	recall: 0.7194, 	specificity: 0.9962, 	f1: 0.8294
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 63: 100%|██████████| 6195/6195 [11:17<00:00,  9.15it/s, loss=0.42]
Train Epoch 63 ==> 	accuracy: 0.8532, 	precision: 0.9996, 	recall: 0.7066, 	specificity: 0.9997, 	f1: 0.8279
Test Epoch 63: 100%|██████████| 1715/1715 [01:12<00:00, 23.56it/s, loss=0.615]
Test Epoch 63 ==> 	accuracy: 0.9420, 	precision: 0.9753, 	recall: 0.7298, 	specificity: 0.9954, 	f1: 0.8349
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 64: 100%|██████████| 6195/6195 [11:39<00:00,  8.85it/s, loss=0.378]
Train Epoch 64 ==> 	accuracy: 0.8523, 	precision: 0.9996, 	recall: 0.7050, 	specificity: 0.9997, 	f1: 0.8268
Test Epoch 64: 100%|██████████| 1715/1715 [01:12<00:00, 23.61it/s, loss=0.676]
Test Epoch 64 ==> 	accuracy: 0.9416, 	precision: 0.9723, 	recall: 0.7298, 	specificity: 0.9948, 	f1: 0.8338
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 65: 100%|██████████| 6195/6195 [11:15<00:00,  9.17it/s, loss=0.396]
Train Epoch 65 ==> 	accuracy: 0.8525, 	precision: 0.9996, 	recall: 0.7053, 	specificity: 0.9997, 	f1: 0.8270
Test Epoch 65: 100%|██████████| 1715/1715 [01:12<00:00, 23.54it/s, loss=0.143]
Test Epoch 65 ==> 	accuracy: 0.9442, 	precision: 0.9754, 	recall: 0.7405, 	specificity: 0.9953, 	f1: 0.8419
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 66: 100%|██████████| 6195/6195 [11:19<00:00,  9.12it/s, loss=0.435]
Train Epoch 66 ==> 	accuracy: 0.8528, 	precision: 0.9996, 	recall: 0.7058, 	specificity: 0.9997, 	f1: 0.8274
Test Epoch 66: 100%|██████████| 1715/1715 [01:16<00:00, 22.48it/s, loss=0.326]
Test Epoch 66 ==> 	accuracy: 0.9439, 	precision: 0.9758, 	recall: 0.7387, 	specificity: 0.9954, 	f1: 0.8409
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 67: 100%|██████████| 6195/6195 [11:30<00:00,  8.97it/s, loss=0.466]
Train Epoch 67 ==> 	accuracy: 0.8568, 	precision: 0.9996, 	recall: 0.7140, 	specificity: 0.9997, 	f1: 0.8330
Test Epoch 67: 100%|██████████| 1715/1715 [01:20<00:00, 21.41it/s, loss=0.609]
Test Epoch 67 ==> 	accuracy: 0.9433, 	precision: 0.9773, 	recall: 0.7348, 	specificity: 0.9957, 	f1: 0.8389
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 68: 100%|██████████| 6195/6195 [11:13<00:00,  9.20it/s, loss=0.412]
Train Epoch 68 ==> 	accuracy: 0.8538, 	precision: 0.9996, 	recall: 0.7078, 	specificity: 0.9997, 	f1: 0.8288
Test Epoch 68: 100%|██████████| 1715/1715 [01:18<00:00, 21.87it/s, loss=3.08]
Test Epoch 68 ==> 	accuracy: 0.9442, 	precision: 0.9771, 	recall: 0.7395, 	specificity: 0.9957, 	f1: 0.8419
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 69: 100%|██████████| 6195/6195 [11:20<00:00,  9.10it/s, loss=0.462]
Train Epoch 69 ==> 	accuracy: 0.8576, 	precision: 0.9996, 	recall: 0.7154, 	specificity: 0.9997, 	f1: 0.8340
Test Epoch 69: 100%|██████████| 1715/1715 [01:16<00:00, 22.49it/s, loss=0.286]
Test Epoch 69 ==> 	accuracy: 0.9441, 	precision: 0.9661, 	recall: 0.7478, 	specificity: 0.9934, 	f1: 0.8430
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 70: 100%|██████████| 6195/6195 [11:20<00:00,  9.11it/s, loss=0.335]
Train Epoch 70 ==> 	accuracy: 0.8556, 	precision: 0.9996, 	recall: 0.7115, 	specificity: 0.9997, 	f1: 0.8313
Test Epoch 70: 100%|██████████| 1715/1715 [01:05<00:00, 25.99it/s, loss=0.249]
Test Epoch 70 ==> 	accuracy: 0.9438, 	precision: 0.9801, 	recall: 0.7351, 	specificity: 0.9963, 	f1: 0.8401
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 71: 100%|██████████| 6195/6195 [11:37<00:00,  8.88it/s, loss=0.529]
Train Epoch 71 ==> 	accuracy: 0.8573, 	precision: 0.9996, 	recall: 0.7149, 	specificity: 0.9997, 	f1: 0.8336
Test Epoch 71: 100%|██████████| 1715/1715 [01:14<00:00, 23.00it/s, loss=0.57]
Test Epoch 71 ==> 	accuracy: 0.9449, 	precision: 0.9765, 	recall: 0.7433, 	specificity: 0.9955, 	f1: 0.8441
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 72: 100%|██████████| 6195/6195 [11:08<00:00,  9.27it/s, loss=0.468]
Train Epoch 72 ==> 	accuracy: 0.8565, 	precision: 0.9996, 	recall: 0.7133, 	specificity: 0.9997, 	f1: 0.8325
Test Epoch 72: 100%|██████████| 1715/1715 [01:10<00:00, 24.49it/s, loss=1.19]
Test Epoch 72 ==> 	accuracy: 0.9441, 	precision: 0.9784, 	recall: 0.7381, 	specificity: 0.9959, 	f1: 0.8414
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 73: 100%|██████████| 6195/6195 [11:15<00:00,  9.17it/s, loss=0.372]
Train Epoch 73 ==> 	accuracy: 0.8599, 	precision: 0.9996, 	recall: 0.7201, 	specificity: 0.9997, 	f1: 0.8371
Test Epoch 73: 100%|██████████| 1715/1715 [01:15<00:00, 22.75it/s, loss=0.813]
Test Epoch 73 ==> 	accuracy: 0.9450, 	precision: 0.9749, 	recall: 0.7455, 	specificity: 0.9952, 	f1: 0.8449
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 74: 100%|██████████| 6195/6195 [11:17<00:00,  9.14it/s, loss=0.429]
Train Epoch 74 ==> 	accuracy: 0.8578, 	precision: 0.9996, 	recall: 0.7159, 	specificity: 0.9997, 	f1: 0.8343
Test Epoch 74: 100%|██████████| 1715/1715 [01:21<00:00, 21.07it/s, loss=0.262]
Test Epoch 74 ==> 	accuracy: 0.9444, 	precision: 0.9738, 	recall: 0.7429, 	specificity: 0.9950, 	f1: 0.8428
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 75: 100%|██████████| 6195/6195 [11:16<00:00,  9.16it/s, loss=0.395]
Train Epoch 75 ==> 	accuracy: 0.8612, 	precision: 0.9996, 	recall: 0.7227, 	specificity: 0.9997, 	f1: 0.8389
Test Epoch 75: 100%|██████████| 1715/1715 [01:15<00:00, 22.67it/s, loss=0.971]
Test Epoch 75 ==> 	accuracy: 0.9456, 	precision: 0.9707, 	recall: 0.7517, 	specificity: 0.9943, 	f1: 0.8473
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 76: 100%|██████████| 6195/6195 [11:19<00:00,  9.12it/s, loss=0.405]
Train Epoch 76 ==> 	accuracy: 0.8625, 	precision: 0.9996, 	recall: 0.7253, 	specificity: 0.9997, 	f1: 0.8406
Test Epoch 76: 100%|██████████| 1715/1715 [01:14<00:00, 22.97it/s, loss=0.65]
Test Epoch 76 ==> 	accuracy: 0.9432, 	precision: 0.9753, 	recall: 0.7360, 	specificity: 0.9953, 	f1: 0.8389
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 77: 100%|██████████| 6195/6195 [11:22<00:00,  9.07it/s, loss=0.425]
Train Epoch 77 ==> 	accuracy: 0.8618, 	precision: 0.9996, 	recall: 0.7239, 	specificity: 0.9997, 	f1: 0.8397
Test Epoch 77: 100%|██████████| 1715/1715 [01:15<00:00, 22.83it/s, loss=0.429]
Test Epoch 77 ==> 	accuracy: 0.9454, 	precision: 0.9763, 	recall: 0.7462, 	specificity: 0.9955, 	f1: 0.8459
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 78: 100%|██████████| 6195/6195 [11:21<00:00,  9.09it/s, loss=0.447]
Train Epoch 78 ==> 	accuracy: 0.8616, 	precision: 0.9996, 	recall: 0.7235, 	specificity: 0.9997, 	f1: 0.8394
Test Epoch 78: 100%|██████████| 1715/1715 [01:14<00:00, 23.07it/s, loss=0.346]
Test Epoch 78 ==> 	accuracy: 0.9447, 	precision: 0.9753, 	recall: 0.7433, 	specificity: 0.9953, 	f1: 0.8436
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 79: 100%|██████████| 6195/6195 [11:26<00:00,  9.02it/s, loss=0.361]
Train Epoch 79 ==> 	accuracy: 0.8634, 	precision: 0.9996, 	recall: 0.7271, 	specificity: 0.9997, 	f1: 0.8418
Test Epoch 79: 100%|██████████| 1715/1715 [01:12<00:00, 23.60it/s, loss=0.407]
Test Epoch 79 ==> 	accuracy: 0.9446, 	precision: 0.9736, 	recall: 0.7445, 	specificity: 0.9949, 	f1: 0.8438
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 80: 100%|██████████| 6195/6195 [11:31<00:00,  8.95it/s, loss=0.391]
Train Epoch 80 ==> 	accuracy: 0.8633, 	precision: 0.9996, 	recall: 0.7268, 	specificity: 0.9997, 	f1: 0.8417
Test Epoch 80: 100%|██████████| 1715/1715 [01:10<00:00, 24.43it/s, loss=2.53]
Test Epoch 80 ==> 	accuracy: 0.9458, 	precision: 0.9719, 	recall: 0.7518, 	specificity: 0.9946, 	f1: 0.8478
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 81: 100%|██████████| 6195/6195 [11:09<00:00,  9.25it/s, loss=0.487]
Train Epoch 81 ==> 	accuracy: 0.8630, 	precision: 0.9996, 	recall: 0.7263, 	specificity: 0.9997, 	f1: 0.8413
Test Epoch 81: 100%|██████████| 1715/1715 [01:16<00:00, 22.45it/s, loss=0.471]
Test Epoch 81 ==> 	accuracy: 0.9464, 	precision: 0.9767, 	recall: 0.7507, 	specificity: 0.9955, 	f1: 0.8489
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 82: 100%|██████████| 6195/6195 [11:20<00:00,  9.11it/s, loss=0.361]
Train Epoch 82 ==> 	accuracy: 0.8643, 	precision: 0.9996, 	recall: 0.7289, 	specificity: 0.9997, 	f1: 0.8431
Test Epoch 82: 100%|██████████| 1715/1715 [01:14<00:00, 22.97it/s, loss=0.797]
Test Epoch 82 ==> 	accuracy: 0.9462, 	precision: 0.9707, 	recall: 0.7546, 	specificity: 0.9943, 	f1: 0.8491
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 83: 100%|██████████| 6195/6195 [11:16<00:00,  9.15it/s, loss=0.477]
Train Epoch 83 ==> 	accuracy: 0.8649, 	precision: 0.9996, 	recall: 0.7300, 	specificity: 0.9997, 	f1: 0.8438
Test Epoch 83: 100%|██████████| 1715/1715 [01:09<00:00, 24.51it/s, loss=2.82]
Test Epoch 83 ==> 	accuracy: 0.9474, 	precision: 0.9717, 	recall: 0.7601, 	specificity: 0.9944, 	f1: 0.8530
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 84: 100%|██████████| 6195/6195 [11:11<00:00,  9.22it/s, loss=0.363]
Train Epoch 84 ==> 	accuracy: 0.8668, 	precision: 0.9997, 	recall: 0.7338, 	specificity: 0.9997, 	f1: 0.8463
Test Epoch 84: 100%|██████████| 1715/1715 [01:14<00:00, 23.01it/s, loss=1.46]
Test Epoch 84 ==> 	accuracy: 0.9470, 	precision: 0.9724, 	recall: 0.7576, 	specificity: 0.9946, 	f1: 0.8517
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 85: 100%|██████████| 6195/6195 [11:28<00:00,  8.99it/s, loss=0.419]
Train Epoch 85 ==> 	accuracy: 0.8654, 	precision: 0.9996, 	recall: 0.7311, 	specificity: 0.9997, 	f1: 0.8445
Test Epoch 85: 100%|██████████| 1715/1715 [01:15<00:00, 22.78it/s, loss=0.292]
Test Epoch 85 ==> 	accuracy: 0.9464, 	precision: 0.9714, 	recall: 0.7550, 	specificity: 0.9944, 	f1: 0.8496
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 86: 100%|██████████| 6195/6195 [11:15<00:00,  9.17it/s, loss=0.405]
Train Epoch 86 ==> 	accuracy: 0.8668, 	precision: 0.9996, 	recall: 0.7339, 	specificity: 0.9997, 	f1: 0.8464
Test Epoch 86: 100%|██████████| 1715/1715 [01:12<00:00, 23.53it/s, loss=0.283]
Test Epoch 86 ==> 	accuracy: 0.9438, 	precision: 0.9715, 	recall: 0.7417, 	specificity: 0.9945, 	f1: 0.8412
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 87: 100%|██████████| 6195/6195 [11:22<00:00,  9.08it/s, loss=0.369]
Train Epoch 87 ==> 	accuracy: 0.8668, 	precision: 0.9996, 	recall: 0.7338, 	specificity: 0.9997, 	f1: 0.8463
Test Epoch 87: 100%|██████████| 1715/1715 [01:15<00:00, 22.58it/s, loss=0.391]
Test Epoch 87 ==> 	accuracy: 0.9449, 	precision: 0.9769, 	recall: 0.7431, 	specificity: 0.9956, 	f1: 0.8441
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 88: 100%|██████████| 6195/6195 [11:19<00:00,  9.12it/s, loss=0.443]
Train Epoch 88 ==> 	accuracy: 0.8683, 	precision: 0.9996, 	recall: 0.7369, 	specificity: 0.9997, 	f1: 0.8484
Test Epoch 88: 100%|██████████| 1715/1715 [01:13<00:00, 23.28it/s, loss=4.54]
Test Epoch 88 ==> 	accuracy: 0.9461, 	precision: 0.9614, 	recall: 0.7623, 	specificity: 0.9923, 	f1: 0.8504
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 89: 100%|██████████| 6195/6195 [11:19<00:00,  9.11it/s, loss=0.299]
Train Epoch 89 ==> 	accuracy: 0.8676, 	precision: 0.9996, 	recall: 0.7355, 	specificity: 0.9997, 	f1: 0.8475
Test Epoch 89: 100%|██████████| 1715/1715 [01:11<00:00, 23.94it/s, loss=1.85]
Test Epoch 89 ==> 	accuracy: 0.9456, 	precision: 0.9674, 	recall: 0.7546, 	specificity: 0.9936, 	f1: 0.8479
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 90: 100%|██████████| 6195/6195 [11:15<00:00,  9.17it/s, loss=0.368]
Train Epoch 90 ==> 	accuracy: 0.8691, 	precision: 0.9996, 	recall: 0.7384, 	specificity: 0.9997, 	f1: 0.8494
Test Epoch 90: 100%|██████████| 1715/1715 [01:15<00:00, 22.65it/s, loss=0.147]
Test Epoch 90 ==> 	accuracy: 0.9479, 	precision: 0.9675, 	recall: 0.7660, 	specificity: 0.9935, 	f1: 0.8550
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 91: 100%|██████████| 6195/6195 [11:05<00:00,  9.30it/s, loss=0.574]
Train Epoch 91 ==> 	accuracy: 0.8690, 	precision: 0.9996, 	recall: 0.7382, 	specificity: 0.9997, 	f1: 0.8493
Test Epoch 91: 100%|██████████| 1715/1715 [01:10<00:00, 24.34it/s, loss=2.29]
Test Epoch 91 ==> 	accuracy: 0.9466, 	precision: 0.9672, 	recall: 0.7599, 	specificity: 0.9935, 	f1: 0.8511
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 92: 100%|██████████| 6195/6195 [11:14<00:00,  9.18it/s, loss=0.334]
Train Epoch 92 ==> 	accuracy: 0.8689, 	precision: 0.9996, 	recall: 0.7380, 	specificity: 0.9997, 	f1: 0.8491
Test Epoch 92: 100%|██████████| 1715/1715 [01:12<00:00, 23.74it/s, loss=0.348]
Test Epoch 92 ==> 	accuracy: 0.9466, 	precision: 0.9690, 	recall: 0.7583, 	specificity: 0.9939, 	f1: 0.8508
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 93: 100%|██████████| 6195/6195 [11:04<00:00,  9.32it/s, loss=0.345]
Train Epoch 93 ==> 	accuracy: 0.8700, 	precision: 0.9996, 	recall: 0.7403, 	specificity: 0.9997, 	f1: 0.8507
Test Epoch 93: 100%|██████████| 1715/1715 [01:15<00:00, 22.63it/s, loss=1.8]
Test Epoch 93 ==> 	accuracy: 0.9479, 	precision: 0.9694, 	recall: 0.7644, 	specificity: 0.9939, 	f1: 0.8548
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 94: 100%|██████████| 6195/6195 [11:08<00:00,  9.26it/s, loss=0.394]
Train Epoch 94 ==> 	accuracy: 0.8676, 	precision: 0.9997, 	recall: 0.7354, 	specificity: 0.9997, 	f1: 0.8474
Test Epoch 94: 100%|██████████| 1715/1715 [01:12<00:00, 23.50it/s, loss=0.318]
Test Epoch 94 ==> 	accuracy: 0.9476, 	precision: 0.9692, 	recall: 0.7631, 	specificity: 0.9939, 	f1: 0.8539
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 95: 100%|██████████| 6195/6195 [11:19<00:00,  9.11it/s, loss=0.341]
Train Epoch 95 ==> 	accuracy: 0.8701, 	precision: 0.9997, 	recall: 0.7404, 	specificity: 0.9997, 	f1: 0.8507
Test Epoch 95: 100%|██████████| 1715/1715 [01:12<00:00, 23.81it/s, loss=0.318]
Test Epoch 95 ==> 	accuracy: 0.9476, 	precision: 0.9712, 	recall: 0.7617, 	specificity: 0.9943, 	f1: 0.8538
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 96: 100%|██████████| 6195/6195 [11:18<00:00,  9.14it/s, loss=0.428]
Train Epoch 96 ==> 	accuracy: 0.8713, 	precision: 0.9996, 	recall: 0.7428, 	specificity: 0.9997, 	f1: 0.8523
Test Epoch 96: 100%|██████████| 1715/1715 [01:11<00:00, 24.15it/s, loss=2.86]
Test Epoch 96 ==> 	accuracy: 0.9477, 	precision: 0.9661, 	recall: 0.7665, 	specificity: 0.9933, 	f1: 0.8548
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 97: 100%|██████████| 6195/6195 [11:10<00:00,  9.24it/s, loss=0.407]
Train Epoch 97 ==> 	accuracy: 0.8705, 	precision: 0.9997, 	recall: 0.7413, 	specificity: 0.9998, 	f1: 0.8513
Test Epoch 97: 100%|██████████| 1715/1715 [01:12<00:00, 23.77it/s, loss=0.353]
Test Epoch 97 ==> 	accuracy: 0.9479, 	precision: 0.9661, 	recall: 0.7672, 	specificity: 0.9932, 	f1: 0.8552
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 98: 100%|██████████| 6195/6195 [11:16<00:00,  9.15it/s, loss=0.433]
Train Epoch 98 ==> 	accuracy: 0.8718, 	precision: 0.9997, 	recall: 0.7438, 	specificity: 0.9997, 	f1: 0.8530
Test Epoch 98: 100%|██████████| 1715/1715 [01:17<00:00, 22.05it/s, loss=0.19]
Test Epoch 98 ==> 	accuracy: 0.9478, 	precision: 0.9687, 	recall: 0.7649, 	specificity: 0.9938, 	f1: 0.8548
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 99: 100%|██████████| 6195/6195 [11:09<00:00,  9.25it/s, loss=0.397]
Train Epoch 99 ==> 	accuracy: 0.8709, 	precision: 0.9997, 	recall: 0.7421, 	specificity: 0.9998, 	f1: 0.8518
Test Epoch 99: 100%|██████████| 1715/1715 [01:15<00:00, 22.76it/s, loss=3.04]
Test Epoch 99 ==> 	accuracy: 0.9474, 	precision: 0.9661, 	recall: 0.7646, 	specificity: 0.9933, 	f1: 0.8536
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 100: 100%|██████████| 6195/6195 [11:09<00:00,  9.26it/s, loss=0.37]
Train Epoch 100 ==> 	accuracy: 0.8729, 	precision: 0.9997, 	recall: 0.7460, 	specificity: 0.9998, 	f1: 0.8544
Test Epoch 100: 100%|██████████| 1715/1715 [01:13<00:00, 23.25it/s, loss=0.117]
Test Epoch 100 ==> 	accuracy: 0.9480, 	precision: 0.9672, 	recall: 0.7668, 	specificity: 0.9935, 	f1: 0.8554
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 101: 100%|██████████| 6195/6195 [11:08<00:00,  9.27it/s, loss=0.48]
Train Epoch 101 ==> 	accuracy: 0.8725, 	precision: 0.9997, 	recall: 0.7453, 	specificity: 0.9997, 	f1: 0.8540
Test Epoch 101: 100%|██████████| 1715/1715 [01:06<00:00, 25.93it/s, loss=0.207]
Test Epoch 101 ==> 	accuracy: 0.9474, 	precision: 0.9677, 	recall: 0.7633, 	specificity: 0.9936, 	f1: 0.8535
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 102: 100%|██████████| 6195/6195 [11:10<00:00,  9.25it/s, loss=0.368]
Train Epoch 102 ==> 	accuracy: 0.8753, 	precision: 0.9997, 	recall: 0.7508, 	specificity: 0.9997, 	f1: 0.8575
Test Epoch 102: 100%|██████████| 1715/1715 [01:10<00:00, 24.35it/s, loss=0.587]
Test Epoch 102 ==> 	accuracy: 0.9490, 	precision: 0.9632, 	recall: 0.7756, 	specificity: 0.9926, 	f1: 0.8593
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 103: 100%|██████████| 6195/6195 [11:10<00:00,  9.25it/s, loss=0.498]
Train Epoch 103 ==> 	accuracy: 0.8748, 	precision: 0.9996, 	recall: 0.7499, 	specificity: 0.9997, 	f1: 0.8569
Test Epoch 103: 100%|██████████| 1715/1715 [01:13<00:00, 23.42it/s, loss=2.1]
Test Epoch 103 ==> 	accuracy: 0.9483, 	precision: 0.9646, 	recall: 0.7708, 	specificity: 0.9929, 	f1: 0.8569
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 104: 100%|██████████| 6195/6195 [11:08<00:00,  9.27it/s, loss=0.364]
Train Epoch 104 ==> 	accuracy: 0.8730, 	precision: 0.9997, 	recall: 0.7462, 	specificity: 0.9997, 	f1: 0.8546
Test Epoch 104: 100%|██████████| 1715/1715 [01:13<00:00, 23.29it/s, loss=0.837]
Test Epoch 104 ==> 	accuracy: 0.9481, 	precision: 0.9710, 	recall: 0.7643, 	specificity: 0.9943, 	f1: 0.8553
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 105: 100%|██████████| 6195/6195 [11:17<00:00,  9.14it/s, loss=0.407]
Train Epoch 105 ==> 	accuracy: 0.8742, 	precision: 0.9996, 	recall: 0.7486, 	specificity: 0.9997, 	f1: 0.8561
Test Epoch 105: 100%|██████████| 1715/1715 [01:09<00:00, 24.63it/s, loss=1.27]
Test Epoch 105 ==> 	accuracy: 0.9493, 	precision: 0.9693, 	recall: 0.7720, 	specificity: 0.9939, 	f1: 0.8594
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 106: 100%|██████████| 6195/6195 [11:11<00:00,  9.22it/s, loss=0.354]
Train Epoch 106 ==> 	accuracy: 0.8750, 	precision: 0.9997, 	recall: 0.7502, 	specificity: 0.9997, 	f1: 0.8571
Test Epoch 106: 100%|██████████| 1715/1715 [01:13<00:00, 23.28it/s, loss=2.87]
Test Epoch 106 ==> 	accuracy: 0.9488, 	precision: 0.9667, 	recall: 0.7717, 	specificity: 0.9933, 	f1: 0.8582
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 107: 100%|██████████| 6195/6195 [11:14<00:00,  9.18it/s, loss=0.328]
Train Epoch 107 ==> 	accuracy: 0.8743, 	precision: 0.9997, 	recall: 0.7488, 	specificity: 0.9998, 	f1: 0.8562
Test Epoch 107: 100%|██████████| 1715/1715 [01:13<00:00, 23.36it/s, loss=0.47]
Test Epoch 107 ==> 	accuracy: 0.9482, 	precision: 0.9672, 	recall: 0.7681, 	specificity: 0.9935, 	f1: 0.8563
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 108: 100%|██████████| 6195/6195 [11:17<00:00,  9.15it/s, loss=0.479]
Train Epoch 108 ==> 	accuracy: 0.8739, 	precision: 0.9997, 	recall: 0.7481, 	specificity: 0.9998, 	f1: 0.8558
Test Epoch 108: 100%|██████████| 1715/1715 [01:10<00:00, 24.22it/s, loss=1.08]
Test Epoch 108 ==> 	accuracy: 0.9476, 	precision: 0.9630, 	recall: 0.7685, 	specificity: 0.9926, 	f1: 0.8548
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 109: 100%|██████████| 6195/6195 [11:11<00:00,  9.23it/s, loss=0.325]
Train Epoch 109 ==> 	accuracy: 0.8772, 	precision: 0.9997, 	recall: 0.7547, 	specificity: 0.9998, 	f1: 0.8601
Test Epoch 109: 100%|██████████| 1715/1715 [01:12<00:00, 23.68it/s, loss=0.22]
Test Epoch 109 ==> 	accuracy: 0.9496, 	precision: 0.9674, 	recall: 0.7749, 	specificity: 0.9934, 	f1: 0.8605
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 110: 100%|██████████| 6195/6195 [11:04<00:00,  9.33it/s, loss=0.388]
Train Epoch 110 ==> 	accuracy: 0.8749, 	precision: 0.9997, 	recall: 0.7500, 	specificity: 0.9998, 	f1: 0.8570
Test Epoch 110: 100%|██████████| 1715/1715 [01:14<00:00, 23.13it/s, loss=0.434]
Test Epoch 110 ==> 	accuracy: 0.9490, 	precision: 0.9656, 	recall: 0.7733, 	specificity: 0.9931, 	f1: 0.8588
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 111: 100%|██████████| 6195/6195 [11:07<00:00,  9.28it/s, loss=0.492]
Train Epoch 111 ==> 	accuracy: 0.8770, 	precision: 0.9997, 	recall: 0.7542, 	specificity: 0.9997, 	f1: 0.8597
Test Epoch 111: 100%|██████████| 1715/1715 [01:11<00:00, 23.92it/s, loss=0.194]
Test Epoch 111 ==> 	accuracy: 0.9493, 	precision: 0.9669, 	recall: 0.7738, 	specificity: 0.9933, 	f1: 0.8596
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 112: 100%|██████████| 6195/6195 [11:32<00:00,  8.95it/s, loss=0.393]
Train Epoch 112 ==> 	accuracy: 0.8750, 	precision: 0.9997, 	recall: 0.7502, 	specificity: 0.9997, 	f1: 0.8572
Test Epoch 112: 100%|██████████| 1715/1715 [01:18<00:00, 21.75it/s, loss=0.442]
Test Epoch 112 ==> 	accuracy: 0.9482, 	precision: 0.9655, 	recall: 0.7694, 	specificity: 0.9931, 	f1: 0.8563
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 113: 100%|██████████| 6195/6195 [11:32<00:00,  8.95it/s, loss=0.351]
Train Epoch 113 ==> 	accuracy: 0.8788, 	precision: 0.9997, 	recall: 0.7578, 	specificity: 0.9998, 	f1: 0.8621
Test Epoch 113: 100%|██████████| 1715/1715 [01:14<00:00, 23.10it/s, loss=0.239]
Test Epoch 113 ==> 	accuracy: 0.9493, 	precision: 0.9627, 	recall: 0.7778, 	specificity: 0.9924, 	f1: 0.8604
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 114: 100%|██████████| 6195/6195 [11:26<00:00,  9.03it/s, loss=0.388]
Train Epoch 114 ==> 	accuracy: 0.8759, 	precision: 0.9997, 	recall: 0.7521, 	specificity: 0.9998, 	f1: 0.8584
Test Epoch 114: 100%|██████████| 1715/1715 [01:15<00:00, 22.59it/s, loss=2.83]
Test Epoch 114 ==> 	accuracy: 0.9490, 	precision: 0.9643, 	recall: 0.7746, 	specificity: 0.9928, 	f1: 0.8591
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 115: 100%|██████████| 6195/6195 [11:28<00:00,  8.99it/s, loss=0.427]
Train Epoch 115 ==> 	accuracy: 0.8775, 	precision: 0.9997, 	recall: 0.7552, 	specificity: 0.9998, 	f1: 0.8604
Test Epoch 115: 100%|██████████| 1715/1715 [01:10<00:00, 24.29it/s, loss=0.602]
Test Epoch 115 ==> 	accuracy: 0.9493, 	precision: 0.9681, 	recall: 0.7728, 	specificity: 0.9936, 	f1: 0.8595
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 116: 100%|██████████| 6195/6195 [11:27<00:00,  9.01it/s, loss=0.427]
Train Epoch 116 ==> 	accuracy: 0.8757, 	precision: 0.9997, 	recall: 0.7516, 	specificity: 0.9998, 	f1: 0.8581
Test Epoch 116: 100%|██████████| 1715/1715 [01:11<00:00, 23.93it/s, loss=0.301]
Test Epoch 116 ==> 	accuracy: 0.9498, 	precision: 0.9687, 	recall: 0.7749, 	specificity: 0.9937, 	f1: 0.8610
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 117: 100%|██████████| 6195/6195 [11:33<00:00,  8.93it/s, loss=0.512]
Train Epoch 117 ==> 	accuracy: 0.8759, 	precision: 0.9997, 	recall: 0.7520, 	specificity: 0.9998, 	f1: 0.8584
Test Epoch 117: 100%|██████████| 1715/1715 [01:17<00:00, 22.20it/s, loss=0.189]
Test Epoch 117 ==> 	accuracy: 0.9493, 	precision: 0.9649, 	recall: 0.7757, 	specificity: 0.9929, 	f1: 0.8600
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 118: 100%|██████████| 6195/6195 [11:23<00:00,  9.06it/s, loss=0.442]
Train Epoch 118 ==> 	accuracy: 0.8773, 	precision: 0.9997, 	recall: 0.7548, 	specificity: 0.9998, 	f1: 0.8602
Test Epoch 118: 100%|██████████| 1715/1715 [01:17<00:00, 22.17it/s, loss=3.48]
Test Epoch 118 ==> 	accuracy: 0.9497, 	precision: 0.9669, 	recall: 0.7758, 	specificity: 0.9933, 	f1: 0.8608
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 119: 100%|██████████| 6195/6195 [11:16<00:00,  9.15it/s, loss=0.374]
Train Epoch 119 ==> 	accuracy: 0.8773, 	precision: 0.9997, 	recall: 0.7549, 	specificity: 0.9998, 	f1: 0.8602
Test Epoch 119: 100%|██████████| 1715/1715 [01:14<00:00, 22.91it/s, loss=0.894]
Test Epoch 119 ==> 	accuracy: 0.9499, 	precision: 0.9694, 	recall: 0.7751, 	specificity: 0.9938, 	f1: 0.8614
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 120: 100%|██████████| 6195/6195 [11:16<00:00,  9.15it/s, loss=0.343]
Train Epoch 120 ==> 	accuracy: 0.8783, 	precision: 0.9997, 	recall: 0.7568, 	specificity: 0.9998, 	f1: 0.8614
Test Epoch 120: 100%|██████████| 1715/1715 [01:17<00:00, 22.14it/s, loss=0.667]
Test Epoch 120 ==> 	accuracy: 0.9491, 	precision: 0.9652, 	recall: 0.7745, 	specificity: 0.9930, 	f1: 0.8594
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 121: 100%|██████████| 6195/6195 [11:23<00:00,  9.06it/s, loss=0.34]
Train Epoch 121 ==> 	accuracy: 0.8775, 	precision: 0.9997, 	recall: 0.7552, 	specificity: 0.9998, 	f1: 0.8604
Test Epoch 121: 100%|██████████| 1715/1715 [01:18<00:00, 21.88it/s, loss=0.335]
Test Epoch 121 ==> 	accuracy: 0.9495, 	precision: 0.9612, 	recall: 0.7800, 	specificity: 0.9921, 	f1: 0.8612
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 122: 100%|██████████| 6195/6195 [11:22<00:00,  9.08it/s, loss=0.382]
Train Epoch 122 ==> 	accuracy: 0.8813, 	precision: 0.9997, 	recall: 0.7628, 	specificity: 0.9997, 	f1: 0.8653
Test Epoch 122: 100%|██████████| 1715/1715 [01:18<00:00, 21.86it/s, loss=5.56]
Test Epoch 122 ==> 	accuracy: 0.9497, 	precision: 0.9602, 	recall: 0.7819, 	specificity: 0.9919, 	f1: 0.8619
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 123: 100%|██████████| 6195/6195 [11:30<00:00,  8.97it/s, loss=0.338]
Train Epoch 123 ==> 	accuracy: 0.8786, 	precision: 0.9997, 	recall: 0.7575, 	specificity: 0.9998, 	f1: 0.8619
Test Epoch 123: 100%|██████████| 1715/1715 [01:17<00:00, 22.24it/s, loss=0.81]
Test Epoch 123 ==> 	accuracy: 0.9494, 	precision: 0.9666, 	recall: 0.7748, 	specificity: 0.9933, 	f1: 0.8602
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 124: 100%|██████████| 6195/6195 [11:23<00:00,  9.06it/s, loss=0.384]
Train Epoch 124 ==> 	accuracy: 0.8794, 	precision: 0.9997, 	recall: 0.7591, 	specificity: 0.9998, 	f1: 0.8630
Test Epoch 124: 100%|██████████| 1715/1715 [01:11<00:00, 24.03it/s, loss=0.176]
Test Epoch 124 ==> 	accuracy: 0.9498, 	precision: 0.9665, 	recall: 0.7771, 	specificity: 0.9932, 	f1: 0.8615
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 125: 100%|██████████| 6195/6195 [11:09<00:00,  9.26it/s, loss=0.327]
Train Epoch 125 ==> 	accuracy: 0.8800, 	precision: 0.9997, 	recall: 0.7603, 	specificity: 0.9997, 	f1: 0.8637
Test Epoch 125: 100%|██████████| 1715/1715 [01:11<00:00, 23.97it/s, loss=0.691]
Test Epoch 125 ==> 	accuracy: 0.9492, 	precision: 0.9627, 	recall: 0.7770, 	specificity: 0.9924, 	f1: 0.8599
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 126: 100%|██████████| 6195/6195 [11:18<00:00,  9.13it/s, loss=0.364]
Train Epoch 126 ==> 	accuracy: 0.8819, 	precision: 0.9997, 	recall: 0.7642, 	specificity: 0.9997, 	f1: 0.8662
Test Epoch 126: 100%|██████████| 1715/1715 [01:18<00:00, 21.82it/s, loss=0.791]
Test Epoch 126 ==> 	accuracy: 0.9506, 	precision: 0.9630, 	recall: 0.7842, 	specificity: 0.9924, 	f1: 0.8644
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 127: 100%|██████████| 6195/6195 [11:18<00:00,  9.13it/s, loss=0.325]
Train Epoch 127 ==> 	accuracy: 0.8774, 	precision: 0.9997, 	recall: 0.7550, 	specificity: 0.9998, 	f1: 0.8603
Test Epoch 127: 100%|██████████| 1715/1715 [01:16<00:00, 22.33it/s, loss=2.31]
Test Epoch 127 ==> 	accuracy: 0.9467, 	precision: 0.9703, 	recall: 0.7576, 	specificity: 0.9942, 	f1: 0.8509
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 128: 100%|██████████| 6195/6195 [11:24<00:00,  9.05it/s, loss=0.511]
Train Epoch 128 ==> 	accuracy: 0.8783, 	precision: 0.9997, 	recall: 0.7569, 	specificity: 0.9998, 	f1: 0.8615
Test Epoch 128: 100%|██████████| 1715/1715 [01:15<00:00, 22.70it/s, loss=2.44]
Test Epoch 128 ==> 	accuracy: 0.9515, 	precision: 0.9723, 	recall: 0.7807, 	specificity: 0.9944, 	f1: 0.8660
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 129: 100%|██████████| 6195/6195 [11:09<00:00,  9.25it/s, loss=0.353]
Train Epoch 129 ==> 	accuracy: 0.8788, 	precision: 0.9997, 	recall: 0.7579, 	specificity: 0.9997, 	f1: 0.8621
Test Epoch 129: 100%|██████████| 1715/1715 [01:17<00:00, 22.00it/s, loss=0.556]
Test Epoch 129 ==> 	accuracy: 0.9512, 	precision: 0.9743, 	recall: 0.7773, 	specificity: 0.9949, 	f1: 0.8647
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 130: 100%|██████████| 6195/6195 [11:11<00:00,  9.23it/s, loss=0.356]
Train Epoch 130 ==> 	accuracy: 0.8787, 	precision: 0.9997, 	recall: 0.7577, 	specificity: 0.9998, 	f1: 0.8620
Test Epoch 130: 100%|██████████| 1715/1715 [01:20<00:00, 21.37it/s, loss=3.09]
Test Epoch 130 ==> 	accuracy: 0.9515, 	precision: 0.9741, 	recall: 0.7793, 	specificity: 0.9948, 	f1: 0.8659
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 131: 100%|██████████| 6195/6195 [11:16<00:00,  9.16it/s, loss=0.28]
Train Epoch 131 ==> 	accuracy: 0.8781, 	precision: 0.9997, 	recall: 0.7564, 	specificity: 0.9998, 	f1: 0.8612
Test Epoch 131: 100%|██████████| 1715/1715 [01:13<00:00, 23.43it/s, loss=0.12]
Test Epoch 131 ==> 	accuracy: 0.9510, 	precision: 0.9750, 	recall: 0.7757, 	specificity: 0.9950, 	f1: 0.8640
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 132: 100%|██████████| 6195/6195 [11:20<00:00,  9.10it/s, loss=0.375]
Train Epoch 132 ==> 	accuracy: 0.8794, 	precision: 0.9997, 	recall: 0.7589, 	specificity: 0.9998, 	f1: 0.8629
Test Epoch 132: 100%|██████████| 1715/1715 [01:12<00:00, 23.53it/s, loss=2.53]
Test Epoch 132 ==> 	accuracy: 0.9514, 	precision: 0.9743, 	recall: 0.7782, 	specificity: 0.9949, 	f1: 0.8653
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 133: 100%|██████████| 6195/6195 [11:09<00:00,  9.26it/s, loss=0.336]
Train Epoch 133 ==> 	accuracy: 0.8805, 	precision: 0.9997, 	recall: 0.7612, 	specificity: 0.9998, 	f1: 0.8643
Test Epoch 133: 100%|██████████| 1715/1715 [01:14<00:00, 23.05it/s, loss=0.469]
Test Epoch 133 ==> 	accuracy: 0.9518, 	precision: 0.9714, 	recall: 0.7830, 	specificity: 0.9942, 	f1: 0.8671
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 134: 100%|██████████| 6195/6195 [11:23<00:00,  9.06it/s, loss=0.392]
Train Epoch 134 ==> 	accuracy: 0.8793, 	precision: 0.9997, 	recall: 0.7589, 	specificity: 0.9998, 	f1: 0.8628
Test Epoch 134: 100%|██████████| 1715/1715 [01:07<00:00, 25.27it/s, loss=0.113]
Test Epoch 134 ==> 	accuracy: 0.9505, 	precision: 0.9735, 	recall: 0.7744, 	specificity: 0.9947, 	f1: 0.8626
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 135: 100%|██████████| 6195/6195 [11:03<00:00,  9.34it/s, loss=0.374]
Train Epoch 135 ==> 	accuracy: 0.8824, 	precision: 0.9997, 	recall: 0.7651, 	specificity: 0.9998, 	f1: 0.8668
Test Epoch 135: 100%|██████████| 1715/1715 [01:12<00:00, 23.70it/s, loss=0.173]
Test Epoch 135 ==> 	accuracy: 0.9516, 	precision: 0.9685, 	recall: 0.7845, 	specificity: 0.9936, 	f1: 0.8668
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 136: 100%|██████████| 6195/6195 [10:59<00:00,  9.40it/s, loss=0.44]
Train Epoch 136 ==> 	accuracy: 0.8816, 	precision: 0.9997, 	recall: 0.7634, 	specificity: 0.9998, 	f1: 0.8657
Test Epoch 136: 100%|██████████| 1715/1715 [01:15<00:00, 22.61it/s, loss=1.18]
Test Epoch 136 ==> 	accuracy: 0.9514, 	precision: 0.9691, 	recall: 0.7828, 	specificity: 0.9937, 	f1: 0.8661
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 137: 100%|██████████| 6195/6195 [11:02<00:00,  9.35it/s, loss=0.339]
Train Epoch 137 ==> 	accuracy: 0.8816, 	precision: 0.9997, 	recall: 0.7635, 	specificity: 0.9998, 	f1: 0.8658
Test Epoch 137: 100%|██████████| 1715/1715 [01:10<00:00, 24.36it/s, loss=1.28]
Test Epoch 137 ==> 	accuracy: 0.9509, 	precision: 0.9696, 	recall: 0.7799, 	specificity: 0.9939, 	f1: 0.8645
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 138: 100%|██████████| 6195/6195 [11:03<00:00,  9.33it/s, loss=0.315]
Train Epoch 138 ==> 	accuracy: 0.8812, 	precision: 0.9997, 	recall: 0.7626, 	specificity: 0.9998, 	f1: 0.8652
Test Epoch 138: 100%|██████████| 1715/1715 [01:12<00:00, 23.58it/s, loss=0.14]
Test Epoch 138 ==> 	accuracy: 0.9506, 	precision: 0.9702, 	recall: 0.7780, 	specificity: 0.9940, 	f1: 0.8635
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 139: 100%|██████████| 6195/6195 [11:07<00:00,  9.28it/s, loss=0.347]
Train Epoch 139 ==> 	accuracy: 0.8837, 	precision: 0.9997, 	recall: 0.7676, 	specificity: 0.9998, 	f1: 0.8684
Test Epoch 139: 100%|██████████| 1715/1715 [01:10<00:00, 24.19it/s, loss=0.216]
Test Epoch 139 ==> 	accuracy: 0.9513, 	precision: 0.9690, 	recall: 0.7824, 	specificity: 0.9937, 	f1: 0.8658
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 140: 100%|██████████| 6195/6195 [11:01<00:00,  9.36it/s, loss=0.352]
Train Epoch 140 ==> 	accuracy: 0.8815, 	precision: 0.9997, 	recall: 0.7633, 	specificity: 0.9998, 	f1: 0.8657
Test Epoch 140: 100%|██████████| 1715/1715 [01:15<00:00, 22.66it/s, loss=4.2]
Test Epoch 140 ==> 	accuracy: 0.9513, 	precision: 0.9710, 	recall: 0.7806, 	specificity: 0.9942, 	f1: 0.8655
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 141: 100%|██████████| 6195/6195 [11:07<00:00,  9.27it/s, loss=0.342]
Train Epoch 141 ==> 	accuracy: 0.8799, 	precision: 0.9997, 	recall: 0.7600, 	specificity: 0.9998, 	f1: 0.8635
Test Epoch 141: 100%|██████████| 1715/1715 [01:14<00:00, 23.04it/s, loss=0.166]
Test Epoch 141 ==> 	accuracy: 0.9504, 	precision: 0.9743, 	recall: 0.7732, 	specificity: 0.9949, 	f1: 0.8622
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 142: 100%|██████████| 6195/6195 [10:57<00:00,  9.43it/s, loss=0.318]
Train Epoch 142 ==> 	accuracy: 0.8810, 	precision: 0.9997, 	recall: 0.7623, 	specificity: 0.9998, 	f1: 0.8650
Test Epoch 142: 100%|██████████| 1715/1715 [01:12<00:00, 23.72it/s, loss=0.123]
Test Epoch 142 ==> 	accuracy: 0.9511, 	precision: 0.9719, 	recall: 0.7790, 	specificity: 0.9943, 	f1: 0.8648
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 143: 100%|██████████| 6195/6195 [11:09<00:00,  9.25it/s, loss=0.5]
Train Epoch 143 ==> 	accuracy: 0.8804, 	precision: 0.9997, 	recall: 0.7610, 	specificity: 0.9998, 	f1: 0.8641
Test Epoch 143: 100%|██████████| 1715/1715 [01:14<00:00, 22.90it/s, loss=1.07]
Test Epoch 143 ==> 	accuracy: 0.9518, 	precision: 0.9724, 	recall: 0.7823, 	specificity: 0.9944, 	f1: 0.8670
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 144: 100%|██████████| 6195/6195 [11:08<00:00,  9.27it/s, loss=0.366]
Train Epoch 144 ==> 	accuracy: 0.8823, 	precision: 0.9997, 	recall: 0.7648, 	specificity: 0.9998, 	f1: 0.8666
Test Epoch 144: 100%|██████████| 1715/1715 [01:05<00:00, 26.00it/s, loss=0.256]
Test Epoch 144 ==> 	accuracy: 0.9518, 	precision: 0.9715, 	recall: 0.7829, 	specificity: 0.9942, 	f1: 0.8671
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 145: 100%|██████████| 6195/6195 [10:48<00:00,  9.56it/s, loss=0.363]
Train Epoch 145 ==> 	accuracy: 0.8814, 	precision: 0.9997, 	recall: 0.7629, 	specificity: 0.9998, 	f1: 0.8654
Test Epoch 145: 100%|██████████| 1715/1715 [01:06<00:00, 25.65it/s, loss=0.141]
Test Epoch 145 ==> 	accuracy: 0.9509, 	precision: 0.9717, 	recall: 0.7780, 	specificity: 0.9943, 	f1: 0.8641
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 146: 100%|██████████| 6195/6195 [10:52<00:00,  9.49it/s, loss=0.313]
Train Epoch 146 ==> 	accuracy: 0.8812, 	precision: 0.9997, 	recall: 0.7626, 	specificity: 0.9998, 	f1: 0.8652
Test Epoch 146: 100%|██████████| 1715/1715 [01:10<00:00, 24.48it/s, loss=0.258]
Test Epoch 146 ==> 	accuracy: 0.9514, 	precision: 0.9700, 	recall: 0.7824, 	specificity: 0.9939, 	f1: 0.8661
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 147: 100%|██████████| 6195/6195 [10:54<00:00,  9.47it/s, loss=0.366]
Train Epoch 147 ==> 	accuracy: 0.8818, 	precision: 0.9997, 	recall: 0.7638, 	specificity: 0.9998, 	f1: 0.8660
Test Epoch 147: 100%|██████████| 1715/1715 [01:14<00:00, 23.00it/s, loss=0.902]
Test Epoch 147 ==> 	accuracy: 0.9516, 	precision: 0.9721, 	recall: 0.7814, 	specificity: 0.9944, 	f1: 0.8664
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 148: 100%|██████████| 6195/6195 [11:01<00:00,  9.36it/s, loss=0.384]
Train Epoch 148 ==> 	accuracy: 0.8831, 	precision: 0.9997, 	recall: 0.7664, 	specificity: 0.9998, 	f1: 0.8677
Test Epoch 148: 100%|██████████| 1715/1715 [01:08<00:00, 24.97it/s, loss=0.392]
Test Epoch 148 ==> 	accuracy: 0.9523, 	precision: 0.9682, 	recall: 0.7883, 	specificity: 0.9935, 	f1: 0.8691
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 149: 100%|██████████| 6195/6195 [11:07<00:00,  9.27it/s, loss=0.341]
Train Epoch 149 ==> 	accuracy: 0.8825, 	precision: 0.9997, 	recall: 0.7652, 	specificity: 0.9998, 	f1: 0.8669
Test Epoch 149: 100%|██████████| 1715/1715 [01:06<00:00, 25.64it/s, loss=0.389]
Test Epoch 149 ==> 	accuracy: 0.9512, 	precision: 0.9697, 	recall: 0.7815, 	specificity: 0.9939, 	f1: 0.8655
Adjusting learning rate of group 0 to 4.2391e-06.

Process finished with exit code 0

'''

'''
'../model_save_sigBlock4_focalWithMs_deformable_ab_mha'
spec 下降
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6195/6195 [04:01<00:00, 25.60it/s, loss=0.56]
Train Epoch 0 ==> 	accuracy: 0.6312, 	precision: 0.9953, 	recall: 0.2637, 	specificity: 0.9988, 	f1: 0.4169
Test Epoch 0: 100%|██████████| 1715/1715 [00:28<00:00, 60.93it/s, loss=0.491]
Test Epoch 0 ==> 	accuracy: 0.8944, 	precision: 0.9655, 	recall: 0.4916, 	specificity: 0.9956, 	f1: 0.6515
Train Epoch 1: 100%|██████████| 6195/6195 [05:13<00:00, 19.77it/s, loss=0.53]
Train Epoch 1 ==> 	accuracy: 0.7156, 	precision: 0.9967, 	recall: 0.4325, 	specificity: 0.9986, 	f1: 0.6033
Test Epoch 1: 100%|██████████| 1715/1715 [00:34<00:00, 50.23it/s, loss=0.676]
Test Epoch 1 ==> 	accuracy: 0.9035, 	precision: 0.9752, 	recall: 0.5328, 	specificity: 0.9966, 	f1: 0.6891
Train Epoch 2: 100%|██████████| 6195/6195 [05:31<00:00, 18.71it/s, loss=0.604]
Train Epoch 2 ==> 	accuracy: 0.7437, 	precision: 0.9973, 	recall: 0.4886, 	specificity: 0.9987, 	f1: 0.6559
Test Epoch 2: 100%|██████████| 1715/1715 [00:31<00:00, 53.70it/s, loss=0.437]
Test Epoch 2 ==> 	accuracy: 0.9023, 	precision: 0.9766, 	recall: 0.5258, 	specificity: 0.9968, 	f1: 0.6836
Train Epoch 3: 100%|██████████| 6195/6195 [05:35<00:00, 18.48it/s, loss=0.561]
Train Epoch 3 ==> 	accuracy: 0.7506, 	precision: 0.9975, 	recall: 0.5025, 	specificity: 0.9987, 	f1: 0.6683
Test Epoch 3: 100%|██████████| 1715/1715 [00:33<00:00, 51.08it/s, loss=0.271]
Test Epoch 3 ==> 	accuracy: 0.9083, 	precision: 0.9557, 	recall: 0.5695, 	specificity: 0.9934, 	f1: 0.7137
Train Epoch 4: 100%|██████████| 6195/6195 [05:32<00:00, 18.65it/s, loss=0.594]
Train Epoch 4 ==> 	accuracy: 0.7616, 	precision: 0.9978, 	recall: 0.5243, 	specificity: 0.9988, 	f1: 0.6874
Test Epoch 4: 100%|██████████| 1715/1715 [00:33<00:00, 50.86it/s, loss=0.396]
Test Epoch 4 ==> 	accuracy: 0.9131, 	precision: 0.9689, 	recall: 0.5862, 	specificity: 0.9953, 	f1: 0.7304
Train Epoch 5: 100%|██████████| 6195/6195 [05:44<00:00, 17.96it/s, loss=0.535]
Train Epoch 5 ==> 	accuracy: 0.7633, 	precision: 0.9979, 	recall: 0.5276, 	specificity: 0.9989, 	f1: 0.6903
Test Epoch 5: 100%|██████████| 1715/1715 [00:35<00:00, 47.73it/s, loss=0.965]
Test Epoch 5 ==> 	accuracy: 0.9160, 	precision: 0.9792, 	recall: 0.5941, 	specificity: 0.9968, 	f1: 0.7395
Train Epoch 6: 100%|██████████| 6195/6195 [05:42<00:00, 18.11it/s, loss=0.521]
Train Epoch 6 ==> 	accuracy: 0.7681, 	precision: 0.9980, 	recall: 0.5372, 	specificity: 0.9989, 	f1: 0.6985
Test Epoch 6: 100%|██████████| 1715/1715 [00:35<00:00, 48.78it/s, loss=0.413]
Test Epoch 6 ==> 	accuracy: 0.9174, 	precision: 0.9859, 	recall: 0.5972, 	specificity: 0.9979, 	f1: 0.7439
Train Epoch 7: 100%|██████████| 6195/6195 [05:42<00:00, 18.10it/s, loss=0.545]
Train Epoch 7 ==> 	accuracy: 0.7728, 	precision: 0.9980, 	recall: 0.5467, 	specificity: 0.9989, 	f1: 0.7064
Test Epoch 7: 100%|██████████| 1715/1715 [00:37<00:00, 45.87it/s, loss=0.524]
Test Epoch 7 ==> 	accuracy: 0.9143, 	precision: 0.9668, 	recall: 0.5938, 	specificity: 0.9949, 	f1: 0.7357
Train Epoch 8: 100%|██████████| 6195/6195 [05:45<00:00, 17.91it/s, loss=0.546]
Train Epoch 8 ==> 	accuracy: 0.7739, 	precision: 0.9982, 	recall: 0.5488, 	specificity: 0.9990, 	f1: 0.7082
Test Epoch 8: 100%|██████████| 1715/1715 [00:34<00:00, 49.11it/s, loss=0.372]
Test Epoch 8 ==> 	accuracy: 0.9195, 	precision: 0.9719, 	recall: 0.6167, 	specificity: 0.9955, 	f1: 0.7546
Train Epoch 9: 100%|██████████| 6195/6195 [05:35<00:00, 18.46it/s, loss=0.586]
Train Epoch 9 ==> 	accuracy: 0.7773, 	precision: 0.9983, 	recall: 0.5554, 	specificity: 0.9991, 	f1: 0.7138
Test Epoch 9: 100%|██████████| 1715/1715 [00:34<00:00, 50.16it/s, loss=0.441]
Test Epoch 9 ==> 	accuracy: 0.9196, 	precision: 0.9613, 	recall: 0.6249, 	specificity: 0.9937, 	f1: 0.7574
Train Epoch 10: 100%|██████████| 6195/6195 [05:49<00:00, 17.71it/s, loss=0.621]
Train Epoch 10 ==> 	accuracy: 0.7806, 	precision: 0.9983, 	recall: 0.5622, 	specificity: 0.9991, 	f1: 0.7194
Test Epoch 10: 100%|██████████| 1715/1715 [00:36<00:00, 46.98it/s, loss=0.332]
Test Epoch 10 ==> 	accuracy: 0.9157, 	precision: 0.9860, 	recall: 0.5884, 	specificity: 0.9979, 	f1: 0.7370
Train Epoch 11: 100%|██████████| 6195/6195 [05:39<00:00, 18.26it/s, loss=0.422]
Train Epoch 11 ==> 	accuracy: 0.7837, 	precision: 0.9984, 	recall: 0.5684, 	specificity: 0.9991, 	f1: 0.7244
Test Epoch 11: 100%|██████████| 1715/1715 [00:37<00:00, 46.05it/s, loss=0.272]
Test Epoch 11 ==> 	accuracy: 0.9213, 	precision: 0.9614, 	recall: 0.6334, 	specificity: 0.9936, 	f1: 0.7637
Train Epoch 12: 100%|██████████| 6195/6195 [05:48<00:00, 17.76it/s, loss=0.517]
Train Epoch 12 ==> 	accuracy: 0.7834, 	precision: 0.9986, 	recall: 0.5675, 	specificity: 0.9992, 	f1: 0.7237
Test Epoch 12: 100%|██████████| 1715/1715 [00:37<00:00, 46.18it/s, loss=0.237]
Test Epoch 12 ==> 	accuracy: 0.9240, 	precision: 0.9839, 	recall: 0.6316, 	specificity: 0.9974, 	f1: 0.7693
Train Epoch 13: 100%|██████████| 6195/6195 [05:51<00:00, 17.63it/s, loss=0.513]
Train Epoch 13 ==> 	accuracy: 0.7863, 	precision: 0.9985, 	recall: 0.5734, 	specificity: 0.9992, 	f1: 0.7285
Test Epoch 13: 100%|██████████| 1715/1715 [00:35<00:00, 48.26it/s, loss=1.01]
Test Epoch 13 ==> 	accuracy: 0.9193, 	precision: 0.9508, 	recall: 0.6305, 	specificity: 0.9918, 	f1: 0.7582
Train Epoch 14: 100%|██████████| 6195/6195 [05:44<00:00, 17.97it/s, loss=0.557]
Train Epoch 14 ==> 	accuracy: 0.7919, 	precision: 0.9985, 	recall: 0.5847, 	specificity: 0.9991, 	f1: 0.7375
Test Epoch 14: 100%|██████████| 1715/1715 [00:34<00:00, 49.16it/s, loss=0.267]
Test Epoch 14 ==> 	accuracy: 0.9214, 	precision: 0.9735, 	recall: 0.6257, 	specificity: 0.9957, 	f1: 0.7618
Train Epoch 15: 100%|██████████| 6195/6195 [05:39<00:00, 18.24it/s, loss=0.477]
Train Epoch 15 ==> 	accuracy: 0.7889, 	precision: 0.9987, 	recall: 0.5785, 	specificity: 0.9992, 	f1: 0.7326
Test Epoch 15: 100%|██████████| 1715/1715 [00:33<00:00, 51.20it/s, loss=0.478]
Test Epoch 15 ==> 	accuracy: 0.9157, 	precision: 0.9674, 	recall: 0.6005, 	specificity: 0.9949, 	f1: 0.7410
Train Epoch 16: 100%|██████████| 6195/6195 [05:41<00:00, 18.12it/s, loss=0.467]
Train Epoch 16 ==> 	accuracy: 0.7895, 	precision: 0.9987, 	recall: 0.5798, 	specificity: 0.9992, 	f1: 0.7337
Test Epoch 16: 100%|██████████| 1715/1715 [00:35<00:00, 47.98it/s, loss=0.545]
Test Epoch 16 ==> 	accuracy: 0.9237, 	precision: 0.9725, 	recall: 0.6379, 	specificity: 0.9955, 	f1: 0.7704
Train Epoch 17: 100%|██████████| 6195/6195 [05:37<00:00, 18.38it/s, loss=0.475]
Train Epoch 17 ==> 	accuracy: 0.7943, 	precision: 0.9987, 	recall: 0.5893, 	specificity: 0.9993, 	f1: 0.7412
Test Epoch 17: 100%|██████████| 1715/1715 [00:32<00:00, 52.27it/s, loss=0.296]
Test Epoch 17 ==> 	accuracy: 0.9230, 	precision: 0.9635, 	recall: 0.6407, 	specificity: 0.9939, 	f1: 0.7696
Train Epoch 18: 100%|██████████| 6195/6195 [05:42<00:00, 18.10it/s, loss=0.475]
Train Epoch 18 ==> 	accuracy: 0.7939, 	precision: 0.9987, 	recall: 0.5885, 	specificity: 0.9993, 	f1: 0.7406
Test Epoch 18: 100%|██████████| 1715/1715 [00:36<00:00, 46.64it/s, loss=0.168]
Test Epoch 18 ==> 	accuracy: 0.9248, 	precision: 0.9742, 	recall: 0.6427, 	specificity: 0.9957, 	f1: 0.7745
Train Epoch 19: 100%|██████████| 6195/6195 [05:43<00:00, 18.02it/s, loss=0.506]
Train Epoch 19 ==> 	accuracy: 0.7944, 	precision: 0.9988, 	recall: 0.5895, 	specificity: 0.9993, 	f1: 0.7414
Test Epoch 19: 100%|██████████| 1715/1715 [00:34<00:00, 49.33it/s, loss=0.262]
Test Epoch 19 ==> 	accuracy: 0.9274, 	precision: 0.9875, 	recall: 0.6466, 	specificity: 0.9979, 	f1: 0.7815
Train Epoch 20: 100%|██████████| 6195/6195 [05:42<00:00, 18.11it/s, loss=0.526]
Train Epoch 20 ==> 	accuracy: 0.7933, 	precision: 0.9988, 	recall: 0.5874, 	specificity: 0.9993, 	f1: 0.7397
Test Epoch 20: 100%|██████████| 1715/1715 [00:37<00:00, 45.90it/s, loss=0.471]
Test Epoch 20 ==> 	accuracy: 0.9236, 	precision: 0.9836, 	recall: 0.6301, 	specificity: 0.9974, 	f1: 0.7681
Train Epoch 21: 100%|██████████| 6195/6195 [05:39<00:00, 18.23it/s, loss=0.751]
Train Epoch 21 ==> 	accuracy: 0.7999, 	precision: 0.9989, 	recall: 0.6005, 	specificity: 0.9993, 	f1: 0.7501
Test Epoch 21: 100%|██████████| 1715/1715 [00:32<00:00, 52.35it/s, loss=0.404]
Test Epoch 21 ==> 	accuracy: 0.9250, 	precision: 0.9864, 	recall: 0.6354, 	specificity: 0.9978, 	f1: 0.7729
Train Epoch 22: 100%|██████████| 6195/6195 [05:38<00:00, 18.28it/s, loss=0.458]
Train Epoch 22 ==> 	accuracy: 0.7966, 	precision: 0.9989, 	recall: 0.5939, 	specificity: 0.9994, 	f1: 0.7449
Test Epoch 22: 100%|██████████| 1715/1715 [00:37<00:00, 46.33it/s, loss=0.996]
Test Epoch 22 ==> 	accuracy: 0.9224, 	precision: 0.9773, 	recall: 0.6280, 	specificity: 0.9963, 	f1: 0.7647
Train Epoch 23: 100%|██████████| 6195/6195 [05:42<00:00, 18.07it/s, loss=0.703]
Train Epoch 23 ==> 	accuracy: 0.8015, 	precision: 0.9989, 	recall: 0.6038, 	specificity: 0.9993, 	f1: 0.7526
Test Epoch 23: 100%|██████████| 1715/1715 [00:34<00:00, 49.08it/s, loss=0.553]
Test Epoch 23 ==> 	accuracy: 0.9267, 	precision: 0.9640, 	recall: 0.6595, 	specificity: 0.9938, 	f1: 0.7832
Train Epoch 24: 100%|██████████| 6195/6195 [05:43<00:00, 18.06it/s, loss=0.579]
Train Epoch 24 ==> 	accuracy: 0.7972, 	precision: 0.9989, 	recall: 0.5951, 	specificity: 0.9993, 	f1: 0.7459
Test Epoch 24: 100%|██████████| 1715/1715 [00:35<00:00, 47.88it/s, loss=0.421]
Test Epoch 24 ==> 	accuracy: 0.9251, 	precision: 0.9798, 	recall: 0.6402, 	specificity: 0.9967, 	f1: 0.7744
Train Epoch 25: 100%|██████████| 6195/6195 [05:36<00:00, 18.43it/s, loss=0.49]
Train Epoch 25 ==> 	accuracy: 0.8013, 	precision: 0.9990, 	recall: 0.6033, 	specificity: 0.9994, 	f1: 0.7522
Test Epoch 25: 100%|██████████| 1715/1715 [00:33<00:00, 51.08it/s, loss=0.534]
Test Epoch 25 ==> 	accuracy: 0.9275, 	precision: 0.9790, 	recall: 0.6531, 	specificity: 0.9965, 	f1: 0.7835
Train Epoch 26: 100%|██████████| 6195/6195 [05:41<00:00, 18.15it/s, loss=0.466]
Train Epoch 26 ==> 	accuracy: 0.8015, 	precision: 0.9990, 	recall: 0.6035, 	specificity: 0.9994, 	f1: 0.7525
Test Epoch 26: 100%|██████████| 1715/1715 [00:36<00:00, 47.40it/s, loss=0.545]
Test Epoch 26 ==> 	accuracy: 0.9265, 	precision: 0.9618, 	recall: 0.6603, 	specificity: 0.9934, 	f1: 0.7830
Train Epoch 27: 100%|██████████| 6195/6195 [05:44<00:00, 17.97it/s, loss=0.532]
Train Epoch 27 ==> 	accuracy: 0.8049, 	precision: 0.9990, 	recall: 0.6105, 	specificity: 0.9994, 	f1: 0.7578
Test Epoch 27: 100%|██████████| 1715/1715 [00:36<00:00, 46.62it/s, loss=0.341]
Test Epoch 27 ==> 	accuracy: 0.9256, 	precision: 0.9833, 	recall: 0.6403, 	specificity: 0.9973, 	f1: 0.7756
Train Epoch 28: 100%|██████████| 6195/6195 [05:41<00:00, 18.16it/s, loss=0.657]
Train Epoch 28 ==> 	accuracy: 0.8048, 	precision: 0.9990, 	recall: 0.6102, 	specificity: 0.9994, 	f1: 0.7577
Test Epoch 28: 100%|██████████| 1715/1715 [00:35<00:00, 48.18it/s, loss=0.436]
Test Epoch 28 ==> 	accuracy: 0.9279, 	precision: 0.9733, 	recall: 0.6591, 	specificity: 0.9955, 	f1: 0.7860
Train Epoch 29: 100%|██████████| 6195/6195 [05:33<00:00, 18.58it/s, loss=0.546]
Train Epoch 29 ==> 	accuracy: 0.8063, 	precision: 0.9990, 	recall: 0.6132, 	specificity: 0.9994, 	f1: 0.7599
Test Epoch 29: 100%|██████████| 1715/1715 [00:34<00:00, 50.00it/s, loss=0.189]
Test Epoch 29 ==> 	accuracy: 0.9282, 	precision: 0.9641, 	recall: 0.6674, 	specificity: 0.9938, 	f1: 0.7887
Train Epoch 30: 100%|██████████| 6195/6195 [05:28<00:00, 18.87it/s, loss=0.538]
Train Epoch 30 ==> 	accuracy: 0.8059, 	precision: 0.9990, 	recall: 0.6123, 	specificity: 0.9994, 	f1: 0.7593
Test Epoch 30: 100%|██████████| 1715/1715 [00:35<00:00, 48.87it/s, loss=0.534]
Test Epoch 30 ==> 	accuracy: 0.9270, 	precision: 0.9622, 	recall: 0.6624, 	specificity: 0.9935, 	f1: 0.7846
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 31: 100%|██████████| 6195/6195 [05:24<00:00, 19.07it/s, loss=0.507]
Train Epoch 31 ==> 	accuracy: 0.8078, 	precision: 0.9991, 	recall: 0.6162, 	specificity: 0.9994, 	f1: 0.7622
Test Epoch 31: 100%|██████████| 1715/1715 [00:34<00:00, 49.31it/s, loss=0.943]
Test Epoch 31 ==> 	accuracy: 0.9299, 	precision: 0.9739, 	recall: 0.6687, 	specificity: 0.9955, 	f1: 0.7930
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 32: 100%|██████████| 6195/6195 [05:35<00:00, 18.46it/s, loss=0.51]
Train Epoch 32 ==> 	accuracy: 0.8081, 	precision: 0.9991, 	recall: 0.6167, 	specificity: 0.9994, 	f1: 0.7627
Test Epoch 32: 100%|██████████| 1715/1715 [00:33<00:00, 51.69it/s, loss=0.373]
Test Epoch 32 ==> 	accuracy: 0.9303, 	precision: 0.9832, 	recall: 0.6641, 	specificity: 0.9971, 	f1: 0.7927
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 33: 100%|██████████| 6195/6195 [05:25<00:00, 19.01it/s, loss=0.469]
Train Epoch 33 ==> 	accuracy: 0.8086, 	precision: 0.9991, 	recall: 0.6177, 	specificity: 0.9995, 	f1: 0.7634
Test Epoch 33: 100%|██████████| 1715/1715 [00:32<00:00, 53.57it/s, loss=0.226]
Test Epoch 33 ==> 	accuracy: 0.9303, 	precision: 0.9766, 	recall: 0.6686, 	specificity: 0.9960, 	f1: 0.7938
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 34: 100%|██████████| 6195/6195 [05:29<00:00, 18.80it/s, loss=0.435]
Train Epoch 34 ==> 	accuracy: 0.8098, 	precision: 0.9992, 	recall: 0.6202, 	specificity: 0.9995, 	f1: 0.7653
Test Epoch 34: 100%|██████████| 1715/1715 [00:34<00:00, 50.16it/s, loss=0.636]
Test Epoch 34 ==> 	accuracy: 0.9302, 	precision: 0.9759, 	recall: 0.6688, 	specificity: 0.9959, 	f1: 0.7937
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 35: 100%|██████████| 6195/6195 [05:25<00:00, 19.06it/s, loss=0.452]
Train Epoch 35 ==> 	accuracy: 0.8125, 	precision: 0.9992, 	recall: 0.6255, 	specificity: 0.9995, 	f1: 0.7694
Test Epoch 35: 100%|██████████| 1715/1715 [00:35<00:00, 48.98it/s, loss=0.849]
Test Epoch 35 ==> 	accuracy: 0.9294, 	precision: 0.9797, 	recall: 0.6618, 	specificity: 0.9966, 	f1: 0.7900
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 36: 100%|██████████| 6195/6195 [05:31<00:00, 18.71it/s, loss=0.415]
Train Epoch 36 ==> 	accuracy: 0.8126, 	precision: 0.9992, 	recall: 0.6257, 	specificity: 0.9995, 	f1: 0.7695
Test Epoch 36: 100%|██████████| 1715/1715 [00:35<00:00, 48.09it/s, loss=0.279]
Test Epoch 36 ==> 	accuracy: 0.9291, 	precision: 0.9670, 	recall: 0.6697, 	specificity: 0.9943, 	f1: 0.7914
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 37: 100%|██████████| 6195/6195 [05:24<00:00, 19.09it/s, loss=0.518]
Train Epoch 37 ==> 	accuracy: 0.8116, 	precision: 0.9992, 	recall: 0.6238, 	specificity: 0.9995, 	f1: 0.7681
Test Epoch 37: 100%|██████████| 1715/1715 [00:33<00:00, 51.35it/s, loss=0.25]
Test Epoch 37 ==> 	accuracy: 0.9296, 	precision: 0.9792, 	recall: 0.6635, 	specificity: 0.9965, 	f1: 0.7910
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 38: 100%|██████████| 6195/6195 [05:25<00:00, 19.06it/s, loss=0.477]
Train Epoch 38 ==> 	accuracy: 0.8170, 	precision: 0.9992, 	recall: 0.6344, 	specificity: 0.9995, 	f1: 0.7761
Test Epoch 38: 100%|██████████| 1715/1715 [00:34<00:00, 49.43it/s, loss=1.35]
Test Epoch 38 ==> 	accuracy: 0.9297, 	precision: 0.9605, 	recall: 0.6775, 	specificity: 0.9930, 	f1: 0.7945
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 39: 100%|██████████| 6195/6195 [05:33<00:00, 18.55it/s, loss=0.553]
Train Epoch 39 ==> 	accuracy: 0.8176, 	precision: 0.9993, 	recall: 0.6357, 	specificity: 0.9995, 	f1: 0.7771
Test Epoch 39: 100%|██████████| 1715/1715 [00:35<00:00, 48.33it/s, loss=0.392]
Test Epoch 39 ==> 	accuracy: 0.9330, 	precision: 0.9773, 	recall: 0.6819, 	specificity: 0.9960, 	f1: 0.8033
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 40: 100%|██████████| 6195/6195 [05:37<00:00, 18.38it/s, loss=0.548]
Train Epoch 40 ==> 	accuracy: 0.8203, 	precision: 0.9993, 	recall: 0.6410, 	specificity: 0.9995, 	f1: 0.7810
Test Epoch 40: 100%|██████████| 1715/1715 [00:33<00:00, 51.02it/s, loss=0.196]
Test Epoch 40 ==> 	accuracy: 0.9318, 	precision: 0.9653, 	recall: 0.6849, 	specificity: 0.9938, 	f1: 0.8013
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 41: 100%|██████████| 6195/6195 [05:31<00:00, 18.70it/s, loss=0.503]
Train Epoch 41 ==> 	accuracy: 0.8173, 	precision: 0.9993, 	recall: 0.6350, 	specificity: 0.9995, 	f1: 0.7766
Test Epoch 41: 100%|██████████| 1715/1715 [00:35<00:00, 48.10it/s, loss=1.13]
Test Epoch 41 ==> 	accuracy: 0.9299, 	precision: 0.9579, 	recall: 0.6806, 	specificity: 0.9925, 	f1: 0.7958
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 42: 100%|██████████| 6195/6195 [05:25<00:00, 19.01it/s, loss=0.428]
Train Epoch 42 ==> 	accuracy: 0.8230, 	precision: 0.9993, 	recall: 0.6465, 	specificity: 0.9996, 	f1: 0.7851
Test Epoch 42: 100%|██████████| 1715/1715 [00:35<00:00, 48.73it/s, loss=2.31]
Test Epoch 42 ==> 	accuracy: 0.9342, 	precision: 0.9633, 	recall: 0.6989, 	specificity: 0.9933, 	f1: 0.8101
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 43: 100%|██████████| 6195/6195 [05:25<00:00, 19.04it/s, loss=0.494]
Train Epoch 43 ==> 	accuracy: 0.8216, 	precision: 0.9994, 	recall: 0.6436, 	specificity: 0.9996, 	f1: 0.7830
Test Epoch 43: 100%|██████████| 1715/1715 [00:34<00:00, 49.15it/s, loss=0.363]
Test Epoch 43 ==> 	accuracy: 0.9347, 	precision: 0.9628, 	recall: 0.7018, 	specificity: 0.9932, 	f1: 0.8119
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 44: 100%|██████████| 6195/6195 [05:29<00:00, 18.78it/s, loss=0.59]
Train Epoch 44 ==> 	accuracy: 0.8218, 	precision: 0.9994, 	recall: 0.6441, 	specificity: 0.9996, 	f1: 0.7833
Test Epoch 44: 100%|██████████| 1715/1715 [00:32<00:00, 52.29it/s, loss=0.25]
Test Epoch 44 ==> 	accuracy: 0.9327, 	precision: 0.9831, 	recall: 0.6763, 	specificity: 0.9971, 	f1: 0.8013
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 45: 100%|██████████| 6195/6195 [05:26<00:00, 18.97it/s, loss=0.474]
Train Epoch 45 ==> 	accuracy: 0.8226, 	precision: 0.9994, 	recall: 0.6456, 	specificity: 0.9996, 	f1: 0.7844
Test Epoch 45: 100%|██████████| 1715/1715 [00:31<00:00, 53.61it/s, loss=0.572]
Test Epoch 45 ==> 	accuracy: 0.9361, 	precision: 0.9727, 	recall: 0.7014, 	specificity: 0.9951, 	f1: 0.8151
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 46: 100%|██████████| 6195/6195 [05:25<00:00, 19.04it/s, loss=0.46]
Train Epoch 46 ==> 	accuracy: 0.8268, 	precision: 0.9993, 	recall: 0.6541, 	specificity: 0.9996, 	f1: 0.7907
Test Epoch 46: 100%|██████████| 1715/1715 [00:33<00:00, 51.73it/s, loss=0.295]
Test Epoch 46 ==> 	accuracy: 0.9299, 	precision: 0.9449, 	recall: 0.6911, 	specificity: 0.9899, 	f1: 0.7983
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 47: 100%|██████████| 6195/6195 [05:34<00:00, 18.54it/s, loss=0.459]
Train Epoch 47 ==> 	accuracy: 0.8259, 	precision: 0.9994, 	recall: 0.6522, 	specificity: 0.9996, 	f1: 0.7893
Test Epoch 47: 100%|██████████| 1715/1715 [00:34<00:00, 49.03it/s, loss=2.63]
Test Epoch 47 ==> 	accuracy: 0.9336, 	precision: 0.9631, 	recall: 0.6962, 	specificity: 0.9933, 	f1: 0.8081
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 48: 100%|██████████| 6195/6195 [05:32<00:00, 18.66it/s, loss=0.45]
Train Epoch 48 ==> 	accuracy: 0.8269, 	precision: 0.9994, 	recall: 0.6542, 	specificity: 0.9996, 	f1: 0.7908
Test Epoch 48: 100%|██████████| 1715/1715 [00:35<00:00, 48.34it/s, loss=0.46]
Test Epoch 48 ==> 	accuracy: 0.9330, 	precision: 0.9555, 	recall: 0.6986, 	specificity: 0.9918, 	f1: 0.8071
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 49: 100%|██████████| 6195/6195 [05:35<00:00, 18.48it/s, loss=0.361]
Train Epoch 49 ==> 	accuracy: 0.8292, 	precision: 0.9994, 	recall: 0.6588, 	specificity: 0.9996, 	f1: 0.7941
Test Epoch 49: 100%|██████████| 1715/1715 [00:33<00:00, 50.68it/s, loss=0.358]
Test Epoch 49 ==> 	accuracy: 0.9342, 	precision: 0.9736, 	recall: 0.6911, 	specificity: 0.9953, 	f1: 0.8084
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 50: 100%|██████████| 6195/6195 [05:28<00:00, 18.86it/s, loss=0.336]
Train Epoch 50 ==> 	accuracy: 0.8310, 	precision: 0.9994, 	recall: 0.6623, 	specificity: 0.9996, 	f1: 0.7967
Test Epoch 50: 100%|██████████| 1715/1715 [00:33<00:00, 51.84it/s, loss=0.664]
Test Epoch 50 ==> 	accuracy: 0.9379, 	precision: 0.9790, 	recall: 0.7058, 	specificity: 0.9962, 	f1: 0.8202
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 51: 100%|██████████| 6195/6195 [05:33<00:00, 18.58it/s, loss=0.415]
Train Epoch 51 ==> 	accuracy: 0.8287, 	precision: 0.9994, 	recall: 0.6577, 	specificity: 0.9996, 	f1: 0.7933
Test Epoch 51: 100%|██████████| 1715/1715 [00:33<00:00, 51.89it/s, loss=1.9]
Test Epoch 51 ==> 	accuracy: 0.9375, 	precision: 0.9825, 	recall: 0.7011, 	specificity: 0.9969, 	f1: 0.8183
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 52: 100%|██████████| 6195/6195 [05:31<00:00, 18.71it/s, loss=0.563]
Train Epoch 52 ==> 	accuracy: 0.8306, 	precision: 0.9994, 	recall: 0.6616, 	specificity: 0.9996, 	f1: 0.7962
Test Epoch 52: 100%|██████████| 1715/1715 [00:32<00:00, 52.30it/s, loss=0.217]
Test Epoch 52 ==> 	accuracy: 0.9395, 	precision: 0.9795, 	recall: 0.7136, 	specificity: 0.9962, 	f1: 0.8257
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 53: 100%|██████████| 6195/6195 [05:25<00:00, 19.05it/s, loss=0.442]
Train Epoch 53 ==> 	accuracy: 0.8304, 	precision: 0.9994, 	recall: 0.6612, 	specificity: 0.9996, 	f1: 0.7959
Test Epoch 53: 100%|██████████| 1715/1715 [00:34<00:00, 49.69it/s, loss=0.147]
Test Epoch 53 ==> 	accuracy: 0.9390, 	precision: 0.9815, 	recall: 0.7093, 	specificity: 0.9967, 	f1: 0.8235
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 54: 100%|██████████| 6195/6195 [05:30<00:00, 18.75it/s, loss=0.473]
Train Epoch 54 ==> 	accuracy: 0.8328, 	precision: 0.9994, 	recall: 0.6659, 	specificity: 0.9996, 	f1: 0.7993
Test Epoch 54: 100%|██████████| 1715/1715 [00:34<00:00, 50.29it/s, loss=0.271]
Test Epoch 54 ==> 	accuracy: 0.9394, 	precision: 0.9784, 	recall: 0.7137, 	specificity: 0.9960, 	f1: 0.8254
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 55: 100%|██████████| 6195/6195 [05:25<00:00, 19.05it/s, loss=0.571]
Train Epoch 55 ==> 	accuracy: 0.8328, 	precision: 0.9995, 	recall: 0.6660, 	specificity: 0.9996, 	f1: 0.7993
Test Epoch 55: 100%|██████████| 1715/1715 [00:34<00:00, 49.28it/s, loss=2.05]
Test Epoch 55 ==> 	accuracy: 0.9390, 	precision: 0.9808, 	recall: 0.7102, 	specificity: 0.9965, 	f1: 0.8239
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 56: 100%|██████████| 6195/6195 [05:38<00:00, 18.31it/s, loss=0.527]
Train Epoch 56 ==> 	accuracy: 0.8363, 	precision: 0.9995, 	recall: 0.6730, 	specificity: 0.9996, 	f1: 0.8044
Test Epoch 56: 100%|██████████| 1715/1715 [00:35<00:00, 48.27it/s, loss=0.205]
Test Epoch 56 ==> 	accuracy: 0.9366, 	precision: 0.9720, 	recall: 0.7046, 	specificity: 0.9949, 	f1: 0.8170
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 57: 100%|██████████| 6195/6195 [05:27<00:00, 18.89it/s, loss=0.388]
Train Epoch 57 ==> 	accuracy: 0.8345, 	precision: 0.9995, 	recall: 0.6693, 	specificity: 0.9997, 	f1: 0.8017
Test Epoch 57: 100%|██████████| 1715/1715 [00:35<00:00, 48.64it/s, loss=0.078]
Test Epoch 57 ==> 	accuracy: 0.9406, 	precision: 0.9794, 	recall: 0.7195, 	specificity: 0.9962, 	f1: 0.8296
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 58: 100%|██████████| 6195/6195 [05:35<00:00, 18.48it/s, loss=0.343]
Train Epoch 58 ==> 	accuracy: 0.8359, 	precision: 0.9995, 	recall: 0.6721, 	specificity: 0.9997, 	f1: 0.8038
Test Epoch 58: 100%|██████████| 1715/1715 [00:33<00:00, 51.95it/s, loss=0.371]
Test Epoch 58 ==> 	accuracy: 0.9398, 	precision: 0.9779, 	recall: 0.7161, 	specificity: 0.9959, 	f1: 0.8268
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 59: 100%|██████████| 6195/6195 [05:33<00:00, 18.56it/s, loss=0.495]
Train Epoch 59 ==> 	accuracy: 0.8368, 	precision: 0.9995, 	recall: 0.6740, 	specificity: 0.9997, 	f1: 0.8051
Test Epoch 59: 100%|██████████| 1715/1715 [00:34<00:00, 49.98it/s, loss=0.131]
Test Epoch 59 ==> 	accuracy: 0.9404, 	precision: 0.9793, 	recall: 0.7182, 	specificity: 0.9962, 	f1: 0.8287
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 60: 100%|██████████| 6195/6195 [05:41<00:00, 18.11it/s, loss=0.476]
Train Epoch 60 ==> 	accuracy: 0.8392, 	precision: 0.9995, 	recall: 0.6787, 	specificity: 0.9997, 	f1: 0.8085
Test Epoch 60: 100%|██████████| 1715/1715 [00:32<00:00, 52.97it/s, loss=0.37]
Test Epoch 60 ==> 	accuracy: 0.9397, 	precision: 0.9799, 	recall: 0.7142, 	specificity: 0.9963, 	f1: 0.8262
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 61: 100%|██████████| 6195/6195 [05:37<00:00, 18.36it/s, loss=0.419]
Train Epoch 61 ==> 	accuracy: 0.8376, 	precision: 0.9995, 	recall: 0.6754, 	specificity: 0.9997, 	f1: 0.8061
Test Epoch 61: 100%|██████████| 1715/1715 [00:32<00:00, 52.04it/s, loss=0.651]
Test Epoch 61 ==> 	accuracy: 0.9408, 	precision: 0.9817, 	recall: 0.7183, 	specificity: 0.9966, 	f1: 0.8296
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 62: 100%|██████████| 6195/6195 [05:36<00:00, 18.43it/s, loss=0.501]
Train Epoch 62 ==> 	accuracy: 0.8376, 	precision: 0.9995, 	recall: 0.6755, 	specificity: 0.9997, 	f1: 0.8062
Test Epoch 62: 100%|██████████| 1715/1715 [00:35<00:00, 47.81it/s, loss=2.32]
Test Epoch 62 ==> 	accuracy: 0.9417, 	precision: 0.9707, 	recall: 0.7315, 	specificity: 0.9945, 	f1: 0.8343
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 63: 100%|██████████| 6195/6195 [05:40<00:00, 18.17it/s, loss=0.376]
Train Epoch 63 ==> 	accuracy: 0.8420, 	precision: 0.9995, 	recall: 0.6844, 	specificity: 0.9997, 	f1: 0.8124
Test Epoch 63: 100%|██████████| 1715/1715 [00:35<00:00, 48.26it/s, loss=0.603]
Test Epoch 63 ==> 	accuracy: 0.9406, 	precision: 0.9746, 	recall: 0.7228, 	specificity: 0.9953, 	f1: 0.8300
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 64: 100%|██████████| 6195/6195 [05:40<00:00, 18.20it/s, loss=0.538]
Train Epoch 64 ==> 	accuracy: 0.8412, 	precision: 0.9996, 	recall: 0.6827, 	specificity: 0.9997, 	f1: 0.8113
Test Epoch 64: 100%|██████████| 1715/1715 [00:35<00:00, 48.40it/s, loss=1.48]
Test Epoch 64 ==> 	accuracy: 0.9419, 	precision: 0.9719, 	recall: 0.7319, 	specificity: 0.9947, 	f1: 0.8350
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 65: 100%|██████████| 6195/6195 [05:40<00:00, 18.21it/s, loss=0.383]
Train Epoch 65 ==> 	accuracy: 0.8409, 	precision: 0.9995, 	recall: 0.6822, 	specificity: 0.9997, 	f1: 0.8109
Test Epoch 65: 100%|██████████| 1715/1715 [00:34<00:00, 50.22it/s, loss=0.245]
Test Epoch 65 ==> 	accuracy: 0.9392, 	precision: 0.9823, 	recall: 0.7099, 	specificity: 0.9968, 	f1: 0.8242
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 66: 100%|██████████| 6195/6195 [05:27<00:00, 18.91it/s, loss=0.417]
Train Epoch 66 ==> 	accuracy: 0.8409, 	precision: 0.9995, 	recall: 0.6822, 	specificity: 0.9997, 	f1: 0.8109
Test Epoch 66: 100%|██████████| 1715/1715 [00:35<00:00, 48.49it/s, loss=1.13]
Test Epoch 66 ==> 	accuracy: 0.9426, 	precision: 0.9756, 	recall: 0.7323, 	specificity: 0.9954, 	f1: 0.8366
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 67: 100%|██████████| 6195/6195 [05:31<00:00, 18.67it/s, loss=0.37]
Train Epoch 67 ==> 	accuracy: 0.8443, 	precision: 0.9995, 	recall: 0.6889, 	specificity: 0.9997, 	f1: 0.8156
Test Epoch 67: 100%|██████████| 1715/1715 [00:35<00:00, 48.28it/s, loss=1.33]
Test Epoch 67 ==> 	accuracy: 0.9414, 	precision: 0.9763, 	recall: 0.7258, 	specificity: 0.9956, 	f1: 0.8326
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 68: 100%|██████████| 6195/6195 [05:29<00:00, 18.79it/s, loss=0.455]
Train Epoch 68 ==> 	accuracy: 0.8423, 	precision: 0.9996, 	recall: 0.6849, 	specificity: 0.9997, 	f1: 0.8129
Test Epoch 68: 100%|██████████| 1715/1715 [00:35<00:00, 47.73it/s, loss=0.968]
Test Epoch 68 ==> 	accuracy: 0.9424, 	precision: 0.9777, 	recall: 0.7297, 	specificity: 0.9958, 	f1: 0.8357
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 69: 100%|██████████| 6195/6195 [05:34<00:00, 18.54it/s, loss=0.43]
Train Epoch 69 ==> 	accuracy: 0.8444, 	precision: 0.9996, 	recall: 0.6892, 	specificity: 0.9997, 	f1: 0.8158
Test Epoch 69: 100%|██████████| 1715/1715 [00:31<00:00, 54.39it/s, loss=0.287]
Test Epoch 69 ==> 	accuracy: 0.9422, 	precision: 0.9673, 	recall: 0.7370, 	specificity: 0.9937, 	f1: 0.8366
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 70: 100%|██████████| 6195/6195 [05:32<00:00, 18.64it/s, loss=0.485]
Train Epoch 70 ==> 	accuracy: 0.8446, 	precision: 0.9996, 	recall: 0.6895, 	specificity: 0.9997, 	f1: 0.8161
Test Epoch 70: 100%|██████████| 1715/1715 [00:35<00:00, 48.83it/s, loss=0.172]
Test Epoch 70 ==> 	accuracy: 0.9436, 	precision: 0.9767, 	recall: 0.7367, 	specificity: 0.9956, 	f1: 0.8399
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 71: 100%|██████████| 6195/6195 [05:27<00:00, 18.92it/s, loss=0.475]
Train Epoch 71 ==> 	accuracy: 0.8453, 	precision: 0.9996, 	recall: 0.6910, 	specificity: 0.9997, 	f1: 0.8171
Test Epoch 71: 100%|██████████| 1715/1715 [00:34<00:00, 50.27it/s, loss=0.397]
Test Epoch 71 ==> 	accuracy: 0.9428, 	precision: 0.9785, 	recall: 0.7309, 	specificity: 0.9960, 	f1: 0.8368
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 72: 100%|██████████| 6195/6195 [05:40<00:00, 18.22it/s, loss=0.457]
Train Epoch 72 ==> 	accuracy: 0.8451, 	precision: 0.9996, 	recall: 0.6905, 	specificity: 0.9997, 	f1: 0.8168
Test Epoch 72: 100%|██████████| 1715/1715 [00:35<00:00, 48.65it/s, loss=4.4]
Test Epoch 72 ==> 	accuracy: 0.9436, 	precision: 0.9759, 	recall: 0.7374, 	specificity: 0.9954, 	f1: 0.8401
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 73: 100%|██████████| 6195/6195 [05:26<00:00, 18.95it/s, loss=0.404]
Train Epoch 73 ==> 	accuracy: 0.8467, 	precision: 0.9996, 	recall: 0.6938, 	specificity: 0.9997, 	f1: 0.8191
Test Epoch 73: 100%|██████████| 1715/1715 [00:34<00:00, 50.03it/s, loss=2.78]
Test Epoch 73 ==> 	accuracy: 0.9447, 	precision: 0.9710, 	recall: 0.7467, 	specificity: 0.9944, 	f1: 0.8442
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 74: 100%|██████████| 6195/6195 [05:44<00:00, 17.96it/s, loss=0.481]
Train Epoch 74 ==> 	accuracy: 0.8452, 	precision: 0.9996, 	recall: 0.6907, 	specificity: 0.9997, 	f1: 0.8169
Test Epoch 74: 100%|██████████| 1715/1715 [00:35<00:00, 48.44it/s, loss=0.563]
Test Epoch 74 ==> 	accuracy: 0.9397, 	precision: 0.9785, 	recall: 0.7153, 	specificity: 0.9961, 	f1: 0.8265
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 75: 100%|██████████| 6195/6195 [05:35<00:00, 18.46it/s, loss=0.505]
Train Epoch 75 ==> 	accuracy: 0.8466, 	precision: 0.9996, 	recall: 0.6935, 	specificity: 0.9997, 	f1: 0.8189
Test Epoch 75: 100%|██████████| 1715/1715 [00:35<00:00, 47.84it/s, loss=0.338]
Test Epoch 75 ==> 	accuracy: 0.9429, 	precision: 0.9765, 	recall: 0.7332, 	specificity: 0.9956, 	f1: 0.8375
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 76: 100%|██████████| 6195/6195 [05:31<00:00, 18.71it/s, loss=0.422]
Train Epoch 76 ==> 	accuracy: 0.8496, 	precision: 0.9996, 	recall: 0.6995, 	specificity: 0.9997, 	f1: 0.8230
Test Epoch 76: 100%|██████████| 1715/1715 [00:36<00:00, 46.91it/s, loss=1.33]
Test Epoch 76 ==> 	accuracy: 0.9454, 	precision: 0.9736, 	recall: 0.7481, 	specificity: 0.9949, 	f1: 0.8461
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 77: 100%|██████████| 6195/6195 [05:24<00:00, 19.06it/s, loss=0.372]
Train Epoch 77 ==> 	accuracy: 0.8490, 	precision: 0.9995, 	recall: 0.6983, 	specificity: 0.9997, 	f1: 0.8222
Test Epoch 77: 100%|██████████| 1715/1715 [00:35<00:00, 47.91it/s, loss=0.424]
Test Epoch 77 ==> 	accuracy: 0.9444, 	precision: 0.9719, 	recall: 0.7447, 	specificity: 0.9946, 	f1: 0.8432
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 78: 100%|██████████| 6195/6195 [05:32<00:00, 18.61it/s, loss=0.407]
Train Epoch 78 ==> 	accuracy: 0.8484, 	precision: 0.9996, 	recall: 0.6972, 	specificity: 0.9997, 	f1: 0.8214
Test Epoch 78: 100%|██████████| 1715/1715 [00:34<00:00, 49.52it/s, loss=1.14]
Test Epoch 78 ==> 	accuracy: 0.9437, 	precision: 0.9708, 	recall: 0.7419, 	specificity: 0.9944, 	f1: 0.8410
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 79: 100%|██████████| 6195/6195 [05:32<00:00, 18.62it/s, loss=0.448]
Train Epoch 79 ==> 	accuracy: 0.8500, 	precision: 0.9996, 	recall: 0.7004, 	specificity: 0.9997, 	f1: 0.8236
Test Epoch 79: 100%|██████████| 1715/1715 [00:33<00:00, 51.64it/s, loss=0.358]
Test Epoch 79 ==> 	accuracy: 0.9434, 	precision: 0.9540, 	recall: 0.7547, 	specificity: 0.9909, 	f1: 0.8427
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 80: 100%|██████████| 6195/6195 [05:28<00:00, 18.87it/s, loss=0.358]
Train Epoch 80 ==> 	accuracy: 0.8504, 	precision: 0.9996, 	recall: 0.7010, 	specificity: 0.9997, 	f1: 0.8241
Test Epoch 80: 100%|██████████| 1715/1715 [00:36<00:00, 46.59it/s, loss=0.359]
Test Epoch 80 ==> 	accuracy: 0.9447, 	precision: 0.9592, 	recall: 0.7565, 	specificity: 0.9919, 	f1: 0.8459
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 81: 100%|██████████| 6195/6195 [05:31<00:00, 18.67it/s, loss=0.523]
Train Epoch 81 ==> 	accuracy: 0.8506, 	precision: 0.9996, 	recall: 0.7015, 	specificity: 0.9997, 	f1: 0.8245
Test Epoch 81: 100%|██████████| 1715/1715 [00:33<00:00, 50.51it/s, loss=1.22]
Test Epoch 81 ==> 	accuracy: 0.9451, 	precision: 0.9701, 	recall: 0.7494, 	specificity: 0.9942, 	f1: 0.8456
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 82: 100%|██████████| 6195/6195 [05:24<00:00, 19.06it/s, loss=0.396]
Train Epoch 82 ==> 	accuracy: 0.8513, 	precision: 0.9996, 	recall: 0.7028, 	specificity: 0.9997, 	f1: 0.8253
Test Epoch 82: 100%|██████████| 1715/1715 [00:35<00:00, 48.67it/s, loss=1.01]
Test Epoch 82 ==> 	accuracy: 0.9451, 	precision: 0.9666, 	recall: 0.7526, 	specificity: 0.9935, 	f1: 0.8463
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 83: 100%|██████████| 6195/6195 [05:19<00:00, 19.41it/s, loss=0.397]
Train Epoch 83 ==> 	accuracy: 0.8513, 	precision: 0.9996, 	recall: 0.7028, 	specificity: 0.9997, 	f1: 0.8253
Test Epoch 83: 100%|██████████| 1715/1715 [00:34<00:00, 49.01it/s, loss=1.98]
Test Epoch 83 ==> 	accuracy: 0.9448, 	precision: 0.9597, 	recall: 0.7570, 	specificity: 0.9920, 	f1: 0.8464
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 84: 100%|██████████| 6195/6195 [05:33<00:00, 18.60it/s, loss=0.403]
Train Epoch 84 ==> 	accuracy: 0.8531, 	precision: 0.9996, 	recall: 0.7065, 	specificity: 0.9997, 	f1: 0.8279
Test Epoch 84: 100%|██████████| 1715/1715 [00:35<00:00, 48.75it/s, loss=0.257]
Test Epoch 84 ==> 	accuracy: 0.9455, 	precision: 0.9577, 	recall: 0.7622, 	specificity: 0.9916, 	f1: 0.8488
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 85: 100%|██████████| 6195/6195 [05:33<00:00, 18.60it/s, loss=0.307]
Train Epoch 85 ==> 	accuracy: 0.8524, 	precision: 0.9996, 	recall: 0.7050, 	specificity: 0.9997, 	f1: 0.8269
Test Epoch 85: 100%|██████████| 1715/1715 [00:37<00:00, 45.94it/s, loss=0.221]
Test Epoch 85 ==> 	accuracy: 0.9441, 	precision: 0.9573, 	recall: 0.7552, 	specificity: 0.9915, 	f1: 0.8443
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 86: 100%|██████████| 6195/6195 [05:28<00:00, 18.87it/s, loss=0.366]
Train Epoch 86 ==> 	accuracy: 0.8545, 	precision: 0.9996, 	recall: 0.7092, 	specificity: 0.9997, 	f1: 0.8297
Test Epoch 86: 100%|██████████| 1715/1715 [00:35<00:00, 47.95it/s, loss=0.341]
Test Epoch 86 ==> 	accuracy: 0.9452, 	precision: 0.9651, 	recall: 0.7541, 	specificity: 0.9932, 	f1: 0.8467
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 87: 100%|██████████| 6195/6195 [05:30<00:00, 18.73it/s, loss=0.437]
Train Epoch 87 ==> 	accuracy: 0.8525, 	precision: 0.9996, 	recall: 0.7052, 	specificity: 0.9997, 	f1: 0.8270
Test Epoch 87: 100%|██████████| 1715/1715 [00:35<00:00, 48.32it/s, loss=0.357]
Test Epoch 87 ==> 	accuracy: 0.9446, 	precision: 0.9654, 	recall: 0.7512, 	specificity: 0.9932, 	f1: 0.8449
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 88: 100%|██████████| 6195/6195 [05:33<00:00, 18.60it/s, loss=0.438]
Train Epoch 88 ==> 	accuracy: 0.8552, 	precision: 0.9996, 	recall: 0.7106, 	specificity: 0.9997, 	f1: 0.8307
Test Epoch 88: 100%|██████████| 1715/1715 [00:34<00:00, 49.53it/s, loss=1.2]
Test Epoch 88 ==> 	accuracy: 0.9449, 	precision: 0.9590, 	recall: 0.7582, 	specificity: 0.9918, 	f1: 0.8469
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 89: 100%|██████████| 6195/6195 [05:35<00:00, 18.45it/s, loss=0.413]
Train Epoch 89 ==> 	accuracy: 0.8544, 	precision: 0.9996, 	recall: 0.7092, 	specificity: 0.9997, 	f1: 0.8297
Test Epoch 89: 100%|██████████| 1715/1715 [00:34<00:00, 50.33it/s, loss=2.77]
Test Epoch 89 ==> 	accuracy: 0.9449, 	precision: 0.9558, 	recall: 0.7607, 	specificity: 0.9912, 	f1: 0.8472
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 90: 100%|██████████| 6195/6195 [05:30<00:00, 18.72it/s, loss=0.513]
Train Epoch 90 ==> 	accuracy: 0.8545, 	precision: 0.9996, 	recall: 0.7093, 	specificity: 0.9997, 	f1: 0.8298
Test Epoch 90: 100%|██████████| 1715/1715 [00:32<00:00, 52.71it/s, loss=0.377]
Test Epoch 90 ==> 	accuracy: 0.9455, 	precision: 0.9656, 	recall: 0.7554, 	specificity: 0.9933, 	f1: 0.8477
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 91: 100%|██████████| 6195/6195 [05:31<00:00, 18.69it/s, loss=0.516]
Train Epoch 91 ==> 	accuracy: 0.8545, 	precision: 0.9996, 	recall: 0.7092, 	specificity: 0.9997, 	f1: 0.8297
Test Epoch 91: 100%|██████████| 1715/1715 [00:35<00:00, 48.89it/s, loss=1.86]
Test Epoch 91 ==> 	accuracy: 0.9458, 	precision: 0.9626, 	recall: 0.7595, 	specificity: 0.9926, 	f1: 0.8491
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 92: 100%|██████████| 6195/6195 [05:26<00:00, 18.97it/s, loss=0.377]
Train Epoch 92 ==> 	accuracy: 0.8559, 	precision: 0.9996, 	recall: 0.7121, 	specificity: 0.9997, 	f1: 0.8317
Test Epoch 92: 100%|██████████| 1715/1715 [00:32<00:00, 53.26it/s, loss=0.28]
Test Epoch 92 ==> 	accuracy: 0.9460, 	precision: 0.9602, 	recall: 0.7628, 	specificity: 0.9921, 	f1: 0.8502
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 93: 100%|██████████| 6195/6195 [05:27<00:00, 18.89it/s, loss=0.371]
Train Epoch 93 ==> 	accuracy: 0.8559, 	precision: 0.9996, 	recall: 0.7120, 	specificity: 0.9997, 	f1: 0.8317
Test Epoch 93: 100%|██████████| 1715/1715 [00:34<00:00, 49.24it/s, loss=0.606]
Test Epoch 93 ==> 	accuracy: 0.9452, 	precision: 0.9676, 	recall: 0.7525, 	specificity: 0.9937, 	f1: 0.8466
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 94: 100%|██████████| 6195/6195 [05:33<00:00, 18.56it/s, loss=0.503]
Train Epoch 94 ==> 	accuracy: 0.8552, 	precision: 0.9997, 	recall: 0.7106, 	specificity: 0.9998, 	f1: 0.8307
Test Epoch 94: 100%|██████████| 1715/1715 [00:33<00:00, 51.17it/s, loss=0.966]
Test Epoch 94 ==> 	accuracy: 0.9461, 	precision: 0.9585, 	recall: 0.7648, 	specificity: 0.9917, 	f1: 0.8508
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 95: 100%|██████████| 6195/6195 [05:34<00:00, 18.50it/s, loss=0.412]
Train Epoch 95 ==> 	accuracy: 0.8553, 	precision: 0.9996, 	recall: 0.7109, 	specificity: 0.9997, 	f1: 0.8309
Test Epoch 95: 100%|██████████| 1715/1715 [00:32<00:00, 52.95it/s, loss=3.01]
Test Epoch 95 ==> 	accuracy: 0.9459, 	precision: 0.9653, 	recall: 0.7578, 	specificity: 0.9932, 	f1: 0.8491
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 96: 100%|██████████| 6195/6195 [05:36<00:00, 18.40it/s, loss=0.526]
Train Epoch 96 ==> 	accuracy: 0.8564, 	precision: 0.9996, 	recall: 0.7130, 	specificity: 0.9997, 	f1: 0.8323
Test Epoch 96: 100%|██████████| 1715/1715 [00:34<00:00, 49.95it/s, loss=0.507]
Test Epoch 96 ==> 	accuracy: 0.9453, 	precision: 0.9546, 	recall: 0.7638, 	specificity: 0.9909, 	f1: 0.8486
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 97: 100%|██████████| 6195/6195 [05:42<00:00, 18.07it/s, loss=0.442]
Train Epoch 97 ==> 	accuracy: 0.8576, 	precision: 0.9997, 	recall: 0.7154, 	specificity: 0.9998, 	f1: 0.8340
Test Epoch 97: 100%|██████████| 1715/1715 [00:34<00:00, 49.32it/s, loss=0.273]
Test Epoch 97 ==> 	accuracy: 0.9447, 	precision: 0.9510, 	recall: 0.7641, 	specificity: 0.9901, 	f1: 0.8474
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 98: 100%|██████████| 6195/6195 [05:41<00:00, 18.12it/s, loss=0.421]
Train Epoch 98 ==> 	accuracy: 0.8586, 	precision: 0.9996, 	recall: 0.7175, 	specificity: 0.9997, 	f1: 0.8354
Test Epoch 98: 100%|██████████| 1715/1715 [00:34<00:00, 49.72it/s, loss=0.31]
Test Epoch 98 ==> 	accuracy: 0.9461, 	precision: 0.9579, 	recall: 0.7650, 	specificity: 0.9916, 	f1: 0.8507
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 99: 100%|██████████| 6195/6195 [05:43<00:00, 18.03it/s, loss=0.423]
Train Epoch 99 ==> 	accuracy: 0.8571, 	precision: 0.9996, 	recall: 0.7146, 	specificity: 0.9997, 	f1: 0.8334
Test Epoch 99: 100%|██████████| 1715/1715 [00:36<00:00, 47.57it/s, loss=0.421]
Test Epoch 99 ==> 	accuracy: 0.9465, 	precision: 0.9608, 	recall: 0.7647, 	specificity: 0.9922, 	f1: 0.8516
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 100: 100%|██████████| 6195/6195 [05:31<00:00, 18.68it/s, loss=0.498]
Train Epoch 100 ==> 	accuracy: 0.8579, 	precision: 0.9996, 	recall: 0.7161, 	specificity: 0.9997, 	f1: 0.8344
Test Epoch 100: 100%|██████████| 1715/1715 [00:33<00:00, 51.89it/s, loss=1]
Test Epoch 100 ==> 	accuracy: 0.9452, 	precision: 0.9607, 	recall: 0.7581, 	specificity: 0.9922, 	f1: 0.8474
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 101: 100%|██████████| 6195/6195 [05:33<00:00, 18.55it/s, loss=0.631]
Train Epoch 101 ==> 	accuracy: 0.8572, 	precision: 0.9996, 	recall: 0.7147, 	specificity: 0.9997, 	f1: 0.8335
Test Epoch 101: 100%|██████████| 1715/1715 [00:35<00:00, 48.17it/s, loss=2.57]
Test Epoch 101 ==> 	accuracy: 0.9457, 	precision: 0.9610, 	recall: 0.7602, 	specificity: 0.9922, 	f1: 0.8489
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 102: 100%|██████████| 6195/6195 [05:33<00:00, 18.60it/s, loss=0.398]
Train Epoch 102 ==> 	accuracy: 0.8604, 	precision: 0.9996, 	recall: 0.7210, 	specificity: 0.9997, 	f1: 0.8378
Test Epoch 102: 100%|██████████| 1715/1715 [00:33<00:00, 51.45it/s, loss=0.252]
Test Epoch 102 ==> 	accuracy: 0.9438, 	precision: 0.9589, 	recall: 0.7524, 	specificity: 0.9919, 	f1: 0.8432
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 103: 100%|██████████| 6195/6195 [05:31<00:00, 18.68it/s, loss=0.473]
Train Epoch 103 ==> 	accuracy: 0.8589, 	precision: 0.9996, 	recall: 0.7180, 	specificity: 0.9997, 	f1: 0.8357
Test Epoch 103: 100%|██████████| 1715/1715 [00:33<00:00, 51.73it/s, loss=2.15]
Test Epoch 103 ==> 	accuracy: 0.9466, 	precision: 0.9537, 	recall: 0.7714, 	specificity: 0.9906, 	f1: 0.8529
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 104: 100%|██████████| 6195/6195 [05:31<00:00, 18.66it/s, loss=0.384]
Train Epoch 104 ==> 	accuracy: 0.8577, 	precision: 0.9996, 	recall: 0.7156, 	specificity: 0.9997, 	f1: 0.8341
Test Epoch 104: 100%|██████████| 1715/1715 [00:35<00:00, 47.99it/s, loss=1.04]
Test Epoch 104 ==> 	accuracy: 0.9473, 	precision: 0.9672, 	recall: 0.7631, 	specificity: 0.9935, 	f1: 0.8531
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 105: 100%|██████████| 6195/6195 [05:32<00:00, 18.62it/s, loss=0.339]
Train Epoch 105 ==> 	accuracy: 0.8590, 	precision: 0.9996, 	recall: 0.7182, 	specificity: 0.9997, 	f1: 0.8359
Test Epoch 105: 100%|██████████| 1715/1715 [00:32<00:00, 52.03it/s, loss=0.173]
Test Epoch 105 ==> 	accuracy: 0.9460, 	precision: 0.9601, 	recall: 0.7628, 	specificity: 0.9920, 	f1: 0.8502
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 106: 100%|██████████| 6195/6195 [05:33<00:00, 18.59it/s, loss=0.31]
Train Epoch 106 ==> 	accuracy: 0.8603, 	precision: 0.9997, 	recall: 0.7208, 	specificity: 0.9998, 	f1: 0.8376
Test Epoch 106: 100%|██████████| 1715/1715 [00:32<00:00, 52.92it/s, loss=2.56]
Test Epoch 106 ==> 	accuracy: 0.9451, 	precision: 0.9631, 	recall: 0.7556, 	specificity: 0.9927, 	f1: 0.8469
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 107: 100%|██████████| 6195/6195 [05:28<00:00, 18.87it/s, loss=0.342]
Train Epoch 107 ==> 	accuracy: 0.8597, 	precision: 0.9996, 	recall: 0.7196, 	specificity: 0.9997, 	f1: 0.8368
Test Epoch 107: 100%|██████████| 1715/1715 [00:34<00:00, 49.18it/s, loss=0.394]
Test Epoch 107 ==> 	accuracy: 0.9470, 	precision: 0.9583, 	recall: 0.7697, 	specificity: 0.9916, 	f1: 0.8537
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 108: 100%|██████████| 6195/6195 [05:36<00:00, 18.43it/s, loss=0.487]
Train Epoch 108 ==> 	accuracy: 0.8582, 	precision: 0.9997, 	recall: 0.7166, 	specificity: 0.9998, 	f1: 0.8348
Test Epoch 108: 100%|██████████| 1715/1715 [00:37<00:00, 46.19it/s, loss=3.99]
Test Epoch 108 ==> 	accuracy: 0.9456, 	precision: 0.9598, 	recall: 0.7607, 	specificity: 0.9920, 	f1: 0.8487
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 109: 100%|██████████| 6195/6195 [05:34<00:00, 18.53it/s, loss=0.422]
Train Epoch 109 ==> 	accuracy: 0.8619, 	precision: 0.9997, 	recall: 0.7240, 	specificity: 0.9998, 	f1: 0.8398
Test Epoch 109: 100%|██████████| 1715/1715 [00:34<00:00, 49.73it/s, loss=0.649]
Test Epoch 109 ==> 	accuracy: 0.9464, 	precision: 0.9527, 	recall: 0.7716, 	specificity: 0.9904, 	f1: 0.8526
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 110: 100%|██████████| 6195/6195 [05:29<00:00, 18.81it/s, loss=0.333]
Train Epoch 110 ==> 	accuracy: 0.8605, 	precision: 0.9996, 	recall: 0.7213, 	specificity: 0.9997, 	f1: 0.8380
Test Epoch 110: 100%|██████████| 1715/1715 [00:34<00:00, 49.62it/s, loss=0.378]
Test Epoch 110 ==> 	accuracy: 0.9469, 	precision: 0.9569, 	recall: 0.7705, 	specificity: 0.9913, 	f1: 0.8536
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 111: 100%|██████████| 6195/6195 [05:29<00:00, 18.82it/s, loss=0.536]
Train Epoch 111 ==> 	accuracy: 0.8614, 	precision: 0.9997, 	recall: 0.7231, 	specificity: 0.9998, 	f1: 0.8392
Test Epoch 111: 100%|██████████| 1715/1715 [00:35<00:00, 48.44it/s, loss=0.543]
Test Epoch 111 ==> 	accuracy: 0.9472, 	precision: 0.9556, 	recall: 0.7729, 	specificity: 0.9910, 	f1: 0.8546
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 112: 100%|██████████| 6195/6195 [05:29<00:00, 18.79it/s, loss=0.456]
Train Epoch 112 ==> 	accuracy: 0.8609, 	precision: 0.9996, 	recall: 0.7220, 	specificity: 0.9997, 	f1: 0.8384
Test Epoch 112: 100%|██████████| 1715/1715 [00:32<00:00, 52.47it/s, loss=0.355]
Test Epoch 112 ==> 	accuracy: 0.9451, 	precision: 0.9438, 	recall: 0.7724, 	specificity: 0.9885, 	f1: 0.8496
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 113: 100%|██████████| 6195/6195 [05:34<00:00, 18.53it/s, loss=0.386]
Train Epoch 113 ==> 	accuracy: 0.8628, 	precision: 0.9996, 	recall: 0.7259, 	specificity: 0.9997, 	f1: 0.8411
Test Epoch 113: 100%|██████████| 1715/1715 [00:35<00:00, 47.73it/s, loss=1.16]
Test Epoch 113 ==> 	accuracy: 0.9454, 	precision: 0.9422, 	recall: 0.7756, 	specificity: 0.9880, 	f1: 0.8508
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 114: 100%|██████████| 6195/6195 [05:28<00:00, 18.86it/s, loss=0.399]
Train Epoch 114 ==> 	accuracy: 0.8607, 	precision: 0.9997, 	recall: 0.7217, 	specificity: 0.9998, 	f1: 0.8382
Test Epoch 114: 100%|██████████| 1715/1715 [00:35<00:00, 48.53it/s, loss=0.216]
Test Epoch 114 ==> 	accuracy: 0.9476, 	precision: 0.9628, 	recall: 0.7685, 	specificity: 0.9925, 	f1: 0.8548
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 115: 100%|██████████| 6195/6195 [05:30<00:00, 18.72it/s, loss=0.449]
Train Epoch 115 ==> 	accuracy: 0.8621, 	precision: 0.9996, 	recall: 0.7244, 	specificity: 0.9997, 	f1: 0.8401
Test Epoch 115: 100%|██████████| 1715/1715 [00:34<00:00, 50.29it/s, loss=0.457]
Test Epoch 115 ==> 	accuracy: 0.9472, 	precision: 0.9547, 	recall: 0.7738, 	specificity: 0.9908, 	f1: 0.8548
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 116: 100%|██████████| 6195/6195 [05:30<00:00, 18.72it/s, loss=0.399]
Train Epoch 116 ==> 	accuracy: 0.8613, 	precision: 0.9997, 	recall: 0.7228, 	specificity: 0.9998, 	f1: 0.8390
Test Epoch 116: 100%|██████████| 1715/1715 [00:36<00:00, 47.05it/s, loss=0.0837]
Test Epoch 116 ==> 	accuracy: 0.9484, 	precision: 0.9588, 	recall: 0.7762, 	specificity: 0.9916, 	f1: 0.8579
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 117: 100%|██████████| 6195/6195 [05:39<00:00, 18.24it/s, loss=0.398]
Train Epoch 117 ==> 	accuracy: 0.8615, 	precision: 0.9997, 	recall: 0.7233, 	specificity: 0.9998, 	f1: 0.8393
Test Epoch 117: 100%|██████████| 1715/1715 [00:33<00:00, 50.82it/s, loss=1.47]
Test Epoch 117 ==> 	accuracy: 0.9470, 	precision: 0.9607, 	recall: 0.7672, 	specificity: 0.9921, 	f1: 0.8531
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 118: 100%|██████████| 6195/6195 [05:29<00:00, 18.79it/s, loss=0.483]
Train Epoch 118 ==> 	accuracy: 0.8621, 	precision: 0.9997, 	recall: 0.7244, 	specificity: 0.9997, 	f1: 0.8401
Test Epoch 118: 100%|██████████| 1715/1715 [00:35<00:00, 48.51it/s, loss=0.375]
Test Epoch 118 ==> 	accuracy: 0.9471, 	precision: 0.9542, 	recall: 0.7735, 	specificity: 0.9907, 	f1: 0.8544
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 119: 100%|██████████| 6195/6195 [05:30<00:00, 18.72it/s, loss=0.419]
Train Epoch 119 ==> 	accuracy: 0.8636, 	precision: 0.9997, 	recall: 0.7274, 	specificity: 0.9998, 	f1: 0.8421
Test Epoch 119: 100%|██████████| 1715/1715 [00:33<00:00, 50.46it/s, loss=0.4]
Test Epoch 119 ==> 	accuracy: 0.9476, 	precision: 0.9585, 	recall: 0.7722, 	specificity: 0.9916, 	f1: 0.8554
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 120: 100%|██████████| 6195/6195 [05:26<00:00, 19.00it/s, loss=0.395]
Train Epoch 120 ==> 	accuracy: 0.8626, 	precision: 0.9996, 	recall: 0.7254, 	specificity: 0.9997, 	f1: 0.8407
Test Epoch 120: 100%|██████████| 1715/1715 [00:32<00:00, 52.17it/s, loss=2.39]
Test Epoch 120 ==> 	accuracy: 0.9472, 	precision: 0.9564, 	recall: 0.7720, 	specificity: 0.9912, 	f1: 0.8544
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 121: 100%|██████████| 6195/6195 [05:29<00:00, 18.81it/s, loss=0.337]
Train Epoch 121 ==> 	accuracy: 0.8634, 	precision: 0.9997, 	recall: 0.7270, 	specificity: 0.9998, 	f1: 0.8418
Test Epoch 121: 100%|██████████| 1715/1715 [00:33<00:00, 51.02it/s, loss=1.68]
Test Epoch 121 ==> 	accuracy: 0.9466, 	precision: 0.9553, 	recall: 0.7703, 	specificity: 0.9909, 	f1: 0.8529
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 122: 100%|██████████| 6195/6195 [05:26<00:00, 18.95it/s, loss=0.378]
Train Epoch 122 ==> 	accuracy: 0.8651, 	precision: 0.9996, 	recall: 0.7304, 	specificity: 0.9997, 	f1: 0.8440
Test Epoch 122: 100%|██████████| 1715/1715 [00:33<00:00, 51.45it/s, loss=0.323]
Test Epoch 122 ==> 	accuracy: 0.9478, 	precision: 0.9498, 	recall: 0.7814, 	specificity: 0.9896, 	f1: 0.8574
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 123: 100%|██████████| 6195/6195 [05:27<00:00, 18.94it/s, loss=0.41]
Train Epoch 123 ==> 	accuracy: 0.8631, 	precision: 0.9996, 	recall: 0.7265, 	specificity: 0.9997, 	f1: 0.8415
Test Epoch 123: 100%|██████████| 1715/1715 [00:35<00:00, 48.54it/s, loss=0.306]
Test Epoch 123 ==> 	accuracy: 0.9480, 	precision: 0.9560, 	recall: 0.7767, 	specificity: 0.9910, 	f1: 0.8571
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 124: 100%|██████████| 6195/6195 [05:37<00:00, 18.36it/s, loss=0.536]
Train Epoch 124 ==> 	accuracy: 0.8634, 	precision: 0.9997, 	recall: 0.7271, 	specificity: 0.9997, 	f1: 0.8419
Test Epoch 124: 100%|██████████| 1715/1715 [00:35<00:00, 48.80it/s, loss=0.617]
Test Epoch 124 ==> 	accuracy: 0.9470, 	precision: 0.9472, 	recall: 0.7793, 	specificity: 0.9891, 	f1: 0.8551
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 125: 100%|██████████| 6195/6195 [05:37<00:00, 18.38it/s, loss=0.35]
Train Epoch 125 ==> 	accuracy: 0.8648, 	precision: 0.9997, 	recall: 0.7299, 	specificity: 0.9998, 	f1: 0.8438
Test Epoch 125: 100%|██████████| 1715/1715 [00:33<00:00, 51.52it/s, loss=0.259]
Test Epoch 125 ==> 	accuracy: 0.9478, 	precision: 0.9525, 	recall: 0.7791, 	specificity: 0.9902, 	f1: 0.8571
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 126: 100%|██████████| 6195/6195 [05:44<00:00, 18.00it/s, loss=0.442]
Train Epoch 126 ==> 	accuracy: 0.8668, 	precision: 0.9996, 	recall: 0.7338, 	specificity: 0.9997, 	f1: 0.8463
Test Epoch 126: 100%|██████████| 1715/1715 [00:34<00:00, 50.13it/s, loss=4]
Test Epoch 126 ==> 	accuracy: 0.9486, 	precision: 0.9528, 	recall: 0.7828, 	specificity: 0.9903, 	f1: 0.8595
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 127: 100%|██████████| 6195/6195 [05:34<00:00, 18.54it/s, loss=0.47]
Train Epoch 127 ==> 	accuracy: 0.8632, 	precision: 0.9997, 	recall: 0.7266, 	specificity: 0.9998, 	f1: 0.8415
Test Epoch 127: 100%|██████████| 1715/1715 [00:37<00:00, 45.90it/s, loss=2.45]
Test Epoch 127 ==> 	accuracy: 0.9478, 	precision: 0.9581, 	recall: 0.7739, 	specificity: 0.9915, 	f1: 0.8562
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 128: 100%|██████████| 6195/6195 [05:44<00:00, 17.98it/s, loss=0.394]
Train Epoch 128 ==> 	accuracy: 0.8630, 	precision: 0.9997, 	recall: 0.7263, 	specificity: 0.9998, 	f1: 0.8413
Test Epoch 128: 100%|██████████| 1715/1715 [00:35<00:00, 48.45it/s, loss=0.354]
Test Epoch 128 ==> 	accuracy: 0.9498, 	precision: 0.9667, 	recall: 0.7766, 	specificity: 0.9933, 	f1: 0.8613
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 129: 100%|██████████| 6195/6195 [05:47<00:00, 17.85it/s, loss=0.328]
Train Epoch 129 ==> 	accuracy: 0.8627, 	precision: 0.9996, 	recall: 0.7258, 	specificity: 0.9997, 	f1: 0.8410
Test Epoch 129: 100%|██████████| 1715/1715 [00:36<00:00, 46.68it/s, loss=0.442]
Test Epoch 129 ==> 	accuracy: 0.9496, 	precision: 0.9651, 	recall: 0.7773, 	specificity: 0.9929, 	f1: 0.8611
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 130: 100%|██████████| 6195/6195 [05:56<00:00, 17.35it/s, loss=0.4]
Train Epoch 130 ==> 	accuracy: 0.8634, 	precision: 0.9996, 	recall: 0.7270, 	specificity: 0.9997, 	f1: 0.8418
Test Epoch 130: 100%|██████████| 1715/1715 [00:33<00:00, 50.96it/s, loss=0.157]
Test Epoch 130 ==> 	accuracy: 0.9505, 	precision: 0.9693, 	recall: 0.7782, 	specificity: 0.9938, 	f1: 0.8633
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 131: 100%|██████████| 6195/6195 [05:43<00:00, 18.05it/s, loss=0.492]
Train Epoch 131 ==> 	accuracy: 0.8633, 	precision: 0.9997, 	recall: 0.7268, 	specificity: 0.9998, 	f1: 0.8417
Test Epoch 131: 100%|██████████| 1715/1715 [00:38<00:00, 44.49it/s, loss=1.51]
Test Epoch 131 ==> 	accuracy: 0.9504, 	precision: 0.9708, 	recall: 0.7760, 	specificity: 0.9941, 	f1: 0.8626
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 132: 100%|██████████| 6195/6195 [05:44<00:00, 18.00it/s, loss=0.394]
Train Epoch 132 ==> 	accuracy: 0.8633, 	precision: 0.9996, 	recall: 0.7268, 	specificity: 0.9997, 	f1: 0.8417
Test Epoch 132: 100%|██████████| 1715/1715 [00:35<00:00, 47.73it/s, loss=1.13]
Test Epoch 132 ==> 	accuracy: 0.9504, 	precision: 0.9713, 	recall: 0.7760, 	specificity: 0.9942, 	f1: 0.8627
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 133: 100%|██████████| 6195/6195 [05:41<00:00, 18.16it/s, loss=0.498]
Train Epoch 133 ==> 	accuracy: 0.8649, 	precision: 0.9996, 	recall: 0.7301, 	specificity: 0.9997, 	f1: 0.8439
Test Epoch 133: 100%|██████████| 1715/1715 [00:37<00:00, 45.37it/s, loss=0.296]
Test Epoch 133 ==> 	accuracy: 0.9506, 	precision: 0.9682, 	recall: 0.7796, 	specificity: 0.9936, 	f1: 0.8637
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 134: 100%|██████████| 6195/6195 [05:47<00:00, 17.85it/s, loss=0.376]
Train Epoch 134 ==> 	accuracy: 0.8632, 	precision: 0.9997, 	recall: 0.7265, 	specificity: 0.9998, 	f1: 0.8415
Test Epoch 134: 100%|██████████| 1715/1715 [00:35<00:00, 47.76it/s, loss=2.24]
Test Epoch 134 ==> 	accuracy: 0.9495, 	precision: 0.9689, 	recall: 0.7735, 	specificity: 0.9938, 	f1: 0.8602
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 135: 100%|██████████| 6195/6195 [05:41<00:00, 18.12it/s, loss=0.413]
Train Epoch 135 ==> 	accuracy: 0.8658, 	precision: 0.9997, 	recall: 0.7317, 	specificity: 0.9998, 	f1: 0.8450
Test Epoch 135: 100%|██████████| 1715/1715 [00:35<00:00, 48.45it/s, loss=0.504]
Test Epoch 135 ==> 	accuracy: 0.9490, 	precision: 0.9691, 	recall: 0.7705, 	specificity: 0.9938, 	f1: 0.8585
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 136: 100%|██████████| 6195/6195 [05:47<00:00, 17.85it/s, loss=0.359]
Train Epoch 136 ==> 	accuracy: 0.8651, 	precision: 0.9997, 	recall: 0.7303, 	specificity: 0.9998, 	f1: 0.8441
Test Epoch 136: 100%|██████████| 1715/1715 [00:33<00:00, 51.76it/s, loss=3.45]
Test Epoch 136 ==> 	accuracy: 0.9502, 	precision: 0.9673, 	recall: 0.7780, 	specificity: 0.9934, 	f1: 0.8624
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 137: 100%|██████████| 6195/6195 [05:28<00:00, 18.84it/s, loss=0.391]
Train Epoch 137 ==> 	accuracy: 0.8658, 	precision: 0.9997, 	recall: 0.7319, 	specificity: 0.9998, 	f1: 0.8451
Test Epoch 137: 100%|██████████| 1715/1715 [00:33<00:00, 50.63it/s, loss=0.369]
Test Epoch 137 ==> 	accuracy: 0.9501, 	precision: 0.9655, 	recall: 0.7793, 	specificity: 0.9930, 	f1: 0.8625
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 138: 100%|██████████| 6195/6195 [05:40<00:00, 18.18it/s, loss=0.465]
Train Epoch 138 ==> 	accuracy: 0.8645, 	precision: 0.9997, 	recall: 0.7293, 	specificity: 0.9998, 	f1: 0.8433
Test Epoch 138: 100%|██████████| 1715/1715 [00:38<00:00, 45.09it/s, loss=2.24]
Test Epoch 138 ==> 	accuracy: 0.9486, 	precision: 0.9674, 	recall: 0.7700, 	specificity: 0.9935, 	f1: 0.8575
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 139: 100%|██████████| 6195/6195 [05:46<00:00, 17.86it/s, loss=0.508]
Train Epoch 139 ==> 	accuracy: 0.8672, 	precision: 0.9997, 	recall: 0.7346, 	specificity: 0.9998, 	f1: 0.8469
Test Epoch 139: 100%|██████████| 1715/1715 [00:35<00:00, 48.20it/s, loss=0.371]
Test Epoch 139 ==> 	accuracy: 0.9501, 	precision: 0.9642, 	recall: 0.7804, 	specificity: 0.9927, 	f1: 0.8626
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 140: 100%|██████████| 6195/6195 [05:42<00:00, 18.10it/s, loss=0.385]
Train Epoch 140 ==> 	accuracy: 0.8653, 	precision: 0.9997, 	recall: 0.7308, 	specificity: 0.9998, 	f1: 0.8444
Test Epoch 140: 100%|██████████| 1715/1715 [00:34<00:00, 50.15it/s, loss=0.123]
Test Epoch 140 ==> 	accuracy: 0.9508, 	precision: 0.9665, 	recall: 0.7819, 	specificity: 0.9932, 	f1: 0.8644
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 141: 100%|██████████| 6195/6195 [05:46<00:00, 17.88it/s, loss=0.394]
Train Epoch 141 ==> 	accuracy: 0.8641, 	precision: 0.9997, 	recall: 0.7284, 	specificity: 0.9998, 	f1: 0.8427
Test Epoch 141: 100%|██████████| 1715/1715 [00:35<00:00, 48.75it/s, loss=0.533]
Test Epoch 141 ==> 	accuracy: 0.9504, 	precision: 0.9687, 	recall: 0.7780, 	specificity: 0.9937, 	f1: 0.8630
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 142: 100%|██████████| 6195/6195 [05:40<00:00, 18.22it/s, loss=0.414]
Train Epoch 142 ==> 	accuracy: 0.8652, 	precision: 0.9997, 	recall: 0.7307, 	specificity: 0.9998, 	f1: 0.8443
Test Epoch 142: 100%|██████████| 1715/1715 [00:38<00:00, 44.26it/s, loss=0.427]
Test Epoch 142 ==> 	accuracy: 0.9504, 	precision: 0.9695, 	recall: 0.7776, 	specificity: 0.9938, 	f1: 0.8630
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 143: 100%|██████████| 6195/6195 [05:46<00:00, 17.89it/s, loss=0.423]
Train Epoch 143 ==> 	accuracy: 0.8646, 	precision: 0.9997, 	recall: 0.7295, 	specificity: 0.9998, 	f1: 0.8435
Test Epoch 143: 100%|██████████| 1715/1715 [00:34<00:00, 50.09it/s, loss=0.404]
Test Epoch 143 ==> 	accuracy: 0.9509, 	precision: 0.9690, 	recall: 0.7803, 	specificity: 0.9937, 	f1: 0.8645
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 144: 100%|██████████| 6195/6195 [05:42<00:00, 18.09it/s, loss=0.363]
Train Epoch 144 ==> 	accuracy: 0.8663, 	precision: 0.9997, 	recall: 0.7328, 	specificity: 0.9998, 	f1: 0.8457
Test Epoch 144: 100%|██████████| 1715/1715 [00:33<00:00, 50.50it/s, loss=0.214]
Test Epoch 144 ==> 	accuracy: 0.9508, 	precision: 0.9649, 	recall: 0.7833, 	specificity: 0.9928, 	f1: 0.8646
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 145: 100%|██████████| 6195/6195 [05:37<00:00, 18.36it/s, loss=0.461]
Train Epoch 145 ==> 	accuracy: 0.8658, 	precision: 0.9997, 	recall: 0.7318, 	specificity: 0.9998, 	f1: 0.8450
Test Epoch 145: 100%|██████████| 1715/1715 [00:34<00:00, 49.72it/s, loss=0.21]
Test Epoch 145 ==> 	accuracy: 0.9501, 	precision: 0.9611, 	recall: 0.7830, 	specificity: 0.9921, 	f1: 0.8629
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 146: 100%|██████████| 6195/6195 [05:44<00:00, 17.99it/s, loss=0.391]
Train Epoch 146 ==> 	accuracy: 0.8664, 	precision: 0.9997, 	recall: 0.7330, 	specificity: 0.9997, 	f1: 0.8458
Test Epoch 146: 100%|██████████| 1715/1715 [00:35<00:00, 48.78it/s, loss=0.506]
Test Epoch 146 ==> 	accuracy: 0.9496, 	precision: 0.9628, 	recall: 0.7792, 	specificity: 0.9924, 	f1: 0.8613
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 147: 100%|██████████| 6195/6195 [05:36<00:00, 18.42it/s, loss=0.418]
Train Epoch 147 ==> 	accuracy: 0.8661, 	precision: 0.9997, 	recall: 0.7325, 	specificity: 0.9998, 	f1: 0.8455
Test Epoch 147: 100%|██████████| 1715/1715 [00:38<00:00, 45.10it/s, loss=0.791]
Test Epoch 147 ==> 	accuracy: 0.9506, 	precision: 0.9662, 	recall: 0.7811, 	specificity: 0.9931, 	f1: 0.8639
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 148: 100%|██████████| 6195/6195 [05:42<00:00, 18.08it/s, loss=0.489]
Train Epoch 148 ==> 	accuracy: 0.8670, 	precision: 0.9997, 	recall: 0.7342, 	specificity: 0.9998, 	f1: 0.8466
Test Epoch 148: 100%|██████████| 1715/1715 [00:38<00:00, 44.93it/s, loss=0.372]
Test Epoch 148 ==> 	accuracy: 0.9509, 	precision: 0.9642, 	recall: 0.7844, 	specificity: 0.9927, 	f1: 0.8651
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 149: 100%|██████████| 6195/6195 [05:40<00:00, 18.17it/s, loss=0.409]
Train Epoch 149 ==> 	accuracy: 0.8661, 	precision: 0.9997, 	recall: 0.7325, 	specificity: 0.9998, 	f1: 0.8455
Test Epoch 149: 100%|██████████| 1715/1715 [00:35<00:00, 48.01it/s, loss=1.28]
Test Epoch 149 ==> 	accuracy: 0.9503, 	precision: 0.9651, 	recall: 0.7807, 	specificity: 0.9929, 	f1: 0.8632
Adjusting learning rate of group 0 to 4.2391e-06.

Process finished with exit code 0


'''

'''
'../model_save_sigBlock4_focalWithMs_deformable_ab_multConv'
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6195/6195 [04:24<00:00, 23.40it/s, loss=0.726]
Train Epoch 0 ==> 	accuracy: 0.6286, 	precision: 0.9953, 	recall: 0.2585, 	specificity: 0.9988, 	f1: 0.4104
Test Epoch 0: 100%|██████████| 1715/1715 [00:30<00:00, 56.31it/s, loss=0.866]
Test Epoch 0 ==> 	accuracy: 0.8958, 	precision: 0.9639, 	recall: 0.4998, 	specificity: 0.9953, 	f1: 0.6582
Train Epoch 1: 100%|██████████| 6195/6195 [05:27<00:00, 18.93it/s, loss=0.608]
Train Epoch 1 ==> 	accuracy: 0.7094, 	precision: 0.9967, 	recall: 0.4202, 	specificity: 0.9986, 	f1: 0.5912
Test Epoch 1: 100%|██████████| 1715/1715 [00:35<00:00, 48.74it/s, loss=0.326]
Test Epoch 1 ==> 	accuracy: 0.8970, 	precision: 0.9829, 	recall: 0.4954, 	specificity: 0.9978, 	f1: 0.6587
Train Epoch 2: 100%|██████████| 6195/6195 [05:51<00:00, 17.64it/s, loss=0.599]
Train Epoch 2 ==> 	accuracy: 0.7384, 	precision: 0.9972, 	recall: 0.4781, 	specificity: 0.9987, 	f1: 0.6463
Test Epoch 2: 100%|██████████| 1715/1715 [00:38<00:00, 44.90it/s, loss=0.286]
Test Epoch 2 ==> 	accuracy: 0.9077, 	precision: 0.9844, 	recall: 0.5490, 	specificity: 0.9978, 	f1: 0.7049
Train Epoch 3: 100%|██████████| 6195/6195 [05:57<00:00, 17.31it/s, loss=0.558]
Train Epoch 3 ==> 	accuracy: 0.7459, 	precision: 0.9974, 	recall: 0.4932, 	specificity: 0.9987, 	f1: 0.6600
Test Epoch 3: 100%|██████████| 1715/1715 [00:38<00:00, 44.49it/s, loss=0.503]
Test Epoch 3 ==> 	accuracy: 0.9086, 	precision: 0.9655, 	recall: 0.5648, 	specificity: 0.9949, 	f1: 0.7127
Train Epoch 4: 100%|██████████| 6195/6195 [05:58<00:00, 17.28it/s, loss=0.527]
Train Epoch 4 ==> 	accuracy: 0.7593, 	precision: 0.9977, 	recall: 0.5198, 	specificity: 0.9988, 	f1: 0.6835
Test Epoch 4: 100%|██████████| 1715/1715 [00:38<00:00, 45.00it/s, loss=0.249]
Test Epoch 4 ==> 	accuracy: 0.9098, 	precision: 0.9827, 	recall: 0.5606, 	specificity: 0.9975, 	f1: 0.7139
Train Epoch 5: 100%|██████████| 6195/6195 [05:53<00:00, 17.54it/s, loss=0.735]
Train Epoch 5 ==> 	accuracy: 0.7601, 	precision: 0.9980, 	recall: 0.5212, 	specificity: 0.9990, 	f1: 0.6848
Test Epoch 5: 100%|██████████| 1715/1715 [00:38<00:00, 44.99it/s, loss=0.272]
Test Epoch 5 ==> 	accuracy: 0.9141, 	precision: 0.9624, 	recall: 0.5956, 	specificity: 0.9941, 	f1: 0.7358
Train Epoch 6: 100%|██████████| 6195/6195 [06:06<00:00, 16.91it/s, loss=0.534]
Train Epoch 6 ==> 	accuracy: 0.7688, 	precision: 0.9981, 	recall: 0.5386, 	specificity: 0.9990, 	f1: 0.6996
Test Epoch 6: 100%|██████████| 1715/1715 [00:40<00:00, 42.22it/s, loss=0.845]
Test Epoch 6 ==> 	accuracy: 0.9171, 	precision: 0.9705, 	recall: 0.6054, 	specificity: 0.9954, 	f1: 0.7456
Train Epoch 7: 100%|██████████| 6195/6195 [06:06<00:00, 16.89it/s, loss=0.514]
Train Epoch 7 ==> 	accuracy: 0.7741, 	precision: 0.9981, 	recall: 0.5493, 	specificity: 0.9989, 	f1: 0.7086
Test Epoch 7: 100%|██████████| 1715/1715 [00:40<00:00, 42.63it/s, loss=0.372]
Test Epoch 7 ==> 	accuracy: 0.9136, 	precision: 0.9897, 	recall: 0.5757, 	specificity: 0.9985, 	f1: 0.7280
Train Epoch 8: 100%|██████████| 6195/6195 [06:12<00:00, 16.65it/s, loss=0.722]
Train Epoch 8 ==> 	accuracy: 0.7715, 	precision: 0.9982, 	recall: 0.5441, 	specificity: 0.9990, 	f1: 0.7043
Test Epoch 8: 100%|██████████| 1715/1715 [00:41<00:00, 41.83it/s, loss=0.503]
Test Epoch 8 ==> 	accuracy: 0.9213, 	precision: 0.9579, 	recall: 0.6360, 	specificity: 0.9930, 	f1: 0.7645
Train Epoch 9: 100%|██████████| 6195/6195 [06:14<00:00, 16.54it/s, loss=0.534]
Train Epoch 9 ==> 	accuracy: 0.7767, 	precision: 0.9984, 	recall: 0.5544, 	specificity: 0.9991, 	f1: 0.7129
Test Epoch 9: 100%|██████████| 1715/1715 [00:40<00:00, 41.89it/s, loss=0.313]
Test Epoch 9 ==> 	accuracy: 0.9130, 	precision: 0.9887, 	recall: 0.5733, 	specificity: 0.9984, 	f1: 0.7258
Train Epoch 10: 100%|██████████| 6195/6195 [06:10<00:00, 16.71it/s, loss=0.499]
Train Epoch 10 ==> 	accuracy: 0.7811, 	precision: 0.9984, 	recall: 0.5632, 	specificity: 0.9991, 	f1: 0.7201
Test Epoch 10: 100%|██████████| 1715/1715 [00:39<00:00, 43.30it/s, loss=0.264]
Test Epoch 10 ==> 	accuracy: 0.9222, 	precision: 0.9747, 	recall: 0.6289, 	specificity: 0.9959, 	f1: 0.7645
Train Epoch 11: 100%|██████████| 6195/6195 [06:08<00:00, 16.81it/s, loss=0.411]
Train Epoch 11 ==> 	accuracy: 0.7823, 	precision: 0.9984, 	recall: 0.5654, 	specificity: 0.9991, 	f1: 0.7220
Test Epoch 11: 100%|██████████| 1715/1715 [00:39<00:00, 43.07it/s, loss=0.233]
Test Epoch 11 ==> 	accuracy: 0.9234, 	precision: 0.9835, 	recall: 0.6288, 	specificity: 0.9974, 	f1: 0.7671
Train Epoch 12: 100%|██████████| 6195/6195 [06:16<00:00, 16.45it/s, loss=0.478]
Train Epoch 12 ==> 	accuracy: 0.7874, 	precision: 0.9985, 	recall: 0.5757, 	specificity: 0.9991, 	f1: 0.7303
Test Epoch 12: 100%|██████████| 1715/1715 [00:41<00:00, 41.18it/s, loss=0.472]
Test Epoch 12 ==> 	accuracy: 0.9185, 	precision: 0.9775, 	recall: 0.6083, 	specificity: 0.9965, 	f1: 0.7499
Train Epoch 13: 100%|██████████| 6195/6195 [06:06<00:00, 16.91it/s, loss=0.636]
Train Epoch 13 ==> 	accuracy: 0.7883, 	precision: 0.9986, 	recall: 0.5775, 	specificity: 0.9992, 	f1: 0.7318
Test Epoch 13: 100%|██████████| 1715/1715 [00:39<00:00, 42.99it/s, loss=0.713]
Test Epoch 13 ==> 	accuracy: 0.9223, 	precision: 0.9650, 	recall: 0.6360, 	specificity: 0.9942, 	f1: 0.7667
Train Epoch 14: 100%|██████████| 6195/6195 [06:16<00:00, 16.44it/s, loss=0.594]
Train Epoch 14 ==> 	accuracy: 0.7927, 	precision: 0.9986, 	recall: 0.5862, 	specificity: 0.9992, 	f1: 0.7387
Test Epoch 14: 100%|██████████| 1715/1715 [00:38<00:00, 44.74it/s, loss=0.844]
Test Epoch 14 ==> 	accuracy: 0.9127, 	precision: 0.9704, 	recall: 0.5828, 	specificity: 0.9955, 	f1: 0.7283
Train Epoch 15: 100%|██████████| 6195/6195 [06:16<00:00, 16.47it/s, loss=0.567]
Train Epoch 15 ==> 	accuracy: 0.7906, 	precision: 0.9987, 	recall: 0.5819, 	specificity: 0.9992, 	f1: 0.7354
Test Epoch 15: 100%|██████████| 1715/1715 [00:39<00:00, 42.91it/s, loss=0.383]
Test Epoch 15 ==> 	accuracy: 0.9217, 	precision: 0.9810, 	recall: 0.6219, 	specificity: 0.9970, 	f1: 0.7612
Train Epoch 16: 100%|██████████| 6195/6195 [06:17<00:00, 16.39it/s, loss=0.46]
Train Epoch 16 ==> 	accuracy: 0.7932, 	precision: 0.9987, 	recall: 0.5871, 	specificity: 0.9992, 	f1: 0.7395
Test Epoch 16: 100%|██████████| 1715/1715 [00:44<00:00, 38.39it/s, loss=0.603]
Test Epoch 16 ==> 	accuracy: 0.9156, 	precision: 0.9805, 	recall: 0.5913, 	specificity: 0.9970, 	f1: 0.7377
Train Epoch 17: 100%|██████████| 6195/6195 [06:17<00:00, 16.43it/s, loss=0.474]
Train Epoch 17 ==> 	accuracy: 0.7967, 	precision: 0.9988, 	recall: 0.5941, 	specificity: 0.9993, 	f1: 0.7450
Test Epoch 17: 100%|██████████| 1715/1715 [00:43<00:00, 39.42it/s, loss=0.533]
Test Epoch 17 ==> 	accuracy: 0.9219, 	precision: 0.9759, 	recall: 0.6264, 	specificity: 0.9961, 	f1: 0.7630
Train Epoch 18: 100%|██████████| 6195/6195 [06:18<00:00, 16.36it/s, loss=0.461]
Train Epoch 18 ==> 	accuracy: 0.7965, 	precision: 0.9988, 	recall: 0.5937, 	specificity: 0.9993, 	f1: 0.7447
Test Epoch 18: 100%|██████████| 1715/1715 [00:44<00:00, 38.75it/s, loss=0.236]
Test Epoch 18 ==> 	accuracy: 0.9211, 	precision: 0.9801, 	recall: 0.6196, 	specificity: 0.9968, 	f1: 0.7592
Train Epoch 19: 100%|██████████| 6195/6195 [06:21<00:00, 16.23it/s, loss=0.513]
Train Epoch 19 ==> 	accuracy: 0.7967, 	precision: 0.9988, 	recall: 0.5941, 	specificity: 0.9993, 	f1: 0.7450
Test Epoch 19: 100%|██████████| 1715/1715 [00:48<00:00, 35.70it/s, loss=0.648]
Test Epoch 19 ==> 	accuracy: 0.9186, 	precision: 0.9793, 	recall: 0.6075, 	specificity: 0.9968, 	f1: 0.7499
Train Epoch 20: 100%|██████████| 6195/6195 [06:14<00:00, 16.55it/s, loss=0.583]
Train Epoch 20 ==> 	accuracy: 0.7951, 	precision: 0.9989, 	recall: 0.5909, 	specificity: 0.9994, 	f1: 0.7425
Test Epoch 20: 100%|██████████| 1715/1715 [00:41<00:00, 41.30it/s, loss=0.565]
Test Epoch 20 ==> 	accuracy: 0.9264, 	precision: 0.9740, 	recall: 0.6508, 	specificity: 0.9956, 	f1: 0.7802
Train Epoch 21: 100%|██████████| 6195/6195 [06:20<00:00, 16.29it/s, loss=0.526]
Train Epoch 21 ==> 	accuracy: 0.8057, 	precision: 0.9989, 	recall: 0.6122, 	specificity: 0.9993, 	f1: 0.7591
Test Epoch 21: 100%|██████████| 1715/1715 [00:39<00:00, 42.88it/s, loss=0.697]
Test Epoch 21 ==> 	accuracy: 0.9280, 	precision: 0.9791, 	recall: 0.6551, 	specificity: 0.9965, 	f1: 0.7850
Train Epoch 22: 100%|██████████| 6195/6195 [06:14<00:00, 16.53it/s, loss=0.555]
Train Epoch 22 ==> 	accuracy: 0.8001, 	precision: 0.9990, 	recall: 0.6009, 	specificity: 0.9994, 	f1: 0.7504
Test Epoch 22: 100%|██████████| 1715/1715 [00:42<00:00, 40.35it/s, loss=0.421]
Test Epoch 22 ==> 	accuracy: 0.9277, 	precision: 0.9819, 	recall: 0.6521, 	specificity: 0.9970, 	f1: 0.7837
Train Epoch 23: 100%|██████████| 6195/6195 [06:17<00:00, 16.42it/s, loss=0.472]
Train Epoch 23 ==> 	accuracy: 0.8040, 	precision: 0.9989, 	recall: 0.6087, 	specificity: 0.9993, 	f1: 0.7564
Test Epoch 23: 100%|██████████| 1715/1715 [00:41<00:00, 41.46it/s, loss=0.646]
Test Epoch 23 ==> 	accuracy: 0.9234, 	precision: 0.9691, 	recall: 0.6386, 	specificity: 0.9949, 	f1: 0.7699
Train Epoch 24: 100%|██████████| 6195/6195 [06:22<00:00, 16.21it/s, loss=0.503]
Train Epoch 24 ==> 	accuracy: 0.8010, 	precision: 0.9989, 	recall: 0.6027, 	specificity: 0.9993, 	f1: 0.7518
Test Epoch 24: 100%|██████████| 1715/1715 [00:38<00:00, 44.09it/s, loss=0.287]
Test Epoch 24 ==> 	accuracy: 0.9241, 	precision: 0.9814, 	recall: 0.6340, 	specificity: 0.9970, 	f1: 0.7703
Train Epoch 25: 100%|██████████| 6195/6195 [06:17<00:00, 16.42it/s, loss=0.493]
Train Epoch 25 ==> 	accuracy: 0.8058, 	precision: 0.9990, 	recall: 0.6123, 	specificity: 0.9994, 	f1: 0.7592
Test Epoch 25: 100%|██████████| 1715/1715 [00:35<00:00, 48.57it/s, loss=0.535]
Test Epoch 25 ==> 	accuracy: 0.9242, 	precision: 0.9775, 	recall: 0.6370, 	specificity: 0.9963, 	f1: 0.7714
Train Epoch 26: 100%|██████████| 6195/6195 [06:11<00:00, 16.67it/s, loss=0.584]
Train Epoch 26 ==> 	accuracy: 0.8064, 	precision: 0.9990, 	recall: 0.6135, 	specificity: 0.9994, 	f1: 0.7602
Test Epoch 26: 100%|██████████| 1715/1715 [00:38<00:00, 44.42it/s, loss=0.275]
Test Epoch 26 ==> 	accuracy: 0.9281, 	precision: 0.9783, 	recall: 0.6564, 	specificity: 0.9963, 	f1: 0.7856
Train Epoch 27: 100%|██████████| 6195/6195 [06:17<00:00, 16.40it/s, loss=0.698]
Train Epoch 27 ==> 	accuracy: 0.8062, 	precision: 0.9990, 	recall: 0.6130, 	specificity: 0.9994, 	f1: 0.7598
Test Epoch 27: 100%|██████████| 1715/1715 [00:38<00:00, 44.13it/s, loss=2.21]
Test Epoch 27 ==> 	accuracy: 0.9283, 	precision: 0.9714, 	recall: 0.6624, 	specificity: 0.9951, 	f1: 0.7877
Train Epoch 28: 100%|██████████| 6195/6195 [06:10<00:00, 16.71it/s, loss=0.407]
Train Epoch 28 ==> 	accuracy: 0.8077, 	precision: 0.9991, 	recall: 0.6160, 	specificity: 0.9994, 	f1: 0.7621
Test Epoch 28: 100%|██████████| 1715/1715 [00:38<00:00, 45.09it/s, loss=1.34]
Test Epoch 28 ==> 	accuracy: 0.9249, 	precision: 0.9864, 	recall: 0.6348, 	specificity: 0.9978, 	f1: 0.7725
Train Epoch 29: 100%|██████████| 6195/6195 [06:13<00:00, 16.60it/s, loss=0.511]
Train Epoch 29 ==> 	accuracy: 0.8092, 	precision: 0.9991, 	recall: 0.6190, 	specificity: 0.9994, 	f1: 0.7644
Test Epoch 29: 100%|██████████| 1715/1715 [00:38<00:00, 44.34it/s, loss=0.313]
Test Epoch 29 ==> 	accuracy: 0.9284, 	precision: 0.9761, 	recall: 0.6597, 	specificity: 0.9959, 	f1: 0.7873
Train Epoch 30: 100%|██████████| 6195/6195 [06:17<00:00, 16.42it/s, loss=0.418]
Train Epoch 30 ==> 	accuracy: 0.8104, 	precision: 0.9991, 	recall: 0.6214, 	specificity: 0.9994, 	f1: 0.7662
Test Epoch 30: 100%|██████████| 1715/1715 [00:39<00:00, 43.28it/s, loss=0.278]
Test Epoch 30 ==> 	accuracy: 0.9293, 	precision: 0.9763, 	recall: 0.6639, 	specificity: 0.9960, 	f1: 0.7903
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 31: 100%|██████████| 6195/6195 [06:18<00:00, 16.39it/s, loss=0.39]
Train Epoch 31 ==> 	accuracy: 0.8125, 	precision: 0.9991, 	recall: 0.6256, 	specificity: 0.9995, 	f1: 0.7694
Test Epoch 31: 100%|██████████| 1715/1715 [00:38<00:00, 44.38it/s, loss=0.401]
Test Epoch 31 ==> 	accuracy: 0.9278, 	precision: 0.9758, 	recall: 0.6568, 	specificity: 0.9959, 	f1: 0.7851
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 32: 100%|██████████| 6195/6195 [06:10<00:00, 16.73it/s, loss=0.552]
Train Epoch 32 ==> 	accuracy: 0.8100, 	precision: 0.9991, 	recall: 0.6205, 	specificity: 0.9995, 	f1: 0.7656
Test Epoch 32: 100%|██████████| 1715/1715 [00:37<00:00, 45.15it/s, loss=0.543]
Test Epoch 32 ==> 	accuracy: 0.9288, 	precision: 0.9829, 	recall: 0.6567, 	specificity: 0.9971, 	f1: 0.7874
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 33: 100%|██████████| 6195/6195 [06:13<00:00, 16.57it/s, loss=0.513]
Train Epoch 33 ==> 	accuracy: 0.8114, 	precision: 0.9991, 	recall: 0.6233, 	specificity: 0.9994, 	f1: 0.7677
Test Epoch 33: 100%|██████████| 1715/1715 [00:38<00:00, 44.27it/s, loss=0.217]
Test Epoch 33 ==> 	accuracy: 0.9300, 	precision: 0.9827, 	recall: 0.6630, 	specificity: 0.9971, 	f1: 0.7918
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 34: 100%|██████████| 6195/6195 [06:09<00:00, 16.77it/s, loss=0.455]
Train Epoch 34 ==> 	accuracy: 0.8166, 	precision: 0.9991, 	recall: 0.6338, 	specificity: 0.9994, 	f1: 0.7756
Test Epoch 34: 100%|██████████| 1715/1715 [00:39<00:00, 43.84it/s, loss=0.449]
Test Epoch 34 ==> 	accuracy: 0.9341, 	precision: 0.9804, 	recall: 0.6852, 	specificity: 0.9966, 	f1: 0.8067
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 35: 100%|██████████| 6195/6195 [06:15<00:00, 16.51it/s, loss=0.398]
Train Epoch 35 ==> 	accuracy: 0.8206, 	precision: 0.9992, 	recall: 0.6416, 	specificity: 0.9995, 	f1: 0.7815
Test Epoch 35: 100%|██████████| 1715/1715 [00:38<00:00, 44.32it/s, loss=0.62]
Test Epoch 35 ==> 	accuracy: 0.9313, 	precision: 0.9742, 	recall: 0.6759, 	specificity: 0.9955, 	f1: 0.7981
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 36: 100%|██████████| 6195/6195 [06:17<00:00, 16.43it/s, loss=0.467]
Train Epoch 36 ==> 	accuracy: 0.8191, 	precision: 0.9992, 	recall: 0.6388, 	specificity: 0.9995, 	f1: 0.7793
Test Epoch 36: 100%|██████████| 1715/1715 [00:40<00:00, 42.52it/s, loss=0.509]
Test Epoch 36 ==> 	accuracy: 0.9275, 	precision: 0.9673, 	recall: 0.6613, 	specificity: 0.9944, 	f1: 0.7855
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 37: 100%|██████████| 6195/6195 [06:26<00:00, 16.02it/s, loss=0.425]
Train Epoch 37 ==> 	accuracy: 0.8178, 	precision: 0.9993, 	recall: 0.6361, 	specificity: 0.9995, 	f1: 0.7774
Test Epoch 37: 100%|██████████| 1715/1715 [00:38<00:00, 44.45it/s, loss=0.318]
Test Epoch 37 ==> 	accuracy: 0.9333, 	precision: 0.9753, 	recall: 0.6852, 	specificity: 0.9956, 	f1: 0.8049
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 38: 100%|██████████| 6195/6195 [06:15<00:00, 16.48it/s, loss=0.471]
Train Epoch 38 ==> 	accuracy: 0.8230, 	precision: 0.9992, 	recall: 0.6465, 	specificity: 0.9995, 	f1: 0.7851
Test Epoch 38: 100%|██████████| 1715/1715 [00:43<00:00, 39.85it/s, loss=0.284]
Test Epoch 38 ==> 	accuracy: 0.9331, 	precision: 0.9731, 	recall: 0.6859, 	specificity: 0.9952, 	f1: 0.8046
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 39: 100%|██████████| 6195/6195 [06:23<00:00, 16.16it/s, loss=0.407]
Train Epoch 39 ==> 	accuracy: 0.8228, 	precision: 0.9992, 	recall: 0.6460, 	specificity: 0.9995, 	f1: 0.7847
Test Epoch 39: 100%|██████████| 1715/1715 [00:40<00:00, 42.03it/s, loss=0.409]
Test Epoch 39 ==> 	accuracy: 0.9314, 	precision: 0.9806, 	recall: 0.6715, 	specificity: 0.9967, 	f1: 0.7971
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 40: 100%|██████████| 6195/6195 [06:19<00:00, 16.31it/s, loss=0.472]
Train Epoch 40 ==> 	accuracy: 0.8249, 	precision: 0.9993, 	recall: 0.6502, 	specificity: 0.9995, 	f1: 0.7878
Test Epoch 40: 100%|██████████| 1715/1715 [00:39<00:00, 43.18it/s, loss=0.328]
Test Epoch 40 ==> 	accuracy: 0.9316, 	precision: 0.9726, 	recall: 0.6784, 	specificity: 0.9952, 	f1: 0.7993
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 41: 100%|██████████| 6195/6195 [06:19<00:00, 16.33it/s, loss=0.597]
Train Epoch 41 ==> 	accuracy: 0.8218, 	precision: 0.9993, 	recall: 0.6441, 	specificity: 0.9996, 	f1: 0.7833
Test Epoch 41: 100%|██████████| 1715/1715 [00:41<00:00, 41.37it/s, loss=0.95]
Test Epoch 41 ==> 	accuracy: 0.9305, 	precision: 0.9708, 	recall: 0.6741, 	specificity: 0.9949, 	f1: 0.7957
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 42: 100%|██████████| 6195/6195 [06:19<00:00, 16.30it/s, loss=0.389]
Train Epoch 42 ==> 	accuracy: 0.8294, 	precision: 0.9992, 	recall: 0.6593, 	specificity: 0.9995, 	f1: 0.7944
Test Epoch 42: 100%|██████████| 1715/1715 [00:43<00:00, 39.80it/s, loss=1.28]
Test Epoch 42 ==> 	accuracy: 0.9347, 	precision: 0.9649, 	recall: 0.7000, 	specificity: 0.9936, 	f1: 0.8114
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 43: 100%|██████████| 6195/6195 [06:16<00:00, 16.45it/s, loss=0.534]
Train Epoch 43 ==> 	accuracy: 0.8263, 	precision: 0.9993, 	recall: 0.6531, 	specificity: 0.9996, 	f1: 0.7899
Test Epoch 43: 100%|██████████| 1715/1715 [00:42<00:00, 40.70it/s, loss=0.181]
Test Epoch 43 ==> 	accuracy: 0.9332, 	precision: 0.9725, 	recall: 0.6865, 	specificity: 0.9951, 	f1: 0.8049
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 44: 100%|██████████| 6195/6195 [06:22<00:00, 16.20it/s, loss=0.495]
Train Epoch 44 ==> 	accuracy: 0.8272, 	precision: 0.9993, 	recall: 0.6548, 	specificity: 0.9996, 	f1: 0.7912
Test Epoch 44: 100%|██████████| 1715/1715 [00:40<00:00, 42.42it/s, loss=0.716]
Test Epoch 44 ==> 	accuracy: 0.9292, 	precision: 0.9825, 	recall: 0.6589, 	specificity: 0.9971, 	f1: 0.7888
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 45: 100%|██████████| 6195/6195 [06:23<00:00, 16.14it/s, loss=0.397]
Train Epoch 45 ==> 	accuracy: 0.8257, 	precision: 0.9993, 	recall: 0.6518, 	specificity: 0.9996, 	f1: 0.7890
Test Epoch 45: 100%|██████████| 1715/1715 [00:43<00:00, 39.09it/s, loss=0.329]
Test Epoch 45 ==> 	accuracy: 0.9340, 	precision: 0.9799, 	recall: 0.6853, 	specificity: 0.9965, 	f1: 0.8066
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 46: 100%|██████████| 6195/6195 [06:22<00:00, 16.20it/s, loss=0.479]
Train Epoch 46 ==> 	accuracy: 0.8310, 	precision: 0.9994, 	recall: 0.6623, 	specificity: 0.9996, 	f1: 0.7967
Test Epoch 46: 100%|██████████| 1715/1715 [00:40<00:00, 42.66it/s, loss=0.332]
Test Epoch 46 ==> 	accuracy: 0.9336, 	precision: 0.9721, 	recall: 0.6891, 	specificity: 0.9950, 	f1: 0.8065
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 47: 100%|██████████| 6195/6195 [06:20<00:00, 16.26it/s, loss=0.562]
Train Epoch 47 ==> 	accuracy: 0.8304, 	precision: 0.9994, 	recall: 0.6611, 	specificity: 0.9996, 	f1: 0.7958
Test Epoch 47: 100%|██████████| 1715/1715 [00:41<00:00, 40.88it/s, loss=2]
Test Epoch 47 ==> 	accuracy: 0.9350, 	precision: 0.9763, 	recall: 0.6929, 	specificity: 0.9958, 	f1: 0.8105
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 48: 100%|██████████| 6195/6195 [06:22<00:00, 16.18it/s, loss=0.507]
Train Epoch 48 ==> 	accuracy: 0.8297, 	precision: 0.9994, 	recall: 0.6598, 	specificity: 0.9996, 	f1: 0.7949
Test Epoch 48: 100%|██████████| 1715/1715 [00:39<00:00, 43.42it/s, loss=0.223]
Test Epoch 48 ==> 	accuracy: 0.9333, 	precision: 0.9823, 	recall: 0.6799, 	specificity: 0.9969, 	f1: 0.8036
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 49: 100%|██████████| 6195/6195 [06:23<00:00, 16.17it/s, loss=0.617]
Train Epoch 49 ==> 	accuracy: 0.8311, 	precision: 0.9994, 	recall: 0.6625, 	specificity: 0.9996, 	f1: 0.7968
Test Epoch 49: 100%|██████████| 1715/1715 [00:41<00:00, 41.37it/s, loss=0.686]
Test Epoch 49 ==> 	accuracy: 0.9320, 	precision: 0.9743, 	recall: 0.6791, 	specificity: 0.9955, 	f1: 0.8003
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 50: 100%|██████████| 6195/6195 [06:21<00:00, 16.23it/s, loss=0.44]
Train Epoch 50 ==> 	accuracy: 0.8335, 	precision: 0.9994, 	recall: 0.6673, 	specificity: 0.9996, 	f1: 0.8003
Test Epoch 50: 100%|██████████| 1715/1715 [00:40<00:00, 42.17it/s, loss=1.24]
Test Epoch 50 ==> 	accuracy: 0.9366, 	precision: 0.9862, 	recall: 0.6940, 	specificity: 0.9976, 	f1: 0.8147
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 51: 100%|██████████| 6195/6195 [06:12<00:00, 16.63it/s, loss=0.435]
Train Epoch 51 ==> 	accuracy: 0.8346, 	precision: 0.9994, 	recall: 0.6696, 	specificity: 0.9996, 	f1: 0.8019
Test Epoch 51: 100%|██████████| 1715/1715 [00:38<00:00, 44.40it/s, loss=0.28]
Test Epoch 51 ==> 	accuracy: 0.9368, 	precision: 0.9858, 	recall: 0.6953, 	specificity: 0.9975, 	f1: 0.8155
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 52: 100%|██████████| 6195/6195 [06:08<00:00, 16.81it/s, loss=0.485]
Train Epoch 52 ==> 	accuracy: 0.8339, 	precision: 0.9995, 	recall: 0.6681, 	specificity: 0.9997, 	f1: 0.8009
Test Epoch 52: 100%|██████████| 1715/1715 [00:38<00:00, 44.67it/s, loss=0.142]
Test Epoch 52 ==> 	accuracy: 0.9372, 	precision: 0.9852, 	recall: 0.6976, 	specificity: 0.9974, 	f1: 0.8168
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 53: 100%|██████████| 6195/6195 [06:11<00:00, 16.66it/s, loss=0.397]
Train Epoch 53 ==> 	accuracy: 0.8363, 	precision: 0.9994, 	recall: 0.6730, 	specificity: 0.9996, 	f1: 0.8043
Test Epoch 53: 100%|██████████| 1715/1715 [00:39<00:00, 43.29it/s, loss=0.41]
Test Epoch 53 ==> 	accuracy: 0.9374, 	precision: 0.9842, 	recall: 0.6995, 	specificity: 0.9972, 	f1: 0.8178
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 54: 100%|██████████| 6195/6195 [06:18<00:00, 16.38it/s, loss=0.43]
Train Epoch 54 ==> 	accuracy: 0.8363, 	precision: 0.9994, 	recall: 0.6731, 	specificity: 0.9996, 	f1: 0.8044
Test Epoch 54: 100%|██████████| 1715/1715 [00:41<00:00, 41.14it/s, loss=0.247]
Test Epoch 54 ==> 	accuracy: 0.9365, 	precision: 0.9877, 	recall: 0.6923, 	specificity: 0.9978, 	f1: 0.8140
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 55: 100%|██████████| 6195/6195 [06:17<00:00, 16.41it/s, loss=0.427]
Train Epoch 55 ==> 	accuracy: 0.8364, 	precision: 0.9994, 	recall: 0.6731, 	specificity: 0.9996, 	f1: 0.8044
Test Epoch 55: 100%|██████████| 1715/1715 [00:39<00:00, 43.81it/s, loss=0.26]
Test Epoch 55 ==> 	accuracy: 0.9349, 	precision: 0.9874, 	recall: 0.6846, 	specificity: 0.9978, 	f1: 0.8086
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 56: 100%|██████████| 6195/6195 [06:08<00:00, 16.80it/s, loss=0.469]
Train Epoch 56 ==> 	accuracy: 0.8398, 	precision: 0.9994, 	recall: 0.6799, 	specificity: 0.9996, 	f1: 0.8093
Test Epoch 56: 100%|██████████| 1715/1715 [00:43<00:00, 39.82it/s, loss=1.96]
Test Epoch 56 ==> 	accuracy: 0.9383, 	precision: 0.9724, 	recall: 0.7130, 	specificity: 0.9949, 	f1: 0.8227
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 57: 100%|██████████| 6195/6195 [06:15<00:00, 16.51it/s, loss=0.356]
Train Epoch 57 ==> 	accuracy: 0.8387, 	precision: 0.9995, 	recall: 0.6777, 	specificity: 0.9997, 	f1: 0.8077
Test Epoch 57: 100%|██████████| 1715/1715 [00:44<00:00, 38.75it/s, loss=0.168]
Test Epoch 57 ==> 	accuracy: 0.9379, 	precision: 0.9845, 	recall: 0.7018, 	specificity: 0.9972, 	f1: 0.8194
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 58: 100%|██████████| 6195/6195 [06:15<00:00, 16.51it/s, loss=0.47]
Train Epoch 58 ==> 	accuracy: 0.8382, 	precision: 0.9995, 	recall: 0.6767, 	specificity: 0.9996, 	f1: 0.8070
Test Epoch 58: 100%|██████████| 1715/1715 [00:43<00:00, 39.58it/s, loss=0.411]
Test Epoch 58 ==> 	accuracy: 0.9372, 	precision: 0.9835, 	recall: 0.6987, 	specificity: 0.9971, 	f1: 0.8170
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 59: 100%|██████████| 6195/6195 [06:15<00:00, 16.48it/s, loss=0.417]
Train Epoch 59 ==> 	accuracy: 0.8416, 	precision: 0.9995, 	recall: 0.6835, 	specificity: 0.9997, 	f1: 0.8118
Test Epoch 59: 100%|██████████| 1715/1715 [00:39<00:00, 43.00it/s, loss=2.02]
Test Epoch 59 ==> 	accuracy: 0.9389, 	precision: 0.9833, 	recall: 0.7077, 	specificity: 0.9970, 	f1: 0.8230
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 60: 100%|██████████| 6195/6195 [06:12<00:00, 16.64it/s, loss=0.485]
Train Epoch 60 ==> 	accuracy: 0.8416, 	precision: 0.9995, 	recall: 0.6836, 	specificity: 0.9996, 	f1: 0.8119
Test Epoch 60: 100%|██████████| 1715/1715 [00:43<00:00, 39.17it/s, loss=0.282]
Test Epoch 60 ==> 	accuracy: 0.9398, 	precision: 0.9793, 	recall: 0.7150, 	specificity: 0.9962, 	f1: 0.8266
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 61: 100%|██████████| 6195/6195 [06:16<00:00, 16.46it/s, loss=0.433]
Train Epoch 61 ==> 	accuracy: 0.8429, 	precision: 0.9995, 	recall: 0.6862, 	specificity: 0.9996, 	f1: 0.8137
Test Epoch 61: 100%|██████████| 1715/1715 [00:39<00:00, 43.01it/s, loss=0.372]
Test Epoch 61 ==> 	accuracy: 0.9379, 	precision: 0.9842, 	recall: 0.7018, 	specificity: 0.9972, 	f1: 0.8194
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 62: 100%|██████████| 6195/6195 [06:14<00:00, 16.54it/s, loss=0.36]
Train Epoch 62 ==> 	accuracy: 0.8441, 	precision: 0.9995, 	recall: 0.6885, 	specificity: 0.9997, 	f1: 0.8153
Test Epoch 62: 100%|██████████| 1715/1715 [00:39<00:00, 42.94it/s, loss=0.298]
Test Epoch 62 ==> 	accuracy: 0.9389, 	precision: 0.9812, 	recall: 0.7090, 	specificity: 0.9966, 	f1: 0.8232
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 63: 100%|██████████| 6195/6195 [06:12<00:00, 16.64it/s, loss=0.44]
Train Epoch 63 ==> 	accuracy: 0.8467, 	precision: 0.9995, 	recall: 0.6937, 	specificity: 0.9997, 	f1: 0.8190
Test Epoch 63: 100%|██████████| 1715/1715 [00:39<00:00, 43.69it/s, loss=0.252]
Test Epoch 63 ==> 	accuracy: 0.9383, 	precision: 0.9798, 	recall: 0.7070, 	specificity: 0.9963, 	f1: 0.8214
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 64: 100%|██████████| 6195/6195 [06:14<00:00, 16.56it/s, loss=0.386]
Train Epoch 64 ==> 	accuracy: 0.8448, 	precision: 0.9995, 	recall: 0.6900, 	specificity: 0.9997, 	f1: 0.8164
Test Epoch 64: 100%|██████████| 1715/1715 [00:43<00:00, 39.66it/s, loss=0.192]
Test Epoch 64 ==> 	accuracy: 0.9391, 	precision: 0.9804, 	recall: 0.7110, 	specificity: 0.9964, 	f1: 0.8242
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 65: 100%|██████████| 6195/6195 [06:14<00:00, 16.56it/s, loss=0.396]
Train Epoch 65 ==> 	accuracy: 0.8449, 	precision: 0.9995, 	recall: 0.6901, 	specificity: 0.9997, 	f1: 0.8165
Test Epoch 65: 100%|██████████| 1715/1715 [00:39<00:00, 43.10it/s, loss=0.308]
Test Epoch 65 ==> 	accuracy: 0.9397, 	precision: 0.9825, 	recall: 0.7123, 	specificity: 0.9968, 	f1: 0.8258
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 66: 100%|██████████| 6195/6195 [06:13<00:00, 16.58it/s, loss=0.357]
Train Epoch 66 ==> 	accuracy: 0.8459, 	precision: 0.9995, 	recall: 0.6921, 	specificity: 0.9996, 	f1: 0.8178
Test Epoch 66: 100%|██████████| 1715/1715 [00:43<00:00, 39.61it/s, loss=0.224]
Test Epoch 66 ==> 	accuracy: 0.9383, 	precision: 0.9824, 	recall: 0.7052, 	specificity: 0.9968, 	f1: 0.8211
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 67: 100%|██████████| 6195/6195 [06:10<00:00, 16.71it/s, loss=0.438]
Train Epoch 67 ==> 	accuracy: 0.8483, 	precision: 0.9995, 	recall: 0.6969, 	specificity: 0.9997, 	f1: 0.8213
Test Epoch 67: 100%|██████████| 1715/1715 [00:42<00:00, 40.80it/s, loss=0.115]
Test Epoch 67 ==> 	accuracy: 0.9404, 	precision: 0.9797, 	recall: 0.7181, 	specificity: 0.9963, 	f1: 0.8288
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 68: 100%|██████████| 6195/6195 [06:13<00:00, 16.61it/s, loss=0.369]
Train Epoch 68 ==> 	accuracy: 0.8463, 	precision: 0.9995, 	recall: 0.6930, 	specificity: 0.9997, 	f1: 0.8185
Test Epoch 68: 100%|██████████| 1715/1715 [00:44<00:00, 38.47it/s, loss=0.209]
Test Epoch 68 ==> 	accuracy: 0.9418, 	precision: 0.9799, 	recall: 0.7249, 	specificity: 0.9963, 	f1: 0.8334
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 69: 100%|██████████| 6195/6195 [06:19<00:00, 16.31it/s, loss=0.425]
Train Epoch 69 ==> 	accuracy: 0.8478, 	precision: 0.9996, 	recall: 0.6959, 	specificity: 0.9997, 	f1: 0.8205
Test Epoch 69: 100%|██████████| 1715/1715 [00:40<00:00, 42.49it/s, loss=1.13]
Test Epoch 69 ==> 	accuracy: 0.9405, 	precision: 0.9765, 	recall: 0.7210, 	specificity: 0.9956, 	f1: 0.8295
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 70: 100%|██████████| 6195/6195 [06:21<00:00, 16.23it/s, loss=0.397]
Train Epoch 70 ==> 	accuracy: 0.8469, 	precision: 0.9995, 	recall: 0.6942, 	specificity: 0.9997, 	f1: 0.8193
Test Epoch 70: 100%|██████████| 1715/1715 [00:39<00:00, 43.15it/s, loss=0.151]
Test Epoch 70 ==> 	accuracy: 0.9410, 	precision: 0.9802, 	recall: 0.7206, 	specificity: 0.9963, 	f1: 0.8306
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 71: 100%|██████████| 6195/6195 [06:23<00:00, 16.17it/s, loss=0.491]
Train Epoch 71 ==> 	accuracy: 0.8489, 	precision: 0.9995, 	recall: 0.6980, 	specificity: 0.9997, 	f1: 0.8220
Test Epoch 71: 100%|██████████| 1715/1715 [00:39<00:00, 43.00it/s, loss=0.59]
Test Epoch 71 ==> 	accuracy: 0.9418, 	precision: 0.9815, 	recall: 0.7239, 	specificity: 0.9966, 	f1: 0.8333
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 72: 100%|██████████| 6195/6195 [06:19<00:00, 16.31it/s, loss=0.596]
Train Epoch 72 ==> 	accuracy: 0.8480, 	precision: 0.9996, 	recall: 0.6962, 	specificity: 0.9997, 	f1: 0.8208
Test Epoch 72: 100%|██████████| 1715/1715 [00:43<00:00, 39.55it/s, loss=0.408]
Test Epoch 72 ==> 	accuracy: 0.9436, 	precision: 0.9793, 	recall: 0.7347, 	specificity: 0.9961, 	f1: 0.8395
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 73: 100%|██████████| 6195/6195 [06:18<00:00, 16.35it/s, loss=0.425]
Train Epoch 73 ==> 	accuracy: 0.8512, 	precision: 0.9996, 	recall: 0.7026, 	specificity: 0.9997, 	f1: 0.8252
Test Epoch 73: 100%|██████████| 1715/1715 [00:40<00:00, 41.95it/s, loss=0.218]
Test Epoch 73 ==> 	accuracy: 0.9429, 	precision: 0.9766, 	recall: 0.7333, 	specificity: 0.9956, 	f1: 0.8376
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 74: 100%|██████████| 6195/6195 [06:18<00:00, 16.37it/s, loss=0.653]
Train Epoch 74 ==> 	accuracy: 0.8492, 	precision: 0.9996, 	recall: 0.6986, 	specificity: 0.9997, 	f1: 0.8224
Test Epoch 74: 100%|██████████| 1715/1715 [00:42<00:00, 40.04it/s, loss=0.174]
Test Epoch 74 ==> 	accuracy: 0.9401, 	precision: 0.9820, 	recall: 0.7148, 	specificity: 0.9967, 	f1: 0.8274
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 75: 100%|██████████| 6195/6195 [06:19<00:00, 16.32it/s, loss=0.374]
Train Epoch 75 ==> 	accuracy: 0.8511, 	precision: 0.9996, 	recall: 0.7024, 	specificity: 0.9997, 	f1: 0.8251
Test Epoch 75: 100%|██████████| 1715/1715 [00:40<00:00, 42.60it/s, loss=0.71]
Test Epoch 75 ==> 	accuracy: 0.9401, 	precision: 0.9822, 	recall: 0.7146, 	specificity: 0.9968, 	f1: 0.8273
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 76: 100%|██████████| 6195/6195 [06:06<00:00, 16.91it/s, loss=0.321]
Train Epoch 76 ==> 	accuracy: 0.8515, 	precision: 0.9996, 	recall: 0.7034, 	specificity: 0.9997, 	f1: 0.8257
Test Epoch 76: 100%|██████████| 1715/1715 [00:44<00:00, 38.95it/s, loss=0.524]
Test Epoch 76 ==> 	accuracy: 0.9424, 	precision: 0.9788, 	recall: 0.7289, 	specificity: 0.9960, 	f1: 0.8355
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 77: 100%|██████████| 6195/6195 [06:17<00:00, 16.43it/s, loss=0.364]
Train Epoch 77 ==> 	accuracy: 0.8510, 	precision: 0.9996, 	recall: 0.7023, 	specificity: 0.9997, 	f1: 0.8250
Test Epoch 77: 100%|██████████| 1715/1715 [00:39<00:00, 43.48it/s, loss=0.371]
Test Epoch 77 ==> 	accuracy: 0.9434, 	precision: 0.9795, 	recall: 0.7335, 	specificity: 0.9961, 	f1: 0.8389
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 78: 100%|██████████| 6195/6195 [06:16<00:00, 16.44it/s, loss=0.397]
Train Epoch 78 ==> 	accuracy: 0.8520, 	precision: 0.9996, 	recall: 0.7044, 	specificity: 0.9997, 	f1: 0.8264
Test Epoch 78: 100%|██████████| 1715/1715 [00:42<00:00, 40.23it/s, loss=1.32]
Test Epoch 78 ==> 	accuracy: 0.9428, 	precision: 0.9788, 	recall: 0.7307, 	specificity: 0.9960, 	f1: 0.8367
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 79: 100%|██████████| 6195/6195 [06:21<00:00, 16.24it/s, loss=0.398]
Train Epoch 79 ==> 	accuracy: 0.8531, 	precision: 0.9996, 	recall: 0.7065, 	specificity: 0.9997, 	f1: 0.8279
Test Epoch 79: 100%|██████████| 1715/1715 [00:41<00:00, 41.05it/s, loss=0.253]
Test Epoch 79 ==> 	accuracy: 0.9424, 	precision: 0.9735, 	recall: 0.7329, 	specificity: 0.9950, 	f1: 0.8362
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 80: 100%|██████████| 6195/6195 [06:19<00:00, 16.33it/s, loss=0.401]
Train Epoch 80 ==> 	accuracy: 0.8529, 	precision: 0.9996, 	recall: 0.7062, 	specificity: 0.9997, 	f1: 0.8277
Test Epoch 80: 100%|██████████| 1715/1715 [00:42<00:00, 40.28it/s, loss=0.771]
Test Epoch 80 ==> 	accuracy: 0.9407, 	precision: 0.9786, 	recall: 0.7204, 	specificity: 0.9960, 	f1: 0.8299
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 81: 100%|██████████| 6195/6195 [06:20<00:00, 16.27it/s, loss=0.46]
Train Epoch 81 ==> 	accuracy: 0.8533, 	precision: 0.9996, 	recall: 0.7068, 	specificity: 0.9997, 	f1: 0.8281
Test Epoch 81: 100%|██████████| 1715/1715 [00:41<00:00, 41.12it/s, loss=0.149]
Test Epoch 81 ==> 	accuracy: 0.9434, 	precision: 0.9796, 	recall: 0.7334, 	specificity: 0.9962, 	f1: 0.8388
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 82: 100%|██████████| 6195/6195 [06:15<00:00, 16.49it/s, loss=0.49]
Train Epoch 82 ==> 	accuracy: 0.8543, 	precision: 0.9996, 	recall: 0.7090, 	specificity: 0.9997, 	f1: 0.8295
Test Epoch 82: 100%|██████████| 1715/1715 [00:41<00:00, 41.01it/s, loss=0.581]
Test Epoch 82 ==> 	accuracy: 0.9436, 	precision: 0.9792, 	recall: 0.7348, 	specificity: 0.9961, 	f1: 0.8396
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 83: 100%|██████████| 6195/6195 [06:14<00:00, 16.54it/s, loss=0.489]
Train Epoch 83 ==> 	accuracy: 0.8548, 	precision: 0.9996, 	recall: 0.7098, 	specificity: 0.9997, 	f1: 0.8302
Test Epoch 83: 100%|██████████| 1715/1715 [00:39<00:00, 43.29it/s, loss=0.443]
Test Epoch 83 ==> 	accuracy: 0.9431, 	precision: 0.9697, 	recall: 0.7394, 	specificity: 0.9942, 	f1: 0.8391
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 84: 100%|██████████| 6195/6195 [06:20<00:00, 16.28it/s, loss=0.468]
Train Epoch 84 ==> 	accuracy: 0.8563, 	precision: 0.9996, 	recall: 0.7128, 	specificity: 0.9997, 	f1: 0.8322
Test Epoch 84: 100%|██████████| 1715/1715 [00:41<00:00, 40.88it/s, loss=0.642]
Test Epoch 84 ==> 	accuracy: 0.9403, 	precision: 0.9814, 	recall: 0.7161, 	specificity: 0.9966, 	f1: 0.8280
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 85: 100%|██████████| 6195/6195 [06:24<00:00, 16.11it/s, loss=0.476]
Train Epoch 85 ==> 	accuracy: 0.8559, 	precision: 0.9996, 	recall: 0.7121, 	specificity: 0.9997, 	f1: 0.8317
Test Epoch 85: 100%|██████████| 1715/1715 [00:38<00:00, 45.05it/s, loss=0.336]
Test Epoch 85 ==> 	accuracy: 0.9433, 	precision: 0.9773, 	recall: 0.7348, 	specificity: 0.9957, 	f1: 0.8389
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 86: 100%|██████████| 6195/6195 [06:24<00:00, 16.13it/s, loss=0.385]
Train Epoch 86 ==> 	accuracy: 0.8567, 	precision: 0.9996, 	recall: 0.7137, 	specificity: 0.9997, 	f1: 0.8328
Test Epoch 86: 100%|██████████| 1715/1715 [00:39<00:00, 43.77it/s, loss=0.388]
Test Epoch 86 ==> 	accuracy: 0.9429, 	precision: 0.9733, 	recall: 0.7358, 	specificity: 0.9949, 	f1: 0.8381
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 87: 100%|██████████| 6195/6195 [06:14<00:00, 16.54it/s, loss=0.435]
Train Epoch 87 ==> 	accuracy: 0.8551, 	precision: 0.9996, 	recall: 0.7105, 	specificity: 0.9997, 	f1: 0.8306
Test Epoch 87: 100%|██████████| 1715/1715 [00:39<00:00, 43.33it/s, loss=0.293]
Test Epoch 87 ==> 	accuracy: 0.9430, 	precision: 0.9801, 	recall: 0.7311, 	specificity: 0.9963, 	f1: 0.8375
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 88: 100%|██████████| 6195/6195 [06:16<00:00, 16.45it/s, loss=0.335]
Train Epoch 88 ==> 	accuracy: 0.8580, 	precision: 0.9996, 	recall: 0.7162, 	specificity: 0.9997, 	f1: 0.8345
Test Epoch 88: 100%|██████████| 1715/1715 [00:38<00:00, 44.29it/s, loss=0.351]
Test Epoch 88 ==> 	accuracy: 0.9426, 	precision: 0.9731, 	recall: 0.7344, 	specificity: 0.9949, 	f1: 0.8371
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 89: 100%|██████████| 6195/6195 [06:20<00:00, 16.27it/s, loss=0.393]
Train Epoch 89 ==> 	accuracy: 0.8573, 	precision: 0.9996, 	recall: 0.7150, 	specificity: 0.9997, 	f1: 0.8337
Test Epoch 89: 100%|██████████| 1715/1715 [00:41<00:00, 41.80it/s, loss=0.277]
Test Epoch 89 ==> 	accuracy: 0.9432, 	precision: 0.9721, 	recall: 0.7384, 	specificity: 0.9947, 	f1: 0.8393
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 90: 100%|██████████| 6195/6195 [06:10<00:00, 16.72it/s, loss=0.388]
Train Epoch 90 ==> 	accuracy: 0.8579, 	precision: 0.9996, 	recall: 0.7161, 	specificity: 0.9997, 	f1: 0.8344
Test Epoch 90: 100%|██████████| 1715/1715 [00:40<00:00, 42.68it/s, loss=2.74]
Test Epoch 90 ==> 	accuracy: 0.9445, 	precision: 0.9744, 	recall: 0.7429, 	specificity: 0.9951, 	f1: 0.8430
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 91: 100%|██████████| 6195/6195 [06:23<00:00, 16.14it/s, loss=0.296]
Train Epoch 91 ==> 	accuracy: 0.8577, 	precision: 0.9996, 	recall: 0.7157, 	specificity: 0.9997, 	f1: 0.8342
Test Epoch 91: 100%|██████████| 1715/1715 [00:36<00:00, 47.51it/s, loss=0.184]
Test Epoch 91 ==> 	accuracy: 0.9435, 	precision: 0.9748, 	recall: 0.7376, 	specificity: 0.9952, 	f1: 0.8398
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 92: 100%|██████████| 6195/6195 [06:16<00:00, 16.46it/s, loss=0.399]
Train Epoch 92 ==> 	accuracy: 0.8577, 	precision: 0.9996, 	recall: 0.7156, 	specificity: 0.9997, 	f1: 0.8341
Test Epoch 92: 100%|██████████| 1715/1715 [00:38<00:00, 44.22it/s, loss=0.117]
Test Epoch 92 ==> 	accuracy: 0.9442, 	precision: 0.9726, 	recall: 0.7432, 	specificity: 0.9947, 	f1: 0.8426
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 93: 100%|██████████| 6195/6195 [06:14<00:00, 16.56it/s, loss=0.446]
Train Epoch 93 ==> 	accuracy: 0.8580, 	precision: 0.9996, 	recall: 0.7162, 	specificity: 0.9997, 	f1: 0.8345
Test Epoch 93: 100%|██████████| 1715/1715 [00:38<00:00, 44.64it/s, loss=0.286]
Test Epoch 93 ==> 	accuracy: 0.9445, 	precision: 0.9758, 	recall: 0.7422, 	specificity: 0.9954, 	f1: 0.8431
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 94: 100%|██████████| 6195/6195 [06:13<00:00, 16.58it/s, loss=0.326]
Train Epoch 94 ==> 	accuracy: 0.8570, 	precision: 0.9996, 	recall: 0.7143, 	specificity: 0.9997, 	f1: 0.8332
Test Epoch 94: 100%|██████████| 1715/1715 [00:37<00:00, 45.27it/s, loss=0.17]
Test Epoch 94 ==> 	accuracy: 0.9435, 	precision: 0.9797, 	recall: 0.7338, 	specificity: 0.9962, 	f1: 0.8391
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 95: 100%|██████████| 6195/6195 [06:15<00:00, 16.48it/s, loss=0.4]
Train Epoch 95 ==> 	accuracy: 0.8583, 	precision: 0.9996, 	recall: 0.7168, 	specificity: 0.9997, 	f1: 0.8349
Test Epoch 95: 100%|██████████| 1715/1715 [00:39<00:00, 43.34it/s, loss=0.236]
Test Epoch 95 ==> 	accuracy: 0.9424, 	precision: 0.9792, 	recall: 0.7286, 	specificity: 0.9961, 	f1: 0.8356
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 96: 100%|██████████| 6195/6195 [06:17<00:00, 16.43it/s, loss=0.432]
Train Epoch 96 ==> 	accuracy: 0.8595, 	precision: 0.9996, 	recall: 0.7192, 	specificity: 0.9997, 	f1: 0.8365
Test Epoch 96: 100%|██████████| 1715/1715 [00:41<00:00, 41.25it/s, loss=2.17]
Test Epoch 96 ==> 	accuracy: 0.9431, 	precision: 0.9762, 	recall: 0.7345, 	specificity: 0.9955, 	f1: 0.8383
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 97: 100%|██████████| 6195/6195 [06:14<00:00, 16.55it/s, loss=0.498]
Train Epoch 97 ==> 	accuracy: 0.8591, 	precision: 0.9996, 	recall: 0.7186, 	specificity: 0.9997, 	f1: 0.8361
Test Epoch 97: 100%|██████████| 1715/1715 [00:41<00:00, 41.53it/s, loss=0.507]
Test Epoch 97 ==> 	accuracy: 0.9437, 	precision: 0.9758, 	recall: 0.7379, 	specificity: 0.9954, 	f1: 0.8403
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 98: 100%|██████████| 6195/6195 [06:23<00:00, 16.17it/s, loss=0.44]
Train Epoch 98 ==> 	accuracy: 0.8605, 	precision: 0.9996, 	recall: 0.7213, 	specificity: 0.9997, 	f1: 0.8380
Test Epoch 98: 100%|██████████| 1715/1715 [00:41<00:00, 41.83it/s, loss=2.52]
Test Epoch 98 ==> 	accuracy: 0.9448, 	precision: 0.9737, 	recall: 0.7450, 	specificity: 0.9949, 	f1: 0.8441
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 99: 100%|██████████| 6195/6195 [06:14<00:00, 16.53it/s, loss=0.431]
Train Epoch 99 ==> 	accuracy: 0.8614, 	precision: 0.9996, 	recall: 0.7230, 	specificity: 0.9997, 	f1: 0.8391
Test Epoch 99: 100%|██████████| 1715/1715 [00:39<00:00, 43.35it/s, loss=0.264]
Test Epoch 99 ==> 	accuracy: 0.9446, 	precision: 0.9737, 	recall: 0.7442, 	specificity: 0.9950, 	f1: 0.8436
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 100: 100%|██████████| 6195/6195 [06:16<00:00, 16.46it/s, loss=0.387]
Train Epoch 100 ==> 	accuracy: 0.8609, 	precision: 0.9996, 	recall: 0.7221, 	specificity: 0.9997, 	f1: 0.8385
Test Epoch 100: 100%|██████████| 1715/1715 [00:44<00:00, 38.83it/s, loss=0.152]
Test Epoch 100 ==> 	accuracy: 0.9440, 	precision: 0.9778, 	recall: 0.7380, 	specificity: 0.9958, 	f1: 0.8411
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 101: 100%|██████████| 6195/6195 [06:21<00:00, 16.24it/s, loss=0.346]
Train Epoch 101 ==> 	accuracy: 0.8600, 	precision: 0.9996, 	recall: 0.7204, 	specificity: 0.9997, 	f1: 0.8373
Test Epoch 101: 100%|██████████| 1715/1715 [00:39<00:00, 43.42it/s, loss=1.57]
Test Epoch 101 ==> 	accuracy: 0.9446, 	precision: 0.9733, 	recall: 0.7447, 	specificity: 0.9949, 	f1: 0.8438
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 102: 100%|██████████| 6195/6195 [06:15<00:00, 16.50it/s, loss=0.397]
Train Epoch 102 ==> 	accuracy: 0.8632, 	precision: 0.9996, 	recall: 0.7267, 	specificity: 0.9997, 	f1: 0.8416
Test Epoch 102: 100%|██████████| 1715/1715 [00:40<00:00, 42.82it/s, loss=2.07]
Test Epoch 102 ==> 	accuracy: 0.9444, 	precision: 0.9704, 	recall: 0.7460, 	specificity: 0.9943, 	f1: 0.8435
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 103: 100%|██████████| 6195/6195 [06:14<00:00, 16.54it/s, loss=0.42]
Train Epoch 103 ==> 	accuracy: 0.8608, 	precision: 0.9996, 	recall: 0.7219, 	specificity: 0.9997, 	f1: 0.8383
Test Epoch 103: 100%|██████████| 1715/1715 [00:40<00:00, 42.80it/s, loss=0.184]
Test Epoch 103 ==> 	accuracy: 0.9446, 	precision: 0.9737, 	recall: 0.7440, 	specificity: 0.9950, 	f1: 0.8435
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 104: 100%|██████████| 6195/6195 [06:10<00:00, 16.71it/s, loss=0.398]
Train Epoch 104 ==> 	accuracy: 0.8596, 	precision: 0.9996, 	recall: 0.7195, 	specificity: 0.9997, 	f1: 0.8367
Test Epoch 104: 100%|██████████| 1715/1715 [00:37<00:00, 45.62it/s, loss=0.162]
Test Epoch 104 ==> 	accuracy: 0.9441, 	precision: 0.9783, 	recall: 0.7377, 	specificity: 0.9959, 	f1: 0.8411
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 105: 100%|██████████| 6195/6195 [06:08<00:00, 16.82it/s, loss=0.436]
Train Epoch 105 ==> 	accuracy: 0.8610, 	precision: 0.9996, 	recall: 0.7222, 	specificity: 0.9997, 	f1: 0.8386
Test Epoch 105: 100%|██████████| 1715/1715 [00:38<00:00, 45.13it/s, loss=0.746]
Test Epoch 105 ==> 	accuracy: 0.9450, 	precision: 0.9753, 	recall: 0.7450, 	specificity: 0.9953, 	f1: 0.8447
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 106: 100%|██████████| 6195/6195 [05:59<00:00, 17.22it/s, loss=0.393]
Train Epoch 106 ==> 	accuracy: 0.8626, 	precision: 0.9996, 	recall: 0.7254, 	specificity: 0.9997, 	f1: 0.8407
Test Epoch 106: 100%|██████████| 1715/1715 [00:42<00:00, 40.30it/s, loss=0.199]
Test Epoch 106 ==> 	accuracy: 0.9447, 	precision: 0.9726, 	recall: 0.7458, 	specificity: 0.9947, 	f1: 0.8442
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 107: 100%|██████████| 6195/6195 [06:09<00:00, 16.75it/s, loss=0.418]
Train Epoch 107 ==> 	accuracy: 0.8612, 	precision: 0.9996, 	recall: 0.7226, 	specificity: 0.9997, 	f1: 0.8388
Test Epoch 107: 100%|██████████| 1715/1715 [00:39<00:00, 43.04it/s, loss=0.36]
Test Epoch 107 ==> 	accuracy: 0.9438, 	precision: 0.9780, 	recall: 0.7366, 	specificity: 0.9958, 	f1: 0.8403
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 108: 100%|██████████| 6195/6195 [06:03<00:00, 17.05it/s, loss=0.414]
Train Epoch 108 ==> 	accuracy: 0.8597, 	precision: 0.9996, 	recall: 0.7197, 	specificity: 0.9997, 	f1: 0.8369
Test Epoch 108: 100%|██████████| 1715/1715 [00:38<00:00, 44.50it/s, loss=1.12]
Test Epoch 108 ==> 	accuracy: 0.9442, 	precision: 0.9737, 	recall: 0.7419, 	specificity: 0.9950, 	f1: 0.8421
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 109: 100%|██████████| 6195/6195 [06:05<00:00, 16.97it/s, loss=0.456]
Train Epoch 109 ==> 	accuracy: 0.8649, 	precision: 0.9997, 	recall: 0.7300, 	specificity: 0.9997, 	f1: 0.8438
Test Epoch 109: 100%|██████████| 1715/1715 [00:39<00:00, 43.64it/s, loss=3.89]
Test Epoch 109 ==> 	accuracy: 0.9455, 	precision: 0.9717, 	recall: 0.7505, 	specificity: 0.9945, 	f1: 0.8469
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 110: 100%|██████████| 6195/6195 [06:13<00:00, 16.60it/s, loss=0.378]
Train Epoch 110 ==> 	accuracy: 0.8622, 	precision: 0.9996, 	recall: 0.7246, 	specificity: 0.9997, 	f1: 0.8402
Test Epoch 110: 100%|██████████| 1715/1715 [00:38<00:00, 44.24it/s, loss=1.65]
Test Epoch 110 ==> 	accuracy: 0.9462, 	precision: 0.9692, 	recall: 0.7560, 	specificity: 0.9940, 	f1: 0.8494
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 111: 100%|██████████| 6195/6195 [06:04<00:00, 17.00it/s, loss=0.448]
Train Epoch 111 ==> 	accuracy: 0.8643, 	precision: 0.9996, 	recall: 0.7288, 	specificity: 0.9997, 	f1: 0.8430
Test Epoch 111: 100%|██████████| 1715/1715 [00:40<00:00, 42.65it/s, loss=2.42]
Test Epoch 111 ==> 	accuracy: 0.9452, 	precision: 0.9764, 	recall: 0.7452, 	specificity: 0.9955, 	f1: 0.8452
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 112: 100%|██████████| 6195/6195 [06:04<00:00, 17.01it/s, loss=0.384]
Train Epoch 112 ==> 	accuracy: 0.8624, 	precision: 0.9996, 	recall: 0.7250, 	specificity: 0.9997, 	f1: 0.8405
Test Epoch 112: 100%|██████████| 1715/1715 [00:40<00:00, 42.62it/s, loss=0.27]
Test Epoch 112 ==> 	accuracy: 0.9450, 	precision: 0.9688, 	recall: 0.7501, 	specificity: 0.9939, 	f1: 0.8455
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 113: 100%|██████████| 6195/6195 [06:20<00:00, 16.28it/s, loss=0.381]
Train Epoch 113 ==> 	accuracy: 0.8650, 	precision: 0.9996, 	recall: 0.7303, 	specificity: 0.9997, 	f1: 0.8440
Test Epoch 113: 100%|██████████| 1715/1715 [00:40<00:00, 42.76it/s, loss=0.393]
Test Epoch 113 ==> 	accuracy: 0.9452, 	precision: 0.9707, 	recall: 0.7499, 	specificity: 0.9943, 	f1: 0.8461
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 114: 100%|██████████| 6195/6195 [06:11<00:00, 16.66it/s, loss=0.358]
Train Epoch 114 ==> 	accuracy: 0.8628, 	precision: 0.9996, 	recall: 0.7258, 	specificity: 0.9997, 	f1: 0.8410
Test Epoch 114: 100%|██████████| 1715/1715 [00:42<00:00, 39.96it/s, loss=0.335]
Test Epoch 114 ==> 	accuracy: 0.9446, 	precision: 0.9782, 	recall: 0.7407, 	specificity: 0.9959, 	f1: 0.8431
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 115: 100%|██████████| 6195/6195 [06:17<00:00, 16.42it/s, loss=0.286]
Train Epoch 115 ==> 	accuracy: 0.8649, 	precision: 0.9996, 	recall: 0.7300, 	specificity: 0.9997, 	f1: 0.8438
Test Epoch 115: 100%|██████████| 1715/1715 [00:43<00:00, 39.70it/s, loss=0.349]
Test Epoch 115 ==> 	accuracy: 0.9451, 	precision: 0.9764, 	recall: 0.7446, 	specificity: 0.9955, 	f1: 0.8449
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 116: 100%|██████████| 6195/6195 [06:15<00:00, 16.51it/s, loss=0.403]
Train Epoch 116 ==> 	accuracy: 0.8629, 	precision: 0.9996, 	recall: 0.7262, 	specificity: 0.9997, 	f1: 0.8412
Test Epoch 116: 100%|██████████| 1715/1715 [00:40<00:00, 42.39it/s, loss=0.253]
Test Epoch 116 ==> 	accuracy: 0.9456, 	precision: 0.9763, 	recall: 0.7471, 	specificity: 0.9954, 	f1: 0.8464
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 117: 100%|██████████| 6195/6195 [06:12<00:00, 16.63it/s, loss=0.446]
Train Epoch 117 ==> 	accuracy: 0.8635, 	precision: 0.9997, 	recall: 0.7272, 	specificity: 0.9998, 	f1: 0.8420
Test Epoch 117: 100%|██████████| 1715/1715 [00:38<00:00, 45.06it/s, loss=0.225]
Test Epoch 117 ==> 	accuracy: 0.9431, 	precision: 0.9752, 	recall: 0.7352, 	specificity: 0.9953, 	f1: 0.8383
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 118: 100%|██████████| 6195/6195 [06:11<00:00, 16.66it/s, loss=0.364]
Train Epoch 118 ==> 	accuracy: 0.8640, 	precision: 0.9996, 	recall: 0.7283, 	specificity: 0.9997, 	f1: 0.8427
Test Epoch 118: 100%|██████████| 1715/1715 [00:38<00:00, 44.19it/s, loss=0.468]
Test Epoch 118 ==> 	accuracy: 0.9452, 	precision: 0.9739, 	recall: 0.7469, 	specificity: 0.9950, 	f1: 0.8454
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 119: 100%|██████████| 6195/6195 [06:06<00:00, 16.90it/s, loss=0.348]
Train Epoch 119 ==> 	accuracy: 0.8640, 	precision: 0.9996, 	recall: 0.7283, 	specificity: 0.9997, 	f1: 0.8426
Test Epoch 119: 100%|██████████| 1715/1715 [00:41<00:00, 41.29it/s, loss=0.306]
Test Epoch 119 ==> 	accuracy: 0.9449, 	precision: 0.9761, 	recall: 0.7435, 	specificity: 0.9954, 	f1: 0.8441
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 120: 100%|██████████| 6195/6195 [06:08<00:00, 16.82it/s, loss=0.458]
Train Epoch 120 ==> 	accuracy: 0.8649, 	precision: 0.9997, 	recall: 0.7300, 	specificity: 0.9997, 	f1: 0.8438
Test Epoch 120: 100%|██████████| 1715/1715 [00:42<00:00, 40.81it/s, loss=1.25]
Test Epoch 120 ==> 	accuracy: 0.9458, 	precision: 0.9706, 	recall: 0.7529, 	specificity: 0.9943, 	f1: 0.8480
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 121: 100%|██████████| 6195/6195 [06:07<00:00, 16.85it/s, loss=0.473]
Train Epoch 121 ==> 	accuracy: 0.8647, 	precision: 0.9996, 	recall: 0.7298, 	specificity: 0.9997, 	f1: 0.8436
Test Epoch 121: 100%|██████████| 1715/1715 [00:41<00:00, 41.41it/s, loss=0.695]
Test Epoch 121 ==> 	accuracy: 0.9455, 	precision: 0.9719, 	recall: 0.7500, 	specificity: 0.9946, 	f1: 0.8467
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 122: 100%|██████████| 6195/6195 [06:07<00:00, 16.84it/s, loss=0.297]
Train Epoch 122 ==> 	accuracy: 0.8666, 	precision: 0.9996, 	recall: 0.7334, 	specificity: 0.9997, 	f1: 0.8461
Test Epoch 122: 100%|██████████| 1715/1715 [00:40<00:00, 42.01it/s, loss=0.256]
Test Epoch 122 ==> 	accuracy: 0.9460, 	precision: 0.9694, 	recall: 0.7550, 	specificity: 0.9940, 	f1: 0.8489
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 123: 100%|██████████| 6195/6195 [06:13<00:00, 16.56it/s, loss=0.419]
Train Epoch 123 ==> 	accuracy: 0.8656, 	precision: 0.9997, 	recall: 0.7313, 	specificity: 0.9998, 	f1: 0.8447
Test Epoch 123: 100%|██████████| 1715/1715 [00:39<00:00, 43.44it/s, loss=2.44]
Test Epoch 123 ==> 	accuracy: 0.9458, 	precision: 0.9731, 	recall: 0.7508, 	specificity: 0.9948, 	f1: 0.8477
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 124: 100%|██████████| 6195/6195 [06:10<00:00, 16.72it/s, loss=0.363]
Train Epoch 124 ==> 	accuracy: 0.8664, 	precision: 0.9997, 	recall: 0.7330, 	specificity: 0.9998, 	f1: 0.8458
Test Epoch 124: 100%|██████████| 1715/1715 [00:38<00:00, 44.21it/s, loss=0.191]
Test Epoch 124 ==> 	accuracy: 0.9452, 	precision: 0.9696, 	recall: 0.7509, 	specificity: 0.9941, 	f1: 0.8463
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 125: 100%|██████████| 6195/6195 [06:07<00:00, 16.85it/s, loss=0.277]
Train Epoch 125 ==> 	accuracy: 0.8658, 	precision: 0.9997, 	recall: 0.7319, 	specificity: 0.9997, 	f1: 0.8451
Test Epoch 125: 100%|██████████| 1715/1715 [00:40<00:00, 41.96it/s, loss=0.412]
Test Epoch 125 ==> 	accuracy: 0.9457, 	precision: 0.9721, 	recall: 0.7510, 	specificity: 0.9946, 	f1: 0.8474
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 126: 100%|██████████| 6195/6195 [06:06<00:00, 16.91it/s, loss=0.408]
Train Epoch 126 ==> 	accuracy: 0.8665, 	precision: 0.9996, 	recall: 0.7333, 	specificity: 0.9997, 	f1: 0.8460
Test Epoch 126: 100%|██████████| 1715/1715 [00:39<00:00, 43.83it/s, loss=0.349]
Test Epoch 126 ==> 	accuracy: 0.9464, 	precision: 0.9691, 	recall: 0.7571, 	specificity: 0.9939, 	f1: 0.8501
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 127: 100%|██████████| 6195/6195 [06:13<00:00, 16.59it/s, loss=0.3]
Train Epoch 127 ==> 	accuracy: 0.8642, 	precision: 0.9996, 	recall: 0.7288, 	specificity: 0.9997, 	f1: 0.8430
Test Epoch 127: 100%|██████████| 1715/1715 [00:41<00:00, 41.74it/s, loss=0.153]
Test Epoch 127 ==> 	accuracy: 0.9459, 	precision: 0.9729, 	recall: 0.7514, 	specificity: 0.9947, 	f1: 0.8479
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 128: 100%|██████████| 6195/6195 [06:07<00:00, 16.87it/s, loss=0.459]
Train Epoch 128 ==> 	accuracy: 0.8640, 	precision: 0.9996, 	recall: 0.7284, 	specificity: 0.9997, 	f1: 0.8427
Test Epoch 128: 100%|██████████| 1715/1715 [00:39<00:00, 43.84it/s, loss=0.273]
Test Epoch 128 ==> 	accuracy: 0.9460, 	precision: 0.9806, 	recall: 0.7460, 	specificity: 0.9963, 	f1: 0.8474
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 129: 100%|██████████| 6195/6195 [06:00<00:00, 17.20it/s, loss=0.415]
Train Epoch 129 ==> 	accuracy: 0.8632, 	precision: 0.9997, 	recall: 0.7267, 	specificity: 0.9998, 	f1: 0.8416
Test Epoch 129: 100%|██████████| 1715/1715 [00:39<00:00, 43.53it/s, loss=0.145]
Test Epoch 129 ==> 	accuracy: 0.9461, 	precision: 0.9806, 	recall: 0.7463, 	specificity: 0.9963, 	f1: 0.8475
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 130: 100%|██████████| 6195/6195 [06:11<00:00, 16.68it/s, loss=0.573]
Train Epoch 130 ==> 	accuracy: 0.8639, 	precision: 0.9997, 	recall: 0.7280, 	specificity: 0.9998, 	f1: 0.8425
Test Epoch 130: 100%|██████████| 1715/1715 [00:40<00:00, 42.85it/s, loss=1.78]
Test Epoch 130 ==> 	accuracy: 0.9465, 	precision: 0.9814, 	recall: 0.7475, 	specificity: 0.9964, 	f1: 0.8486
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 131: 100%|██████████| 6195/6195 [06:14<00:00, 16.55it/s, loss=0.437]
Train Epoch 131 ==> 	accuracy: 0.8643, 	precision: 0.9997, 	recall: 0.7288, 	specificity: 0.9998, 	f1: 0.8430
Test Epoch 131: 100%|██████████| 1715/1715 [00:39<00:00, 43.35it/s, loss=0.182]
Test Epoch 131 ==> 	accuracy: 0.9467, 	precision: 0.9800, 	recall: 0.7500, 	specificity: 0.9962, 	f1: 0.8497
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 132: 100%|██████████| 6195/6195 [06:17<00:00, 16.41it/s, loss=0.378]
Train Epoch 132 ==> 	accuracy: 0.8645, 	precision: 0.9997, 	recall: 0.7293, 	specificity: 0.9998, 	f1: 0.8434
Test Epoch 132: 100%|██████████| 1715/1715 [00:42<00:00, 40.31it/s, loss=0.32]
Test Epoch 132 ==> 	accuracy: 0.9466, 	precision: 0.9809, 	recall: 0.7487, 	specificity: 0.9963, 	f1: 0.8492
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 133: 100%|██████████| 6195/6195 [06:17<00:00, 16.42it/s, loss=0.382]
Train Epoch 133 ==> 	accuracy: 0.8650, 	precision: 0.9996, 	recall: 0.7302, 	specificity: 0.9997, 	f1: 0.8439
Test Epoch 133: 100%|██████████| 1715/1715 [00:42<00:00, 40.82it/s, loss=0.12]
Test Epoch 133 ==> 	accuracy: 0.9477, 	precision: 0.9783, 	recall: 0.7565, 	specificity: 0.9958, 	f1: 0.8532
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 134: 100%|██████████| 6195/6195 [06:11<00:00, 16.67it/s, loss=0.455]
Train Epoch 134 ==> 	accuracy: 0.8643, 	precision: 0.9997, 	recall: 0.7289, 	specificity: 0.9998, 	f1: 0.8431
Test Epoch 134: 100%|██████████| 1715/1715 [00:39<00:00, 43.12it/s, loss=1.23]
Test Epoch 134 ==> 	accuracy: 0.9463, 	precision: 0.9804, 	recall: 0.7477, 	specificity: 0.9963, 	f1: 0.8484
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 135: 100%|██████████| 6195/6195 [06:16<00:00, 16.45it/s, loss=0.346]
Train Epoch 135 ==> 	accuracy: 0.8670, 	precision: 0.9997, 	recall: 0.7343, 	specificity: 0.9998, 	f1: 0.8467
Test Epoch 135: 100%|██████████| 1715/1715 [00:36<00:00, 47.05it/s, loss=1.08]
Test Epoch 135 ==> 	accuracy: 0.9474, 	precision: 0.9772, 	recall: 0.7554, 	specificity: 0.9956, 	f1: 0.8521
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 136: 100%|██████████| 6195/6195 [06:08<00:00, 16.80it/s, loss=0.368]
Train Epoch 136 ==> 	accuracy: 0.8661, 	precision: 0.9997, 	recall: 0.7324, 	specificity: 0.9997, 	f1: 0.8454
Test Epoch 136: 100%|██████████| 1715/1715 [00:38<00:00, 44.78it/s, loss=0.479]
Test Epoch 136 ==> 	accuracy: 0.9466, 	precision: 0.9783, 	recall: 0.7506, 	specificity: 0.9958, 	f1: 0.8495
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 137: 100%|██████████| 6195/6195 [06:15<00:00, 16.50it/s, loss=0.374]
Train Epoch 137 ==> 	accuracy: 0.8668, 	precision: 0.9997, 	recall: 0.7339, 	specificity: 0.9998, 	f1: 0.8464
Test Epoch 137: 100%|██████████| 1715/1715 [00:40<00:00, 42.36it/s, loss=0.308]
Test Epoch 137 ==> 	accuracy: 0.9434, 	precision: 0.9808, 	recall: 0.7322, 	specificity: 0.9964, 	f1: 0.8385
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 138: 100%|██████████| 6195/6195 [06:08<00:00, 16.83it/s, loss=0.404]
Train Epoch 138 ==> 	accuracy: 0.8646, 	precision: 0.9997, 	recall: 0.7294, 	specificity: 0.9998, 	f1: 0.8434
Test Epoch 138: 100%|██████████| 1715/1715 [00:37<00:00, 45.41it/s, loss=0.138]
Test Epoch 138 ==> 	accuracy: 0.9467, 	precision: 0.9778, 	recall: 0.7513, 	specificity: 0.9957, 	f1: 0.8497
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 139: 100%|██████████| 6195/6195 [06:01<00:00, 17.15it/s, loss=0.36]
Train Epoch 139 ==> 	accuracy: 0.8682, 	precision: 0.9997, 	recall: 0.7367, 	specificity: 0.9998, 	f1: 0.8483
Test Epoch 139: 100%|██████████| 1715/1715 [00:40<00:00, 42.37it/s, loss=0.359]
Test Epoch 139 ==> 	accuracy: 0.9474, 	precision: 0.9769, 	recall: 0.7561, 	specificity: 0.9955, 	f1: 0.8524
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 140: 100%|██████████| 6195/6195 [06:09<00:00, 16.74it/s, loss=0.452]
Train Epoch 140 ==> 	accuracy: 0.8661, 	precision: 0.9997, 	recall: 0.7325, 	specificity: 0.9997, 	f1: 0.8455
Test Epoch 140: 100%|██████████| 1715/1715 [00:36<00:00, 47.20it/s, loss=0.33]
Test Epoch 140 ==> 	accuracy: 0.9474, 	precision: 0.9770, 	recall: 0.7556, 	specificity: 0.9955, 	f1: 0.8521
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 141: 100%|██████████| 6195/6195 [06:08<00:00, 16.82it/s, loss=0.38]
Train Epoch 141 ==> 	accuracy: 0.8646, 	precision: 0.9996, 	recall: 0.7295, 	specificity: 0.9997, 	f1: 0.8435
Test Epoch 141: 100%|██████████| 1715/1715 [00:38<00:00, 45.06it/s, loss=0.448]
Test Epoch 141 ==> 	accuracy: 0.9469, 	precision: 0.9782, 	recall: 0.7521, 	specificity: 0.9958, 	f1: 0.8504
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 142: 100%|██████████| 6195/6195 [06:06<00:00, 16.89it/s, loss=0.344]
Train Epoch 142 ==> 	accuracy: 0.8665, 	precision: 0.9996, 	recall: 0.7332, 	specificity: 0.9997, 	f1: 0.8460
Test Epoch 142: 100%|██████████| 1715/1715 [00:40<00:00, 41.90it/s, loss=0.236]
Test Epoch 142 ==> 	accuracy: 0.9467, 	precision: 0.9790, 	recall: 0.7505, 	specificity: 0.9960, 	f1: 0.8496
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 143: 100%|██████████| 6195/6195 [06:13<00:00, 16.59it/s, loss=0.404]
Train Epoch 143 ==> 	accuracy: 0.8658, 	precision: 0.9997, 	recall: 0.7319, 	specificity: 0.9998, 	f1: 0.8451
Test Epoch 143: 100%|██████████| 1715/1715 [00:42<00:00, 40.51it/s, loss=0.263]
Test Epoch 143 ==> 	accuracy: 0.9462, 	precision: 0.9807, 	recall: 0.7466, 	specificity: 0.9963, 	f1: 0.8477
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 144: 100%|██████████| 6195/6195 [06:10<00:00, 16.72it/s, loss=0.294]
Train Epoch 144 ==> 	accuracy: 0.8672, 	precision: 0.9997, 	recall: 0.7346, 	specificity: 0.9998, 	f1: 0.8469
Test Epoch 144: 100%|██████████| 1715/1715 [00:38<00:00, 45.09it/s, loss=0.213]
Test Epoch 144 ==> 	accuracy: 0.9471, 	precision: 0.9785, 	recall: 0.7532, 	specificity: 0.9958, 	f1: 0.8512
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 145: 100%|██████████| 6195/6195 [06:19<00:00, 16.34it/s, loss=0.408]
Train Epoch 145 ==> 	accuracy: 0.8658, 	precision: 0.9997, 	recall: 0.7319, 	specificity: 0.9997, 	f1: 0.8451
Test Epoch 145: 100%|██████████| 1715/1715 [00:38<00:00, 44.23it/s, loss=2.27]
Test Epoch 145 ==> 	accuracy: 0.9472, 	precision: 0.9757, 	recall: 0.7556, 	specificity: 0.9953, 	f1: 0.8517
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 146: 100%|██████████| 6195/6195 [06:23<00:00, 16.17it/s, loss=0.419]
Train Epoch 146 ==> 	accuracy: 0.8661, 	precision: 0.9997, 	recall: 0.7324, 	specificity: 0.9998, 	f1: 0.8454
Test Epoch 146: 100%|██████████| 1715/1715 [00:40<00:00, 42.62it/s, loss=2.18]
Test Epoch 146 ==> 	accuracy: 0.9468, 	precision: 0.9775, 	recall: 0.7526, 	specificity: 0.9956, 	f1: 0.8504
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 147: 100%|██████████| 6195/6195 [06:15<00:00, 16.48it/s, loss=0.364]
Train Epoch 147 ==> 	accuracy: 0.8667, 	precision: 0.9997, 	recall: 0.7337, 	specificity: 0.9998, 	f1: 0.8463
Test Epoch 147: 100%|██████████| 1715/1715 [00:40<00:00, 42.43it/s, loss=0.499]
Test Epoch 147 ==> 	accuracy: 0.9476, 	precision: 0.9773, 	recall: 0.7563, 	specificity: 0.9956, 	f1: 0.8527
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 148: 100%|██████████| 6195/6195 [06:10<00:00, 16.70it/s, loss=0.407]
Train Epoch 148 ==> 	accuracy: 0.8674, 	precision: 0.9997, 	recall: 0.7351, 	specificity: 0.9998, 	f1: 0.8472
Test Epoch 148: 100%|██████████| 1715/1715 [00:39<00:00, 43.16it/s, loss=0.252]
Test Epoch 148 ==> 	accuracy: 0.9474, 	precision: 0.9773, 	recall: 0.7555, 	specificity: 0.9956, 	f1: 0.8522
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 149: 100%|██████████| 6195/6195 [06:14<00:00, 16.56it/s, loss=0.348]
Train Epoch 149 ==> 	accuracy: 0.8669, 	precision: 0.9997, 	recall: 0.7340, 	specificity: 0.9998, 	f1: 0.8465
Test Epoch 149: 100%|██████████| 1715/1715 [00:40<00:00, 42.80it/s, loss=6.19]
Test Epoch 149 ==> 	accuracy: 0.9470, 	precision: 0.9758, 	recall: 0.7548, 	specificity: 0.9953, 	f1: 0.8512
Adjusting learning rate of group 0 to 4.2391e-06.

Process finished with exit code 0

'''

'''
'../model_save_sigBlock4_focalWithMs_deformable_ab_deformable'
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6195/6195 [04:06<00:00, 25.14it/s, loss=0.567]
Train Epoch 0 ==> 	accuracy: 0.6279, 	precision: 0.9953, 	recall: 0.2569, 	specificity: 0.9988, 	f1: 0.4084
Test Epoch 0: 100%|██████████| 1715/1715 [00:31<00:00, 55.17it/s, loss=0.537]
Test Epoch 0 ==> 	accuracy: 0.8975, 	precision: 0.9582, 	recall: 0.5118, 	specificity: 0.9944, 	f1: 0.6672
Train Epoch 1: 100%|██████████| 6195/6195 [05:03<00:00, 20.39it/s, loss=0.581]
Train Epoch 1 ==> 	accuracy: 0.7137, 	precision: 0.9967, 	recall: 0.4289, 	specificity: 0.9986, 	f1: 0.5997
Test Epoch 1: 100%|██████████| 1715/1715 [00:36<00:00, 47.23it/s, loss=0.258]
Test Epoch 1 ==> 	accuracy: 0.9035, 	precision: 0.9722, 	recall: 0.5344, 	specificity: 0.9962, 	f1: 0.6897
Train Epoch 2: 100%|██████████| 6195/6195 [05:24<00:00, 19.07it/s, loss=0.58]
Train Epoch 2 ==> 	accuracy: 0.7399, 	precision: 0.9973, 	recall: 0.4811, 	specificity: 0.9987, 	f1: 0.6491
Test Epoch 2: 100%|██████████| 1715/1715 [00:36<00:00, 46.53it/s, loss=0.733]
Test Epoch 2 ==> 	accuracy: 0.9038, 	precision: 0.9761, 	recall: 0.5341, 	specificity: 0.9967, 	f1: 0.6904
Train Epoch 3: 100%|██████████| 6195/6195 [05:38<00:00, 18.31it/s, loss=1.16]
Train Epoch 3 ==> 	accuracy: 0.7482, 	precision: 0.9976, 	recall: 0.4977, 	specificity: 0.9988, 	f1: 0.6641
Test Epoch 3: 100%|██████████| 1715/1715 [00:37<00:00, 45.25it/s, loss=0.484]
Test Epoch 3 ==> 	accuracy: 0.9087, 	precision: 0.9742, 	recall: 0.5601, 	specificity: 0.9963, 	f1: 0.7112
Train Epoch 4: 100%|██████████| 6195/6195 [05:35<00:00, 18.45it/s, loss=0.486]
Train Epoch 4 ==> 	accuracy: 0.7613, 	precision: 0.9977, 	recall: 0.5238, 	specificity: 0.9988, 	f1: 0.6869
Test Epoch 4: 100%|██████████| 1715/1715 [00:34<00:00, 49.41it/s, loss=1.32]
Test Epoch 4 ==> 	accuracy: 0.9179, 	precision: 0.9741, 	recall: 0.6070, 	specificity: 0.9959, 	f1: 0.7479
Train Epoch 5: 100%|██████████| 6195/6195 [05:34<00:00, 18.52it/s, loss=0.56]
Train Epoch 5 ==> 	accuracy: 0.7640, 	precision: 0.9979, 	recall: 0.5292, 	specificity: 0.9989, 	f1: 0.6916
Test Epoch 5: 100%|██████████| 1715/1715 [00:36<00:00, 47.51it/s, loss=0.846]
Test Epoch 5 ==> 	accuracy: 0.9118, 	precision: 0.9851, 	recall: 0.5692, 	specificity: 0.9978, 	f1: 0.7215
Train Epoch 6: 100%|██████████| 6195/6195 [05:41<00:00, 18.12it/s, loss=0.483]
Train Epoch 6 ==> 	accuracy: 0.7692, 	precision: 0.9981, 	recall: 0.5395, 	specificity: 0.9990, 	f1: 0.7004
Test Epoch 6: 100%|██████████| 1715/1715 [00:38<00:00, 43.99it/s, loss=0.724]
Test Epoch 6 ==> 	accuracy: 0.9201, 	precision: 0.9665, 	recall: 0.6234, 	specificity: 0.9946, 	f1: 0.7579
Train Epoch 7: 100%|██████████| 6195/6195 [05:42<00:00, 18.07it/s, loss=0.575]
Train Epoch 7 ==> 	accuracy: 0.7734, 	precision: 0.9981, 	recall: 0.5478, 	specificity: 0.9990, 	f1: 0.7074
Test Epoch 7: 100%|██████████| 1715/1715 [00:38<00:00, 44.12it/s, loss=0.299]
Test Epoch 7 ==> 	accuracy: 0.9171, 	precision: 0.9683, 	recall: 0.6072, 	specificity: 0.9950, 	f1: 0.7463
Train Epoch 8: 100%|██████████| 6195/6195 [05:49<00:00, 17.71it/s, loss=0.561]
Train Epoch 8 ==> 	accuracy: 0.7729, 	precision: 0.9982, 	recall: 0.5467, 	specificity: 0.9990, 	f1: 0.7065
Test Epoch 8: 100%|██████████| 1715/1715 [00:37<00:00, 46.03it/s, loss=0.18]
Test Epoch 8 ==> 	accuracy: 0.9143, 	precision: 0.9807, 	recall: 0.5846, 	specificity: 0.9971, 	f1: 0.7326
Train Epoch 9: 100%|██████████| 6195/6195 [05:41<00:00, 18.12it/s, loss=0.52]
Train Epoch 9 ==> 	accuracy: 0.7779, 	precision: 0.9983, 	recall: 0.5567, 	specificity: 0.9991, 	f1: 0.7148
Test Epoch 9: 100%|██████████| 1715/1715 [00:37<00:00, 45.32it/s, loss=0.519]
Test Epoch 9 ==> 	accuracy: 0.9203, 	precision: 0.9599, 	recall: 0.6291, 	specificity: 0.9934, 	f1: 0.7601
Train Epoch 10: 100%|██████████| 6195/6195 [05:49<00:00, 17.72it/s, loss=0.593]
Train Epoch 10 ==> 	accuracy: 0.7787, 	precision: 0.9984, 	recall: 0.5584, 	specificity: 0.9991, 	f1: 0.7162
Test Epoch 10: 100%|██████████| 1715/1715 [00:38<00:00, 44.40it/s, loss=0.26]
Test Epoch 10 ==> 	accuracy: 0.9164, 	precision: 0.9872, 	recall: 0.5915, 	specificity: 0.9981, 	f1: 0.7398
Train Epoch 11: 100%|██████████| 6195/6195 [05:46<00:00, 17.89it/s, loss=0.579]
Train Epoch 11 ==> 	accuracy: 0.7856, 	precision: 0.9984, 	recall: 0.5721, 	specificity: 0.9991, 	f1: 0.7274
Test Epoch 11: 100%|██████████| 1715/1715 [00:38<00:00, 44.66it/s, loss=0.479]
Test Epoch 11 ==> 	accuracy: 0.9176, 	precision: 0.9610, 	recall: 0.6143, 	specificity: 0.9937, 	f1: 0.7495
Train Epoch 12: 100%|██████████| 6195/6195 [05:50<00:00, 17.69it/s, loss=0.464]
Train Epoch 12 ==> 	accuracy: 0.7840, 	precision: 0.9985, 	recall: 0.5688, 	specificity: 0.9992, 	f1: 0.7247
Test Epoch 12: 100%|██████████| 1715/1715 [00:37<00:00, 46.33it/s, loss=0.271]
Test Epoch 12 ==> 	accuracy: 0.9277, 	precision: 0.9756, 	recall: 0.6562, 	specificity: 0.9959, 	f1: 0.7846
Train Epoch 13: 100%|██████████| 6195/6195 [05:51<00:00, 17.61it/s, loss=0.462]
Train Epoch 13 ==> 	accuracy: 0.7889, 	precision: 0.9986, 	recall: 0.5786, 	specificity: 0.9992, 	f1: 0.7326
Test Epoch 13: 100%|██████████| 1715/1715 [00:39<00:00, 43.69it/s, loss=0.634]
Test Epoch 13 ==> 	accuracy: 0.9176, 	precision: 0.9535, 	recall: 0.6197, 	specificity: 0.9924, 	f1: 0.7512
Train Epoch 14: 100%|██████████| 6195/6195 [05:48<00:00, 17.77it/s, loss=0.497]
Train Epoch 14 ==> 	accuracy: 0.7942, 	precision: 0.9986, 	recall: 0.5892, 	specificity: 0.9992, 	f1: 0.7411
Test Epoch 14: 100%|██████████| 1715/1715 [00:37<00:00, 45.80it/s, loss=0.629]
Test Epoch 14 ==> 	accuracy: 0.9223, 	precision: 0.9784, 	recall: 0.6269, 	specificity: 0.9965, 	f1: 0.7642
Train Epoch 15: 100%|██████████| 6195/6195 [05:54<00:00, 17.49it/s, loss=0.461]
Train Epoch 15 ==> 	accuracy: 0.7922, 	precision: 0.9987, 	recall: 0.5851, 	specificity: 0.9992, 	f1: 0.7379
Test Epoch 15: 100%|██████████| 1715/1715 [00:36<00:00, 47.48it/s, loss=0.717]
Test Epoch 15 ==> 	accuracy: 0.9180, 	precision: 0.9732, 	recall: 0.6083, 	specificity: 0.9958, 	f1: 0.7486
Train Epoch 16: 100%|██████████| 6195/6195 [05:42<00:00, 18.09it/s, loss=0.727]
Train Epoch 16 ==> 	accuracy: 0.7885, 	precision: 0.9988, 	recall: 0.5778, 	specificity: 0.9993, 	f1: 0.7321
Test Epoch 16: 100%|██████████| 1715/1715 [00:40<00:00, 42.13it/s, loss=0.16]
Test Epoch 16 ==> 	accuracy: 0.9249, 	precision: 0.9713, 	recall: 0.6448, 	specificity: 0.9952, 	f1: 0.7750
Train Epoch 17: 100%|██████████| 6195/6195 [05:47<00:00, 17.81it/s, loss=0.494]
Train Epoch 17 ==> 	accuracy: 0.7950, 	precision: 0.9987, 	recall: 0.5907, 	specificity: 0.9992, 	f1: 0.7423
Test Epoch 17: 100%|██████████| 1715/1715 [00:38<00:00, 44.18it/s, loss=0.47]
Test Epoch 17 ==> 	accuracy: 0.9251, 	precision: 0.9687, 	recall: 0.6479, 	specificity: 0.9947, 	f1: 0.7765
Train Epoch 18: 100%|██████████| 6195/6195 [05:46<00:00, 17.89it/s, loss=0.481]
Train Epoch 18 ==> 	accuracy: 0.7958, 	precision: 0.9988, 	recall: 0.5923, 	specificity: 0.9993, 	f1: 0.7436
Test Epoch 18: 100%|██████████| 1715/1715 [00:38<00:00, 44.81it/s, loss=0.229]
Test Epoch 18 ==> 	accuracy: 0.9263, 	precision: 0.9711, 	recall: 0.6526, 	specificity: 0.9951, 	f1: 0.7806
Train Epoch 19: 100%|██████████| 6195/6195 [05:44<00:00, 17.99it/s, loss=0.552]
Train Epoch 19 ==> 	accuracy: 0.7964, 	precision: 0.9988, 	recall: 0.5935, 	specificity: 0.9993, 	f1: 0.7446
Test Epoch 19: 100%|██████████| 1715/1715 [00:36<00:00, 47.03it/s, loss=0.852]
Test Epoch 19 ==> 	accuracy: 0.9260, 	precision: 0.9675, 	recall: 0.6533, 	specificity: 0.9945, 	f1: 0.7799
Train Epoch 20: 100%|██████████| 6195/6195 [05:49<00:00, 17.72it/s, loss=0.449]
Train Epoch 20 ==> 	accuracy: 0.7970, 	precision: 0.9988, 	recall: 0.5947, 	specificity: 0.9993, 	f1: 0.7455
Test Epoch 20: 100%|██████████| 1715/1715 [00:39<00:00, 43.79it/s, loss=0.368]
Test Epoch 20 ==> 	accuracy: 0.9183, 	precision: 0.9795, 	recall: 0.6057, 	specificity: 0.9968, 	f1: 0.7486
Train Epoch 21: 100%|██████████| 6195/6195 [05:46<00:00, 17.89it/s, loss=0.567]
Train Epoch 21 ==> 	accuracy: 0.8014, 	precision: 0.9989, 	recall: 0.6034, 	specificity: 0.9993, 	f1: 0.7523
Test Epoch 21: 100%|██████████| 1715/1715 [00:37<00:00, 45.45it/s, loss=0.419]
Test Epoch 21 ==> 	accuracy: 0.9227, 	precision: 0.9759, 	recall: 0.6307, 	specificity: 0.9961, 	f1: 0.7662
Train Epoch 22: 100%|██████████| 6195/6195 [05:49<00:00, 17.70it/s, loss=0.587]
Train Epoch 22 ==> 	accuracy: 0.8003, 	precision: 0.9989, 	recall: 0.6013, 	specificity: 0.9993, 	f1: 0.7507
Test Epoch 22: 100%|██████████| 1715/1715 [00:37<00:00, 45.59it/s, loss=0.335]
Test Epoch 22 ==> 	accuracy: 0.9161, 	precision: 0.9871, 	recall: 0.5900, 	specificity: 0.9981, 	f1: 0.7386
Train Epoch 23: 100%|██████████| 6195/6195 [05:54<00:00, 17.49it/s, loss=0.559]
Train Epoch 23 ==> 	accuracy: 0.8041, 	precision: 0.9989, 	recall: 0.6090, 	specificity: 0.9993, 	f1: 0.7566
Test Epoch 23: 100%|██████████| 1715/1715 [00:35<00:00, 48.38it/s, loss=0.613]
Test Epoch 23 ==> 	accuracy: 0.9114, 	precision: 0.9737, 	recall: 0.5743, 	specificity: 0.9961, 	f1: 0.7225
Train Epoch 24: 100%|██████████| 6195/6195 [05:46<00:00, 17.86it/s, loss=0.73]
Train Epoch 24 ==> 	accuracy: 0.8015, 	precision: 0.9989, 	recall: 0.6037, 	specificity: 0.9993, 	f1: 0.7526
Test Epoch 24: 100%|██████████| 1715/1715 [00:36<00:00, 47.61it/s, loss=0.262]
Test Epoch 24 ==> 	accuracy: 0.9234, 	precision: 0.9735, 	recall: 0.6357, 	specificity: 0.9957, 	f1: 0.7691
Train Epoch 25: 100%|██████████| 6195/6195 [05:42<00:00, 18.09it/s, loss=0.486]
Train Epoch 25 ==> 	accuracy: 0.8057, 	precision: 0.9991, 	recall: 0.6119, 	specificity: 0.9994, 	f1: 0.7590
Test Epoch 25: 100%|██████████| 1715/1715 [00:38<00:00, 44.63it/s, loss=0.616]
Test Epoch 25 ==> 	accuracy: 0.9229, 	precision: 0.9760, 	recall: 0.6314, 	specificity: 0.9961, 	f1: 0.7668
Train Epoch 26: 100%|██████████| 6195/6195 [05:48<00:00, 17.79it/s, loss=0.449]
Train Epoch 26 ==> 	accuracy: 0.8046, 	precision: 0.9990, 	recall: 0.6098, 	specificity: 0.9994, 	f1: 0.7573
Test Epoch 26: 100%|██████████| 1715/1715 [00:39<00:00, 43.76it/s, loss=0.445]
Test Epoch 26 ==> 	accuracy: 0.9230, 	precision: 0.9774, 	recall: 0.6310, 	specificity: 0.9963, 	f1: 0.7669
Train Epoch 27: 100%|██████████| 6195/6195 [05:47<00:00, 17.81it/s, loss=0.535]
Train Epoch 27 ==> 	accuracy: 0.8076, 	precision: 0.9990, 	recall: 0.6159, 	specificity: 0.9994, 	f1: 0.7620
Test Epoch 27: 100%|██████████| 1715/1715 [00:37<00:00, 45.39it/s, loss=0.38]
Test Epoch 27 ==> 	accuracy: 0.9277, 	precision: 0.9638, 	recall: 0.6647, 	specificity: 0.9937, 	f1: 0.7868
Train Epoch 28: 100%|██████████| 6195/6195 [05:47<00:00, 17.84it/s, loss=0.481]
Train Epoch 28 ==> 	accuracy: 0.8100, 	precision: 0.9990, 	recall: 0.6206, 	specificity: 0.9994, 	f1: 0.7656
Test Epoch 28: 100%|██████████| 1715/1715 [00:37<00:00, 46.09it/s, loss=0.417]
Test Epoch 28 ==> 	accuracy: 0.9258, 	precision: 0.9691, 	recall: 0.6513, 	specificity: 0.9948, 	f1: 0.7790
Train Epoch 29: 100%|██████████| 6195/6195 [05:44<00:00, 17.97it/s, loss=0.466]
Train Epoch 29 ==> 	accuracy: 0.8096, 	precision: 0.9990, 	recall: 0.6199, 	specificity: 0.9994, 	f1: 0.7650
Test Epoch 29: 100%|██████████| 1715/1715 [00:37<00:00, 45.73it/s, loss=0.224]
Test Epoch 29 ==> 	accuracy: 0.9254, 	precision: 0.9662, 	recall: 0.6512, 	specificity: 0.9943, 	f1: 0.7780
Train Epoch 30: 100%|██████████| 6195/6195 [05:44<00:00, 17.98it/s, loss=0.567]
Train Epoch 30 ==> 	accuracy: 0.8091, 	precision: 0.9991, 	recall: 0.6187, 	specificity: 0.9994, 	f1: 0.7642
Test Epoch 30: 100%|██████████| 1715/1715 [00:34<00:00, 49.35it/s, loss=0.381]
Test Epoch 30 ==> 	accuracy: 0.9277, 	precision: 0.9657, 	recall: 0.6633, 	specificity: 0.9941, 	f1: 0.7864
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 31: 100%|██████████| 6195/6195 [05:34<00:00, 18.52it/s, loss=0.421]
Train Epoch 31 ==> 	accuracy: 0.8122, 	precision: 0.9991, 	recall: 0.6250, 	specificity: 0.9995, 	f1: 0.7690
Test Epoch 31: 100%|██████████| 1715/1715 [00:38<00:00, 44.80it/s, loss=0.593]
Test Epoch 31 ==> 	accuracy: 0.9272, 	precision: 0.9606, 	recall: 0.6649, 	specificity: 0.9931, 	f1: 0.7858
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 32: 100%|██████████| 6195/6195 [05:32<00:00, 18.64it/s, loss=0.506]
Train Epoch 32 ==> 	accuracy: 0.8103, 	precision: 0.9991, 	recall: 0.6212, 	specificity: 0.9994, 	f1: 0.7661
Test Epoch 32: 100%|██████████| 1715/1715 [00:34<00:00, 49.01it/s, loss=0.36]
Test Epoch 32 ==> 	accuracy: 0.9271, 	precision: 0.9724, 	recall: 0.6557, 	specificity: 0.9953, 	f1: 0.7832
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 33: 100%|██████████| 6195/6195 [05:30<00:00, 18.74it/s, loss=0.354]
Train Epoch 33 ==> 	accuracy: 0.8102, 	precision: 0.9991, 	recall: 0.6210, 	specificity: 0.9995, 	f1: 0.7660
Test Epoch 33: 100%|██████████| 1715/1715 [00:36<00:00, 47.31it/s, loss=1.56]
Test Epoch 33 ==> 	accuracy: 0.9263, 	precision: 0.9744, 	recall: 0.6501, 	specificity: 0.9957, 	f1: 0.7799
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 34: 100%|██████████| 6195/6195 [05:30<00:00, 18.76it/s, loss=0.508]
Train Epoch 34 ==> 	accuracy: 0.8130, 	precision: 0.9992, 	recall: 0.6265, 	specificity: 0.9995, 	f1: 0.7702
Test Epoch 34: 100%|██████████| 1715/1715 [00:35<00:00, 48.81it/s, loss=0.376]
Test Epoch 34 ==> 	accuracy: 0.9292, 	precision: 0.9823, 	recall: 0.6593, 	specificity: 0.9970, 	f1: 0.7890
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 35: 100%|██████████| 6195/6195 [05:29<00:00, 18.80it/s, loss=0.514]
Train Epoch 35 ==> 	accuracy: 0.8180, 	precision: 0.9992, 	recall: 0.6366, 	specificity: 0.9995, 	f1: 0.7777
Test Epoch 35: 100%|██████████| 1715/1715 [00:33<00:00, 50.59it/s, loss=0.461]
Test Epoch 35 ==> 	accuracy: 0.9300, 	precision: 0.9668, 	recall: 0.6743, 	specificity: 0.9942, 	f1: 0.7945
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 36: 100%|██████████| 6195/6195 [05:45<00:00, 17.94it/s, loss=0.493]
Train Epoch 36 ==> 	accuracy: 0.8192, 	precision: 0.9992, 	recall: 0.6389, 	specificity: 0.9995, 	f1: 0.7794
Test Epoch 36: 100%|██████████| 1715/1715 [00:38<00:00, 44.29it/s, loss=2.1]
Test Epoch 36 ==> 	accuracy: 0.9319, 	precision: 0.9507, 	recall: 0.6971, 	specificity: 0.9909, 	f1: 0.8044
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 37: 100%|██████████| 6195/6195 [05:42<00:00, 18.10it/s, loss=0.441]
Train Epoch 37 ==> 	accuracy: 0.8184, 	precision: 0.9993, 	recall: 0.6373, 	specificity: 0.9995, 	f1: 0.7783
Test Epoch 37: 100%|██████████| 1715/1715 [00:35<00:00, 47.70it/s, loss=0.58]
Test Epoch 37 ==> 	accuracy: 0.9303, 	precision: 0.9710, 	recall: 0.6730, 	specificity: 0.9950, 	f1: 0.7950
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 38: 100%|██████████| 6195/6195 [05:29<00:00, 18.81it/s, loss=0.517]
Train Epoch 38 ==> 	accuracy: 0.8230, 	precision: 0.9992, 	recall: 0.6466, 	specificity: 0.9995, 	f1: 0.7851
Test Epoch 38: 100%|██████████| 1715/1715 [00:34<00:00, 50.16it/s, loss=0.526]
Test Epoch 38 ==> 	accuracy: 0.9321, 	precision: 0.9696, 	recall: 0.6831, 	specificity: 0.9946, 	f1: 0.8015
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 39: 100%|██████████| 6195/6195 [05:36<00:00, 18.39it/s, loss=0.392]
Train Epoch 39 ==> 	accuracy: 0.8206, 	precision: 0.9993, 	recall: 0.6418, 	specificity: 0.9995, 	f1: 0.7816
Test Epoch 39: 100%|██████████| 1715/1715 [00:34<00:00, 49.02it/s, loss=0.393]
Test Epoch 39 ==> 	accuracy: 0.9323, 	precision: 0.9766, 	recall: 0.6793, 	specificity: 0.9959, 	f1: 0.8012
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 40: 100%|██████████| 6195/6195 [05:28<00:00, 18.88it/s, loss=0.63]
Train Epoch 40 ==> 	accuracy: 0.8262, 	precision: 0.9993, 	recall: 0.6528, 	specificity: 0.9995, 	f1: 0.7897
Test Epoch 40: 100%|██████████| 1715/1715 [00:33<00:00, 51.25it/s, loss=0.321]
Test Epoch 40 ==> 	accuracy: 0.9323, 	precision: 0.9744, 	recall: 0.6804, 	specificity: 0.9955, 	f1: 0.8013
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 41: 100%|██████████| 6195/6195 [05:40<00:00, 18.21it/s, loss=0.435]
Train Epoch 41 ==> 	accuracy: 0.8214, 	precision: 0.9993, 	recall: 0.6433, 	specificity: 0.9995, 	f1: 0.7827
Test Epoch 41: 100%|██████████| 1715/1715 [00:36<00:00, 46.41it/s, loss=0.5]
Test Epoch 41 ==> 	accuracy: 0.9323, 	precision: 0.9638, 	recall: 0.6888, 	specificity: 0.9935, 	f1: 0.8034
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 42: 100%|██████████| 6195/6195 [05:35<00:00, 18.44it/s, loss=0.509]
Train Epoch 42 ==> 	accuracy: 0.8305, 	precision: 0.9993, 	recall: 0.6615, 	specificity: 0.9995, 	f1: 0.7961
Test Epoch 42: 100%|██████████| 1715/1715 [00:36<00:00, 47.25it/s, loss=0.48]
Test Epoch 42 ==> 	accuracy: 0.9341, 	precision: 0.9662, 	recall: 0.6962, 	specificity: 0.9939, 	f1: 0.8093
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 43: 100%|██████████| 6195/6195 [05:46<00:00, 17.89it/s, loss=0.568]
Train Epoch 43 ==> 	accuracy: 0.8282, 	precision: 0.9993, 	recall: 0.6569, 	specificity: 0.9996, 	f1: 0.7927
Test Epoch 43: 100%|██████████| 1715/1715 [00:36<00:00, 46.65it/s, loss=0.419]
Test Epoch 43 ==> 	accuracy: 0.9328, 	precision: 0.9641, 	recall: 0.6911, 	specificity: 0.9935, 	f1: 0.8051
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 44: 100%|██████████| 6195/6195 [05:43<00:00, 18.05it/s, loss=0.506]
Train Epoch 44 ==> 	accuracy: 0.8270, 	precision: 0.9994, 	recall: 0.6544, 	specificity: 0.9996, 	f1: 0.7909
Test Epoch 44: 100%|██████████| 1715/1715 [00:37<00:00, 45.62it/s, loss=0.255]
Test Epoch 44 ==> 	accuracy: 0.9311, 	precision: 0.9801, 	recall: 0.6706, 	specificity: 0.9966, 	f1: 0.7964
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 45: 100%|██████████| 6195/6195 [05:46<00:00, 17.89it/s, loss=0.563]
Train Epoch 45 ==> 	accuracy: 0.8297, 	precision: 0.9994, 	recall: 0.6598, 	specificity: 0.9996, 	f1: 0.7949
Test Epoch 45: 100%|██████████| 1715/1715 [00:38<00:00, 44.42it/s, loss=0.491]
Test Epoch 45 ==> 	accuracy: 0.9350, 	precision: 0.9705, 	recall: 0.6974, 	specificity: 0.9947, 	f1: 0.8116
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 46: 100%|██████████| 6195/6195 [05:42<00:00, 18.11it/s, loss=0.381]
Train Epoch 46 ==> 	accuracy: 0.8325, 	precision: 0.9994, 	recall: 0.6654, 	specificity: 0.9996, 	f1: 0.7989
Test Epoch 46: 100%|██████████| 1715/1715 [00:37<00:00, 45.81it/s, loss=1.74]
Test Epoch 46 ==> 	accuracy: 0.9327, 	precision: 0.9485, 	recall: 0.7028, 	specificity: 0.9904, 	f1: 0.8073
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 47: 100%|██████████| 6195/6195 [05:45<00:00, 17.95it/s, loss=0.399]
Train Epoch 47 ==> 	accuracy: 0.8358, 	precision: 0.9994, 	recall: 0.6721, 	specificity: 0.9996, 	f1: 0.8037
Test Epoch 47: 100%|██████████| 1715/1715 [00:38<00:00, 45.12it/s, loss=0.507]
Test Epoch 47 ==> 	accuracy: 0.9358, 	precision: 0.9748, 	recall: 0.6984, 	specificity: 0.9955, 	f1: 0.8138
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 48: 100%|██████████| 6195/6195 [05:38<00:00, 18.28it/s, loss=0.476]
Train Epoch 48 ==> 	accuracy: 0.8319, 	precision: 0.9994, 	recall: 0.6643, 	specificity: 0.9996, 	f1: 0.7981
Test Epoch 48: 100%|██████████| 1715/1715 [00:38<00:00, 44.85it/s, loss=0.745]
Test Epoch 48 ==> 	accuracy: 0.9328, 	precision: 0.9676, 	recall: 0.6884, 	specificity: 0.9942, 	f1: 0.8045
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 49: 100%|██████████| 6195/6195 [05:40<00:00, 18.21it/s, loss=0.508]
Train Epoch 49 ==> 	accuracy: 0.8325, 	precision: 0.9994, 	recall: 0.6654, 	specificity: 0.9996, 	f1: 0.7989
Test Epoch 49: 100%|██████████| 1715/1715 [00:35<00:00, 48.59it/s, loss=0.212]
Test Epoch 49 ==> 	accuracy: 0.9368, 	precision: 0.9751, 	recall: 0.7029, 	specificity: 0.9955, 	f1: 0.8169
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 50: 100%|██████████| 6195/6195 [05:40<00:00, 18.20it/s, loss=0.423]
Train Epoch 50 ==> 	accuracy: 0.8367, 	precision: 0.9994, 	recall: 0.6738, 	specificity: 0.9996, 	f1: 0.8049
Test Epoch 50: 100%|██████████| 1715/1715 [00:37<00:00, 45.26it/s, loss=0.208]
Test Epoch 50 ==> 	accuracy: 0.9394, 	precision: 0.9811, 	recall: 0.7116, 	specificity: 0.9966, 	f1: 0.8249
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 51: 100%|██████████| 6195/6195 [05:40<00:00, 18.20it/s, loss=0.417]
Train Epoch 51 ==> 	accuracy: 0.8363, 	precision: 0.9994, 	recall: 0.6730, 	specificity: 0.9996, 	f1: 0.8044
Test Epoch 51: 100%|██████████| 1715/1715 [00:34<00:00, 49.06it/s, loss=0.217]
Test Epoch 51 ==> 	accuracy: 0.9377, 	precision: 0.9850, 	recall: 0.7006, 	specificity: 0.9973, 	f1: 0.8188
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 52: 100%|██████████| 6195/6195 [05:43<00:00, 18.04it/s, loss=0.441]
Train Epoch 52 ==> 	accuracy: 0.8372, 	precision: 0.9994, 	recall: 0.6748, 	specificity: 0.9996, 	f1: 0.8057
Test Epoch 52: 100%|██████████| 1715/1715 [00:36<00:00, 47.37it/s, loss=0.148]
Test Epoch 52 ==> 	accuracy: 0.9344, 	precision: 0.9866, 	recall: 0.6826, 	specificity: 0.9977, 	f1: 0.8069
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 53: 100%|██████████| 6195/6195 [05:41<00:00, 18.15it/s, loss=0.433]
Train Epoch 53 ==> 	accuracy: 0.8382, 	precision: 0.9994, 	recall: 0.6768, 	specificity: 0.9996, 	f1: 0.8071
Test Epoch 53: 100%|██████████| 1715/1715 [00:40<00:00, 42.48it/s, loss=0.285]
Test Epoch 53 ==> 	accuracy: 0.9367, 	precision: 0.9851, 	recall: 0.6954, 	specificity: 0.9974, 	f1: 0.8153
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 54: 100%|██████████| 6195/6195 [05:42<00:00, 18.08it/s, loss=0.43]
Train Epoch 54 ==> 	accuracy: 0.8401, 	precision: 0.9995, 	recall: 0.6805, 	specificity: 0.9996, 	f1: 0.8097
Test Epoch 54: 100%|██████████| 1715/1715 [00:35<00:00, 48.08it/s, loss=0.439]
Test Epoch 54 ==> 	accuracy: 0.9368, 	precision: 0.9872, 	recall: 0.6944, 	specificity: 0.9977, 	f1: 0.8153
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 55: 100%|██████████| 6195/6195 [05:36<00:00, 18.42it/s, loss=0.453]
Train Epoch 55 ==> 	accuracy: 0.8385, 	precision: 0.9995, 	recall: 0.6774, 	specificity: 0.9997, 	f1: 0.8075
Test Epoch 55: 100%|██████████| 1715/1715 [00:35<00:00, 48.86it/s, loss=0.495]
Test Epoch 55 ==> 	accuracy: 0.9404, 	precision: 0.9829, 	recall: 0.7154, 	specificity: 0.9969, 	f1: 0.8281
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 56: 100%|██████████| 6195/6195 [05:32<00:00, 18.63it/s, loss=0.383]
Train Epoch 56 ==> 	accuracy: 0.8424, 	precision: 0.9995, 	recall: 0.6852, 	specificity: 0.9997, 	f1: 0.8131
Test Epoch 56: 100%|██████████| 1715/1715 [00:36<00:00, 46.59it/s, loss=0.146]
Test Epoch 56 ==> 	accuracy: 0.9379, 	precision: 0.9842, 	recall: 0.7020, 	specificity: 0.9972, 	f1: 0.8195
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 57: 100%|██████████| 6195/6195 [05:30<00:00, 18.77it/s, loss=0.368]
Train Epoch 57 ==> 	accuracy: 0.8402, 	precision: 0.9995, 	recall: 0.6807, 	specificity: 0.9997, 	f1: 0.8099
Test Epoch 57: 100%|██████████| 1715/1715 [00:33<00:00, 51.22it/s, loss=0.365]
Test Epoch 57 ==> 	accuracy: 0.9415, 	precision: 0.9816, 	recall: 0.7223, 	specificity: 0.9966, 	f1: 0.8322
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 58: 100%|██████████| 6195/6195 [05:31<00:00, 18.69it/s, loss=0.493]
Train Epoch 58 ==> 	accuracy: 0.8416, 	precision: 0.9996, 	recall: 0.6836, 	specificity: 0.9997, 	f1: 0.8119
Test Epoch 58: 100%|██████████| 1715/1715 [00:35<00:00, 47.71it/s, loss=0.147]
Test Epoch 58 ==> 	accuracy: 0.9393, 	precision: 0.9821, 	recall: 0.7108, 	specificity: 0.9968, 	f1: 0.8247
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 59: 100%|██████████| 6195/6195 [05:30<00:00, 18.74it/s, loss=0.535]
Train Epoch 59 ==> 	accuracy: 0.8433, 	precision: 0.9996, 	recall: 0.6869, 	specificity: 0.9997, 	f1: 0.8143
Test Epoch 59: 100%|██████████| 1715/1715 [00:34<00:00, 49.64it/s, loss=0.287]
Test Epoch 59 ==> 	accuracy: 0.9393, 	precision: 0.9820, 	recall: 0.7105, 	specificity: 0.9967, 	f1: 0.8244
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 60: 100%|██████████| 6195/6195 [05:28<00:00, 18.84it/s, loss=0.451]
Train Epoch 60 ==> 	accuracy: 0.8456, 	precision: 0.9995, 	recall: 0.6915, 	specificity: 0.9996, 	f1: 0.8175
Test Epoch 60: 100%|██████████| 1715/1715 [00:32<00:00, 53.11it/s, loss=0.177]
Test Epoch 60 ==> 	accuracy: 0.9416, 	precision: 0.9799, 	recall: 0.7240, 	specificity: 0.9963, 	f1: 0.8327
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 61: 100%|██████████| 6195/6195 [05:29<00:00, 18.78it/s, loss=0.397]
Train Epoch 61 ==> 	accuracy: 0.8442, 	precision: 0.9995, 	recall: 0.6887, 	specificity: 0.9997, 	f1: 0.8155
Test Epoch 61: 100%|██████████| 1715/1715 [00:37<00:00, 45.78it/s, loss=0.557]
Test Epoch 61 ==> 	accuracy: 0.9411, 	precision: 0.9829, 	recall: 0.7189, 	specificity: 0.9969, 	f1: 0.8305
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 62: 100%|██████████| 6195/6195 [05:31<00:00, 18.70it/s, loss=0.461]
Train Epoch 62 ==> 	accuracy: 0.8459, 	precision: 0.9995, 	recall: 0.6921, 	specificity: 0.9997, 	f1: 0.8179
Test Epoch 62: 100%|██████████| 1715/1715 [00:35<00:00, 47.91it/s, loss=0.164]
Test Epoch 62 ==> 	accuracy: 0.9393, 	precision: 0.9808, 	recall: 0.7115, 	specificity: 0.9965, 	f1: 0.8247
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 63: 100%|██████████| 6195/6195 [05:25<00:00, 19.04it/s, loss=0.367]
Train Epoch 63 ==> 	accuracy: 0.8489, 	precision: 0.9995, 	recall: 0.6981, 	specificity: 0.9997, 	f1: 0.8221
Test Epoch 63: 100%|██████████| 1715/1715 [00:37<00:00, 46.22it/s, loss=2.06]
Test Epoch 63 ==> 	accuracy: 0.9432, 	precision: 0.9745, 	recall: 0.7362, 	specificity: 0.9952, 	f1: 0.8387
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 64: 100%|██████████| 6195/6195 [05:24<00:00, 19.08it/s, loss=0.397]
Train Epoch 64 ==> 	accuracy: 0.8466, 	precision: 0.9995, 	recall: 0.6935, 	specificity: 0.9997, 	f1: 0.8189
Test Epoch 64: 100%|██████████| 1715/1715 [00:34<00:00, 49.15it/s, loss=2.49]
Test Epoch 64 ==> 	accuracy: 0.9402, 	precision: 0.9754, 	recall: 0.7204, 	specificity: 0.9954, 	f1: 0.8287
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 65: 100%|██████████| 6195/6195 [05:33<00:00, 18.56it/s, loss=0.526]
Train Epoch 65 ==> 	accuracy: 0.8468, 	precision: 0.9996, 	recall: 0.6939, 	specificity: 0.9997, 	f1: 0.8191
Test Epoch 65: 100%|██████████| 1715/1715 [00:35<00:00, 48.90it/s, loss=0.914]
Test Epoch 65 ==> 	accuracy: 0.9418, 	precision: 0.9800, 	recall: 0.7248, 	specificity: 0.9963, 	f1: 0.8333
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 66: 100%|██████████| 6195/6195 [05:27<00:00, 18.90it/s, loss=0.435]
Train Epoch 66 ==> 	accuracy: 0.8472, 	precision: 0.9996, 	recall: 0.6947, 	specificity: 0.9997, 	f1: 0.8197
Test Epoch 66: 100%|██████████| 1715/1715 [00:36<00:00, 47.45it/s, loss=0.278]
Test Epoch 66 ==> 	accuracy: 0.9433, 	precision: 0.9767, 	recall: 0.7354, 	specificity: 0.9956, 	f1: 0.8390
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 67: 100%|██████████| 6195/6195 [05:23<00:00, 19.13it/s, loss=0.437]
Train Epoch 67 ==> 	accuracy: 0.8504, 	precision: 0.9996, 	recall: 0.7011, 	specificity: 0.9997, 	f1: 0.8242
Test Epoch 67: 100%|██████████| 1715/1715 [00:34<00:00, 50.01it/s, loss=0.274]
Test Epoch 67 ==> 	accuracy: 0.9408, 	precision: 0.9832, 	recall: 0.7173, 	specificity: 0.9969, 	f1: 0.8294
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 68: 100%|██████████| 6195/6195 [05:27<00:00, 18.94it/s, loss=0.411]
Train Epoch 68 ==> 	accuracy: 0.8481, 	precision: 0.9996, 	recall: 0.6966, 	specificity: 0.9997, 	f1: 0.8210
Test Epoch 68: 100%|██████████| 1715/1715 [00:34<00:00, 49.51it/s, loss=0.133]
Test Epoch 68 ==> 	accuracy: 0.9419, 	precision: 0.9788, 	recall: 0.7264, 	specificity: 0.9960, 	f1: 0.8339
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 69: 100%|██████████| 6195/6195 [05:23<00:00, 19.13it/s, loss=0.377]
Train Epoch 69 ==> 	accuracy: 0.8514, 	precision: 0.9996, 	recall: 0.7030, 	specificity: 0.9997, 	f1: 0.8255
Test Epoch 69: 100%|██████████| 1715/1715 [00:36<00:00, 47.05it/s, loss=1.53]
Test Epoch 69 ==> 	accuracy: 0.9415, 	precision: 0.9801, 	recall: 0.7232, 	specificity: 0.9963, 	f1: 0.8323
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 70: 100%|██████████| 6195/6195 [05:25<00:00, 19.02it/s, loss=0.39]
Train Epoch 70 ==> 	accuracy: 0.8514, 	precision: 0.9996, 	recall: 0.7031, 	specificity: 0.9997, 	f1: 0.8255
Test Epoch 70: 100%|██████████| 1715/1715 [00:37<00:00, 45.23it/s, loss=0.379]
Test Epoch 70 ==> 	accuracy: 0.9402, 	precision: 0.9820, 	recall: 0.7153, 	specificity: 0.9967, 	f1: 0.8277
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 71: 100%|██████████| 6195/6195 [05:21<00:00, 19.30it/s, loss=0.478]
Train Epoch 71 ==> 	accuracy: 0.8526, 	precision: 0.9996, 	recall: 0.7054, 	specificity: 0.9997, 	f1: 0.8271
Test Epoch 71: 100%|██████████| 1715/1715 [00:34<00:00, 49.54it/s, loss=0.211]
Test Epoch 71 ==> 	accuracy: 0.9432, 	precision: 0.9785, 	recall: 0.7329, 	specificity: 0.9960, 	f1: 0.8381
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 72: 100%|██████████| 6195/6195 [05:25<00:00, 19.06it/s, loss=0.36]
Train Epoch 72 ==> 	accuracy: 0.8513, 	precision: 0.9996, 	recall: 0.7030, 	specificity: 0.9997, 	f1: 0.8254
Test Epoch 72: 100%|██████████| 1715/1715 [00:35<00:00, 47.67it/s, loss=0.28]
Test Epoch 72 ==> 	accuracy: 0.9449, 	precision: 0.9774, 	recall: 0.7425, 	specificity: 0.9957, 	f1: 0.8439
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 73: 100%|██████████| 6195/6195 [05:26<00:00, 18.98it/s, loss=0.458]
Train Epoch 73 ==> 	accuracy: 0.8561, 	precision: 0.9996, 	recall: 0.7126, 	specificity: 0.9997, 	f1: 0.8320
Test Epoch 73: 100%|██████████| 1715/1715 [00:33<00:00, 50.80it/s, loss=0.523]
Test Epoch 73 ==> 	accuracy: 0.9441, 	precision: 0.9767, 	recall: 0.7392, 	specificity: 0.9956, 	f1: 0.8415
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 74: 100%|██████████| 6195/6195 [05:24<00:00, 19.09it/s, loss=0.331]
Train Epoch 74 ==> 	accuracy: 0.8534, 	precision: 0.9996, 	recall: 0.7071, 	specificity: 0.9997, 	f1: 0.8283
Test Epoch 74: 100%|██████████| 1715/1715 [00:34<00:00, 50.15it/s, loss=0.148]
Test Epoch 74 ==> 	accuracy: 0.9440, 	precision: 0.9778, 	recall: 0.7375, 	specificity: 0.9958, 	f1: 0.8409
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 75: 100%|██████████| 6195/6195 [05:14<00:00, 19.72it/s, loss=0.387]
Train Epoch 75 ==> 	accuracy: 0.8561, 	precision: 0.9996, 	recall: 0.7125, 	specificity: 0.9997, 	f1: 0.8320
Test Epoch 75: 100%|██████████| 1715/1715 [00:35<00:00, 48.24it/s, loss=1.55]
Test Epoch 75 ==> 	accuracy: 0.9431, 	precision: 0.9777, 	recall: 0.7333, 	specificity: 0.9958, 	f1: 0.8380
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 76: 100%|██████████| 6195/6195 [05:20<00:00, 19.35it/s, loss=0.484]
Train Epoch 76 ==> 	accuracy: 0.8576, 	precision: 0.9996, 	recall: 0.7155, 	specificity: 0.9997, 	f1: 0.8341
Test Epoch 76: 100%|██████████| 1715/1715 [00:34<00:00, 49.29it/s, loss=0.488]
Test Epoch 76 ==> 	accuracy: 0.9440, 	precision: 0.9779, 	recall: 0.7376, 	specificity: 0.9958, 	f1: 0.8409
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 77: 100%|██████████| 6195/6195 [05:29<00:00, 18.78it/s, loss=0.403]
Train Epoch 77 ==> 	accuracy: 0.8583, 	precision: 0.9996, 	recall: 0.7169, 	specificity: 0.9997, 	f1: 0.8350
Test Epoch 77: 100%|██████████| 1715/1715 [00:33<00:00, 51.96it/s, loss=0.157]
Test Epoch 77 ==> 	accuracy: 0.9440, 	precision: 0.9772, 	recall: 0.7383, 	specificity: 0.9957, 	f1: 0.8411
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 78: 100%|██████████| 6195/6195 [05:24<00:00, 19.09it/s, loss=0.392]
Train Epoch 78 ==> 	accuracy: 0.8556, 	precision: 0.9996, 	recall: 0.7114, 	specificity: 0.9997, 	f1: 0.8313
Test Epoch 78: 100%|██████████| 1715/1715 [00:34<00:00, 49.17it/s, loss=0.321]
Test Epoch 78 ==> 	accuracy: 0.9429, 	precision: 0.9750, 	recall: 0.7342, 	specificity: 0.9953, 	f1: 0.8376
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 79: 100%|██████████| 6195/6195 [05:25<00:00, 19.06it/s, loss=0.394]
Train Epoch 79 ==> 	accuracy: 0.8577, 	precision: 0.9996, 	recall: 0.7158, 	specificity: 0.9997, 	f1: 0.8342
Test Epoch 79: 100%|██████████| 1715/1715 [00:32<00:00, 52.18it/s, loss=4.68]
Test Epoch 79 ==> 	accuracy: 0.9425, 	precision: 0.9561, 	recall: 0.7479, 	specificity: 0.9914, 	f1: 0.8393
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 80: 100%|██████████| 6195/6195 [05:25<00:00, 19.03it/s, loss=0.83]
Train Epoch 80 ==> 	accuracy: 0.8582, 	precision: 0.9996, 	recall: 0.7167, 	specificity: 0.9997, 	f1: 0.8349
Test Epoch 80: 100%|██████████| 1715/1715 [00:35<00:00, 48.43it/s, loss=0.307]
Test Epoch 80 ==> 	accuracy: 0.9443, 	precision: 0.9694, 	recall: 0.7463, 	specificity: 0.9941, 	f1: 0.8434
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 81: 100%|██████████| 6195/6195 [05:25<00:00, 19.03it/s, loss=0.403]
Train Epoch 81 ==> 	accuracy: 0.8580, 	precision: 0.9996, 	recall: 0.7163, 	specificity: 0.9997, 	f1: 0.8345
Test Epoch 81: 100%|██████████| 1715/1715 [00:34<00:00, 49.74it/s, loss=0.369]
Test Epoch 81 ==> 	accuracy: 0.9439, 	precision: 0.9730, 	recall: 0.7410, 	specificity: 0.9948, 	f1: 0.8413
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 82: 100%|██████████| 6195/6195 [05:22<00:00, 19.22it/s, loss=0.409]
Train Epoch 82 ==> 	accuracy: 0.8581, 	precision: 0.9996, 	recall: 0.7165, 	specificity: 0.9997, 	f1: 0.8347
Test Epoch 82: 100%|██████████| 1715/1715 [00:34<00:00, 49.98it/s, loss=0.241]
Test Epoch 82 ==> 	accuracy: 0.9431, 	precision: 0.9774, 	recall: 0.7336, 	specificity: 0.9957, 	f1: 0.8381
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 83: 100%|██████████| 6195/6195 [05:28<00:00, 18.86it/s, loss=0.499]
Train Epoch 83 ==> 	accuracy: 0.8604, 	precision: 0.9996, 	recall: 0.7210, 	specificity: 0.9997, 	f1: 0.8378
Test Epoch 83: 100%|██████████| 1715/1715 [00:33<00:00, 51.26it/s, loss=0.36]
Test Epoch 83 ==> 	accuracy: 0.9444, 	precision: 0.9726, 	recall: 0.7439, 	specificity: 0.9947, 	f1: 0.8430
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 84: 100%|██████████| 6195/6195 [05:29<00:00, 18.82it/s, loss=0.343]
Train Epoch 84 ==> 	accuracy: 0.8615, 	precision: 0.9996, 	recall: 0.7232, 	specificity: 0.9997, 	f1: 0.8392
Test Epoch 84: 100%|██████████| 1715/1715 [00:35<00:00, 48.18it/s, loss=0.622]
Test Epoch 84 ==> 	accuracy: 0.9452, 	precision: 0.9692, 	recall: 0.7507, 	specificity: 0.9940, 	f1: 0.8461
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 85: 100%|██████████| 6195/6195 [05:21<00:00, 19.28it/s, loss=0.385]
Train Epoch 85 ==> 	accuracy: 0.8602, 	precision: 0.9996, 	recall: 0.7207, 	specificity: 0.9997, 	f1: 0.8375
Test Epoch 85: 100%|██████████| 1715/1715 [00:36<00:00, 47.23it/s, loss=0.404]
Test Epoch 85 ==> 	accuracy: 0.9448, 	precision: 0.9669, 	recall: 0.7506, 	specificity: 0.9935, 	f1: 0.8452
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 86: 100%|██████████| 6195/6195 [05:28<00:00, 18.85it/s, loss=0.303]
Train Epoch 86 ==> 	accuracy: 0.8615, 	precision: 0.9996, 	recall: 0.7233, 	specificity: 0.9997, 	f1: 0.8393
Test Epoch 86: 100%|██████████| 1715/1715 [00:34<00:00, 50.20it/s, loss=0.299]
Test Epoch 86 ==> 	accuracy: 0.9446, 	precision: 0.9722, 	recall: 0.7452, 	specificity: 0.9946, 	f1: 0.8437
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 87: 100%|██████████| 6195/6195 [05:25<00:00, 19.05it/s, loss=0.393]
Train Epoch 87 ==> 	accuracy: 0.8619, 	precision: 0.9996, 	recall: 0.7241, 	specificity: 0.9997, 	f1: 0.8398
Test Epoch 87: 100%|██████████| 1715/1715 [00:33<00:00, 50.45it/s, loss=0.142]
Test Epoch 87 ==> 	accuracy: 0.9450, 	precision: 0.9686, 	recall: 0.7502, 	specificity: 0.9939, 	f1: 0.8455
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 88: 100%|██████████| 6195/6195 [05:24<00:00, 19.09it/s, loss=0.467]
Train Epoch 88 ==> 	accuracy: 0.8634, 	precision: 0.9996, 	recall: 0.7271, 	specificity: 0.9997, 	f1: 0.8419
Test Epoch 88: 100%|██████████| 1715/1715 [00:34<00:00, 49.29it/s, loss=2.18]
Test Epoch 88 ==> 	accuracy: 0.9459, 	precision: 0.9658, 	recall: 0.7573, 	specificity: 0.9933, 	f1: 0.8489
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 89: 100%|██████████| 6195/6195 [05:28<00:00, 18.86it/s, loss=0.42]
Train Epoch 89 ==> 	accuracy: 0.8628, 	precision: 0.9996, 	recall: 0.7259, 	specificity: 0.9997, 	f1: 0.8411
Test Epoch 89: 100%|██████████| 1715/1715 [00:34<00:00, 49.42it/s, loss=0.891]
Test Epoch 89 ==> 	accuracy: 0.9433, 	precision: 0.9621, 	recall: 0.7470, 	specificity: 0.9926, 	f1: 0.8410
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 90: 100%|██████████| 6195/6195 [05:22<00:00, 19.21it/s, loss=0.376]
Train Epoch 90 ==> 	accuracy: 0.8631, 	precision: 0.9996, 	recall: 0.7264, 	specificity: 0.9997, 	f1: 0.8414
Test Epoch 90: 100%|██████████| 1715/1715 [00:34<00:00, 50.04it/s, loss=0.273]
Test Epoch 90 ==> 	accuracy: 0.9447, 	precision: 0.9648, 	recall: 0.7521, 	specificity: 0.9931, 	f1: 0.8453
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 91: 100%|██████████| 6195/6195 [05:18<00:00, 19.47it/s, loss=0.371]
Train Epoch 91 ==> 	accuracy: 0.8626, 	precision: 0.9996, 	recall: 0.7255, 	specificity: 0.9997, 	f1: 0.8407
Test Epoch 91: 100%|██████████| 1715/1715 [00:34<00:00, 49.32it/s, loss=0.456]
Test Epoch 91 ==> 	accuracy: 0.9456, 	precision: 0.9676, 	recall: 0.7542, 	specificity: 0.9937, 	f1: 0.8477
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 92: 100%|██████████| 6195/6195 [05:24<00:00, 19.08it/s, loss=0.418]
Train Epoch 92 ==> 	accuracy: 0.8635, 	precision: 0.9996, 	recall: 0.7273, 	specificity: 0.9997, 	f1: 0.8420
Test Epoch 92: 100%|██████████| 1715/1715 [00:34<00:00, 49.46it/s, loss=1.12]
Test Epoch 92 ==> 	accuracy: 0.9453, 	precision: 0.9655, 	recall: 0.7546, 	specificity: 0.9932, 	f1: 0.8472
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 93: 100%|██████████| 6195/6195 [05:20<00:00, 19.32it/s, loss=0.388]
Train Epoch 93 ==> 	accuracy: 0.8638, 	precision: 0.9996, 	recall: 0.7278, 	specificity: 0.9997, 	f1: 0.8423
Test Epoch 93: 100%|██████████| 1715/1715 [00:33<00:00, 51.12it/s, loss=0.447]
Test Epoch 93 ==> 	accuracy: 0.9456, 	precision: 0.9745, 	recall: 0.7486, 	specificity: 0.9951, 	f1: 0.8467
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 94: 100%|██████████| 6195/6195 [05:19<00:00, 19.42it/s, loss=0.41]
Train Epoch 94 ==> 	accuracy: 0.8635, 	precision: 0.9996, 	recall: 0.7273, 	specificity: 0.9997, 	f1: 0.8420
Test Epoch 94: 100%|██████████| 1715/1715 [00:35<00:00, 48.13it/s, loss=0.309]
Test Epoch 94 ==> 	accuracy: 0.9471, 	precision: 0.9718, 	recall: 0.7583, 	specificity: 0.9945, 	f1: 0.8519
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 95: 100%|██████████| 6195/6195 [05:21<00:00, 19.25it/s, loss=0.366]
Train Epoch 95 ==> 	accuracy: 0.8646, 	precision: 0.9996, 	recall: 0.7294, 	specificity: 0.9997, 	f1: 0.8434
Test Epoch 95: 100%|██████████| 1715/1715 [00:33<00:00, 50.61it/s, loss=0.169]
Test Epoch 95 ==> 	accuracy: 0.9463, 	precision: 0.9716, 	recall: 0.7547, 	specificity: 0.9945, 	f1: 0.8495
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 96: 100%|██████████| 6195/6195 [05:31<00:00, 18.69it/s, loss=0.408]
Train Epoch 96 ==> 	accuracy: 0.8653, 	precision: 0.9996, 	recall: 0.7309, 	specificity: 0.9997, 	f1: 0.8444
Test Epoch 96: 100%|██████████| 1715/1715 [00:35<00:00, 48.45it/s, loss=0.988]
Test Epoch 96 ==> 	accuracy: 0.9440, 	precision: 0.9665, 	recall: 0.7469, 	specificity: 0.9935, 	f1: 0.8426
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 97: 100%|██████████| 6195/6195 [05:20<00:00, 19.33it/s, loss=0.405]
Train Epoch 97 ==> 	accuracy: 0.8649, 	precision: 0.9996, 	recall: 0.7301, 	specificity: 0.9997, 	f1: 0.8439
Test Epoch 97: 100%|██████████| 1715/1715 [00:32<00:00, 53.16it/s, loss=0.336]
Test Epoch 97 ==> 	accuracy: 0.9459, 	precision: 0.9684, 	recall: 0.7552, 	specificity: 0.9938, 	f1: 0.8486
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 98: 100%|██████████| 6195/6195 [05:19<00:00, 19.38it/s, loss=0.409]
Train Epoch 98 ==> 	accuracy: 0.8663, 	precision: 0.9996, 	recall: 0.7330, 	specificity: 0.9997, 	f1: 0.8458
Test Epoch 98: 100%|██████████| 1715/1715 [00:34<00:00, 50.20it/s, loss=0.138]
Test Epoch 98 ==> 	accuracy: 0.9456, 	precision: 0.9642, 	recall: 0.7569, 	specificity: 0.9929, 	f1: 0.8481
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 99: 100%|██████████| 6195/6195 [05:29<00:00, 18.78it/s, loss=0.41]
Train Epoch 99 ==> 	accuracy: 0.8663, 	precision: 0.9996, 	recall: 0.7328, 	specificity: 0.9997, 	f1: 0.8457
Test Epoch 99: 100%|██████████| 1715/1715 [00:36<00:00, 47.34it/s, loss=0.192]
Test Epoch 99 ==> 	accuracy: 0.9453, 	precision: 0.9738, 	recall: 0.7478, 	specificity: 0.9949, 	f1: 0.8460
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 100: 100%|██████████| 6195/6195 [05:27<00:00, 18.91it/s, loss=0.3]
Train Epoch 100 ==> 	accuracy: 0.8676, 	precision: 0.9997, 	recall: 0.7355, 	specificity: 0.9997, 	f1: 0.8475
Test Epoch 100: 100%|██████████| 1715/1715 [00:33<00:00, 50.85it/s, loss=1.31]
Test Epoch 100 ==> 	accuracy: 0.9466, 	precision: 0.9716, 	recall: 0.7559, 	specificity: 0.9945, 	f1: 0.8503
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 101: 100%|██████████| 6195/6195 [05:32<00:00, 18.65it/s, loss=0.423]
Train Epoch 101 ==> 	accuracy: 0.8658, 	precision: 0.9996, 	recall: 0.7319, 	specificity: 0.9997, 	f1: 0.8451
Test Epoch 101: 100%|██████████| 1715/1715 [00:35<00:00, 48.02it/s, loss=0.614]
Test Epoch 101 ==> 	accuracy: 0.9454, 	precision: 0.9666, 	recall: 0.7543, 	specificity: 0.9935, 	f1: 0.8473
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 102: 100%|██████████| 6195/6195 [05:18<00:00, 19.45it/s, loss=0.354]
Train Epoch 102 ==> 	accuracy: 0.8676, 	precision: 0.9997, 	recall: 0.7355, 	specificity: 0.9997, 	f1: 0.8475
Test Epoch 102: 100%|██████████| 1715/1715 [00:32<00:00, 52.21it/s, loss=2.03]
Test Epoch 102 ==> 	accuracy: 0.9457, 	precision: 0.9651, 	recall: 0.7571, 	specificity: 0.9931, 	f1: 0.8485
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 103: 100%|██████████| 6195/6195 [05:24<00:00, 19.09it/s, loss=0.33]
Train Epoch 103 ==> 	accuracy: 0.8666, 	precision: 0.9996, 	recall: 0.7334, 	specificity: 0.9997, 	f1: 0.8461
Test Epoch 103: 100%|██████████| 1715/1715 [00:35<00:00, 48.10it/s, loss=0.278]
Test Epoch 103 ==> 	accuracy: 0.9468, 	precision: 0.9720, 	recall: 0.7569, 	specificity: 0.9945, 	f1: 0.8510
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 104: 100%|██████████| 6195/6195 [05:28<00:00, 18.88it/s, loss=0.403]
Train Epoch 104 ==> 	accuracy: 0.8657, 	precision: 0.9997, 	recall: 0.7317, 	specificity: 0.9997, 	f1: 0.8449
Test Epoch 104: 100%|██████████| 1715/1715 [00:35<00:00, 48.45it/s, loss=0.147]
Test Epoch 104 ==> 	accuracy: 0.9460, 	precision: 0.9725, 	recall: 0.7525, 	specificity: 0.9947, 	f1: 0.8485
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 105: 100%|██████████| 6195/6195 [05:34<00:00, 18.51it/s, loss=0.395]
Train Epoch 105 ==> 	accuracy: 0.8678, 	precision: 0.9997, 	recall: 0.7359, 	specificity: 0.9997, 	f1: 0.8477
Test Epoch 105: 100%|██████████| 1715/1715 [00:36<00:00, 47.23it/s, loss=0.52]
Test Epoch 105 ==> 	accuracy: 0.9471, 	precision: 0.9707, 	recall: 0.7594, 	specificity: 0.9943, 	f1: 0.8522
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 106: 100%|██████████| 6195/6195 [05:25<00:00, 19.02it/s, loss=0.392]
Train Epoch 106 ==> 	accuracy: 0.8689, 	precision: 0.9996, 	recall: 0.7380, 	specificity: 0.9997, 	f1: 0.8491
Test Epoch 106: 100%|██████████| 1715/1715 [00:37<00:00, 46.08it/s, loss=0.456]
Test Epoch 106 ==> 	accuracy: 0.9470, 	precision: 0.9686, 	recall: 0.7608, 	specificity: 0.9938, 	f1: 0.8522
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 107: 100%|██████████| 6195/6195 [05:29<00:00, 18.81it/s, loss=0.378]
Train Epoch 107 ==> 	accuracy: 0.8685, 	precision: 0.9997, 	recall: 0.7372, 	specificity: 0.9998, 	f1: 0.8486
Test Epoch 107: 100%|██████████| 1715/1715 [00:33<00:00, 50.69it/s, loss=0.196]
Test Epoch 107 ==> 	accuracy: 0.9458, 	precision: 0.9736, 	recall: 0.7502, 	specificity: 0.9949, 	f1: 0.8474
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 108: 100%|██████████| 6195/6195 [05:21<00:00, 19.26it/s, loss=0.481]
Train Epoch 108 ==> 	accuracy: 0.8664, 	precision: 0.9996, 	recall: 0.7331, 	specificity: 0.9997, 	f1: 0.8459
Test Epoch 108: 100%|██████████| 1715/1715 [00:32<00:00, 52.38it/s, loss=1.43]
Test Epoch 108 ==> 	accuracy: 0.9454, 	precision: 0.9676, 	recall: 0.7535, 	specificity: 0.9937, 	f1: 0.8472
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 109: 100%|██████████| 6195/6195 [05:19<00:00, 19.38it/s, loss=0.349]
Train Epoch 109 ==> 	accuracy: 0.8713, 	precision: 0.9996, 	recall: 0.7429, 	specificity: 0.9997, 	f1: 0.8523
Test Epoch 109: 100%|██████████| 1715/1715 [00:35<00:00, 48.41it/s, loss=0.356]
Test Epoch 109 ==> 	accuracy: 0.9469, 	precision: 0.9665, 	recall: 0.7620, 	specificity: 0.9934, 	f1: 0.8522
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 110: 100%|██████████| 6195/6195 [05:20<00:00, 19.32it/s, loss=0.37]
Train Epoch 110 ==> 	accuracy: 0.8695, 	precision: 0.9997, 	recall: 0.7393, 	specificity: 0.9998, 	f1: 0.8500
Test Epoch 110: 100%|██████████| 1715/1715 [00:35<00:00, 48.44it/s, loss=0.339]
Test Epoch 110 ==> 	accuracy: 0.9467, 	precision: 0.9721, 	recall: 0.7563, 	specificity: 0.9946, 	f1: 0.8507
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 111: 100%|██████████| 6195/6195 [05:25<00:00, 19.04it/s, loss=0.462]
Train Epoch 111 ==> 	accuracy: 0.8697, 	precision: 0.9996, 	recall: 0.7397, 	specificity: 0.9997, 	f1: 0.8502
Test Epoch 111: 100%|██████████| 1715/1715 [00:34<00:00, 50.18it/s, loss=2.24]
Test Epoch 111 ==> 	accuracy: 0.9474, 	precision: 0.9638, 	recall: 0.7670, 	specificity: 0.9928, 	f1: 0.8542
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 112: 100%|██████████| 6195/6195 [05:23<00:00, 19.13it/s, loss=0.318]
Train Epoch 112 ==> 	accuracy: 0.8684, 	precision: 0.9996, 	recall: 0.7370, 	specificity: 0.9997, 	f1: 0.8485
Test Epoch 112: 100%|██████████| 1715/1715 [00:35<00:00, 47.93it/s, loss=0.475]
Test Epoch 112 ==> 	accuracy: 0.9449, 	precision: 0.9618, 	recall: 0.7558, 	specificity: 0.9925, 	f1: 0.8464
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 113: 100%|██████████| 6195/6195 [05:25<00:00, 19.01it/s, loss=0.35]
Train Epoch 113 ==> 	accuracy: 0.8717, 	precision: 0.9997, 	recall: 0.7436, 	specificity: 0.9998, 	f1: 0.8529
Test Epoch 113: 100%|██████████| 1715/1715 [00:32<00:00, 52.49it/s, loss=1.85]
Test Epoch 113 ==> 	accuracy: 0.9480, 	precision: 0.9636, 	recall: 0.7701, 	specificity: 0.9927, 	f1: 0.8561
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 114: 100%|██████████| 6195/6195 [05:25<00:00, 19.03it/s, loss=0.388]
Train Epoch 114 ==> 	accuracy: 0.8702, 	precision: 0.9997, 	recall: 0.7406, 	specificity: 0.9997, 	f1: 0.8508
Test Epoch 114: 100%|██████████| 1715/1715 [00:36<00:00, 46.99it/s, loss=0.214]
Test Epoch 114 ==> 	accuracy: 0.9471, 	precision: 0.9736, 	recall: 0.7571, 	specificity: 0.9948, 	f1: 0.8518
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 115: 100%|██████████| 6195/6195 [05:35<00:00, 18.48it/s, loss=0.316]
Train Epoch 115 ==> 	accuracy: 0.8703, 	precision: 0.9997, 	recall: 0.7409, 	specificity: 0.9998, 	f1: 0.8510
Test Epoch 115: 100%|██████████| 1715/1715 [00:34<00:00, 49.16it/s, loss=0.574]
Test Epoch 115 ==> 	accuracy: 0.9479, 	precision: 0.9664, 	recall: 0.7673, 	specificity: 0.9933, 	f1: 0.8554
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 116: 100%|██████████| 6195/6195 [05:27<00:00, 18.90it/s, loss=0.307]
Train Epoch 116 ==> 	accuracy: 0.8702, 	precision: 0.9997, 	recall: 0.7406, 	specificity: 0.9998, 	f1: 0.8509
Test Epoch 116: 100%|██████████| 1715/1715 [00:38<00:00, 44.80it/s, loss=2.39]
Test Epoch 116 ==> 	accuracy: 0.9479, 	precision: 0.9699, 	recall: 0.7639, 	specificity: 0.9941, 	f1: 0.8547
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 117: 100%|██████████| 6195/6195 [05:35<00:00, 18.47it/s, loss=0.321]
Train Epoch 117 ==> 	accuracy: 0.8704, 	precision: 0.9997, 	recall: 0.7411, 	specificity: 0.9998, 	f1: 0.8512
Test Epoch 117: 100%|██████████| 1715/1715 [00:36<00:00, 46.77it/s, loss=0.126]
Test Epoch 117 ==> 	accuracy: 0.9474, 	precision: 0.9682, 	recall: 0.7631, 	specificity: 0.9937, 	f1: 0.8535
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 118: 100%|██████████| 6195/6195 [05:33<00:00, 18.59it/s, loss=0.44]
Train Epoch 118 ==> 	accuracy: 0.8700, 	precision: 0.9997, 	recall: 0.7403, 	specificity: 0.9998, 	f1: 0.8506
Test Epoch 118: 100%|██████████| 1715/1715 [00:35<00:00, 48.48it/s, loss=0.312]
Test Epoch 118 ==> 	accuracy: 0.9477, 	precision: 0.9655, 	recall: 0.7670, 	specificity: 0.9931, 	f1: 0.8549
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 119: 100%|██████████| 6195/6195 [05:30<00:00, 18.76it/s, loss=0.333]
Train Epoch 119 ==> 	accuracy: 0.8707, 	precision: 0.9997, 	recall: 0.7416, 	specificity: 0.9998, 	f1: 0.8515
Test Epoch 119: 100%|██████████| 1715/1715 [00:38<00:00, 45.06it/s, loss=1.22]
Test Epoch 119 ==> 	accuracy: 0.9469, 	precision: 0.9728, 	recall: 0.7565, 	specificity: 0.9947, 	f1: 0.8511
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 120: 100%|██████████| 6195/6195 [05:36<00:00, 18.38it/s, loss=0.315]
Train Epoch 120 ==> 	accuracy: 0.8707, 	precision: 0.9997, 	recall: 0.7417, 	specificity: 0.9997, 	f1: 0.8516
Test Epoch 120: 100%|██████████| 1715/1715 [00:36<00:00, 46.53it/s, loss=0.399]
Test Epoch 120 ==> 	accuracy: 0.9475, 	precision: 0.9683, 	recall: 0.7635, 	specificity: 0.9937, 	f1: 0.8538
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 121: 100%|██████████| 6195/6195 [05:31<00:00, 18.68it/s, loss=0.332]
Train Epoch 121 ==> 	accuracy: 0.8717, 	precision: 0.9997, 	recall: 0.7437, 	specificity: 0.9998, 	f1: 0.8529
Test Epoch 121: 100%|██████████| 1715/1715 [00:35<00:00, 48.22it/s, loss=4.23]
Test Epoch 121 ==> 	accuracy: 0.9473, 	precision: 0.9654, 	recall: 0.7651, 	specificity: 0.9931, 	f1: 0.8537
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 122: 100%|██████████| 6195/6195 [05:36<00:00, 18.41it/s, loss=0.396]
Train Epoch 122 ==> 	accuracy: 0.8745, 	precision: 0.9997, 	recall: 0.7493, 	specificity: 0.9997, 	f1: 0.8566
Test Epoch 122: 100%|██████████| 1715/1715 [00:36<00:00, 47.17it/s, loss=0.15]
Test Epoch 122 ==> 	accuracy: 0.9477, 	precision: 0.9644, 	recall: 0.7681, 	specificity: 0.9929, 	f1: 0.8551
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 123: 100%|██████████| 6195/6195 [05:31<00:00, 18.68it/s, loss=0.451]
Train Epoch 123 ==> 	accuracy: 0.8718, 	precision: 0.9997, 	recall: 0.7439, 	specificity: 0.9998, 	f1: 0.8530
Test Epoch 123: 100%|██████████| 1715/1715 [00:34<00:00, 49.68it/s, loss=0.294]
Test Epoch 123 ==> 	accuracy: 0.9478, 	precision: 0.9666, 	recall: 0.7663, 	specificity: 0.9934, 	f1: 0.8549
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 124: 100%|██████████| 6195/6195 [05:20<00:00, 19.32it/s, loss=0.323]
Train Epoch 124 ==> 	accuracy: 0.8728, 	precision: 0.9997, 	recall: 0.7459, 	specificity: 0.9998, 	f1: 0.8544
Test Epoch 124: 100%|██████████| 1715/1715 [00:35<00:00, 48.04it/s, loss=0.222]
Test Epoch 124 ==> 	accuracy: 0.9476, 	precision: 0.9623, 	recall: 0.7691, 	specificity: 0.9924, 	f1: 0.8549
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 125: 100%|██████████| 6195/6195 [05:30<00:00, 18.77it/s, loss=0.402]
Train Epoch 125 ==> 	accuracy: 0.8725, 	precision: 0.9997, 	recall: 0.7453, 	specificity: 0.9997, 	f1: 0.8539
Test Epoch 125: 100%|██████████| 1715/1715 [00:34<00:00, 49.61it/s, loss=2.62]
Test Epoch 125 ==> 	accuracy: 0.9474, 	precision: 0.9610, 	recall: 0.7695, 	specificity: 0.9922, 	f1: 0.8546
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 126: 100%|██████████| 6195/6195 [05:25<00:00, 19.04it/s, loss=0.426]
Train Epoch 126 ==> 	accuracy: 0.8754, 	precision: 0.9997, 	recall: 0.7510, 	specificity: 0.9997, 	f1: 0.8576
Test Epoch 126: 100%|██████████| 1715/1715 [00:35<00:00, 48.22it/s, loss=1.14]
Test Epoch 126 ==> 	accuracy: 0.9486, 	precision: 0.9640, 	recall: 0.7727, 	specificity: 0.9927, 	f1: 0.8578
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 127: 100%|██████████| 6195/6195 [05:28<00:00, 18.88it/s, loss=0.367]
Train Epoch 127 ==> 	accuracy: 0.8712, 	precision: 0.9997, 	recall: 0.7426, 	specificity: 0.9998, 	f1: 0.8521
Test Epoch 127: 100%|██████████| 1715/1715 [00:32<00:00, 52.63it/s, loss=1.42]
Test Epoch 127 ==> 	accuracy: 0.9473, 	precision: 0.9671, 	recall: 0.7634, 	specificity: 0.9935, 	f1: 0.8533
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 128: 100%|██████████| 6195/6195 [05:23<00:00, 19.15it/s, loss=0.403]
Train Epoch 128 ==> 	accuracy: 0.8715, 	precision: 0.9997, 	recall: 0.7433, 	specificity: 0.9998, 	f1: 0.8526
Test Epoch 128: 100%|██████████| 1715/1715 [00:34<00:00, 49.52it/s, loss=1.04]
Test Epoch 128 ==> 	accuracy: 0.9502, 	precision: 0.9749, 	recall: 0.7719, 	specificity: 0.9950, 	f1: 0.8616
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 129: 100%|██████████| 6195/6195 [05:28<00:00, 18.89it/s, loss=0.355]
Train Epoch 129 ==> 	accuracy: 0.8717, 	precision: 0.9997, 	recall: 0.7436, 	specificity: 0.9997, 	f1: 0.8528
Test Epoch 129: 100%|██████████| 1715/1715 [00:37<00:00, 45.25it/s, loss=2.3]
Test Epoch 129 ==> 	accuracy: 0.9495, 	precision: 0.9771, 	recall: 0.7665, 	specificity: 0.9955, 	f1: 0.8591
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 130: 100%|██████████| 6195/6195 [05:25<00:00, 19.01it/s, loss=0.351]
Train Epoch 130 ==> 	accuracy: 0.8723, 	precision: 0.9997, 	recall: 0.7448, 	specificity: 0.9998, 	f1: 0.8536
Test Epoch 130: 100%|██████████| 1715/1715 [00:36<00:00, 47.49it/s, loss=1.28]
Test Epoch 130 ==> 	accuracy: 0.9497, 	precision: 0.9765, 	recall: 0.7681, 	specificity: 0.9954, 	f1: 0.8598
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 131: 100%|██████████| 6195/6195 [05:30<00:00, 18.74it/s, loss=0.377]
Train Epoch 131 ==> 	accuracy: 0.8709, 	precision: 0.9997, 	recall: 0.7421, 	specificity: 0.9998, 	f1: 0.8519
Test Epoch 131: 100%|██████████| 1715/1715 [00:34<00:00, 50.05it/s, loss=0.248]
Test Epoch 131 ==> 	accuracy: 0.9495, 	precision: 0.9770, 	recall: 0.7663, 	specificity: 0.9955, 	f1: 0.8589
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 132: 100%|██████████| 6195/6195 [05:39<00:00, 18.23it/s, loss=0.425]
Train Epoch 132 ==> 	accuracy: 0.8723, 	precision: 0.9997, 	recall: 0.7449, 	specificity: 0.9998, 	f1: 0.8537
Test Epoch 132: 100%|██████████| 1715/1715 [00:33<00:00, 51.07it/s, loss=2.14]
Test Epoch 132 ==> 	accuracy: 0.9493, 	precision: 0.9775, 	recall: 0.7651, 	specificity: 0.9956, 	f1: 0.8583
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 133: 100%|██████████| 6195/6195 [05:36<00:00, 18.42it/s, loss=0.412]
Train Epoch 133 ==> 	accuracy: 0.8731, 	precision: 0.9997, 	recall: 0.7464, 	specificity: 0.9997, 	f1: 0.8547
Test Epoch 133: 100%|██████████| 1715/1715 [00:37<00:00, 45.73it/s, loss=2]
Test Epoch 133 ==> 	accuracy: 0.9497, 	precision: 0.9764, 	recall: 0.7681, 	specificity: 0.9953, 	f1: 0.8598
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 134: 100%|██████████| 6195/6195 [05:30<00:00, 18.76it/s, loss=0.468]
Train Epoch 134 ==> 	accuracy: 0.8718, 	precision: 0.9997, 	recall: 0.7439, 	specificity: 0.9998, 	f1: 0.8530
Test Epoch 134: 100%|██████████| 1715/1715 [00:35<00:00, 48.27it/s, loss=2.83]
Test Epoch 134 ==> 	accuracy: 0.9496, 	precision: 0.9748, 	recall: 0.7691, 	specificity: 0.9950, 	f1: 0.8598
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 135: 100%|██████████| 6195/6195 [05:32<00:00, 18.65it/s, loss=0.535]
Train Epoch 135 ==> 	accuracy: 0.8746, 	precision: 0.9997, 	recall: 0.7495, 	specificity: 0.9997, 	f1: 0.8567
Test Epoch 135: 100%|██████████| 1715/1715 [00:35<00:00, 48.72it/s, loss=1.07]
Test Epoch 135 ==> 	accuracy: 0.9494, 	precision: 0.9716, 	recall: 0.7707, 	specificity: 0.9943, 	f1: 0.8595
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 136: 100%|██████████| 6195/6195 [05:29<00:00, 18.82it/s, loss=0.299]
Train Epoch 136 ==> 	accuracy: 0.8741, 	precision: 0.9997, 	recall: 0.7485, 	specificity: 0.9998, 	f1: 0.8560
Test Epoch 136: 100%|██████████| 1715/1715 [00:34<00:00, 49.04it/s, loss=0.199]
Test Epoch 136 ==> 	accuracy: 0.9504, 	precision: 0.9735, 	recall: 0.7742, 	specificity: 0.9947, 	f1: 0.8625
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 137: 100%|██████████| 6195/6195 [05:32<00:00, 18.66it/s, loss=0.517]
Train Epoch 137 ==> 	accuracy: 0.8740, 	precision: 0.9997, 	recall: 0.7482, 	specificity: 0.9998, 	f1: 0.8558
Test Epoch 137: 100%|██████████| 1715/1715 [00:35<00:00, 48.50it/s, loss=0.359]
Test Epoch 137 ==> 	accuracy: 0.9503, 	precision: 0.9715, 	recall: 0.7752, 	specificity: 0.9943, 	f1: 0.8624
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 138: 100%|██████████| 6195/6195 [05:32<00:00, 18.64it/s, loss=0.386]
Train Epoch 138 ==> 	accuracy: 0.8738, 	precision: 0.9997, 	recall: 0.7477, 	specificity: 0.9998, 	f1: 0.8555
Test Epoch 138: 100%|██████████| 1715/1715 [00:35<00:00, 48.75it/s, loss=0.565]
Test Epoch 138 ==> 	accuracy: 0.9494, 	precision: 0.9734, 	recall: 0.7689, 	specificity: 0.9947, 	f1: 0.8592
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 139: 100%|██████████| 6195/6195 [05:23<00:00, 19.17it/s, loss=0.429]
Train Epoch 139 ==> 	accuracy: 0.8761, 	precision: 0.9997, 	recall: 0.7523, 	specificity: 0.9998, 	f1: 0.8586
Test Epoch 139: 100%|██████████| 1715/1715 [00:34<00:00, 49.82it/s, loss=1.31]
Test Epoch 139 ==> 	accuracy: 0.9507, 	precision: 0.9706, 	recall: 0.7779, 	specificity: 0.9941, 	f1: 0.8637
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 140: 100%|██████████| 6195/6195 [05:24<00:00, 19.08it/s, loss=0.37]
Train Epoch 140 ==> 	accuracy: 0.8738, 	precision: 0.9997, 	recall: 0.7479, 	specificity: 0.9997, 	f1: 0.8557
Test Epoch 140: 100%|██████████| 1715/1715 [00:36<00:00, 47.60it/s, loss=0.0737]
Test Epoch 140 ==> 	accuracy: 0.9503, 	precision: 0.9729, 	recall: 0.7740, 	specificity: 0.9946, 	f1: 0.8621
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 141: 100%|██████████| 6195/6195 [05:19<00:00, 19.37it/s, loss=0.399]
Train Epoch 141 ==> 	accuracy: 0.8735, 	precision: 0.9997, 	recall: 0.7472, 	specificity: 0.9998, 	f1: 0.8552
Test Epoch 141: 100%|██████████| 1715/1715 [00:33<00:00, 50.68it/s, loss=2.29]
Test Epoch 141 ==> 	accuracy: 0.9497, 	precision: 0.9742, 	recall: 0.7700, 	specificity: 0.9949, 	f1: 0.8602
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 142: 100%|██████████| 6195/6195 [05:27<00:00, 18.90it/s, loss=0.368]
Train Epoch 142 ==> 	accuracy: 0.8736, 	precision: 0.9997, 	recall: 0.7475, 	specificity: 0.9998, 	f1: 0.8554
Test Epoch 142: 100%|██████████| 1715/1715 [00:34<00:00, 49.61it/s, loss=0.194]
Test Epoch 142 ==> 	accuracy: 0.9499, 	precision: 0.9749, 	recall: 0.7704, 	specificity: 0.9950, 	f1: 0.8607
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 143: 100%|██████████| 6195/6195 [05:32<00:00, 18.63it/s, loss=0.462]
Train Epoch 143 ==> 	accuracy: 0.8735, 	precision: 0.9997, 	recall: 0.7472, 	specificity: 0.9998, 	f1: 0.8552
Test Epoch 143: 100%|██████████| 1715/1715 [00:35<00:00, 48.23it/s, loss=0.169]
Test Epoch 143 ==> 	accuracy: 0.9503, 	precision: 0.9752, 	recall: 0.7721, 	specificity: 0.9951, 	f1: 0.8618
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 144: 100%|██████████| 6195/6195 [05:23<00:00, 19.14it/s, loss=0.416]
Train Epoch 144 ==> 	accuracy: 0.8755, 	precision: 0.9997, 	recall: 0.7513, 	specificity: 0.9998, 	f1: 0.8579
Test Epoch 144: 100%|██████████| 1715/1715 [00:37<00:00, 45.56it/s, loss=2.7]
Test Epoch 144 ==> 	accuracy: 0.9510, 	precision: 0.9725, 	recall: 0.7780, 	specificity: 0.9945, 	f1: 0.8644
Adjusting learning rate of group 0 to 5.2335e-06.
Train Epoch 145: 100%|██████████| 6195/6195 [05:29<00:00, 18.82it/s, loss=0.403]
Train Epoch 145 ==> 	accuracy: 0.8737, 	precision: 0.9997, 	recall: 0.7477, 	specificity: 0.9998, 	f1: 0.8555
Test Epoch 145: 100%|██████████| 1715/1715 [00:35<00:00, 47.68it/s, loss=0.363]
Test Epoch 145 ==> 	accuracy: 0.9493, 	precision: 0.9735, 	recall: 0.7685, 	specificity: 0.9948, 	f1: 0.8590
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 146: 100%|██████████| 6195/6195 [05:24<00:00, 19.10it/s, loss=0.397]
Train Epoch 146 ==> 	accuracy: 0.8746, 	precision: 0.9997, 	recall: 0.7495, 	specificity: 0.9998, 	f1: 0.8567
Test Epoch 146: 100%|██████████| 1715/1715 [00:34<00:00, 50.18it/s, loss=0.599]
Test Epoch 146 ==> 	accuracy: 0.9503, 	precision: 0.9743, 	recall: 0.7727, 	specificity: 0.9949, 	f1: 0.8619
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 147: 100%|██████████| 6195/6195 [05:24<00:00, 19.10it/s, loss=0.362]
Train Epoch 147 ==> 	accuracy: 0.8748, 	precision: 0.9997, 	recall: 0.7498, 	specificity: 0.9998, 	f1: 0.8569
Test Epoch 147: 100%|██████████| 1715/1715 [00:37<00:00, 46.20it/s, loss=0.515]
Test Epoch 147 ==> 	accuracy: 0.9504, 	precision: 0.9729, 	recall: 0.7748, 	specificity: 0.9946, 	f1: 0.8626
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 148: 100%|██████████| 6195/6195 [05:22<00:00, 19.20it/s, loss=0.218]
Train Epoch 148 ==> 	accuracy: 0.8754, 	precision: 0.9997, 	recall: 0.7511, 	specificity: 0.9998, 	f1: 0.8578
Test Epoch 148: 100%|██████████| 1715/1715 [00:33<00:00, 51.09it/s, loss=2.52]
Test Epoch 148 ==> 	accuracy: 0.9506, 	precision: 0.9719, 	recall: 0.7766, 	specificity: 0.9944, 	f1: 0.8633
Adjusting learning rate of group 0 to 4.7101e-06.
Train Epoch 149: 100%|██████████| 6195/6195 [05:19<00:00, 19.41it/s, loss=0.449]
Train Epoch 149 ==> 	accuracy: 0.8752, 	precision: 0.9997, 	recall: 0.7506, 	specificity: 0.9998, 	f1: 0.8574
Test Epoch 149: 100%|██████████| 1715/1715 [00:34<00:00, 49.90it/s, loss=4.78]
Test Epoch 149 ==> 	accuracy: 0.9493, 	precision: 0.9746, 	recall: 0.7675, 	specificity: 0.9950, 	f1: 0.8587
Adjusting learning rate of group 0 to 4.2391e-06.

Process finished with exit code 0

'''

#   7mer
"""
'../model_save_sigBlock4_focalWithMs_deformable_7mer_signal&seq'

/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6507/6507 [09:55<00:00, 10.93it/s, loss=0.0117]
Train Epoch 0 ==> 	accuracy: 0.7768, 	precision: 0.9987, 	recall: 0.5543, 	specificity: 0.9993, 	f1: 0.7129
Test Epoch 0: 100%|██████████| 1768/1768 [01:02<00:00, 28.26it/s, loss=0.161]
Test Epoch 0 ==> 	accuracy: 0.9048, 	precision: 0.9993, 	recall: 0.5348, 	specificity: 0.9999, 	f1: 0.6968
Train Epoch 1: 100%|██████████| 6507/6507 [10:09<00:00, 10.68it/s, loss=0.0369]
Train Epoch 1 ==> 	accuracy: 0.8559, 	precision: 0.9993, 	recall: 0.7123, 	specificity: 0.9995, 	f1: 0.8317
Test Epoch 1: 100%|██████████| 1768/1768 [01:05<00:00, 26.93it/s, loss=0.114]
Test Epoch 1 ==> 	accuracy: 0.9385, 	precision: 0.9996, 	recall: 0.6995, 	specificity: 0.9999, 	f1: 0.8231
Train Epoch 2: 100%|██████████| 6507/6507 [10:20<00:00, 10.49it/s, loss=1.42]
Train Epoch 2 ==> 	accuracy: 0.8829, 	precision: 0.9994, 	recall: 0.7663, 	specificity: 0.9995, 	f1: 0.8674
Test Epoch 2: 100%|██████████| 1768/1768 [01:04<00:00, 27.46it/s, loss=0.113]
Test Epoch 2 ==> 	accuracy: 0.9437, 	precision: 0.9983, 	recall: 0.7262, 	specificity: 0.9997, 	f1: 0.8408
Train Epoch 3: 100%|██████████| 6507/6507 [10:15<00:00, 10.57it/s, loss=0.0753]
Train Epoch 3 ==> 	accuracy: 0.8895, 	precision: 0.9995, 	recall: 0.7794, 	specificity: 0.9996, 	f1: 0.8758
Test Epoch 3: 100%|██████████| 1768/1768 [01:05<00:00, 27.02it/s, loss=0.0916]
Test Epoch 3 ==> 	accuracy: 0.9560, 	precision: 0.9991, 	recall: 0.7857, 	specificity: 0.9998, 	f1: 0.8797
Train Epoch 4: 100%|██████████| 6507/6507 [10:14<00:00, 10.59it/s, loss=0.285]
Train Epoch 4 ==> 	accuracy: 0.8979, 	precision: 0.9994, 	recall: 0.7963, 	specificity: 0.9996, 	f1: 0.8864
Test Epoch 4: 100%|██████████| 1768/1768 [01:04<00:00, 27.52it/s, loss=0.172]
Test Epoch 4 ==> 	accuracy: 0.9548, 	precision: 0.9982, 	recall: 0.7803, 	specificity: 0.9996, 	f1: 0.8759
Train Epoch 5: 100%|██████████| 6507/6507 [10:15<00:00, 10.58it/s, loss=0.0585]
Train Epoch 5 ==> 	accuracy: 0.9046, 	precision: 0.9995, 	recall: 0.8096, 	specificity: 0.9996, 	f1: 0.8946
Test Epoch 5: 100%|██████████| 1768/1768 [01:02<00:00, 28.37it/s, loss=0.169]
Test Epoch 5 ==> 	accuracy: 0.9575, 	precision: 0.9991, 	recall: 0.7927, 	specificity: 0.9998, 	f1: 0.8840
Train Epoch 6: 100%|██████████| 6507/6507 [10:15<00:00, 10.57it/s, loss=0.0007]
Train Epoch 6 ==> 	accuracy: 0.9076, 	precision: 0.9996, 	recall: 0.8155, 	specificity: 0.9997, 	f1: 0.8982
Test Epoch 6: 100%|██████████| 1768/1768 [01:02<00:00, 28.46it/s, loss=0.155]
Test Epoch 6 ==> 	accuracy: 0.9583, 	precision: 0.9985, 	recall: 0.7971, 	specificity: 0.9997, 	f1: 0.8865
Train Epoch 7: 100%|██████████| 6507/6507 [10:17<00:00, 10.53it/s, loss=0.0115]
Train Epoch 7 ==> 	accuracy: 0.9126, 	precision: 0.9996, 	recall: 0.8255, 	specificity: 0.9996, 	f1: 0.9042
Test Epoch 7: 100%|██████████| 1768/1768 [01:04<00:00, 27.26it/s, loss=0.0697]
Test Epoch 7 ==> 	accuracy: 0.9656, 	precision: 0.9980, 	recall: 0.8337, 	specificity: 0.9996, 	f1: 0.9085
Train Epoch 8: 100%|██████████| 6507/6507 [10:14<00:00, 10.60it/s, loss=1.47]
Train Epoch 8 ==> 	accuracy: 0.9098, 	precision: 0.9996, 	recall: 0.8199, 	specificity: 0.9997, 	f1: 0.9009
Test Epoch 8: 100%|██████████| 1768/1768 [01:06<00:00, 26.64it/s, loss=0.0373]
Test Epoch 8 ==> 	accuracy: 0.9633, 	precision: 0.9982, 	recall: 0.8222, 	specificity: 0.9996, 	f1: 0.9017
Train Epoch 9: 100%|██████████| 6507/6507 [10:20<00:00, 10.49it/s, loss=0.0031]
Train Epoch 9 ==> 	accuracy: 0.9154, 	precision: 0.9996, 	recall: 0.8310, 	specificity: 0.9997, 	f1: 0.9076
Test Epoch 9: 100%|██████████| 1768/1768 [01:04<00:00, 27.51it/s, loss=0.0716]
Test Epoch 9 ==> 	accuracy: 0.9690, 	precision: 0.9977, 	recall: 0.8505, 	specificity: 0.9995, 	f1: 0.9182
Train Epoch 10: 100%|██████████| 6507/6507 [10:13<00:00, 10.60it/s, loss=0.563]
Train Epoch 10 ==> 	accuracy: 0.9167, 	precision: 0.9996, 	recall: 0.8337, 	specificity: 0.9997, 	f1: 0.9092
Test Epoch 10: 100%|██████████| 1768/1768 [01:03<00:00, 28.01it/s, loss=0.131]
Test Epoch 10 ==> 	accuracy: 0.9661, 	precision: 0.9974, 	recall: 0.8365, 	specificity: 0.9994, 	f1: 0.9098
Train Epoch 11: 100%|██████████| 6507/6507 [10:18<00:00, 10.51it/s, loss=0.0723]
Train Epoch 11 ==> 	accuracy: 0.9180, 	precision: 0.9996, 	recall: 0.8362, 	specificity: 0.9997, 	f1: 0.9107
Test Epoch 11: 100%|██████████| 1768/1768 [01:05<00:00, 26.90it/s, loss=0.0535]
Test Epoch 11 ==> 	accuracy: 0.9686, 	precision: 0.9986, 	recall: 0.8475, 	specificity: 0.9997, 	f1: 0.9169
Train Epoch 12: 100%|██████████| 6507/6507 [10:19<00:00, 10.51it/s, loss=2.16]
Train Epoch 12 ==> 	accuracy: 0.9186, 	precision: 0.9997, 	recall: 0.8374, 	specificity: 0.9997, 	f1: 0.9114
Test Epoch 12: 100%|██████████| 1768/1768 [01:05<00:00, 26.81it/s, loss=0.0873]
Test Epoch 12 ==> 	accuracy: 0.9670, 	precision: 0.9982, 	recall: 0.8404, 	specificity: 0.9996, 	f1: 0.9125
Train Epoch 13: 100%|██████████| 6507/6507 [10:11<00:00, 10.63it/s, loss=0.002]
Train Epoch 13 ==> 	accuracy: 0.9214, 	precision: 0.9997, 	recall: 0.8431, 	specificity: 0.9997, 	f1: 0.9147
Test Epoch 13: 100%|██████████| 1768/1768 [01:06<00:00, 26.41it/s, loss=0.0929]
Test Epoch 13 ==> 	accuracy: 0.9693, 	precision: 0.9978, 	recall: 0.8517, 	specificity: 0.9995, 	f1: 0.9190
Train Epoch 14: 100%|██████████| 6507/6507 [10:23<00:00, 10.43it/s, loss=1.76]
Train Epoch 14 ==> 	accuracy: 0.9239, 	precision: 0.9997, 	recall: 0.8481, 	specificity: 0.9997, 	f1: 0.9177
Test Epoch 14: 100%|██████████| 1768/1768 [01:10<00:00, 25.02it/s, loss=0.138]
Test Epoch 14 ==> 	accuracy: 0.9719, 	precision: 0.9971, 	recall: 0.8654, 	specificity: 0.9993, 	f1: 0.9266
Train Epoch 15: 100%|██████████| 6507/6507 [10:22<00:00, 10.45it/s, loss=0.118]
Train Epoch 15 ==> 	accuracy: 0.9213, 	precision: 0.9997, 	recall: 0.8428, 	specificity: 0.9997, 	f1: 0.9146
Test Epoch 15: 100%|██████████| 1768/1768 [01:03<00:00, 27.75it/s, loss=0.0384]
Test Epoch 15 ==> 	accuracy: 0.9693, 	precision: 0.9986, 	recall: 0.8510, 	specificity: 0.9997, 	f1: 0.9189
Train Epoch 16: 100%|██████████| 6507/6507 [10:19<00:00, 10.50it/s, loss=0.0001]
Train Epoch 16 ==> 	accuracy: 0.9254, 	precision: 0.9997, 	recall: 0.8511, 	specificity: 0.9997, 	f1: 0.9195
Test Epoch 16: 100%|██████████| 1768/1768 [01:03<00:00, 27.97it/s, loss=0.0338]
Test Epoch 16 ==> 	accuracy: 0.9717, 	precision: 0.9974, 	recall: 0.8640, 	specificity: 0.9994, 	f1: 0.9260
Train Epoch 17: 100%|██████████| 6507/6507 [10:26<00:00, 10.38it/s, loss=0.2]
Train Epoch 17 ==> 	accuracy: 0.9250, 	precision: 0.9997, 	recall: 0.8502, 	specificity: 0.9997, 	f1: 0.9189
Test Epoch 17: 100%|██████████| 1768/1768 [01:05<00:00, 26.89it/s, loss=0.0983]
Test Epoch 17 ==> 	accuracy: 0.9721, 	precision: 0.9961, 	recall: 0.8671, 	specificity: 0.9991, 	f1: 0.9271
Train Epoch 18: 100%|██████████| 6507/6507 [10:22<00:00, 10.45it/s, loss=0.0407]
Train Epoch 18 ==> 	accuracy: 0.9287, 	precision: 0.9997, 	recall: 0.8576, 	specificity: 0.9998, 	f1: 0.9232
Test Epoch 18: 100%|██████████| 1768/1768 [01:04<00:00, 27.48it/s, loss=0.0646]
Test Epoch 18 ==> 	accuracy: 0.9719, 	precision: 0.9973, 	recall: 0.8648, 	specificity: 0.9994, 	f1: 0.9264
Train Epoch 19: 100%|██████████| 6507/6507 [10:22<00:00, 10.46it/s, loss=0.0012]
Train Epoch 19 ==> 	accuracy: 0.9271, 	precision: 0.9997, 	recall: 0.8544, 	specificity: 0.9998, 	f1: 0.9213
Test Epoch 19: 100%|██████████| 1768/1768 [01:04<00:00, 27.33it/s, loss=0.119]
Test Epoch 19 ==> 	accuracy: 0.9679, 	precision: 0.9990, 	recall: 0.8437, 	specificity: 0.9998, 	f1: 0.9148
Train Epoch 20: 100%|██████████| 6507/6507 [10:26<00:00, 10.39it/s, loss=0]
Train Epoch 20 ==> 	accuracy: 0.9281, 	precision: 0.9997, 	recall: 0.8564, 	specificity: 0.9998, 	f1: 0.9226
Test Epoch 20: 100%|██████████| 1768/1768 [01:04<00:00, 27.47it/s, loss=0.0312]
Test Epoch 20 ==> 	accuracy: 0.9726, 	precision: 0.9967, 	recall: 0.8687, 	specificity: 0.9993, 	f1: 0.9283
Train Epoch 21: 100%|██████████| 6507/6507 [10:22<00:00, 10.46it/s, loss=0.0001]
Train Epoch 21 ==> 	accuracy: 0.9292, 	precision: 0.9997, 	recall: 0.8587, 	specificity: 0.9998, 	f1: 0.9238
Test Epoch 21: 100%|██████████| 1768/1768 [01:04<00:00, 27.41it/s, loss=0.0689]
Test Epoch 21 ==> 	accuracy: 0.9740, 	precision: 0.9971, 	recall: 0.8754, 	specificity: 0.9993, 	f1: 0.9323
Train Epoch 22: 100%|██████████| 6507/6507 [10:29<00:00, 10.34it/s, loss=0.0013]
Train Epoch 22 ==> 	accuracy: 0.9303, 	precision: 0.9997, 	recall: 0.8608, 	specificity: 0.9998, 	f1: 0.9251
Test Epoch 22: 100%|██████████| 1768/1768 [01:07<00:00, 26.33it/s, loss=0.0474]
Test Epoch 22 ==> 	accuracy: 0.9753, 	precision: 0.9968, 	recall: 0.8821, 	specificity: 0.9993, 	f1: 0.9360
Train Epoch 23: 100%|██████████| 6507/6507 [10:16<00:00, 10.56it/s, loss=0.0608]
Train Epoch 23 ==> 	accuracy: 0.9299, 	precision: 0.9997, 	recall: 0.8600, 	specificity: 0.9998, 	f1: 0.9246
Test Epoch 23: 100%|██████████| 1768/1768 [01:03<00:00, 27.92it/s, loss=0.0913]
Test Epoch 23 ==> 	accuracy: 0.9728, 	precision: 0.9972, 	recall: 0.8695, 	specificity: 0.9994, 	f1: 0.9289
Train Epoch 24: 100%|██████████| 6507/6507 [10:22<00:00, 10.45it/s, loss=0.0005]
Train Epoch 24 ==> 	accuracy: 0.9318, 	precision: 0.9997, 	recall: 0.8638, 	specificity: 0.9998, 	f1: 0.9268
Test Epoch 24: 100%|██████████| 1768/1768 [01:02<00:00, 28.30it/s, loss=0.0857]
Test Epoch 24 ==> 	accuracy: 0.9749, 	precision: 0.9970, 	recall: 0.8798, 	specificity: 0.9993, 	f1: 0.9348
Train Epoch 25: 100%|██████████| 6507/6507 [10:33<00:00, 10.28it/s, loss=0.0018]
Train Epoch 25 ==> 	accuracy: 0.9315, 	precision: 0.9997, 	recall: 0.8633, 	specificity: 0.9998, 	f1: 0.9265
Test Epoch 25: 100%|██████████| 1768/1768 [01:02<00:00, 28.23it/s, loss=0.0451]
Test Epoch 25 ==> 	accuracy: 0.9750, 	precision: 0.9959, 	recall: 0.8813, 	specificity: 0.9991, 	f1: 0.9351
Train Epoch 26: 100%|██████████| 6507/6507 [10:31<00:00, 10.31it/s, loss=0.0009]
Train Epoch 26 ==> 	accuracy: 0.9340, 	precision: 0.9997, 	recall: 0.8682, 	specificity: 0.9998, 	f1: 0.9294
Test Epoch 26: 100%|██████████| 1768/1768 [01:04<00:00, 27.28it/s, loss=0.0717]
Test Epoch 26 ==> 	accuracy: 0.9735, 	precision: 0.9970, 	recall: 0.8730, 	specificity: 0.9993, 	f1: 0.9309
Train Epoch 27: 100%|██████████| 6507/6507 [10:22<00:00, 10.46it/s, loss=0.103]
Train Epoch 27 ==> 	accuracy: 0.9344, 	precision: 0.9998, 	recall: 0.8691, 	specificity: 0.9998, 	f1: 0.9298
Test Epoch 27: 100%|██████████| 1768/1768 [01:04<00:00, 27.21it/s, loss=0.13]
Test Epoch 27 ==> 	accuracy: 0.9738, 	precision: 0.9955, 	recall: 0.8756, 	specificity: 0.9990, 	f1: 0.9317
Train Epoch 28: 100%|██████████| 6507/6507 [10:25<00:00, 10.40it/s, loss=0.0303]
Train Epoch 28 ==> 	accuracy: 0.9344, 	precision: 0.9998, 	recall: 0.8691, 	specificity: 0.9998, 	f1: 0.9299
Test Epoch 28: 100%|██████████| 1768/1768 [01:04<00:00, 27.20it/s, loss=0.0362]
Test Epoch 28 ==> 	accuracy: 0.9771, 	precision: 0.9964, 	recall: 0.8915, 	specificity: 0.9992, 	f1: 0.9410
Train Epoch 29: 100%|██████████| 6507/6507 [10:20<00:00, 10.49it/s, loss=0.0035]
Train Epoch 29 ==> 	accuracy: 0.9338, 	precision: 0.9998, 	recall: 0.8678, 	specificity: 0.9998, 	f1: 0.9291
Test Epoch 29: 100%|██████████| 1768/1768 [01:08<00:00, 25.81it/s, loss=0.1]
Test Epoch 29 ==> 	accuracy: 0.9735, 	precision: 0.9968, 	recall: 0.8732, 	specificity: 0.9993, 	f1: 0.9309
Train Epoch 30: 100%|██████████| 6507/6507 [10:12<00:00, 10.62it/s, loss=0.0009]
Train Epoch 30 ==> 	accuracy: 0.9360, 	precision: 0.9998, 	recall: 0.8723, 	specificity: 0.9998, 	f1: 0.9317
Test Epoch 30: 100%|██████████| 1768/1768 [01:07<00:00, 26.05it/s, loss=0.0184]
Test Epoch 30 ==> 	accuracy: 0.9767, 	precision: 0.9963, 	recall: 0.8896, 	specificity: 0.9991, 	f1: 0.9399
Train Epoch 31: 100%|██████████| 6507/6507 [10:24<00:00, 10.42it/s, loss=0.0019]
Train Epoch 31 ==> 	accuracy: 0.9369, 	precision: 0.9997, 	recall: 0.8739, 	specificity: 0.9998, 	f1: 0.9326
Test Epoch 31: 100%|██████████| 1768/1768 [01:01<00:00, 28.90it/s, loss=0.45]
Test Epoch 31 ==> 	accuracy: 0.9771, 	precision: 0.9966, 	recall: 0.8909, 	specificity: 0.9992, 	f1: 0.9408
Train Epoch 32: 100%|██████████| 6507/6507 [10:14<00:00, 10.58it/s, loss=4.47]
Train Epoch 32 ==> 	accuracy: 0.9363, 	precision: 0.9998, 	recall: 0.8729, 	specificity: 0.9998, 	f1: 0.9320
Test Epoch 32: 100%|██████████| 1768/1768 [01:06<00:00, 26.65it/s, loss=0.0314]
Test Epoch 32 ==> 	accuracy: 0.9762, 	precision: 0.9977, 	recall: 0.8858, 	specificity: 0.9995, 	f1: 0.9384
Train Epoch 33: 100%|██████████| 6507/6507 [10:07<00:00, 10.71it/s, loss=0.126]
Train Epoch 33 ==> 	accuracy: 0.9384, 	precision: 0.9998, 	recall: 0.8770, 	specificity: 0.9998, 	f1: 0.9344
Test Epoch 33: 100%|██████████| 1768/1768 [01:04<00:00, 27.44it/s, loss=0.118]
Test Epoch 33 ==> 	accuracy: 0.9784, 	precision: 0.9958, 	recall: 0.8980, 	specificity: 0.9990, 	f1: 0.9444
Train Epoch 34: 100%|██████████| 6507/6507 [10:15<00:00, 10.57it/s, loss=2.6]
Train Epoch 34 ==> 	accuracy: 0.9375, 	precision: 0.9998, 	recall: 0.8753, 	specificity: 0.9998, 	f1: 0.9334
Test Epoch 34: 100%|██████████| 1768/1768 [01:03<00:00, 27.68it/s, loss=0.0495]
Test Epoch 34 ==> 	accuracy: 0.9766, 	precision: 0.9959, 	recall: 0.8891, 	specificity: 0.9991, 	f1: 0.9395
Train Epoch 35: 100%|██████████| 6507/6507 [10:23<00:00, 10.43it/s, loss=0.0015]
Train Epoch 35 ==> 	accuracy: 0.9384, 	precision: 0.9998, 	recall: 0.8771, 	specificity: 0.9998, 	f1: 0.9344
Test Epoch 35: 100%|██████████| 1768/1768 [01:06<00:00, 26.56it/s, loss=0.0594]
Test Epoch 35 ==> 	accuracy: 0.9766, 	precision: 0.9971, 	recall: 0.8884, 	specificity: 0.9993, 	f1: 0.9396
Train Epoch 36: 100%|██████████| 6507/6507 [10:15<00:00, 10.58it/s, loss=0.0005]
Train Epoch 36 ==> 	accuracy: 0.9392, 	precision: 0.9998, 	recall: 0.8786, 	specificity: 0.9998, 	f1: 0.9353
Test Epoch 36: 100%|██████████| 1768/1768 [01:04<00:00, 27.24it/s, loss=0.0215]
Test Epoch 36 ==> 	accuracy: 0.9751, 	precision: 0.9966, 	recall: 0.8814, 	specificity: 0.9992, 	f1: 0.9355
Train Epoch 37: 100%|██████████| 6507/6507 [10:18<00:00, 10.53it/s, loss=0.0037]
Train Epoch 37 ==> 	accuracy: 0.9394, 	precision: 0.9998, 	recall: 0.8791, 	specificity: 0.9998, 	f1: 0.9355
Test Epoch 37: 100%|██████████| 1768/1768 [01:02<00:00, 28.16it/s, loss=0.0659]
Test Epoch 37 ==> 	accuracy: 0.9767, 	precision: 0.9968, 	recall: 0.8890, 	specificity: 0.9993, 	f1: 0.9398
Train Epoch 38: 100%|██████████| 6507/6507 [10:26<00:00, 10.39it/s, loss=0.122]
Train Epoch 38 ==> 	accuracy: 0.9403, 	precision: 0.9998, 	recall: 0.8807, 	specificity: 0.9998, 	f1: 0.9365
Test Epoch 38: 100%|██████████| 1768/1768 [01:00<00:00, 29.11it/s, loss=0.0648]
Test Epoch 38 ==> 	accuracy: 0.9765, 	precision: 0.9939, 	recall: 0.8908, 	specificity: 0.9986, 	f1: 0.9395
Train Epoch 39: 100%|██████████| 6507/6507 [10:11<00:00, 10.65it/s, loss=0]
Train Epoch 39 ==> 	accuracy: 0.9419, 	precision: 0.9998, 	recall: 0.8841, 	specificity: 0.9998, 	f1: 0.9384
Test Epoch 39: 100%|██████████| 1768/1768 [01:02<00:00, 28.36it/s, loss=0.0206]
Test Epoch 39 ==> 	accuracy: 0.9786, 	precision: 0.9956, 	recall: 0.8991, 	specificity: 0.9990, 	f1: 0.9449
Train Epoch 40: 100%|██████████| 6507/6507 [10:27<00:00, 10.37it/s, loss=0]
Train Epoch 40 ==> 	accuracy: 0.9412, 	precision: 0.9998, 	recall: 0.8827, 	specificity: 0.9998, 	f1: 0.9376
Test Epoch 40: 100%|██████████| 1768/1768 [01:07<00:00, 26.22it/s, loss=0.512]
Test Epoch 40 ==> 	accuracy: 0.9782, 	precision: 0.9941, 	recall: 0.8986, 	specificity: 0.9986, 	f1: 0.9439
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 41: 100%|██████████| 6507/6507 [10:24<00:00, 10.42it/s, loss=0.0017]
Train Epoch 41 ==> 	accuracy: 0.9405, 	precision: 0.9998, 	recall: 0.8813, 	specificity: 0.9998, 	f1: 0.9368
Test Epoch 41: 100%|██████████| 1768/1768 [01:06<00:00, 26.48it/s, loss=0.0458]
Test Epoch 41 ==> 	accuracy: 0.9787, 	precision: 0.9951, 	recall: 0.9004, 	specificity: 0.9989, 	f1: 0.9454
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 42: 100%|██████████| 6507/6507 [10:28<00:00, 10.35it/s, loss=0.932]
Train Epoch 42 ==> 	accuracy: 0.9416, 	precision: 0.9998, 	recall: 0.8835, 	specificity: 0.9998, 	f1: 0.9380
Test Epoch 42: 100%|██████████| 1768/1768 [01:02<00:00, 28.15it/s, loss=0.6]
Test Epoch 42 ==> 	accuracy: 0.9769, 	precision: 0.9948, 	recall: 0.8918, 	specificity: 0.9988, 	f1: 0.9405
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 43: 100%|██████████| 6507/6507 [10:14<00:00, 10.60it/s, loss=2.87]
Train Epoch 43 ==> 	accuracy: 0.9413, 	precision: 0.9998, 	recall: 0.8828, 	specificity: 0.9998, 	f1: 0.9377
Test Epoch 43: 100%|██████████| 1768/1768 [01:10<00:00, 25.08it/s, loss=0.108]
Test Epoch 43 ==> 	accuracy: 0.9788, 	precision: 0.9948, 	recall: 0.9011, 	specificity: 0.9988, 	f1: 0.9457
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 44: 100%|██████████| 6507/6507 [10:21<00:00, 10.47it/s, loss=0]
Train Epoch 44 ==> 	accuracy: 0.9428, 	precision: 0.9998, 	recall: 0.8858, 	specificity: 0.9998, 	f1: 0.9393
Test Epoch 44: 100%|██████████| 1768/1768 [01:03<00:00, 27.66it/s, loss=0.199]
Test Epoch 44 ==> 	accuracy: 0.9793, 	precision: 0.9957, 	recall: 0.9027, 	specificity: 0.9990, 	f1: 0.9470
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 45: 100%|██████████| 6507/6507 [10:24<00:00, 10.42it/s, loss=0.0001]
Train Epoch 45 ==> 	accuracy: 0.9427, 	precision: 0.9998, 	recall: 0.8855, 	specificity: 0.9998, 	f1: 0.9392
Test Epoch 45: 100%|██████████| 1768/1768 [01:09<00:00, 25.61it/s, loss=0.0862]
Test Epoch 45 ==> 	accuracy: 0.9771, 	precision: 0.9965, 	recall: 0.8912, 	specificity: 0.9992, 	f1: 0.9409
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 46: 100%|██████████| 6507/6507 [10:17<00:00, 10.54it/s, loss=0.0451]
Train Epoch 46 ==> 	accuracy: 0.9446, 	precision: 0.9998, 	recall: 0.8894, 	specificity: 0.9998, 	f1: 0.9414
Test Epoch 46: 100%|██████████| 1768/1768 [01:05<00:00, 26.99it/s, loss=0.0319]
Test Epoch 46 ==> 	accuracy: 0.9806, 	precision: 0.9952, 	recall: 0.9098, 	specificity: 0.9989, 	f1: 0.9506
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 47: 100%|██████████| 6507/6507 [10:18<00:00, 10.51it/s, loss=2.8]
Train Epoch 47 ==> 	accuracy: 0.9436, 	precision: 0.9998, 	recall: 0.8873, 	specificity: 0.9998, 	f1: 0.9402
Test Epoch 47: 100%|██████████| 1768/1768 [01:06<00:00, 26.45it/s, loss=0.331]
Test Epoch 47 ==> 	accuracy: 0.9790, 	precision: 0.9935, 	recall: 0.9033, 	specificity: 0.9985, 	f1: 0.9462
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 48: 100%|██████████| 6507/6507 [10:17<00:00, 10.54it/s, loss=0.0045]
Train Epoch 48 ==> 	accuracy: 0.9460, 	precision: 0.9998, 	recall: 0.8922, 	specificity: 0.9999, 	f1: 0.9430
Test Epoch 48: 100%|██████████| 1768/1768 [01:03<00:00, 27.92it/s, loss=0.212]
Test Epoch 48 ==> 	accuracy: 0.9804, 	precision: 0.9952, 	recall: 0.9087, 	specificity: 0.9989, 	f1: 0.9499
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 49: 100%|██████████| 6507/6507 [10:26<00:00, 10.38it/s, loss=0.0271]
Train Epoch 49 ==> 	accuracy: 0.9445, 	precision: 0.9998, 	recall: 0.8892, 	specificity: 0.9998, 	f1: 0.9413
Test Epoch 49: 100%|██████████| 1768/1768 [01:05<00:00, 27.19it/s, loss=0.0302]
Test Epoch 49 ==> 	accuracy: 0.9800, 	precision: 0.9954, 	recall: 0.9067, 	specificity: 0.9989, 	f1: 0.9489
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 50: 100%|██████████| 6507/6507 [10:12<00:00, 10.62it/s, loss=0.0005]
Train Epoch 50 ==> 	accuracy: 0.9482, 	precision: 0.9998, 	recall: 0.8966, 	specificity: 0.9998, 	f1: 0.9454
Test Epoch 50: 100%|██████████| 1768/1768 [01:05<00:00, 27.18it/s, loss=0.0228]
Test Epoch 50 ==> 	accuracy: 0.9798, 	precision: 0.9932, 	recall: 0.9075, 	specificity: 0.9984, 	f1: 0.9484
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 51: 100%|██████████| 6507/6507 [10:12<00:00, 10.63it/s, loss=0.0002]
Train Epoch 51 ==> 	accuracy: 0.9469, 	precision: 0.9999, 	recall: 0.8938, 	specificity: 0.9999, 	f1: 0.9439
Test Epoch 51: 100%|██████████| 1768/1768 [01:01<00:00, 28.67it/s, loss=0.0234]
Test Epoch 51 ==> 	accuracy: 0.9795, 	precision: 0.9952, 	recall: 0.9039, 	specificity: 0.9989, 	f1: 0.9474
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 52: 100%|██████████| 6507/6507 [10:27<00:00, 10.37it/s, loss=0.0873]
Train Epoch 52 ==> 	accuracy: 0.9486, 	precision: 0.9998, 	recall: 0.8973, 	specificity: 0.9999, 	f1: 0.9458
Test Epoch 52: 100%|██████████| 1768/1768 [01:02<00:00, 28.33it/s, loss=0.0506]
Test Epoch 52 ==> 	accuracy: 0.9808, 	precision: 0.9920, 	recall: 0.9134, 	specificity: 0.9981, 	f1: 0.9511
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 53: 100%|██████████| 6507/6507 [10:12<00:00, 10.62it/s, loss=0.001]
Train Epoch 53 ==> 	accuracy: 0.9481, 	precision: 0.9998, 	recall: 0.8964, 	specificity: 0.9998, 	f1: 0.9453
Test Epoch 53: 100%|██████████| 1768/1768 [01:05<00:00, 27.16it/s, loss=0.0519]
Test Epoch 53 ==> 	accuracy: 0.9817, 	precision: 0.9929, 	recall: 0.9170, 	specificity: 0.9983, 	f1: 0.9534
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 54: 100%|██████████| 6507/6507 [10:24<00:00, 10.42it/s, loss=0.737]
Train Epoch 54 ==> 	accuracy: 0.9488, 	precision: 0.9998, 	recall: 0.8977, 	specificity: 0.9998, 	f1: 0.9460
Test Epoch 54: 100%|██████████| 1768/1768 [01:08<00:00, 25.87it/s, loss=0.113]
Test Epoch 54 ==> 	accuracy: 0.9811, 	precision: 0.9954, 	recall: 0.9118, 	specificity: 0.9989, 	f1: 0.9518
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 55: 100%|██████████| 6507/6507 [10:22<00:00, 10.45it/s, loss=0.0171]
Train Epoch 55 ==> 	accuracy: 0.9499, 	precision: 0.9998, 	recall: 0.9000, 	specificity: 0.9999, 	f1: 0.9473
Test Epoch 55: 100%|██████████| 1768/1768 [01:07<00:00, 26.21it/s, loss=0.0063]
Test Epoch 55 ==> 	accuracy: 0.9802, 	precision: 0.9964, 	recall: 0.9065, 	specificity: 0.9991, 	f1: 0.9493
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 56: 100%|██████████| 6507/6507 [10:10<00:00, 10.66it/s, loss=0.0007]
Train Epoch 56 ==> 	accuracy: 0.9491, 	precision: 0.9998, 	recall: 0.8983, 	specificity: 0.9999, 	f1: 0.9464
Test Epoch 56: 100%|██████████| 1768/1768 [01:06<00:00, 26.58it/s, loss=0.033]
Test Epoch 56 ==> 	accuracy: 0.9813, 	precision: 0.9948, 	recall: 0.9132, 	specificity: 0.9988, 	f1: 0.9523
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 57: 100%|██████████| 6507/6507 [10:19<00:00, 10.50it/s, loss=0.0029]
Train Epoch 57 ==> 	accuracy: 0.9520, 	precision: 0.9998, 	recall: 0.9042, 	specificity: 0.9999, 	f1: 0.9496
Test Epoch 57: 100%|██████████| 1768/1768 [01:04<00:00, 27.29it/s, loss=0.41]
Test Epoch 57 ==> 	accuracy: 0.9813, 	precision: 0.9917, 	recall: 0.9164, 	specificity: 0.9980, 	f1: 0.9526
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 58: 100%|██████████| 6507/6507 [10:12<00:00, 10.63it/s, loss=0.26]
Train Epoch 58 ==> 	accuracy: 0.9501, 	precision: 0.9998, 	recall: 0.9003, 	specificity: 0.9998, 	f1: 0.9475
Test Epoch 58: 100%|██████████| 1768/1768 [01:05<00:00, 27.10it/s, loss=0.0485]
Test Epoch 58 ==> 	accuracy: 0.9807, 	precision: 0.9932, 	recall: 0.9120, 	specificity: 0.9984, 	f1: 0.9508
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 59: 100%|██████████| 6507/6507 [10:09<00:00, 10.68it/s, loss=0.0009]
Train Epoch 59 ==> 	accuracy: 0.9520, 	precision: 0.9998, 	recall: 0.9041, 	specificity: 0.9998, 	f1: 0.9496
Test Epoch 59: 100%|██████████| 1768/1768 [01:05<00:00, 27.09it/s, loss=0.138]
Test Epoch 59 ==> 	accuracy: 0.9823, 	precision: 0.9902, 	recall: 0.9225, 	specificity: 0.9977, 	f1: 0.9551
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 60: 100%|██████████| 6507/6507 [10:17<00:00, 10.53it/s, loss=0.0325]
Train Epoch 60 ==> 	accuracy: 0.9521, 	precision: 0.9998, 	recall: 0.9043, 	specificity: 0.9999, 	f1: 0.9497
Test Epoch 60: 100%|██████████| 1768/1768 [01:04<00:00, 27.62it/s, loss=0.0308]
Test Epoch 60 ==> 	accuracy: 0.9820, 	precision: 0.9929, 	recall: 0.9185, 	specificity: 0.9983, 	f1: 0.9542
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 61: 100%|██████████| 6507/6507 [10:18<00:00, 10.52it/s, loss=0.0627]
Train Epoch 61 ==> 	accuracy: 0.9534, 	precision: 0.9999, 	recall: 0.9070, 	specificity: 0.9999, 	f1: 0.9512
Test Epoch 61: 100%|██████████| 1768/1768 [01:01<00:00, 28.67it/s, loss=0.0536]
Test Epoch 61 ==> 	accuracy: 0.9828, 	precision: 0.9904, 	recall: 0.9250, 	specificity: 0.9977, 	f1: 0.9566
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 62: 100%|██████████| 6507/6507 [10:10<00:00, 10.66it/s, loss=0.033]
Train Epoch 62 ==> 	accuracy: 0.9525, 	precision: 0.9999, 	recall: 0.9051, 	specificity: 0.9999, 	f1: 0.9501
Test Epoch 62: 100%|██████████| 1768/1768 [01:06<00:00, 26.57it/s, loss=0.134]
Test Epoch 62 ==> 	accuracy: 0.9820, 	precision: 0.9916, 	recall: 0.9199, 	specificity: 0.9980, 	f1: 0.9544
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 63: 100%|██████████| 6507/6507 [10:16<00:00, 10.56it/s, loss=0.0146]
Train Epoch 63 ==> 	accuracy: 0.9543, 	precision: 0.9999, 	recall: 0.9087, 	specificity: 0.9999, 	f1: 0.9521
Test Epoch 63: 100%|██████████| 1768/1768 [01:02<00:00, 28.49it/s, loss=0.21]
Test Epoch 63 ==> 	accuracy: 0.9828, 	precision: 0.9903, 	recall: 0.9249, 	specificity: 0.9977, 	f1: 0.9565
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 64: 100%|██████████| 6507/6507 [10:16<00:00, 10.55it/s, loss=0.0004]
Train Epoch 64 ==> 	accuracy: 0.9538, 	precision: 0.9999, 	recall: 0.9076, 	specificity: 0.9999, 	f1: 0.9515
Test Epoch 64: 100%|██████████| 1768/1768 [01:05<00:00, 26.84it/s, loss=0.0929]
Test Epoch 64 ==> 	accuracy: 0.9821, 	precision: 0.9909, 	recall: 0.9212, 	specificity: 0.9978, 	f1: 0.9548
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 65: 100%|██████████| 6507/6507 [10:08<00:00, 10.69it/s, loss=0.0001]
Train Epoch 65 ==> 	accuracy: 0.9546, 	precision: 0.9999, 	recall: 0.9094, 	specificity: 0.9999, 	f1: 0.9525
Test Epoch 65: 100%|██████████| 1768/1768 [01:06<00:00, 26.69it/s, loss=0.0354]
Test Epoch 65 ==> 	accuracy: 0.9825, 	precision: 0.9877, 	recall: 0.9261, 	specificity: 0.9970, 	f1: 0.9559
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 66: 100%|██████████| 6507/6507 [10:18<00:00, 10.53it/s, loss=1.39]
Train Epoch 66 ==> 	accuracy: 0.9543, 	precision: 0.9999, 	recall: 0.9088, 	specificity: 0.9999, 	f1: 0.9522
Test Epoch 66: 100%|██████████| 1768/1768 [01:06<00:00, 26.58it/s, loss=0.0259]
Test Epoch 66 ==> 	accuracy: 0.9824, 	precision: 0.9931, 	recall: 0.9204, 	specificity: 0.9983, 	f1: 0.9553
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 67: 100%|██████████| 6507/6507 [10:13<00:00, 10.60it/s, loss=1.85]
Train Epoch 67 ==> 	accuracy: 0.9549, 	precision: 0.9999, 	recall: 0.9100, 	specificity: 0.9999, 	f1: 0.9528
Test Epoch 67: 100%|██████████| 1768/1768 [01:06<00:00, 26.46it/s, loss=0.096]
Test Epoch 67 ==> 	accuracy: 0.9827, 	precision: 0.9932, 	recall: 0.9220, 	specificity: 0.9984, 	f1: 0.9563
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 68: 100%|██████████| 6507/6507 [10:11<00:00, 10.64it/s, loss=0.001]
Train Epoch 68 ==> 	accuracy: 0.9560, 	precision: 0.9999, 	recall: 0.9121, 	specificity: 0.9999, 	f1: 0.9540
Test Epoch 68: 100%|██████████| 1768/1768 [01:03<00:00, 27.85it/s, loss=0.0418]
Test Epoch 68 ==> 	accuracy: 0.9827, 	precision: 0.9895, 	recall: 0.9251, 	specificity: 0.9975, 	f1: 0.9562
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 69: 100%|██████████| 6507/6507 [10:21<00:00, 10.46it/s, loss=0.0003]
Train Epoch 69 ==> 	accuracy: 0.9555, 	precision: 0.9999, 	recall: 0.9111, 	specificity: 0.9999, 	f1: 0.9534
Test Epoch 69: 100%|██████████| 1768/1768 [01:08<00:00, 25.92it/s, loss=0.0143]
Test Epoch 69 ==> 	accuracy: 0.9832, 	precision: 0.9929, 	recall: 0.9246, 	specificity: 0.9983, 	f1: 0.9576
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 70: 100%|██████████| 6507/6507 [10:21<00:00, 10.47it/s, loss=2.56]
Train Epoch 70 ==> 	accuracy: 0.9568, 	precision: 0.9999, 	recall: 0.9136, 	specificity: 0.9999, 	f1: 0.9548
Test Epoch 70: 100%|██████████| 1768/1768 [01:05<00:00, 26.85it/s, loss=2.59]
Test Epoch 70 ==> 	accuracy: 0.9830, 	precision: 0.9862, 	recall: 0.9297, 	specificity: 0.9966, 	f1: 0.9571
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 71: 100%|██████████| 6507/6507 [10:25<00:00, 10.40it/s, loss=0.0292]
Train Epoch 71 ==> 	accuracy: 0.9562, 	precision: 0.9999, 	recall: 0.9126, 	specificity: 0.9999, 	f1: 0.9542
Test Epoch 71: 100%|██████████| 1768/1768 [01:04<00:00, 27.61it/s, loss=0.0489]
Test Epoch 71 ==> 	accuracy: 0.9826, 	precision: 0.9932, 	recall: 0.9212, 	specificity: 0.9984, 	f1: 0.9559
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 72: 100%|██████████| 6507/6507 [10:18<00:00, 10.51it/s, loss=0.0013]
Train Epoch 72 ==> 	accuracy: 0.9578, 	precision: 0.9999, 	recall: 0.9158, 	specificity: 0.9999, 	f1: 0.9560
Test Epoch 72: 100%|██████████| 1768/1768 [01:06<00:00, 26.77it/s, loss=0.0358]
Test Epoch 72 ==> 	accuracy: 0.9837, 	precision: 0.9930, 	recall: 0.9268, 	specificity: 0.9983, 	f1: 0.9588
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 73: 100%|██████████| 6507/6507 [10:11<00:00, 10.64it/s, loss=0.0002]
Train Epoch 73 ==> 	accuracy: 0.9567, 	precision: 0.9999, 	recall: 0.9135, 	specificity: 0.9999, 	f1: 0.9547
Test Epoch 73: 100%|██████████| 1768/1768 [01:09<00:00, 25.40it/s, loss=0.0862]
Test Epoch 73 ==> 	accuracy: 0.9835, 	precision: 0.9938, 	recall: 0.9250, 	specificity: 0.9985, 	f1: 0.9582
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 74: 100%|██████████| 6507/6507 [10:06<00:00, 10.72it/s, loss=0]
Train Epoch 74 ==> 	accuracy: 0.9589, 	precision: 0.9999, 	recall: 0.9179, 	specificity: 0.9999, 	f1: 0.9571
Test Epoch 74: 100%|██████████| 1768/1768 [01:05<00:00, 27.10it/s, loss=0.0998]
Test Epoch 74 ==> 	accuracy: 0.9844, 	precision: 0.9912, 	recall: 0.9317, 	specificity: 0.9979, 	f1: 0.9606
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 75: 100%|██████████| 6507/6507 [10:14<00:00, 10.58it/s, loss=0]
Train Epoch 75 ==> 	accuracy: 0.9573, 	precision: 0.9999, 	recall: 0.9148, 	specificity: 0.9999, 	f1: 0.9554
Test Epoch 75: 100%|██████████| 1768/1768 [01:06<00:00, 26.74it/s, loss=2.79]
Test Epoch 75 ==> 	accuracy: 0.9837, 	precision: 0.9920, 	recall: 0.9278, 	specificity: 0.9981, 	f1: 0.9588
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 76: 100%|██████████| 6507/6507 [10:09<00:00, 10.68it/s, loss=1.49]
Train Epoch 76 ==> 	accuracy: 0.9596, 	precision: 0.9999, 	recall: 0.9193, 	specificity: 0.9999, 	f1: 0.9579
Test Epoch 76: 100%|██████████| 1768/1768 [01:04<00:00, 27.28it/s, loss=0.0454]
Test Epoch 76 ==> 	accuracy: 0.9839, 	precision: 0.9906, 	recall: 0.9301, 	specificity: 0.9977, 	f1: 0.9594
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 77: 100%|██████████| 6507/6507 [10:15<00:00, 10.58it/s, loss=0.068]
Train Epoch 77 ==> 	accuracy: 0.9582, 	precision: 0.9999, 	recall: 0.9164, 	specificity: 0.9999, 	f1: 0.9563
Test Epoch 77: 100%|██████████| 1768/1768 [01:02<00:00, 28.21it/s, loss=0.0136]
Test Epoch 77 ==> 	accuracy: 0.9843, 	precision: 0.9898, 	recall: 0.9331, 	specificity: 0.9975, 	f1: 0.9606
Adjusting learning rate of group 0 to 3.8742e-05.

"""


'''
7mer 5mer base
'../model_save_sigBlock4_focalWithMs_deformable_7mer'

/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6507/6507 [09:51<00:00, 11.00it/s, loss=0.0987]
Train Epoch 0 ==> 	accuracy: 0.6451, 	precision: 0.9962, 	recall: 0.2913, 	specificity: 0.9989, 	f1: 0.4508
Test Epoch 0: 100%|██████████| 1768/1768 [01:05<00:00, 26.84it/s, loss=0.996]
Test Epoch 0 ==> 	accuracy: 0.9181, 	precision: 0.9665, 	recall: 0.6212, 	specificity: 0.9945, 	f1: 0.7563
Train Epoch 1: 100%|██████████| 6507/6507 [10:08<00:00, 10.70it/s, loss=0.0773]
Train Epoch 1 ==> 	accuracy: 0.7484, 	precision: 0.9976, 	recall: 0.4980, 	specificity: 0.9988, 	f1: 0.6644
Test Epoch 1: 100%|██████████| 1768/1768 [01:03<00:00, 27.64it/s, loss=0.558]
Test Epoch 1 ==> 	accuracy: 0.9139, 	precision: 0.9820, 	recall: 0.5901, 	specificity: 0.9972, 	f1: 0.7372
Train Epoch 2: 100%|██████████| 6507/6507 [10:04<00:00, 10.76it/s, loss=0.14]
Train Epoch 2 ==> 	accuracy: 0.7771, 	precision: 0.9981, 	recall: 0.5552, 	specificity: 0.9990, 	f1: 0.7135
Test Epoch 2: 100%|██████████| 1768/1768 [01:06<00:00, 26.47it/s, loss=0.533]
Test Epoch 2 ==> 	accuracy: 0.9208, 	precision: 0.9759, 	recall: 0.6283, 	specificity: 0.9960, 	f1: 0.7644
Train Epoch 3: 100%|██████████| 6507/6507 [10:02<00:00, 10.80it/s, loss=0.0371]
Train Epoch 3 ==> 	accuracy: 0.7895, 	precision: 0.9984, 	recall: 0.5800, 	specificity: 0.9991, 	f1: 0.7337
Test Epoch 3: 100%|██████████| 1768/1768 [01:03<00:00, 27.87it/s, loss=0.278]
Test Epoch 3 ==> 	accuracy: 0.9246, 	precision: 0.9862, 	recall: 0.6404, 	specificity: 0.9977, 	f1: 0.7766
Train Epoch 4: 100%|██████████| 6507/6507 [10:13<00:00, 10.60it/s, loss=0.0551]
Train Epoch 4 ==> 	accuracy: 0.8035, 	precision: 0.9985, 	recall: 0.6079, 	specificity: 0.9991, 	f1: 0.7557
Test Epoch 4: 100%|██████████| 1768/1768 [01:00<00:00, 29.42it/s, loss=0.126]
Test Epoch 4 ==> 	accuracy: 0.9331, 	precision: 0.9810, 	recall: 0.6862, 	specificity: 0.9966, 	f1: 0.8075
Train Epoch 5: 100%|██████████| 6507/6507 [10:03<00:00, 10.78it/s, loss=0.089]
Train Epoch 5 ==> 	accuracy: 0.8069, 	precision: 0.9986, 	recall: 0.6146, 	specificity: 0.9992, 	f1: 0.7609
Test Epoch 5: 100%|██████████| 1768/1768 [01:03<00:00, 27.65it/s, loss=0.489]
Test Epoch 5 ==> 	accuracy: 0.9277, 	precision: 0.9801, 	recall: 0.6600, 	specificity: 0.9966, 	f1: 0.7888
Train Epoch 6: 100%|██████████| 6507/6507 [10:07<00:00, 10.71it/s, loss=0.0849]
Train Epoch 6 ==> 	accuracy: 0.8091, 	precision: 0.9988, 	recall: 0.6190, 	specificity: 0.9992, 	f1: 0.7643
Test Epoch 6: 100%|██████████| 1768/1768 [01:05<00:00, 27.14it/s, loss=0.313]
Test Epoch 6 ==> 	accuracy: 0.9299, 	precision: 0.9861, 	recall: 0.6668, 	specificity: 0.9976, 	f1: 0.7956
Train Epoch 7: 100%|██████████| 6507/6507 [10:02<00:00, 10.80it/s, loss=0.0862]
Train Epoch 7 ==> 	accuracy: 0.8200, 	precision: 0.9989, 	recall: 0.6408, 	specificity: 0.9993, 	f1: 0.7807
Test Epoch 7: 100%|██████████| 1768/1768 [01:02<00:00, 28.26it/s, loss=0.295]
Test Epoch 7 ==> 	accuracy: 0.9351, 	precision: 0.9911, 	recall: 0.6888, 	specificity: 0.9984, 	f1: 0.8127
Train Epoch 8: 100%|██████████| 6507/6507 [09:59<00:00, 10.85it/s, loss=0.102]
Train Epoch 8 ==> 	accuracy: 0.8192, 	precision: 0.9989, 	recall: 0.6390, 	specificity: 0.9993, 	f1: 0.7794
Test Epoch 8: 100%|██████████| 1768/1768 [01:03<00:00, 28.03it/s, loss=0.182]
Test Epoch 8 ==> 	accuracy: 0.9365, 	precision: 0.9803, 	recall: 0.7036, 	specificity: 0.9964, 	f1: 0.8192
Train Epoch 9: 100%|██████████| 6507/6507 [10:09<00:00, 10.68it/s, loss=0.102]
Train Epoch 9 ==> 	accuracy: 0.8243, 	precision: 0.9989, 	recall: 0.6493, 	specificity: 0.9993, 	f1: 0.7871
Test Epoch 9: 100%|██████████| 1768/1768 [01:03<00:00, 27.86it/s, loss=0.18]
Test Epoch 9 ==> 	accuracy: 0.9298, 	precision: 0.9849, 	recall: 0.6669, 	specificity: 0.9974, 	f1: 0.7953
Train Epoch 10: 100%|██████████| 6507/6507 [10:00<00:00, 10.83it/s, loss=0.117]
Train Epoch 10 ==> 	accuracy: 0.8280, 	precision: 0.9990, 	recall: 0.6566, 	specificity: 0.9993, 	f1: 0.7924
Test Epoch 10: 100%|██████████| 1768/1768 [01:03<00:00, 28.05it/s, loss=0.278]
Test Epoch 10 ==> 	accuracy: 0.9343, 	precision: 0.9779, 	recall: 0.6945, 	specificity: 0.9960, 	f1: 0.8122
Train Epoch 11: 100%|██████████| 6507/6507 [09:59<00:00, 10.85it/s, loss=0.0584]
Train Epoch 11 ==> 	accuracy: 0.8325, 	precision: 0.9990, 	recall: 0.6656, 	specificity: 0.9994, 	f1: 0.7989
Test Epoch 11: 100%|██████████| 1768/1768 [01:05<00:00, 26.95it/s, loss=0.185]
Test Epoch 11 ==> 	accuracy: 0.9390, 	precision: 0.9881, 	recall: 0.7104, 	specificity: 0.9978, 	f1: 0.8266
Train Epoch 12: 100%|██████████| 6507/6507 [10:08<00:00, 10.69it/s, loss=0.0801]
Train Epoch 12 ==> 	accuracy: 0.8314, 	precision: 0.9991, 	recall: 0.6633, 	specificity: 0.9994, 	f1: 0.7973
Test Epoch 12: 100%|██████████| 1768/1768 [01:06<00:00, 26.45it/s, loss=0.144]
Test Epoch 12 ==> 	accuracy: 0.9399, 	precision: 0.9855, 	recall: 0.7169, 	specificity: 0.9973, 	f1: 0.8300
Train Epoch 13: 100%|██████████| 6507/6507 [10:02<00:00, 10.80it/s, loss=0.124]
Train Epoch 13 ==> 	accuracy: 0.8367, 	precision: 0.9991, 	recall: 0.6740, 	specificity: 0.9994, 	f1: 0.8049
Test Epoch 13: 100%|██████████| 1768/1768 [01:08<00:00, 25.76it/s, loss=0.163]
Test Epoch 13 ==> 	accuracy: 0.9399, 	precision: 0.9906, 	recall: 0.7127, 	specificity: 0.9983, 	f1: 0.8290
Train Epoch 14: 100%|██████████| 6507/6507 [10:02<00:00, 10.80it/s, loss=0.175]
Train Epoch 14 ==> 	accuracy: 0.8418, 	precision: 0.9991, 	recall: 0.6842, 	specificity: 0.9994, 	f1: 0.8122
Test Epoch 14: 100%|██████████| 1768/1768 [01:02<00:00, 28.31it/s, loss=0.72]
Test Epoch 14 ==> 	accuracy: 0.9416, 	precision: 0.9747, 	recall: 0.7336, 	specificity: 0.9951, 	f1: 0.8371
Train Epoch 15: 100%|██████████| 6507/6507 [10:02<00:00, 10.80it/s, loss=0.0789]
Train Epoch 15 ==> 	accuracy: 0.8395, 	precision: 0.9992, 	recall: 0.6796, 	specificity: 0.9994, 	f1: 0.8089
Test Epoch 15: 100%|██████████| 1768/1768 [01:04<00:00, 27.48it/s, loss=0.207]
Test Epoch 15 ==> 	accuracy: 0.9353, 	precision: 0.9880, 	recall: 0.6920, 	specificity: 0.9978, 	f1: 0.8139
Train Epoch 16: 100%|██████████| 6507/6507 [10:13<00:00, 10.61it/s, loss=0.0332]
Train Epoch 16 ==> 	accuracy: 0.8405, 	precision: 0.9992, 	recall: 0.6816, 	specificity: 0.9994, 	f1: 0.8104
Test Epoch 16: 100%|██████████| 1768/1768 [01:01<00:00, 28.57it/s, loss=0.257]
Test Epoch 16 ==> 	accuracy: 0.9400, 	precision: 0.9892, 	recall: 0.7143, 	specificity: 0.9980, 	f1: 0.8296
Train Epoch 17: 100%|██████████| 6507/6507 [10:18<00:00, 10.52it/s, loss=0.0174]
Train Epoch 17 ==> 	accuracy: 0.8425, 	precision: 0.9992, 	recall: 0.6856, 	specificity: 0.9995, 	f1: 0.8132
Test Epoch 17: 100%|██████████| 1768/1768 [01:06<00:00, 26.69it/s, loss=0.141]
Test Epoch 17 ==> 	accuracy: 0.9439, 	precision: 0.9898, 	recall: 0.7331, 	specificity: 0.9981, 	f1: 0.8424
Train Epoch 18: 100%|██████████| 6507/6507 [10:17<00:00, 10.53it/s, loss=0.0822]
Train Epoch 18 ==> 	accuracy: 0.8489, 	precision: 0.9992, 	recall: 0.6984, 	specificity: 0.9994, 	f1: 0.8221
Test Epoch 18: 100%|██████████| 1768/1768 [01:07<00:00, 26.26it/s, loss=0.412]
Test Epoch 18 ==> 	accuracy: 0.9451, 	precision: 0.9916, 	recall: 0.7381, 	specificity: 0.9984, 	f1: 0.8463
Train Epoch 19: 100%|██████████| 6507/6507 [10:13<00:00, 10.61it/s, loss=0.274]
Train Epoch 19 ==> 	accuracy: 0.8465, 	precision: 0.9992, 	recall: 0.6935, 	specificity: 0.9995, 	f1: 0.8187
Test Epoch 19: 100%|██████████| 1768/1768 [01:06<00:00, 26.64it/s, loss=0.153]
Test Epoch 19 ==> 	accuracy: 0.9449, 	precision: 0.9898, 	recall: 0.7384, 	specificity: 0.9980, 	f1: 0.8458
Train Epoch 20: 100%|██████████| 6507/6507 [10:15<00:00, 10.57it/s, loss=0.179]
Train Epoch 20 ==> 	accuracy: 0.8494, 	precision: 0.9992, 	recall: 0.6994, 	specificity: 0.9995, 	f1: 0.8228
Test Epoch 20: 100%|██████████| 1768/1768 [01:05<00:00, 27.09it/s, loss=1.06]
Test Epoch 20 ==> 	accuracy: 0.9459, 	precision: 0.9866, 	recall: 0.7458, 	specificity: 0.9974, 	f1: 0.8495
Train Epoch 21: 100%|██████████| 6507/6507 [10:15<00:00, 10.57it/s, loss=0.122]
Train Epoch 21 ==> 	accuracy: 0.8524, 	precision: 0.9993, 	recall: 0.7053, 	specificity: 0.9995, 	f1: 0.8269
Test Epoch 21: 100%|██████████| 1768/1768 [01:03<00:00, 27.76it/s, loss=0.695]
Test Epoch 21 ==> 	accuracy: 0.9431, 	precision: 0.9844, 	recall: 0.7336, 	specificity: 0.9970, 	f1: 0.8407
Train Epoch 22: 100%|██████████| 6507/6507 [10:09<00:00, 10.68it/s, loss=0.0997]
Train Epoch 22 ==> 	accuracy: 0.8507, 	precision: 0.9993, 	recall: 0.7020, 	specificity: 0.9995, 	f1: 0.8246
Test Epoch 22: 100%|██████████| 1768/1768 [01:07<00:00, 26.32it/s, loss=0.15]
Test Epoch 22 ==> 	accuracy: 0.9434, 	precision: 0.9832, 	recall: 0.7359, 	specificity: 0.9968, 	f1: 0.8418
Train Epoch 23: 100%|██████████| 6507/6507 [10:18<00:00, 10.53it/s, loss=0.101]
Train Epoch 23 ==> 	accuracy: 0.8543, 	precision: 0.9993, 	recall: 0.7090, 	specificity: 0.9995, 	f1: 0.8295
Test Epoch 23: 100%|██████████| 1768/1768 [01:04<00:00, 27.44it/s, loss=0.608]
Test Epoch 23 ==> 	accuracy: 0.9445, 	precision: 0.9833, 	recall: 0.7412, 	specificity: 0.9968, 	f1: 0.8452
Train Epoch 24: 100%|██████████| 6507/6507 [10:18<00:00, 10.52it/s, loss=0.0697]
Train Epoch 24 ==> 	accuracy: 0.8570, 	precision: 0.9993, 	recall: 0.7145, 	specificity: 0.9995, 	f1: 0.8332
Test Epoch 24: 100%|██████████| 1768/1768 [01:02<00:00, 28.17it/s, loss=0.153]
Test Epoch 24 ==> 	accuracy: 0.9459, 	precision: 0.9869, 	recall: 0.7454, 	specificity: 0.9974, 	f1: 0.8493
Train Epoch 25: 100%|██████████| 6507/6507 [10:16<00:00, 10.56it/s, loss=0.0855]
Train Epoch 25 ==> 	accuracy: 0.8556, 	precision: 0.9993, 	recall: 0.7117, 	specificity: 0.9995, 	f1: 0.8313
Test Epoch 25: 100%|██████████| 1768/1768 [01:05<00:00, 27.09it/s, loss=0.24]
Test Epoch 25 ==> 	accuracy: 0.9468, 	precision: 0.9864, 	recall: 0.7501, 	specificity: 0.9973, 	f1: 0.8522
Train Epoch 26: 100%|██████████| 6507/6507 [10:19<00:00, 10.51it/s, loss=0.113]
Train Epoch 26 ==> 	accuracy: 0.8574, 	precision: 0.9993, 	recall: 0.7153, 	specificity: 0.9995, 	f1: 0.8338
Test Epoch 26: 100%|██████████| 1768/1768 [01:05<00:00, 26.84it/s, loss=0.152]
Test Epoch 26 ==> 	accuracy: 0.9476, 	precision: 0.9878, 	recall: 0.7532, 	specificity: 0.9976, 	f1: 0.8547
Train Epoch 27: 100%|██████████| 6507/6507 [10:00<00:00, 10.84it/s, loss=0.0668]
Train Epoch 27 ==> 	accuracy: 0.8604, 	precision: 0.9993, 	recall: 0.7212, 	specificity: 0.9995, 	f1: 0.8378
Test Epoch 27: 100%|██████████| 1768/1768 [01:04<00:00, 27.56it/s, loss=0.18]
Test Epoch 27 ==> 	accuracy: 0.9469, 	precision: 0.9919, 	recall: 0.7465, 	specificity: 0.9984, 	f1: 0.8519
Train Epoch 28: 100%|██████████| 6507/6507 [10:02<00:00, 10.80it/s, loss=0.0931]
Train Epoch 28 ==> 	accuracy: 0.8585, 	precision: 0.9994, 	recall: 0.7175, 	specificity: 0.9995, 	f1: 0.8353
Test Epoch 28: 100%|██████████| 1768/1768 [01:00<00:00, 29.26it/s, loss=0.203]
Test Epoch 28 ==> 	accuracy: 0.9503, 	precision: 0.9840, 	recall: 0.7697, 	specificity: 0.9968, 	f1: 0.8637
Train Epoch 29: 100%|██████████| 6507/6507 [10:06<00:00, 10.73it/s, loss=0.0555]
Train Epoch 29 ==> 	accuracy: 0.8586, 	precision: 0.9994, 	recall: 0.7177, 	specificity: 0.9996, 	f1: 0.8354
Test Epoch 29: 100%|██████████| 1768/1768 [01:07<00:00, 26.30it/s, loss=0.216]
Test Epoch 29 ==> 	accuracy: 0.9457, 	precision: 0.9909, 	recall: 0.7411, 	specificity: 0.9983, 	f1: 0.8480
Train Epoch 30: 100%|██████████| 6507/6507 [10:09<00:00, 10.68it/s, loss=0.098]
Train Epoch 30 ==> 	accuracy: 0.8609, 	precision: 0.9994, 	recall: 0.7223, 	specificity: 0.9995, 	f1: 0.8385
Test Epoch 30: 100%|██████████| 1768/1768 [01:02<00:00, 28.33it/s, loss=0.166]
Test Epoch 30 ==> 	accuracy: 0.9424, 	precision: 0.9834, 	recall: 0.7306, 	specificity: 0.9968, 	f1: 0.8384
Train Epoch 31: 100%|██████████| 6507/6507 [10:03<00:00, 10.77it/s, loss=0.0148]
Train Epoch 31 ==> 	accuracy: 0.8630, 	precision: 0.9994, 	recall: 0.7264, 	specificity: 0.9996, 	f1: 0.8413
Test Epoch 31: 100%|██████████| 1768/1768 [01:01<00:00, 28.74it/s, loss=0.288]
Test Epoch 31 ==> 	accuracy: 0.9515, 	precision: 0.9834, 	recall: 0.7760, 	specificity: 0.9966, 	f1: 0.8675
Train Epoch 32: 100%|██████████| 6507/6507 [10:10<00:00, 10.66it/s, loss=0.0585]
Train Epoch 32 ==> 	accuracy: 0.8618, 	precision: 0.9994, 	recall: 0.7239, 	specificity: 0.9996, 	f1: 0.8397
Test Epoch 32: 100%|██████████| 1768/1768 [01:09<00:00, 25.28it/s, loss=0.141]
Test Epoch 32 ==> 	accuracy: 0.9484, 	precision: 0.9888, 	recall: 0.7561, 	specificity: 0.9978, 	f1: 0.8570
Train Epoch 33: 100%|██████████| 6507/6507 [10:03<00:00, 10.79it/s, loss=0.101]
Train Epoch 33 ==> 	accuracy: 0.8633, 	precision: 0.9995, 	recall: 0.7270, 	specificity: 0.9996, 	f1: 0.8418
Test Epoch 33: 100%|██████████| 1768/1768 [01:05<00:00, 27.02it/s, loss=0.12]
Test Epoch 33 ==> 	accuracy: 0.9481, 	precision: 0.9839, 	recall: 0.7589, 	specificity: 0.9968, 	f1: 0.8569
Train Epoch 34: 100%|██████████| 6507/6507 [10:14<00:00, 10.59it/s, loss=0.0657]
Train Epoch 34 ==> 	accuracy: 0.8652, 	precision: 0.9994, 	recall: 0.7308, 	specificity: 0.9996, 	f1: 0.8443
Test Epoch 34: 100%|██████████| 1768/1768 [01:04<00:00, 27.36it/s, loss=0.203]
Test Epoch 34 ==> 	accuracy: 0.9468, 	precision: 0.9875, 	recall: 0.7494, 	specificity: 0.9976, 	f1: 0.8521
Train Epoch 35: 100%|██████████| 6507/6507 [10:01<00:00, 10.81it/s, loss=0.116]
Train Epoch 35 ==> 	accuracy: 0.8673, 	precision: 0.9994, 	recall: 0.7349, 	specificity: 0.9996, 	f1: 0.8470
Test Epoch 35: 100%|██████████| 1768/1768 [01:03<00:00, 27.87it/s, loss=0.229]
Test Epoch 35 ==> 	accuracy: 0.9404, 	precision: 0.9869, 	recall: 0.7180, 	specificity: 0.9975, 	f1: 0.8312
Train Epoch 36: 100%|██████████| 6507/6507 [10:09<00:00, 10.67it/s, loss=0.0845]
Train Epoch 36 ==> 	accuracy: 0.8666, 	precision: 0.9994, 	recall: 0.7337, 	specificity: 0.9996, 	f1: 0.8462
Test Epoch 36: 100%|██████████| 1768/1768 [01:04<00:00, 27.53it/s, loss=0.351]
Test Epoch 36 ==> 	accuracy: 0.9512, 	precision: 0.9838, 	recall: 0.7744, 	specificity: 0.9967, 	f1: 0.8666
Train Epoch 37: 100%|██████████| 6507/6507 [10:12<00:00, 10.63it/s, loss=0.189]
Train Epoch 37 ==> 	accuracy: 0.8693, 	precision: 0.9994, 	recall: 0.7390, 	specificity: 0.9996, 	f1: 0.8497
Test Epoch 37: 100%|██████████| 1768/1768 [01:02<00:00, 28.50it/s, loss=1.46]
Test Epoch 37 ==> 	accuracy: 0.9506, 	precision: 0.9778, 	recall: 0.7759, 	specificity: 0.9955, 	f1: 0.8652
Train Epoch 38: 100%|██████████| 6507/6507 [10:04<00:00, 10.76it/s, loss=0.0515]
Train Epoch 38 ==> 	accuracy: 0.8686, 	precision: 0.9995, 	recall: 0.7377, 	specificity: 0.9996, 	f1: 0.8488
Test Epoch 38: 100%|██████████| 1768/1768 [01:02<00:00, 28.34it/s, loss=0.283]
Test Epoch 38 ==> 	accuracy: 0.9517, 	precision: 0.9859, 	recall: 0.7751, 	specificity: 0.9971, 	f1: 0.8679
Train Epoch 39: 100%|██████████| 6507/6507 [09:57<00:00, 10.89it/s, loss=0.0944]
Train Epoch 39 ==> 	accuracy: 0.8716, 	precision: 0.9995, 	recall: 0.7436, 	specificity: 0.9996, 	f1: 0.8527
Test Epoch 39: 100%|██████████| 1768/1768 [01:04<00:00, 27.57it/s, loss=0.61]
Test Epoch 39 ==> 	accuracy: 0.9542, 	precision: 0.9678, 	recall: 0.8029, 	specificity: 0.9931, 	f1: 0.8776
Train Epoch 40: 100%|██████████| 6507/6507 [09:38<00:00, 11.26it/s, loss=0.0085]
Train Epoch 40 ==> 	accuracy: 0.8706, 	precision: 0.9995, 	recall: 0.7416, 	specificity: 0.9996, 	f1: 0.8514
Test Epoch 40: 100%|██████████| 1768/1768 [01:01<00:00, 28.64it/s, loss=0.13]
Test Epoch 40 ==> 	accuracy: 0.9497, 	precision: 0.9792, 	recall: 0.7704, 	specificity: 0.9958, 	f1: 0.8624
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 41: 100%|██████████| 6507/6507 [09:43<00:00, 11.15it/s, loss=0.0226]
Train Epoch 41 ==> 	accuracy: 0.8701, 	precision: 0.9994, 	recall: 0.7407, 	specificity: 0.9996, 	f1: 0.8508
Test Epoch 41: 100%|██████████| 1768/1768 [01:01<00:00, 28.65it/s, loss=0.355]
Test Epoch 41 ==> 	accuracy: 0.9512, 	precision: 0.9806, 	recall: 0.7768, 	specificity: 0.9961, 	f1: 0.8669
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 42: 100%|██████████| 6507/6507 [09:38<00:00, 11.24it/s, loss=0.0312]
Train Epoch 42 ==> 	accuracy: 0.8720, 	precision: 0.9995, 	recall: 0.7445, 	specificity: 0.9996, 	f1: 0.8533
Test Epoch 42: 100%|██████████| 1768/1768 [01:04<00:00, 27.49it/s, loss=0.0722]
Test Epoch 42 ==> 	accuracy: 0.9526, 	precision: 0.9735, 	recall: 0.7900, 	specificity: 0.9945, 	f1: 0.8722
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 43: 100%|██████████| 6507/6507 [09:49<00:00, 11.04it/s, loss=0.0199]
Train Epoch 43 ==> 	accuracy: 0.8725, 	precision: 0.9994, 	recall: 0.7455, 	specificity: 0.9995, 	f1: 0.8540
Test Epoch 43: 100%|██████████| 1768/1768 [01:03<00:00, 27.70it/s, loss=0.0915]
Test Epoch 43 ==> 	accuracy: 0.9549, 	precision: 0.9840, 	recall: 0.7923, 	specificity: 0.9967, 	f1: 0.8778
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 44: 100%|██████████| 6507/6507 [09:48<00:00, 11.05it/s, loss=0.0779]
Train Epoch 44 ==> 	accuracy: 0.8766, 	precision: 0.9995, 	recall: 0.7536, 	specificity: 0.9996, 	f1: 0.8593
Test Epoch 44: 100%|██████████| 1768/1768 [01:03<00:00, 27.83it/s, loss=0.107]
Test Epoch 44 ==> 	accuracy: 0.9518, 	precision: 0.9886, 	recall: 0.7733, 	specificity: 0.9977, 	f1: 0.8678
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 45: 100%|██████████| 6507/6507 [09:45<00:00, 11.12it/s, loss=0.0013]
Train Epoch 45 ==> 	accuracy: 0.8718, 	precision: 0.9995, 	recall: 0.7440, 	specificity: 0.9996, 	f1: 0.8530
Test Epoch 45: 100%|██████████| 1768/1768 [01:03<00:00, 27.76it/s, loss=0.208]
Test Epoch 45 ==> 	accuracy: 0.9522, 	precision: 0.9863, 	recall: 0.7772, 	specificity: 0.9972, 	f1: 0.8694
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 46: 100%|██████████| 6507/6507 [09:46<00:00, 11.10it/s, loss=0.0053]
Train Epoch 46 ==> 	accuracy: 0.8758, 	precision: 0.9995, 	recall: 0.7519, 	specificity: 0.9996, 	f1: 0.8582
Test Epoch 46: 100%|██████████| 1768/1768 [01:01<00:00, 28.82it/s, loss=0.259]
Test Epoch 46 ==> 	accuracy: 0.9537, 	precision: 0.9869, 	recall: 0.7840, 	specificity: 0.9973, 	f1: 0.8738
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 47: 100%|██████████| 6507/6507 [09:53<00:00, 10.97it/s, loss=0.0189]
Train Epoch 47 ==> 	accuracy: 0.8760, 	precision: 0.9995, 	recall: 0.7524, 	specificity: 0.9996, 	f1: 0.8585
Test Epoch 47: 100%|██████████| 1768/1768 [01:03<00:00, 27.63it/s, loss=0.117]
Test Epoch 47 ==> 	accuracy: 0.9563, 	precision: 0.9813, 	recall: 0.8014, 	specificity: 0.9961, 	f1: 0.8823
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 48: 100%|██████████| 6507/6507 [09:40<00:00, 11.21it/s, loss=0.0086]
Train Epoch 48 ==> 	accuracy: 0.8780, 	precision: 0.9996, 	recall: 0.7564, 	specificity: 0.9997, 	f1: 0.8612
Test Epoch 48: 100%|██████████| 1768/1768 [01:01<00:00, 28.67it/s, loss=0.197]
Test Epoch 48 ==> 	accuracy: 0.9500, 	precision: 0.9871, 	recall: 0.7656, 	specificity: 0.9974, 	f1: 0.8624
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 49: 100%|██████████| 6507/6507 [09:33<00:00, 11.35it/s, loss=0.0087]
Train Epoch 49 ==> 	accuracy: 0.8773, 	precision: 0.9995, 	recall: 0.7550, 	specificity: 0.9996, 	f1: 0.8602
Test Epoch 49: 100%|██████████| 1768/1768 [01:03<00:00, 27.89it/s, loss=0.139]
Test Epoch 49 ==> 	accuracy: 0.9544, 	precision: 0.9878, 	recall: 0.7869, 	specificity: 0.9975, 	f1: 0.8759
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 50: 100%|██████████| 6507/6507 [09:40<00:00, 11.21it/s, loss=0.0014]
Train Epoch 50 ==> 	accuracy: 0.8813, 	precision: 0.9995, 	recall: 0.7629, 	specificity: 0.9996, 	f1: 0.8653
Test Epoch 50: 100%|██████████| 1768/1768 [01:03<00:00, 27.70it/s, loss=0.187]
Test Epoch 50 ==> 	accuracy: 0.9562, 	precision: 0.9834, 	recall: 0.7993, 	specificity: 0.9965, 	f1: 0.8819
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 51: 100%|██████████| 6507/6507 [09:36<00:00, 11.29it/s, loss=0.0015]
Train Epoch 51 ==> 	accuracy: 0.8805, 	precision: 0.9995, 	recall: 0.7613, 	specificity: 0.9996, 	f1: 0.8643
Test Epoch 51: 100%|██████████| 1768/1768 [01:01<00:00, 28.64it/s, loss=0.177]
Test Epoch 51 ==> 	accuracy: 0.9531, 	precision: 0.9895, 	recall: 0.7789, 	specificity: 0.9979, 	f1: 0.8717
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 52: 100%|██████████| 6507/6507 [09:40<00:00, 11.21it/s, loss=0.0261]
Train Epoch 52 ==> 	accuracy: 0.8829, 	precision: 0.9996, 	recall: 0.7661, 	specificity: 0.9997, 	f1: 0.8674
Test Epoch 52: 100%|██████████| 1768/1768 [01:02<00:00, 28.18it/s, loss=0.109]
Test Epoch 52 ==> 	accuracy: 0.9559, 	precision: 0.9861, 	recall: 0.7957, 	specificity: 0.9971, 	f1: 0.8807
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 53: 100%|██████████| 6507/6507 [09:43<00:00, 11.15it/s, loss=0.0056]
Train Epoch 53 ==> 	accuracy: 0.8847, 	precision: 0.9996, 	recall: 0.7697, 	specificity: 0.9997, 	f1: 0.8697
Test Epoch 53: 100%|██████████| 1768/1768 [01:05<00:00, 26.87it/s, loss=0.16]
Test Epoch 53 ==> 	accuracy: 0.9571, 	precision: 0.9824, 	recall: 0.8049, 	specificity: 0.9963, 	f1: 0.8848
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 54: 100%|██████████| 6507/6507 [09:41<00:00, 11.19it/s, loss=0.021]
Train Epoch 54 ==> 	accuracy: 0.8847, 	precision: 0.9996, 	recall: 0.7698, 	specificity: 0.9997, 	f1: 0.8697
Test Epoch 54: 100%|██████████| 1768/1768 [01:03<00:00, 27.99it/s, loss=0.122]
Test Epoch 54 ==> 	accuracy: 0.9586, 	precision: 0.9827, 	recall: 0.8119, 	specificity: 0.9963, 	f1: 0.8891
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 55: 100%|██████████| 6507/6507 [09:31<00:00, 11.38it/s, loss=0.0027]
Train Epoch 55 ==> 	accuracy: 0.8863, 	precision: 0.9996, 	recall: 0.7729, 	specificity: 0.9997, 	f1: 0.8717
Test Epoch 55: 100%|██████████| 1768/1768 [01:00<00:00, 29.12it/s, loss=0.208]
Test Epoch 55 ==> 	accuracy: 0.9575, 	precision: 0.9838, 	recall: 0.8053, 	specificity: 0.9966, 	f1: 0.8857
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 56: 100%|██████████| 6507/6507 [09:31<00:00, 11.39it/s, loss=0.0039]
Train Epoch 56 ==> 	accuracy: 0.8854, 	precision: 0.9996, 	recall: 0.7712, 	specificity: 0.9997, 	f1: 0.8707
Test Epoch 56: 100%|██████████| 1768/1768 [01:01<00:00, 28.54it/s, loss=0.171]
Test Epoch 56 ==> 	accuracy: 0.9570, 	precision: 0.9821, 	recall: 0.8044, 	specificity: 0.9962, 	f1: 0.8844
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 57: 100%|██████████| 6507/6507 [09:37<00:00, 11.26it/s, loss=0.165]
Train Epoch 57 ==> 	accuracy: 0.8904, 	precision: 0.9996, 	recall: 0.7811, 	specificity: 0.9997, 	f1: 0.8770
Test Epoch 57: 100%|██████████| 1768/1768 [01:00<00:00, 29.11it/s, loss=0.84]
Test Epoch 57 ==> 	accuracy: 0.9575, 	precision: 0.9817, 	recall: 0.8075, 	specificity: 0.9961, 	f1: 0.8861
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 58: 100%|██████████| 6507/6507 [09:35<00:00, 11.30it/s, loss=0.0665]
Train Epoch 58 ==> 	accuracy: 0.8876, 	precision: 0.9996, 	recall: 0.7755, 	specificity: 0.9997, 	f1: 0.8734
Test Epoch 58: 100%|██████████| 1768/1768 [01:07<00:00, 26.13it/s, loss=2.91]
Test Epoch 58 ==> 	accuracy: 0.9579, 	precision: 0.9865, 	recall: 0.8053, 	specificity: 0.9972, 	f1: 0.8867
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 59: 100%|██████████| 6507/6507 [09:40<00:00, 11.21it/s, loss=0.0641]
Train Epoch 59 ==> 	accuracy: 0.8908, 	precision: 0.9996, 	recall: 0.7818, 	specificity: 0.9997, 	f1: 0.8774
Test Epoch 59: 100%|██████████| 1768/1768 [01:04<00:00, 27.28it/s, loss=0.278]
Test Epoch 59 ==> 	accuracy: 0.9569, 	precision: 0.9836, 	recall: 0.8026, 	specificity: 0.9966, 	f1: 0.8839
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 60: 100%|██████████| 6507/6507 [09:34<00:00, 11.33it/s, loss=0.0183]
Train Epoch 60 ==> 	accuracy: 0.8908, 	precision: 0.9996, 	recall: 0.7818, 	specificity: 0.9997, 	f1: 0.8774
Test Epoch 60: 100%|██████████| 1768/1768 [01:03<00:00, 28.06it/s, loss=0.114]
Test Epoch 60 ==> 	accuracy: 0.9588, 	precision: 0.9823, 	recall: 0.8132, 	specificity: 0.9962, 	f1: 0.8898
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 61: 100%|██████████| 6507/6507 [09:38<00:00, 11.25it/s, loss=0.124]
Train Epoch 61 ==> 	accuracy: 0.8941, 	precision: 0.9996, 	recall: 0.7884, 	specificity: 0.9997, 	f1: 0.8816
Test Epoch 61: 100%|██████████| 1768/1768 [01:01<00:00, 28.65it/s, loss=0.238]
Test Epoch 61 ==> 	accuracy: 0.9603, 	precision: 0.9795, 	recall: 0.8232, 	specificity: 0.9956, 	f1: 0.8946
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 62: 100%|██████████| 6507/6507 [09:43<00:00, 11.16it/s, loss=0.0483]
Train Epoch 62 ==> 	accuracy: 0.8898, 	precision: 0.9996, 	recall: 0.7799, 	specificity: 0.9997, 	f1: 0.8762
Test Epoch 62: 100%|██████████| 1768/1768 [01:06<00:00, 26.57it/s, loss=1.06]
Test Epoch 62 ==> 	accuracy: 0.9582, 	precision: 0.9843, 	recall: 0.8083, 	specificity: 0.9967, 	f1: 0.8877
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 63: 100%|██████████| 6507/6507 [09:42<00:00, 11.17it/s, loss=0.0153]
Train Epoch 63 ==> 	accuracy: 0.8941, 	precision: 0.9996, 	recall: 0.7885, 	specificity: 0.9997, 	f1: 0.8816
Test Epoch 63: 100%|██████████| 1768/1768 [01:03<00:00, 27.74it/s, loss=0.429]
Test Epoch 63 ==> 	accuracy: 0.9557, 	precision: 0.9843, 	recall: 0.7960, 	specificity: 0.9967, 	f1: 0.8802
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 64: 100%|██████████| 6507/6507 [09:36<00:00, 11.29it/s, loss=0.0143]
Train Epoch 64 ==> 	accuracy: 0.8947, 	precision: 0.9996, 	recall: 0.7897, 	specificity: 0.9997, 	f1: 0.8824
Test Epoch 64: 100%|██████████| 1768/1768 [01:00<00:00, 29.17it/s, loss=0.127]
Test Epoch 64 ==> 	accuracy: 0.9599, 	precision: 0.9770, 	recall: 0.8233, 	specificity: 0.9950, 	f1: 0.8936
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 65: 100%|██████████| 6507/6507 [09:26<00:00, 11.48it/s, loss=0.0102]
Train Epoch 65 ==> 	accuracy: 0.8961, 	precision: 0.9997, 	recall: 0.7924, 	specificity: 0.9997, 	f1: 0.8840
Test Epoch 65: 100%|██████████| 1768/1768 [01:06<00:00, 26.57it/s, loss=1.22]
Test Epoch 65 ==> 	accuracy: 0.9599, 	precision: 0.9773, 	recall: 0.8231, 	specificity: 0.9951, 	f1: 0.8936
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 66: 100%|██████████| 6507/6507 [09:38<00:00, 11.25it/s, loss=0.0424]
Train Epoch 66 ==> 	accuracy: 0.8945, 	precision: 0.9996, 	recall: 0.7893, 	specificity: 0.9997, 	f1: 0.8821
Test Epoch 66: 100%|██████████| 1768/1768 [01:02<00:00, 28.45it/s, loss=0.087]
Test Epoch 66 ==> 	accuracy: 0.9604, 	precision: 0.9803, 	recall: 0.8229, 	specificity: 0.9957, 	f1: 0.8947
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 67: 100%|██████████| 6507/6507 [09:41<00:00, 11.19it/s, loss=0.0036]
Train Epoch 67 ==> 	accuracy: 0.8971, 	precision: 0.9996, 	recall: 0.7946, 	specificity: 0.9997, 	f1: 0.8854
Test Epoch 67: 100%|██████████| 1768/1768 [01:05<00:00, 26.92it/s, loss=0.106]
Test Epoch 67 ==> 	accuracy: 0.9583, 	precision: 0.9845, 	recall: 0.8089, 	specificity: 0.9967, 	f1: 0.8881
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 68: 100%|██████████| 6507/6507 [09:35<00:00, 11.31it/s, loss=0.0117]
Train Epoch 68 ==> 	accuracy: 0.8987, 	precision: 0.9996, 	recall: 0.7978, 	specificity: 0.9997, 	f1: 0.8874
Test Epoch 68: 100%|██████████| 1768/1768 [01:02<00:00, 28.19it/s, loss=0.523]
Test Epoch 68 ==> 	accuracy: 0.9602, 	precision: 0.9805, 	recall: 0.8217, 	specificity: 0.9958, 	f1: 0.8941
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 69: 100%|██████████| 6507/6507 [09:33<00:00, 11.35it/s, loss=0.0114]
Train Epoch 69 ==> 	accuracy: 0.8968, 	precision: 0.9997, 	recall: 0.7939, 	specificity: 0.9997, 	f1: 0.8850
Test Epoch 69: 100%|██████████| 1768/1768 [01:05<00:00, 27.11it/s, loss=0.254]
Test Epoch 69 ==> 	accuracy: 0.9585, 	precision: 0.9835, 	recall: 0.8107, 	specificity: 0.9965, 	f1: 0.8887
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 70: 100%|██████████| 6507/6507 [09:32<00:00, 11.36it/s, loss=0.0117]
Train Epoch 70 ==> 	accuracy: 0.8980, 	precision: 0.9996, 	recall: 0.7963, 	specificity: 0.9997, 	f1: 0.8865
Test Epoch 70: 100%|██████████| 1768/1768 [01:07<00:00, 26.07it/s, loss=0.377]
Test Epoch 70 ==> 	accuracy: 0.9610, 	precision: 0.9758, 	recall: 0.8301, 	specificity: 0.9947, 	f1: 0.8971
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 71: 100%|██████████| 6507/6507 [09:31<00:00, 11.38it/s, loss=0.0138]
Train Epoch 71 ==> 	accuracy: 0.8989, 	precision: 0.9997, 	recall: 0.7980, 	specificity: 0.9997, 	f1: 0.8875
Test Epoch 71: 100%|██████████| 1768/1768 [01:02<00:00, 28.32it/s, loss=0.387]
Test Epoch 71 ==> 	accuracy: 0.9622, 	precision: 0.9826, 	recall: 0.8300, 	specificity: 0.9962, 	f1: 0.8999
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 72: 100%|██████████| 6507/6507 [09:36<00:00, 11.30it/s, loss=0.0199]
Train Epoch 72 ==> 	accuracy: 0.9001, 	precision: 0.9996, 	recall: 0.8005, 	specificity: 0.9997, 	f1: 0.8891
Test Epoch 72: 100%|██████████| 1768/1768 [01:07<00:00, 26.04it/s, loss=0.102]
Test Epoch 72 ==> 	accuracy: 0.9624, 	precision: 0.9837, 	recall: 0.8298, 	specificity: 0.9965, 	f1: 0.9002
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 73: 100%|██████████| 6507/6507 [09:38<00:00, 11.25it/s, loss=0.0111]
Train Epoch 73 ==> 	accuracy: 0.9014, 	precision: 0.9997, 	recall: 0.8031, 	specificity: 0.9997, 	f1: 0.8906
Test Epoch 73: 100%|██████████| 1768/1768 [01:01<00:00, 28.92it/s, loss=0.0366]
Test Epoch 73 ==> 	accuracy: 0.9620, 	precision: 0.9800, 	recall: 0.8311, 	specificity: 0.9956, 	f1: 0.8994
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 74: 100%|██████████| 6507/6507 [09:32<00:00, 11.37it/s, loss=0.161]
Train Epoch 74 ==> 	accuracy: 0.9019, 	precision: 0.9996, 	recall: 0.8041, 	specificity: 0.9997, 	f1: 0.8913
Test Epoch 74: 100%|██████████| 1768/1768 [01:02<00:00, 28.39it/s, loss=0.194]
Test Epoch 74 ==> 	accuracy: 0.9594, 	precision: 0.9845, 	recall: 0.8142, 	specificity: 0.9967, 	f1: 0.8913
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 75: 100%|██████████| 6507/6507 [09:34<00:00, 11.33it/s, loss=0.0125]
Train Epoch 75 ==> 	accuracy: 0.9011, 	precision: 0.9997, 	recall: 0.8024, 	specificity: 0.9997, 	f1: 0.8902
Test Epoch 75: 100%|██████████| 1768/1768 [01:03<00:00, 27.88it/s, loss=0.381]
Test Epoch 75 ==> 	accuracy: 0.9620, 	precision: 0.9811, 	recall: 0.8301, 	specificity: 0.9959, 	f1: 0.8993
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 76: 100%|██████████| 6507/6507 [09:36<00:00, 11.28it/s, loss=0.558]
Train Epoch 76 ==> 	accuracy: 0.9044, 	precision: 0.9997, 	recall: 0.8091, 	specificity: 0.9997, 	f1: 0.8943
Test Epoch 76: 100%|██████████| 1768/1768 [01:00<00:00, 29.05it/s, loss=0.137]
Test Epoch 76 ==> 	accuracy: 0.9623, 	precision: 0.9810, 	recall: 0.8318, 	specificity: 0.9959, 	f1: 0.9003
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 77: 100%|██████████| 6507/6507 [09:28<00:00, 11.45it/s, loss=0.0067]
Train Epoch 77 ==> 	accuracy: 0.9030, 	precision: 0.9997, 	recall: 0.8062, 	specificity: 0.9997, 	f1: 0.8926
Test Epoch 77: 100%|██████████| 1768/1768 [01:04<00:00, 27.59it/s, loss=0.319]
Test Epoch 77 ==> 	accuracy: 0.9626, 	precision: 0.9771, 	recall: 0.8366, 	specificity: 0.9950, 	f1: 0.9014
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 78: 100%|██████████| 6507/6507 [09:30<00:00, 11.40it/s, loss=0.0603]
Train Epoch 78 ==> 	accuracy: 0.9046, 	precision: 0.9997, 	recall: 0.8095, 	specificity: 0.9997, 	f1: 0.8946
Test Epoch 78: 100%|██████████| 1768/1768 [01:03<00:00, 27.99it/s, loss=0.253]
Test Epoch 78 ==> 	accuracy: 0.9630, 	precision: 0.9805, 	recall: 0.8359, 	specificity: 0.9957, 	f1: 0.9024
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 79: 100%|██████████| 6507/6507 [09:32<00:00, 11.36it/s, loss=2.87]
Train Epoch 79 ==> 	accuracy: 0.9024, 	precision: 0.9997, 	recall: 0.8050, 	specificity: 0.9997, 	f1: 0.8918
Test Epoch 79: 100%|██████████| 1768/1768 [01:03<00:00, 27.83it/s, loss=0.107]
Test Epoch 79 ==> 	accuracy: 0.9630, 	precision: 0.9800, 	recall: 0.8363, 	specificity: 0.9956, 	f1: 0.9025
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 80: 100%|██████████| 6507/6507 [09:30<00:00, 11.41it/s, loss=0.041]
Train Epoch 80 ==> 	accuracy: 0.9043, 	precision: 0.9997, 	recall: 0.8089, 	specificity: 0.9998, 	f1: 0.8942
Test Epoch 80: 100%|██████████| 1768/1768 [01:05<00:00, 26.84it/s, loss=0.0866]
Test Epoch 80 ==> 	accuracy: 0.9633, 	precision: 0.9804, 	recall: 0.8374, 	specificity: 0.9957, 	f1: 0.9033
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 81: 100%|██████████| 6507/6507 [09:34<00:00, 11.32it/s, loss=0.01]
Train Epoch 81 ==> 	accuracy: 0.9056, 	precision: 0.9997, 	recall: 0.8115, 	specificity: 0.9998, 	f1: 0.8958
Test Epoch 81: 100%|██████████| 1768/1768 [01:02<00:00, 28.30it/s, loss=0.196]
Test Epoch 81 ==> 	accuracy: 0.9627, 	precision: 0.9833, 	recall: 0.8318, 	specificity: 0.9964, 	f1: 0.9012
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 82: 100%|██████████| 6507/6507 [09:38<00:00, 11.25it/s, loss=0.0065]
Train Epoch 82 ==> 	accuracy: 0.9045, 	precision: 0.9997, 	recall: 0.8093, 	specificity: 0.9998, 	f1: 0.8945
Test Epoch 82: 100%|██████████| 1768/1768 [01:05<00:00, 26.99it/s, loss=0.153]
Test Epoch 82 ==> 	accuracy: 0.9632, 	precision: 0.9838, 	recall: 0.8338, 	specificity: 0.9965, 	f1: 0.9026
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 83: 100%|██████████| 6507/6507 [09:40<00:00, 11.22it/s, loss=0.007]
Train Epoch 83 ==> 	accuracy: 0.9052, 	precision: 0.9997, 	recall: 0.8107, 	specificity: 0.9997, 	f1: 0.8953
Test Epoch 83: 100%|██████████| 1768/1768 [01:04<00:00, 27.51it/s, loss=0.326]
Test Epoch 83 ==> 	accuracy: 0.9629, 	precision: 0.9785, 	recall: 0.8372, 	specificity: 0.9953, 	f1: 0.9023
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 84: 100%|██████████| 6507/6507 [09:31<00:00, 11.38it/s, loss=1.3]
Train Epoch 84 ==> 	accuracy: 0.9067, 	precision: 0.9997, 	recall: 0.8136, 	specificity: 0.9998, 	f1: 0.8971
Test Epoch 84: 100%|██████████| 1768/1768 [01:07<00:00, 26.38it/s, loss=0.746]
Test Epoch 84 ==> 	accuracy: 0.9636, 	precision: 0.9806, 	recall: 0.8385, 	specificity: 0.9957, 	f1: 0.9040
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 85: 100%|██████████| 6507/6507 [09:35<00:00, 11.31it/s, loss=1.27]
Train Epoch 85 ==> 	accuracy: 0.9091, 	precision: 0.9997, 	recall: 0.8184, 	specificity: 0.9997, 	f1: 0.9000
Test Epoch 85: 100%|██████████| 1768/1768 [01:04<00:00, 27.21it/s, loss=0.24]
Test Epoch 85 ==> 	accuracy: 0.9634, 	precision: 0.9827, 	recall: 0.8356, 	specificity: 0.9962, 	f1: 0.9032
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 86: 100%|██████████| 6507/6507 [09:33<00:00, 11.35it/s, loss=0.028]
Train Epoch 86 ==> 	accuracy: 0.9098, 	precision: 0.9997, 	recall: 0.8198, 	specificity: 0.9998, 	f1: 0.9008
Test Epoch 86: 100%|██████████| 1768/1768 [01:04<00:00, 27.33it/s, loss=0.417]
Test Epoch 86 ==> 	accuracy: 0.9648, 	precision: 0.9786, 	recall: 0.8462, 	specificity: 0.9952, 	f1: 0.9076
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 87: 100%|██████████| 6507/6507 [09:29<00:00, 11.43it/s, loss=1.25]
Train Epoch 87 ==> 	accuracy: 0.9083, 	precision: 0.9997, 	recall: 0.8168, 	specificity: 0.9997, 	f1: 0.8990
Test Epoch 87: 100%|██████████| 1768/1768 [01:01<00:00, 28.61it/s, loss=0.104]
Test Epoch 87 ==> 	accuracy: 0.9634, 	precision: 0.9786, 	recall: 0.8393, 	specificity: 0.9953, 	f1: 0.9036
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 88: 100%|██████████| 6507/6507 [09:38<00:00, 11.25it/s, loss=0.242]
Train Epoch 88 ==> 	accuracy: 0.9094, 	precision: 0.9997, 	recall: 0.8191, 	specificity: 0.9997, 	f1: 0.9004
Test Epoch 88: 100%|██████████| 1768/1768 [01:03<00:00, 27.87it/s, loss=2.23]
Test Epoch 88 ==> 	accuracy: 0.9620, 	precision: 0.9802, 	recall: 0.8312, 	specificity: 0.9957, 	f1: 0.8996
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 89: 100%|██████████| 6507/6507 [09:47<00:00, 11.07it/s, loss=0.0066]
Train Epoch 89 ==> 	accuracy: 0.9107, 	precision: 0.9997, 	recall: 0.8216, 	specificity: 0.9998, 	f1: 0.9019
Test Epoch 89: 100%|██████████| 1768/1768 [01:00<00:00, 29.15it/s, loss=0.231]
Test Epoch 89 ==> 	accuracy: 0.9650, 	precision: 0.9755, 	recall: 0.8502, 	specificity: 0.9945, 	f1: 0.9086
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 90: 100%|██████████| 6507/6507 [09:37<00:00, 11.27it/s, loss=0.0494]
Train Epoch 90 ==> 	accuracy: 0.9088, 	precision: 0.9997, 	recall: 0.8178, 	specificity: 0.9998, 	f1: 0.8997
Test Epoch 90: 100%|██████████| 1768/1768 [01:04<00:00, 27.44it/s, loss=0.262]
Test Epoch 90 ==> 	accuracy: 0.9634, 	precision: 0.9812, 	recall: 0.8371, 	specificity: 0.9959, 	f1: 0.9035
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 91: 100%|██████████| 6507/6507 [09:40<00:00, 11.21it/s, loss=0.0364]
Train Epoch 91 ==> 	accuracy: 0.9128, 	precision: 0.9997, 	recall: 0.8258, 	specificity: 0.9998, 	f1: 0.9045
Test Epoch 91: 100%|██████████| 1768/1768 [01:03<00:00, 27.79it/s, loss=1.33]
Test Epoch 91 ==> 	accuracy: 0.9659, 	precision: 0.9745, 	recall: 0.8559, 	specificity: 0.9942, 	f1: 0.9113
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 92: 100%|██████████| 6507/6507 [09:47<00:00, 11.08it/s, loss=0.0582]
Train Epoch 92 ==> 	accuracy: 0.9100, 	precision: 0.9997, 	recall: 0.8202, 	specificity: 0.9998, 	f1: 0.9011
Test Epoch 92: 100%|██████████| 1768/1768 [01:07<00:00, 26.20it/s, loss=0.119]
Test Epoch 92 ==> 	accuracy: 0.9642, 	precision: 0.9820, 	recall: 0.8403, 	specificity: 0.9960, 	f1: 0.9056
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 93: 100%|██████████| 6507/6507 [09:31<00:00, 11.39it/s, loss=0.0318]
Train Epoch 93 ==> 	accuracy: 0.9124, 	precision: 0.9997, 	recall: 0.8250, 	specificity: 0.9998, 	f1: 0.9040
Test Epoch 93: 100%|██████████| 1768/1768 [01:00<00:00, 29.10it/s, loss=0.172]
Test Epoch 93 ==> 	accuracy: 0.9651, 	precision: 0.9773, 	recall: 0.8493, 	specificity: 0.9949, 	f1: 0.9088
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 94: 100%|██████████| 6507/6507 [09:37<00:00, 11.26it/s, loss=0.0429]
Train Epoch 94 ==> 	accuracy: 0.9108, 	precision: 0.9997, 	recall: 0.8219, 	specificity: 0.9998, 	f1: 0.9021
Test Epoch 94: 100%|██████████| 1768/1768 [01:02<00:00, 28.40it/s, loss=0.194]
Test Epoch 94 ==> 	accuracy: 0.9643, 	precision: 0.9784, 	recall: 0.8443, 	specificity: 0.9952, 	f1: 0.9064
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 95: 100%|██████████| 6507/6507 [09:32<00:00, 11.36it/s, loss=0.0046]
Train Epoch 95 ==> 	accuracy: 0.9105, 	precision: 0.9997, 	recall: 0.8212, 	specificity: 0.9998, 	f1: 0.9017
Test Epoch 95: 100%|██████████| 1768/1768 [01:01<00:00, 28.75it/s, loss=0.323]
Test Epoch 95 ==> 	accuracy: 0.9648, 	precision: 0.9792, 	recall: 0.8460, 	specificity: 0.9954, 	f1: 0.9077
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 96: 100%|██████████| 6507/6507 [09:28<00:00, 11.45it/s, loss=0.0006]
Train Epoch 96 ==> 	accuracy: 0.9117, 	precision: 0.9997, 	recall: 0.8237, 	specificity: 0.9998, 	f1: 0.9032
Test Epoch 96: 100%|██████████| 1768/1768 [01:05<00:00, 27.11it/s, loss=0.262]
Test Epoch 96 ==> 	accuracy: 0.9641, 	precision: 0.9814, 	recall: 0.8403, 	specificity: 0.9959, 	f1: 0.9054
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 97: 100%|██████████| 6507/6507 [09:34<00:00, 11.33it/s, loss=0.017]
Train Epoch 97 ==> 	accuracy: 0.9122, 	precision: 0.9997, 	recall: 0.8246, 	specificity: 0.9998, 	f1: 0.9037
Test Epoch 97: 100%|██████████| 1768/1768 [01:02<00:00, 28.43it/s, loss=4.67]
Test Epoch 97 ==> 	accuracy: 0.9653, 	precision: 0.9770, 	recall: 0.8502, 	specificity: 0.9949, 	f1: 0.9092
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 98: 100%|██████████| 6507/6507 [09:31<00:00, 11.38it/s, loss=0.0374]
Train Epoch 98 ==> 	accuracy: 0.9149, 	precision: 0.9997, 	recall: 0.8300, 	specificity: 0.9998, 	f1: 0.9070
Test Epoch 98: 100%|██████████| 1768/1768 [01:06<00:00, 26.72it/s, loss=0.0739]
Test Epoch 98 ==> 	accuracy: 0.9661, 	precision: 0.9766, 	recall: 0.8548, 	specificity: 0.9947, 	f1: 0.9117
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 99: 100%|██████████| 6507/6507 [09:38<00:00, 11.25it/s, loss=0.0046]
Train Epoch 99 ==> 	accuracy: 0.9125, 	precision: 0.9997, 	recall: 0.8252, 	specificity: 0.9997, 	f1: 0.9041
Test Epoch 99: 100%|██████████| 1768/1768 [01:01<00:00, 28.91it/s, loss=0.331]
Test Epoch 99 ==> 	accuracy: 0.9646, 	precision: 0.9797, 	recall: 0.8447, 	specificity: 0.9955, 	f1: 0.9072
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 100: 100%|██████████| 6507/6507 [09:45<00:00, 11.12it/s, loss=0.0155]
Train Epoch 100 ==> 	accuracy: 0.9155, 	precision: 0.9997, 	recall: 0.8313, 	specificity: 0.9998, 	f1: 0.9077
Test Epoch 100: 100%|██████████| 1768/1768 [00:58<00:00, 30.02it/s, loss=3.83]
Test Epoch 100 ==> 	accuracy: 0.9656, 	precision: 0.9766, 	recall: 0.8522, 	specificity: 0.9947, 	f1: 0.9102
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 101: 100%|██████████| 6507/6507 [09:29<00:00, 11.42it/s, loss=0.032]
Train Epoch 101 ==> 	accuracy: 0.9158, 	precision: 0.9997, 	recall: 0.8319, 	specificity: 0.9998, 	f1: 0.9081
Test Epoch 101: 100%|██████████| 1768/1768 [01:03<00:00, 27.78it/s, loss=0.227]
Test Epoch 101 ==> 	accuracy: 0.9662, 	precision: 0.9753, 	recall: 0.8563, 	specificity: 0.9944, 	f1: 0.9119
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 102: 100%|██████████| 6507/6507 [09:37<00:00, 11.28it/s, loss=0.013]
Train Epoch 102 ==> 	accuracy: 0.9155, 	precision: 0.9997, 	recall: 0.8313, 	specificity: 0.9998, 	f1: 0.9078
Test Epoch 102: 100%|██████████| 1768/1768 [01:03<00:00, 27.72it/s, loss=0.306]
Test Epoch 102 ==> 	accuracy: 0.9660, 	precision: 0.9769, 	recall: 0.8541, 	specificity: 0.9948, 	f1: 0.9114
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 103: 100%|██████████| 6507/6507 [09:31<00:00, 11.39it/s, loss=0.0208]
Train Epoch 103 ==> 	accuracy: 0.9142, 	precision: 0.9997, 	recall: 0.8287, 	specificity: 0.9998, 	f1: 0.9062
Test Epoch 103: 100%|██████████| 1768/1768 [00:58<00:00, 30.14it/s, loss=0.0899]
Test Epoch 103 ==> 	accuracy: 0.9658, 	precision: 0.9772, 	recall: 0.8529, 	specificity: 0.9949, 	f1: 0.9108
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 104: 100%|██████████| 6507/6507 [09:25<00:00, 11.51it/s, loss=0.0034]
Train Epoch 104 ==> 	accuracy: 0.9145, 	precision: 0.9997, 	recall: 0.8293, 	specificity: 0.9998, 	f1: 0.9066
Test Epoch 104: 100%|██████████| 1768/1768 [01:01<00:00, 28.81it/s, loss=1.02]
Test Epoch 104 ==> 	accuracy: 0.9667, 	precision: 0.9747, 	recall: 0.8593, 	specificity: 0.9943, 	f1: 0.9134
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 105: 100%|██████████| 6507/6507 [09:19<00:00, 11.62it/s, loss=0.026]
Train Epoch 105 ==> 	accuracy: 0.9155, 	precision: 0.9997, 	recall: 0.8313, 	specificity: 0.9998, 	f1: 0.9078
Test Epoch 105: 100%|██████████| 1768/1768 [01:00<00:00, 29.26it/s, loss=0.842]
Test Epoch 105 ==> 	accuracy: 0.9654, 	precision: 0.9824, 	recall: 0.8459, 	specificity: 0.9961, 	f1: 0.9090
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 106: 100%|██████████| 6507/6507 [09:36<00:00, 11.29it/s, loss=0.0551]
Train Epoch 106 ==> 	accuracy: 0.9160, 	precision: 0.9997, 	recall: 0.8322, 	specificity: 0.9998, 	f1: 0.9083
Test Epoch 106: 100%|██████████| 1768/1768 [01:00<00:00, 29.34it/s, loss=0.558]
Test Epoch 106 ==> 	accuracy: 0.9671, 	precision: 0.9734, 	recall: 0.8627, 	specificity: 0.9939, 	f1: 0.9147
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 107: 100%|██████████| 6507/6507 [09:30<00:00, 11.41it/s, loss=0.0154]
Train Epoch 107 ==> 	accuracy: 0.9170, 	precision: 0.9998, 	recall: 0.8343, 	specificity: 0.9998, 	f1: 0.9095
Test Epoch 107: 100%|██████████| 1768/1768 [01:03<00:00, 27.68it/s, loss=1.58]
Test Epoch 107 ==> 	accuracy: 0.9667, 	precision: 0.9722, 	recall: 0.8616, 	specificity: 0.9937, 	f1: 0.9136
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 108: 100%|██████████| 6507/6507 [09:29<00:00, 11.43it/s, loss=0.287]
Train Epoch 108 ==> 	accuracy: 0.9173, 	precision: 0.9997, 	recall: 0.8349, 	specificity: 0.9998, 	f1: 0.9099
Test Epoch 108: 100%|██████████| 1768/1768 [01:06<00:00, 26.54it/s, loss=0.253]
Test Epoch 108 ==> 	accuracy: 0.9660, 	precision: 0.9774, 	recall: 0.8535, 	specificity: 0.9949, 	f1: 0.9113
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 109: 100%|██████████| 6507/6507 [09:41<00:00, 11.20it/s, loss=1.7]
Train Epoch 109 ==> 	accuracy: 0.9174, 	precision: 0.9998, 	recall: 0.8349, 	specificity: 0.9998, 	f1: 0.9099
Test Epoch 109: 100%|██████████| 1768/1768 [01:00<00:00, 29.22it/s, loss=0.185]
Test Epoch 109 ==> 	accuracy: 0.9643, 	precision: 0.9794, 	recall: 0.8431, 	specificity: 0.9954, 	f1: 0.9062
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 110: 100%|██████████| 6507/6507 [09:32<00:00, 11.36it/s, loss=0.0081]
Train Epoch 110 ==> 	accuracy: 0.9164, 	precision: 0.9997, 	recall: 0.8330, 	specificity: 0.9998, 	f1: 0.9088
Test Epoch 110: 100%|██████████| 1768/1768 [00:58<00:00, 30.44it/s, loss=0.398]
Test Epoch 110 ==> 	accuracy: 0.9661, 	precision: 0.9785, 	recall: 0.8528, 	specificity: 0.9952, 	f1: 0.9114
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 111: 100%|██████████| 6507/6507 [09:29<00:00, 11.42it/s, loss=0.117]
Train Epoch 111 ==> 	accuracy: 0.9190, 	precision: 0.9997, 	recall: 0.8383, 	specificity: 0.9998, 	f1: 0.9119
Test Epoch 111: 100%|██████████| 1768/1768 [01:02<00:00, 28.16it/s, loss=0.317]
Test Epoch 111 ==> 	accuracy: 0.9667, 	precision: 0.9767, 	recall: 0.8577, 	specificity: 0.9947, 	f1: 0.9134
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 112: 100%|██████████| 6507/6507 [09:30<00:00, 11.40it/s, loss=0.0009]
Train Epoch 112 ==> 	accuracy: 0.9158, 	precision: 0.9997, 	recall: 0.8317, 	specificity: 0.9998, 	f1: 0.9080
Test Epoch 112: 100%|██████████| 1768/1768 [01:02<00:00, 28.17it/s, loss=0.0765]
Test Epoch 112 ==> 	accuracy: 0.9666, 	precision: 0.9762, 	recall: 0.8578, 	specificity: 0.9946, 	f1: 0.9132
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 113: 100%|██████████| 6507/6507 [09:40<00:00, 11.20it/s, loss=0.0154]
Train Epoch 113 ==> 	accuracy: 0.9176, 	precision: 0.9998, 	recall: 0.8353, 	specificity: 0.9998, 	f1: 0.9102
Test Epoch 113: 100%|██████████| 1768/1768 [01:01<00:00, 28.87it/s, loss=0.0558]
Test Epoch 113 ==> 	accuracy: 0.9666, 	precision: 0.9712, 	recall: 0.8624, 	specificity: 0.9934, 	f1: 0.9136
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 114: 100%|██████████| 6507/6507 [09:28<00:00, 11.44it/s, loss=0.0158]
Train Epoch 114 ==> 	accuracy: 0.9179, 	precision: 0.9997, 	recall: 0.8360, 	specificity: 0.9998, 	f1: 0.9106
Test Epoch 114: 100%|██████████| 1768/1768 [01:04<00:00, 27.29it/s, loss=0.188]
Test Epoch 114 ==> 	accuracy: 0.9668, 	precision: 0.9742, 	recall: 0.8606, 	specificity: 0.9941, 	f1: 0.9138
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 115: 100%|██████████| 6507/6507 [09:26<00:00, 11.48it/s, loss=0.0009]
Train Epoch 115 ==> 	accuracy: 0.9195, 	precision: 0.9997, 	recall: 0.8392, 	specificity: 0.9998, 	f1: 0.9125
Test Epoch 115: 100%|██████████| 1768/1768 [01:04<00:00, 27.38it/s, loss=0.107]
Test Epoch 115 ==> 	accuracy: 0.9671, 	precision: 0.9747, 	recall: 0.8615, 	specificity: 0.9942, 	f1: 0.9146
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 116: 100%|██████████| 6507/6507 [09:39<00:00, 11.22it/s, loss=0.0016]
Train Epoch 116 ==> 	accuracy: 0.9164, 	precision: 0.9998, 	recall: 0.8331, 	specificity: 0.9998, 	f1: 0.9088
Test Epoch 116: 100%|██████████| 1768/1768 [01:01<00:00, 28.70it/s, loss=1.22]
Test Epoch 116 ==> 	accuracy: 0.9665, 	precision: 0.9777, 	recall: 0.8558, 	specificity: 0.9950, 	f1: 0.9127
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 117: 100%|██████████| 6507/6507 [09:28<00:00, 11.45it/s, loss=0.717]
Train Epoch 117 ==> 	accuracy: 0.9198, 	precision: 0.9998, 	recall: 0.8399, 	specificity: 0.9998, 	f1: 0.9129
Test Epoch 117: 100%|██████████| 1768/1768 [01:08<00:00, 25.89it/s, loss=0.158]
Test Epoch 117 ==> 	accuracy: 0.9668, 	precision: 0.9744, 	recall: 0.8603, 	specificity: 0.9942, 	f1: 0.9138
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 118: 100%|██████████| 6507/6507 [09:47<00:00, 11.08it/s, loss=0.673]
Train Epoch 118 ==> 	accuracy: 0.9186, 	precision: 0.9997, 	recall: 0.8375, 	specificity: 0.9998, 	f1: 0.9114
Test Epoch 118: 100%|██████████| 1768/1768 [01:05<00:00, 27.12it/s, loss=0.387]
Test Epoch 118 ==> 	accuracy: 0.9678, 	precision: 0.9776, 	recall: 0.8624, 	specificity: 0.9949, 	f1: 0.9164
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 119: 100%|██████████| 6507/6507 [09:48<00:00, 11.05it/s, loss=0.0103]
Train Epoch 119 ==> 	accuracy: 0.9189, 	precision: 0.9997, 	recall: 0.8381, 	specificity: 0.9998, 	f1: 0.9118
Test Epoch 119: 100%|██████████| 1768/1768 [01:02<00:00, 28.40it/s, loss=0.801]
Test Epoch 119 ==> 	accuracy: 0.9672, 	precision: 0.9799, 	recall: 0.8574, 	specificity: 0.9955, 	f1: 0.9145
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 120: 100%|██████████| 6507/6507 [09:46<00:00, 11.09it/s, loss=0.0162]
Train Epoch 120 ==> 	accuracy: 0.9176, 	precision: 0.9998, 	recall: 0.8354, 	specificity: 0.9998, 	f1: 0.9102
Test Epoch 120: 100%|██████████| 1768/1768 [01:09<00:00, 25.52it/s, loss=1.13]
Test Epoch 120 ==> 	accuracy: 0.9673, 	precision: 0.9796, 	recall: 0.8582, 	specificity: 0.9954, 	f1: 0.9149
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 121: 100%|██████████| 6507/6507 [09:49<00:00, 11.05it/s, loss=0.0082]
Train Epoch 121 ==> 	accuracy: 0.9202, 	precision: 0.9998, 	recall: 0.8405, 	specificity: 0.9998, 	f1: 0.9133
Test Epoch 121: 100%|██████████| 1768/1768 [01:07<00:00, 26.30it/s, loss=2.77]
Test Epoch 121 ==> 	accuracy: 0.9684, 	precision: 0.9763, 	recall: 0.8665, 	specificity: 0.9946, 	f1: 0.9181
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 122: 100%|██████████| 6507/6507 [09:43<00:00, 11.15it/s, loss=0.0003]
Train Epoch 122 ==> 	accuracy: 0.9207, 	precision: 0.9997, 	recall: 0.8416, 	specificity: 0.9998, 	f1: 0.9139
Test Epoch 122: 100%|██████████| 1768/1768 [01:02<00:00, 28.07it/s, loss=3.1]
Test Epoch 122 ==> 	accuracy: 0.9675, 	precision: 0.9778, 	recall: 0.8605, 	specificity: 0.9950, 	f1: 0.9154
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 123: 100%|██████████| 6507/6507 [09:45<00:00, 11.11it/s, loss=0.246]
Train Epoch 123 ==> 	accuracy: 0.9182, 	precision: 0.9998, 	recall: 0.8366, 	specificity: 0.9998, 	f1: 0.9109
Test Epoch 123: 100%|██████████| 1768/1768 [01:03<00:00, 28.02it/s, loss=0.0676]
Test Epoch 123 ==> 	accuracy: 0.9675, 	precision: 0.9765, 	recall: 0.8620, 	specificity: 0.9947, 	f1: 0.9157
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 124: 100%|██████████| 6507/6507 [09:47<00:00, 11.08it/s, loss=0.0219]
Train Epoch 124 ==> 	accuracy: 0.9217, 	precision: 0.9998, 	recall: 0.8435, 	specificity: 0.9998, 	f1: 0.9150
Test Epoch 124: 100%|██████████| 1768/1768 [01:06<00:00, 26.61it/s, loss=0.972]
Test Epoch 124 ==> 	accuracy: 0.9680, 	precision: 0.9730, 	recall: 0.8675, 	specificity: 0.9938, 	f1: 0.9173
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 125: 100%|██████████| 6507/6507 [09:36<00:00, 11.29it/s, loss=0.0104]
Train Epoch 125 ==> 	accuracy: 0.9213, 	precision: 0.9998, 	recall: 0.8428, 	specificity: 0.9998, 	f1: 0.9146
Test Epoch 125: 100%|██████████| 1768/1768 [01:05<00:00, 27.08it/s, loss=1.16]
Test Epoch 125 ==> 	accuracy: 0.9677, 	precision: 0.9744, 	recall: 0.8648, 	specificity: 0.9942, 	f1: 0.9163
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 126: 100%|██████████| 6507/6507 [09:45<00:00, 11.12it/s, loss=0.0066]
Train Epoch 126 ==> 	accuracy: 0.9212, 	precision: 0.9997, 	recall: 0.8427, 	specificity: 0.9998, 	f1: 0.9145
Test Epoch 126: 100%|██████████| 1768/1768 [01:08<00:00, 25.64it/s, loss=0.13]
Test Epoch 126 ==> 	accuracy: 0.9671, 	precision: 0.9753, 	recall: 0.8611, 	specificity: 0.9944, 	f1: 0.9146
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 127: 100%|██████████| 6507/6507 [09:42<00:00, 11.18it/s, loss=0.0035]
Train Epoch 127 ==> 	accuracy: 0.9199, 	precision: 0.9997, 	recall: 0.8400, 	specificity: 0.9998, 	f1: 0.9129
Test Epoch 127: 100%|██████████| 1768/1768 [01:03<00:00, 27.93it/s, loss=1.57]
Test Epoch 127 ==> 	accuracy: 0.9673, 	precision: 0.9739, 	recall: 0.8633, 	specificity: 0.9941, 	f1: 0.9153
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 128: 100%|██████████| 6507/6507 [09:58<00:00, 10.87it/s, loss=0.0028]
Train Epoch 128 ==> 	accuracy: 0.9212, 	precision: 0.9997, 	recall: 0.8426, 	specificity: 0.9998, 	f1: 0.9145
Test Epoch 128: 100%|██████████| 1768/1768 [01:03<00:00, 27.79it/s, loss=0.85]
Test Epoch 128 ==> 	accuracy: 0.9679, 	precision: 0.9759, 	recall: 0.8646, 	specificity: 0.9945, 	f1: 0.9169
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 129: 100%|██████████| 6507/6507 [09:54<00:00, 10.94it/s, loss=0.0029]
Train Epoch 129 ==> 	accuracy: 0.9195, 	precision: 0.9997, 	recall: 0.8392, 	specificity: 0.9998, 	f1: 0.9125
Test Epoch 129: 100%|██████████| 1768/1768 [01:05<00:00, 26.85it/s, loss=0.171]
Test Epoch 129 ==> 	accuracy: 0.9675, 	precision: 0.9764, 	recall: 0.8618, 	specificity: 0.9946, 	f1: 0.9155
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 130: 100%|██████████| 6507/6507 [09:52<00:00, 10.98it/s, loss=0.0475]
Train Epoch 130 ==> 	accuracy: 0.9207, 	precision: 0.9998, 	recall: 0.8417, 	specificity: 0.9998, 	f1: 0.9139
Test Epoch 130: 100%|██████████| 1768/1768 [01:03<00:00, 27.81it/s, loss=0.151]
Test Epoch 130 ==> 	accuracy: 0.9679, 	precision: 0.9755, 	recall: 0.8649, 	specificity: 0.9944, 	f1: 0.9168
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 131: 100%|██████████| 6507/6507 [09:53<00:00, 10.96it/s, loss=0.002]
Train Epoch 131 ==> 	accuracy: 0.9206, 	precision: 0.9998, 	recall: 0.8413, 	specificity: 0.9998, 	f1: 0.9137
Test Epoch 131: 100%|██████████| 1768/1768 [01:05<00:00, 27.15it/s, loss=0.961]
Test Epoch 131 ==> 	accuracy: 0.9680, 	precision: 0.9763, 	recall: 0.8643, 	specificity: 0.9946, 	f1: 0.9169
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 132: 100%|██████████| 6507/6507 [10:07<00:00, 10.72it/s, loss=0.0087]
Train Epoch 132 ==> 	accuracy: 0.9217, 	precision: 0.9998, 	recall: 0.8435, 	specificity: 0.9998, 	f1: 0.9150
Test Epoch 132: 100%|██████████| 1768/1768 [01:02<00:00, 28.15it/s, loss=0.254]
Test Epoch 132 ==> 	accuracy: 0.9684, 	precision: 0.9744, 	recall: 0.8681, 	specificity: 0.9941, 	f1: 0.9182
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 133: 100%|██████████| 6507/6507 [09:55<00:00, 10.92it/s, loss=0.0454]
Train Epoch 133 ==> 	accuracy: 0.9212, 	precision: 0.9997, 	recall: 0.8427, 	specificity: 0.9998, 	f1: 0.9145
Test Epoch 133: 100%|██████████| 1768/1768 [01:08<00:00, 25.95it/s, loss=1.47]
Test Epoch 133 ==> 	accuracy: 0.9679, 	precision: 0.9749, 	recall: 0.8655, 	specificity: 0.9943, 	f1: 0.9169
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 134: 100%|██████████| 6507/6507 [10:06<00:00, 10.73it/s, loss=0.0011]
Train Epoch 134 ==> 	accuracy: 0.9224, 	precision: 0.9998, 	recall: 0.8451, 	specificity: 0.9998, 	f1: 0.9159
Test Epoch 134: 100%|██████████| 1768/1768 [01:10<00:00, 25.01it/s, loss=0.149]
Test Epoch 134 ==> 	accuracy: 0.9682, 	precision: 0.9737, 	recall: 0.8680, 	specificity: 0.9940, 	f1: 0.9178
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 135: 100%|██████████| 6507/6507 [10:07<00:00, 10.70it/s, loss=0.891]
Train Epoch 135 ==> 	accuracy: 0.9229, 	precision: 0.9998, 	recall: 0.8461, 	specificity: 0.9998, 	f1: 0.9165
Test Epoch 135: 100%|██████████| 1768/1768 [01:03<00:00, 28.00it/s, loss=0.936]
Test Epoch 135 ==> 	accuracy: 0.9676, 	precision: 0.9732, 	recall: 0.8654, 	specificity: 0.9939, 	f1: 0.9162
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 136: 100%|██████████| 6507/6507 [10:07<00:00, 10.71it/s, loss=0.0043]
Train Epoch 136 ==> 	accuracy: 0.9225, 	precision: 0.9998, 	recall: 0.8452, 	specificity: 0.9998, 	f1: 0.9160
Test Epoch 136: 100%|██████████| 1768/1768 [01:03<00:00, 27.97it/s, loss=0.202]
Test Epoch 136 ==> 	accuracy: 0.9675, 	precision: 0.9739, 	recall: 0.8641, 	specificity: 0.9940, 	f1: 0.9157
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 137: 100%|██████████| 6507/6507 [10:05<00:00, 10.75it/s, loss=0.0291]
Train Epoch 137 ==> 	accuracy: 0.9239, 	precision: 0.9998, 	recall: 0.8480, 	specificity: 0.9998, 	f1: 0.9176
Test Epoch 137: 100%|██████████| 1768/1768 [01:06<00:00, 26.65it/s, loss=0.9]
Test Epoch 137 ==> 	accuracy: 0.9677, 	precision: 0.9735, 	recall: 0.8657, 	specificity: 0.9939, 	f1: 0.9165
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 138: 100%|██████████| 6507/6507 [10:17<00:00, 10.54it/s, loss=0.0035]
Train Epoch 138 ==> 	accuracy: 0.9231, 	precision: 0.9998, 	recall: 0.8464, 	specificity: 0.9998, 	f1: 0.9167
Test Epoch 138: 100%|██████████| 1768/1768 [01:06<00:00, 26.49it/s, loss=0.119]
Test Epoch 138 ==> 	accuracy: 0.9678, 	precision: 0.9721, 	recall: 0.8676, 	specificity: 0.9936, 	f1: 0.9169
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 139: 100%|██████████| 6507/6507 [10:16<00:00, 10.55it/s, loss=0.0017]
Train Epoch 139 ==> 	accuracy: 0.9238, 	precision: 0.9998, 	recall: 0.8478, 	specificity: 0.9998, 	f1: 0.9175
Test Epoch 139: 100%|██████████| 1768/1768 [01:08<00:00, 25.79it/s, loss=0.399]
Test Epoch 139 ==> 	accuracy: 0.9682, 	precision: 0.9713, 	recall: 0.8704, 	specificity: 0.9934, 	f1: 0.9181
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 140: 100%|██████████| 6507/6507 [10:06<00:00, 10.74it/s, loss=0.0067]
Train Epoch 140 ==> 	accuracy: 0.9213, 	precision: 0.9998, 	recall: 0.8428, 	specificity: 0.9998, 	f1: 0.9146
Test Epoch 140: 100%|██████████| 1768/1768 [01:04<00:00, 27.23it/s, loss=0.14]
Test Epoch 140 ==> 	accuracy: 0.9674, 	precision: 0.9767, 	recall: 0.8614, 	specificity: 0.9947, 	f1: 0.9154
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 141: 100%|██████████| 6507/6507 [09:58<00:00, 10.88it/s, loss=0.0563]
Train Epoch 141 ==> 	accuracy: 0.9240, 	precision: 0.9998, 	recall: 0.8483, 	specificity: 0.9998, 	f1: 0.9178
Test Epoch 141: 100%|██████████| 1768/1768 [01:05<00:00, 27.18it/s, loss=0.566]
Test Epoch 141 ==> 	accuracy: 0.9681, 	precision: 0.9731, 	recall: 0.8678, 	specificity: 0.9938, 	f1: 0.9175
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 142: 100%|██████████| 6507/6507 [10:08<00:00, 10.70it/s, loss=0.696]
Train Epoch 142 ==> 	accuracy: 0.9219, 	precision: 0.9998, 	recall: 0.8440, 	specificity: 0.9998, 	f1: 0.9153
Test Epoch 142: 100%|██████████| 1768/1768 [01:07<00:00, 26.07it/s, loss=2.03]
Test Epoch 142 ==> 	accuracy: 0.9677, 	precision: 0.9746, 	recall: 0.8647, 	specificity: 0.9942, 	f1: 0.9163
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 143: 100%|██████████| 6507/6507 [10:10<00:00, 10.66it/s, loss=0.007]
Train Epoch 143 ==> 	accuracy: 0.9242, 	precision: 0.9998, 	recall: 0.8486, 	specificity: 0.9998, 	f1: 0.9180
Test Epoch 143: 100%|██████████| 1768/1768 [01:05<00:00, 27.13it/s, loss=2.97]
Test Epoch 143 ==> 	accuracy: 0.9681, 	precision: 0.9758, 	recall: 0.8655, 	specificity: 0.9945, 	f1: 0.9173
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 144: 100%|██████████| 6507/6507 [10:11<00:00, 10.64it/s, loss=0.0056]
Train Epoch 144 ==> 	accuracy: 0.9228, 	precision: 0.9998, 	recall: 0.8457, 	specificity: 0.9998, 	f1: 0.9163
Test Epoch 144: 100%|██████████| 1768/1768 [01:05<00:00, 27.08it/s, loss=0.366]
Test Epoch 144 ==> 	accuracy: 0.9684, 	precision: 0.9767, 	recall: 0.8664, 	specificity: 0.9947, 	f1: 0.9182
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 145: 100%|██████████| 6507/6507 [10:15<00:00, 10.58it/s, loss=0.0375]
Train Epoch 145 ==> 	accuracy: 0.9226, 	precision: 0.9997, 	recall: 0.8454, 	specificity: 0.9998, 	f1: 0.9161
Test Epoch 145: 100%|██████████| 1768/1768 [01:04<00:00, 27.40it/s, loss=0.332]
Test Epoch 145 ==> 	accuracy: 0.9294, 	precision: 0.9645, 	recall: 0.6798, 	specificity: 0.9936, 	f1: 0.7975
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 146: 100%|██████████| 6507/6507 [10:01<00:00, 10.82it/s, loss=0.0011]
Train Epoch 146 ==> 	accuracy: 0.9223, 	precision: 0.9998, 	recall: 0.8448, 	specificity: 0.9998, 	f1: 0.9158
Test Epoch 146: 100%|██████████| 1768/1768 [01:05<00:00, 26.97it/s, loss=0.171]
Test Epoch 146 ==> 	accuracy: 0.9680, 	precision: 0.9767, 	recall: 0.8640, 	specificity: 0.9947, 	f1: 0.9169
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 147: 100%|██████████| 6507/6507 [10:04<00:00, 10.77it/s, loss=0.0092]
Train Epoch 147 ==> 	accuracy: 0.9234, 	precision: 0.9998, 	recall: 0.8469, 	specificity: 0.9998, 	f1: 0.9170
Test Epoch 147: 100%|██████████| 1768/1768 [01:04<00:00, 27.49it/s, loss=0.0985]
Test Epoch 147 ==> 	accuracy: 0.9680, 	precision: 0.9728, 	recall: 0.8680, 	specificity: 0.9938, 	f1: 0.9174
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 148: 100%|██████████| 6507/6507 [10:05<00:00, 10.74it/s, loss=0.002]
Train Epoch 148 ==> 	accuracy: 0.9250, 	precision: 0.9998, 	recall: 0.8501, 	specificity: 0.9998, 	f1: 0.9189
Test Epoch 148: 100%|██████████| 1768/1768 [01:05<00:00, 27.05it/s, loss=0.086]
Test Epoch 148 ==> 	accuracy: 0.9684, 	precision: 0.9726, 	recall: 0.8702, 	specificity: 0.9937, 	f1: 0.9186
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 149: 100%|██████████| 6507/6507 [09:55<00:00, 10.92it/s, loss=0.0032]
Train Epoch 149 ==> 	accuracy: 0.9236, 	precision: 0.9998, 	recall: 0.8473, 	specificity: 0.9998, 	f1: 0.9173
Test Epoch 149: 100%|██████████| 1768/1768 [01:06<00:00, 26.63it/s, loss=0.104]
Test Epoch 149 ==> 	accuracy: 0.9681, 	precision: 0.9736, 	recall: 0.8677, 	specificity: 0.9939, 	f1: 0.9176
Adjusting learning rate of group 0 to 5.8150e-06.

进程已结束，退出代码为 0

'''


'''
mha ab
'../model_save_sigBlock4_focalWithMs_deformable_7mer_ab_mha'
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6507/6507 [06:25<00:00, 16.90it/s, loss=0.112]
Train Epoch 0 ==> 	accuracy: 0.6557, 	precision: 0.9962, 	recall: 0.3125, 	specificity: 0.9988, 	f1: 0.4758
Test Epoch 0: 100%|██████████| 1768/1768 [00:46<00:00, 38.01it/s, loss=0.888]
Test Epoch 0 ==> 	accuracy: 0.9169, 	precision: 0.9578, 	recall: 0.6213, 	specificity: 0.9930, 	f1: 0.7537
Train Epoch 1: 100%|██████████| 6507/6507 [07:43<00:00, 14.03it/s, loss=0.318]
Train Epoch 1 ==> 	accuracy: 0.7579, 	precision: 0.9977, 	recall: 0.5170, 	specificity: 0.9988, 	f1: 0.6810
Test Epoch 1: 100%|██████████| 1768/1768 [00:49<00:00, 35.60it/s, loss=0.572]
Test Epoch 1 ==> 	accuracy: 0.9246, 	precision: 0.9635, 	recall: 0.6562, 	specificity: 0.9936, 	f1: 0.7807
Train Epoch 2: 100%|██████████| 6507/6507 [07:44<00:00, 14.00it/s, loss=0.0753]
Train Epoch 2 ==> 	accuracy: 0.7822, 	precision: 0.9982, 	recall: 0.5654, 	specificity: 0.9990, 	f1: 0.7219
Test Epoch 2: 100%|██████████| 1768/1768 [00:45<00:00, 38.82it/s, loss=0.328]
Test Epoch 2 ==> 	accuracy: 0.9244, 	precision: 0.9853, 	recall: 0.6400, 	specificity: 0.9975, 	f1: 0.7759
Train Epoch 3: 100%|██████████| 6507/6507 [07:53<00:00, 13.75it/s, loss=0.12]
Train Epoch 3 ==> 	accuracy: 0.7916, 	precision: 0.9983, 	recall: 0.5842, 	specificity: 0.9990, 	f1: 0.7371
Test Epoch 3: 100%|██████████| 1768/1768 [00:47<00:00, 37.36it/s, loss=0.247]
Test Epoch 3 ==> 	accuracy: 0.9262, 	precision: 0.9761, 	recall: 0.6554, 	specificity: 0.9959, 	f1: 0.7842
Train Epoch 4: 100%|██████████| 6507/6507 [07:33<00:00, 14.34it/s, loss=0.117]
Train Epoch 4 ==> 	accuracy: 0.8016, 	precision: 0.9985, 	recall: 0.6041, 	specificity: 0.9991, 	f1: 0.7528
Test Epoch 4: 100%|██████████| 1768/1768 [00:48<00:00, 36.12it/s, loss=0.415]
Test Epoch 4 ==> 	accuracy: 0.9229, 	precision: 0.9815, 	recall: 0.6351, 	specificity: 0.9969, 	f1: 0.7712
Train Epoch 5: 100%|██████████| 6507/6507 [07:57<00:00, 13.63it/s, loss=0.0251]
Train Epoch 5 ==> 	accuracy: 0.8065, 	precision: 0.9986, 	recall: 0.6139, 	specificity: 0.9991, 	f1: 0.7604
Test Epoch 5: 100%|██████████| 1768/1768 [00:51<00:00, 34.62it/s, loss=0.202]
Test Epoch 5 ==> 	accuracy: 0.9319, 	precision: 0.9845, 	recall: 0.6779, 	specificity: 0.9972, 	f1: 0.8029
Train Epoch 6: 100%|██████████| 6507/6507 [07:56<00:00, 13.64it/s, loss=0.0842]
Train Epoch 6 ==> 	accuracy: 0.8120, 	precision: 0.9987, 	recall: 0.6247, 	specificity: 0.9992, 	f1: 0.7687
Test Epoch 6: 100%|██████████| 1768/1768 [00:50<00:00, 35.32it/s, loss=0.175]
Test Epoch 6 ==> 	accuracy: 0.9327, 	precision: 0.9710, 	recall: 0.6918, 	specificity: 0.9947, 	f1: 0.8079
Train Epoch 7: 100%|██████████| 6507/6507 [07:38<00:00, 14.20it/s, loss=0.0884]
Train Epoch 7 ==> 	accuracy: 0.8202, 	precision: 0.9988, 	recall: 0.6412, 	specificity: 0.9992, 	f1: 0.7810
Test Epoch 7: 100%|██████████| 1768/1768 [00:50<00:00, 35.02it/s, loss=0.463]
Test Epoch 7 ==> 	accuracy: 0.9288, 	precision: 0.9798, 	recall: 0.6657, 	specificity: 0.9965, 	f1: 0.7927
Train Epoch 8: 100%|██████████| 6507/6507 [07:45<00:00, 13.97it/s, loss=0.0803]
Train Epoch 8 ==> 	accuracy: 0.8153, 	precision: 0.9988, 	recall: 0.6313, 	specificity: 0.9993, 	f1: 0.7736
Test Epoch 8: 100%|██████████| 1768/1768 [00:50<00:00, 35.23it/s, loss=0.797]
Test Epoch 8 ==> 	accuracy: 0.9292, 	precision: 0.9857, 	recall: 0.6633, 	specificity: 0.9975, 	f1: 0.7930
Train Epoch 9: 100%|██████████| 6507/6507 [07:45<00:00, 13.99it/s, loss=0.107]
Train Epoch 9 ==> 	accuracy: 0.8224, 	precision: 0.9989, 	recall: 0.6454, 	specificity: 0.9993, 	f1: 0.7842
Test Epoch 9: 100%|██████████| 1768/1768 [00:48<00:00, 36.56it/s, loss=0.128]
Test Epoch 9 ==> 	accuracy: 0.9338, 	precision: 0.9863, 	recall: 0.6858, 	specificity: 0.9975, 	f1: 0.8090
Train Epoch 10: 100%|██████████| 6507/6507 [07:46<00:00, 13.95it/s, loss=0.0956]
Train Epoch 10 ==> 	accuracy: 0.8228, 	precision: 0.9989, 	recall: 0.6462, 	specificity: 0.9993, 	f1: 0.7847
Test Epoch 10: 100%|██████████| 1768/1768 [00:45<00:00, 39.16it/s, loss=0.672]
Test Epoch 10 ==> 	accuracy: 0.9343, 	precision: 0.9788, 	recall: 0.6940, 	specificity: 0.9961, 	f1: 0.8122
Train Epoch 11: 100%|██████████| 6507/6507 [07:41<00:00, 14.09it/s, loss=0.23]
Train Epoch 11 ==> 	accuracy: 0.8274, 	precision: 0.9990, 	recall: 0.6555, 	specificity: 0.9993, 	f1: 0.7916
Test Epoch 11: 100%|██████████| 1768/1768 [00:49<00:00, 35.54it/s, loss=0.259]
Test Epoch 11 ==> 	accuracy: 0.9349, 	precision: 0.9824, 	recall: 0.6942, 	specificity: 0.9968, 	f1: 0.8136
Train Epoch 12: 100%|██████████| 6507/6507 [07:56<00:00, 13.67it/s, loss=0.0638]
Train Epoch 12 ==> 	accuracy: 0.8257, 	precision: 0.9990, 	recall: 0.6521, 	specificity: 0.9993, 	f1: 0.7891
Test Epoch 12: 100%|██████████| 1768/1768 [00:46<00:00, 37.88it/s, loss=0.126]
Test Epoch 12 ==> 	accuracy: 0.9352, 	precision: 0.9928, 	recall: 0.6884, 	specificity: 0.9987, 	f1: 0.8131
Train Epoch 13: 100%|██████████| 6507/6507 [07:55<00:00, 13.70it/s, loss=0.177]
Train Epoch 13 ==> 	accuracy: 0.8315, 	precision: 0.9990, 	recall: 0.6636, 	specificity: 0.9994, 	f1: 0.7975
Test Epoch 13: 100%|██████████| 1768/1768 [00:47<00:00, 36.94it/s, loss=0.133]
Test Epoch 13 ==> 	accuracy: 0.9360, 	precision: 0.9895, 	recall: 0.6946, 	specificity: 0.9981, 	f1: 0.8163
Train Epoch 14: 100%|██████████| 6507/6507 [07:41<00:00, 14.11it/s, loss=0.0945]
Train Epoch 14 ==> 	accuracy: 0.8356, 	precision: 0.9991, 	recall: 0.6719, 	specificity: 0.9994, 	f1: 0.8034
Test Epoch 14: 100%|██████████| 1768/1768 [00:47<00:00, 37.23it/s, loss=0.593]
Test Epoch 14 ==> 	accuracy: 0.9398, 	precision: 0.9715, 	recall: 0.7268, 	specificity: 0.9945, 	f1: 0.8315
Train Epoch 15: 100%|██████████| 6507/6507 [07:35<00:00, 14.28it/s, loss=0.0889]
Train Epoch 15 ==> 	accuracy: 0.8329, 	precision: 0.9991, 	recall: 0.6665, 	specificity: 0.9994, 	f1: 0.7996
Test Epoch 15: 100%|██████████| 1768/1768 [00:51<00:00, 34.19it/s, loss=0.161]
Test Epoch 15 ==> 	accuracy: 0.9398, 	precision: 0.9784, 	recall: 0.7219, 	specificity: 0.9959, 	f1: 0.8308
Train Epoch 16: 100%|██████████| 6507/6507 [07:41<00:00, 14.10it/s, loss=0.0219]
Train Epoch 16 ==> 	accuracy: 0.8350, 	precision: 0.9991, 	recall: 0.6706, 	specificity: 0.9994, 	f1: 0.8025
Test Epoch 16: 100%|██████████| 1768/1768 [00:50<00:00, 35.05it/s, loss=0.445]
Test Epoch 16 ==> 	accuracy: 0.9416, 	precision: 0.9792, 	recall: 0.7300, 	specificity: 0.9960, 	f1: 0.8364
Train Epoch 17: 100%|██████████| 6507/6507 [07:44<00:00, 14.02it/s, loss=0.0974]
Train Epoch 17 ==> 	accuracy: 0.8390, 	precision: 0.9991, 	recall: 0.6786, 	specificity: 0.9994, 	f1: 0.8083
Test Epoch 17: 100%|██████████| 1768/1768 [00:46<00:00, 37.86it/s, loss=0.331]
Test Epoch 17 ==> 	accuracy: 0.9365, 	precision: 0.9809, 	recall: 0.7034, 	specificity: 0.9965, 	f1: 0.8193
Train Epoch 18: 100%|██████████| 6507/6507 [07:35<00:00, 14.29it/s, loss=0.0731]
Train Epoch 18 ==> 	accuracy: 0.8405, 	precision: 0.9992, 	recall: 0.6815, 	specificity: 0.9994, 	f1: 0.8103
Test Epoch 18: 100%|██████████| 1768/1768 [00:43<00:00, 40.22it/s, loss=0.471]
Test Epoch 18 ==> 	accuracy: 0.9432, 	precision: 0.9818, 	recall: 0.7361, 	specificity: 0.9965, 	f1: 0.8413
Train Epoch 19: 100%|██████████| 6507/6507 [07:42<00:00, 14.06it/s, loss=0.0526]
Train Epoch 19 ==> 	accuracy: 0.8398, 	precision: 0.9992, 	recall: 0.6801, 	specificity: 0.9994, 	f1: 0.8093
Test Epoch 19: 100%|██████████| 1768/1768 [00:45<00:00, 38.48it/s, loss=0.135]
Test Epoch 19 ==> 	accuracy: 0.9408, 	precision: 0.9868, 	recall: 0.7201, 	specificity: 0.9975, 	f1: 0.8326
Train Epoch 20: 100%|██████████| 6507/6507 [07:35<00:00, 14.28it/s, loss=0.0379]
Train Epoch 20 ==> 	accuracy: 0.8408, 	precision: 0.9992, 	recall: 0.6822, 	specificity: 0.9994, 	f1: 0.8108
Test Epoch 20: 100%|██████████| 1768/1768 [00:46<00:00, 38.06it/s, loss=0.405]
Test Epoch 20 ==> 	accuracy: 0.9433, 	precision: 0.9819, 	recall: 0.7364, 	specificity: 0.9965, 	f1: 0.8416
Train Epoch 21: 100%|██████████| 6507/6507 [07:41<00:00, 14.11it/s, loss=0.073]
Train Epoch 21 ==> 	accuracy: 0.8457, 	precision: 0.9992, 	recall: 0.6919, 	specificity: 0.9994, 	f1: 0.8176
Test Epoch 21: 100%|██████████| 1768/1768 [00:45<00:00, 39.14it/s, loss=0.255]
Test Epoch 21 ==> 	accuracy: 0.9443, 	precision: 0.9781, 	recall: 0.7443, 	specificity: 0.9957, 	f1: 0.8454
Train Epoch 22: 100%|██████████| 6507/6507 [07:40<00:00, 14.12it/s, loss=0.0425]
Train Epoch 22 ==> 	accuracy: 0.8455, 	precision: 0.9992, 	recall: 0.6915, 	specificity: 0.9995, 	f1: 0.8174
Test Epoch 22: 100%|██████████| 1768/1768 [00:46<00:00, 38.02it/s, loss=0.328]
Test Epoch 22 ==> 	accuracy: 0.9411, 	precision: 0.9824, 	recall: 0.7251, 	specificity: 0.9967, 	f1: 0.8343
Train Epoch 23: 100%|██████████| 6507/6507 [07:36<00:00, 14.24it/s, loss=0.0643]
Train Epoch 23 ==> 	accuracy: 0.8434, 	precision: 0.9992, 	recall: 0.6873, 	specificity: 0.9995, 	f1: 0.8144
Test Epoch 23: 100%|██████████| 1768/1768 [00:44<00:00, 39.87it/s, loss=0.211]
Test Epoch 23 ==> 	accuracy: 0.9419, 	precision: 0.9837, 	recall: 0.7280, 	specificity: 0.9969, 	f1: 0.8367
Train Epoch 24: 100%|██████████| 6507/6507 [07:43<00:00, 14.04it/s, loss=0.0998]
Train Epoch 24 ==> 	accuracy: 0.8475, 	precision: 0.9993, 	recall: 0.6955, 	specificity: 0.9995, 	f1: 0.8201
Test Epoch 24: 100%|██████████| 1768/1768 [00:45<00:00, 38.78it/s, loss=0.189]
Test Epoch 24 ==> 	accuracy: 0.9438, 	precision: 0.9875, 	recall: 0.7344, 	specificity: 0.9976, 	f1: 0.8424
Train Epoch 25: 100%|██████████| 6507/6507 [07:44<00:00, 14.01it/s, loss=0.11]
Train Epoch 25 ==> 	accuracy: 0.8470, 	precision: 0.9992, 	recall: 0.6945, 	specificity: 0.9995, 	f1: 0.8195
Test Epoch 25: 100%|██████████| 1768/1768 [00:46<00:00, 38.40it/s, loss=0.426]
Test Epoch 25 ==> 	accuracy: 0.9460, 	precision: 0.9833, 	recall: 0.7487, 	specificity: 0.9967, 	f1: 0.8501
Train Epoch 26: 100%|██████████| 6507/6507 [07:32<00:00, 14.38it/s, loss=0.0708]
Train Epoch 26 ==> 	accuracy: 0.8494, 	precision: 0.9993, 	recall: 0.6992, 	specificity: 0.9995, 	f1: 0.8227
Test Epoch 26: 100%|██████████| 1768/1768 [00:46<00:00, 38.01it/s, loss=0.235]
Test Epoch 26 ==> 	accuracy: 0.9456, 	precision: 0.9800, 	recall: 0.7496, 	specificity: 0.9961, 	f1: 0.8494
Train Epoch 27: 100%|██████████| 6507/6507 [07:37<00:00, 14.21it/s, loss=0.0306]
Train Epoch 27 ==> 	accuracy: 0.8515, 	precision: 0.9993, 	recall: 0.7034, 	specificity: 0.9995, 	f1: 0.8256
Test Epoch 27: 100%|██████████| 1768/1768 [00:45<00:00, 39.12it/s, loss=0.167]
Test Epoch 27 ==> 	accuracy: 0.9482, 	precision: 0.9875, 	recall: 0.7566, 	specificity: 0.9975, 	f1: 0.8568
Train Epoch 28: 100%|██████████| 6507/6507 [07:34<00:00, 14.31it/s, loss=0.0407]
Train Epoch 28 ==> 	accuracy: 0.8504, 	precision: 0.9993, 	recall: 0.7013, 	specificity: 0.9995, 	f1: 0.8242
Test Epoch 28: 100%|██████████| 1768/1768 [00:45<00:00, 39.06it/s, loss=0.408]
Test Epoch 28 ==> 	accuracy: 0.9443, 	precision: 0.9849, 	recall: 0.7392, 	specificity: 0.9971, 	f1: 0.8445
Train Epoch 29: 100%|██████████| 6507/6507 [07:29<00:00, 14.48it/s, loss=0.0379]
Train Epoch 29 ==> 	accuracy: 0.8509, 	precision: 0.9993, 	recall: 0.7023, 	specificity: 0.9995, 	f1: 0.8249
Test Epoch 29: 100%|██████████| 1768/1768 [00:48<00:00, 36.53it/s, loss=0.328]
Test Epoch 29 ==> 	accuracy: 0.9468, 	precision: 0.9786, 	recall: 0.7565, 	specificity: 0.9957, 	f1: 0.8533
Train Epoch 30: 100%|██████████| 6507/6507 [07:21<00:00, 14.75it/s, loss=0.162]
Train Epoch 30 ==> 	accuracy: 0.8534, 	precision: 0.9993, 	recall: 0.7074, 	specificity: 0.9995, 	f1: 0.8284
Test Epoch 30: 100%|██████████| 1768/1768 [00:44<00:00, 39.77it/s, loss=0.224]
Test Epoch 30 ==> 	accuracy: 0.9468, 	precision: 0.9856, 	recall: 0.7510, 	specificity: 0.9972, 	f1: 0.8524
Train Epoch 31: 100%|██████████| 6507/6507 [07:28<00:00, 14.51it/s, loss=0.106]
Train Epoch 31 ==> 	accuracy: 0.8550, 	precision: 0.9993, 	recall: 0.7104, 	specificity: 0.9995, 	f1: 0.8305
Test Epoch 31: 100%|██████████| 1768/1768 [00:48<00:00, 36.45it/s, loss=0.246]
Test Epoch 31 ==> 	accuracy: 0.9483, 	precision: 0.9824, 	recall: 0.7608, 	specificity: 0.9965, 	f1: 0.8575
Train Epoch 32: 100%|██████████| 6507/6507 [07:41<00:00, 14.10it/s, loss=0.0516]
Train Epoch 32 ==> 	accuracy: 0.8519, 	precision: 0.9994, 	recall: 0.7042, 	specificity: 0.9995, 	f1: 0.8262
Test Epoch 32: 100%|██████████| 1768/1768 [00:44<00:00, 39.82it/s, loss=0.129]
Test Epoch 32 ==> 	accuracy: 0.9476, 	precision: 0.9865, 	recall: 0.7544, 	specificity: 0.9973, 	f1: 0.8549
Train Epoch 33: 100%|██████████| 6507/6507 [07:27<00:00, 14.53it/s, loss=0.148]
Train Epoch 33 ==> 	accuracy: 0.8531, 	precision: 0.9993, 	recall: 0.7067, 	specificity: 0.9995, 	f1: 0.8279
Test Epoch 33: 100%|██████████| 1768/1768 [00:45<00:00, 38.92it/s, loss=0.764]
Test Epoch 33 ==> 	accuracy: 0.9452, 	precision: 0.9866, 	recall: 0.7421, 	specificity: 0.9974, 	f1: 0.8471
Train Epoch 34: 100%|██████████| 6507/6507 [07:33<00:00, 14.35it/s, loss=0.0628]
Train Epoch 34 ==> 	accuracy: 0.8571, 	precision: 0.9994, 	recall: 0.7146, 	specificity: 0.9996, 	f1: 0.8333
Test Epoch 34: 100%|██████████| 1768/1768 [00:45<00:00, 38.66it/s, loss=0.147]
Test Epoch 34 ==> 	accuracy: 0.9494, 	precision: 0.9835, 	recall: 0.7652, 	specificity: 0.9967, 	f1: 0.8607
Train Epoch 35: 100%|██████████| 6507/6507 [07:40<00:00, 14.13it/s, loss=0.0962]
Train Epoch 35 ==> 	accuracy: 0.8578, 	precision: 0.9994, 	recall: 0.7161, 	specificity: 0.9995, 	f1: 0.8343
Test Epoch 35: 100%|██████████| 1768/1768 [00:45<00:00, 39.27it/s, loss=0.133]
Test Epoch 35 ==> 	accuracy: 0.9464, 	precision: 0.9851, 	recall: 0.7491, 	specificity: 0.9971, 	f1: 0.8510
Train Epoch 36: 100%|██████████| 6507/6507 [07:32<00:00, 14.37it/s, loss=0.117]
Train Epoch 36 ==> 	accuracy: 0.8580, 	precision: 0.9994, 	recall: 0.7164, 	specificity: 0.9995, 	f1: 0.8345
Test Epoch 36: 100%|██████████| 1768/1768 [00:46<00:00, 37.77it/s, loss=0.604]
Test Epoch 36 ==> 	accuracy: 0.9483, 	precision: 0.9893, 	recall: 0.7555, 	specificity: 0.9979, 	f1: 0.8567
Train Epoch 37: 100%|██████████| 6507/6507 [07:42<00:00, 14.06it/s, loss=0.0788]
Train Epoch 37 ==> 	accuracy: 0.8574, 	precision: 0.9994, 	recall: 0.7153, 	specificity: 0.9995, 	f1: 0.8338
Test Epoch 37: 100%|██████████| 1768/1768 [00:48<00:00, 36.13it/s, loss=0.303]
Test Epoch 37 ==> 	accuracy: 0.9478, 	precision: 0.9834, 	recall: 0.7574, 	specificity: 0.9967, 	f1: 0.8557
Train Epoch 38: 100%|██████████| 6507/6507 [07:31<00:00, 14.42it/s, loss=0.108]
Train Epoch 38 ==> 	accuracy: 0.8602, 	precision: 0.9994, 	recall: 0.7208, 	specificity: 0.9996, 	f1: 0.8376
Test Epoch 38: 100%|██████████| 1768/1768 [00:48<00:00, 36.69it/s, loss=0.0487]
Test Epoch 38 ==> 	accuracy: 0.9490, 	precision: 0.9801, 	recall: 0.7664, 	specificity: 0.9960, 	f1: 0.8601
Train Epoch 39: 100%|██████████| 6507/6507 [07:26<00:00, 14.57it/s, loss=0.0787]
Train Epoch 39 ==> 	accuracy: 0.8612, 	precision: 0.9994, 	recall: 0.7229, 	specificity: 0.9995, 	f1: 0.8390
Test Epoch 39: 100%|██████████| 1768/1768 [00:46<00:00, 38.40it/s, loss=0.285]
Test Epoch 39 ==> 	accuracy: 0.9506, 	precision: 0.9811, 	recall: 0.7734, 	specificity: 0.9962, 	f1: 0.8650
Train Epoch 40: 100%|██████████| 6507/6507 [07:31<00:00, 14.41it/s, loss=0.149]
Train Epoch 40 ==> 	accuracy: 0.8609, 	precision: 0.9994, 	recall: 0.7223, 	specificity: 0.9996, 	f1: 0.8386
Test Epoch 40: 100%|██████████| 1768/1768 [00:47<00:00, 37.40it/s, loss=0.181]
Test Epoch 40 ==> 	accuracy: 0.9464, 	precision: 0.9852, 	recall: 0.7490, 	specificity: 0.9971, 	f1: 0.8510
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 41: 100%|██████████| 6507/6507 [07:21<00:00, 14.73it/s, loss=0.102]
Train Epoch 41 ==> 	accuracy: 0.8606, 	precision: 0.9994, 	recall: 0.7217, 	specificity: 0.9996, 	f1: 0.8381
Test Epoch 41: 100%|██████████| 1768/1768 [00:45<00:00, 38.57it/s, loss=0.134]
Test Epoch 41 ==> 	accuracy: 0.9491, 	precision: 0.9847, 	recall: 0.7629, 	specificity: 0.9970, 	f1: 0.8597
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 42: 100%|██████████| 6507/6507 [07:04<00:00, 15.33it/s, loss=0.0104]
Train Epoch 42 ==> 	accuracy: 0.8619, 	precision: 0.9995, 	recall: 0.7242, 	specificity: 0.9996, 	f1: 0.8398
Test Epoch 42: 100%|██████████| 1768/1768 [00:49<00:00, 35.73it/s, loss=0.242]
Test Epoch 42 ==> 	accuracy: 0.9477, 	precision: 0.9867, 	recall: 0.7543, 	specificity: 0.9974, 	f1: 0.8550
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 43: 100%|██████████| 6507/6507 [07:20<00:00, 14.79it/s, loss=0.0193]
Train Epoch 43 ==> 	accuracy: 0.8622, 	precision: 0.9995, 	recall: 0.7249, 	specificity: 0.9996, 	f1: 0.8403
Test Epoch 43: 100%|██████████| 1768/1768 [00:46<00:00, 37.62it/s, loss=0.409]
Test Epoch 43 ==> 	accuracy: 0.9490, 	precision: 0.9820, 	recall: 0.7648, 	specificity: 0.9964, 	f1: 0.8599
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 44: 100%|██████████| 6507/6507 [07:26<00:00, 14.59it/s, loss=0.0041]
Train Epoch 44 ==> 	accuracy: 0.8651, 	precision: 0.9994, 	recall: 0.7307, 	specificity: 0.9996, 	f1: 0.8442
Test Epoch 44: 100%|██████████| 1768/1768 [00:48<00:00, 36.55it/s, loss=0.102]
Test Epoch 44 ==> 	accuracy: 0.9489, 	precision: 0.9780, 	recall: 0.7672, 	specificity: 0.9956, 	f1: 0.8599
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 45: 100%|██████████| 6507/6507 [07:15<00:00, 14.93it/s, loss=0.0081]
Train Epoch 45 ==> 	accuracy: 0.8622, 	precision: 0.9994, 	recall: 0.7248, 	specificity: 0.9996, 	f1: 0.8402
Test Epoch 45: 100%|██████████| 1768/1768 [00:48<00:00, 36.61it/s, loss=0.148]
Test Epoch 45 ==> 	accuracy: 0.9512, 	precision: 0.9857, 	recall: 0.7726, 	specificity: 0.9971, 	f1: 0.8662
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 46: 100%|██████████| 6507/6507 [07:17<00:00, 14.86it/s, loss=0.013]
Train Epoch 46 ==> 	accuracy: 0.8651, 	precision: 0.9994, 	recall: 0.7306, 	specificity: 0.9996, 	f1: 0.8441
Test Epoch 46: 100%|██████████| 1768/1768 [00:42<00:00, 41.98it/s, loss=0.171]
Test Epoch 46 ==> 	accuracy: 0.9503, 	precision: 0.9877, 	recall: 0.7666, 	specificity: 0.9975, 	f1: 0.8632
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 47: 100%|██████████| 6507/6507 [07:06<00:00, 15.25it/s, loss=0.0203]
Train Epoch 47 ==> 	accuracy: 0.8643, 	precision: 0.9994, 	recall: 0.7290, 	specificity: 0.9996, 	f1: 0.8430
Test Epoch 47: 100%|██████████| 1768/1768 [00:48<00:00, 36.59it/s, loss=0.13]
Test Epoch 47 ==> 	accuracy: 0.9491, 	precision: 0.9715, 	recall: 0.7738, 	specificity: 0.9942, 	f1: 0.8615
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 48: 100%|██████████| 6507/6507 [07:12<00:00, 15.06it/s, loss=0.029]
Train Epoch 48 ==> 	accuracy: 0.8693, 	precision: 0.9995, 	recall: 0.7390, 	specificity: 0.9996, 	f1: 0.8497
Test Epoch 48: 100%|██████████| 1768/1768 [00:44<00:00, 39.44it/s, loss=0.0809]
Test Epoch 48 ==> 	accuracy: 0.9515, 	precision: 0.9796, 	recall: 0.7793, 	specificity: 0.9958, 	f1: 0.8680
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 49: 100%|██████████| 6507/6507 [06:59<00:00, 15.50it/s, loss=0.0202]
Train Epoch 49 ==> 	accuracy: 0.8665, 	precision: 0.9995, 	recall: 0.7334, 	specificity: 0.9996, 	f1: 0.8460
Test Epoch 49: 100%|██████████| 1768/1768 [00:51<00:00, 34.44it/s, loss=0.068]
Test Epoch 49 ==> 	accuracy: 0.9524, 	precision: 0.9903, 	recall: 0.7747, 	specificity: 0.9980, 	f1: 0.8693
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 50: 100%|██████████| 6507/6507 [07:16<00:00, 14.89it/s, loss=0.106]
Train Epoch 50 ==> 	accuracy: 0.8718, 	precision: 0.9995, 	recall: 0.7439, 	specificity: 0.9996, 	f1: 0.8530
Test Epoch 50: 100%|██████████| 1768/1768 [00:45<00:00, 39.19it/s, loss=1.24]
Test Epoch 50 ==> 	accuracy: 0.9536, 	precision: 0.9853, 	recall: 0.7851, 	specificity: 0.9970, 	f1: 0.8739
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 51: 100%|██████████| 6507/6507 [07:21<00:00, 14.75it/s, loss=0.0111]
Train Epoch 51 ==> 	accuracy: 0.8714, 	precision: 0.9995, 	recall: 0.7432, 	specificity: 0.9996, 	f1: 0.8525
Test Epoch 51: 100%|██████████| 1768/1768 [00:45<00:00, 38.46it/s, loss=0.129]
Test Epoch 51 ==> 	accuracy: 0.9543, 	precision: 0.9857, 	recall: 0.7881, 	specificity: 0.9970, 	f1: 0.8759
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 52: 100%|██████████| 6507/6507 [07:36<00:00, 14.24it/s, loss=0.0719]
Train Epoch 52 ==> 	accuracy: 0.8738, 	precision: 0.9995, 	recall: 0.7479, 	specificity: 0.9996, 	f1: 0.8556
Test Epoch 52: 100%|██████████| 1768/1768 [00:45<00:00, 39.05it/s, loss=0.104]
Test Epoch 52 ==> 	accuracy: 0.9544, 	precision: 0.9839, 	recall: 0.7901, 	specificity: 0.9967, 	f1: 0.8764
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 53: 100%|██████████| 6507/6507 [07:21<00:00, 14.74it/s, loss=0.151]
Train Epoch 53 ==> 	accuracy: 0.8742, 	precision: 0.9995, 	recall: 0.7487, 	specificity: 0.9996, 	f1: 0.8561
Test Epoch 53: 100%|██████████| 1768/1768 [00:49<00:00, 36.07it/s, loss=1.37]
Test Epoch 53 ==> 	accuracy: 0.9556, 	precision: 0.9822, 	recall: 0.7975, 	specificity: 0.9963, 	f1: 0.8803
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 54: 100%|██████████| 6507/6507 [07:11<00:00, 15.07it/s, loss=0.008]
Train Epoch 54 ==> 	accuracy: 0.8753, 	precision: 0.9995, 	recall: 0.7509, 	specificity: 0.9996, 	f1: 0.8575
Test Epoch 54: 100%|██████████| 1768/1768 [00:49<00:00, 35.38it/s, loss=1.32]
Test Epoch 54 ==> 	accuracy: 0.9521, 	precision: 0.9864, 	recall: 0.7765, 	specificity: 0.9973, 	f1: 0.8690
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 55: 100%|██████████| 6507/6507 [07:14<00:00, 14.98it/s, loss=0.0076]
Train Epoch 55 ==> 	accuracy: 0.8749, 	precision: 0.9995, 	recall: 0.7502, 	specificity: 0.9997, 	f1: 0.8571
Test Epoch 55: 100%|██████████| 1768/1768 [00:50<00:00, 35.34it/s, loss=0.184]
Test Epoch 55 ==> 	accuracy: 0.9547, 	precision: 0.9850, 	recall: 0.7904, 	specificity: 0.9969, 	f1: 0.8770
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 56: 100%|██████████| 6507/6507 [07:20<00:00, 14.79it/s, loss=0.0064]
Train Epoch 56 ==> 	accuracy: 0.8752, 	precision: 0.9996, 	recall: 0.7508, 	specificity: 0.9997, 	f1: 0.8575
Test Epoch 56: 100%|██████████| 1768/1768 [00:44<00:00, 39.72it/s, loss=0.103]
Test Epoch 56 ==> 	accuracy: 0.9516, 	precision: 0.9847, 	recall: 0.7753, 	specificity: 0.9969, 	f1: 0.8676
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 57: 100%|██████████| 6507/6507 [07:14<00:00, 14.96it/s, loss=0.111]
Train Epoch 57 ==> 	accuracy: 0.8800, 	precision: 0.9996, 	recall: 0.7604, 	specificity: 0.9997, 	f1: 0.8637
Test Epoch 57: 100%|██████████| 1768/1768 [00:44<00:00, 39.53it/s, loss=0.647]
Test Epoch 57 ==> 	accuracy: 0.9560, 	precision: 0.9846, 	recall: 0.7974, 	specificity: 0.9968, 	f1: 0.8811
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 58: 100%|██████████| 6507/6507 [07:14<00:00, 14.99it/s, loss=0.0169]
Train Epoch 58 ==> 	accuracy: 0.8757, 	precision: 0.9995, 	recall: 0.7517, 	specificity: 0.9997, 	f1: 0.8581
Test Epoch 58: 100%|██████████| 1768/1768 [00:46<00:00, 38.05it/s, loss=0.214]
Test Epoch 58 ==> 	accuracy: 0.9557, 	precision: 0.9813, 	recall: 0.7984, 	specificity: 0.9961, 	f1: 0.8805
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 59: 100%|██████████| 6507/6507 [07:18<00:00, 14.83it/s, loss=0.0063]
Train Epoch 59 ==> 	accuracy: 0.8793, 	precision: 0.9996, 	recall: 0.7588, 	specificity: 0.9997, 	f1: 0.8627
Test Epoch 59: 100%|██████████| 1768/1768 [00:45<00:00, 38.94it/s, loss=0.468]
Test Epoch 59 ==> 	accuracy: 0.9565, 	precision: 0.9772, 	recall: 0.8062, 	specificity: 0.9952, 	f1: 0.8835
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 60: 100%|██████████| 6507/6507 [07:12<00:00, 15.03it/s, loss=0.0341]
Train Epoch 60 ==> 	accuracy: 0.8800, 	precision: 0.9996, 	recall: 0.7603, 	specificity: 0.9997, 	f1: 0.8637
Test Epoch 60: 100%|██████████| 1768/1768 [00:44<00:00, 39.48it/s, loss=0.21]
Test Epoch 60 ==> 	accuracy: 0.9549, 	precision: 0.9813, 	recall: 0.7947, 	specificity: 0.9961, 	f1: 0.8782
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 61: 100%|██████████| 6507/6507 [07:06<00:00, 15.25it/s, loss=0.0168]
Train Epoch 61 ==> 	accuracy: 0.8836, 	precision: 0.9996, 	recall: 0.7676, 	specificity: 0.9997, 	f1: 0.8683
Test Epoch 61: 100%|██████████| 1768/1768 [00:45<00:00, 39.01it/s, loss=0.354]
Test Epoch 61 ==> 	accuracy: 0.9568, 	precision: 0.9822, 	recall: 0.8034, 	specificity: 0.9962, 	f1: 0.8839
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 62: 100%|██████████| 6507/6507 [07:13<00:00, 15.02it/s, loss=0.0114]
Train Epoch 62 ==> 	accuracy: 0.8780, 	precision: 0.9996, 	recall: 0.7563, 	specificity: 0.9997, 	f1: 0.8611
Test Epoch 62: 100%|██████████| 1768/1768 [00:43<00:00, 40.53it/s, loss=0.182]
Test Epoch 62 ==> 	accuracy: 0.9583, 	precision: 0.9800, 	recall: 0.8128, 	specificity: 0.9957, 	f1: 0.8886
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 63: 100%|██████████| 6507/6507 [07:22<00:00, 14.70it/s, loss=0.0387]
Train Epoch 63 ==> 	accuracy: 0.8825, 	precision: 0.9996, 	recall: 0.7653, 	specificity: 0.9997, 	f1: 0.8669
Test Epoch 63: 100%|██████████| 1768/1768 [00:45<00:00, 38.89it/s, loss=0.152]
Test Epoch 63 ==> 	accuracy: 0.9553, 	precision: 0.9769, 	recall: 0.8004, 	specificity: 0.9951, 	f1: 0.8799
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 64: 100%|██████████| 6507/6507 [07:17<00:00, 14.86it/s, loss=0.0218]
Train Epoch 64 ==> 	accuracy: 0.8827, 	precision: 0.9996, 	recall: 0.7658, 	specificity: 0.9997, 	f1: 0.8672
Test Epoch 64: 100%|██████████| 1768/1768 [00:45<00:00, 38.89it/s, loss=0.0457]
Test Epoch 64 ==> 	accuracy: 0.9574, 	precision: 0.9739, 	recall: 0.8137, 	specificity: 0.9944, 	f1: 0.8866
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 65: 100%|██████████| 6507/6507 [07:12<00:00, 15.03it/s, loss=0.0405]
Train Epoch 65 ==> 	accuracy: 0.8841, 	precision: 0.9996, 	recall: 0.7685, 	specificity: 0.9997, 	f1: 0.8689
Test Epoch 65: 100%|██████████| 1768/1768 [00:45<00:00, 38.60it/s, loss=0.606]
Test Epoch 65 ==> 	accuracy: 0.9595, 	precision: 0.9758, 	recall: 0.8226, 	specificity: 0.9948, 	f1: 0.8927
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 66: 100%|██████████| 6507/6507 [07:18<00:00, 14.84it/s, loss=0.156]
Train Epoch 66 ==> 	accuracy: 0.8829, 	precision: 0.9996, 	recall: 0.7662, 	specificity: 0.9997, 	f1: 0.8674
Test Epoch 66: 100%|██████████| 1768/1768 [00:44<00:00, 39.57it/s, loss=0.0427]
Test Epoch 66 ==> 	accuracy: 0.9584, 	precision: 0.9791, 	recall: 0.8142, 	specificity: 0.9955, 	f1: 0.8891
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 67: 100%|██████████| 6507/6507 [07:13<00:00, 15.02it/s, loss=0.0055]
Train Epoch 67 ==> 	accuracy: 0.8858, 	precision: 0.9996, 	recall: 0.7720, 	specificity: 0.9997, 	f1: 0.8711
Test Epoch 67: 100%|██████████| 1768/1768 [00:43<00:00, 40.31it/s, loss=0.168]
Test Epoch 67 ==> 	accuracy: 0.9563, 	precision: 0.9844, 	recall: 0.7990, 	specificity: 0.9967, 	f1: 0.8821
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 68: 100%|██████████| 6507/6507 [06:59<00:00, 15.52it/s, loss=0.0037]
Train Epoch 68 ==> 	accuracy: 0.8875, 	precision: 0.9996, 	recall: 0.7753, 	specificity: 0.9997, 	f1: 0.8733
Test Epoch 68: 100%|██████████| 1768/1768 [00:50<00:00, 34.77it/s, loss=0.117]
Test Epoch 68 ==> 	accuracy: 0.9556, 	precision: 0.9843, 	recall: 0.7957, 	specificity: 0.9967, 	f1: 0.8800
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 69: 100%|██████████| 6507/6507 [07:12<00:00, 15.06it/s, loss=0.0608]
Train Epoch 69 ==> 	accuracy: 0.8853, 	precision: 0.9996, 	recall: 0.7710, 	specificity: 0.9997, 	f1: 0.8705
Test Epoch 69: 100%|██████████| 1768/1768 [00:47<00:00, 37.54it/s, loss=0.848]
Test Epoch 69 ==> 	accuracy: 0.9587, 	precision: 0.9811, 	recall: 0.8138, 	specificity: 0.9960, 	f1: 0.8897
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 70: 100%|██████████| 6507/6507 [07:13<00:00, 15.02it/s, loss=0.0074]
Train Epoch 70 ==> 	accuracy: 0.8869, 	precision: 0.9996, 	recall: 0.7741, 	specificity: 0.9997, 	f1: 0.8725
Test Epoch 70: 100%|██████████| 1768/1768 [00:49<00:00, 35.72it/s, loss=2.32]
Test Epoch 70 ==> 	accuracy: 0.9590, 	precision: 0.9785, 	recall: 0.8177, 	specificity: 0.9954, 	f1: 0.8909
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 71: 100%|██████████| 6507/6507 [07:08<00:00, 15.18it/s, loss=0.163]
Train Epoch 71 ==> 	accuracy: 0.8877, 	precision: 0.9996, 	recall: 0.7758, 	specificity: 0.9997, 	f1: 0.8736
Test Epoch 71: 100%|██████████| 1768/1768 [00:47<00:00, 37.41it/s, loss=0.942]
Test Epoch 71 ==> 	accuracy: 0.9590, 	precision: 0.9821, 	recall: 0.8143, 	specificity: 0.9962, 	f1: 0.8903
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 72: 100%|██████████| 6507/6507 [07:05<00:00, 15.28it/s, loss=0.459]
Train Epoch 72 ==> 	accuracy: 0.8885, 	precision: 0.9996, 	recall: 0.7772, 	specificity: 0.9997, 	f1: 0.8745
Test Epoch 72: 100%|██████████| 1768/1768 [00:44<00:00, 40.04it/s, loss=0.12]
Test Epoch 72 ==> 	accuracy: 0.9602, 	precision: 0.9823, 	recall: 0.8204, 	specificity: 0.9962, 	f1: 0.8940
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 73: 100%|██████████| 6507/6507 [07:11<00:00, 15.07it/s, loss=0.0336]
Train Epoch 73 ==> 	accuracy: 0.8889, 	precision: 0.9996, 	recall: 0.7780, 	specificity: 0.9997, 	f1: 0.8750
Test Epoch 73: 100%|██████████| 1768/1768 [00:40<00:00, 43.39it/s, loss=0.105]
Test Epoch 73 ==> 	accuracy: 0.9603, 	precision: 0.9819, 	recall: 0.8210, 	specificity: 0.9961, 	f1: 0.8943
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 74: 100%|██████████| 6507/6507 [07:18<00:00, 14.85it/s, loss=0.586]
Train Epoch 74 ==> 	accuracy: 0.8907, 	precision: 0.9996, 	recall: 0.7816, 	specificity: 0.9997, 	f1: 0.8773
Test Epoch 74: 100%|██████████| 1768/1768 [00:45<00:00, 38.69it/s, loss=0.081]
Test Epoch 74 ==> 	accuracy: 0.9599, 	precision: 0.9835, 	recall: 0.8175, 	specificity: 0.9965, 	f1: 0.8928
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 75: 100%|██████████| 6507/6507 [07:11<00:00, 15.09it/s, loss=0.0292]
Train Epoch 75 ==> 	accuracy: 0.8886, 	precision: 0.9996, 	recall: 0.7776, 	specificity: 0.9997, 	f1: 0.8747
Test Epoch 75: 100%|██████████| 1768/1768 [00:44<00:00, 39.39it/s, loss=0.171]
Test Epoch 75 ==> 	accuracy: 0.9604, 	precision: 0.9756, 	recall: 0.8271, 	specificity: 0.9947, 	f1: 0.8952
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 76: 100%|██████████| 6507/6507 [07:33<00:00, 14.36it/s, loss=0.0342]
Train Epoch 76 ==> 	accuracy: 0.8927, 	precision: 0.9996, 	recall: 0.7856, 	specificity: 0.9997, 	f1: 0.8798
Test Epoch 76: 100%|██████████| 1768/1768 [00:44<00:00, 40.12it/s, loss=0.246]
Test Epoch 76 ==> 	accuracy: 0.9614, 	precision: 0.9729, 	recall: 0.8344, 	specificity: 0.9940, 	f1: 0.8984
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 77: 100%|██████████| 6507/6507 [07:25<00:00, 14.61it/s, loss=0.084]
Train Epoch 77 ==> 	accuracy: 0.8914, 	precision: 0.9996, 	recall: 0.7832, 	specificity: 0.9997, 	f1: 0.8783
Test Epoch 77: 100%|██████████| 1768/1768 [00:47<00:00, 37.47it/s, loss=1.53]
Test Epoch 77 ==> 	accuracy: 0.9599, 	precision: 0.9746, 	recall: 0.8253, 	specificity: 0.9945, 	f1: 0.8938
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 78: 100%|██████████| 6507/6507 [07:14<00:00, 14.96it/s, loss=2.59]
Train Epoch 78 ==> 	accuracy: 0.8935, 	precision: 0.9996, 	recall: 0.7872, 	specificity: 0.9997, 	f1: 0.8808
Test Epoch 78: 100%|██████████| 1768/1768 [00:47<00:00, 36.96it/s, loss=1.49]
Test Epoch 78 ==> 	accuracy: 0.9609, 	precision: 0.9723, 	recall: 0.8326, 	specificity: 0.9939, 	f1: 0.8970
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 79: 100%|██████████| 6507/6507 [07:07<00:00, 15.21it/s, loss=0.0171]
Train Epoch 79 ==> 	accuracy: 0.8918, 	precision: 0.9996, 	recall: 0.7840, 	specificity: 0.9997, 	f1: 0.8788
Test Epoch 79: 100%|██████████| 1768/1768 [00:44<00:00, 39.92it/s, loss=0.0915]
Test Epoch 79 ==> 	accuracy: 0.9598, 	precision: 0.9782, 	recall: 0.8216, 	specificity: 0.9953, 	f1: 0.8931
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 80: 100%|██████████| 6507/6507 [07:04<00:00, 15.31it/s, loss=1.01]
Train Epoch 80 ==> 	accuracy: 0.8924, 	precision: 0.9996, 	recall: 0.7851, 	specificity: 0.9997, 	f1: 0.8795
Test Epoch 80: 100%|██████████| 1768/1768 [00:45<00:00, 38.45it/s, loss=0.154]
Test Epoch 80 ==> 	accuracy: 0.9605, 	precision: 0.9810, 	recall: 0.8230, 	specificity: 0.9959, 	f1: 0.8951
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 81: 100%|██████████| 6507/6507 [07:16<00:00, 14.91it/s, loss=0.0446]
Train Epoch 81 ==> 	accuracy: 0.8949, 	precision: 0.9997, 	recall: 0.7900, 	specificity: 0.9997, 	f1: 0.8825
Test Epoch 81: 100%|██████████| 1768/1768 [00:48<00:00, 36.44it/s, loss=0.194]
Test Epoch 81 ==> 	accuracy: 0.9613, 	precision: 0.9836, 	recall: 0.8246, 	specificity: 0.9965, 	f1: 0.8971
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 82: 100%|██████████| 6507/6507 [07:14<00:00, 14.97it/s, loss=0.0137]
Train Epoch 82 ==> 	accuracy: 0.8933, 	precision: 0.9997, 	recall: 0.7868, 	specificity: 0.9997, 	f1: 0.8805
Test Epoch 82: 100%|██████████| 1768/1768 [00:45<00:00, 38.49it/s, loss=1.74]
Test Epoch 82 ==> 	accuracy: 0.9624, 	precision: 0.9759, 	recall: 0.8370, 	specificity: 0.9947, 	f1: 0.9011
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 83: 100%|██████████| 6507/6507 [07:17<00:00, 14.89it/s, loss=0.0565]
Train Epoch 83 ==> 	accuracy: 0.8938, 	precision: 0.9997, 	recall: 0.7877, 	specificity: 0.9998, 	f1: 0.8812
Test Epoch 83: 100%|██████████| 1768/1768 [00:44<00:00, 39.60it/s, loss=0.372]
Test Epoch 83 ==> 	accuracy: 0.9627, 	precision: 0.9754, 	recall: 0.8386, 	specificity: 0.9946, 	f1: 0.9019
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 84: 100%|██████████| 6507/6507 [07:12<00:00, 15.05it/s, loss=0.021]
Train Epoch 84 ==> 	accuracy: 0.8950, 	precision: 0.9997, 	recall: 0.7902, 	specificity: 0.9997, 	f1: 0.8827
Test Epoch 84: 100%|██████████| 1768/1768 [00:45<00:00, 38.73it/s, loss=0.117]
Test Epoch 84 ==> 	accuracy: 0.9625, 	precision: 0.9789, 	recall: 0.8344, 	specificity: 0.9954, 	f1: 0.9009
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 85: 100%|██████████| 6507/6507 [07:14<00:00, 14.99it/s, loss=0.011]
Train Epoch 85 ==> 	accuracy: 0.8966, 	precision: 0.9996, 	recall: 0.7936, 	specificity: 0.9997, 	f1: 0.8848
Test Epoch 85: 100%|██████████| 1768/1768 [00:46<00:00, 37.97it/s, loss=0.0674]
Test Epoch 85 ==> 	accuracy: 0.9628, 	precision: 0.9750, 	recall: 0.8398, 	specificity: 0.9945, 	f1: 0.9024
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 86: 100%|██████████| 6507/6507 [07:16<00:00, 14.89it/s, loss=0.0881]
Train Epoch 86 ==> 	accuracy: 0.8968, 	precision: 0.9996, 	recall: 0.7939, 	specificity: 0.9997, 	f1: 0.8850
Test Epoch 86: 100%|██████████| 1768/1768 [00:48<00:00, 36.30it/s, loss=0.134]
Test Epoch 86 ==> 	accuracy: 0.9620, 	precision: 0.9734, 	recall: 0.8370, 	specificity: 0.9941, 	f1: 0.9001
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 87: 100%|██████████| 6507/6507 [07:14<00:00, 14.98it/s, loss=0.0215]
Train Epoch 87 ==> 	accuracy: 0.8962, 	precision: 0.9997, 	recall: 0.7928, 	specificity: 0.9997, 	f1: 0.8843
Test Epoch 87: 100%|██████████| 1768/1768 [00:46<00:00, 37.94it/s, loss=0.0782]
Test Epoch 87 ==> 	accuracy: 0.9622, 	precision: 0.9747, 	recall: 0.8368, 	specificity: 0.9944, 	f1: 0.9005
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 88: 100%|██████████| 6507/6507 [07:09<00:00, 15.16it/s, loss=0.0321]
Train Epoch 88 ==> 	accuracy: 0.8977, 	precision: 0.9997, 	recall: 0.7957, 	specificity: 0.9997, 	f1: 0.8861
Test Epoch 88: 100%|██████████| 1768/1768 [00:52<00:00, 33.54it/s, loss=0.223]
Test Epoch 88 ==> 	accuracy: 0.9621, 	precision: 0.9779, 	recall: 0.8335, 	specificity: 0.9952, 	f1: 0.9000
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 89: 100%|██████████| 6507/6507 [07:23<00:00, 14.68it/s, loss=0.0078]
Train Epoch 89 ==> 	accuracy: 0.9000, 	precision: 0.9996, 	recall: 0.8003, 	specificity: 0.9997, 	f1: 0.8889
Test Epoch 89: 100%|██████████| 1768/1768 [00:47<00:00, 37.44it/s, loss=0.497]
Test Epoch 89 ==> 	accuracy: 0.9626, 	precision: 0.9751, 	recall: 0.8388, 	specificity: 0.9945, 	f1: 0.9018
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 90: 100%|██████████| 6507/6507 [07:18<00:00, 14.83it/s, loss=0.0091]
Train Epoch 90 ==> 	accuracy: 0.8969, 	precision: 0.9997, 	recall: 0.7940, 	specificity: 0.9998, 	f1: 0.8851
Test Epoch 90: 100%|██████████| 1768/1768 [00:50<00:00, 35.20it/s, loss=0.338]
Test Epoch 90 ==> 	accuracy: 0.9624, 	precision: 0.9743, 	recall: 0.8381, 	specificity: 0.9943, 	f1: 0.9011
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 91: 100%|██████████| 6507/6507 [07:22<00:00, 14.71it/s, loss=0.128]
Train Epoch 91 ==> 	accuracy: 0.8996, 	precision: 0.9997, 	recall: 0.7995, 	specificity: 0.9997, 	f1: 0.8884
Test Epoch 91: 100%|██████████| 1768/1768 [00:48<00:00, 36.49it/s, loss=0.79]
Test Epoch 91 ==> 	accuracy: 0.9631, 	precision: 0.9726, 	recall: 0.8435, 	specificity: 0.9939, 	f1: 0.9035
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 92: 100%|██████████| 6507/6507 [07:18<00:00, 14.85it/s, loss=0.0182]
Train Epoch 92 ==> 	accuracy: 0.8965, 	precision: 0.9997, 	recall: 0.7933, 	specificity: 0.9997, 	f1: 0.8846
Test Epoch 92: 100%|██████████| 1768/1768 [00:44<00:00, 39.73it/s, loss=0.108]
Test Epoch 92 ==> 	accuracy: 0.9637, 	precision: 0.9731, 	recall: 0.8457, 	specificity: 0.9940, 	f1: 0.9049
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 93: 100%|██████████| 6507/6507 [07:10<00:00, 15.10it/s, loss=0.0261]
Train Epoch 93 ==> 	accuracy: 0.8993, 	precision: 0.9997, 	recall: 0.7989, 	specificity: 0.9997, 	f1: 0.8881
Test Epoch 93: 100%|██████████| 1768/1768 [00:45<00:00, 38.46it/s, loss=0.248]
Test Epoch 93 ==> 	accuracy: 0.9633, 	precision: 0.9738, 	recall: 0.8435, 	specificity: 0.9942, 	f1: 0.9040
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 94: 100%|██████████| 6507/6507 [07:29<00:00, 14.48it/s, loss=0.0716]
Train Epoch 94 ==> 	accuracy: 0.8993, 	precision: 0.9997, 	recall: 0.7989, 	specificity: 0.9997, 	f1: 0.8881
Test Epoch 94: 100%|██████████| 1768/1768 [00:46<00:00, 38.01it/s, loss=0.119]
Test Epoch 94 ==> 	accuracy: 0.9631, 	precision: 0.9673, 	recall: 0.8482, 	specificity: 0.9926, 	f1: 0.9038
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 95: 100%|██████████| 6507/6507 [07:17<00:00, 14.87it/s, loss=0.0387]
Train Epoch 95 ==> 	accuracy: 0.8989, 	precision: 0.9997, 	recall: 0.7980, 	specificity: 0.9997, 	f1: 0.8875
Test Epoch 95: 100%|██████████| 1768/1768 [00:50<00:00, 34.73it/s, loss=0.0363]
Test Epoch 95 ==> 	accuracy: 0.9632, 	precision: 0.9783, 	recall: 0.8388, 	specificity: 0.9952, 	f1: 0.9032
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 96: 100%|██████████| 6507/6507 [07:07<00:00, 15.24it/s, loss=0.0141]
Train Epoch 96 ==> 	accuracy: 0.9002, 	precision: 0.9997, 	recall: 0.8007, 	specificity: 0.9997, 	f1: 0.8892
Test Epoch 96: 100%|██████████| 1768/1768 [00:49<00:00, 35.68it/s, loss=0.127]
Test Epoch 96 ==> 	accuracy: 0.9629, 	precision: 0.9741, 	recall: 0.8411, 	specificity: 0.9943, 	f1: 0.9027
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 97: 100%|██████████| 6507/6507 [07:09<00:00, 15.16it/s, loss=0.0284]
Train Epoch 97 ==> 	accuracy: 0.8989, 	precision: 0.9997, 	recall: 0.7982, 	specificity: 0.9997, 	f1: 0.8876
Test Epoch 97: 100%|██████████| 1768/1768 [00:50<00:00, 35.17it/s, loss=0.0324]
Test Epoch 97 ==> 	accuracy: 0.9632, 	precision: 0.9738, 	recall: 0.8427, 	specificity: 0.9942, 	f1: 0.9035
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 98: 100%|██████████| 6507/6507 [07:22<00:00, 14.69it/s, loss=0.0072]
Train Epoch 98 ==> 	accuracy: 0.9015, 	precision: 0.9997, 	recall: 0.8032, 	specificity: 0.9997, 	f1: 0.8907
Test Epoch 98: 100%|██████████| 1768/1768 [00:44<00:00, 39.53it/s, loss=0.551]
Test Epoch 98 ==> 	accuracy: 0.9632, 	precision: 0.9787, 	recall: 0.8383, 	specificity: 0.9953, 	f1: 0.9031
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 99: 100%|██████████| 6507/6507 [07:21<00:00, 14.74it/s, loss=0.0114]
Train Epoch 99 ==> 	accuracy: 0.8999, 	precision: 0.9997, 	recall: 0.8001, 	specificity: 0.9997, 	f1: 0.8888
Test Epoch 99: 100%|██████████| 1768/1768 [00:45<00:00, 39.01it/s, loss=2.7]
Test Epoch 99 ==> 	accuracy: 0.9620, 	precision: 0.9694, 	recall: 0.8407, 	specificity: 0.9932, 	f1: 0.9005
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 100: 100%|██████████| 6507/6507 [07:15<00:00, 14.94it/s, loss=0.0178]
Train Epoch 100 ==> 	accuracy: 0.9023, 	precision: 0.9997, 	recall: 0.8049, 	specificity: 0.9997, 	f1: 0.8918
Test Epoch 100: 100%|██████████| 1768/1768 [00:48<00:00, 36.80it/s, loss=0.0934]
Test Epoch 100 ==> 	accuracy: 0.9639, 	precision: 0.9756, 	recall: 0.8445, 	specificity: 0.9946, 	f1: 0.9053
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 101: 100%|██████████| 6507/6507 [07:09<00:00, 15.14it/s, loss=0.0107]
Train Epoch 101 ==> 	accuracy: 0.9030, 	precision: 0.9997, 	recall: 0.8062, 	specificity: 0.9998, 	f1: 0.8926
Test Epoch 101: 100%|██████████| 1768/1768 [00:43<00:00, 40.65it/s, loss=0.111]
Test Epoch 101 ==> 	accuracy: 0.9647, 	precision: 0.9734, 	recall: 0.8506, 	specificity: 0.9940, 	f1: 0.9079
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 102: 100%|██████████| 6507/6507 [07:10<00:00, 15.10it/s, loss=0.0169]
Train Epoch 102 ==> 	accuracy: 0.9032, 	precision: 0.9997, 	recall: 0.8067, 	specificity: 0.9998, 	f1: 0.8929
Test Epoch 102: 100%|██████████| 1768/1768 [00:44<00:00, 39.36it/s, loss=0.089]
Test Epoch 102 ==> 	accuracy: 0.9648, 	precision: 0.9678, 	recall: 0.8564, 	specificity: 0.9927, 	f1: 0.9087
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 103: 100%|██████████| 6507/6507 [07:18<00:00, 14.83it/s, loss=0.0081]
Train Epoch 103 ==> 	accuracy: 0.9015, 	precision: 0.9997, 	recall: 0.8033, 	specificity: 0.9998, 	f1: 0.8908
Test Epoch 103: 100%|██████████| 1768/1768 [00:49<00:00, 35.58it/s, loss=0.409]
Test Epoch 103 ==> 	accuracy: 0.9634, 	precision: 0.9733, 	recall: 0.8444, 	specificity: 0.9940, 	f1: 0.9043
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 104: 100%|██████████| 6507/6507 [07:04<00:00, 15.32it/s, loss=0.0016]
Train Epoch 104 ==> 	accuracy: 0.9023, 	precision: 0.9997, 	recall: 0.8048, 	specificity: 0.9997, 	f1: 0.8917
Test Epoch 104: 100%|██████████| 1768/1768 [00:42<00:00, 41.67it/s, loss=0.441]
Test Epoch 104 ==> 	accuracy: 0.9638, 	precision: 0.9726, 	recall: 0.8468, 	specificity: 0.9939, 	f1: 0.9053
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 105: 100%|██████████| 6507/6507 [07:06<00:00, 15.24it/s, loss=0.141]
Train Epoch 105 ==> 	accuracy: 0.9030, 	precision: 0.9997, 	recall: 0.8063, 	specificity: 0.9998, 	f1: 0.8927
Test Epoch 105: 100%|██████████| 1768/1768 [00:44<00:00, 39.33it/s, loss=0.633]
Test Epoch 105 ==> 	accuracy: 0.9640, 	precision: 0.9698, 	recall: 0.8506, 	specificity: 0.9932, 	f1: 0.9063
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 106: 100%|██████████| 6507/6507 [07:11<00:00, 15.08it/s, loss=0.007]
Train Epoch 106 ==> 	accuracy: 0.9035, 	precision: 0.9997, 	recall: 0.8072, 	specificity: 0.9998, 	f1: 0.8932
Test Epoch 106: 100%|██████████| 1768/1768 [00:47<00:00, 37.58it/s, loss=1.41]
Test Epoch 106 ==> 	accuracy: 0.9636, 	precision: 0.9716, 	recall: 0.8468, 	specificity: 0.9936, 	f1: 0.9049
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 107: 100%|██████████| 6507/6507 [07:12<00:00, 15.06it/s, loss=0.0457]
Train Epoch 107 ==> 	accuracy: 0.9045, 	precision: 0.9997, 	recall: 0.8092, 	specificity: 0.9998, 	f1: 0.8944
Test Epoch 107: 100%|██████████| 1768/1768 [00:49<00:00, 35.96it/s, loss=1.65]
Test Epoch 107 ==> 	accuracy: 0.9642, 	precision: 0.9719, 	recall: 0.8498, 	specificity: 0.9937, 	f1: 0.9067
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 108: 100%|██████████| 6507/6507 [07:16<00:00, 14.89it/s, loss=0.014]
Train Epoch 108 ==> 	accuracy: 0.9035, 	precision: 0.9997, 	recall: 0.8073, 	specificity: 0.9998, 	f1: 0.8933
Test Epoch 108: 100%|██████████| 1768/1768 [00:46<00:00, 38.33it/s, loss=0.758]
Test Epoch 108 ==> 	accuracy: 0.9644, 	precision: 0.9678, 	recall: 0.8544, 	specificity: 0.9927, 	f1: 0.9076
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 109: 100%|██████████| 6507/6507 [07:17<00:00, 14.87it/s, loss=0.0253]
Train Epoch 109 ==> 	accuracy: 0.9050, 	precision: 0.9997, 	recall: 0.8102, 	specificity: 0.9998, 	f1: 0.8951
Test Epoch 109: 100%|██████████| 1768/1768 [00:48<00:00, 36.45it/s, loss=3.19]
Test Epoch 109 ==> 	accuracy: 0.9635, 	precision: 0.9678, 	recall: 0.8496, 	specificity: 0.9927, 	f1: 0.9049
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 110: 100%|██████████| 6507/6507 [07:16<00:00, 14.90it/s, loss=0.0192]
Train Epoch 110 ==> 	accuracy: 0.9035, 	precision: 0.9997, 	recall: 0.8073, 	specificity: 0.9997, 	f1: 0.8932
Test Epoch 110: 100%|██████████| 1768/1768 [00:45<00:00, 38.75it/s, loss=0.173]
Test Epoch 110 ==> 	accuracy: 0.9640, 	precision: 0.9692, 	recall: 0.8510, 	specificity: 0.9930, 	f1: 0.9063
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 111: 100%|██████████| 6507/6507 [07:23<00:00, 14.67it/s, loss=0.0284]
Train Epoch 111 ==> 	accuracy: 0.9067, 	precision: 0.9997, 	recall: 0.8136, 	specificity: 0.9998, 	f1: 0.8971
Test Epoch 111: 100%|██████████| 1768/1768 [00:44<00:00, 39.83it/s, loss=0.228]
Test Epoch 111 ==> 	accuracy: 0.9648, 	precision: 0.9711, 	recall: 0.8531, 	specificity: 0.9935, 	f1: 0.9083
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 112: 100%|██████████| 6507/6507 [07:22<00:00, 14.71it/s, loss=0.0277]
Train Epoch 112 ==> 	accuracy: 0.9035, 	precision: 0.9997, 	recall: 0.8072, 	specificity: 0.9998, 	f1: 0.8932
Test Epoch 112: 100%|██████████| 1768/1768 [00:48<00:00, 36.42it/s, loss=0.58]
Test Epoch 112 ==> 	accuracy: 0.9641, 	precision: 0.9731, 	recall: 0.8481, 	specificity: 0.9940, 	f1: 0.9063
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 113: 100%|██████████| 6507/6507 [07:24<00:00, 14.65it/s, loss=0.0814]
Train Epoch 113 ==> 	accuracy: 0.9050, 	precision: 0.9997, 	recall: 0.8103, 	specificity: 0.9998, 	f1: 0.8951
Test Epoch 113: 100%|██████████| 1768/1768 [00:50<00:00, 35.06it/s, loss=0.157]
Test Epoch 113 ==> 	accuracy: 0.9643, 	precision: 0.9694, 	recall: 0.8524, 	specificity: 0.9931, 	f1: 0.9072
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 114: 100%|██████████| 6507/6507 [07:23<00:00, 14.67it/s, loss=0.136]
Train Epoch 114 ==> 	accuracy: 0.9049, 	precision: 0.9997, 	recall: 0.8101, 	specificity: 0.9997, 	f1: 0.8949
Test Epoch 114: 100%|██████████| 1768/1768 [00:44<00:00, 39.63it/s, loss=0.138]
Test Epoch 114 ==> 	accuracy: 0.9645, 	precision: 0.9705, 	recall: 0.8524, 	specificity: 0.9933, 	f1: 0.9077
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 115: 100%|██████████| 6507/6507 [07:17<00:00, 14.86it/s, loss=0.378]
Train Epoch 115 ==> 	accuracy: 0.9057, 	precision: 0.9997, 	recall: 0.8117, 	specificity: 0.9997, 	f1: 0.8959
Test Epoch 115: 100%|██████████| 1768/1768 [00:49<00:00, 35.82it/s, loss=0.0882]
Test Epoch 115 ==> 	accuracy: 0.9648, 	precision: 0.9716, 	recall: 0.8527, 	specificity: 0.9936, 	f1: 0.9083
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 116: 100%|██████████| 6507/6507 [07:20<00:00, 14.76it/s, loss=0.0123]
Train Epoch 116 ==> 	accuracy: 0.9036, 	precision: 0.9997, 	recall: 0.8073, 	specificity: 0.9998, 	f1: 0.8933
Test Epoch 116: 100%|██████████| 1768/1768 [00:49<00:00, 35.59it/s, loss=1.76]
Test Epoch 116 ==> 	accuracy: 0.9647, 	precision: 0.9700, 	recall: 0.8538, 	specificity: 0.9932, 	f1: 0.9082
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 117: 100%|██████████| 6507/6507 [07:01<00:00, 15.45it/s, loss=0.0764]
Train Epoch 117 ==> 	accuracy: 0.9065, 	precision: 0.9997, 	recall: 0.8132, 	specificity: 0.9998, 	f1: 0.8969
Test Epoch 117: 100%|██████████| 1768/1768 [00:51<00:00, 34.26it/s, loss=0.0769]
Test Epoch 117 ==> 	accuracy: 0.9652, 	precision: 0.9691, 	recall: 0.8574, 	specificity: 0.9930, 	f1: 0.9098
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 118: 100%|██████████| 6507/6507 [07:10<00:00, 15.12it/s, loss=0.0244]
Train Epoch 118 ==> 	accuracy: 0.9060, 	precision: 0.9997, 	recall: 0.8123, 	specificity: 0.9998, 	f1: 0.8963
Test Epoch 118: 100%|██████████| 1768/1768 [00:50<00:00, 35.30it/s, loss=0.118]
Test Epoch 118 ==> 	accuracy: 0.9665, 	precision: 0.9762, 	recall: 0.8571, 	specificity: 0.9946, 	f1: 0.9128
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 119: 100%|██████████| 6507/6507 [07:14<00:00, 14.99it/s, loss=0.0078]
Train Epoch 119 ==> 	accuracy: 0.9055, 	precision: 0.9997, 	recall: 0.8113, 	specificity: 0.9998, 	f1: 0.8957
Test Epoch 119: 100%|██████████| 1768/1768 [00:49<00:00, 35.94it/s, loss=2.89]
Test Epoch 119 ==> 	accuracy: 0.9662, 	precision: 0.9760, 	recall: 0.8560, 	specificity: 0.9946, 	f1: 0.9120
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 120: 100%|██████████| 6507/6507 [07:13<00:00, 15.00it/s, loss=0.0081]
Train Epoch 120 ==> 	accuracy: 0.9052, 	precision: 0.9997, 	recall: 0.8106, 	specificity: 0.9998, 	f1: 0.8953
Test Epoch 120: 100%|██████████| 1768/1768 [00:47<00:00, 37.09it/s, loss=0.542]
Test Epoch 120 ==> 	accuracy: 0.9657, 	precision: 0.9783, 	recall: 0.8514, 	specificity: 0.9951, 	f1: 0.9105
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 121: 100%|██████████| 6507/6507 [07:20<00:00, 14.79it/s, loss=0.0203]
Train Epoch 121 ==> 	accuracy: 0.9072, 	precision: 0.9997, 	recall: 0.8147, 	specificity: 0.9998, 	f1: 0.8977
Test Epoch 121: 100%|██████████| 1768/1768 [00:47<00:00, 37.34it/s, loss=0.481]
Test Epoch 121 ==> 	accuracy: 0.9660, 	precision: 0.9763, 	recall: 0.8545, 	specificity: 0.9947, 	f1: 0.9113
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 122: 100%|██████████| 6507/6507 [07:03<00:00, 15.37it/s, loss=0.0006]
Train Epoch 122 ==> 	accuracy: 0.9081, 	precision: 0.9997, 	recall: 0.8165, 	specificity: 0.9998, 	f1: 0.8988
Test Epoch 122: 100%|██████████| 1768/1768 [00:49<00:00, 36.04it/s, loss=0.0998]
Test Epoch 122 ==> 	accuracy: 0.9665, 	precision: 0.9720, 	recall: 0.8612, 	specificity: 0.9936, 	f1: 0.9133
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 123: 100%|██████████| 6507/6507 [07:20<00:00, 14.77it/s, loss=0.0136]
Train Epoch 123 ==> 	accuracy: 0.9054, 	precision: 0.9997, 	recall: 0.8110, 	specificity: 0.9998, 	f1: 0.8955
Test Epoch 123: 100%|██████████| 1768/1768 [00:48<00:00, 36.81it/s, loss=0.359]
Test Epoch 123 ==> 	accuracy: 0.9655, 	precision: 0.9764, 	recall: 0.8520, 	specificity: 0.9947, 	f1: 0.9099
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 124: 100%|██████████| 6507/6507 [07:09<00:00, 15.14it/s, loss=0.0293]
Train Epoch 124 ==> 	accuracy: 0.9084, 	precision: 0.9997, 	recall: 0.8170, 	specificity: 0.9998, 	f1: 0.8992
Test Epoch 124: 100%|██████████| 1768/1768 [00:50<00:00, 35.21it/s, loss=1.53]
Test Epoch 124 ==> 	accuracy: 0.9661, 	precision: 0.9742, 	recall: 0.8572, 	specificity: 0.9942, 	f1: 0.9119
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 125: 100%|██████████| 6507/6507 [07:08<00:00, 15.19it/s, loss=0.0211]
Train Epoch 125 ==> 	accuracy: 0.9083, 	precision: 0.9997, 	recall: 0.8169, 	specificity: 0.9998, 	f1: 0.8991
Test Epoch 125: 100%|██████████| 1768/1768 [00:51<00:00, 34.45it/s, loss=2.28]
Test Epoch 125 ==> 	accuracy: 0.9663, 	precision: 0.9712, 	recall: 0.8607, 	specificity: 0.9934, 	f1: 0.9126
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 126: 100%|██████████| 6507/6507 [07:26<00:00, 14.57it/s, loss=0.197]
Train Epoch 126 ==> 	accuracy: 0.9080, 	precision: 0.9997, 	recall: 0.8161, 	specificity: 0.9998, 	f1: 0.8986
Test Epoch 126: 100%|██████████| 1768/1768 [00:50<00:00, 35.00it/s, loss=0.0772]
Test Epoch 126 ==> 	accuracy: 0.9648, 	precision: 0.9708, 	recall: 0.8533, 	specificity: 0.9934, 	f1: 0.9083
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 127: 100%|██████████| 6507/6507 [07:15<00:00, 14.95it/s, loss=0.0222]
Train Epoch 127 ==> 	accuracy: 0.9078, 	precision: 0.9997, 	recall: 0.8159, 	specificity: 0.9998, 	f1: 0.8985
Test Epoch 127: 100%|██████████| 1768/1768 [00:49<00:00, 35.69it/s, loss=1.99]
Test Epoch 127 ==> 	accuracy: 0.9656, 	precision: 0.9755, 	recall: 0.8530, 	specificity: 0.9945, 	f1: 0.9102
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 128: 100%|██████████| 6507/6507 [07:19<00:00, 14.81it/s, loss=0.366]
Train Epoch 128 ==> 	accuracy: 0.9075, 	precision: 0.9997, 	recall: 0.8152, 	specificity: 0.9998, 	f1: 0.8981
Test Epoch 128: 100%|██████████| 1768/1768 [00:49<00:00, 35.94it/s, loss=0.0876]
Test Epoch 128 ==> 	accuracy: 0.9667, 	precision: 0.9760, 	recall: 0.8585, 	specificity: 0.9946, 	f1: 0.9135
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 129: 100%|██████████| 6507/6507 [07:21<00:00, 14.73it/s, loss=0.133]
Train Epoch 129 ==> 	accuracy: 0.9064, 	precision: 0.9997, 	recall: 0.8131, 	specificity: 0.9998, 	f1: 0.8968
Test Epoch 129: 100%|██████████| 1768/1768 [00:50<00:00, 35.16it/s, loss=0.568]
Test Epoch 129 ==> 	accuracy: 0.9659, 	precision: 0.9726, 	recall: 0.8573, 	specificity: 0.9938, 	f1: 0.9113
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 130: 100%|██████████| 6507/6507 [07:30<00:00, 14.44it/s, loss=0.0121]
Train Epoch 130 ==> 	accuracy: 0.9079, 	precision: 0.9997, 	recall: 0.8160, 	specificity: 0.9998, 	f1: 0.8986
Test Epoch 130: 100%|██████████| 1768/1768 [00:49<00:00, 35.97it/s, loss=0.124]
Test Epoch 130 ==> 	accuracy: 0.9663, 	precision: 0.9721, 	recall: 0.8599, 	specificity: 0.9936, 	f1: 0.9125
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 131: 100%|██████████| 6507/6507 [07:32<00:00, 14.39it/s, loss=0.026]
Train Epoch 131 ==> 	accuracy: 0.9076, 	precision: 0.9997, 	recall: 0.8153, 	specificity: 0.9998, 	f1: 0.8982
Test Epoch 131: 100%|██████████| 1768/1768 [00:54<00:00, 32.37it/s, loss=0.674]
Test Epoch 131 ==> 	accuracy: 0.9663, 	precision: 0.9767, 	recall: 0.8556, 	specificity: 0.9947, 	f1: 0.9121
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 132: 100%|██████████| 6507/6507 [07:29<00:00, 14.49it/s, loss=0.0448]
Train Epoch 132 ==> 	accuracy: 0.9086, 	precision: 0.9997, 	recall: 0.8174, 	specificity: 0.9998, 	f1: 0.8994
Test Epoch 132: 100%|██████████| 1768/1768 [00:50<00:00, 35.08it/s, loss=0.107]
Test Epoch 132 ==> 	accuracy: 0.9663, 	precision: 0.9715, 	recall: 0.8605, 	specificity: 0.9935, 	f1: 0.9127
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 133: 100%|██████████| 6507/6507 [07:28<00:00, 14.51it/s, loss=0.636]
Train Epoch 133 ==> 	accuracy: 0.9076, 	precision: 0.9997, 	recall: 0.8154, 	specificity: 0.9998, 	f1: 0.8982
Test Epoch 133: 100%|██████████| 1768/1768 [00:52<00:00, 33.87it/s, loss=2.07]
Test Epoch 133 ==> 	accuracy: 0.9660, 	precision: 0.9735, 	recall: 0.8573, 	specificity: 0.9940, 	f1: 0.9117
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 134: 100%|██████████| 6507/6507 [07:26<00:00, 14.56it/s, loss=0.0064]
Train Epoch 134 ==> 	accuracy: 0.9103, 	precision: 0.9997, 	recall: 0.8207, 	specificity: 0.9998, 	f1: 0.9014
Test Epoch 134: 100%|██████████| 1768/1768 [00:45<00:00, 38.83it/s, loss=2.68]
Test Epoch 134 ==> 	accuracy: 0.9670, 	precision: 0.9691, 	recall: 0.8665, 	specificity: 0.9929, 	f1: 0.9149
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 135: 100%|██████████| 6507/6507 [07:28<00:00, 14.51it/s, loss=0.0872]
Train Epoch 135 ==> 	accuracy: 0.9105, 	precision: 0.9997, 	recall: 0.8212, 	specificity: 0.9998, 	f1: 0.9017
Test Epoch 135: 100%|██████████| 1768/1768 [00:50<00:00, 34.94it/s, loss=0.181]
Test Epoch 135 ==> 	accuracy: 0.9666, 	precision: 0.9699, 	recall: 0.8635, 	specificity: 0.9931, 	f1: 0.9136
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 136: 100%|██████████| 6507/6507 [07:28<00:00, 14.51it/s, loss=0.0227]
Train Epoch 136 ==> 	accuracy: 0.9089, 	precision: 0.9997, 	recall: 0.8181, 	specificity: 0.9998, 	f1: 0.8998
Test Epoch 136: 100%|██████████| 1768/1768 [00:51<00:00, 34.53it/s, loss=0.256]
Test Epoch 136 ==> 	accuracy: 0.9658, 	precision: 0.9720, 	recall: 0.8576, 	specificity: 0.9936, 	f1: 0.9112
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 137: 100%|██████████| 6507/6507 [07:37<00:00, 14.21it/s, loss=0.0171]
Train Epoch 137 ==> 	accuracy: 0.9099, 	precision: 0.9997, 	recall: 0.8201, 	specificity: 0.9998, 	f1: 0.9011
Test Epoch 137: 100%|██████████| 1768/1768 [00:53<00:00, 33.32it/s, loss=1.34]
Test Epoch 137 ==> 	accuracy: 0.9665, 	precision: 0.9722, 	recall: 0.8606, 	specificity: 0.9937, 	f1: 0.9130
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 138: 100%|██████████| 6507/6507 [07:30<00:00, 14.44it/s, loss=0.159]
Train Epoch 138 ==> 	accuracy: 0.9111, 	precision: 0.9997, 	recall: 0.8225, 	specificity: 0.9998, 	f1: 0.9025
Test Epoch 138: 100%|██████████| 1768/1768 [00:49<00:00, 35.94it/s, loss=2.79]
Test Epoch 138 ==> 	accuracy: 0.9661, 	precision: 0.9692, 	recall: 0.8615, 	specificity: 0.9930, 	f1: 0.9121
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 139: 100%|██████████| 6507/6507 [07:23<00:00, 14.67it/s, loss=0.0206]
Train Epoch 139 ==> 	accuracy: 0.9100, 	precision: 0.9997, 	recall: 0.8202, 	specificity: 0.9998, 	f1: 0.9011
Test Epoch 139: 100%|██████████| 1768/1768 [00:50<00:00, 35.26it/s, loss=0.347]
Test Epoch 139 ==> 	accuracy: 0.9660, 	precision: 0.9708, 	recall: 0.8598, 	specificity: 0.9934, 	f1: 0.9119
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 140: 100%|██████████| 6507/6507 [07:26<00:00, 14.59it/s, loss=0.009]
Train Epoch 140 ==> 	accuracy: 0.9084, 	precision: 0.9997, 	recall: 0.8170, 	specificity: 0.9998, 	f1: 0.8992
Test Epoch 140: 100%|██████████| 1768/1768 [00:50<00:00, 35.03it/s, loss=2.42]
Test Epoch 140 ==> 	accuracy: 0.9658, 	precision: 0.9717, 	recall: 0.8580, 	specificity: 0.9936, 	f1: 0.9113
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 141: 100%|██████████| 6507/6507 [07:32<00:00, 14.39it/s, loss=0.0019]
Train Epoch 141 ==> 	accuracy: 0.9102, 	precision: 0.9997, 	recall: 0.8207, 	specificity: 0.9998, 	f1: 0.9014
Test Epoch 141: 100%|██████████| 1768/1768 [00:55<00:00, 31.65it/s, loss=0.0824]
Test Epoch 141 ==> 	accuracy: 0.9664, 	precision: 0.9717, 	recall: 0.8607, 	specificity: 0.9935, 	f1: 0.9128
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 142: 100%|██████████| 6507/6507 [07:34<00:00, 14.30it/s, loss=0.0123]
Train Epoch 142 ==> 	accuracy: 0.9086, 	precision: 0.9997, 	recall: 0.8175, 	specificity: 0.9998, 	f1: 0.8995
Test Epoch 142: 100%|██████████| 1768/1768 [00:53<00:00, 32.77it/s, loss=0.242]
Test Epoch 142 ==> 	accuracy: 0.9662, 	precision: 0.9741, 	recall: 0.8575, 	specificity: 0.9941, 	f1: 0.9121
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 143: 100%|██████████| 6507/6507 [07:33<00:00, 14.34it/s, loss=0.0132]
Train Epoch 143 ==> 	accuracy: 0.9103, 	precision: 0.9997, 	recall: 0.8209, 	specificity: 0.9997, 	f1: 0.9015
Test Epoch 143: 100%|██████████| 1768/1768 [00:49<00:00, 36.05it/s, loss=0.179]
Test Epoch 143 ==> 	accuracy: 0.9664, 	precision: 0.9722, 	recall: 0.8603, 	specificity: 0.9937, 	f1: 0.9129
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 144: 100%|██████████| 6507/6507 [07:32<00:00, 14.38it/s, loss=0.0036]
Train Epoch 144 ==> 	accuracy: 0.9088, 	precision: 0.9997, 	recall: 0.8179, 	specificity: 0.9998, 	f1: 0.8997
Test Epoch 144: 100%|██████████| 1768/1768 [00:51<00:00, 34.52it/s, loss=0.361]
Test Epoch 144 ==> 	accuracy: 0.9669, 	precision: 0.9760, 	recall: 0.8592, 	specificity: 0.9946, 	f1: 0.9139
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 145: 100%|██████████| 6507/6507 [07:36<00:00, 14.25it/s, loss=0.0135]
Train Epoch 145 ==> 	accuracy: 0.9090, 	precision: 0.9997, 	recall: 0.8183, 	specificity: 0.9998, 	f1: 0.9000
Test Epoch 145: 100%|██████████| 1768/1768 [00:49<00:00, 35.48it/s, loss=0.358]
Test Epoch 145 ==> 	accuracy: 0.9674, 	precision: 0.9729, 	recall: 0.8645, 	specificity: 0.9938, 	f1: 0.9155
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 146: 100%|██████████| 6507/6507 [07:36<00:00, 14.25it/s, loss=0.0193]
Train Epoch 146 ==> 	accuracy: 0.9097, 	precision: 0.9997, 	recall: 0.8196, 	specificity: 0.9998, 	f1: 0.9007
Test Epoch 146: 100%|██████████| 1768/1768 [00:51<00:00, 34.17it/s, loss=5.05]
Test Epoch 146 ==> 	accuracy: 0.9664, 	precision: 0.9705, 	recall: 0.8618, 	specificity: 0.9933, 	f1: 0.9129
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 147: 100%|██████████| 6507/6507 [07:30<00:00, 14.44it/s, loss=0.503]
Train Epoch 147 ==> 	accuracy: 0.9110, 	precision: 0.9997, 	recall: 0.8222, 	specificity: 0.9998, 	f1: 0.9023
Test Epoch 147: 100%|██████████| 1768/1768 [00:47<00:00, 37.20it/s, loss=3.06]
Test Epoch 147 ==> 	accuracy: 0.9674, 	precision: 0.9697, 	recall: 0.8679, 	specificity: 0.9930, 	f1: 0.9160
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 148: 100%|██████████| 6507/6507 [07:43<00:00, 14.05it/s, loss=0.0518]
Train Epoch 148 ==> 	accuracy: 0.9126, 	precision: 0.9997, 	recall: 0.8254, 	specificity: 0.9998, 	f1: 0.9043
Test Epoch 148: 100%|██████████| 1768/1768 [00:49<00:00, 35.46it/s, loss=0.0724]
Test Epoch 148 ==> 	accuracy: 0.9674, 	precision: 0.9682, 	recall: 0.8692, 	specificity: 0.9927, 	f1: 0.9160
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 149: 100%|██████████| 6507/6507 [07:34<00:00, 14.32it/s, loss=0.005]
Train Epoch 149 ==> 	accuracy: 0.9095, 	precision: 0.9997, 	recall: 0.8193, 	specificity: 0.9998, 	f1: 0.9006
Test Epoch 149: 100%|██████████| 1768/1768 [00:50<00:00, 35.12it/s, loss=1.64]
Test Epoch 149 ==> 	accuracy: 0.9662, 	precision: 0.9681, 	recall: 0.8631, 	specificity: 0.9927, 	f1: 0.9126
Adjusting learning rate of group 0 to 5.8150e-06.

进程已结束，退出代码为 0

'''

'''
'../model_save_sigBlock4_focalWithMs_deformable_7mer_ab_deformable'
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6507/6507 [06:19<00:00, 17.16it/s, loss=0.104]
Train Epoch 0 ==> 	accuracy: 0.6357, 	precision: 0.9960, 	recall: 0.2725, 	specificity: 0.9989, 	f1: 0.4280
Test Epoch 0: 100%|██████████| 1768/1768 [00:43<00:00, 40.55it/s, loss=0.668]
Test Epoch 0 ==> 	accuracy: 0.9019, 	precision: 0.9555, 	recall: 0.5460, 	specificity: 0.9935, 	f1: 0.6949
Train Epoch 1: 100%|██████████| 6507/6507 [07:19<00:00, 14.81it/s, loss=0.119]
Train Epoch 1 ==> 	accuracy: 0.7423, 	precision: 0.9974, 	recall: 0.4858, 	specificity: 0.9987, 	f1: 0.6534
Test Epoch 1: 100%|██████████| 1768/1768 [00:43<00:00, 40.82it/s, loss=0.658]
Test Epoch 1 ==> 	accuracy: 0.9180, 	precision: 0.9636, 	recall: 0.6225, 	specificity: 0.9940, 	f1: 0.7564
Train Epoch 2: 100%|██████████| 6507/6507 [07:32<00:00, 14.37it/s, loss=0.0972]
Train Epoch 2 ==> 	accuracy: 0.7753, 	precision: 0.9980, 	recall: 0.5517, 	specificity: 0.9989, 	f1: 0.7106
Test Epoch 2: 100%|██████████| 1768/1768 [00:45<00:00, 38.77it/s, loss=0.519]
Test Epoch 2 ==> 	accuracy: 0.9201, 	precision: 0.9556, 	recall: 0.6393, 	specificity: 0.9924, 	f1: 0.7661
Train Epoch 3: 100%|██████████| 6507/6507 [07:19<00:00, 14.80it/s, loss=0.0685]
Train Epoch 3 ==> 	accuracy: 0.7851, 	precision: 0.9983, 	recall: 0.5712, 	specificity: 0.9990, 	f1: 0.7267
Test Epoch 3: 100%|██████████| 1768/1768 [00:48<00:00, 36.46it/s, loss=0.233]
Test Epoch 3 ==> 	accuracy: 0.9162, 	precision: 0.9908, 	recall: 0.5960, 	specificity: 0.9986, 	f1: 0.7443
Train Epoch 4: 100%|██████████| 6507/6507 [07:32<00:00, 14.39it/s, loss=0.0837]
Train Epoch 4 ==> 	accuracy: 0.8024, 	precision: 0.9985, 	recall: 0.6056, 	specificity: 0.9991, 	f1: 0.7540
Test Epoch 4: 100%|██████████| 1768/1768 [00:46<00:00, 37.81it/s, loss=0.276]
Test Epoch 4 ==> 	accuracy: 0.9249, 	precision: 0.9719, 	recall: 0.6515, 	specificity: 0.9952, 	f1: 0.7801
Train Epoch 5: 100%|██████████| 6507/6507 [07:20<00:00, 14.76it/s, loss=0.15]
Train Epoch 5 ==> 	accuracy: 0.8048, 	precision: 0.9987, 	recall: 0.6104, 	specificity: 0.9992, 	f1: 0.7577
Test Epoch 5: 100%|██████████| 1768/1768 [00:49<00:00, 35.57it/s, loss=0.316]
Test Epoch 5 ==> 	accuracy: 0.9246, 	precision: 0.9814, 	recall: 0.6438, 	specificity: 0.9969, 	f1: 0.7775
Train Epoch 6: 100%|██████████| 6507/6507 [07:27<00:00, 14.54it/s, loss=0.0893]
Train Epoch 6 ==> 	accuracy: 0.8115, 	precision: 0.9988, 	recall: 0.6238, 	specificity: 0.9992, 	f1: 0.7679
Test Epoch 6: 100%|██████████| 1768/1768 [00:50<00:00, 35.02it/s, loss=0.236]
Test Epoch 6 ==> 	accuracy: 0.9274, 	precision: 0.9727, 	recall: 0.6639, 	specificity: 0.9952, 	f1: 0.7892
Train Epoch 7: 100%|██████████| 6507/6507 [07:34<00:00, 14.31it/s, loss=0.09]
Train Epoch 7 ==> 	accuracy: 0.8192, 	precision: 0.9987, 	recall: 0.6392, 	specificity: 0.9992, 	f1: 0.7795
Test Epoch 7: 100%|██████████| 1768/1768 [00:49<00:00, 36.06it/s, loss=0.242]
Test Epoch 7 ==> 	accuracy: 0.9305, 	precision: 0.9846, 	recall: 0.6709, 	specificity: 0.9973, 	f1: 0.7981
Train Epoch 8: 100%|██████████| 6507/6507 [07:29<00:00, 14.47it/s, loss=0.107]
Train Epoch 8 ==> 	accuracy: 0.8166, 	precision: 0.9989, 	recall: 0.6339, 	specificity: 0.9993, 	f1: 0.7756
Test Epoch 8: 100%|██████████| 1768/1768 [00:44<00:00, 39.38it/s, loss=0.249]
Test Epoch 8 ==> 	accuracy: 0.9377, 	precision: 0.9809, 	recall: 0.7093, 	specificity: 0.9965, 	f1: 0.8233
Train Epoch 9: 100%|██████████| 6507/6507 [07:30<00:00, 14.46it/s, loss=0.0933]
Train Epoch 9 ==> 	accuracy: 0.8216, 	precision: 0.9990, 	recall: 0.6439, 	specificity: 0.9993, 	f1: 0.7831
Test Epoch 9: 100%|██████████| 1768/1768 [00:47<00:00, 37.11it/s, loss=0.211]
Test Epoch 9 ==> 	accuracy: 0.9368, 	precision: 0.9801, 	recall: 0.7051, 	specificity: 0.9963, 	f1: 0.8202
Train Epoch 10: 100%|██████████| 6507/6507 [07:19<00:00, 14.80it/s, loss=0.0929]
Train Epoch 10 ==> 	accuracy: 0.8250, 	precision: 0.9990, 	recall: 0.6507, 	specificity: 0.9994, 	f1: 0.7880
Test Epoch 10: 100%|██████████| 1768/1768 [00:46<00:00, 38.35it/s, loss=0.385]
Test Epoch 10 ==> 	accuracy: 0.9315, 	precision: 0.9819, 	recall: 0.6779, 	specificity: 0.9968, 	f1: 0.8020
Train Epoch 11: 100%|██████████| 6507/6507 [07:25<00:00, 14.59it/s, loss=0.168]
Train Epoch 11 ==> 	accuracy: 0.8322, 	precision: 0.9990, 	recall: 0.6650, 	specificity: 0.9993, 	f1: 0.7985
Test Epoch 11: 100%|██████████| 1768/1768 [00:49<00:00, 35.89it/s, loss=0.611]
Test Epoch 11 ==> 	accuracy: 0.9367, 	precision: 0.9802, 	recall: 0.7050, 	specificity: 0.9963, 	f1: 0.8201
Train Epoch 12: 100%|██████████| 6507/6507 [07:32<00:00, 14.37it/s, loss=0.0949]
Train Epoch 12 ==> 	accuracy: 0.8300, 	precision: 0.9991, 	recall: 0.6606, 	specificity: 0.9994, 	f1: 0.7953
Test Epoch 12: 100%|██████████| 1768/1768 [00:47<00:00, 36.93it/s, loss=0.229]
Test Epoch 12 ==> 	accuracy: 0.9408, 	precision: 0.9855, 	recall: 0.7213, 	specificity: 0.9973, 	f1: 0.8330
Train Epoch 13: 100%|██████████| 6507/6507 [07:26<00:00, 14.57it/s, loss=0.0805]
Train Epoch 13 ==> 	accuracy: 0.8329, 	precision: 0.9991, 	recall: 0.6663, 	specificity: 0.9994, 	f1: 0.7994
Test Epoch 13: 100%|██████████| 1768/1768 [00:45<00:00, 39.03it/s, loss=0.224]
Test Epoch 13 ==> 	accuracy: 0.9387, 	precision: 0.9841, 	recall: 0.7116, 	specificity: 0.9970, 	f1: 0.8260
Train Epoch 14: 100%|██████████| 6507/6507 [07:28<00:00, 14.51it/s, loss=0.016]
Train Epoch 14 ==> 	accuracy: 0.8399, 	precision: 0.9991, 	recall: 0.6805, 	specificity: 0.9994, 	f1: 0.8096
Test Epoch 14: 100%|██████████| 1768/1768 [00:45<00:00, 38.72it/s, loss=0.166]
Test Epoch 14 ==> 	accuracy: 0.9397, 	precision: 0.9826, 	recall: 0.7178, 	specificity: 0.9967, 	f1: 0.8296
Train Epoch 15: 100%|██████████| 6507/6507 [07:15<00:00, 14.95it/s, loss=0.148]
Train Epoch 15 ==> 	accuracy: 0.8381, 	precision: 0.9991, 	recall: 0.6768, 	specificity: 0.9994, 	f1: 0.8069
Test Epoch 15: 100%|██████████| 1768/1768 [00:44<00:00, 39.60it/s, loss=0.0976]
Test Epoch 15 ==> 	accuracy: 0.9408, 	precision: 0.9728, 	recall: 0.7308, 	specificity: 0.9947, 	f1: 0.8346
Train Epoch 16: 100%|██████████| 6507/6507 [07:23<00:00, 14.67it/s, loss=0.119]
Train Epoch 16 ==> 	accuracy: 0.8394, 	precision: 0.9992, 	recall: 0.6794, 	specificity: 0.9994, 	f1: 0.8088
Test Epoch 16: 100%|██████████| 1768/1768 [00:46<00:00, 37.76it/s, loss=1.42]
Test Epoch 16 ==> 	accuracy: 0.9427, 	precision: 0.9805, 	recall: 0.7345, 	specificity: 0.9962, 	f1: 0.8398
Train Epoch 17: 100%|██████████| 6507/6507 [07:26<00:00, 14.59it/s, loss=0.0723]
Train Epoch 17 ==> 	accuracy: 0.8407, 	precision: 0.9992, 	recall: 0.6819, 	specificity: 0.9994, 	f1: 0.8106
Test Epoch 17: 100%|██████████| 1768/1768 [00:46<00:00, 38.29it/s, loss=0.121]
Test Epoch 17 ==> 	accuracy: 0.9453, 	precision: 0.9814, 	recall: 0.7466, 	specificity: 0.9964, 	f1: 0.8480
Train Epoch 18: 100%|██████████| 6507/6507 [07:31<00:00, 14.40it/s, loss=0.088]
Train Epoch 18 ==> 	accuracy: 0.8451, 	precision: 0.9992, 	recall: 0.6907, 	specificity: 0.9995, 	f1: 0.8168
Test Epoch 18: 100%|██████████| 1768/1768 [00:48<00:00, 36.33it/s, loss=0.684]
Test Epoch 18 ==> 	accuracy: 0.9426, 	precision: 0.9858, 	recall: 0.7298, 	specificity: 0.9973, 	f1: 0.8387
Train Epoch 19: 100%|██████████| 6507/6507 [07:34<00:00, 14.31it/s, loss=0.0828]
Train Epoch 19 ==> 	accuracy: 0.8430, 	precision: 0.9993, 	recall: 0.6866, 	specificity: 0.9995, 	f1: 0.8139
Test Epoch 19: 100%|██████████| 1768/1768 [00:47<00:00, 37.58it/s, loss=0.343]
Test Epoch 19 ==> 	accuracy: 0.9434, 	precision: 0.9888, 	recall: 0.7315, 	specificity: 0.9979, 	f1: 0.8409
Train Epoch 20: 100%|██████████| 6507/6507 [07:38<00:00, 14.19it/s, loss=0.0873]
Train Epoch 20 ==> 	accuracy: 0.8462, 	precision: 0.9993, 	recall: 0.6930, 	specificity: 0.9995, 	f1: 0.8184
Test Epoch 20: 100%|██████████| 1768/1768 [00:44<00:00, 39.34it/s, loss=0.509]
Test Epoch 20 ==> 	accuracy: 0.9436, 	precision: 0.9736, 	recall: 0.7443, 	specificity: 0.9948, 	f1: 0.8437
Train Epoch 21: 100%|██████████| 6507/6507 [07:29<00:00, 14.48it/s, loss=0.0177]
Train Epoch 21 ==> 	accuracy: 0.8482, 	precision: 0.9993, 	recall: 0.6968, 	specificity: 0.9995, 	f1: 0.8211
Test Epoch 21: 100%|██████████| 1768/1768 [00:46<00:00, 37.73it/s, loss=1.19]
Test Epoch 21 ==> 	accuracy: 0.9474, 	precision: 0.9828, 	recall: 0.7559, 	specificity: 0.9966, 	f1: 0.8546
Train Epoch 22: 100%|██████████| 6507/6507 [07:32<00:00, 14.39it/s, loss=0.0786]
Train Epoch 22 ==> 	accuracy: 0.8497, 	precision: 0.9993, 	recall: 0.6998, 	specificity: 0.9995, 	f1: 0.8232
Test Epoch 22: 100%|██████████| 1768/1768 [00:46<00:00, 37.98it/s, loss=0.513]
Test Epoch 22 ==> 	accuracy: 0.9453, 	precision: 0.9858, 	recall: 0.7431, 	specificity: 0.9972, 	f1: 0.8474
Train Epoch 23: 100%|██████████| 6507/6507 [07:33<00:00, 14.36it/s, loss=0.0821]
Train Epoch 23 ==> 	accuracy: 0.8507, 	precision: 0.9993, 	recall: 0.7019, 	specificity: 0.9995, 	f1: 0.8246
Test Epoch 23: 100%|██████████| 1768/1768 [00:45<00:00, 39.18it/s, loss=0.0888]
Test Epoch 23 ==> 	accuracy: 0.9421, 	precision: 0.9882, 	recall: 0.7255, 	specificity: 0.9978, 	f1: 0.8367
Train Epoch 24: 100%|██████████| 6507/6507 [07:30<00:00, 14.44it/s, loss=0.0636]
Train Epoch 24 ==> 	accuracy: 0.8528, 	precision: 0.9993, 	recall: 0.7062, 	specificity: 0.9995, 	f1: 0.8275
Test Epoch 24: 100%|██████████| 1768/1768 [00:48<00:00, 36.80it/s, loss=0.316]
Test Epoch 24 ==> 	accuracy: 0.9454, 	precision: 0.9766, 	recall: 0.7509, 	specificity: 0.9954, 	f1: 0.8490
Train Epoch 25: 100%|██████████| 6507/6507 [07:29<00:00, 14.46it/s, loss=0.171]
Train Epoch 25 ==> 	accuracy: 0.8536, 	precision: 0.9993, 	recall: 0.7076, 	specificity: 0.9995, 	f1: 0.8285
Test Epoch 25: 100%|██████████| 1768/1768 [00:48<00:00, 36.48it/s, loss=0.234]
Test Epoch 25 ==> 	accuracy: 0.9439, 	precision: 0.9814, 	recall: 0.7396, 	specificity: 0.9964, 	f1: 0.8435
Train Epoch 26: 100%|██████████| 6507/6507 [07:32<00:00, 14.37it/s, loss=0.128]
Train Epoch 26 ==> 	accuracy: 0.8565, 	precision: 0.9993, 	recall: 0.7135, 	specificity: 0.9995, 	f1: 0.8326
Test Epoch 26: 100%|██████████| 1768/1768 [00:50<00:00, 34.79it/s, loss=0.152]
Test Epoch 26 ==> 	accuracy: 0.9456, 	precision: 0.9860, 	recall: 0.7445, 	specificity: 0.9973, 	f1: 0.8484
Train Epoch 27: 100%|██████████| 6507/6507 [07:31<00:00, 14.42it/s, loss=0.104]
Train Epoch 27 ==> 	accuracy: 0.8566, 	precision: 0.9994, 	recall: 0.7137, 	specificity: 0.9995, 	f1: 0.8327
Test Epoch 27: 100%|██████████| 1768/1768 [00:45<00:00, 39.14it/s, loss=0.674]
Test Epoch 27 ==> 	accuracy: 0.9500, 	precision: 0.9800, 	recall: 0.7714, 	specificity: 0.9959, 	f1: 0.8633
Train Epoch 28: 100%|██████████| 6507/6507 [07:34<00:00, 14.33it/s, loss=0.0018]
Train Epoch 28 ==> 	accuracy: 0.8554, 	precision: 0.9994, 	recall: 0.7113, 	specificity: 0.9995, 	f1: 0.8311
Test Epoch 28: 100%|██████████| 1768/1768 [00:43<00:00, 40.20it/s, loss=0.344]
Test Epoch 28 ==> 	accuracy: 0.9490, 	precision: 0.9815, 	recall: 0.7649, 	specificity: 0.9963, 	f1: 0.8598
Train Epoch 29: 100%|██████████| 6507/6507 [07:27<00:00, 14.53it/s, loss=0.212]
Train Epoch 29 ==> 	accuracy: 0.8558, 	precision: 0.9994, 	recall: 0.7120, 	specificity: 0.9995, 	f1: 0.8316
Test Epoch 29: 100%|██████████| 1768/1768 [00:46<00:00, 37.93it/s, loss=0.153]
Test Epoch 29 ==> 	accuracy: 0.9430, 	precision: 0.9917, 	recall: 0.7274, 	specificity: 0.9984, 	f1: 0.8392
Train Epoch 30: 100%|██████████| 6507/6507 [07:29<00:00, 14.48it/s, loss=0.0395]
Train Epoch 30 ==> 	accuracy: 0.8589, 	precision: 0.9994, 	recall: 0.7182, 	specificity: 0.9995, 	f1: 0.8358
Test Epoch 30: 100%|██████████| 1768/1768 [00:46<00:00, 37.91it/s, loss=0.226]
Test Epoch 30 ==> 	accuracy: 0.9459, 	precision: 0.9869, 	recall: 0.7453, 	specificity: 0.9974, 	f1: 0.8492
Train Epoch 31: 100%|██████████| 6507/6507 [07:37<00:00, 14.21it/s, loss=0.401]
Train Epoch 31 ==> 	accuracy: 0.8615, 	precision: 0.9994, 	recall: 0.7235, 	specificity: 0.9995, 	f1: 0.8394
Test Epoch 31: 100%|██████████| 1768/1768 [00:46<00:00, 38.13it/s, loss=0.15]
Test Epoch 31 ==> 	accuracy: 0.9492, 	precision: 0.9778, 	recall: 0.7691, 	specificity: 0.9955, 	f1: 0.8610
Train Epoch 32: 100%|██████████| 6507/6507 [07:30<00:00, 14.45it/s, loss=0.0655]
Train Epoch 32 ==> 	accuracy: 0.8591, 	precision: 0.9994, 	recall: 0.7186, 	specificity: 0.9996, 	f1: 0.8360
Test Epoch 32: 100%|██████████| 1768/1768 [00:48<00:00, 36.28it/s, loss=0.196]
Test Epoch 32 ==> 	accuracy: 0.9473, 	precision: 0.9861, 	recall: 0.7530, 	specificity: 0.9973, 	f1: 0.8539
Train Epoch 33: 100%|██████████| 6507/6507 [07:31<00:00, 14.41it/s, loss=0.0844]
Train Epoch 33 ==> 	accuracy: 0.8616, 	precision: 0.9994, 	recall: 0.7236, 	specificity: 0.9995, 	f1: 0.8394
Test Epoch 33: 100%|██████████| 1768/1768 [00:48<00:00, 36.61it/s, loss=0.185]
Test Epoch 33 ==> 	accuracy: 0.9466, 	precision: 0.9806, 	recall: 0.7539, 	specificity: 0.9962, 	f1: 0.8524
Train Epoch 34: 100%|██████████| 6507/6507 [07:32<00:00, 14.38it/s, loss=0.0703]
Train Epoch 34 ==> 	accuracy: 0.8638, 	precision: 0.9994, 	recall: 0.7280, 	specificity: 0.9996, 	f1: 0.8424
Test Epoch 34: 100%|██████████| 1768/1768 [00:46<00:00, 37.81it/s, loss=0.0939]
Test Epoch 34 ==> 	accuracy: 0.9467, 	precision: 0.9816, 	recall: 0.7537, 	specificity: 0.9964, 	f1: 0.8527
Train Epoch 35: 100%|██████████| 6507/6507 [07:31<00:00, 14.41it/s, loss=0.0857]
Train Epoch 35 ==> 	accuracy: 0.8646, 	precision: 0.9994, 	recall: 0.7297, 	specificity: 0.9995, 	f1: 0.8435
Test Epoch 35: 100%|██████████| 1768/1768 [00:48<00:00, 36.12it/s, loss=0.293]
Test Epoch 35 ==> 	accuracy: 0.9456, 	precision: 0.9835, 	recall: 0.7466, 	specificity: 0.9968, 	f1: 0.8488
Train Epoch 36: 100%|██████████| 6507/6507 [07:24<00:00, 14.65it/s, loss=0.102]
Train Epoch 36 ==> 	accuracy: 0.8643, 	precision: 0.9994, 	recall: 0.7291, 	specificity: 0.9996, 	f1: 0.8431
Test Epoch 36: 100%|██████████| 1768/1768 [00:45<00:00, 38.83it/s, loss=0.215]
Test Epoch 36 ==> 	accuracy: 0.9522, 	precision: 0.9849, 	recall: 0.7780, 	specificity: 0.9969, 	f1: 0.8693
Train Epoch 37: 100%|██████████| 6507/6507 [07:25<00:00, 14.61it/s, loss=0.0858]
Train Epoch 37 ==> 	accuracy: 0.8649, 	precision: 0.9994, 	recall: 0.7302, 	specificity: 0.9996, 	f1: 0.8439
Test Epoch 37: 100%|██████████| 1768/1768 [00:49<00:00, 36.08it/s, loss=0.182]
Test Epoch 37 ==> 	accuracy: 0.9426, 	precision: 0.9834, 	recall: 0.7317, 	specificity: 0.9968, 	f1: 0.8391
Train Epoch 38: 100%|██████████| 6507/6507 [07:28<00:00, 14.51it/s, loss=0.181]
Train Epoch 38 ==> 	accuracy: 0.8656, 	precision: 0.9994, 	recall: 0.7316, 	specificity: 0.9996, 	f1: 0.8448
Test Epoch 38: 100%|██████████| 1768/1768 [00:49<00:00, 35.49it/s, loss=0.083]
Test Epoch 38 ==> 	accuracy: 0.9491, 	precision: 0.9858, 	recall: 0.7620, 	specificity: 0.9972, 	f1: 0.8596
Train Epoch 39: 100%|██████████| 6507/6507 [07:24<00:00, 14.64it/s, loss=0.0602]
Train Epoch 39 ==> 	accuracy: 0.8673, 	precision: 0.9994, 	recall: 0.7351, 	specificity: 0.9996, 	f1: 0.8471
Test Epoch 39: 100%|██████████| 1768/1768 [00:50<00:00, 35.12it/s, loss=0.327]
Test Epoch 39 ==> 	accuracy: 0.9519, 	precision: 0.9814, 	recall: 0.7796, 	specificity: 0.9962, 	f1: 0.8690
Train Epoch 40: 100%|██████████| 6507/6507 [07:10<00:00, 15.11it/s, loss=0.0203]
Train Epoch 40 ==> 	accuracy: 0.8688, 	precision: 0.9994, 	recall: 0.7381, 	specificity: 0.9996, 	f1: 0.8491
Test Epoch 40: 100%|██████████| 1768/1768 [00:47<00:00, 37.22it/s, loss=0.602]
Test Epoch 40 ==> 	accuracy: 0.9524, 	precision: 0.9674, 	recall: 0.7940, 	specificity: 0.9931, 	f1: 0.8721
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 41: 100%|██████████| 6507/6507 [07:05<00:00, 15.29it/s, loss=0.0138]
Train Epoch 41 ==> 	accuracy: 0.8682, 	precision: 0.9994, 	recall: 0.7368, 	specificity: 0.9996, 	f1: 0.8483
Test Epoch 41: 100%|██████████| 1768/1768 [00:50<00:00, 34.71it/s, loss=0.16]
Test Epoch 41 ==> 	accuracy: 0.9492, 	precision: 0.9809, 	recall: 0.7667, 	specificity: 0.9962, 	f1: 0.8607
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 42: 100%|██████████| 6507/6507 [07:09<00:00, 15.16it/s, loss=0.0271]
Train Epoch 42 ==> 	accuracy: 0.8683, 	precision: 0.9995, 	recall: 0.7370, 	specificity: 0.9996, 	f1: 0.8484
Test Epoch 42: 100%|██████████| 1768/1768 [00:49<00:00, 35.52it/s, loss=0.548]
Test Epoch 42 ==> 	accuracy: 0.9547, 	precision: 0.9816, 	recall: 0.7935, 	specificity: 0.9962, 	f1: 0.8776
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 43: 100%|██████████| 6507/6507 [07:06<00:00, 15.25it/s, loss=0.0025]
Train Epoch 43 ==> 	accuracy: 0.8692, 	precision: 0.9995, 	recall: 0.7388, 	specificity: 0.9996, 	f1: 0.8496
Test Epoch 43: 100%|██████████| 1768/1768 [00:47<00:00, 37.25it/s, loss=0.193]
Test Epoch 43 ==> 	accuracy: 0.9509, 	precision: 0.9829, 	recall: 0.7733, 	specificity: 0.9965, 	f1: 0.8656
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 44: 100%|██████████| 6507/6507 [07:05<00:00, 15.29it/s, loss=0.0062]
Train Epoch 44 ==> 	accuracy: 0.8704, 	precision: 0.9995, 	recall: 0.7411, 	specificity: 0.9996, 	f1: 0.8511
Test Epoch 44: 100%|██████████| 1768/1768 [00:44<00:00, 40.17it/s, loss=0.385]
Test Epoch 44 ==> 	accuracy: 0.9516, 	precision: 0.9874, 	recall: 0.7735, 	specificity: 0.9975, 	f1: 0.8674
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 45: 100%|██████████| 6507/6507 [07:17<00:00, 14.88it/s, loss=0.0341]
Train Epoch 45 ==> 	accuracy: 0.8693, 	precision: 0.9995, 	recall: 0.7389, 	specificity: 0.9996, 	f1: 0.8497
Test Epoch 45: 100%|██████████| 1768/1768 [00:44<00:00, 39.87it/s, loss=0.583]
Test Epoch 45 ==> 	accuracy: 0.9493, 	precision: 0.9839, 	recall: 0.7645, 	specificity: 0.9968, 	f1: 0.8604
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 46: 100%|██████████| 6507/6507 [07:06<00:00, 15.25it/s, loss=0.0866]
Train Epoch 46 ==> 	accuracy: 0.8734, 	precision: 0.9995, 	recall: 0.7472, 	specificity: 0.9996, 	f1: 0.8551
Test Epoch 46: 100%|██████████| 1768/1768 [00:47<00:00, 36.86it/s, loss=0.0765]
Test Epoch 46 ==> 	accuracy: 0.9532, 	precision: 0.9873, 	recall: 0.7812, 	specificity: 0.9974, 	f1: 0.8723
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 47: 100%|██████████| 6507/6507 [07:03<00:00, 15.38it/s, loss=0.0033]
Train Epoch 47 ==> 	accuracy: 0.8718, 	precision: 0.9995, 	recall: 0.7439, 	specificity: 0.9996, 	f1: 0.8530
Test Epoch 47: 100%|██████████| 1768/1768 [00:47<00:00, 37.32it/s, loss=0.153]
Test Epoch 47 ==> 	accuracy: 0.9567, 	precision: 0.9792, 	recall: 0.8052, 	specificity: 0.9956, 	f1: 0.8838
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 48: 100%|██████████| 6507/6507 [06:58<00:00, 15.54it/s, loss=0.0103]
Train Epoch 48 ==> 	accuracy: 0.8764, 	precision: 0.9995, 	recall: 0.7531, 	specificity: 0.9996, 	f1: 0.8590
Test Epoch 48: 100%|██████████| 1768/1768 [00:48<00:00, 36.53it/s, loss=0.249]
Test Epoch 48 ==> 	accuracy: 0.9561, 	precision: 0.9870, 	recall: 0.7960, 	specificity: 0.9973, 	f1: 0.8812
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 49: 100%|██████████| 6507/6507 [07:04<00:00, 15.34it/s, loss=0.0188]
Train Epoch 49 ==> 	accuracy: 0.8736, 	precision: 0.9995, 	recall: 0.7476, 	specificity: 0.9996, 	f1: 0.8554
Test Epoch 49: 100%|██████████| 1768/1768 [00:44<00:00, 39.33it/s, loss=0.127]
Test Epoch 49 ==> 	accuracy: 0.9541, 	precision: 0.9883, 	recall: 0.7848, 	specificity: 0.9976, 	f1: 0.8749
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 50: 100%|██████████| 6507/6507 [07:12<00:00, 15.06it/s, loss=0.0012]
Train Epoch 50 ==> 	accuracy: 0.8804, 	precision: 0.9995, 	recall: 0.7612, 	specificity: 0.9996, 	f1: 0.8642
Test Epoch 50: 100%|██████████| 1768/1768 [00:45<00:00, 38.67it/s, loss=0.119]
Test Epoch 50 ==> 	accuracy: 0.9545, 	precision: 0.9872, 	recall: 0.7879, 	specificity: 0.9974, 	f1: 0.8763
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 51: 100%|██████████| 6507/6507 [07:01<00:00, 15.45it/s, loss=0.168]
Train Epoch 51 ==> 	accuracy: 0.8786, 	precision: 0.9995, 	recall: 0.7575, 	specificity: 0.9996, 	f1: 0.8618
Test Epoch 51: 100%|██████████| 1768/1768 [00:46<00:00, 37.68it/s, loss=0.635]
Test Epoch 51 ==> 	accuracy: 0.9587, 	precision: 0.9830, 	recall: 0.8121, 	specificity: 0.9964, 	f1: 0.8894
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 52: 100%|██████████| 6507/6507 [06:56<00:00, 15.62it/s, loss=0.0145]
Train Epoch 52 ==> 	accuracy: 0.8817, 	precision: 0.9996, 	recall: 0.7637, 	specificity: 0.9997, 	f1: 0.8658
Test Epoch 52: 100%|██████████| 1768/1768 [00:50<00:00, 34.78it/s, loss=0.184]
Test Epoch 52 ==> 	accuracy: 0.9536, 	precision: 0.9851, 	recall: 0.7848, 	specificity: 0.9969, 	f1: 0.8736
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 53: 100%|██████████| 6507/6507 [07:05<00:00, 15.28it/s, loss=0.0024]
Train Epoch 53 ==> 	accuracy: 0.8818, 	precision: 0.9995, 	recall: 0.7639, 	specificity: 0.9997, 	f1: 0.8660
Test Epoch 53: 100%|██████████| 1768/1768 [00:48<00:00, 36.60it/s, loss=1.22]
Test Epoch 53 ==> 	accuracy: 0.9573, 	precision: 0.9830, 	recall: 0.8053, 	specificity: 0.9964, 	f1: 0.8853
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 54: 100%|██████████| 6507/6507 [07:11<00:00, 15.07it/s, loss=0.051]
Train Epoch 54 ==> 	accuracy: 0.8829, 	precision: 0.9995, 	recall: 0.7662, 	specificity: 0.9996, 	f1: 0.8674
Test Epoch 54: 100%|██████████| 1768/1768 [00:51<00:00, 34.24it/s, loss=0.202]
Test Epoch 54 ==> 	accuracy: 0.9585, 	precision: 0.9784, 	recall: 0.8149, 	specificity: 0.9954, 	f1: 0.8892
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 55: 100%|██████████| 6507/6507 [07:07<00:00, 15.23it/s, loss=0.0146]
Train Epoch 55 ==> 	accuracy: 0.8839, 	precision: 0.9996, 	recall: 0.7681, 	specificity: 0.9997, 	f1: 0.8687
Test Epoch 55: 100%|██████████| 1768/1768 [00:48<00:00, 36.13it/s, loss=0.314]
Test Epoch 55 ==> 	accuracy: 0.9560, 	precision: 0.9848, 	recall: 0.7973, 	specificity: 0.9968, 	f1: 0.8812
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 56: 100%|██████████| 6507/6507 [07:05<00:00, 15.30it/s, loss=0.0247]
Train Epoch 56 ==> 	accuracy: 0.8826, 	precision: 0.9996, 	recall: 0.7655, 	specificity: 0.9997, 	f1: 0.8671
Test Epoch 56: 100%|██████████| 1768/1768 [00:48<00:00, 36.34it/s, loss=0.176]
Test Epoch 56 ==> 	accuracy: 0.9551, 	precision: 0.9886, 	recall: 0.7898, 	specificity: 0.9976, 	f1: 0.8781
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 57: 100%|██████████| 6507/6507 [07:15<00:00, 14.95it/s, loss=0.0081]
Train Epoch 57 ==> 	accuracy: 0.8883, 	precision: 0.9996, 	recall: 0.7770, 	specificity: 0.9997, 	f1: 0.8743
Test Epoch 57: 100%|██████████| 1768/1768 [00:46<00:00, 37.62it/s, loss=0.567]
Test Epoch 57 ==> 	accuracy: 0.9579, 	precision: 0.9826, 	recall: 0.8087, 	specificity: 0.9963, 	f1: 0.8872
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 58: 100%|██████████| 6507/6507 [07:14<00:00, 14.98it/s, loss=0.0143]
Train Epoch 58 ==> 	accuracy: 0.8849, 	precision: 0.9996, 	recall: 0.7702, 	specificity: 0.9997, 	f1: 0.8700
Test Epoch 58: 100%|██████████| 1768/1768 [00:46<00:00, 37.83it/s, loss=0.394]
Test Epoch 58 ==> 	accuracy: 0.9585, 	precision: 0.9842, 	recall: 0.8103, 	specificity: 0.9966, 	f1: 0.8888
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 59: 100%|██████████| 6507/6507 [07:06<00:00, 15.26it/s, loss=0.0395]
Train Epoch 59 ==> 	accuracy: 0.8878, 	precision: 0.9996, 	recall: 0.7760, 	specificity: 0.9997, 	f1: 0.8737
Test Epoch 59: 100%|██████████| 1768/1768 [00:46<00:00, 38.04it/s, loss=0.207]
Test Epoch 59 ==> 	accuracy: 0.9592, 	precision: 0.9812, 	recall: 0.8162, 	specificity: 0.9960, 	f1: 0.8911
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 60: 100%|██████████| 6507/6507 [07:07<00:00, 15.21it/s, loss=0.294]
Train Epoch 60 ==> 	accuracy: 0.8872, 	precision: 0.9996, 	recall: 0.7748, 	specificity: 0.9997, 	f1: 0.8729
Test Epoch 60: 100%|██████████| 1768/1768 [00:50<00:00, 35.18it/s, loss=0.191]
Test Epoch 60 ==> 	accuracy: 0.9566, 	precision: 0.9834, 	recall: 0.8016, 	specificity: 0.9965, 	f1: 0.8832
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 61: 100%|██████████| 6507/6507 [07:14<00:00, 14.97it/s, loss=0.0085]
Train Epoch 61 ==> 	accuracy: 0.8923, 	precision: 0.9996, 	recall: 0.7849, 	specificity: 0.9997, 	f1: 0.8793
Test Epoch 61: 100%|██████████| 1768/1768 [00:48<00:00, 36.31it/s, loss=0.221]
Test Epoch 61 ==> 	accuracy: 0.9573, 	precision: 0.9843, 	recall: 0.8043, 	specificity: 0.9967, 	f1: 0.8853
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 62: 100%|██████████| 6507/6507 [07:10<00:00, 15.11it/s, loss=0.506]
Train Epoch 62 ==> 	accuracy: 0.8874, 	precision: 0.9996, 	recall: 0.7751, 	specificity: 0.9997, 	f1: 0.8731
Test Epoch 62: 100%|██████████| 1768/1768 [00:45<00:00, 39.15it/s, loss=0.982]
Test Epoch 62 ==> 	accuracy: 0.9566, 	precision: 0.9842, 	recall: 0.8006, 	specificity: 0.9967, 	f1: 0.8829
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 63: 100%|██████████| 6507/6507 [07:10<00:00, 15.13it/s, loss=0.0176]
Train Epoch 63 ==> 	accuracy: 0.8917, 	precision: 0.9996, 	recall: 0.7838, 	specificity: 0.9997, 	f1: 0.8787
Test Epoch 63: 100%|██████████| 1768/1768 [00:48<00:00, 36.71it/s, loss=0.221]
Test Epoch 63 ==> 	accuracy: 0.9604, 	precision: 0.9808, 	recall: 0.8225, 	specificity: 0.9959, 	f1: 0.8947
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 64: 100%|██████████| 6507/6507 [07:10<00:00, 15.13it/s, loss=0.0025]
Train Epoch 64 ==> 	accuracy: 0.8913, 	precision: 0.9996, 	recall: 0.7829, 	specificity: 0.9997, 	f1: 0.8781
Test Epoch 64: 100%|██████████| 1768/1768 [00:46<00:00, 38.35it/s, loss=0.0905]
Test Epoch 64 ==> 	accuracy: 0.9581, 	precision: 0.9826, 	recall: 0.8094, 	specificity: 0.9963, 	f1: 0.8877
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 65: 100%|██████████| 6507/6507 [07:00<00:00, 15.48it/s, loss=0.391]
Train Epoch 65 ==> 	accuracy: 0.8932, 	precision: 0.9996, 	recall: 0.7868, 	specificity: 0.9997, 	f1: 0.8805
Test Epoch 65: 100%|██████████| 1768/1768 [00:45<00:00, 38.73it/s, loss=1.02]
Test Epoch 65 ==> 	accuracy: 0.9601, 	precision: 0.9785, 	recall: 0.8231, 	specificity: 0.9953, 	f1: 0.8941
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 66: 100%|██████████| 6507/6507 [07:13<00:00, 15.00it/s, loss=0.0122]
Train Epoch 66 ==> 	accuracy: 0.8919, 	precision: 0.9996, 	recall: 0.7841, 	specificity: 0.9997, 	f1: 0.8788
Test Epoch 66: 100%|██████████| 1768/1768 [00:50<00:00, 34.87it/s, loss=0.0562]
Test Epoch 66 ==> 	accuracy: 0.9589, 	precision: 0.9807, 	recall: 0.8153, 	specificity: 0.9959, 	f1: 0.8904
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 67: 100%|██████████| 6507/6507 [07:11<00:00, 15.09it/s, loss=0.018]
Train Epoch 67 ==> 	accuracy: 0.8949, 	precision: 0.9996, 	recall: 0.7901, 	specificity: 0.9997, 	f1: 0.8826
Test Epoch 67: 100%|██████████| 1768/1768 [00:45<00:00, 38.66it/s, loss=0.157]
Test Epoch 67 ==> 	accuracy: 0.9608, 	precision: 0.9789, 	recall: 0.8262, 	specificity: 0.9954, 	f1: 0.8961
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 68: 100%|██████████| 6507/6507 [06:58<00:00, 15.55it/s, loss=0.369]
Train Epoch 68 ==> 	accuracy: 0.8963, 	precision: 0.9996, 	recall: 0.7928, 	specificity: 0.9997, 	f1: 0.8843
Test Epoch 68: 100%|██████████| 1768/1768 [00:50<00:00, 35.33it/s, loss=0.494]
Test Epoch 68 ==> 	accuracy: 0.9591, 	precision: 0.9832, 	recall: 0.8137, 	specificity: 0.9964, 	f1: 0.8905
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 69: 100%|██████████| 6507/6507 [07:02<00:00, 15.39it/s, loss=0.0963]
Train Epoch 69 ==> 	accuracy: 0.8951, 	precision: 0.9997, 	recall: 0.7904, 	specificity: 0.9997, 	f1: 0.8828
Test Epoch 69: 100%|██████████| 1768/1768 [00:43<00:00, 40.73it/s, loss=0.128]
Test Epoch 69 ==> 	accuracy: 0.9619, 	precision: 0.9778, 	recall: 0.8325, 	specificity: 0.9951, 	f1: 0.8993
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 70: 100%|██████████| 6507/6507 [07:04<00:00, 15.32it/s, loss=0.0072]
Train Epoch 70 ==> 	accuracy: 0.8955, 	precision: 0.9996, 	recall: 0.7914, 	specificity: 0.9997, 	f1: 0.8834
Test Epoch 70: 100%|██████████| 1768/1768 [00:49<00:00, 35.77it/s, loss=0.138]
Test Epoch 70 ==> 	accuracy: 0.9604, 	precision: 0.9769, 	recall: 0.8260, 	specificity: 0.9950, 	f1: 0.8951
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 71: 100%|██████████| 6507/6507 [07:03<00:00, 15.35it/s, loss=0.0037]
Train Epoch 71 ==> 	accuracy: 0.8976, 	precision: 0.9997, 	recall: 0.7955, 	specificity: 0.9997, 	f1: 0.8859
Test Epoch 71: 100%|██████████| 1768/1768 [00:51<00:00, 34.25it/s, loss=2.01]
Test Epoch 71 ==> 	accuracy: 0.9613, 	precision: 0.9775, 	recall: 0.8300, 	specificity: 0.9951, 	f1: 0.8977
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 72: 100%|██████████| 6507/6507 [07:01<00:00, 15.43it/s, loss=0.0145]
Train Epoch 72 ==> 	accuracy: 0.8981, 	precision: 0.9997, 	recall: 0.7965, 	specificity: 0.9997, 	f1: 0.8866
Test Epoch 72: 100%|██████████| 1768/1768 [00:49<00:00, 35.69it/s, loss=0.164]
Test Epoch 72 ==> 	accuracy: 0.9622, 	precision: 0.9810, 	recall: 0.8315, 	specificity: 0.9959, 	f1: 0.9001
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 73: 100%|██████████| 6507/6507 [07:03<00:00, 15.37it/s, loss=0.003]
Train Epoch 73 ==> 	accuracy: 0.8990, 	precision: 0.9996, 	recall: 0.7982, 	specificity: 0.9997, 	f1: 0.8877
Test Epoch 73: 100%|██████████| 1768/1768 [00:45<00:00, 39.23it/s, loss=0.133]
Test Epoch 73 ==> 	accuracy: 0.9628, 	precision: 0.9804, 	recall: 0.8349, 	specificity: 0.9957, 	f1: 0.9018
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 74: 100%|██████████| 6507/6507 [07:05<00:00, 15.29it/s, loss=2.07]
Train Epoch 74 ==> 	accuracy: 0.9008, 	precision: 0.9996, 	recall: 0.8019, 	specificity: 0.9997, 	f1: 0.8899
Test Epoch 74: 100%|██████████| 1768/1768 [00:47<00:00, 37.25it/s, loss=1.13]
Test Epoch 74 ==> 	accuracy: 0.9633, 	precision: 0.9748, 	recall: 0.8424, 	specificity: 0.9944, 	f1: 0.9038
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 75: 100%|██████████| 6507/6507 [07:04<00:00, 15.31it/s, loss=0.124]
Train Epoch 75 ==> 	accuracy: 0.8997, 	precision: 0.9996, 	recall: 0.7998, 	specificity: 0.9997, 	f1: 0.8886
Test Epoch 75: 100%|██████████| 1768/1768 [00:50<00:00, 35.27it/s, loss=0.169]
Test Epoch 75 ==> 	accuracy: 0.9617, 	precision: 0.9776, 	recall: 0.8318, 	specificity: 0.9951, 	f1: 0.8988
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 76: 100%|██████████| 6507/6507 [06:59<00:00, 15.53it/s, loss=0.125]
Train Epoch 76 ==> 	accuracy: 0.9022, 	precision: 0.9997, 	recall: 0.8048, 	specificity: 0.9997, 	f1: 0.8917
Test Epoch 76: 100%|██████████| 1768/1768 [00:46<00:00, 37.71it/s, loss=0.247]
Test Epoch 76 ==> 	accuracy: 0.9609, 	precision: 0.9804, 	recall: 0.8255, 	specificity: 0.9958, 	f1: 0.8963
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 77: 100%|██████████| 6507/6507 [07:09<00:00, 15.16it/s, loss=0.144]
Train Epoch 77 ==> 	accuracy: 0.9010, 	precision: 0.9997, 	recall: 0.8024, 	specificity: 0.9997, 	f1: 0.8902
Test Epoch 77: 100%|██████████| 1768/1768 [00:43<00:00, 40.49it/s, loss=0.509]
Test Epoch 77 ==> 	accuracy: 0.9624, 	precision: 0.9776, 	recall: 0.8352, 	specificity: 0.9951, 	f1: 0.9008
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 78: 100%|██████████| 6507/6507 [07:03<00:00, 15.35it/s, loss=0.0006]
Train Epoch 78 ==> 	accuracy: 0.9023, 	precision: 0.9997, 	recall: 0.8048, 	specificity: 0.9997, 	f1: 0.8917
Test Epoch 78: 100%|██████████| 1768/1768 [00:46<00:00, 37.70it/s, loss=0.133]
Test Epoch 78 ==> 	accuracy: 0.9600, 	precision: 0.9856, 	recall: 0.8163, 	specificity: 0.9969, 	f1: 0.8930
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 79: 100%|██████████| 6507/6507 [07:08<00:00, 15.18it/s, loss=0.0261]
Train Epoch 79 ==> 	accuracy: 0.9003, 	precision: 0.9997, 	recall: 0.8008, 	specificity: 0.9997, 	f1: 0.8892
Test Epoch 79: 100%|██████████| 1768/1768 [00:48<00:00, 36.35it/s, loss=0.192]
Test Epoch 79 ==> 	accuracy: 0.9623, 	precision: 0.9782, 	recall: 0.8345, 	specificity: 0.9952, 	f1: 0.9006
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 80: 100%|██████████| 6507/6507 [07:04<00:00, 15.34it/s, loss=0.0072]
Train Epoch 80 ==> 	accuracy: 0.9020, 	precision: 0.9997, 	recall: 0.8043, 	specificity: 0.9997, 	f1: 0.8914
Test Epoch 80: 100%|██████████| 1768/1768 [00:49<00:00, 35.72it/s, loss=0.654]
Test Epoch 80 ==> 	accuracy: 0.9625, 	precision: 0.9820, 	recall: 0.8317, 	specificity: 0.9961, 	f1: 0.9006
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 81: 100%|██████████| 6507/6507 [07:08<00:00, 15.18it/s, loss=0.002]
Train Epoch 81 ==> 	accuracy: 0.9039, 	precision: 0.9997, 	recall: 0.8081, 	specificity: 0.9997, 	f1: 0.8938
Test Epoch 81: 100%|██████████| 1768/1768 [00:49<00:00, 35.72it/s, loss=0.0817]
Test Epoch 81 ==> 	accuracy: 0.9635, 	precision: 0.9809, 	recall: 0.8377, 	specificity: 0.9958, 	f1: 0.9037
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 82: 100%|██████████| 6507/6507 [07:06<00:00, 15.27it/s, loss=0.102]
Train Epoch 82 ==> 	accuracy: 0.9027, 	precision: 0.9997, 	recall: 0.8057, 	specificity: 0.9998, 	f1: 0.8923
Test Epoch 82: 100%|██████████| 1768/1768 [00:49<00:00, 35.89it/s, loss=0.19]
Test Epoch 82 ==> 	accuracy: 0.9627, 	precision: 0.9811, 	recall: 0.8338, 	specificity: 0.9959, 	f1: 0.9015
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 83: 100%|██████████| 6507/6507 [07:04<00:00, 15.32it/s, loss=0.0067]
Train Epoch 83 ==> 	accuracy: 0.9039, 	precision: 0.9997, 	recall: 0.8080, 	specificity: 0.9997, 	f1: 0.8937
Test Epoch 83: 100%|██████████| 1768/1768 [00:49<00:00, 35.63it/s, loss=0.878]
Test Epoch 83 ==> 	accuracy: 0.9633, 	precision: 0.9775, 	recall: 0.8398, 	specificity: 0.9950, 	f1: 0.9035
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 84: 100%|██████████| 6507/6507 [07:19<00:00, 14.81it/s, loss=0.0062]
Train Epoch 84 ==> 	accuracy: 0.9061, 	precision: 0.9997, 	recall: 0.8124, 	specificity: 0.9997, 	f1: 0.8964
Test Epoch 84: 100%|██████████| 1768/1768 [00:47<00:00, 36.86it/s, loss=0.161]
Test Epoch 84 ==> 	accuracy: 0.9626, 	precision: 0.9826, 	recall: 0.8317, 	specificity: 0.9962, 	f1: 0.9009
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 85: 100%|██████████| 6507/6507 [07:18<00:00, 14.84it/s, loss=0.104]
Train Epoch 85 ==> 	accuracy: 0.9071, 	precision: 0.9997, 	recall: 0.8145, 	specificity: 0.9998, 	f1: 0.8976
Test Epoch 85: 100%|██████████| 1768/1768 [00:49<00:00, 35.44it/s, loss=0.105]
Test Epoch 85 ==> 	accuracy: 0.9633, 	precision: 0.9787, 	recall: 0.8388, 	specificity: 0.9953, 	f1: 0.9034
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 86: 100%|██████████| 6507/6507 [07:01<00:00, 15.44it/s, loss=0.0233]
Train Epoch 86 ==> 	accuracy: 0.9061, 	precision: 0.9997, 	recall: 0.8125, 	specificity: 0.9997, 	f1: 0.8964
Test Epoch 86: 100%|██████████| 1768/1768 [00:50<00:00, 35.19it/s, loss=0.203]
Test Epoch 86 ==> 	accuracy: 0.9632, 	precision: 0.9753, 	recall: 0.8413, 	specificity: 0.9945, 	f1: 0.9033
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 87: 100%|██████████| 6507/6507 [07:12<00:00, 15.04it/s, loss=0.0018]
Train Epoch 87 ==> 	accuracy: 0.9067, 	precision: 0.9997, 	recall: 0.8136, 	specificity: 0.9997, 	f1: 0.8971
Test Epoch 87: 100%|██████████| 1768/1768 [00:50<00:00, 34.86it/s, loss=0.238]
Test Epoch 87 ==> 	accuracy: 0.9629, 	precision: 0.9760, 	recall: 0.8391, 	specificity: 0.9947, 	f1: 0.9024
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 88: 100%|██████████| 6507/6507 [07:12<00:00, 15.05it/s, loss=0.013]
Train Epoch 88 ==> 	accuracy: 0.9073, 	precision: 0.9997, 	recall: 0.8148, 	specificity: 0.9998, 	f1: 0.8979
Test Epoch 88: 100%|██████████| 1768/1768 [00:49<00:00, 35.58it/s, loss=0.128]
Test Epoch 88 ==> 	accuracy: 0.9641, 	precision: 0.9690, 	recall: 0.8517, 	specificity: 0.9930, 	f1: 0.9066
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 89: 100%|██████████| 6507/6507 [07:15<00:00, 14.93it/s, loss=0.0888]
Train Epoch 89 ==> 	accuracy: 0.9092, 	precision: 0.9997, 	recall: 0.8187, 	specificity: 0.9997, 	f1: 0.9002
Test Epoch 89: 100%|██████████| 1768/1768 [00:48<00:00, 36.13it/s, loss=0.11]
Test Epoch 89 ==> 	accuracy: 0.9641, 	precision: 0.9736, 	recall: 0.8473, 	specificity: 0.9941, 	f1: 0.9061
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 90: 100%|██████████| 6507/6507 [07:14<00:00, 14.97it/s, loss=0.0076]
Train Epoch 90 ==> 	accuracy: 0.9068, 	precision: 0.9997, 	recall: 0.8138, 	specificity: 0.9998, 	f1: 0.8972
Test Epoch 90: 100%|██████████| 1768/1768 [00:48<00:00, 36.54it/s, loss=0.0975]
Test Epoch 90 ==> 	accuracy: 0.9634, 	precision: 0.9748, 	recall: 0.8429, 	specificity: 0.9944, 	f1: 0.9041
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 91: 100%|██████████| 6507/6507 [07:11<00:00, 15.10it/s, loss=0.0393]
Train Epoch 91 ==> 	accuracy: 0.9105, 	precision: 0.9997, 	recall: 0.8213, 	specificity: 0.9997, 	f1: 0.9018
Test Epoch 91: 100%|██████████| 1768/1768 [00:47<00:00, 37.05it/s, loss=0.305]
Test Epoch 91 ==> 	accuracy: 0.9652, 	precision: 0.9712, 	recall: 0.8553, 	specificity: 0.9935, 	f1: 0.9095
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 92: 100%|██████████| 6507/6507 [07:12<00:00, 15.04it/s, loss=0.0022]
Train Epoch 92 ==> 	accuracy: 0.9074, 	precision: 0.9997, 	recall: 0.8150, 	specificity: 0.9997, 	f1: 0.8979
Test Epoch 92: 100%|██████████| 1768/1768 [00:50<00:00, 34.73it/s, loss=0.792]
Test Epoch 92 ==> 	accuracy: 0.9644, 	precision: 0.9756, 	recall: 0.8473, 	specificity: 0.9945, 	f1: 0.9069
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 93: 100%|██████████| 6507/6507 [07:15<00:00, 14.93it/s, loss=0.132]
Train Epoch 93 ==> 	accuracy: 0.9097, 	precision: 0.9997, 	recall: 0.8196, 	specificity: 0.9998, 	f1: 0.9007
Test Epoch 93: 100%|██████████| 1768/1768 [00:51<00:00, 34.22it/s, loss=0.141]
Test Epoch 93 ==> 	accuracy: 0.9649, 	precision: 0.9753, 	recall: 0.8502, 	specificity: 0.9945, 	f1: 0.9084
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 94: 100%|██████████| 6507/6507 [07:08<00:00, 15.17it/s, loss=0.391]
Train Epoch 94 ==> 	accuracy: 0.9084, 	precision: 0.9997, 	recall: 0.8169, 	specificity: 0.9998, 	f1: 0.8991
Test Epoch 94: 100%|██████████| 1768/1768 [00:48<00:00, 36.40it/s, loss=2.95]
Test Epoch 94 ==> 	accuracy: 0.9643, 	precision: 0.9786, 	recall: 0.8441, 	specificity: 0.9953, 	f1: 0.9064
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 95: 100%|██████████| 6507/6507 [07:14<00:00, 14.97it/s, loss=0.007]
Train Epoch 95 ==> 	accuracy: 0.9085, 	precision: 0.9997, 	recall: 0.8172, 	specificity: 0.9998, 	f1: 0.8993
Test Epoch 95: 100%|██████████| 1768/1768 [00:51<00:00, 34.45it/s, loss=1.76]
Test Epoch 95 ==> 	accuracy: 0.9641, 	precision: 0.9773, 	recall: 0.8442, 	specificity: 0.9950, 	f1: 0.9059
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 96: 100%|██████████| 6507/6507 [07:14<00:00, 14.97it/s, loss=0.0381]
Train Epoch 96 ==> 	accuracy: 0.9114, 	precision: 0.9997, 	recall: 0.8231, 	specificity: 0.9997, 	f1: 0.9028
Test Epoch 96: 100%|██████████| 1768/1768 [00:51<00:00, 34.37it/s, loss=0.196]
Test Epoch 96 ==> 	accuracy: 0.9646, 	precision: 0.9803, 	recall: 0.8441, 	specificity: 0.9956, 	f1: 0.9071
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 97: 100%|██████████| 6507/6507 [07:08<00:00, 15.20it/s, loss=0.0186]
Train Epoch 97 ==> 	accuracy: 0.9098, 	precision: 0.9997, 	recall: 0.8198, 	specificity: 0.9997, 	f1: 0.9009
Test Epoch 97: 100%|██████████| 1768/1768 [00:48<00:00, 36.76it/s, loss=0.0681]
Test Epoch 97 ==> 	accuracy: 0.9645, 	precision: 0.9779, 	recall: 0.8456, 	specificity: 0.9951, 	f1: 0.9069
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 98: 100%|██████████| 6507/6507 [07:09<00:00, 15.14it/s, loss=2.55]
Train Epoch 98 ==> 	accuracy: 0.9126, 	precision: 0.9997, 	recall: 0.8255, 	specificity: 0.9998, 	f1: 0.9043
Test Epoch 98: 100%|██████████| 1768/1768 [00:53<00:00, 32.75it/s, loss=0.105]
Test Epoch 98 ==> 	accuracy: 0.9647, 	precision: 0.9785, 	recall: 0.8461, 	specificity: 0.9952, 	f1: 0.9075
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 99: 100%|██████████| 6507/6507 [07:14<00:00, 14.99it/s, loss=0.0108]
Train Epoch 99 ==> 	accuracy: 0.9096, 	precision: 0.9997, 	recall: 0.8195, 	specificity: 0.9997, 	f1: 0.9006
Test Epoch 99: 100%|██████████| 1768/1768 [00:49<00:00, 35.64it/s, loss=0.185]
Test Epoch 99 ==> 	accuracy: 0.9643, 	precision: 0.9701, 	recall: 0.8519, 	specificity: 0.9932, 	f1: 0.9071
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 100: 100%|██████████| 6507/6507 [07:14<00:00, 14.97it/s, loss=0.328]
Train Epoch 100 ==> 	accuracy: 0.9137, 	precision: 0.9997, 	recall: 0.8277, 	specificity: 0.9998, 	f1: 0.9056
Test Epoch 100: 100%|██████████| 1768/1768 [00:49<00:00, 35.76it/s, loss=0.183]
Test Epoch 100 ==> 	accuracy: 0.9657, 	precision: 0.9735, 	recall: 0.8556, 	specificity: 0.9940, 	f1: 0.9108
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 101: 100%|██████████| 6507/6507 [07:18<00:00, 14.85it/s, loss=0.0892]
Train Epoch 101 ==> 	accuracy: 0.9141, 	precision: 0.9997, 	recall: 0.8285, 	specificity: 0.9998, 	f1: 0.9061
Test Epoch 101: 100%|██████████| 1768/1768 [00:50<00:00, 34.81it/s, loss=0.125]
Test Epoch 101 ==> 	accuracy: 0.9652, 	precision: 0.9705, 	recall: 0.8561, 	specificity: 0.9933, 	f1: 0.9097
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 102: 100%|██████████| 6507/6507 [07:13<00:00, 15.00it/s, loss=0.0148]
Train Epoch 102 ==> 	accuracy: 0.9142, 	precision: 0.9997, 	recall: 0.8286, 	specificity: 0.9998, 	f1: 0.9061
Test Epoch 102: 100%|██████████| 1768/1768 [00:48<00:00, 36.34it/s, loss=0.409]
Test Epoch 102 ==> 	accuracy: 0.9646, 	precision: 0.9757, 	recall: 0.8482, 	specificity: 0.9946, 	f1: 0.9075
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 103: 100%|██████████| 6507/6507 [07:17<00:00, 14.86it/s, loss=0.0281]
Train Epoch 103 ==> 	accuracy: 0.9124, 	precision: 0.9997, 	recall: 0.8251, 	specificity: 0.9998, 	f1: 0.9040
Test Epoch 103: 100%|██████████| 1768/1768 [00:50<00:00, 34.77it/s, loss=0.128]
Test Epoch 103 ==> 	accuracy: 0.9646, 	precision: 0.9766, 	recall: 0.8472, 	specificity: 0.9948, 	f1: 0.9073
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 104: 100%|██████████| 6507/6507 [07:30<00:00, 14.44it/s, loss=0.0218]
Train Epoch 104 ==> 	accuracy: 0.9144, 	precision: 0.9997, 	recall: 0.8290, 	specificity: 0.9998, 	f1: 0.9064
Test Epoch 104: 100%|██████████| 1768/1768 [00:50<00:00, 35.02it/s, loss=0.14]
Test Epoch 104 ==> 	accuracy: 0.9658, 	precision: 0.9743, 	recall: 0.8555, 	specificity: 0.9942, 	f1: 0.9111
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 105: 100%|██████████| 6507/6507 [07:17<00:00, 14.86it/s, loss=0.234]
Train Epoch 105 ==> 	accuracy: 0.9141, 	precision: 0.9997, 	recall: 0.8285, 	specificity: 0.9998, 	f1: 0.9061
Test Epoch 105: 100%|██████████| 1768/1768 [00:55<00:00, 31.59it/s, loss=0.237]
Test Epoch 105 ==> 	accuracy: 0.9652, 	precision: 0.9755, 	recall: 0.8513, 	specificity: 0.9945, 	f1: 0.9092
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 106: 100%|██████████| 6507/6507 [07:16<00:00, 14.89it/s, loss=0.0083]
Train Epoch 106 ==> 	accuracy: 0.9141, 	precision: 0.9997, 	recall: 0.8284, 	specificity: 0.9998, 	f1: 0.9060
Test Epoch 106: 100%|██████████| 1768/1768 [00:55<00:00, 32.01it/s, loss=1.44]
Test Epoch 106 ==> 	accuracy: 0.9662, 	precision: 0.9717, 	recall: 0.8596, 	specificity: 0.9936, 	f1: 0.9122
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 107: 100%|██████████| 6507/6507 [07:17<00:00, 14.87it/s, loss=0.0184]
Train Epoch 107 ==> 	accuracy: 0.9148, 	precision: 0.9997, 	recall: 0.8298, 	specificity: 0.9998, 	f1: 0.9069
Test Epoch 107: 100%|██████████| 1768/1768 [00:49<00:00, 35.83it/s, loss=0.587]
Test Epoch 107 ==> 	accuracy: 0.9658, 	precision: 0.9745, 	recall: 0.8551, 	specificity: 0.9943, 	f1: 0.9109
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 108: 100%|██████████| 6507/6507 [07:20<00:00, 14.78it/s, loss=0.0016]
Train Epoch 108 ==> 	accuracy: 0.9146, 	precision: 0.9997, 	recall: 0.8293, 	specificity: 0.9998, 	f1: 0.9066
Test Epoch 108: 100%|██████████| 1768/1768 [00:49<00:00, 35.90it/s, loss=0.0343]
Test Epoch 108 ==> 	accuracy: 0.9652, 	precision: 0.9728, 	recall: 0.8536, 	specificity: 0.9939, 	f1: 0.9093
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 109: 100%|██████████| 6507/6507 [07:24<00:00, 14.63it/s, loss=0.0374]
Train Epoch 109 ==> 	accuracy: 0.9151, 	precision: 0.9997, 	recall: 0.8304, 	specificity: 0.9998, 	f1: 0.9072
Test Epoch 109: 100%|██████████| 1768/1768 [00:49<00:00, 35.93it/s, loss=0.102]
Test Epoch 109 ==> 	accuracy: 0.9657, 	precision: 0.9715, 	recall: 0.8573, 	specificity: 0.9935, 	f1: 0.9108
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 110: 100%|██████████| 6507/6507 [07:21<00:00, 14.75it/s, loss=0.0258]
Train Epoch 110 ==> 	accuracy: 0.9145, 	precision: 0.9998, 	recall: 0.8292, 	specificity: 0.9998, 	f1: 0.9065
Test Epoch 110: 100%|██████████| 1768/1768 [00:49<00:00, 35.77it/s, loss=0.124]
Test Epoch 110 ==> 	accuracy: 0.9655, 	precision: 0.9689, 	recall: 0.8589, 	specificity: 0.9929, 	f1: 0.9106
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 111:  12%|█▏        | 799/6507 [00:37<04:36, 20.68it/s, loss=0.366]
Train Epoch 111: 100%|██████████| 6507/6507 [07:24<00:00, 14.65it/s, loss=0.0024]
Train Epoch 111 ==> 	accuracy: 0.9160, 	precision: 0.9997, 	recall: 0.8321, 	specificity: 0.9998, 	f1: 0.9083
Test Epoch 111: 100%|██████████| 1768/1768 [00:50<00:00, 35.03it/s, loss=0.178]
Test Epoch 111 ==> 	accuracy: 0.9660, 	precision: 0.9739, 	recall: 0.8568, 	specificity: 0.9941, 	f1: 0.9116
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 112: 100%|██████████| 6507/6507 [07:40<00:00, 14.13it/s, loss=1.82]
Train Epoch 112 ==> 	accuracy: 0.9139, 	precision: 0.9997, 	recall: 0.8279, 	specificity: 0.9998, 	f1: 0.9058
Test Epoch 112: 100%|██████████| 1768/1768 [00:49<00:00, 35.67it/s, loss=0.0676]
Test Epoch 112 ==> 	accuracy: 0.9656, 	precision: 0.9756, 	recall: 0.8533, 	specificity: 0.9945, 	f1: 0.9104
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 113: 100%|██████████| 6507/6507 [07:37<00:00, 14.22it/s, loss=0.0207]
Train Epoch 113 ==> 	accuracy: 0.9148, 	precision: 0.9997, 	recall: 0.8298, 	specificity: 0.9998, 	f1: 0.9069
Test Epoch 113: 100%|██████████| 1768/1768 [00:49<00:00, 35.63it/s, loss=1.07]
Test Epoch 113 ==> 	accuracy: 0.9658, 	precision: 0.9701, 	recall: 0.8591, 	specificity: 0.9932, 	f1: 0.9112
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 114: 100%|██████████| 6507/6507 [07:30<00:00, 14.43it/s, loss=0.0382]
Train Epoch 114 ==> 	accuracy: 0.9157, 	precision: 0.9997, 	recall: 0.8317, 	specificity: 0.9998, 	f1: 0.9080
Test Epoch 114: 100%|██████████| 1768/1768 [00:51<00:00, 34.15it/s, loss=0.247]
Test Epoch 114 ==> 	accuracy: 0.9659, 	precision: 0.9692, 	recall: 0.8606, 	specificity: 0.9930, 	f1: 0.9117
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 115: 100%|██████████| 6507/6507 [07:35<00:00, 14.30it/s, loss=0.0038]
Train Epoch 115 ==> 	accuracy: 0.9171, 	precision: 0.9997, 	recall: 0.8345, 	specificity: 0.9998, 	f1: 0.9097
Test Epoch 115: 100%|██████████| 1768/1768 [00:49<00:00, 35.99it/s, loss=0.785]
Test Epoch 115 ==> 	accuracy: 0.9665, 	precision: 0.9745, 	recall: 0.8586, 	specificity: 0.9942, 	f1: 0.9129
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 116: 100%|██████████| 6507/6507 [07:38<00:00, 14.19it/s, loss=0.0011]
Train Epoch 116 ==> 	accuracy: 0.9148, 	precision: 0.9997, 	recall: 0.8299, 	specificity: 0.9998, 	f1: 0.9069
Test Epoch 116: 100%|██████████| 1768/1768 [00:48<00:00, 36.28it/s, loss=0.076]
Test Epoch 116 ==> 	accuracy: 0.9659, 	precision: 0.9772, 	recall: 0.8534, 	specificity: 0.9949, 	f1: 0.9111
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 117: 100%|██████████| 6507/6507 [07:32<00:00, 14.37it/s, loss=0.0503]
Train Epoch 117 ==> 	accuracy: 0.9186, 	precision: 0.9998, 	recall: 0.8374, 	specificity: 0.9998, 	f1: 0.9114
Test Epoch 117: 100%|██████████| 1768/1768 [00:48<00:00, 36.70it/s, loss=0.0818]
Test Epoch 117 ==> 	accuracy: 0.9656, 	precision: 0.9713, 	recall: 0.8572, 	specificity: 0.9935, 	f1: 0.9107
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 118: 100%|██████████| 6507/6507 [07:39<00:00, 14.16it/s, loss=0.0029]
Train Epoch 118 ==> 	accuracy: 0.9164, 	precision: 0.9997, 	recall: 0.8329, 	specificity: 0.9998, 	f1: 0.9087
Test Epoch 118: 100%|██████████| 1768/1768 [00:52<00:00, 33.65it/s, loss=4.5]
Test Epoch 118 ==> 	accuracy: 0.9671, 	precision: 0.9775, 	recall: 0.8587, 	specificity: 0.9949, 	f1: 0.9143
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 119: 100%|██████████| 6507/6507 [07:33<00:00, 14.36it/s, loss=0.0187]
Train Epoch 119 ==> 	accuracy: 0.9162, 	precision: 0.9997, 	recall: 0.8327, 	specificity: 0.9998, 	f1: 0.9086
Test Epoch 119: 100%|██████████| 1768/1768 [00:50<00:00, 34.74it/s, loss=0.13]
Test Epoch 119 ==> 	accuracy: 0.9674, 	precision: 0.9769, 	recall: 0.8608, 	specificity: 0.9948, 	f1: 0.9152
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 120: 100%|██████████| 6507/6507 [07:33<00:00, 14.36it/s, loss=0.0421]
Train Epoch 120 ==> 	accuracy: 0.9163, 	precision: 0.9998, 	recall: 0.8328, 	specificity: 0.9998, 	f1: 0.9086
Test Epoch 120: 100%|██████████| 1768/1768 [00:51<00:00, 34.23it/s, loss=1.09]
Test Epoch 120 ==> 	accuracy: 0.9671, 	precision: 0.9769, 	recall: 0.8595, 	specificity: 0.9948, 	f1: 0.9144
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 121: 100%|██████████| 6507/6507 [07:34<00:00, 14.32it/s, loss=0.0222]
Train Epoch 121 ==> 	accuracy: 0.9182, 	precision: 0.9997, 	recall: 0.8365, 	specificity: 0.9998, 	f1: 0.9109
Test Epoch 121: 100%|██████████| 1768/1768 [00:49<00:00, 35.63it/s, loss=5.4]
Test Epoch 121 ==> 	accuracy: 0.9678, 	precision: 0.9769, 	recall: 0.8631, 	specificity: 0.9947, 	f1: 0.9165
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 122: 100%|██████████| 6507/6507 [07:32<00:00, 14.38it/s, loss=0.0251]
Train Epoch 122 ==> 	accuracy: 0.9179, 	precision: 0.9997, 	recall: 0.8361, 	specificity: 0.9998, 	f1: 0.9106
Test Epoch 122: 100%|██████████| 1768/1768 [00:49<00:00, 35.74it/s, loss=1.47]
Test Epoch 122 ==> 	accuracy: 0.9665, 	precision: 0.9778, 	recall: 0.8559, 	specificity: 0.9950, 	f1: 0.9128
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 123: 100%|██████████| 6507/6507 [07:33<00:00, 14.35it/s, loss=0.0121]
Train Epoch 123 ==> 	accuracy: 0.9163, 	precision: 0.9998, 	recall: 0.8327, 	specificity: 0.9998, 	f1: 0.9086
Test Epoch 123: 100%|██████████| 1768/1768 [00:51<00:00, 34.19it/s, loss=0.196]
Test Epoch 123 ==> 	accuracy: 0.9668, 	precision: 0.9758, 	recall: 0.8591, 	specificity: 0.9945, 	f1: 0.9137
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 124: 100%|██████████| 6507/6507 [07:32<00:00, 14.39it/s, loss=0.0031]
Train Epoch 124 ==> 	accuracy: 0.9194, 	precision: 0.9998, 	recall: 0.8391, 	specificity: 0.9998, 	f1: 0.9124
Test Epoch 124: 100%|██████████| 1768/1768 [00:50<00:00, 35.02it/s, loss=0.11]
Test Epoch 124 ==> 	accuracy: 0.9675, 	precision: 0.9756, 	recall: 0.8628, 	specificity: 0.9945, 	f1: 0.9158
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 125: 100%|██████████| 6507/6507 [07:33<00:00, 14.34it/s, loss=0.0102]
Train Epoch 125 ==> 	accuracy: 0.9177, 	precision: 0.9998, 	recall: 0.8356, 	specificity: 0.9998, 	f1: 0.9104
Test Epoch 125: 100%|██████████| 1768/1768 [00:50<00:00, 34.69it/s, loss=1.17]
Test Epoch 125 ==> 	accuracy: 0.9667, 	precision: 0.9733, 	recall: 0.8609, 	specificity: 0.9939, 	f1: 0.9137
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 126: 100%|██████████| 6507/6507 [07:39<00:00, 14.16it/s, loss=0.0242]
Train Epoch 126 ==> 	accuracy: 0.9189, 	precision: 0.9998, 	recall: 0.8380, 	specificity: 0.9998, 	f1: 0.9117
Test Epoch 126: 100%|██████████| 1768/1768 [00:50<00:00, 35.29it/s, loss=0.123]
Test Epoch 126 ==> 	accuracy: 0.9674, 	precision: 0.9737, 	recall: 0.8640, 	specificity: 0.9940, 	f1: 0.9156
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 127: 100%|██████████| 6507/6507 [07:29<00:00, 14.48it/s, loss=0.381]
Train Epoch 127 ==> 	accuracy: 0.9191, 	precision: 0.9997, 	recall: 0.8384, 	specificity: 0.9998, 	f1: 0.9120
Test Epoch 127: 100%|██████████| 1768/1768 [00:49<00:00, 35.63it/s, loss=0.0769]
Test Epoch 127 ==> 	accuracy: 0.9674, 	precision: 0.9715, 	recall: 0.8658, 	specificity: 0.9935, 	f1: 0.9156
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 128: 100%|██████████| 6507/6507 [07:37<00:00, 14.23it/s, loss=0.0438]
Train Epoch 128 ==> 	accuracy: 0.9189, 	precision: 0.9997, 	recall: 0.8380, 	specificity: 0.9998, 	f1: 0.9117
Test Epoch 128: 100%|██████████| 1768/1768 [00:50<00:00, 35.32it/s, loss=0.0773]
Test Epoch 128 ==> 	accuracy: 0.9676, 	precision: 0.9780, 	recall: 0.8611, 	specificity: 0.9950, 	f1: 0.9158
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 129: 100%|██████████| 6507/6507 [07:27<00:00, 14.53it/s, loss=1.73]
Train Epoch 129 ==> 	accuracy: 0.9171, 	precision: 0.9997, 	recall: 0.8344, 	specificity: 0.9998, 	f1: 0.9096
Test Epoch 129: 100%|██████████| 1768/1768 [00:51<00:00, 34.30it/s, loss=0.152]
Test Epoch 129 ==> 	accuracy: 0.9678, 	precision: 0.9758, 	recall: 0.8640, 	specificity: 0.9945, 	f1: 0.9165
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 130: 100%|██████████| 6507/6507 [07:32<00:00, 14.37it/s, loss=0.209]
Train Epoch 130 ==> 	accuracy: 0.9191, 	precision: 0.9998, 	recall: 0.8383, 	specificity: 0.9998, 	f1: 0.9119
Test Epoch 130: 100%|██████████| 1768/1768 [00:50<00:00, 34.99it/s, loss=0.16]
Test Epoch 130 ==> 	accuracy: 0.9672, 	precision: 0.9777, 	recall: 0.8591, 	specificity: 0.9950, 	f1: 0.9146
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 131: 100%|██████████| 6507/6507 [07:36<00:00, 14.27it/s, loss=0.0042]
Train Epoch 131 ==> 	accuracy: 0.9181, 	precision: 0.9998, 	recall: 0.8365, 	specificity: 0.9998, 	f1: 0.9109
Test Epoch 131: 100%|██████████| 1768/1768 [00:51<00:00, 34.52it/s, loss=0.109]
Test Epoch 131 ==> 	accuracy: 0.9676, 	precision: 0.9759, 	recall: 0.8630, 	specificity: 0.9945, 	f1: 0.9160
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 132: 100%|██████████| 6507/6507 [07:27<00:00, 14.54it/s, loss=0.0157]
Train Epoch 132 ==> 	accuracy: 0.9198, 	precision: 0.9998, 	recall: 0.8397, 	specificity: 0.9998, 	f1: 0.9128
Test Epoch 132: 100%|██████████| 1768/1768 [00:50<00:00, 34.97it/s, loss=3.27]
Test Epoch 132 ==> 	accuracy: 0.9680, 	precision: 0.9733, 	recall: 0.8672, 	specificity: 0.9939, 	f1: 0.9172
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 133: 100%|██████████| 6507/6507 [07:33<00:00, 14.36it/s, loss=0.0149]
Train Epoch 133 ==> 	accuracy: 0.9192, 	precision: 0.9998, 	recall: 0.8386, 	specificity: 0.9998, 	f1: 0.9121
Test Epoch 133: 100%|██████████| 1768/1768 [00:49<00:00, 35.72it/s, loss=0.148]
Test Epoch 133 ==> 	accuracy: 0.9672, 	precision: 0.9749, 	recall: 0.8617, 	specificity: 0.9943, 	f1: 0.9148
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 134: 100%|██████████| 6507/6507 [07:31<00:00, 14.42it/s, loss=0.0067]
Train Epoch 134 ==> 	accuracy: 0.9205, 	precision: 0.9998, 	recall: 0.8411, 	specificity: 0.9998, 	f1: 0.9136
Test Epoch 134: 100%|██████████| 1768/1768 [00:54<00:00, 32.32it/s, loss=0.0437]
Test Epoch 134 ==> 	accuracy: 0.9676, 	precision: 0.9756, 	recall: 0.8632, 	specificity: 0.9945, 	f1: 0.9160
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 135: 100%|██████████| 6507/6507 [07:29<00:00, 14.49it/s, loss=0.025]
Train Epoch 135 ==> 	accuracy: 0.9212, 	precision: 0.9998, 	recall: 0.8426, 	specificity: 0.9998, 	f1: 0.9145
Test Epoch 135: 100%|██████████| 1768/1768 [00:50<00:00, 34.97it/s, loss=0.0444]
Test Epoch 135 ==> 	accuracy: 0.9678, 	precision: 0.9731, 	recall: 0.8664, 	specificity: 0.9938, 	f1: 0.9166
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 136: 100%|██████████| 6507/6507 [07:33<00:00, 14.35it/s, loss=0.147]
Train Epoch 136 ==> 	accuracy: 0.9204, 	precision: 0.9998, 	recall: 0.8410, 	specificity: 0.9998, 	f1: 0.9135
Test Epoch 136: 100%|██████████| 1768/1768 [00:48<00:00, 36.30it/s, loss=0.118]
Test Epoch 136 ==> 	accuracy: 0.9669, 	precision: 0.9733, 	recall: 0.8617, 	specificity: 0.9939, 	f1: 0.9141
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 137: 100%|██████████| 6507/6507 [07:19<00:00, 14.82it/s, loss=0.0091]
Train Epoch 137 ==> 	accuracy: 0.9207, 	precision: 0.9998, 	recall: 0.8417, 	specificity: 0.9998, 	f1: 0.9139
Test Epoch 137: 100%|██████████| 1768/1768 [00:53<00:00, 32.75it/s, loss=0.09]
Test Epoch 137 ==> 	accuracy: 0.9676, 	precision: 0.9733, 	recall: 0.8653, 	specificity: 0.9939, 	f1: 0.9162
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 138: 100%|██████████| 6507/6507 [07:20<00:00, 14.77it/s, loss=0.0183]
Train Epoch 138 ==> 	accuracy: 0.9213, 	precision: 0.9998, 	recall: 0.8428, 	specificity: 0.9998, 	f1: 0.9146
Test Epoch 138: 100%|██████████| 1768/1768 [00:49<00:00, 35.49it/s, loss=0.119]
Test Epoch 138 ==> 	accuracy: 0.9673, 	precision: 0.9724, 	recall: 0.8648, 	specificity: 0.9937, 	f1: 0.9154
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 139: 100%|██████████| 6507/6507 [07:22<00:00, 14.71it/s, loss=0.05]
Train Epoch 139 ==> 	accuracy: 0.9212, 	precision: 0.9998, 	recall: 0.8426, 	specificity: 0.9998, 	f1: 0.9144
Test Epoch 139: 100%|██████████| 1768/1768 [00:50<00:00, 35.19it/s, loss=0.0686]
Test Epoch 139 ==> 	accuracy: 0.9674, 	precision: 0.9731, 	recall: 0.8645, 	specificity: 0.9939, 	f1: 0.9156
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 140: 100%|██████████| 6507/6507 [07:27<00:00, 14.53it/s, loss=2.55]
Train Epoch 140 ==> 	accuracy: 0.9190, 	precision: 0.9997, 	recall: 0.8383, 	specificity: 0.9998, 	f1: 0.9119
Test Epoch 140: 100%|██████████| 1768/1768 [00:49<00:00, 36.05it/s, loss=0.0935]
Test Epoch 140 ==> 	accuracy: 0.9673, 	precision: 0.9755, 	recall: 0.8618, 	specificity: 0.9944, 	f1: 0.9151
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 141: 100%|██████████| 6507/6507 [07:32<00:00, 14.37it/s, loss=0.0069]
Train Epoch 141 ==> 	accuracy: 0.9212, 	precision: 0.9998, 	recall: 0.8426, 	specificity: 0.9998, 	f1: 0.9145
Test Epoch 141: 100%|██████████| 1768/1768 [00:48<00:00, 36.09it/s, loss=0.168]
Test Epoch 141 ==> 	accuracy: 0.9675, 	precision: 0.9753, 	recall: 0.8628, 	specificity: 0.9944, 	f1: 0.9156
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 142: 100%|██████████| 6507/6507 [07:21<00:00, 14.73it/s, loss=0.0499]
Train Epoch 142 ==> 	accuracy: 0.9200, 	precision: 0.9997, 	recall: 0.8402, 	specificity: 0.9998, 	f1: 0.9131
Test Epoch 142: 100%|██████████| 1768/1768 [00:52<00:00, 33.44it/s, loss=0.277]
Test Epoch 142 ==> 	accuracy: 0.9670, 	precision: 0.9742, 	recall: 0.8617, 	specificity: 0.9941, 	f1: 0.9145
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 143: 100%|██████████| 6507/6507 [07:28<00:00, 14.50it/s, loss=0.0069]
Train Epoch 143 ==> 	accuracy: 0.9224, 	precision: 0.9998, 	recall: 0.8451, 	specificity: 0.9998, 	f1: 0.9159
Test Epoch 143: 100%|██████████| 1768/1768 [00:48<00:00, 36.53it/s, loss=0.196]
Test Epoch 143 ==> 	accuracy: 0.9682, 	precision: 0.9732, 	recall: 0.8683, 	specificity: 0.9939, 	f1: 0.9177
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 144: 100%|██████████| 6507/6507 [07:17<00:00, 14.88it/s, loss=0.022]
Train Epoch 144 ==> 	accuracy: 0.9204, 	precision: 0.9998, 	recall: 0.8409, 	specificity: 0.9998, 	f1: 0.9135
Test Epoch 144: 100%|██████████| 1768/1768 [00:49<00:00, 36.01it/s, loss=0.0966]
Test Epoch 144 ==> 	accuracy: 0.9679, 	precision: 0.9759, 	recall: 0.8642, 	specificity: 0.9945, 	f1: 0.9167
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 145: 100%|██████████| 6507/6507 [07:24<00:00, 14.64it/s, loss=0.195]
Train Epoch 145 ==> 	accuracy: 0.9208, 	precision: 0.9997, 	recall: 0.8418, 	specificity: 0.9998, 	f1: 0.9140
Test Epoch 145: 100%|██████████| 1768/1768 [00:50<00:00, 35.15it/s, loss=0.105]
Test Epoch 145 ==> 	accuracy: 0.9682, 	precision: 0.9751, 	recall: 0.8665, 	specificity: 0.9943, 	f1: 0.9176
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 146: 100%|██████████| 6507/6507 [07:21<00:00, 14.75it/s, loss=2.52]
Train Epoch 146 ==> 	accuracy: 0.9204, 	precision: 0.9997, 	recall: 0.8409, 	specificity: 0.9998, 	f1: 0.9135
Test Epoch 146: 100%|██████████| 1768/1768 [00:48<00:00, 36.59it/s, loss=2.86]
Test Epoch 146 ==> 	accuracy: 0.9680, 	precision: 0.9744, 	recall: 0.8664, 	specificity: 0.9941, 	f1: 0.9172
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 147: 100%|██████████| 6507/6507 [07:29<00:00, 14.47it/s, loss=0.168]
Train Epoch 147 ==> 	accuracy: 0.9210, 	precision: 0.9997, 	recall: 0.8422, 	specificity: 0.9998, 	f1: 0.9142
Test Epoch 147: 100%|██████████| 1768/1768 [00:49<00:00, 35.86it/s, loss=0.185]
Test Epoch 147 ==> 	accuracy: 0.9673, 	precision: 0.9779, 	recall: 0.8597, 	specificity: 0.9950, 	f1: 0.9150
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 148: 100%|██████████| 6507/6507 [07:19<00:00, 14.79it/s, loss=0.0037]
Train Epoch 148 ==> 	accuracy: 0.9233, 	precision: 0.9998, 	recall: 0.8467, 	specificity: 0.9998, 	f1: 0.9169
Test Epoch 148: 100%|██████████| 1768/1768 [00:50<00:00, 34.94it/s, loss=0.662]
Test Epoch 148 ==> 	accuracy: 0.9682, 	precision: 0.9675, 	recall: 0.8740, 	specificity: 0.9924, 	f1: 0.9183
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 149: 100%|██████████| 6507/6507 [07:22<00:00, 14.71it/s, loss=0.158]
Train Epoch 149 ==> 	accuracy: 0.9212, 	precision: 0.9998, 	recall: 0.8426, 	specificity: 0.9998, 	f1: 0.9145
Test Epoch 149: 100%|██████████| 1768/1768 [00:50<00:00, 34.70it/s, loss=0.595]
Test Epoch 149 ==> 	accuracy: 0.9674, 	precision: 0.9686, 	recall: 0.8686, 	specificity: 0.9928, 	f1: 0.9159
Adjusting learning rate of group 0 to 5.8150e-06.

进程已结束，退出代码为 0

'''

'''
'../model_save_sigBlock4_focalWithMs_deformable_7mer_ab_deformable_mha'
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6507/6507 [03:43<00:00, 29.06it/s, loss=0.102]
Train Epoch 0 ==> 	accuracy: 0.6467, 	precision: 0.9961, 	recall: 0.2946, 	specificity: 0.9988, 	f1: 0.4547
Test Epoch 0: 100%|██████████| 1768/1768 [00:27<00:00, 64.39it/s, loss=0.544]
Test Epoch 0 ==> 	accuracy: 0.9020, 	precision: 0.9765, 	recall: 0.5336, 	specificity: 0.9967, 	f1: 0.6901
Train Epoch 1: 100%|██████████| 6507/6507 [04:34<00:00, 23.67it/s, loss=0.0936]
Train Epoch 1 ==> 	accuracy: 0.7493, 	precision: 0.9976, 	recall: 0.4999, 	specificity: 0.9988, 	f1: 0.6660
Test Epoch 1: 100%|██████████| 1768/1768 [00:33<00:00, 53.03it/s, loss=0.221]
Test Epoch 1 ==> 	accuracy: 0.9208, 	precision: 0.9792, 	recall: 0.6260, 	specificity: 0.9966, 	f1: 0.7637
Train Epoch 2: 100%|██████████| 6507/6507 [04:52<00:00, 22.23it/s, loss=0.0296]
Train Epoch 2 ==> 	accuracy: 0.7744, 	precision: 0.9981, 	recall: 0.5497, 	specificity: 0.9990, 	f1: 0.7090
Test Epoch 2: 100%|██████████| 1768/1768 [00:33<00:00, 52.34it/s, loss=0.483]
Test Epoch 2 ==> 	accuracy: 0.9257, 	precision: 0.9840, 	recall: 0.6472, 	specificity: 0.9973, 	f1: 0.7809
Train Epoch 3: 100%|██████████| 6507/6507 [04:48<00:00, 22.59it/s, loss=0.176]
Train Epoch 3 ==> 	accuracy: 0.7850, 	precision: 0.9983, 	recall: 0.5710, 	specificity: 0.9990, 	f1: 0.7265
Test Epoch 3: 100%|██████████| 1768/1768 [00:32<00:00, 54.91it/s, loss=0.529]
Test Epoch 3 ==> 	accuracy: 0.9251, 	precision: 0.9785, 	recall: 0.6481, 	specificity: 0.9963, 	f1: 0.7798
Train Epoch 4: 100%|██████████| 6507/6507 [04:55<00:00, 22.03it/s, loss=0.247]
Train Epoch 4 ==> 	accuracy: 0.7961, 	precision: 0.9985, 	recall: 0.5931, 	specificity: 0.9991, 	f1: 0.7442
Test Epoch 4: 100%|██████████| 1768/1768 [00:36<00:00, 48.10it/s, loss=0.236]
Test Epoch 4 ==> 	accuracy: 0.9278, 	precision: 0.9680, 	recall: 0.6693, 	specificity: 0.9943, 	f1: 0.7914
Train Epoch 5: 100%|██████████| 6507/6507 [04:51<00:00, 22.30it/s, loss=0.092]
Train Epoch 5 ==> 	accuracy: 0.8010, 	precision: 0.9985, 	recall: 0.6029, 	specificity: 0.9991, 	f1: 0.7519
Test Epoch 5: 100%|██████████| 1768/1768 [00:34<00:00, 51.58it/s, loss=0.232]
Test Epoch 5 ==> 	accuracy: 0.9276, 	precision: 0.9805, 	recall: 0.6594, 	specificity: 0.9966, 	f1: 0.7885
Train Epoch 6: 100%|██████████| 6507/6507 [05:03<00:00, 21.43it/s, loss=0.0011]
Train Epoch 6 ==> 	accuracy: 0.8057, 	precision: 0.9987, 	recall: 0.6122, 	specificity: 0.9992, 	f1: 0.7591
Test Epoch 6: 100%|██████████| 1768/1768 [00:34<00:00, 50.95it/s, loss=0.167]
Test Epoch 6 ==> 	accuracy: 0.9323, 	precision: 0.9768, 	recall: 0.6854, 	specificity: 0.9958, 	f1: 0.8056
Train Epoch 7: 100%|██████████| 6507/6507 [04:57<00:00, 21.87it/s, loss=0.0679]
Train Epoch 7 ==> 	accuracy: 0.8115, 	precision: 0.9987, 	recall: 0.6239, 	specificity: 0.9992, 	f1: 0.7680
Test Epoch 7: 100%|██████████| 1768/1768 [00:38<00:00, 45.51it/s, loss=0.199]
Test Epoch 7 ==> 	accuracy: 0.9261, 	precision: 0.9926, 	recall: 0.6435, 	specificity: 0.9988, 	f1: 0.7808
Train Epoch 8: 100%|██████████| 6507/6507 [04:56<00:00, 21.94it/s, loss=0.121]
Train Epoch 8 ==> 	accuracy: 0.8093, 	precision: 0.9988, 	recall: 0.6193, 	specificity: 0.9993, 	f1: 0.7646
Test Epoch 8: 100%|██████████| 1768/1768 [00:36<00:00, 48.05it/s, loss=0.219]
Test Epoch 8 ==> 	accuracy: 0.9298, 	precision: 0.9838, 	recall: 0.6677, 	specificity: 0.9972, 	f1: 0.7955
Train Epoch 9: 100%|██████████| 6507/6507 [04:53<00:00, 22.20it/s, loss=0.0999]
Train Epoch 9 ==> 	accuracy: 0.8160, 	precision: 0.9989, 	recall: 0.6327, 	specificity: 0.9993, 	f1: 0.7747
Test Epoch 9: 100%|██████████| 1768/1768 [00:35<00:00, 49.94it/s, loss=0.243]
Test Epoch 9 ==> 	accuracy: 0.9252, 	precision: 0.9886, 	recall: 0.6415, 	specificity: 0.9981, 	f1: 0.7781
Train Epoch 10: 100%|██████████| 6507/6507 [04:55<00:00, 22.02it/s, loss=0.122]
Train Epoch 10 ==> 	accuracy: 0.8182, 	precision: 0.9989, 	recall: 0.6372, 	specificity: 0.9993, 	f1: 0.7781
Test Epoch 10: 100%|██████████| 1768/1768 [00:33<00:00, 52.24it/s, loss=0.154]
Test Epoch 10 ==> 	accuracy: 0.9344, 	precision: 0.9762, 	recall: 0.6961, 	specificity: 0.9956, 	f1: 0.8127
Train Epoch 11: 100%|██████████| 6507/6507 [04:57<00:00, 21.88it/s, loss=0.0865]
Train Epoch 11 ==> 	accuracy: 0.8233, 	precision: 0.9988, 	recall: 0.6474, 	specificity: 0.9992, 	f1: 0.7856
Test Epoch 11: 100%|██████████| 1768/1768 [00:35<00:00, 49.87it/s, loss=0.162]
Test Epoch 11 ==> 	accuracy: 0.9311, 	precision: 0.9730, 	recall: 0.6821, 	specificity: 0.9951, 	f1: 0.8020
Train Epoch 12: 100%|██████████| 6507/6507 [04:49<00:00, 22.45it/s, loss=0.0769]
Train Epoch 12 ==> 	accuracy: 0.8195, 	precision: 0.9990, 	recall: 0.6396, 	specificity: 0.9993, 	f1: 0.7799
Test Epoch 12: 100%|██████████| 1768/1768 [00:36<00:00, 48.03it/s, loss=0.485]
Test Epoch 12 ==> 	accuracy: 0.9337, 	precision: 0.9878, 	recall: 0.6845, 	specificity: 0.9978, 	f1: 0.8086
Train Epoch 13: 100%|██████████| 6507/6507 [05:00<00:00, 21.63it/s, loss=0.199]
Train Epoch 13 ==> 	accuracy: 0.8226, 	precision: 0.9990, 	recall: 0.6459, 	specificity: 0.9993, 	f1: 0.7845
Test Epoch 13: 100%|██████████| 1768/1768 [00:34<00:00, 50.71it/s, loss=0.295]
Test Epoch 13 ==> 	accuracy: 0.9355, 	precision: 0.9793, 	recall: 0.6995, 	specificity: 0.9962, 	f1: 0.8161
Train Epoch 14: 100%|██████████| 6507/6507 [04:50<00:00, 22.39it/s, loss=0.0995]
Train Epoch 14 ==> 	accuracy: 0.8289, 	precision: 0.9990, 	recall: 0.6585, 	specificity: 0.9994, 	f1: 0.7938
Test Epoch 14: 100%|██████████| 1768/1768 [00:34<00:00, 51.86it/s, loss=0.214]
Test Epoch 14 ==> 	accuracy: 0.9374, 	precision: 0.9759, 	recall: 0.7115, 	specificity: 0.9955, 	f1: 0.8230
Train Epoch 15: 100%|██████████| 6507/6507 [04:56<00:00, 21.91it/s, loss=0.103]
Train Epoch 15 ==> 	accuracy: 0.8266, 	precision: 0.9991, 	recall: 0.6538, 	specificity: 0.9994, 	f1: 0.7904
Test Epoch 15: 100%|██████████| 1768/1768 [00:34<00:00, 50.63it/s, loss=0.196]
Test Epoch 15 ==> 	accuracy: 0.9338, 	precision: 0.9884, 	recall: 0.6843, 	specificity: 0.9979, 	f1: 0.8087
Train Epoch 16: 100%|██████████| 6507/6507 [05:00<00:00, 21.66it/s, loss=0.0464]
Train Epoch 16 ==> 	accuracy: 0.8265, 	precision: 0.9991, 	recall: 0.6537, 	specificity: 0.9994, 	f1: 0.7903
Test Epoch 16: 100%|██████████| 1768/1768 [00:31<00:00, 56.62it/s, loss=0.108]
Test Epoch 16 ==> 	accuracy: 0.9400, 	precision: 0.9824, 	recall: 0.7197, 	specificity: 0.9967, 	f1: 0.8308
Train Epoch 17: 100%|██████████| 6507/6507 [04:45<00:00, 22.81it/s, loss=0.0773]
Train Epoch 17 ==> 	accuracy: 0.8305, 	precision: 0.9990, 	recall: 0.6616, 	specificity: 0.9994, 	f1: 0.7960
Test Epoch 17: 100%|██████████| 1768/1768 [00:33<00:00, 52.62it/s, loss=0.356]
Test Epoch 17 ==> 	accuracy: 0.9397, 	precision: 0.9866, 	recall: 0.7148, 	specificity: 0.9975, 	f1: 0.8290
Train Epoch 18: 100%|██████████| 6507/6507 [04:50<00:00, 22.40it/s, loss=0.0932]
Train Epoch 18 ==> 	accuracy: 0.8353, 	precision: 0.9991, 	recall: 0.6712, 	specificity: 0.9994, 	f1: 0.8030
Test Epoch 18: 100%|██████████| 1768/1768 [00:35<00:00, 50.42it/s, loss=0.156]
Test Epoch 18 ==> 	accuracy: 0.9390, 	precision: 0.9826, 	recall: 0.7142, 	specificity: 0.9967, 	f1: 0.8272
Train Epoch 19: 100%|██████████| 6507/6507 [04:59<00:00, 21.75it/s, loss=0.114]
Train Epoch 19 ==> 	accuracy: 0.8336, 	precision: 0.9992, 	recall: 0.6677, 	specificity: 0.9995, 	f1: 0.8005
Test Epoch 19: 100%|██████████| 1768/1768 [00:35<00:00, 50.24it/s, loss=0.146]
Test Epoch 19 ==> 	accuracy: 0.9380, 	precision: 0.9832, 	recall: 0.7088, 	specificity: 0.9969, 	f1: 0.8237
Train Epoch 20: 100%|██████████| 6507/6507 [04:51<00:00, 22.35it/s, loss=0.102]
Train Epoch 20 ==> 	accuracy: 0.8341, 	precision: 0.9991, 	recall: 0.6687, 	specificity: 0.9994, 	f1: 0.8012
Test Epoch 20: 100%|██████████| 1768/1768 [00:29<00:00, 60.02it/s, loss=0.451]
Test Epoch 20 ==> 	accuracy: 0.9388, 	precision: 0.9858, 	recall: 0.7111, 	specificity: 0.9974, 	f1: 0.8262
Train Epoch 21: 100%|██████████| 6507/6507 [04:48<00:00, 22.54it/s, loss=0.145]
Train Epoch 21 ==> 	accuracy: 0.8371, 	precision: 0.9992, 	recall: 0.6747, 	specificity: 0.9995, 	f1: 0.8055
Test Epoch 21: 100%|██████████| 1768/1768 [00:34<00:00, 50.94it/s, loss=0.281]
Test Epoch 21 ==> 	accuracy: 0.9387, 	precision: 0.9847, 	recall: 0.7113, 	specificity: 0.9972, 	f1: 0.8260
Train Epoch 22: 100%|██████████| 6507/6507 [04:49<00:00, 22.47it/s, loss=0.0858]
Train Epoch 22 ==> 	accuracy: 0.8389, 	precision: 0.9993, 	recall: 0.6783, 	specificity: 0.9995, 	f1: 0.8081
Test Epoch 22: 100%|██████████| 1768/1768 [00:31<00:00, 55.27it/s, loss=0.156]
Test Epoch 22 ==> 	accuracy: 0.9408, 	precision: 0.9860, 	recall: 0.7209, 	specificity: 0.9974, 	f1: 0.8329
Train Epoch 23: 100%|██████████| 6507/6507 [04:46<00:00, 22.73it/s, loss=0.0329]
Train Epoch 23 ==> 	accuracy: 0.8388, 	precision: 0.9992, 	recall: 0.6781, 	specificity: 0.9995, 	f1: 0.8079
Test Epoch 23: 100%|██████████| 1768/1768 [00:34<00:00, 51.41it/s, loss=0.204]
Test Epoch 23 ==> 	accuracy: 0.9377, 	precision: 0.9781, 	recall: 0.7112, 	specificity: 0.9959, 	f1: 0.8236
Train Epoch 24: 100%|██████████| 6507/6507 [04:52<00:00, 22.24it/s, loss=0.0737]
Train Epoch 24 ==> 	accuracy: 0.8414, 	precision: 0.9992, 	recall: 0.6833, 	specificity: 0.9994, 	f1: 0.8116
Test Epoch 24: 100%|██████████| 1768/1768 [00:33<00:00, 52.03it/s, loss=0.148]
Test Epoch 24 ==> 	accuracy: 0.9418, 	precision: 0.9817, 	recall: 0.7292, 	specificity: 0.9965, 	f1: 0.8368
Train Epoch 25: 100%|██████████| 6507/6507 [04:53<00:00, 22.15it/s, loss=0.077]
Train Epoch 25 ==> 	accuracy: 0.8426, 	precision: 0.9993, 	recall: 0.6858, 	specificity: 0.9995, 	f1: 0.8134
Test Epoch 25: 100%|██████████| 1768/1768 [00:30<00:00, 57.83it/s, loss=0.335]
Test Epoch 25 ==> 	accuracy: 0.9398, 	precision: 0.9864, 	recall: 0.7155, 	specificity: 0.9975, 	f1: 0.8294
Train Epoch 26: 100%|██████████| 6507/6507 [04:50<00:00, 22.41it/s, loss=0.123]
Train Epoch 26 ==> 	accuracy: 0.8418, 	precision: 0.9992, 	recall: 0.6841, 	specificity: 0.9995, 	f1: 0.8122
Test Epoch 26: 100%|██████████| 1768/1768 [00:32<00:00, 53.78it/s, loss=0.224]
Test Epoch 26 ==> 	accuracy: 0.9426, 	precision: 0.9875, 	recall: 0.7284, 	specificity: 0.9976, 	f1: 0.8384
Train Epoch 27: 100%|██████████| 6507/6507 [04:58<00:00, 21.77it/s, loss=0.25]
Train Epoch 27 ==> 	accuracy: 0.8435, 	precision: 0.9993, 	recall: 0.6876, 	specificity: 0.9995, 	f1: 0.8146
Test Epoch 27: 100%|██████████| 1768/1768 [00:32<00:00, 53.98it/s, loss=0.195]
Test Epoch 27 ==> 	accuracy: 0.9453, 	precision: 0.9804, 	recall: 0.7475, 	specificity: 0.9961, 	f1: 0.8483
Train Epoch 28: 100%|██████████| 6507/6507 [04:46<00:00, 22.72it/s, loss=0.0018]
Train Epoch 28 ==> 	accuracy: 0.8448, 	precision: 0.9992, 	recall: 0.6901, 	specificity: 0.9994, 	f1: 0.8164
Test Epoch 28: 100%|██████████| 1768/1768 [00:36<00:00, 48.36it/s, loss=0.169]
Test Epoch 28 ==> 	accuracy: 0.9442, 	precision: 0.9810, 	recall: 0.7416, 	specificity: 0.9963, 	f1: 0.8447
Train Epoch 29: 100%|██████████| 6507/6507 [04:55<00:00, 22.00it/s, loss=0.0128]
Train Epoch 29 ==> 	accuracy: 0.8443, 	precision: 0.9993, 	recall: 0.6892, 	specificity: 0.9995, 	f1: 0.8157
Test Epoch 29: 100%|██████████| 1768/1768 [00:28<00:00, 62.58it/s, loss=0.256]
Test Epoch 29 ==> 	accuracy: 0.9425, 	precision: 0.9893, 	recall: 0.7269, 	specificity: 0.9980, 	f1: 0.8380
Train Epoch 30: 100%|██████████| 6507/6507 [04:52<00:00, 22.24it/s, loss=0.0296]
Train Epoch 30 ==> 	accuracy: 0.8456, 	precision: 0.9993, 	recall: 0.6916, 	specificity: 0.9995, 	f1: 0.8174
Test Epoch 30: 100%|██████████| 1768/1768 [00:34<00:00, 50.87it/s, loss=0.184]
Test Epoch 30 ==> 	accuracy: 0.9398, 	precision: 0.9903, 	recall: 0.7127, 	specificity: 0.9982, 	f1: 0.8288
Train Epoch 31: 100%|██████████| 6507/6507 [04:52<00:00, 22.26it/s, loss=0.111]
Train Epoch 31 ==> 	accuracy: 0.8492, 	precision: 0.9993, 	recall: 0.6989, 	specificity: 0.9995, 	f1: 0.8225
Test Epoch 31: 100%|██████████| 1768/1768 [00:35<00:00, 50.23it/s, loss=0.479]
Test Epoch 31 ==> 	accuracy: 0.9464, 	precision: 0.9711, 	recall: 0.7605, 	specificity: 0.9942, 	f1: 0.8530
Train Epoch 32: 100%|██████████| 6507/6507 [04:53<00:00, 22.16it/s, loss=0.0657]
Train Epoch 32 ==> 	accuracy: 0.8469, 	precision: 0.9993, 	recall: 0.6942, 	specificity: 0.9995, 	f1: 0.8193
Test Epoch 32: 100%|██████████| 1768/1768 [00:36<00:00, 48.76it/s, loss=0.171]
Test Epoch 32 ==> 	accuracy: 0.9408, 	precision: 0.9834, 	recall: 0.7228, 	specificity: 0.9969, 	f1: 0.8332
Train Epoch 33: 100%|██████████| 6507/6507 [04:53<00:00, 22.15it/s, loss=0.141]
Train Epoch 33 ==> 	accuracy: 0.8472, 	precision: 0.9993, 	recall: 0.6948, 	specificity: 0.9995, 	f1: 0.8197
Test Epoch 33: 100%|██████████| 1768/1768 [00:32<00:00, 53.91it/s, loss=0.324]
Test Epoch 33 ==> 	accuracy: 0.9465, 	precision: 0.9842, 	recall: 0.7505, 	specificity: 0.9969, 	f1: 0.8516
Train Epoch 34: 100%|██████████| 6507/6507 [04:50<00:00, 22.37it/s, loss=0.144]
Train Epoch 34 ==> 	accuracy: 0.8505, 	precision: 0.9993, 	recall: 0.7015, 	specificity: 0.9995, 	f1: 0.8243
Test Epoch 34: 100%|██████████| 1768/1768 [00:34<00:00, 51.15it/s, loss=0.166]
Test Epoch 34 ==> 	accuracy: 0.9414, 	precision: 0.9901, 	recall: 0.7209, 	specificity: 0.9981, 	f1: 0.8343
Train Epoch 35: 100%|██████████| 6507/6507 [04:53<00:00, 22.19it/s, loss=0.103]
Train Epoch 35 ==> 	accuracy: 0.8511, 	precision: 0.9994, 	recall: 0.7027, 	specificity: 0.9996, 	f1: 0.8252
Test Epoch 35: 100%|██████████| 1768/1768 [00:33<00:00, 53.08it/s, loss=0.257]
Test Epoch 35 ==> 	accuracy: 0.9480, 	precision: 0.9837, 	recall: 0.7585, 	specificity: 0.9968, 	f1: 0.8565
Train Epoch 36: 100%|██████████| 6507/6507 [04:51<00:00, 22.34it/s, loss=0.427]
Train Epoch 36 ==> 	accuracy: 0.8507, 	precision: 0.9993, 	recall: 0.7018, 	specificity: 0.9995, 	f1: 0.8246
Test Epoch 36: 100%|██████████| 1768/1768 [00:33<00:00, 53.45it/s, loss=0.123]
Test Epoch 36 ==> 	accuracy: 0.9438, 	precision: 0.9884, 	recall: 0.7340, 	specificity: 0.9978, 	f1: 0.8424
Train Epoch 37: 100%|██████████| 6507/6507 [04:53<00:00, 22.18it/s, loss=0.174]
Train Epoch 37 ==> 	accuracy: 0.8517, 	precision: 0.9993, 	recall: 0.7039, 	specificity: 0.9995, 	f1: 0.8260
Test Epoch 37: 100%|██████████| 1768/1768 [00:32<00:00, 54.19it/s, loss=1.15]
Test Epoch 37 ==> 	accuracy: 0.9468, 	precision: 0.9859, 	recall: 0.7504, 	specificity: 0.9972, 	f1: 0.8522
Train Epoch 38: 100%|██████████| 6507/6507 [04:52<00:00, 22.25it/s, loss=0.0033]
Train Epoch 38 ==> 	accuracy: 0.8530, 	precision: 0.9993, 	recall: 0.7064, 	specificity: 0.9995, 	f1: 0.8277
Test Epoch 38: 100%|██████████| 1768/1768 [00:31<00:00, 55.28it/s, loss=0.346]
Test Epoch 38 ==> 	accuracy: 0.9459, 	precision: 0.9846, 	recall: 0.7470, 	specificity: 0.9970, 	f1: 0.8495
Train Epoch 39: 100%|██████████| 6507/6507 [04:46<00:00, 22.72it/s, loss=0.0236]
Train Epoch 39 ==> 	accuracy: 0.8551, 	precision: 0.9993, 	recall: 0.7106, 	specificity: 0.9995, 	f1: 0.8306
Test Epoch 39: 100%|██████████| 1768/1768 [00:32<00:00, 54.65it/s, loss=0.144]
Test Epoch 39 ==> 	accuracy: 0.9494, 	precision: 0.9818, 	recall: 0.7669, 	specificity: 0.9963, 	f1: 0.8611
Train Epoch 40: 100%|██████████| 6507/6507 [04:24<00:00, 24.58it/s, loss=0.0083]
Train Epoch 40 ==> 	accuracy: 0.8555, 	precision: 0.9993, 	recall: 0.7114, 	specificity: 0.9995, 	f1: 0.8311
Test Epoch 40: 100%|██████████| 1768/1768 [00:34<00:00, 50.58it/s, loss=0.292]
Test Epoch 40 ==> 	accuracy: 0.9481, 	precision: 0.9727, 	recall: 0.7680, 	specificity: 0.9945, 	f1: 0.8583
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 41: 100%|██████████| 6507/6507 [04:33<00:00, 23.76it/s, loss=0.0219]
Train Epoch 41 ==> 	accuracy: 0.8550, 	precision: 0.9993, 	recall: 0.7105, 	specificity: 0.9995, 	f1: 0.8305
Test Epoch 41: 100%|██████████| 1768/1768 [00:34<00:00, 51.41it/s, loss=0.185]
Test Epoch 41 ==> 	accuracy: 0.9465, 	precision: 0.9811, 	recall: 0.7530, 	specificity: 0.9963, 	f1: 0.8521
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 42: 100%|██████████| 6507/6507 [04:26<00:00, 24.45it/s, loss=0.0246]
Train Epoch 42 ==> 	accuracy: 0.8564, 	precision: 0.9994, 	recall: 0.7132, 	specificity: 0.9996, 	f1: 0.8324
Test Epoch 42: 100%|██████████| 1768/1768 [00:33<00:00, 52.76it/s, loss=0.65]
Test Epoch 42 ==> 	accuracy: 0.9463, 	precision: 0.9878, 	recall: 0.7467, 	specificity: 0.9976, 	f1: 0.8505
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 43: 100%|██████████| 6507/6507 [04:33<00:00, 23.83it/s, loss=0.0282]
Train Epoch 43 ==> 	accuracy: 0.8575, 	precision: 0.9994, 	recall: 0.7156, 	specificity: 0.9995, 	f1: 0.8340
Test Epoch 43: 100%|██████████| 1768/1768 [00:35<00:00, 50.47it/s, loss=0.186]
Test Epoch 43 ==> 	accuracy: 0.9490, 	precision: 0.9861, 	recall: 0.7614, 	specificity: 0.9972, 	f1: 0.8593
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 44: 100%|██████████| 6507/6507 [04:29<00:00, 24.15it/s, loss=0.0104]
Train Epoch 44 ==> 	accuracy: 0.8600, 	precision: 0.9994, 	recall: 0.7204, 	specificity: 0.9996, 	f1: 0.8373
Test Epoch 44: 100%|██████████| 1768/1768 [00:37<00:00, 47.66it/s, loss=0.127]
Test Epoch 44 ==> 	accuracy: 0.9521, 	precision: 0.9752, 	recall: 0.7855, 	specificity: 0.9949, 	f1: 0.8702
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 45: 100%|██████████| 6507/6507 [04:34<00:00, 23.68it/s, loss=0.0232]
Train Epoch 45 ==> 	accuracy: 0.8577, 	precision: 0.9994, 	recall: 0.7158, 	specificity: 0.9996, 	f1: 0.8342
Test Epoch 45: 100%|██████████| 1768/1768 [00:34<00:00, 50.78it/s, loss=0.172]
Test Epoch 45 ==> 	accuracy: 0.9489, 	precision: 0.9762, 	recall: 0.7687, 	specificity: 0.9952, 	f1: 0.8601
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 46: 100%|██████████| 6507/6507 [04:35<00:00, 23.66it/s, loss=0.0094]
Train Epoch 46 ==> 	accuracy: 0.8606, 	precision: 0.9994, 	recall: 0.7217, 	specificity: 0.9996, 	f1: 0.8381
Test Epoch 46: 100%|██████████| 1768/1768 [00:36<00:00, 48.93it/s, loss=0.187]
Test Epoch 46 ==> 	accuracy: 0.9467, 	precision: 0.9913, 	recall: 0.7459, 	specificity: 0.9983, 	f1: 0.8513
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 47: 100%|██████████| 6507/6507 [04:32<00:00, 23.85it/s, loss=0.019]
Train Epoch 47 ==> 	accuracy: 0.8599, 	precision: 0.9994, 	recall: 0.7202, 	specificity: 0.9996, 	f1: 0.8371
Test Epoch 47: 100%|██████████| 1768/1768 [00:36<00:00, 48.16it/s, loss=0.211]
Test Epoch 47 ==> 	accuracy: 0.9542, 	precision: 0.9742, 	recall: 0.7973, 	specificity: 0.9946, 	f1: 0.8769
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 48: 100%|██████████| 6507/6507 [04:33<00:00, 23.81it/s, loss=0.0091]
Train Epoch 48 ==> 	accuracy: 0.8624, 	precision: 0.9995, 	recall: 0.7252, 	specificity: 0.9996, 	f1: 0.8405
Test Epoch 48: 100%|██████████| 1768/1768 [00:35<00:00, 49.13it/s, loss=0.206]
Test Epoch 48 ==> 	accuracy: 0.9507, 	precision: 0.9829, 	recall: 0.7726, 	specificity: 0.9965, 	f1: 0.8651
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 49: 100%|██████████| 6507/6507 [04:36<00:00, 23.51it/s, loss=0.007]
Train Epoch 49 ==> 	accuracy: 0.8608, 	precision: 0.9995, 	recall: 0.7219, 	specificity: 0.9996, 	f1: 0.8383
Test Epoch 49: 100%|██████████| 1768/1768 [00:33<00:00, 53.00it/s, loss=1.38]
Test Epoch 49 ==> 	accuracy: 0.9511, 	precision: 0.9884, 	recall: 0.7702, 	specificity: 0.9977, 	f1: 0.8657
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 50: 100%|██████████| 6507/6507 [04:29<00:00, 24.12it/s, loss=0.0127]
Train Epoch 50 ==> 	accuracy: 0.8670, 	precision: 0.9994, 	recall: 0.7344, 	specificity: 0.9996, 	f1: 0.8467
Test Epoch 50: 100%|██████████| 1768/1768 [00:34<00:00, 50.75it/s, loss=0.198]
Test Epoch 50 ==> 	accuracy: 0.9503, 	precision: 0.9879, 	recall: 0.7664, 	specificity: 0.9976, 	f1: 0.8632
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 51: 100%|██████████| 6507/6507 [04:37<00:00, 23.45it/s, loss=0.0148]
Train Epoch 51 ==> 	accuracy: 0.8659, 	precision: 0.9995, 	recall: 0.7321, 	specificity: 0.9996, 	f1: 0.8452
Test Epoch 51: 100%|██████████| 1768/1768 [00:33<00:00, 52.02it/s, loss=0.563]
Test Epoch 51 ==> 	accuracy: 0.9545, 	precision: 0.9810, 	recall: 0.7931, 	specificity: 0.9960, 	f1: 0.8771
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 52: 100%|██████████| 6507/6507 [04:34<00:00, 23.75it/s, loss=0.016]
Train Epoch 52 ==> 	accuracy: 0.8684, 	precision: 0.9995, 	recall: 0.7371, 	specificity: 0.9996, 	f1: 0.8485
Test Epoch 52: 100%|██████████| 1768/1768 [00:33<00:00, 53.00it/s, loss=0.137]
Test Epoch 52 ==> 	accuracy: 0.9526, 	precision: 0.9830, 	recall: 0.7816, 	specificity: 0.9965, 	f1: 0.8708
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 53: 100%|██████████| 6507/6507 [04:30<00:00, 24.06it/s, loss=0.0366]
Train Epoch 53 ==> 	accuracy: 0.8682, 	precision: 0.9995, 	recall: 0.7368, 	specificity: 0.9996, 	f1: 0.8483
Test Epoch 53: 100%|██████████| 1768/1768 [00:34<00:00, 51.54it/s, loss=0.141]
Test Epoch 53 ==> 	accuracy: 0.9540, 	precision: 0.9811, 	recall: 0.7904, 	specificity: 0.9961, 	f1: 0.8755
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 54: 100%|██████████| 6507/6507 [04:31<00:00, 23.96it/s, loss=0.0228]
Train Epoch 54 ==> 	accuracy: 0.8662, 	precision: 0.9995, 	recall: 0.7328, 	specificity: 0.9996, 	f1: 0.8456
Test Epoch 54: 100%|██████████| 1768/1768 [00:34<00:00, 51.44it/s, loss=0.138]
Test Epoch 54 ==> 	accuracy: 0.9512, 	precision: 0.9878, 	recall: 0.7710, 	specificity: 0.9976, 	f1: 0.8661
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 55: 100%|██████████| 6507/6507 [04:37<00:00, 23.42it/s, loss=0.0169]
Train Epoch 55 ==> 	accuracy: 0.8701, 	precision: 0.9995, 	recall: 0.7405, 	specificity: 0.9997, 	f1: 0.8507
Test Epoch 55: 100%|██████████| 1768/1768 [00:32<00:00, 53.96it/s, loss=0.0852]
Test Epoch 55 ==> 	accuracy: 0.9550, 	precision: 0.9834, 	recall: 0.7932, 	specificity: 0.9966, 	f1: 0.8781
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 56: 100%|██████████| 6507/6507 [04:40<00:00, 23.21it/s, loss=0.812]
Train Epoch 56 ==> 	accuracy: 0.8689, 	precision: 0.9995, 	recall: 0.7381, 	specificity: 0.9996, 	f1: 0.8492
Test Epoch 56: 100%|██████████| 1768/1768 [00:33<00:00, 53.28it/s, loss=0.483]
Test Epoch 56 ==> 	accuracy: 0.9543, 	precision: 0.9814, 	recall: 0.7915, 	specificity: 0.9961, 	f1: 0.8762
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 57: 100%|██████████| 6507/6507 [04:28<00:00, 24.24it/s, loss=0.0202]
Train Epoch 57 ==> 	accuracy: 0.8726, 	precision: 0.9995, 	recall: 0.7456, 	specificity: 0.9997, 	f1: 0.8541
Test Epoch 57: 100%|██████████| 1768/1768 [00:35<00:00, 49.31it/s, loss=0.643]
Test Epoch 57 ==> 	accuracy: 0.9541, 	precision: 0.9838, 	recall: 0.7884, 	specificity: 0.9967, 	f1: 0.8753
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 58: 100%|██████████| 6507/6507 [04:33<00:00, 23.79it/s, loss=0.0275]
Train Epoch 58 ==> 	accuracy: 0.8707, 	precision: 0.9995, 	recall: 0.7418, 	specificity: 0.9997, 	f1: 0.8516
Test Epoch 58: 100%|██████████| 1768/1768 [00:33<00:00, 52.24it/s, loss=0.252]
Test Epoch 58 ==> 	accuracy: 0.9566, 	precision: 0.9782, 	recall: 0.8056, 	specificity: 0.9954, 	f1: 0.8836
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 59: 100%|██████████| 6507/6507 [04:33<00:00, 23.83it/s, loss=0.015]
Train Epoch 59 ==> 	accuracy: 0.8730, 	precision: 0.9995, 	recall: 0.7463, 	specificity: 0.9996, 	f1: 0.8546
Test Epoch 59: 100%|██████████| 1768/1768 [00:35<00:00, 49.37it/s, loss=1.24]
Test Epoch 59 ==> 	accuracy: 0.9538, 	precision: 0.9821, 	recall: 0.7886, 	specificity: 0.9963, 	f1: 0.8748
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 60: 100%|██████████| 6507/6507 [04:37<00:00, 23.41it/s, loss=0.0056]
Train Epoch 60 ==> 	accuracy: 0.8732, 	precision: 0.9996, 	recall: 0.7467, 	specificity: 0.9997, 	f1: 0.8548
Test Epoch 60: 100%|██████████| 1768/1768 [00:34<00:00, 51.41it/s, loss=0.171]
Test Epoch 60 ==> 	accuracy: 0.9572, 	precision: 0.9800, 	recall: 0.8070, 	specificity: 0.9958, 	f1: 0.8852
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 61: 100%|██████████| 6507/6507 [04:39<00:00, 23.30it/s, loss=0.0245]
Train Epoch 61 ==> 	accuracy: 0.8768, 	precision: 0.9995, 	recall: 0.7540, 	specificity: 0.9997, 	f1: 0.8596
Test Epoch 61: 100%|██████████| 1768/1768 [00:34<00:00, 51.67it/s, loss=0.223]
Test Epoch 61 ==> 	accuracy: 0.9543, 	precision: 0.9840, 	recall: 0.7891, 	specificity: 0.9967, 	f1: 0.8759
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 62: 100%|██████████| 6507/6507 [04:33<00:00, 23.83it/s, loss=0.0153]
Train Epoch 62 ==> 	accuracy: 0.8730, 	precision: 0.9996, 	recall: 0.7463, 	specificity: 0.9997, 	f1: 0.8545
Test Epoch 62: 100%|██████████| 1768/1768 [00:35<00:00, 50.22it/s, loss=0.175]
Test Epoch 62 ==> 	accuracy: 0.9545, 	precision: 0.9850, 	recall: 0.7897, 	specificity: 0.9969, 	f1: 0.8766
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 63: 100%|██████████| 6507/6507 [04:26<00:00, 24.40it/s, loss=0.0807]
Train Epoch 63 ==> 	accuracy: 0.8769, 	precision: 0.9995, 	recall: 0.7542, 	specificity: 0.9997, 	f1: 0.8597
Test Epoch 63: 100%|██████████| 1768/1768 [00:33<00:00, 52.36it/s, loss=0.117]
Test Epoch 63 ==> 	accuracy: 0.9546, 	precision: 0.9773, 	recall: 0.7966, 	specificity: 0.9952, 	f1: 0.8777
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 64: 100%|██████████| 6507/6507 [04:30<00:00, 24.09it/s, loss=0.0163]
Train Epoch 64 ==> 	accuracy: 0.8781, 	precision: 0.9996, 	recall: 0.7565, 	specificity: 0.9997, 	f1: 0.8612
Test Epoch 64: 100%|██████████| 1768/1768 [00:30<00:00, 57.32it/s, loss=2.14]
Test Epoch 64 ==> 	accuracy: 0.9548, 	precision: 0.9820, 	recall: 0.7935, 	specificity: 0.9963, 	f1: 0.8778
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 65: 100%|██████████| 6507/6507 [04:32<00:00, 23.90it/s, loss=0.0824]
Train Epoch 65 ==> 	accuracy: 0.8789, 	precision: 0.9996, 	recall: 0.7581, 	specificity: 0.9997, 	f1: 0.8622
Test Epoch 65: 100%|██████████| 1768/1768 [00:31<00:00, 55.72it/s, loss=0.194]
Test Epoch 65 ==> 	accuracy: 0.9576, 	precision: 0.9799, 	recall: 0.8091, 	specificity: 0.9957, 	f1: 0.8864
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 66: 100%|██████████| 6507/6507 [04:30<00:00, 24.06it/s, loss=0.0327]
Train Epoch 66 ==> 	accuracy: 0.8758, 	precision: 0.9996, 	recall: 0.7519, 	specificity: 0.9997, 	f1: 0.8582
Test Epoch 66: 100%|██████████| 1768/1768 [00:32<00:00, 54.01it/s, loss=0.234]
Test Epoch 66 ==> 	accuracy: 0.9549, 	precision: 0.9816, 	recall: 0.7943, 	specificity: 0.9962, 	f1: 0.8781
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 67: 100%|██████████| 6507/6507 [04:20<00:00, 25.02it/s, loss=0.0133]
Train Epoch 67 ==> 	accuracy: 0.8801, 	precision: 0.9996, 	recall: 0.7604, 	specificity: 0.9997, 	f1: 0.8638
Test Epoch 67: 100%|██████████| 1768/1768 [00:35<00:00, 50.17it/s, loss=0.27]
Test Epoch 67 ==> 	accuracy: 0.9568, 	precision: 0.9818, 	recall: 0.8035, 	specificity: 0.9962, 	f1: 0.8837
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 68: 100%|██████████| 6507/6507 [04:35<00:00, 23.61it/s, loss=0.0728]
Train Epoch 68 ==> 	accuracy: 0.8815, 	precision: 0.9996, 	recall: 0.7633, 	specificity: 0.9997, 	f1: 0.8656
Test Epoch 68: 100%|██████████| 1768/1768 [00:31<00:00, 55.74it/s, loss=0.135]
Test Epoch 68 ==> 	accuracy: 0.9574, 	precision: 0.9808, 	recall: 0.8073, 	specificity: 0.9959, 	f1: 0.8856
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 69: 100%|██████████| 6507/6507 [04:24<00:00, 24.56it/s, loss=0.0269]
Train Epoch 69 ==> 	accuracy: 0.8810, 	precision: 0.9996, 	recall: 0.7624, 	specificity: 0.9997, 	f1: 0.8650
Test Epoch 69: 100%|██████████| 1768/1768 [00:35<00:00, 50.36it/s, loss=0.109]
Test Epoch 69 ==> 	accuracy: 0.9567, 	precision: 0.9805, 	recall: 0.8044, 	specificity: 0.9959, 	f1: 0.8837
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 70: 100%|██████████| 6507/6507 [04:28<00:00, 24.20it/s, loss=0.0002]
Train Epoch 70 ==> 	accuracy: 0.8821, 	precision: 0.9996, 	recall: 0.7646, 	specificity: 0.9997, 	f1: 0.8664
Test Epoch 70: 100%|██████████| 1768/1768 [00:35<00:00, 50.21it/s, loss=0.0741]
Test Epoch 70 ==> 	accuracy: 0.9576, 	precision: 0.9768, 	recall: 0.8119, 	specificity: 0.9951, 	f1: 0.8868
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 71: 100%|██████████| 6507/6507 [04:29<00:00, 24.12it/s, loss=0.184]
Train Epoch 71 ==> 	accuracy: 0.8817, 	precision: 0.9996, 	recall: 0.7637, 	specificity: 0.9997, 	f1: 0.8658
Test Epoch 71: 100%|██████████| 1768/1768 [00:33<00:00, 53.38it/s, loss=0.0979]
Test Epoch 71 ==> 	accuracy: 0.9574, 	precision: 0.9845, 	recall: 0.8046, 	specificity: 0.9967, 	f1: 0.8855
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 72: 100%|██████████| 6507/6507 [04:33<00:00, 23.80it/s, loss=0.0347]
Train Epoch 72 ==> 	accuracy: 0.8833, 	precision: 0.9996, 	recall: 0.7668, 	specificity: 0.9997, 	f1: 0.8679
Test Epoch 72: 100%|██████████| 1768/1768 [00:36<00:00, 48.03it/s, loss=1.19]
Test Epoch 72 ==> 	accuracy: 0.9587, 	precision: 0.9829, 	recall: 0.8122, 	specificity: 0.9964, 	f1: 0.8895
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 73: 100%|██████████| 6507/6507 [04:24<00:00, 24.57it/s, loss=0.0373]
Train Epoch 73 ==> 	accuracy: 0.8824, 	precision: 0.9996, 	recall: 0.7652, 	specificity: 0.9997, 	f1: 0.8668
Test Epoch 73: 100%|██████████| 1768/1768 [00:35<00:00, 49.48it/s, loss=0.882]
Test Epoch 73 ==> 	accuracy: 0.9584, 	precision: 0.9834, 	recall: 0.8102, 	specificity: 0.9965, 	f1: 0.8884
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 74: 100%|██████████| 6507/6507 [04:25<00:00, 24.46it/s, loss=0.0114]
Train Epoch 74 ==> 	accuracy: 0.8846, 	precision: 0.9996, 	recall: 0.7695, 	specificity: 0.9997, 	f1: 0.8696
Test Epoch 74: 100%|██████████| 1768/1768 [00:36<00:00, 48.03it/s, loss=0.323]
Test Epoch 74 ==> 	accuracy: 0.9575, 	precision: 0.9817, 	recall: 0.8075, 	specificity: 0.9961, 	f1: 0.8861
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 75: 100%|██████████| 6507/6507 [04:29<00:00, 24.16it/s, loss=0.0106]
Train Epoch 75 ==> 	accuracy: 0.8834, 	precision: 0.9996, 	recall: 0.7672, 	specificity: 0.9997, 	f1: 0.8681
Test Epoch 75: 100%|██████████| 1768/1768 [00:36<00:00, 48.84it/s, loss=0.154]
Test Epoch 75 ==> 	accuracy: 0.9591, 	precision: 0.9768, 	recall: 0.8194, 	specificity: 0.9950, 	f1: 0.8912
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 76: 100%|██████████| 6507/6507 [04:29<00:00, 24.12it/s, loss=0.301]
Train Epoch 76 ==> 	accuracy: 0.8870, 	precision: 0.9996, 	recall: 0.7744, 	specificity: 0.9997, 	f1: 0.8727
Test Epoch 76: 100%|██████████| 1768/1768 [00:34<00:00, 50.58it/s, loss=0.277]
Test Epoch 76 ==> 	accuracy: 0.9596, 	precision: 0.9788, 	recall: 0.8203, 	specificity: 0.9954, 	f1: 0.8926
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 77: 100%|██████████| 6507/6507 [04:26<00:00, 24.37it/s, loss=1.87]
Train Epoch 77 ==> 	accuracy: 0.8860, 	precision: 0.9996, 	recall: 0.7722, 	specificity: 0.9997, 	f1: 0.8713
Test Epoch 77: 100%|██████████| 1768/1768 [00:35<00:00, 49.57it/s, loss=0.119]
Test Epoch 77 ==> 	accuracy: 0.9583, 	precision: 0.9794, 	recall: 0.8131, 	specificity: 0.9956, 	f1: 0.8885
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 78: 100%|██████████| 6507/6507 [04:32<00:00, 23.88it/s, loss=0.0024]
Train Epoch 78 ==> 	accuracy: 0.8874, 	precision: 0.9996, 	recall: 0.7750, 	specificity: 0.9997, 	f1: 0.8731
Test Epoch 78: 100%|██████████| 1768/1768 [00:38<00:00, 46.02it/s, loss=0.176]
Test Epoch 78 ==> 	accuracy: 0.9605, 	precision: 0.9794, 	recall: 0.8242, 	specificity: 0.9955, 	f1: 0.8951
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 79: 100%|██████████| 6507/6507 [04:32<00:00, 23.85it/s, loss=0.0593]
Train Epoch 79 ==> 	accuracy: 0.8850, 	precision: 0.9996, 	recall: 0.7704, 	specificity: 0.9997, 	f1: 0.8701
Test Epoch 79: 100%|██████████| 1768/1768 [00:34<00:00, 50.55it/s, loss=0.117]
Test Epoch 79 ==> 	accuracy: 0.9592, 	precision: 0.9817, 	recall: 0.8159, 	specificity: 0.9961, 	f1: 0.8912
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 80: 100%|██████████| 6507/6507 [04:37<00:00, 23.45it/s, loss=0.0087]
Train Epoch 80 ==> 	accuracy: 0.8858, 	precision: 0.9996, 	recall: 0.7719, 	specificity: 0.9997, 	f1: 0.8711
Test Epoch 80: 100%|██████████| 1768/1768 [00:31<00:00, 55.57it/s, loss=0.371]
Test Epoch 80 ==> 	accuracy: 0.9582, 	precision: 0.9831, 	recall: 0.8097, 	specificity: 0.9964, 	f1: 0.8880
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 81: 100%|██████████| 6507/6507 [04:31<00:00, 23.99it/s, loss=0.0258]
Train Epoch 81 ==> 	accuracy: 0.8887, 	precision: 0.9997, 	recall: 0.7776, 	specificity: 0.9997, 	f1: 0.8747
Test Epoch 81: 100%|██████████| 1768/1768 [00:34<00:00, 51.72it/s, loss=0.127]
Test Epoch 81 ==> 	accuracy: 0.9589, 	precision: 0.9819, 	recall: 0.8143, 	specificity: 0.9961, 	f1: 0.8903
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 82: 100%|██████████| 6507/6507 [04:45<00:00, 22.79it/s, loss=0.0018]
Train Epoch 82 ==> 	accuracy: 0.8873, 	precision: 0.9996, 	recall: 0.7749, 	specificity: 0.9997, 	f1: 0.8730
Test Epoch 82: 100%|██████████| 1768/1768 [00:34<00:00, 51.68it/s, loss=0.268]
Test Epoch 82 ==> 	accuracy: 0.9597, 	precision: 0.9804, 	recall: 0.8192, 	specificity: 0.9958, 	f1: 0.8926
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 83: 100%|██████████| 6507/6507 [04:30<00:00, 24.03it/s, loss=0.0323]
Train Epoch 83 ==> 	accuracy: 0.8877, 	precision: 0.9996, 	recall: 0.7756, 	specificity: 0.9997, 	f1: 0.8735
Test Epoch 83: 100%|██████████| 1768/1768 [00:35<00:00, 49.86it/s, loss=2.48]
Test Epoch 83 ==> 	accuracy: 0.9606, 	precision: 0.9760, 	recall: 0.8275, 	specificity: 0.9948, 	f1: 0.8956
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 84: 100%|██████████| 6507/6507 [04:31<00:00, 23.96it/s, loss=0.0104]
Train Epoch 84 ==> 	accuracy: 0.8891, 	precision: 0.9996, 	recall: 0.7784, 	specificity: 0.9997, 	f1: 0.8753
Test Epoch 84: 100%|██████████| 1768/1768 [00:33<00:00, 52.68it/s, loss=0.157]
Test Epoch 84 ==> 	accuracy: 0.9613, 	precision: 0.9806, 	recall: 0.8272, 	specificity: 0.9958, 	f1: 0.8974
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 85: 100%|██████████| 6507/6507 [04:30<00:00, 24.04it/s, loss=0.0372]
Train Epoch 85 ==> 	accuracy: 0.8914, 	precision: 0.9996, 	recall: 0.7831, 	specificity: 0.9997, 	f1: 0.8782
Test Epoch 85: 100%|██████████| 1768/1768 [00:34<00:00, 51.86it/s, loss=1.42]
Test Epoch 85 ==> 	accuracy: 0.9607, 	precision: 0.9788, 	recall: 0.8258, 	specificity: 0.9954, 	f1: 0.8958
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 86: 100%|██████████| 6507/6507 [04:30<00:00, 24.08it/s, loss=0.561]
Train Epoch 86 ==> 	accuracy: 0.8906, 	precision: 0.9996, 	recall: 0.7815, 	specificity: 0.9997, 	f1: 0.8772
Test Epoch 86: 100%|██████████| 1768/1768 [00:35<00:00, 49.80it/s, loss=0.112]
Test Epoch 86 ==> 	accuracy: 0.9595, 	precision: 0.9818, 	recall: 0.8173, 	specificity: 0.9961, 	f1: 0.8920
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 87: 100%|██████████| 6507/6507 [04:32<00:00, 23.87it/s, loss=0.0208]
Train Epoch 87 ==> 	accuracy: 0.8905, 	precision: 0.9996, 	recall: 0.7813, 	specificity: 0.9997, 	f1: 0.8771
Test Epoch 87: 100%|██████████| 1768/1768 [00:32<00:00, 54.47it/s, loss=0.535]
Test Epoch 87 ==> 	accuracy: 0.9607, 	precision: 0.9758, 	recall: 0.8286, 	specificity: 0.9947, 	f1: 0.8962
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 88: 100%|██████████| 6507/6507 [04:34<00:00, 23.74it/s, loss=1.1]
Train Epoch 88 ==> 	accuracy: 0.8922, 	precision: 0.9997, 	recall: 0.7847, 	specificity: 0.9997, 	f1: 0.8792
Test Epoch 88: 100%|██████████| 1768/1768 [00:34<00:00, 51.22it/s, loss=1.62]
Test Epoch 88 ==> 	accuracy: 0.9600, 	precision: 0.9746, 	recall: 0.8259, 	specificity: 0.9945, 	f1: 0.8941
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 89: 100%|██████████| 6507/6507 [04:33<00:00, 23.80it/s, loss=0.655]
Train Epoch 89 ==> 	accuracy: 0.8939, 	precision: 0.9996, 	recall: 0.7881, 	specificity: 0.9997, 	f1: 0.8814
Test Epoch 89: 100%|██████████| 1768/1768 [00:34<00:00, 50.56it/s, loss=0.772]
Test Epoch 89 ==> 	accuracy: 0.9611, 	precision: 0.9731, 	recall: 0.8328, 	specificity: 0.9941, 	f1: 0.8975
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 90: 100%|██████████| 6507/6507 [04:26<00:00, 24.43it/s, loss=2.01]
Train Epoch 90 ==> 	accuracy: 0.8912, 	precision: 0.9997, 	recall: 0.7826, 	specificity: 0.9997, 	f1: 0.8779
Test Epoch 90: 100%|██████████| 1768/1768 [00:33<00:00, 52.58it/s, loss=0.464]
Test Epoch 90 ==> 	accuracy: 0.9604, 	precision: 0.9737, 	recall: 0.8286, 	specificity: 0.9943, 	f1: 0.8953
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 91: 100%|██████████| 6507/6507 [04:28<00:00, 24.26it/s, loss=0.01]
Train Epoch 91 ==> 	accuracy: 0.8945, 	precision: 0.9996, 	recall: 0.7893, 	specificity: 0.9997, 	f1: 0.8821
Test Epoch 91: 100%|██████████| 1768/1768 [00:32<00:00, 55.18it/s, loss=1.21]
Test Epoch 91 ==> 	accuracy: 0.9612, 	precision: 0.9718, 	recall: 0.8344, 	specificity: 0.9938, 	f1: 0.8978
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 92: 100%|██████████| 6507/6507 [04:26<00:00, 24.42it/s, loss=0.0086]
Train Epoch 92 ==> 	accuracy: 0.8915, 	precision: 0.9996, 	recall: 0.7833, 	specificity: 0.9997, 	f1: 0.8783
Test Epoch 92: 100%|██████████| 1768/1768 [00:34<00:00, 50.78it/s, loss=0.247]
Test Epoch 92 ==> 	accuracy: 0.9613, 	precision: 0.9796, 	recall: 0.8281, 	specificity: 0.9956, 	f1: 0.8975
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 93: 100%|██████████| 6507/6507 [04:36<00:00, 23.57it/s, loss=0.0014]
Train Epoch 93 ==> 	accuracy: 0.8942, 	precision: 0.9996, 	recall: 0.7886, 	specificity: 0.9997, 	f1: 0.8817
Test Epoch 93: 100%|██████████| 1768/1768 [00:32<00:00, 54.70it/s, loss=0.0881]
Test Epoch 93 ==> 	accuracy: 0.9619, 	precision: 0.9760, 	recall: 0.8342, 	specificity: 0.9947, 	f1: 0.8996
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 94: 100%|██████████| 6507/6507 [04:33<00:00, 23.78it/s, loss=0.0042]
Train Epoch 94 ==> 	accuracy: 0.8940, 	precision: 0.9997, 	recall: 0.7882, 	specificity: 0.9997, 	f1: 0.8814
Test Epoch 94: 100%|██████████| 1768/1768 [00:33<00:00, 52.52it/s, loss=0.298]
Test Epoch 94 ==> 	accuracy: 0.9620, 	precision: 0.9704, 	recall: 0.8400, 	specificity: 0.9934, 	f1: 0.9005
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 95: 100%|██████████| 6507/6507 [04:33<00:00, 23.75it/s, loss=0.0237]
Train Epoch 95 ==> 	accuracy: 0.8922, 	precision: 0.9997, 	recall: 0.7846, 	specificity: 0.9997, 	f1: 0.8792
Test Epoch 95: 100%|██████████| 1768/1768 [00:34<00:00, 50.70it/s, loss=0.455]
Test Epoch 95 ==> 	accuracy: 0.9615, 	precision: 0.9759, 	recall: 0.8326, 	specificity: 0.9947, 	f1: 0.8986
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 96: 100%|██████████| 6507/6507 [04:32<00:00, 23.91it/s, loss=0.0866]
Train Epoch 96 ==> 	accuracy: 0.8944, 	precision: 0.9997, 	recall: 0.7891, 	specificity: 0.9997, 	f1: 0.8820
Test Epoch 96: 100%|██████████| 1768/1768 [00:36<00:00, 49.11it/s, loss=0.0982]
Test Epoch 96 ==> 	accuracy: 0.9619, 	precision: 0.9727, 	recall: 0.8374, 	specificity: 0.9940, 	f1: 0.9000
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 97: 100%|██████████| 6507/6507 [04:29<00:00, 24.14it/s, loss=0.0356]
Train Epoch 97 ==> 	accuracy: 0.8938, 	precision: 0.9996, 	recall: 0.7879, 	specificity: 0.9997, 	f1: 0.8812
Test Epoch 97: 100%|██████████| 1768/1768 [00:33<00:00, 52.61it/s, loss=0.0791]
Test Epoch 97 ==> 	accuracy: 0.9616, 	precision: 0.9761, 	recall: 0.8326, 	specificity: 0.9948, 	f1: 0.8986
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 98: 100%|██████████| 6507/6507 [04:25<00:00, 24.54it/s, loss=0.24]
Train Epoch 98 ==> 	accuracy: 0.8957, 	precision: 0.9997, 	recall: 0.7917, 	specificity: 0.9997, 	f1: 0.8836
Test Epoch 98: 100%|██████████| 1768/1768 [00:33<00:00, 52.33it/s, loss=0.624]
Test Epoch 98 ==> 	accuracy: 0.9624, 	precision: 0.9729, 	recall: 0.8398, 	specificity: 0.9940, 	f1: 0.9014
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 99: 100%|██████████| 6507/6507 [04:32<00:00, 23.88it/s, loss=0.0231]
Train Epoch 99 ==> 	accuracy: 0.8939, 	precision: 0.9997, 	recall: 0.7882, 	specificity: 0.9997, 	f1: 0.8814
Test Epoch 99: 100%|██████████| 1768/1768 [00:36<00:00, 49.03it/s, loss=0.142]
Test Epoch 99 ==> 	accuracy: 0.9625, 	precision: 0.9725, 	recall: 0.8404, 	specificity: 0.9939, 	f1: 0.9016
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 100: 100%|██████████| 6507/6507 [04:35<00:00, 23.60it/s, loss=0.0067]
Train Epoch 100 ==> 	accuracy: 0.8970, 	precision: 0.9997, 	recall: 0.7943, 	specificity: 0.9997, 	f1: 0.8852
Test Epoch 100: 100%|██████████| 1768/1768 [00:34<00:00, 51.31it/s, loss=0.138]
Test Epoch 100 ==> 	accuracy: 0.9630, 	precision: 0.9724, 	recall: 0.8429, 	specificity: 0.9938, 	f1: 0.9030
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 101: 100%|██████████| 6507/6507 [04:31<00:00, 23.98it/s, loss=0.0186]
Train Epoch 101 ==> 	accuracy: 0.8984, 	precision: 0.9997, 	recall: 0.7970, 	specificity: 0.9997, 	f1: 0.8869
Test Epoch 101: 100%|██████████| 1768/1768 [00:35<00:00, 49.89it/s, loss=0.0748]
Test Epoch 101 ==> 	accuracy: 0.9623, 	precision: 0.9714, 	recall: 0.8403, 	specificity: 0.9936, 	f1: 0.9011
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 102: 100%|██████████| 6507/6507 [04:33<00:00, 23.81it/s, loss=0.502]
Train Epoch 102 ==> 	accuracy: 0.8965, 	precision: 0.9997, 	recall: 0.7933, 	specificity: 0.9998, 	f1: 0.8846
Test Epoch 102: 100%|██████████| 1768/1768 [00:36<00:00, 48.54it/s, loss=0.171]
Test Epoch 102 ==> 	accuracy: 0.9621, 	precision: 0.9717, 	recall: 0.8391, 	specificity: 0.9937, 	f1: 0.9006
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 103: 100%|██████████| 6507/6507 [04:31<00:00, 23.93it/s, loss=1.31]
Train Epoch 103 ==> 	accuracy: 0.8957, 	precision: 0.9997, 	recall: 0.7916, 	specificity: 0.9997, 	f1: 0.8835
Test Epoch 103: 100%|██████████| 1768/1768 [00:35<00:00, 49.25it/s, loss=0.121]
Test Epoch 103 ==> 	accuracy: 0.9618, 	precision: 0.9752, 	recall: 0.8347, 	specificity: 0.9945, 	f1: 0.8995
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 104: 100%|██████████| 6507/6507 [04:33<00:00, 23.81it/s, loss=0.0584]
Train Epoch 104 ==> 	accuracy: 0.8977, 	precision: 0.9997, 	recall: 0.7957, 	specificity: 0.9997, 	f1: 0.8861
Test Epoch 104: 100%|██████████| 1768/1768 [00:33<00:00, 53.43it/s, loss=0.16]
Test Epoch 104 ==> 	accuracy: 0.9623, 	precision: 0.9709, 	recall: 0.8410, 	specificity: 0.9935, 	f1: 0.9013
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 105: 100%|██████████| 6507/6507 [04:23<00:00, 24.65it/s, loss=0.0193]
Train Epoch 105 ==> 	accuracy: 0.8977, 	precision: 0.9997, 	recall: 0.7957, 	specificity: 0.9997, 	f1: 0.8861
Test Epoch 105: 100%|██████████| 1768/1768 [00:35<00:00, 50.19it/s, loss=2.25]
Test Epoch 105 ==> 	accuracy: 0.9629, 	precision: 0.9755, 	recall: 0.8397, 	specificity: 0.9946, 	f1: 0.9025
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 106: 100%|██████████| 6507/6507 [04:31<00:00, 23.94it/s, loss=0.0096]
Train Epoch 106 ==> 	accuracy: 0.8986, 	precision: 0.9997, 	recall: 0.7975, 	specificity: 0.9997, 	f1: 0.8872
Test Epoch 106: 100%|██████████| 1768/1768 [00:33<00:00, 52.80it/s, loss=0.215]
Test Epoch 106 ==> 	accuracy: 0.9632, 	precision: 0.9681, 	recall: 0.8481, 	specificity: 0.9928, 	f1: 0.9041
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 107: 100%|██████████| 6507/6507 [04:32<00:00, 23.88it/s, loss=0.0147]
Train Epoch 107 ==> 	accuracy: 0.8990, 	precision: 0.9997, 	recall: 0.7983, 	specificity: 0.9998, 	f1: 0.8877
Test Epoch 107: 100%|██████████| 1768/1768 [00:35<00:00, 49.50it/s, loss=2.61]
Test Epoch 107 ==> 	accuracy: 0.9626, 	precision: 0.9738, 	recall: 0.8399, 	specificity: 0.9942, 	f1: 0.9019
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 108: 100%|██████████| 6507/6507 [04:36<00:00, 23.53it/s, loss=0.022]
Train Epoch 108 ==> 	accuracy: 0.8974, 	precision: 0.9997, 	recall: 0.7951, 	specificity: 0.9997, 	f1: 0.8857
Test Epoch 108: 100%|██████████| 1768/1768 [00:33<00:00, 53.27it/s, loss=1.78]
Test Epoch 108 ==> 	accuracy: 0.9621, 	precision: 0.9696, 	recall: 0.8412, 	specificity: 0.9932, 	f1: 0.9009
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 109: 100%|██████████| 6507/6507 [04:26<00:00, 24.44it/s, loss=0.0439]
Train Epoch 109 ==> 	accuracy: 0.8997, 	precision: 0.9997, 	recall: 0.7996, 	specificity: 0.9998, 	f1: 0.8885
Test Epoch 109: 100%|██████████| 1768/1768 [00:35<00:00, 50.36it/s, loss=0.0967]
Test Epoch 109 ==> 	accuracy: 0.9626, 	precision: 0.9727, 	recall: 0.8408, 	specificity: 0.9939, 	f1: 0.9020
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 110: 100%|██████████| 6507/6507 [04:29<00:00, 24.18it/s, loss=0.0035]
Train Epoch 110 ==> 	accuracy: 0.8984, 	precision: 0.9997, 	recall: 0.7971, 	specificity: 0.9997, 	f1: 0.8870
Test Epoch 110: 100%|██████████| 1768/1768 [00:35<00:00, 50.34it/s, loss=5.11]
Test Epoch 110 ==> 	accuracy: 0.9628, 	precision: 0.9727, 	recall: 0.8416, 	specificity: 0.9939, 	f1: 0.9024
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 111: 100%|██████████| 6507/6507 [04:34<00:00, 23.72it/s, loss=0.0187]
Train Epoch 111 ==> 	accuracy: 0.9003, 	precision: 0.9996, 	recall: 0.8009, 	specificity: 0.9997, 	f1: 0.8893
Test Epoch 111: 100%|██████████| 1768/1768 [00:34<00:00, 52.00it/s, loss=2.68]
Test Epoch 111 ==> 	accuracy: 0.9645, 	precision: 0.9666, 	recall: 0.8559, 	specificity: 0.9924, 	f1: 0.9079
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 112: 100%|██████████| 6507/6507 [04:32<00:00, 23.86it/s, loss=0.0217]
Train Epoch 112 ==> 	accuracy: 0.8981, 	precision: 0.9997, 	recall: 0.7965, 	specificity: 0.9997, 	f1: 0.8866
Test Epoch 112: 100%|██████████| 1768/1768 [00:33<00:00, 52.14it/s, loss=0.149]
Test Epoch 112 ==> 	accuracy: 0.9636, 	precision: 0.9700, 	recall: 0.8481, 	specificity: 0.9932, 	f1: 0.9050
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 113: 100%|██████████| 6507/6507 [04:29<00:00, 24.12it/s, loss=0.014]
Train Epoch 113 ==> 	accuracy: 0.8995, 	precision: 0.9997, 	recall: 0.7992, 	specificity: 0.9998, 	f1: 0.8883
Test Epoch 113: 100%|██████████| 1768/1768 [00:35<00:00, 50.39it/s, loss=0.398]
Test Epoch 113 ==> 	accuracy: 0.9631, 	precision: 0.9695, 	recall: 0.8463, 	specificity: 0.9932, 	f1: 0.9037
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 114: 100%|██████████| 6507/6507 [04:28<00:00, 24.25it/s, loss=0.0244]
Train Epoch 114 ==> 	accuracy: 0.8997, 	precision: 0.9997, 	recall: 0.7996, 	specificity: 0.9997, 	f1: 0.8885
Test Epoch 114: 100%|██████████| 1768/1768 [00:37<00:00, 47.72it/s, loss=1.05]
Test Epoch 114 ==> 	accuracy: 0.9629, 	precision: 0.9714, 	recall: 0.8433, 	specificity: 0.9936, 	f1: 0.9028
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 115: 100%|██████████| 6507/6507 [04:31<00:00, 24.00it/s, loss=0.0075]
Train Epoch 115 ==> 	accuracy: 0.9009, 	precision: 0.9997, 	recall: 0.8020, 	specificity: 0.9997, 	f1: 0.8900
Test Epoch 115: 100%|██████████| 1768/1768 [00:33<00:00, 52.18it/s, loss=0.37]
Test Epoch 115 ==> 	accuracy: 0.9629, 	precision: 0.9725, 	recall: 0.8425, 	specificity: 0.9939, 	f1: 0.9028
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 116: 100%|██████████| 6507/6507 [04:30<00:00, 24.02it/s, loss=0.0222]
Train Epoch 116 ==> 	accuracy: 0.8977, 	precision: 0.9997, 	recall: 0.7957, 	specificity: 0.9998, 	f1: 0.8861
Test Epoch 116: 100%|██████████| 1768/1768 [00:36<00:00, 48.50it/s, loss=0.179]
Test Epoch 116 ==> 	accuracy: 0.9629, 	precision: 0.9750, 	recall: 0.8402, 	specificity: 0.9944, 	f1: 0.9026
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 117: 100%|██████████| 6507/6507 [04:26<00:00, 24.45it/s, loss=0.0314]
Train Epoch 117 ==> 	accuracy: 0.9010, 	precision: 0.9997, 	recall: 0.8023, 	specificity: 0.9998, 	f1: 0.8902
Test Epoch 117: 100%|██████████| 1768/1768 [00:34<00:00, 51.65it/s, loss=0.404]
Test Epoch 117 ==> 	accuracy: 0.9635, 	precision: 0.9677, 	recall: 0.8498, 	specificity: 0.9927, 	f1: 0.9049
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 118: 100%|██████████| 6507/6507 [04:23<00:00, 24.69it/s, loss=0.0354]
Train Epoch 118 ==> 	accuracy: 0.9001, 	precision: 0.9997, 	recall: 0.8005, 	specificity: 0.9998, 	f1: 0.8891
Test Epoch 118: 100%|██████████| 1768/1768 [00:34<00:00, 51.80it/s, loss=0.153]
Test Epoch 118 ==> 	accuracy: 0.9647, 	precision: 0.9765, 	recall: 0.8479, 	specificity: 0.9948, 	f1: 0.9077
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 119: 100%|██████████| 6507/6507 [04:29<00:00, 24.14it/s, loss=0.0196]
Train Epoch 119 ==> 	accuracy: 0.8995, 	precision: 0.9997, 	recall: 0.7993, 	specificity: 0.9998, 	f1: 0.8883
Test Epoch 119: 100%|██████████| 1768/1768 [00:35<00:00, 50.02it/s, loss=0.792]
Test Epoch 119 ==> 	accuracy: 0.9645, 	precision: 0.9763, 	recall: 0.8472, 	specificity: 0.9947, 	f1: 0.9072
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 120: 100%|██████████| 6507/6507 [04:26<00:00, 24.40it/s, loss=0.0085]
Train Epoch 120 ==> 	accuracy: 0.8996, 	precision: 0.9997, 	recall: 0.7995, 	specificity: 0.9997, 	f1: 0.8884
Test Epoch 120: 100%|██████████| 1768/1768 [00:32<00:00, 54.09it/s, loss=1.18]
Test Epoch 120 ==> 	accuracy: 0.9649, 	precision: 0.9745, 	recall: 0.8504, 	specificity: 0.9943, 	f1: 0.9082
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 121: 100%|██████████| 6507/6507 [04:31<00:00, 23.95it/s, loss=0.344]
Train Epoch 121 ==> 	accuracy: 0.9019, 	precision: 0.9997, 	recall: 0.8040, 	specificity: 0.9998, 	f1: 0.8912
Test Epoch 121: 100%|██████████| 1768/1768 [00:35<00:00, 50.02it/s, loss=1.15]
Test Epoch 121 ==> 	accuracy: 0.9646, 	precision: 0.9770, 	recall: 0.8470, 	specificity: 0.9949, 	f1: 0.9074
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 122: 100%|██████████| 6507/6507 [04:31<00:00, 23.93it/s, loss=0.0093]
Train Epoch 122 ==> 	accuracy: 0.9017, 	precision: 0.9997, 	recall: 0.8036, 	specificity: 0.9997, 	f1: 0.8910
Test Epoch 122: 100%|██████████| 1768/1768 [00:34<00:00, 50.83it/s, loss=1.96]
Test Epoch 122 ==> 	accuracy: 0.9646, 	precision: 0.9755, 	recall: 0.8481, 	specificity: 0.9945, 	f1: 0.9074
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 123: 100%|██████████| 6507/6507 [04:29<00:00, 24.19it/s, loss=0.0104]
Train Epoch 123 ==> 	accuracy: 0.9004, 	precision: 0.9997, 	recall: 0.8010, 	specificity: 0.9998, 	f1: 0.8894
Test Epoch 123: 100%|██████████| 1768/1768 [00:34<00:00, 51.94it/s, loss=0.184]
Test Epoch 123 ==> 	accuracy: 0.9643, 	precision: 0.9753, 	recall: 0.8469, 	specificity: 0.9945, 	f1: 0.9066
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 124: 100%|██████████| 6507/6507 [04:27<00:00, 24.37it/s, loss=0.0161]
Train Epoch 124 ==> 	accuracy: 0.9030, 	precision: 0.9997, 	recall: 0.8063, 	specificity: 0.9998, 	f1: 0.8926
Test Epoch 124: 100%|██████████| 1768/1768 [00:33<00:00, 52.02it/s, loss=0.133]
Test Epoch 124 ==> 	accuracy: 0.9646, 	precision: 0.9744, 	recall: 0.8493, 	specificity: 0.9943, 	f1: 0.9075
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 125: 100%|██████████| 6507/6507 [04:28<00:00, 24.24it/s, loss=0.135]
Train Epoch 125 ==> 	accuracy: 0.9021, 	precision: 0.9997, 	recall: 0.8044, 	specificity: 0.9997, 	f1: 0.8914
Test Epoch 125: 100%|██████████| 1768/1768 [00:34<00:00, 50.70it/s, loss=0.38]
Test Epoch 125 ==> 	accuracy: 0.9652, 	precision: 0.9722, 	recall: 0.8541, 	specificity: 0.9937, 	f1: 0.9094
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 126: 100%|██████████| 6507/6507 [04:25<00:00, 24.51it/s, loss=0.0125]
Train Epoch 126 ==> 	accuracy: 0.9022, 	precision: 0.9997, 	recall: 0.8046, 	specificity: 0.9998, 	f1: 0.8916
Test Epoch 126: 100%|██████████| 1768/1768 [00:34<00:00, 51.60it/s, loss=0.207]
Test Epoch 126 ==> 	accuracy: 0.9642, 	precision: 0.9739, 	recall: 0.8476, 	specificity: 0.9942, 	f1: 0.9064
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 127: 100%|██████████| 6507/6507 [04:30<00:00, 24.01it/s, loss=0.953]
Train Epoch 127 ==> 	accuracy: 0.9023, 	precision: 0.9997, 	recall: 0.8048, 	specificity: 0.9997, 	f1: 0.8917
Test Epoch 127: 100%|██████████| 1768/1768 [00:33<00:00, 52.13it/s, loss=0.939]
Test Epoch 127 ==> 	accuracy: 0.9638, 	precision: 0.9751, 	recall: 0.8448, 	specificity: 0.9945, 	f1: 0.9053
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 128: 100%|██████████| 6507/6507 [04:31<00:00, 24.00it/s, loss=0.0178]
Train Epoch 128 ==> 	accuracy: 0.9019, 	precision: 0.9997, 	recall: 0.8040, 	specificity: 0.9998, 	f1: 0.8912
Test Epoch 128: 100%|██████████| 1768/1768 [00:34<00:00, 51.52it/s, loss=0.715]
Test Epoch 128 ==> 	accuracy: 0.9651, 	precision: 0.9755, 	recall: 0.8505, 	specificity: 0.9945, 	f1: 0.9087
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 129: 100%|██████████| 6507/6507 [04:36<00:00, 23.52it/s, loss=0.0567]
Train Epoch 129 ==> 	accuracy: 0.9003, 	precision: 0.9997, 	recall: 0.8009, 	specificity: 0.9997, 	f1: 0.8893
Test Epoch 129: 100%|██████████| 1768/1768 [00:33<00:00, 52.28it/s, loss=0.151]
Test Epoch 129 ==> 	accuracy: 0.9642, 	precision: 0.9763, 	recall: 0.8455, 	specificity: 0.9947, 	f1: 0.9062
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 130: 100%|██████████| 6507/6507 [04:32<00:00, 23.88it/s, loss=0.0168]
Train Epoch 130 ==> 	accuracy: 0.9025, 	precision: 0.9997, 	recall: 0.8052, 	specificity: 0.9998, 	f1: 0.8919
Test Epoch 130: 100%|██████████| 1768/1768 [00:36<00:00, 49.07it/s, loss=0.158]
Test Epoch 130 ==> 	accuracy: 0.9646, 	precision: 0.9770, 	recall: 0.8471, 	specificity: 0.9949, 	f1: 0.9074
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 131: 100%|██████████| 6507/6507 [04:34<00:00, 23.72it/s, loss=0.0061]
Train Epoch 131 ==> 	accuracy: 0.9020, 	precision: 0.9997, 	recall: 0.8042, 	specificity: 0.9998, 	f1: 0.8914
Test Epoch 131: 100%|██████████| 1768/1768 [00:33<00:00, 53.24it/s, loss=7.84]
Test Epoch 131 ==> 	accuracy: 0.9654, 	precision: 0.9730, 	recall: 0.8543, 	specificity: 0.9939, 	f1: 0.9098
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 132: 100%|██████████| 6507/6507 [04:32<00:00, 23.90it/s, loss=0.0098]
Train Epoch 132 ==> 	accuracy: 0.9033, 	precision: 0.9997, 	recall: 0.8068, 	specificity: 0.9998, 	f1: 0.8930
Test Epoch 132: 100%|██████████| 1768/1768 [00:35<00:00, 50.16it/s, loss=0.0958]
Test Epoch 132 ==> 	accuracy: 0.9654, 	precision: 0.9731, 	recall: 0.8547, 	specificity: 0.9939, 	f1: 0.9100
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 133: 100%|██████████| 6507/6507 [04:30<00:00, 24.02it/s, loss=0.0129]
Train Epoch 133 ==> 	accuracy: 0.9020, 	precision: 0.9997, 	recall: 0.8042, 	specificity: 0.9998, 	f1: 0.8914
Test Epoch 133: 100%|██████████| 1768/1768 [00:36<00:00, 48.20it/s, loss=0.101]
Test Epoch 133 ==> 	accuracy: 0.9647, 	precision: 0.9726, 	recall: 0.8516, 	specificity: 0.9938, 	f1: 0.9081
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 134: 100%|██████████| 6507/6507 [04:41<00:00, 23.10it/s, loss=0.0177]
Train Epoch 134 ==> 	accuracy: 0.9040, 	precision: 0.9997, 	recall: 0.8083, 	specificity: 0.9998, 	f1: 0.8938
Test Epoch 134: 100%|██████████| 1768/1768 [00:34<00:00, 50.59it/s, loss=0.542]
Test Epoch 134 ==> 	accuracy: 0.9651, 	precision: 0.9727, 	recall: 0.8531, 	specificity: 0.9938, 	f1: 0.9090
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 135: 100%|██████████| 6507/6507 [04:27<00:00, 24.36it/s, loss=0.0113]
Train Epoch 135 ==> 	accuracy: 0.9042, 	precision: 0.9997, 	recall: 0.8087, 	specificity: 0.9998, 	f1: 0.8941
Test Epoch 135: 100%|██████████| 1768/1768 [00:35<00:00, 49.33it/s, loss=0.658]
Test Epoch 135 ==> 	accuracy: 0.9650, 	precision: 0.9740, 	recall: 0.8514, 	specificity: 0.9942, 	f1: 0.9086
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 136: 100%|██████████| 6507/6507 [04:36<00:00, 23.56it/s, loss=0.0528]
Train Epoch 136 ==> 	accuracy: 0.9036, 	precision: 0.9997, 	recall: 0.8074, 	specificity: 0.9998, 	f1: 0.8933
Test Epoch 136: 100%|██████████| 1768/1768 [00:35<00:00, 49.73it/s, loss=0.0455]
Test Epoch 136 ==> 	accuracy: 0.9649, 	precision: 0.9734, 	recall: 0.8518, 	specificity: 0.9940, 	f1: 0.9085
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 137: 100%|██████████| 6507/6507 [04:30<00:00, 24.03it/s, loss=0.348]
Train Epoch 137 ==> 	accuracy: 0.9046, 	precision: 0.9997, 	recall: 0.8094, 	specificity: 0.9998, 	f1: 0.8945
Test Epoch 137: 100%|██████████| 1768/1768 [00:35<00:00, 49.12it/s, loss=1]
Test Epoch 137 ==> 	accuracy: 0.9646, 	precision: 0.9707, 	recall: 0.8528, 	specificity: 0.9934, 	f1: 0.9079
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 138: 100%|██████████| 6507/6507 [04:36<00:00, 23.57it/s, loss=0.0188]
Train Epoch 138 ==> 	accuracy: 0.9043, 	precision: 0.9997, 	recall: 0.8089, 	specificity: 0.9997, 	f1: 0.8942
Test Epoch 138: 100%|██████████| 1768/1768 [00:36<00:00, 48.94it/s, loss=1.5]
Test Epoch 138 ==> 	accuracy: 0.9646, 	precision: 0.9689, 	recall: 0.8546, 	specificity: 0.9930, 	f1: 0.9082
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 139: 100%|██████████| 6507/6507 [04:33<00:00, 23.79it/s, loss=0.0082]
Train Epoch 139 ==> 	accuracy: 0.9050, 	precision: 0.9997, 	recall: 0.8103, 	specificity: 0.9998, 	f1: 0.8951
Test Epoch 139: 100%|██████████| 1768/1768 [00:35<00:00, 50.34it/s, loss=4.36]
Test Epoch 139 ==> 	accuracy: 0.9646, 	precision: 0.9717, 	recall: 0.8515, 	specificity: 0.9936, 	f1: 0.9077
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 140: 100%|██████████| 6507/6507 [04:29<00:00, 24.11it/s, loss=0.0436]
Train Epoch 140 ==> 	accuracy: 0.9028, 	precision: 0.9997, 	recall: 0.8059, 	specificity: 0.9998, 	f1: 0.8924
Test Epoch 140: 100%|██████████| 1768/1768 [00:35<00:00, 49.25it/s, loss=0.525]
Test Epoch 140 ==> 	accuracy: 0.9646, 	precision: 0.9715, 	recall: 0.8518, 	specificity: 0.9936, 	f1: 0.9077
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 141: 100%|██████████| 6507/6507 [04:40<00:00, 23.17it/s, loss=0.0109]
Train Epoch 141 ==> 	accuracy: 0.9052, 	precision: 0.9997, 	recall: 0.8105, 	specificity: 0.9998, 	f1: 0.8952
Test Epoch 141: 100%|██████████| 1768/1768 [00:34<00:00, 50.63it/s, loss=1.91]
Test Epoch 141 ==> 	accuracy: 0.9651, 	precision: 0.9711, 	recall: 0.8551, 	specificity: 0.9934, 	f1: 0.9094
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 142: 100%|██████████| 6507/6507 [04:43<00:00, 22.98it/s, loss=0.0124]
Train Epoch 142 ==> 	accuracy: 0.9027, 	precision: 0.9997, 	recall: 0.8056, 	specificity: 0.9998, 	f1: 0.8922
Test Epoch 142: 100%|██████████| 1768/1768 [00:35<00:00, 49.46it/s, loss=0.055]
Test Epoch 142 ==> 	accuracy: 0.9648, 	precision: 0.9713, 	recall: 0.8532, 	specificity: 0.9935, 	f1: 0.9084
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 143: 100%|██████████| 6507/6507 [04:45<00:00, 22.82it/s, loss=0.0049]
Train Epoch 143 ==> 	accuracy: 0.9051, 	precision: 0.9997, 	recall: 0.8103, 	specificity: 0.9998, 	f1: 0.8951
Test Epoch 143: 100%|██████████| 1768/1768 [00:34<00:00, 50.59it/s, loss=0.0914]
Test Epoch 143 ==> 	accuracy: 0.9651, 	precision: 0.9742, 	recall: 0.8520, 	specificity: 0.9942, 	f1: 0.9090
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 144: 100%|██████████| 6507/6507 [04:40<00:00, 23.23it/s, loss=0.0111]
Train Epoch 144 ==> 	accuracy: 0.9034, 	precision: 0.9997, 	recall: 0.8071, 	specificity: 0.9997, 	f1: 0.8931
Test Epoch 144: 100%|██████████| 1768/1768 [00:37<00:00, 47.15it/s, loss=0.121]
Test Epoch 144 ==> 	accuracy: 0.9655, 	precision: 0.9737, 	recall: 0.8544, 	specificity: 0.9941, 	f1: 0.9102
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 145: 100%|██████████| 6507/6507 [04:36<00:00, 23.53it/s, loss=0.0607]
Train Epoch 145 ==> 	accuracy: 0.9038, 	precision: 0.9997, 	recall: 0.8079, 	specificity: 0.9998, 	f1: 0.8936
Test Epoch 145: 100%|██████████| 1768/1768 [00:38<00:00, 45.72it/s, loss=0.634]
Test Epoch 145 ==> 	accuracy: 0.9658, 	precision: 0.9740, 	recall: 0.8558, 	specificity: 0.9941, 	f1: 0.9111
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 146: 100%|██████████| 6507/6507 [04:45<00:00, 22.81it/s, loss=0.0145]
Train Epoch 146 ==> 	accuracy: 0.9036, 	precision: 0.9997, 	recall: 0.8074, 	specificity: 0.9998, 	f1: 0.8933
Test Epoch 146: 100%|██████████| 1768/1768 [00:33<00:00, 53.53it/s, loss=0.223]
Test Epoch 146 ==> 	accuracy: 0.9652, 	precision: 0.9724, 	recall: 0.8540, 	specificity: 0.9938, 	f1: 0.9093
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 147: 100%|██████████| 6507/6507 [04:40<00:00, 23.20it/s, loss=0.0625]
Train Epoch 147 ==> 	accuracy: 0.9049, 	precision: 0.9997, 	recall: 0.8100, 	specificity: 0.9998, 	f1: 0.8949
Test Epoch 147: 100%|██████████| 1768/1768 [00:37<00:00, 47.08it/s, loss=0.375]
Test Epoch 147 ==> 	accuracy: 0.9652, 	precision: 0.9724, 	recall: 0.8542, 	specificity: 0.9938, 	f1: 0.9095
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 148: 100%|██████████| 6507/6507 [04:39<00:00, 23.28it/s, loss=0.0174]
Train Epoch 148 ==> 	accuracy: 0.9070, 	precision: 0.9997, 	recall: 0.8143, 	specificity: 0.9998, 	f1: 0.8975
Test Epoch 148: 100%|██████████| 1768/1768 [00:36<00:00, 48.80it/s, loss=1.07]
Test Epoch 148 ==> 	accuracy: 0.9658, 	precision: 0.9681, 	recall: 0.8611, 	specificity: 0.9927, 	f1: 0.9115
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 149: 100%|██████████| 6507/6507 [04:41<00:00, 23.08it/s, loss=0.602]
Train Epoch 149 ==> 	accuracy: 0.9037, 	precision: 0.9997, 	recall: 0.8076, 	specificity: 0.9998, 	f1: 0.8934
Test Epoch 149: 100%|██████████| 1768/1768 [00:37<00:00, 46.70it/s, loss=0.392]
Test Epoch 149 ==> 	accuracy: 0.9648, 	precision: 0.9710, 	recall: 0.8536, 	specificity: 0.9934, 	f1: 0.9085
Adjusting learning rate of group 0 to 5.8150e-06.

进程已结束，退出代码为 0

'''

'''
'../model_save_sigBlock4_focalWithMs_deformable_7mer_ab_multConv'
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6507/6507 [07:10<00:00, 15.10it/s, loss=0.156]
Train Epoch 0 ==> 	accuracy: 0.6383, 	precision: 0.9960, 	recall: 0.2776, 	specificity: 0.9989, 	f1: 0.4342
Test Epoch 0: 100%|██████████| 1768/1768 [00:49<00:00, 35.39it/s, loss=0.426]
Test Epoch 0 ==> 	accuracy: 0.9130, 	precision: 0.9653, 	recall: 0.5959, 	specificity: 0.9945, 	f1: 0.7369
Train Epoch 1: 100%|██████████| 6507/6507 [08:14<00:00, 13.15it/s, loss=0.114]
Train Epoch 1 ==> 	accuracy: 0.7379, 	precision: 0.9975, 	recall: 0.4769, 	specificity: 0.9988, 	f1: 0.6453
Test Epoch 1: 100%|██████████| 1768/1768 [00:51<00:00, 34.33it/s, loss=0.542]
Test Epoch 1 ==> 	accuracy: 0.9216, 	precision: 0.9787, 	recall: 0.6306, 	specificity: 0.9965, 	f1: 0.7670
Train Epoch 2: 100%|██████████| 6507/6507 [08:25<00:00, 12.87it/s, loss=0.0975]
Train Epoch 2 ==> 	accuracy: 0.7709, 	precision: 0.9981, 	recall: 0.5429, 	specificity: 0.9990, 	f1: 0.7033
Test Epoch 2: 100%|██████████| 1768/1768 [00:52<00:00, 33.81it/s, loss=0.325]
Test Epoch 2 ==> 	accuracy: 0.9233, 	precision: 0.9831, 	recall: 0.6360, 	specificity: 0.9972, 	f1: 0.7724
Train Epoch 3: 100%|██████████| 6507/6507 [08:28<00:00, 12.80it/s, loss=0.152]
Train Epoch 3 ==> 	accuracy: 0.7845, 	precision: 0.9982, 	recall: 0.5701, 	specificity: 0.9990, 	f1: 0.7257
Test Epoch 3: 100%|██████████| 1768/1768 [00:51<00:00, 34.03it/s, loss=0.138]
Test Epoch 3 ==> 	accuracy: 0.9319, 	precision: 0.9757, 	recall: 0.6839, 	specificity: 0.9956, 	f1: 0.8042
Train Epoch 4: 100%|██████████| 6507/6507 [08:27<00:00, 12.83it/s, loss=0.0785]
Train Epoch 4 ==> 	accuracy: 0.7979, 	precision: 0.9985, 	recall: 0.5967, 	specificity: 0.9991, 	f1: 0.7470
Test Epoch 4: 100%|██████████| 1768/1768 [00:53<00:00, 33.19it/s, loss=0.462]
Test Epoch 4 ==> 	accuracy: 0.9270, 	precision: 0.9826, 	recall: 0.6548, 	specificity: 0.9970, 	f1: 0.7859
Train Epoch 5: 100%|██████████| 6507/6507 [08:38<00:00, 12.55it/s, loss=0.193]
Train Epoch 5 ==> 	accuracy: 0.8050, 	precision: 0.9986, 	recall: 0.6109, 	specificity: 0.9992, 	f1: 0.7581
Test Epoch 5: 100%|██████████| 1768/1768 [00:52<00:00, 33.94it/s, loss=0.207]
Test Epoch 5 ==> 	accuracy: 0.9356, 	precision: 0.9769, 	recall: 0.7018, 	specificity: 0.9957, 	f1: 0.8168
Train Epoch 6: 100%|██████████| 6507/6507 [08:33<00:00, 12.68it/s, loss=0.142]
Train Epoch 6 ==> 	accuracy: 0.8079, 	precision: 0.9988, 	recall: 0.6165, 	specificity: 0.9993, 	f1: 0.7624
Test Epoch 6: 100%|██████████| 1768/1768 [00:53<00:00, 33.23it/s, loss=0.264]
Test Epoch 6 ==> 	accuracy: 0.9318, 	precision: 0.9865, 	recall: 0.6756, 	specificity: 0.9976, 	f1: 0.8020
Train Epoch 7: 100%|██████████| 6507/6507 [08:26<00:00, 12.86it/s, loss=0.116]
Train Epoch 7 ==> 	accuracy: 0.8172, 	precision: 0.9988, 	recall: 0.6353, 	specificity: 0.9992, 	f1: 0.7766
Test Epoch 7: 100%|██████████| 1768/1768 [00:51<00:00, 34.16it/s, loss=0.487]
Test Epoch 7 ==> 	accuracy: 0.9324, 	precision: 0.9906, 	recall: 0.6760, 	specificity: 0.9983, 	f1: 0.8036
Train Epoch 8: 100%|██████████| 6507/6507 [08:27<00:00, 12.82it/s, loss=0.0783]
Train Epoch 8 ==> 	accuracy: 0.8171, 	precision: 0.9988, 	recall: 0.6349, 	specificity: 0.9993, 	f1: 0.7763
Test Epoch 8: 100%|██████████| 1768/1768 [00:53<00:00, 33.18it/s, loss=0.352]
Test Epoch 8 ==> 	accuracy: 0.9419, 	precision: 0.9752, 	recall: 0.7347, 	specificity: 0.9952, 	f1: 0.8381
Train Epoch 9: 100%|██████████| 6507/6507 [08:24<00:00, 12.89it/s, loss=0.091]
Train Epoch 9 ==> 	accuracy: 0.8225, 	precision: 0.9990, 	recall: 0.6456, 	specificity: 0.9993, 	f1: 0.7843
Test Epoch 9: 100%|██████████| 1768/1768 [00:52<00:00, 33.69it/s, loss=0.326]
Test Epoch 9 ==> 	accuracy: 0.9383, 	precision: 0.9754, 	recall: 0.7163, 	specificity: 0.9954, 	f1: 0.8260
Train Epoch 10: 100%|██████████| 6507/6507 [08:29<00:00, 12.78it/s, loss=0.0848]
Train Epoch 10 ==> 	accuracy: 0.8273, 	precision: 0.9989, 	recall: 0.6553, 	specificity: 0.9993, 	f1: 0.7914
Test Epoch 10: 100%|██████████| 1768/1768 [00:52<00:00, 33.53it/s, loss=0.182]
Test Epoch 10 ==> 	accuracy: 0.9391, 	precision: 0.9773, 	recall: 0.7188, 	specificity: 0.9957, 	f1: 0.8283
Train Epoch 11: 100%|██████████| 6507/6507 [08:26<00:00, 12.85it/s, loss=0.273]
Train Epoch 11 ==> 	accuracy: 0.8300, 	precision: 0.9990, 	recall: 0.6607, 	specificity: 0.9994, 	f1: 0.7954
Test Epoch 11: 100%|██████████| 1768/1768 [00:53<00:00, 33.09it/s, loss=0.213]
Test Epoch 11 ==> 	accuracy: 0.9421, 	precision: 0.9810, 	recall: 0.7309, 	specificity: 0.9964, 	f1: 0.8377
Train Epoch 12: 100%|██████████| 6507/6507 [08:26<00:00, 12.85it/s, loss=0.241]
Train Epoch 12 ==> 	accuracy: 0.8293, 	precision: 0.9991, 	recall: 0.6593, 	specificity: 0.9994, 	f1: 0.7943
Test Epoch 12: 100%|██████████| 1768/1768 [00:50<00:00, 35.13it/s, loss=0.239]
Test Epoch 12 ==> 	accuracy: 0.9353, 	precision: 0.9841, 	recall: 0.6949, 	specificity: 0.9971, 	f1: 0.8146
Train Epoch 13: 100%|██████████| 6507/6507 [08:27<00:00, 12.82it/s, loss=0.171]
Train Epoch 13 ==> 	accuracy: 0.8325, 	precision: 0.9991, 	recall: 0.6656, 	specificity: 0.9994, 	f1: 0.7989
Test Epoch 13: 100%|██████████| 1768/1768 [00:53<00:00, 32.77it/s, loss=0.168]
Test Epoch 13 ==> 	accuracy: 0.9407, 	precision: 0.9853, 	recall: 0.7210, 	specificity: 0.9972, 	f1: 0.8327
Train Epoch 14: 100%|██████████| 6507/6507 [08:26<00:00, 12.85it/s, loss=0.0286]
Train Epoch 14 ==> 	accuracy: 0.8398, 	precision: 0.9992, 	recall: 0.6801, 	specificity: 0.9994, 	f1: 0.8093
Test Epoch 14: 100%|██████████| 1768/1768 [01:00<00:00, 29.18it/s, loss=0.614]
Test Epoch 14 ==> 	accuracy: 0.9434, 	precision: 0.9841, 	recall: 0.7350, 	specificity: 0.9970, 	f1: 0.8415
Train Epoch 15: 100%|██████████| 6507/6507 [08:24<00:00, 12.89it/s, loss=0.0812]
Train Epoch 15 ==> 	accuracy: 0.8348, 	precision: 0.9991, 	recall: 0.6703, 	specificity: 0.9994, 	f1: 0.8023
Test Epoch 15: 100%|██████████| 1768/1768 [00:54<00:00, 32.73it/s, loss=0.327]
Test Epoch 15 ==> 	accuracy: 0.9392, 	precision: 0.9807, 	recall: 0.7167, 	specificity: 0.9964, 	f1: 0.8282
Train Epoch 16: 100%|██████████| 6507/6507 [08:31<00:00, 12.72it/s, loss=0.129]
Train Epoch 16 ==> 	accuracy: 0.8393, 	precision: 0.9991, 	recall: 0.6792, 	specificity: 0.9994, 	f1: 0.8087
Test Epoch 16: 100%|██████████| 1768/1768 [00:52<00:00, 33.47it/s, loss=0.266]
Test Epoch 16 ==> 	accuracy: 0.9451, 	precision: 0.9847, 	recall: 0.7434, 	specificity: 0.9970, 	f1: 0.8472
Train Epoch 17: 100%|██████████| 6507/6507 [08:33<00:00, 12.66it/s, loss=0.0659]
Train Epoch 17 ==> 	accuracy: 0.8432, 	precision: 0.9992, 	recall: 0.6869, 	specificity: 0.9995, 	f1: 0.8142
Test Epoch 17: 100%|██████████| 1768/1768 [00:54<00:00, 32.37it/s, loss=0.319]
Test Epoch 17 ==> 	accuracy: 0.9465, 	precision: 0.9768, 	recall: 0.7565, 	specificity: 0.9954, 	f1: 0.8527
Train Epoch 18: 100%|██████████| 6507/6507 [08:33<00:00, 12.66it/s, loss=0.0518]
Train Epoch 18 ==> 	accuracy: 0.8442, 	precision: 0.9992, 	recall: 0.6889, 	specificity: 0.9995, 	f1: 0.8156
Test Epoch 18: 100%|██████████| 1768/1768 [00:55<00:00, 31.77it/s, loss=0.232]
Test Epoch 18 ==> 	accuracy: 0.9432, 	precision: 0.9811, 	recall: 0.7365, 	specificity: 0.9963, 	f1: 0.8414
Train Epoch 19: 100%|██████████| 6507/6507 [08:25<00:00, 12.87it/s, loss=0.113]
Train Epoch 19 ==> 	accuracy: 0.8449, 	precision: 0.9993, 	recall: 0.6903, 	specificity: 0.9995, 	f1: 0.8165
Test Epoch 19: 100%|██████████| 1768/1768 [00:52<00:00, 33.94it/s, loss=0.219]
Test Epoch 19 ==> 	accuracy: 0.9361, 	precision: 0.9893, 	recall: 0.6953, 	specificity: 0.9981, 	f1: 0.8167
Train Epoch 20: 100%|██████████| 6507/6507 [08:25<00:00, 12.86it/s, loss=0.0932]
Train Epoch 20 ==> 	accuracy: 0.8464, 	precision: 0.9993, 	recall: 0.6933, 	specificity: 0.9995, 	f1: 0.8186
Test Epoch 20: 100%|██████████| 1768/1768 [00:51<00:00, 34.26it/s, loss=0.355]
Test Epoch 20 ==> 	accuracy: 0.9470, 	precision: 0.9792, 	recall: 0.7572, 	specificity: 0.9959, 	f1: 0.8540
Train Epoch 21: 100%|██████████| 6507/6507 [08:31<00:00, 12.72it/s, loss=0.102]
Train Epoch 21 ==> 	accuracy: 0.8489, 	precision: 0.9993, 	recall: 0.6982, 	specificity: 0.9995, 	f1: 0.8221
Test Epoch 21: 100%|██████████| 1768/1768 [00:51<00:00, 34.50it/s, loss=0.0948]
Test Epoch 21 ==> 	accuracy: 0.9446, 	precision: 0.9898, 	recall: 0.7368, 	specificity: 0.9981, 	f1: 0.8448
Train Epoch 22: 100%|██████████| 6507/6507 [08:25<00:00, 12.88it/s, loss=0.0649]
Train Epoch 22 ==> 	accuracy: 0.8503, 	precision: 0.9993, 	recall: 0.7011, 	specificity: 0.9995, 	f1: 0.8240
Test Epoch 22: 100%|██████████| 1768/1768 [00:54<00:00, 32.55it/s, loss=0.126]
Test Epoch 22 ==> 	accuracy: 0.9437, 	precision: 0.9820, 	recall: 0.7381, 	specificity: 0.9965, 	f1: 0.8428
Train Epoch 23: 100%|██████████| 6507/6507 [08:28<00:00, 12.80it/s, loss=0.0854]
Train Epoch 23 ==> 	accuracy: 0.8501, 	precision: 0.9993, 	recall: 0.7007, 	specificity: 0.9995, 	f1: 0.8238
Test Epoch 23: 100%|██████████| 1768/1768 [00:53<00:00, 33.11it/s, loss=0.151]
Test Epoch 23 ==> 	accuracy: 0.9435, 	precision: 0.9829, 	recall: 0.7367, 	specificity: 0.9967, 	f1: 0.8422
Train Epoch 24: 100%|██████████| 6507/6507 [08:25<00:00, 12.87it/s, loss=0.0055]
Train Epoch 24 ==> 	accuracy: 0.8537, 	precision: 0.9993, 	recall: 0.7079, 	specificity: 0.9995, 	f1: 0.8287
Test Epoch 24: 100%|██████████| 1768/1768 [00:50<00:00, 34.78it/s, loss=3.03]
Test Epoch 24 ==> 	accuracy: 0.9466, 	precision: 0.9832, 	recall: 0.7520, 	specificity: 0.9967, 	f1: 0.8522
Train Epoch 25: 100%|██████████| 6507/6507 [08:20<00:00, 13.00it/s, loss=0.0752]
Train Epoch 25 ==> 	accuracy: 0.8536, 	precision: 0.9992, 	recall: 0.7077, 	specificity: 0.9995, 	f1: 0.8286
Test Epoch 25: 100%|██████████| 1768/1768 [00:52<00:00, 33.94it/s, loss=0.132]
Test Epoch 25 ==> 	accuracy: 0.9478, 	precision: 0.9760, 	recall: 0.7634, 	specificity: 0.9952, 	f1: 0.8567
Train Epoch 26: 100%|██████████| 6507/6507 [08:14<00:00, 13.15it/s, loss=0.07]
Train Epoch 26 ==> 	accuracy: 0.8544, 	precision: 0.9994, 	recall: 0.7092, 	specificity: 0.9996, 	f1: 0.8296
Test Epoch 26: 100%|██████████| 1768/1768 [00:56<00:00, 31.41it/s, loss=0.215]
Test Epoch 26 ==> 	accuracy: 0.9481, 	precision: 0.9843, 	recall: 0.7583, 	specificity: 0.9969, 	f1: 0.8566
Train Epoch 27: 100%|██████████| 6507/6507 [08:25<00:00, 12.88it/s, loss=0.0307]
Train Epoch 27 ==> 	accuracy: 0.8570, 	precision: 0.9993, 	recall: 0.7144, 	specificity: 0.9995, 	f1: 0.8332
Test Epoch 27: 100%|██████████| 1768/1768 [00:53<00:00, 32.97it/s, loss=0.108]
Test Epoch 27 ==> 	accuracy: 0.9476, 	precision: 0.9871, 	recall: 0.7536, 	specificity: 0.9975, 	f1: 0.8547
Train Epoch 28: 100%|██████████| 6507/6507 [08:32<00:00, 12.71it/s, loss=0.0808]
Train Epoch 28 ==> 	accuracy: 0.8561, 	precision: 0.9994, 	recall: 0.7126, 	specificity: 0.9996, 	f1: 0.8320
Test Epoch 28: 100%|██████████| 1768/1768 [00:51<00:00, 34.17it/s, loss=0.159]
Test Epoch 28 ==> 	accuracy: 0.9476, 	precision: 0.9858, 	recall: 0.7547, 	specificity: 0.9972, 	f1: 0.8549
Train Epoch 29: 100%|██████████| 6507/6507 [08:17<00:00, 13.07it/s, loss=0.11]
Train Epoch 29 ==> 	accuracy: 0.8546, 	precision: 0.9994, 	recall: 0.7096, 	specificity: 0.9996, 	f1: 0.8299
Test Epoch 29: 100%|██████████| 1768/1768 [00:51<00:00, 34.25it/s, loss=1.32]
Test Epoch 29 ==> 	accuracy: 0.9497, 	precision: 0.9869, 	recall: 0.7641, 	specificity: 0.9974, 	f1: 0.8613
Train Epoch 30: 100%|██████████| 6507/6507 [08:21<00:00, 12.97it/s, loss=0.0633]
Train Epoch 30 ==> 	accuracy: 0.8591, 	precision: 0.9994, 	recall: 0.7187, 	specificity: 0.9996, 	f1: 0.8361
Test Epoch 30: 100%|██████████| 1768/1768 [00:50<00:00, 35.13it/s, loss=0.207]
Test Epoch 30 ==> 	accuracy: 0.9487, 	precision: 0.9864, 	recall: 0.7597, 	specificity: 0.9973, 	f1: 0.8583
Train Epoch 31: 100%|██████████| 6507/6507 [08:22<00:00, 12.95it/s, loss=0.0911]
Train Epoch 31 ==> 	accuracy: 0.8626, 	precision: 0.9994, 	recall: 0.7257, 	specificity: 0.9996, 	f1: 0.8409
Test Epoch 31: 100%|██████████| 1768/1768 [00:51<00:00, 34.23it/s, loss=0.416]
Test Epoch 31 ==> 	accuracy: 0.9469, 	precision: 0.9745, 	recall: 0.7604, 	specificity: 0.9949, 	f1: 0.8542
Train Epoch 32: 100%|██████████| 6507/6507 [08:23<00:00, 12.92it/s, loss=0.0783]
Train Epoch 32 ==> 	accuracy: 0.8598, 	precision: 0.9994, 	recall: 0.7200, 	specificity: 0.9996, 	f1: 0.8370
Test Epoch 32: 100%|██████████| 1768/1768 [00:51<00:00, 34.11it/s, loss=0.505]
Test Epoch 32 ==> 	accuracy: 0.9498, 	precision: 0.9862, 	recall: 0.7653, 	specificity: 0.9972, 	f1: 0.8618
Train Epoch 33: 100%|██████████| 6507/6507 [08:17<00:00, 13.08it/s, loss=0.0464]
Train Epoch 33 ==> 	accuracy: 0.8613, 	precision: 0.9994, 	recall: 0.7231, 	specificity: 0.9996, 	f1: 0.8391
Test Epoch 33: 100%|██████████| 1768/1768 [00:51<00:00, 34.25it/s, loss=0.212]
Test Epoch 33 ==> 	accuracy: 0.9507, 	precision: 0.9885, 	recall: 0.7677, 	specificity: 0.9977, 	f1: 0.8642
Train Epoch 34: 100%|██████████| 6507/6507 [08:22<00:00, 12.94it/s, loss=0.0379]
Train Epoch 34 ==> 	accuracy: 0.8630, 	precision: 0.9994, 	recall: 0.7264, 	specificity: 0.9996, 	f1: 0.8413
Test Epoch 34: 100%|██████████| 1768/1768 [00:50<00:00, 35.03it/s, loss=0.151]
Test Epoch 34 ==> 	accuracy: 0.9501, 	precision: 0.9840, 	recall: 0.7687, 	specificity: 0.9968, 	f1: 0.8631
Train Epoch 35: 100%|██████████| 6507/6507 [08:11<00:00, 13.24it/s, loss=0.0697]
Train Epoch 35 ==> 	accuracy: 0.8650, 	precision: 0.9994, 	recall: 0.7305, 	specificity: 0.9996, 	f1: 0.8441
Test Epoch 35: 100%|██████████| 1768/1768 [00:51<00:00, 34.33it/s, loss=0.0934]
Test Epoch 35 ==> 	accuracy: 0.9503, 	precision: 0.9815, 	recall: 0.7717, 	specificity: 0.9963, 	f1: 0.8640
Train Epoch 36: 100%|██████████| 6507/6507 [08:28<00:00, 12.79it/s, loss=0.0753]
Train Epoch 36 ==> 	accuracy: 0.8664, 	precision: 0.9994, 	recall: 0.7332, 	specificity: 0.9996, 	f1: 0.8459
Test Epoch 36: 100%|██████████| 1768/1768 [00:51<00:00, 34.09it/s, loss=1.04]
Test Epoch 36 ==> 	accuracy: 0.9540, 	precision: 0.9767, 	recall: 0.7941, 	specificity: 0.9951, 	f1: 0.8760
Train Epoch 37: 100%|██████████| 6507/6507 [08:12<00:00, 13.22it/s, loss=0.095]
Train Epoch 37 ==> 	accuracy: 0.8647, 	precision: 0.9994, 	recall: 0.7297, 	specificity: 0.9996, 	f1: 0.8435
Test Epoch 37: 100%|██████████| 1768/1768 [00:51<00:00, 34.60it/s, loss=0.2]
Test Epoch 37 ==> 	accuracy: 0.9513, 	precision: 0.9855, 	recall: 0.7735, 	specificity: 0.9971, 	f1: 0.8667
Train Epoch 38: 100%|██████████| 6507/6507 [08:19<00:00, 13.04it/s, loss=0.105]
Train Epoch 38 ==> 	accuracy: 0.8640, 	precision: 0.9994, 	recall: 0.7283, 	specificity: 0.9996, 	f1: 0.8426
Test Epoch 38: 100%|██████████| 1768/1768 [00:51<00:00, 34.56it/s, loss=0.274]
Test Epoch 38 ==> 	accuracy: 0.9497, 	precision: 0.9832, 	recall: 0.7672, 	specificity: 0.9966, 	f1: 0.8619
Train Epoch 39: 100%|██████████| 6507/6507 [08:12<00:00, 13.21it/s, loss=0.153]
Train Epoch 39 ==> 	accuracy: 0.8685, 	precision: 0.9994, 	recall: 0.7374, 	specificity: 0.9996, 	f1: 0.8486
Test Epoch 39: 100%|██████████| 1768/1768 [00:51<00:00, 34.25it/s, loss=0.472]
Test Epoch 39 ==> 	accuracy: 0.9506, 	precision: 0.9810, 	recall: 0.7735, 	specificity: 0.9961, 	f1: 0.8650
Train Epoch 40: 100%|██████████| 6507/6507 [07:57<00:00, 13.62it/s, loss=0.0082]
Train Epoch 40 ==> 	accuracy: 0.8686, 	precision: 0.9995, 	recall: 0.7377, 	specificity: 0.9996, 	f1: 0.8488
Test Epoch 40: 100%|██████████| 1768/1768 [00:53<00:00, 33.15it/s, loss=0.301]
Test Epoch 40 ==> 	accuracy: 0.9516, 	precision: 0.9769, 	recall: 0.7820, 	specificity: 0.9953, 	f1: 0.8687
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 41: 100%|██████████| 6507/6507 [07:57<00:00, 13.63it/s, loss=0.0121]
Train Epoch 41 ==> 	accuracy: 0.8681, 	precision: 0.9994, 	recall: 0.7365, 	specificity: 0.9996, 	f1: 0.8481
Test Epoch 41: 100%|██████████| 1768/1768 [00:53<00:00, 32.87it/s, loss=0.777]
Test Epoch 41 ==> 	accuracy: 0.9486, 	precision: 0.9863, 	recall: 0.7594, 	specificity: 0.9973, 	f1: 0.8581
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 42: 100%|██████████| 6507/6507 [08:01<00:00, 13.52it/s, loss=0.0164]
Train Epoch 42 ==> 	accuracy: 0.8690, 	precision: 0.9995, 	recall: 0.7383, 	specificity: 0.9996, 	f1: 0.8493
Test Epoch 42: 100%|██████████| 1768/1768 [00:51<00:00, 34.12it/s, loss=0.894]
Test Epoch 42 ==> 	accuracy: 0.9497, 	precision: 0.9799, 	recall: 0.7697, 	specificity: 0.9959, 	f1: 0.8622
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 43: 100%|██████████| 6507/6507 [08:01<00:00, 13.50it/s, loss=0.126]
Train Epoch 43 ==> 	accuracy: 0.8688, 	precision: 0.9995, 	recall: 0.7381, 	specificity: 0.9996, 	f1: 0.8491
Test Epoch 43: 100%|██████████| 1768/1768 [00:53<00:00, 32.90it/s, loss=0.0445]
Test Epoch 43 ==> 	accuracy: 0.9498, 	precision: 0.9841, 	recall: 0.7671, 	specificity: 0.9968, 	f1: 0.8621
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 44: 100%|██████████| 6507/6507 [07:51<00:00, 13.80it/s, loss=0.0182]
Train Epoch 44 ==> 	accuracy: 0.8717, 	precision: 0.9995, 	recall: 0.7438, 	specificity: 0.9996, 	f1: 0.8529
Test Epoch 44: 100%|██████████| 1768/1768 [00:50<00:00, 34.95it/s, loss=0.27]
Test Epoch 44 ==> 	accuracy: 0.9530, 	precision: 0.9845, 	recall: 0.7827, 	specificity: 0.9968, 	f1: 0.8721
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 45: 100%|██████████| 6507/6507 [07:54<00:00, 13.72it/s, loss=0.0056]
Train Epoch 45 ==> 	accuracy: 0.8706, 	precision: 0.9995, 	recall: 0.7417, 	specificity: 0.9996, 	f1: 0.8515
Test Epoch 45: 100%|██████████| 1768/1768 [00:53<00:00, 32.97it/s, loss=0.163]
Test Epoch 45 ==> 	accuracy: 0.9498, 	precision: 0.9865, 	recall: 0.7651, 	specificity: 0.9973, 	f1: 0.8618
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 46: 100%|██████████| 6507/6507 [07:53<00:00, 13.75it/s, loss=0.0008]
Train Epoch 46 ==> 	accuracy: 0.8731, 	precision: 0.9995, 	recall: 0.7465, 	specificity: 0.9996, 	f1: 0.8547
Test Epoch 46: 100%|██████████| 1768/1768 [00:51<00:00, 34.31it/s, loss=0.257]
Test Epoch 46 ==> 	accuracy: 0.9523, 	precision: 0.9906, 	recall: 0.7740, 	specificity: 0.9981, 	f1: 0.8690
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 47: 100%|██████████| 6507/6507 [07:57<00:00, 13.61it/s, loss=0.0485]
Train Epoch 47 ==> 	accuracy: 0.8730, 	precision: 0.9995, 	recall: 0.7463, 	specificity: 0.9996, 	f1: 0.8546
Test Epoch 47: 100%|██████████| 1768/1768 [00:53<00:00, 32.99it/s, loss=0.27]
Test Epoch 47 ==> 	accuracy: 0.9526, 	precision: 0.9848, 	recall: 0.7800, 	specificity: 0.9969, 	f1: 0.8706
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 48: 100%|██████████| 6507/6507 [07:50<00:00, 13.83it/s, loss=0.043]
Train Epoch 48 ==> 	accuracy: 0.8757, 	precision: 0.9995, 	recall: 0.7517, 	specificity: 0.9996, 	f1: 0.8581
Test Epoch 48: 100%|██████████| 1768/1768 [00:51<00:00, 34.15it/s, loss=0.114]
Test Epoch 48 ==> 	accuracy: 0.9559, 	precision: 0.9846, 	recall: 0.7968, 	specificity: 0.9968, 	f1: 0.8808
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 49: 100%|██████████| 6507/6507 [07:50<00:00, 13.83it/s, loss=0.145]
Train Epoch 49 ==> 	accuracy: 0.8756, 	precision: 0.9995, 	recall: 0.7515, 	specificity: 0.9996, 	f1: 0.8579
Test Epoch 49: 100%|██████████| 1768/1768 [00:52<00:00, 33.49it/s, loss=0.0885]
Test Epoch 49 ==> 	accuracy: 0.9552, 	precision: 0.9882, 	recall: 0.7906, 	specificity: 0.9976, 	f1: 0.8784
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 50: 100%|██████████| 6507/6507 [07:53<00:00, 13.73it/s, loss=0.0667]
Train Epoch 50 ==> 	accuracy: 0.8802, 	precision: 0.9995, 	recall: 0.7608, 	specificity: 0.9996, 	f1: 0.8640
Test Epoch 50: 100%|██████████| 1768/1768 [00:55<00:00, 31.58it/s, loss=0.24]
Test Epoch 50 ==> 	accuracy: 0.9535, 	precision: 0.9857, 	recall: 0.7839, 	specificity: 0.9971, 	f1: 0.8733
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 51: 100%|██████████| 6507/6507 [07:46<00:00, 13.95it/s, loss=0.011]
Train Epoch 51 ==> 	accuracy: 0.8780, 	precision: 0.9995, 	recall: 0.7563, 	specificity: 0.9997, 	f1: 0.8611
Test Epoch 51: 100%|██████████| 1768/1768 [00:50<00:00, 35.14it/s, loss=0.293]
Test Epoch 51 ==> 	accuracy: 0.9533, 	precision: 0.9861, 	recall: 0.7826, 	specificity: 0.9972, 	f1: 0.8727
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 52: 100%|██████████| 6507/6507 [07:49<00:00, 13.87it/s, loss=0.014]
Train Epoch 52 ==> 	accuracy: 0.8806, 	precision: 0.9996, 	recall: 0.7616, 	specificity: 0.9997, 	f1: 0.8645
Test Epoch 52: 100%|██████████| 1768/1768 [00:52<00:00, 33.64it/s, loss=0.155]
Test Epoch 52 ==> 	accuracy: 0.9579, 	precision: 0.9803, 	recall: 0.8104, 	specificity: 0.9958, 	f1: 0.8873
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 53: 100%|██████████| 6507/6507 [07:49<00:00, 13.86it/s, loss=0.0321]
Train Epoch 53 ==> 	accuracy: 0.8821, 	precision: 0.9995, 	recall: 0.7646, 	specificity: 0.9996, 	f1: 0.8664
Test Epoch 53: 100%|██████████| 1768/1768 [00:50<00:00, 35.13it/s, loss=0.125]
Test Epoch 53 ==> 	accuracy: 0.9557, 	precision: 0.9797, 	recall: 0.7999, 	specificity: 0.9957, 	f1: 0.8807
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 54: 100%|██████████| 6507/6507 [07:52<00:00, 13.76it/s, loss=0.022]
Train Epoch 54 ==> 	accuracy: 0.8794, 	precision: 0.9996, 	recall: 0.7591, 	specificity: 0.9997, 	f1: 0.8629
Test Epoch 54: 100%|██████████| 1768/1768 [00:53<00:00, 33.15it/s, loss=0.306]
Test Epoch 54 ==> 	accuracy: 0.9566, 	precision: 0.9841, 	recall: 0.8009, 	specificity: 0.9967, 	f1: 0.8831
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 55: 100%|██████████| 6507/6507 [07:50<00:00, 13.84it/s, loss=0.0344]
Train Epoch 55 ==> 	accuracy: 0.8827, 	precision: 0.9996, 	recall: 0.7658, 	specificity: 0.9997, 	f1: 0.8672
Test Epoch 55: 100%|██████████| 1768/1768 [00:55<00:00, 31.63it/s, loss=0.0675]
Test Epoch 55 ==> 	accuracy: 0.9568, 	precision: 0.9835, 	recall: 0.8021, 	specificity: 0.9965, 	f1: 0.8836
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 56: 100%|██████████| 6507/6507 [07:52<00:00, 13.78it/s, loss=0.0362]
Train Epoch 56 ==> 	accuracy: 0.8807, 	precision: 0.9996, 	recall: 0.7617, 	specificity: 0.9997, 	f1: 0.8646
Test Epoch 56: 100%|██████████| 1768/1768 [00:54<00:00, 32.65it/s, loss=0.0924]
Test Epoch 56 ==> 	accuracy: 0.9543, 	precision: 0.9854, 	recall: 0.7884, 	specificity: 0.9970, 	f1: 0.8759
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 57: 100%|██████████| 6507/6507 [08:04<00:00, 13.43it/s, loss=0.191]
Train Epoch 57 ==> 	accuracy: 0.8852, 	precision: 0.9996, 	recall: 0.7707, 	specificity: 0.9997, 	f1: 0.8703
Test Epoch 57: 100%|██████████| 1768/1768 [00:51<00:00, 34.43it/s, loss=0.1]
Test Epoch 57 ==> 	accuracy: 0.9586, 	precision: 0.9804, 	recall: 0.8138, 	specificity: 0.9958, 	f1: 0.8894
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 58: 100%|██████████| 6507/6507 [07:53<00:00, 13.74it/s, loss=0.0103]
Train Epoch 58 ==> 	accuracy: 0.8818, 	precision: 0.9996, 	recall: 0.7640, 	specificity: 0.9997, 	f1: 0.8661
Test Epoch 58: 100%|██████████| 1768/1768 [00:58<00:00, 30.13it/s, loss=0.149]
Test Epoch 58 ==> 	accuracy: 0.9548, 	precision: 0.9857, 	recall: 0.7903, 	specificity: 0.9971, 	f1: 0.8773
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 59: 100%|██████████| 6507/6507 [07:58<00:00, 13.59it/s, loss=0.0161]
Train Epoch 59 ==> 	accuracy: 0.8855, 	precision: 0.9996, 	recall: 0.7713, 	specificity: 0.9997, 	f1: 0.8707
Test Epoch 59: 100%|██████████| 1768/1768 [00:51<00:00, 34.31it/s, loss=0.0649]
Test Epoch 59 ==> 	accuracy: 0.9575, 	precision: 0.9806, 	recall: 0.8084, 	specificity: 0.9959, 	f1: 0.8862
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 60: 100%|██████████| 6507/6507 [07:55<00:00, 13.68it/s, loss=0.506]
Train Epoch 60 ==> 	accuracy: 0.8864, 	precision: 0.9996, 	recall: 0.7731, 	specificity: 0.9997, 	f1: 0.8719
Test Epoch 60: 100%|██████████| 1768/1768 [00:50<00:00, 35.04it/s, loss=0.423]
Test Epoch 60 ==> 	accuracy: 0.9547, 	precision: 0.9867, 	recall: 0.7890, 	specificity: 0.9973, 	f1: 0.8769
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 61: 100%|██████████| 6507/6507 [07:47<00:00, 13.91it/s, loss=0.0134]
Train Epoch 61 ==> 	accuracy: 0.8904, 	precision: 0.9996, 	recall: 0.7812, 	specificity: 0.9997, 	f1: 0.8770
Test Epoch 61: 100%|██████████| 1768/1768 [00:50<00:00, 35.16it/s, loss=0.102]
Test Epoch 61 ==> 	accuracy: 0.9573, 	precision: 0.9831, 	recall: 0.8050, 	specificity: 0.9964, 	f1: 0.8852
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 62: 100%|██████████| 6507/6507 [07:53<00:00, 13.75it/s, loss=0.189]
Train Epoch 62 ==> 	accuracy: 0.8861, 	precision: 0.9996, 	recall: 0.7724, 	specificity: 0.9997, 	f1: 0.8715
Test Epoch 62: 100%|██████████| 1768/1768 [00:50<00:00, 34.71it/s, loss=0.159]
Test Epoch 62 ==> 	accuracy: 0.9584, 	precision: 0.9820, 	recall: 0.8113, 	specificity: 0.9962, 	f1: 0.8885
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 63: 100%|██████████| 6507/6507 [07:48<00:00, 13.89it/s, loss=0.512]
Train Epoch 63 ==> 	accuracy: 0.8895, 	precision: 0.9996, 	recall: 0.7792, 	specificity: 0.9997, 	f1: 0.8758
Test Epoch 63: 100%|██████████| 1768/1768 [00:52<00:00, 33.59it/s, loss=0.863]
Test Epoch 63 ==> 	accuracy: 0.9572, 	precision: 0.9845, 	recall: 0.8034, 	specificity: 0.9968, 	f1: 0.8848
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 64: 100%|██████████| 6507/6507 [07:55<00:00, 13.69it/s, loss=0.016]
Train Epoch 64 ==> 	accuracy: 0.8892, 	precision: 0.9996, 	recall: 0.7787, 	specificity: 0.9997, 	f1: 0.8755
Test Epoch 64: 100%|██████████| 1768/1768 [00:49<00:00, 35.43it/s, loss=0.182]
Test Epoch 64 ==> 	accuracy: 0.9567, 	precision: 0.9847, 	recall: 0.8006, 	specificity: 0.9968, 	f1: 0.8832
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 65: 100%|██████████| 6507/6507 [07:54<00:00, 13.71it/s, loss=0.0866]
Train Epoch 65 ==> 	accuracy: 0.8908, 	precision: 0.9996, 	recall: 0.7819, 	specificity: 0.9997, 	f1: 0.8775
Test Epoch 65: 100%|██████████| 1768/1768 [00:51<00:00, 34.56it/s, loss=1.07]
Test Epoch 65 ==> 	accuracy: 0.9579, 	precision: 0.9845, 	recall: 0.8071, 	specificity: 0.9967, 	f1: 0.8870
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 66: 100%|██████████| 6507/6507 [07:55<00:00, 13.69it/s, loss=0.0274]
Train Epoch 66 ==> 	accuracy: 0.8905, 	precision: 0.9996, 	recall: 0.7813, 	specificity: 0.9997, 	f1: 0.8771
Test Epoch 66: 100%|██████████| 1768/1768 [00:53<00:00, 33.20it/s, loss=0.115]
Test Epoch 66 ==> 	accuracy: 0.9575, 	precision: 0.9829, 	recall: 0.8060, 	specificity: 0.9964, 	f1: 0.8857
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 67: 100%|██████████| 6507/6507 [07:55<00:00, 13.67it/s, loss=0.0156]
Train Epoch 67 ==> 	accuracy: 0.8911, 	precision: 0.9996, 	recall: 0.7824, 	specificity: 0.9997, 	f1: 0.8778
Test Epoch 67: 100%|██████████| 1768/1768 [00:50<00:00, 34.69it/s, loss=0.179]
Test Epoch 67 ==> 	accuracy: 0.9589, 	precision: 0.9820, 	recall: 0.8137, 	specificity: 0.9962, 	f1: 0.8900
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 68: 100%|██████████| 6507/6507 [07:48<00:00, 13.88it/s, loss=0.0101]
Train Epoch 68 ==> 	accuracy: 0.8931, 	precision: 0.9996, 	recall: 0.7865, 	specificity: 0.9997, 	f1: 0.8803
Test Epoch 68: 100%|██████████| 1768/1768 [00:52<00:00, 33.74it/s, loss=0.0411]
Test Epoch 68 ==> 	accuracy: 0.9598, 	precision: 0.9831, 	recall: 0.8173, 	specificity: 0.9964, 	f1: 0.8926
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 69: 100%|██████████| 6507/6507 [07:46<00:00, 13.95it/s, loss=0.0147]
Train Epoch 69 ==> 	accuracy: 0.8928, 	precision: 0.9996, 	recall: 0.7860, 	specificity: 0.9997, 	f1: 0.8800
Test Epoch 69: 100%|██████████| 1768/1768 [00:50<00:00, 34.95it/s, loss=0.149]
Test Epoch 69 ==> 	accuracy: 0.9583, 	precision: 0.9826, 	recall: 0.8104, 	specificity: 0.9963, 	f1: 0.8882
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 70: 100%|██████████| 6507/6507 [07:50<00:00, 13.83it/s, loss=0.355]
Train Epoch 70 ==> 	accuracy: 0.8931, 	precision: 0.9996, 	recall: 0.7865, 	specificity: 0.9997, 	f1: 0.8804
Test Epoch 70: 100%|██████████| 1768/1768 [00:53<00:00, 32.82it/s, loss=0.101]
Test Epoch 70 ==> 	accuracy: 0.9583, 	precision: 0.9816, 	recall: 0.8112, 	specificity: 0.9961, 	f1: 0.8883
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 71: 100%|██████████| 6507/6507 [07:46<00:00, 13.94it/s, loss=0.0228]
Train Epoch 71 ==> 	accuracy: 0.8943, 	precision: 0.9997, 	recall: 0.7889, 	specificity: 0.9997, 	f1: 0.8819
Test Epoch 71: 100%|██████████| 1768/1768 [00:53<00:00, 32.81it/s, loss=0.314]
Test Epoch 71 ==> 	accuracy: 0.9609, 	precision: 0.9830, 	recall: 0.8229, 	specificity: 0.9963, 	f1: 0.8958
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 72: 100%|██████████| 6507/6507 [07:50<00:00, 13.83it/s, loss=0.0309]
Train Epoch 72 ==> 	accuracy: 0.8958, 	precision: 0.9996, 	recall: 0.7920, 	specificity: 0.9997, 	f1: 0.8838
Test Epoch 72: 100%|██████████| 1768/1768 [00:52<00:00, 33.36it/s, loss=0.303]
Test Epoch 72 ==> 	accuracy: 0.9597, 	precision: 0.9861, 	recall: 0.8145, 	specificity: 0.9970, 	f1: 0.8921
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 73: 100%|██████████| 6507/6507 [07:54<00:00, 13.70it/s, loss=0.0148]
Train Epoch 73 ==> 	accuracy: 0.8951, 	precision: 0.9996, 	recall: 0.7906, 	specificity: 0.9997, 	f1: 0.8829
Test Epoch 73: 100%|██████████| 1768/1768 [00:49<00:00, 35.74it/s, loss=2.84]
Test Epoch 73 ==> 	accuracy: 0.9606, 	precision: 0.9829, 	recall: 0.8219, 	specificity: 0.9963, 	f1: 0.8952
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 74: 100%|██████████| 6507/6507 [07:51<00:00, 13.80it/s, loss=1.88]
Train Epoch 74 ==> 	accuracy: 0.8963, 	precision: 0.9996, 	recall: 0.7928, 	specificity: 0.9997, 	f1: 0.8843
Test Epoch 74: 100%|██████████| 1768/1768 [00:50<00:00, 34.86it/s, loss=0.0804]
Test Epoch 74 ==> 	accuracy: 0.9588, 	precision: 0.9862, 	recall: 0.8097, 	specificity: 0.9971, 	f1: 0.8893
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 75: 100%|██████████| 6507/6507 [07:43<00:00, 14.05it/s, loss=0.0414]
Train Epoch 75 ==> 	accuracy: 0.8969, 	precision: 0.9997, 	recall: 0.7940, 	specificity: 0.9997, 	f1: 0.8851
Test Epoch 75: 100%|██████████| 1768/1768 [00:53<00:00, 33.16it/s, loss=0.425]
Test Epoch 75 ==> 	accuracy: 0.9597, 	precision: 0.9818, 	recall: 0.8181, 	specificity: 0.9961, 	f1: 0.8925
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 76: 100%|██████████| 6507/6507 [07:45<00:00, 13.99it/s, loss=0.349]
Train Epoch 76 ==> 	accuracy: 0.8994, 	precision: 0.9996, 	recall: 0.7992, 	specificity: 0.9997, 	f1: 0.8882
Test Epoch 76: 100%|██████████| 1768/1768 [00:53<00:00, 33.34it/s, loss=0.128]
Test Epoch 76 ==> 	accuracy: 0.9607, 	precision: 0.9799, 	recall: 0.8247, 	specificity: 0.9957, 	f1: 0.8956
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 77: 100%|██████████| 6507/6507 [07:49<00:00, 13.85it/s, loss=0.0101]
Train Epoch 77 ==> 	accuracy: 0.8982, 	precision: 0.9996, 	recall: 0.7968, 	specificity: 0.9997, 	f1: 0.8867
Test Epoch 77: 100%|██████████| 1768/1768 [00:57<00:00, 30.80it/s, loss=0.824]
Test Epoch 77 ==> 	accuracy: 0.9598, 	precision: 0.9834, 	recall: 0.8172, 	specificity: 0.9965, 	f1: 0.8927
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 78: 100%|██████████| 6507/6507 [07:50<00:00, 13.82it/s, loss=0.0153]
Train Epoch 78 ==> 	accuracy: 0.8991, 	precision: 0.9996, 	recall: 0.7986, 	specificity: 0.9997, 	f1: 0.8879
Test Epoch 78: 100%|██████████| 1768/1768 [00:50<00:00, 34.67it/s, loss=0.118]
Test Epoch 78 ==> 	accuracy: 0.9606, 	precision: 0.9848, 	recall: 0.8201, 	specificity: 0.9967, 	f1: 0.8949
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 79: 100%|██████████| 6507/6507 [07:44<00:00, 14.00it/s, loss=0.013]
Train Epoch 79 ==> 	accuracy: 0.8973, 	precision: 0.9997, 	recall: 0.7948, 	specificity: 0.9997, 	f1: 0.8855
Test Epoch 79: 100%|██████████| 1768/1768 [00:55<00:00, 31.89it/s, loss=0.34]
Test Epoch 79 ==> 	accuracy: 0.9595, 	precision: 0.9850, 	recall: 0.8142, 	specificity: 0.9968, 	f1: 0.8915
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 80: 100%|██████████| 6507/6507 [07:49<00:00, 13.86it/s, loss=0.0268]
Train Epoch 80 ==> 	accuracy: 0.8985, 	precision: 0.9997, 	recall: 0.7973, 	specificity: 0.9997, 	f1: 0.8871
Test Epoch 80: 100%|██████████| 1768/1768 [00:51<00:00, 34.16it/s, loss=0.721]
Test Epoch 80 ==> 	accuracy: 0.9597, 	precision: 0.9837, 	recall: 0.8165, 	specificity: 0.9965, 	f1: 0.8923
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 81: 100%|██████████| 6507/6507 [07:46<00:00, 13.96it/s, loss=1.25]
Train Epoch 81 ==> 	accuracy: 0.9009, 	precision: 0.9996, 	recall: 0.8020, 	specificity: 0.9997, 	f1: 0.8900
Test Epoch 81: 100%|██████████| 1768/1768 [00:53<00:00, 33.00it/s, loss=0.247]
Test Epoch 81 ==> 	accuracy: 0.9609, 	precision: 0.9815, 	recall: 0.8244, 	specificity: 0.9960, 	f1: 0.8961
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 82: 100%|██████████| 6507/6507 [07:45<00:00, 13.97it/s, loss=0.0158]
Train Epoch 82 ==> 	accuracy: 0.8992, 	precision: 0.9997, 	recall: 0.7987, 	specificity: 0.9997, 	f1: 0.8880
Test Epoch 82: 100%|██████████| 1768/1768 [00:51<00:00, 34.45it/s, loss=0.168]
Test Epoch 82 ==> 	accuracy: 0.9615, 	precision: 0.9800, 	recall: 0.8286, 	specificity: 0.9957, 	f1: 0.8980
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 83: 100%|██████████| 6507/6507 [07:42<00:00, 14.08it/s, loss=0.0188]
Train Epoch 83 ==> 	accuracy: 0.9011, 	precision: 0.9997, 	recall: 0.8024, 	specificity: 0.9998, 	f1: 0.8903
Test Epoch 83: 100%|██████████| 1768/1768 [00:55<00:00, 31.64it/s, loss=1.35]
Test Epoch 83 ==> 	accuracy: 0.9617, 	precision: 0.9814, 	recall: 0.8284, 	specificity: 0.9960, 	f1: 0.8984
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 84: 100%|██████████| 6507/6507 [07:44<00:00, 14.01it/s, loss=0.0238]
Train Epoch 84 ==> 	accuracy: 0.9012, 	precision: 0.9997, 	recall: 0.8026, 	specificity: 0.9998, 	f1: 0.8904
Test Epoch 84: 100%|██████████| 1768/1768 [00:53<00:00, 33.31it/s, loss=0.0902]
Test Epoch 84 ==> 	accuracy: 0.9618, 	precision: 0.9836, 	recall: 0.8271, 	specificity: 0.9965, 	f1: 0.8986
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 85: 100%|██████████| 6507/6507 [07:50<00:00, 13.83it/s, loss=0.0229]
Train Epoch 85 ==> 	accuracy: 0.9042, 	precision: 0.9996, 	recall: 0.8086, 	specificity: 0.9997, 	f1: 0.8940
Test Epoch 85: 100%|██████████| 1768/1768 [00:52<00:00, 33.77it/s, loss=0.173]
Test Epoch 85 ==> 	accuracy: 0.9609, 	precision: 0.9831, 	recall: 0.8229, 	specificity: 0.9964, 	f1: 0.8959
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 86: 100%|██████████| 6507/6507 [07:46<00:00, 13.93it/s, loss=0.01]
Train Epoch 86 ==> 	accuracy: 0.9018, 	precision: 0.9997, 	recall: 0.8038, 	specificity: 0.9997, 	f1: 0.8911
Test Epoch 86: 100%|██████████| 1768/1768 [00:52<00:00, 33.77it/s, loss=0.217]
Test Epoch 86 ==> 	accuracy: 0.9620, 	precision: 0.9800, 	recall: 0.8311, 	specificity: 0.9956, 	f1: 0.8995
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 87: 100%|██████████| 6507/6507 [07:51<00:00, 13.81it/s, loss=0.0626]
Train Epoch 87 ==> 	accuracy: 0.9030, 	precision: 0.9997, 	recall: 0.8062, 	specificity: 0.9998, 	f1: 0.8926
Test Epoch 87: 100%|██████████| 1768/1768 [00:51<00:00, 34.34it/s, loss=0.878]
Test Epoch 87 ==> 	accuracy: 0.9613, 	precision: 0.9776, 	recall: 0.8297, 	specificity: 0.9951, 	f1: 0.8976
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 88: 100%|██████████| 6507/6507 [07:46<00:00, 13.94it/s, loss=0.146]
Train Epoch 88 ==> 	accuracy: 0.9041, 	precision: 0.9997, 	recall: 0.8085, 	specificity: 0.9997, 	f1: 0.8940
Test Epoch 88: 100%|██████████| 1768/1768 [00:51<00:00, 34.15it/s, loss=3.37]
Test Epoch 88 ==> 	accuracy: 0.9626, 	precision: 0.9790, 	recall: 0.8353, 	specificity: 0.9954, 	f1: 0.9014
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 89: 100%|██████████| 6507/6507 [07:44<00:00, 14.02it/s, loss=0.0058]
Train Epoch 89 ==> 	accuracy: 0.9058, 	precision: 0.9997, 	recall: 0.8118, 	specificity: 0.9997, 	f1: 0.8960
Test Epoch 89: 100%|██████████| 1768/1768 [00:51<00:00, 34.45it/s, loss=0.456]
Test Epoch 89 ==> 	accuracy: 0.9627, 	precision: 0.9800, 	recall: 0.8346, 	specificity: 0.9956, 	f1: 0.9015
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 90: 100%|██████████| 6507/6507 [07:46<00:00, 13.94it/s, loss=0.0542]
Train Epoch 90 ==> 	accuracy: 0.9031, 	precision: 0.9997, 	recall: 0.8065, 	specificity: 0.9997, 	f1: 0.8928
Test Epoch 90: 100%|██████████| 1768/1768 [00:51<00:00, 34.16it/s, loss=0.156]
Test Epoch 90 ==> 	accuracy: 0.9611, 	precision: 0.9822, 	recall: 0.8246, 	specificity: 0.9962, 	f1: 0.8965
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 91: 100%|██████████| 6507/6507 [07:58<00:00, 13.60it/s, loss=0.0147]
Train Epoch 91 ==> 	accuracy: 0.9059, 	precision: 0.9997, 	recall: 0.8120, 	specificity: 0.9997, 	f1: 0.8961
Test Epoch 91: 100%|██████████| 1768/1768 [00:53<00:00, 32.81it/s, loss=0.0718]
Test Epoch 91 ==> 	accuracy: 0.9628, 	precision: 0.9788, 	recall: 0.8364, 	specificity: 0.9953, 	f1: 0.9020
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 92: 100%|██████████| 6507/6507 [07:52<00:00, 13.78it/s, loss=0.0163]
Train Epoch 92 ==> 	accuracy: 0.9043, 	precision: 0.9997, 	recall: 0.8089, 	specificity: 0.9997, 	f1: 0.8942
Test Epoch 92: 100%|██████████| 1768/1768 [00:50<00:00, 35.01it/s, loss=0.0738]
Test Epoch 92 ==> 	accuracy: 0.9627, 	precision: 0.9819, 	recall: 0.8329, 	specificity: 0.9961, 	f1: 0.9013
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 93: 100%|██████████| 6507/6507 [07:52<00:00, 13.76it/s, loss=0.0163]
Train Epoch 93 ==> 	accuracy: 0.9056, 	precision: 0.9997, 	recall: 0.8114, 	specificity: 0.9998, 	f1: 0.8958
Test Epoch 93: 100%|██████████| 1768/1768 [00:54<00:00, 32.44it/s, loss=0.321]
Test Epoch 93 ==> 	accuracy: 0.9624, 	precision: 0.9772, 	recall: 0.8359, 	specificity: 0.9950, 	f1: 0.9010
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 94: 100%|██████████| 6507/6507 [07:48<00:00, 13.88it/s, loss=0.101]
Train Epoch 94 ==> 	accuracy: 0.9051, 	precision: 0.9997, 	recall: 0.8104, 	specificity: 0.9997, 	f1: 0.8951
Test Epoch 94: 100%|██████████| 1768/1768 [00:53<00:00, 33.22it/s, loss=0.0709]
Test Epoch 94 ==> 	accuracy: 0.9637, 	precision: 0.9770, 	recall: 0.8426, 	specificity: 0.9949, 	f1: 0.9048
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 95: 100%|██████████| 6507/6507 [07:51<00:00, 13.80it/s, loss=0.0423]
Train Epoch 95 ==> 	accuracy: 0.9033, 	precision: 0.9997, 	recall: 0.8068, 	specificity: 0.9998, 	f1: 0.8929
Test Epoch 95: 100%|██████████| 1768/1768 [00:50<00:00, 35.19it/s, loss=0.134]
Test Epoch 95 ==> 	accuracy: 0.9626, 	precision: 0.9818, 	recall: 0.8327, 	specificity: 0.9960, 	f1: 0.9011
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 96: 100%|██████████| 6507/6507 [07:49<00:00, 13.87it/s, loss=0.0363]
Train Epoch 96 ==> 	accuracy: 0.9062, 	precision: 0.9997, 	recall: 0.8127, 	specificity: 0.9997, 	f1: 0.8965
Test Epoch 96: 100%|██████████| 1768/1768 [00:51<00:00, 34.29it/s, loss=2.25]
Test Epoch 96 ==> 	accuracy: 0.9637, 	precision: 0.9807, 	recall: 0.8393, 	specificity: 0.9957, 	f1: 0.9045
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 97: 100%|██████████| 6507/6507 [07:46<00:00, 13.95it/s, loss=0.0089]
Train Epoch 97 ==> 	accuracy: 0.9049, 	precision: 0.9997, 	recall: 0.8100, 	specificity: 0.9997, 	f1: 0.8949
Test Epoch 97: 100%|██████████| 1768/1768 [00:56<00:00, 31.19it/s, loss=3.14]
Test Epoch 97 ==> 	accuracy: 0.9625, 	precision: 0.9802, 	recall: 0.8334, 	specificity: 0.9957, 	f1: 0.9009
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 98: 100%|██████████| 6507/6507 [07:51<00:00, 13.80it/s, loss=0.0477]
Train Epoch 98 ==> 	accuracy: 0.9079, 	precision: 0.9997, 	recall: 0.8161, 	specificity: 0.9998, 	f1: 0.8986
Test Epoch 98: 100%|██████████| 1768/1768 [00:50<00:00, 35.01it/s, loss=0.161]
Test Epoch 98 ==> 	accuracy: 0.9635, 	precision: 0.9816, 	recall: 0.8371, 	specificity: 0.9960, 	f1: 0.9036
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 99: 100%|██████████| 6507/6507 [07:54<00:00, 13.72it/s, loss=0.0007]
Train Epoch 99 ==> 	accuracy: 0.9060, 	precision: 0.9997, 	recall: 0.8123, 	specificity: 0.9997, 	f1: 0.8963
Test Epoch 99: 100%|██████████| 1768/1768 [00:54<00:00, 32.54it/s, loss=0.224]
Test Epoch 99 ==> 	accuracy: 0.9623, 	precision: 0.9833, 	recall: 0.8299, 	specificity: 0.9964, 	f1: 0.9001
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 100: 100%|██████████| 6507/6507 [07:56<00:00, 13.65it/s, loss=0.002]
Train Epoch 100 ==> 	accuracy: 0.9085, 	precision: 0.9997, 	recall: 0.8172, 	specificity: 0.9998, 	f1: 0.8993
Test Epoch 100: 100%|██████████| 1768/1768 [00:56<00:00, 31.28it/s, loss=2.82]
Test Epoch 100 ==> 	accuracy: 0.9635, 	precision: 0.9758, 	recall: 0.8424, 	specificity: 0.9946, 	f1: 0.9042
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 101: 100%|██████████| 6507/6507 [07:44<00:00, 14.02it/s, loss=0.0027]
Train Epoch 101 ==> 	accuracy: 0.9086, 	precision: 0.9997, 	recall: 0.8175, 	specificity: 0.9998, 	f1: 0.8994
Test Epoch 101: 100%|██████████| 1768/1768 [00:56<00:00, 31.06it/s, loss=0.107]
Test Epoch 101 ==> 	accuracy: 0.9637, 	precision: 0.9761, 	recall: 0.8434, 	specificity: 0.9947, 	f1: 0.9049
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 102: 100%|██████████| 6507/6507 [07:47<00:00, 13.92it/s, loss=0.0064]
Train Epoch 102 ==> 	accuracy: 0.9080, 	precision: 0.9997, 	recall: 0.8163, 	specificity: 0.9997, 	f1: 0.8987
Test Epoch 102: 100%|██████████| 1768/1768 [00:52<00:00, 33.84it/s, loss=3.73]
Test Epoch 102 ==> 	accuracy: 0.9633, 	precision: 0.9789, 	recall: 0.8387, 	specificity: 0.9954, 	f1: 0.9034
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 103: 100%|██████████| 6507/6507 [07:45<00:00, 13.97it/s, loss=0.158]
Train Epoch 103 ==> 	accuracy: 0.9069, 	precision: 0.9997, 	recall: 0.8141, 	specificity: 0.9998, 	f1: 0.8974
Test Epoch 103: 100%|██████████| 1768/1768 [00:50<00:00, 35.15it/s, loss=1.47]
Test Epoch 103 ==> 	accuracy: 0.9625, 	precision: 0.9821, 	recall: 0.8316, 	specificity: 0.9961, 	f1: 0.9006
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 104: 100%|██████████| 6507/6507 [07:54<00:00, 13.73it/s, loss=0.324]
Train Epoch 104 ==> 	accuracy: 0.9084, 	precision: 0.9997, 	recall: 0.8171, 	specificity: 0.9998, 	f1: 0.8992
Test Epoch 104: 100%|██████████| 1768/1768 [00:53<00:00, 32.99it/s, loss=0.272]
Test Epoch 104 ==> 	accuracy: 0.9634, 	precision: 0.9795, 	recall: 0.8385, 	specificity: 0.9955, 	f1: 0.9035
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 105: 100%|██████████| 6507/6507 [07:53<00:00, 13.73it/s, loss=0.0163]
Train Epoch 105 ==> 	accuracy: 0.9088, 	precision: 0.9997, 	recall: 0.8179, 	specificity: 0.9997, 	f1: 0.8997
Test Epoch 105: 100%|██████████| 1768/1768 [00:53<00:00, 32.98it/s, loss=0.0506]
Test Epoch 105 ==> 	accuracy: 0.9625, 	precision: 0.9822, 	recall: 0.8319, 	specificity: 0.9961, 	f1: 0.9008
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 106: 100%|██████████| 6507/6507 [07:49<00:00, 13.86it/s, loss=0.0193]
Train Epoch 106 ==> 	accuracy: 0.9085, 	precision: 0.9997, 	recall: 0.8172, 	specificity: 0.9998, 	f1: 0.8993
Test Epoch 106: 100%|██████████| 1768/1768 [00:50<00:00, 35.22it/s, loss=0.184]
Test Epoch 106 ==> 	accuracy: 0.9628, 	precision: 0.9816, 	recall: 0.8336, 	specificity: 0.9960, 	f1: 0.9016
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 107: 100%|██████████| 6507/6507 [07:48<00:00, 13.90it/s, loss=0.703]
Train Epoch 107 ==> 	accuracy: 0.9095, 	precision: 0.9997, 	recall: 0.8193, 	specificity: 0.9998, 	f1: 0.9005
Test Epoch 107: 100%|██████████| 1768/1768 [00:51<00:00, 34.34it/s, loss=0.197]
Test Epoch 107 ==> 	accuracy: 0.9636, 	precision: 0.9794, 	recall: 0.8400, 	specificity: 0.9954, 	f1: 0.9043
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 108: 100%|██████████| 6507/6507 [07:39<00:00, 14.15it/s, loss=0.0095]
Train Epoch 108 ==> 	accuracy: 0.9098, 	precision: 0.9997, 	recall: 0.8198, 	specificity: 0.9998, 	f1: 0.9009
Test Epoch 108: 100%|██████████| 1768/1768 [00:51<00:00, 34.59it/s, loss=1.45]
Test Epoch 108 ==> 	accuracy: 0.9629, 	precision: 0.9766, 	recall: 0.8388, 	specificity: 0.9948, 	f1: 0.9025
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 109: 100%|██████████| 6507/6507 [07:46<00:00, 13.94it/s, loss=0.0037]
Train Epoch 109 ==> 	accuracy: 0.9108, 	precision: 0.9997, 	recall: 0.8219, 	specificity: 0.9997, 	f1: 0.9021
Test Epoch 109: 100%|██████████| 1768/1768 [00:51<00:00, 34.00it/s, loss=0.14]
Test Epoch 109 ==> 	accuracy: 0.9637, 	precision: 0.9793, 	recall: 0.8401, 	specificity: 0.9954, 	f1: 0.9044
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 110: 100%|██████████| 6507/6507 [07:49<00:00, 13.87it/s, loss=0.303]
Train Epoch 110 ==> 	accuracy: 0.9094, 	precision: 0.9997, 	recall: 0.8190, 	specificity: 0.9998, 	f1: 0.9004
Test Epoch 110: 100%|██████████| 1768/1768 [00:50<00:00, 34.68it/s, loss=0.132]
Test Epoch 110 ==> 	accuracy: 0.9620, 	precision: 0.9839, 	recall: 0.8277, 	specificity: 0.9965, 	f1: 0.8991
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 111: 100%|██████████| 6507/6507 [07:51<00:00, 13.81it/s, loss=0.0079]
Train Epoch 111 ==> 	accuracy: 0.9104, 	precision: 0.9997, 	recall: 0.8211, 	specificity: 0.9997, 	f1: 0.9016
Test Epoch 111: 100%|██████████| 1768/1768 [00:51<00:00, 34.18it/s, loss=0.197]
Test Epoch 111 ==> 	accuracy: 0.9645, 	precision: 0.9795, 	recall: 0.8440, 	specificity: 0.9954, 	f1: 0.9067
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 112: 100%|██████████| 6507/6507 [07:44<00:00, 14.01it/s, loss=0.0036]
Train Epoch 112 ==> 	accuracy: 0.9092, 	precision: 0.9997, 	recall: 0.8187, 	specificity: 0.9998, 	f1: 0.9002
Test Epoch 112: 100%|██████████| 1768/1768 [00:51<00:00, 34.30it/s, loss=0.205]
Test Epoch 112 ==> 	accuracy: 0.9631, 	precision: 0.9788, 	recall: 0.8378, 	specificity: 0.9953, 	f1: 0.9028
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 113: 100%|██████████| 6507/6507 [07:48<00:00, 13.90it/s, loss=0.0094]
Train Epoch 113 ==> 	accuracy: 0.9105, 	precision: 0.9997, 	recall: 0.8213, 	specificity: 0.9998, 	f1: 0.9018
Test Epoch 113: 100%|██████████| 1768/1768 [00:51<00:00, 34.45it/s, loss=0.123]
Test Epoch 113 ==> 	accuracy: 0.9636, 	precision: 0.9773, 	recall: 0.8418, 	specificity: 0.9950, 	f1: 0.9045
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 114: 100%|██████████| 6507/6507 [07:39<00:00, 14.16it/s, loss=0.0024]
Train Epoch 114 ==> 	accuracy: 0.9115, 	precision: 0.9997, 	recall: 0.8233, 	specificity: 0.9998, 	f1: 0.9030
Test Epoch 114: 100%|██████████| 1768/1768 [00:56<00:00, 31.35it/s, loss=0.146]
Test Epoch 114 ==> 	accuracy: 0.9637, 	precision: 0.9778, 	recall: 0.8414, 	specificity: 0.9951, 	f1: 0.9045
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 115: 100%|██████████| 6507/6507 [07:43<00:00, 14.05it/s, loss=0.0337]
Train Epoch 115 ==> 	accuracy: 0.9117, 	precision: 0.9997, 	recall: 0.8236, 	specificity: 0.9998, 	f1: 0.9032
Test Epoch 115: 100%|██████████| 1768/1768 [00:53<00:00, 32.76it/s, loss=0.102]
Test Epoch 115 ==> 	accuracy: 0.9637, 	precision: 0.9819, 	recall: 0.8382, 	specificity: 0.9960, 	f1: 0.9044
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 116: 100%|██████████| 6507/6507 [07:47<00:00, 13.91it/s, loss=0.0147]
Train Epoch 116 ==> 	accuracy: 0.9093, 	precision: 0.9997, 	recall: 0.8188, 	specificity: 0.9998, 	f1: 0.9003
Test Epoch 116: 100%|██████████| 1768/1768 [00:54<00:00, 32.15it/s, loss=2.77]
Test Epoch 116 ==> 	accuracy: 0.9640, 	precision: 0.9813, 	recall: 0.8401, 	specificity: 0.9959, 	f1: 0.9052
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 117: 100%|██████████| 6507/6507 [07:46<00:00, 13.96it/s, loss=2.44]
Train Epoch 117 ==> 	accuracy: 0.9120, 	precision: 0.9997, 	recall: 0.8242, 	specificity: 0.9998, 	f1: 0.9035
Test Epoch 117: 100%|██████████| 1768/1768 [00:56<00:00, 31.48it/s, loss=2.12]
Test Epoch 117 ==> 	accuracy: 0.9642, 	precision: 0.9776, 	recall: 0.8445, 	specificity: 0.9950, 	f1: 0.9062
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 118: 100%|██████████| 6507/6507 [07:56<00:00, 13.65it/s, loss=0.0228]
Train Epoch 118 ==> 	accuracy: 0.9108, 	precision: 0.9997, 	recall: 0.8218, 	specificity: 0.9998, 	f1: 0.9021
Test Epoch 118: 100%|██████████| 1768/1768 [00:57<00:00, 30.73it/s, loss=0.295]
Test Epoch 118 ==> 	accuracy: 0.9651, 	precision: 0.9821, 	recall: 0.8448, 	specificity: 0.9960, 	f1: 0.9083
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 119: 100%|██████████| 6507/6507 [07:47<00:00, 13.92it/s, loss=0.138]
Train Epoch 119 ==> 	accuracy: 0.9105, 	precision: 0.9997, 	recall: 0.8213, 	specificity: 0.9998, 	f1: 0.9018
Test Epoch 119: 100%|██████████| 1768/1768 [00:54<00:00, 32.37it/s, loss=0.454]
Test Epoch 119 ==> 	accuracy: 0.9651, 	precision: 0.9823, 	recall: 0.8448, 	specificity: 0.9961, 	f1: 0.9084
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 120: 100%|██████████| 6507/6507 [07:50<00:00, 13.84it/s, loss=0.0192]
Train Epoch 120 ==> 	accuracy: 0.9097, 	precision: 0.9997, 	recall: 0.8195, 	specificity: 0.9998, 	f1: 0.9007
Test Epoch 120: 100%|██████████| 1768/1768 [00:54<00:00, 32.51it/s, loss=0.0653]
Test Epoch 120 ==> 	accuracy: 0.9639, 	precision: 0.9839, 	recall: 0.8372, 	specificity: 0.9965, 	f1: 0.9047
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 121: 100%|██████████| 6507/6507 [07:50<00:00, 13.83it/s, loss=0.125]
Train Epoch 121 ==> 	accuracy: 0.9121, 	precision: 0.9998, 	recall: 0.8244, 	specificity: 0.9998, 	f1: 0.9037
Test Epoch 121: 100%|██████████| 1768/1768 [00:50<00:00, 35.01it/s, loss=0.112]
Test Epoch 121 ==> 	accuracy: 0.9655, 	precision: 0.9819, 	recall: 0.8471, 	specificity: 0.9960, 	f1: 0.9095
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 122: 100%|██████████| 6507/6507 [07:45<00:00, 13.97it/s, loss=0.0055]
Train Epoch 122 ==> 	accuracy: 0.9133, 	precision: 0.9997, 	recall: 0.8268, 	specificity: 0.9998, 	f1: 0.9051
Test Epoch 122: 100%|██████████| 1768/1768 [00:57<00:00, 30.93it/s, loss=0.119]
Test Epoch 122 ==> 	accuracy: 0.9655, 	precision: 0.9803, 	recall: 0.8484, 	specificity: 0.9956, 	f1: 0.9096
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 123: 100%|██████████| 6507/6507 [07:44<00:00, 14.02it/s, loss=0.004]
Train Epoch 123 ==> 	accuracy: 0.9110, 	precision: 0.9997, 	recall: 0.8223, 	specificity: 0.9998, 	f1: 0.9024
Test Epoch 123: 100%|██████████| 1768/1768 [00:51<00:00, 34.40it/s, loss=0.139]
Test Epoch 123 ==> 	accuracy: 0.9654, 	precision: 0.9801, 	recall: 0.8481, 	specificity: 0.9956, 	f1: 0.9093
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 124: 100%|██████████| 6507/6507 [07:49<00:00, 13.85it/s, loss=0.028]
Train Epoch 124 ==> 	accuracy: 0.9142, 	precision: 0.9997, 	recall: 0.8286, 	specificity: 0.9998, 	f1: 0.9061
Test Epoch 124: 100%|██████████| 1768/1768 [00:51<00:00, 34.05it/s, loss=1.92]
Test Epoch 124 ==> 	accuracy: 0.9653, 	precision: 0.9789, 	recall: 0.8488, 	specificity: 0.9953, 	f1: 0.9092
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 125: 100%|██████████| 6507/6507 [07:42<00:00, 14.06it/s, loss=0.0126]
Train Epoch 125 ==> 	accuracy: 0.9133, 	precision: 0.9997, 	recall: 0.8269, 	specificity: 0.9998, 	f1: 0.9051
Test Epoch 125: 100%|██████████| 1768/1768 [00:52<00:00, 33.52it/s, loss=0.728]
Test Epoch 125 ==> 	accuracy: 0.9646, 	precision: 0.9808, 	recall: 0.8434, 	specificity: 0.9957, 	f1: 0.9069
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 126: 100%|██████████| 6507/6507 [07:36<00:00, 14.25it/s, loss=0.0545]
Train Epoch 126 ==> 	accuracy: 0.9132, 	precision: 0.9997, 	recall: 0.8267, 	specificity: 0.9998, 	f1: 0.9050
Test Epoch 126: 100%|██████████| 1768/1768 [00:53<00:00, 32.83it/s, loss=2.27]
Test Epoch 126 ==> 	accuracy: 0.9637, 	precision: 0.9813, 	recall: 0.8387, 	specificity: 0.9959, 	f1: 0.9044
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 127: 100%|██████████| 6507/6507 [07:48<00:00, 13.90it/s, loss=0.0449]
Train Epoch 127 ==> 	accuracy: 0.9135, 	precision: 0.9997, 	recall: 0.8271, 	specificity: 0.9998, 	f1: 0.9053
Test Epoch 127: 100%|██████████| 1768/1768 [00:52<00:00, 33.79it/s, loss=0.0412]
Test Epoch 127 ==> 	accuracy: 0.9648, 	precision: 0.9794, 	recall: 0.8459, 	specificity: 0.9954, 	f1: 0.9078
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 128: 100%|██████████| 6507/6507 [07:49<00:00, 13.86it/s, loss=0.0435]
Train Epoch 128 ==> 	accuracy: 0.9129, 	precision: 0.9997, 	recall: 0.8261, 	specificity: 0.9998, 	f1: 0.9047
Test Epoch 128: 100%|██████████| 1768/1768 [00:48<00:00, 36.31it/s, loss=2.85]
Test Epoch 128 ==> 	accuracy: 0.9651, 	precision: 0.9824, 	recall: 0.8447, 	specificity: 0.9961, 	f1: 0.9083
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 129: 100%|██████████| 6507/6507 [07:51<00:00, 13.81it/s, loss=0.0125]
Train Epoch 129 ==> 	accuracy: 0.9114, 	precision: 0.9998, 	recall: 0.8231, 	specificity: 0.9998, 	f1: 0.9029
Test Epoch 129: 100%|██████████| 1768/1768 [00:55<00:00, 31.66it/s, loss=0.147]
Test Epoch 129 ==> 	accuracy: 0.9650, 	precision: 0.9806, 	recall: 0.8456, 	specificity: 0.9957, 	f1: 0.9081
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 130: 100%|██████████| 6507/6507 [07:41<00:00, 14.10it/s, loss=0.0063]
Train Epoch 130 ==> 	accuracy: 0.9125, 	precision: 0.9997, 	recall: 0.8252, 	specificity: 0.9998, 	f1: 0.9041
Test Epoch 130: 100%|██████████| 1768/1768 [00:57<00:00, 30.70it/s, loss=0.513]
Test Epoch 130 ==> 	accuracy: 0.9649, 	precision: 0.9806, 	recall: 0.8453, 	specificity: 0.9957, 	f1: 0.9080
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 131: 100%|██████████| 6507/6507 [07:53<00:00, 13.73it/s, loss=0.0082]
Train Epoch 131 ==> 	accuracy: 0.9125, 	precision: 0.9997, 	recall: 0.8252, 	specificity: 0.9998, 	f1: 0.9041
Test Epoch 131: 100%|██████████| 1768/1768 [00:51<00:00, 34.51it/s, loss=0.374]
Test Epoch 131 ==> 	accuracy: 0.9651, 	precision: 0.9816, 	recall: 0.8455, 	specificity: 0.9959, 	f1: 0.9085
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 132: 100%|██████████| 6507/6507 [07:53<00:00, 13.73it/s, loss=0.0057]
Train Epoch 132 ==> 	accuracy: 0.9135, 	precision: 0.9997, 	recall: 0.8271, 	specificity: 0.9998, 	f1: 0.9053
Test Epoch 132: 100%|██████████| 1768/1768 [00:57<00:00, 30.54it/s, loss=0.226]
Test Epoch 132 ==> 	accuracy: 0.9651, 	precision: 0.9810, 	recall: 0.8457, 	specificity: 0.9958, 	f1: 0.9083
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 133: 100%|██████████| 6507/6507 [07:54<00:00, 13.71it/s, loss=0.0663]
Train Epoch 133 ==> 	accuracy: 0.9124, 	precision: 0.9997, 	recall: 0.8250, 	specificity: 0.9998, 	f1: 0.9040
Test Epoch 133: 100%|██████████| 1768/1768 [00:54<00:00, 32.63it/s, loss=0.0591]
Test Epoch 133 ==> 	accuracy: 0.9648, 	precision: 0.9820, 	recall: 0.8432, 	specificity: 0.9960, 	f1: 0.9073
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 134: 100%|██████████| 6507/6507 [07:52<00:00, 13.77it/s, loss=0.0117]
Train Epoch 134 ==> 	accuracy: 0.9143, 	precision: 0.9998, 	recall: 0.8289, 	specificity: 0.9998, 	f1: 0.9063
Test Epoch 134: 100%|██████████| 1768/1768 [00:53<00:00, 32.90it/s, loss=2.84]
Test Epoch 134 ==> 	accuracy: 0.9646, 	precision: 0.9805, 	recall: 0.8438, 	specificity: 0.9957, 	f1: 0.9070
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 135: 100%|██████████| 6507/6507 [08:01<00:00, 13.50it/s, loss=0.0019]
Train Epoch 135 ==> 	accuracy: 0.9145, 	precision: 0.9997, 	recall: 0.8292, 	specificity: 0.9998, 	f1: 0.9065
Test Epoch 135: 100%|██████████| 1768/1768 [00:55<00:00, 32.08it/s, loss=0.118]
Test Epoch 135 ==> 	accuracy: 0.9645, 	precision: 0.9808, 	recall: 0.8430, 	specificity: 0.9958, 	f1: 0.9067
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 136: 100%|██████████| 6507/6507 [07:55<00:00, 13.67it/s, loss=2.63]
Train Epoch 136 ==> 	accuracy: 0.9146, 	precision: 0.9997, 	recall: 0.8295, 	specificity: 0.9998, 	f1: 0.9067
Test Epoch 136: 100%|██████████| 1768/1768 [00:57<00:00, 30.86it/s, loss=0.356]
Test Epoch 136 ==> 	accuracy: 0.9647, 	precision: 0.9811, 	recall: 0.8436, 	specificity: 0.9958, 	f1: 0.9072
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 137: 100%|██████████| 6507/6507 [08:08<00:00, 13.31it/s, loss=1.43]
Train Epoch 137 ==> 	accuracy: 0.9149, 	precision: 0.9997, 	recall: 0.8301, 	specificity: 0.9998, 	f1: 0.9071
Test Epoch 137: 100%|██████████| 1768/1768 [00:52<00:00, 33.52it/s, loss=0.154]
Test Epoch 137 ==> 	accuracy: 0.9649, 	precision: 0.9806, 	recall: 0.8449, 	specificity: 0.9957, 	f1: 0.9077
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 138: 100%|██████████| 6507/6507 [07:52<00:00, 13.76it/s, loss=0.0107]
Train Epoch 138 ==> 	accuracy: 0.9145, 	precision: 0.9998, 	recall: 0.8291, 	specificity: 0.9998, 	f1: 0.9065
Test Epoch 138: 100%|██████████| 1768/1768 [00:53<00:00, 32.84it/s, loss=0.0571]
Test Epoch 138 ==> 	accuracy: 0.9649, 	precision: 0.9799, 	recall: 0.8460, 	specificity: 0.9955, 	f1: 0.9080
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 139: 100%|██████████| 6507/6507 [08:00<00:00, 13.53it/s, loss=0.775]
Train Epoch 139 ==> 	accuracy: 0.9155, 	precision: 0.9997, 	recall: 0.8313, 	specificity: 0.9998, 	f1: 0.9078
Test Epoch 139: 100%|██████████| 1768/1768 [00:55<00:00, 31.98it/s, loss=0.0787]
Test Epoch 139 ==> 	accuracy: 0.9649, 	precision: 0.9784, 	recall: 0.8473, 	specificity: 0.9952, 	f1: 0.9081
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 140: 100%|██████████| 6507/6507 [07:58<00:00, 13.61it/s, loss=0.0042]
Train Epoch 140 ==> 	accuracy: 0.9127, 	precision: 0.9997, 	recall: 0.8257, 	specificity: 0.9998, 	f1: 0.9044
Test Epoch 140: 100%|██████████| 1768/1768 [00:53<00:00, 33.20it/s, loss=0.085]
Test Epoch 140 ==> 	accuracy: 0.9654, 	precision: 0.9802, 	recall: 0.8482, 	specificity: 0.9956, 	f1: 0.9094
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 141: 100%|██████████| 6507/6507 [07:59<00:00, 13.57it/s, loss=0.483]
Train Epoch 141 ==> 	accuracy: 0.9153, 	precision: 0.9997, 	recall: 0.8308, 	specificity: 0.9998, 	f1: 0.9075
Test Epoch 141: 100%|██████████| 1768/1768 [00:51<00:00, 34.46it/s, loss=0.0583]
Test Epoch 141 ==> 	accuracy: 0.9659, 	precision: 0.9783, 	recall: 0.8520, 	specificity: 0.9951, 	f1: 0.9108
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 142: 100%|██████████| 6507/6507 [07:59<00:00, 13.56it/s, loss=0.0273]
Train Epoch 142 ==> 	accuracy: 0.9137, 	precision: 0.9998, 	recall: 0.8276, 	specificity: 0.9998, 	f1: 0.9056
Test Epoch 142: 100%|██████████| 1768/1768 [00:52<00:00, 33.85it/s, loss=0.0484]
Test Epoch 142 ==> 	accuracy: 0.9650, 	precision: 0.9804, 	recall: 0.8458, 	specificity: 0.9957, 	f1: 0.9081
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 143: 100%|██████████| 6507/6507 [07:54<00:00, 13.72it/s, loss=0.0012]
Train Epoch 143 ==> 	accuracy: 0.9153, 	precision: 0.9998, 	recall: 0.8307, 	specificity: 0.9998, 	f1: 0.9074
Test Epoch 143: 100%|██████████| 1768/1768 [00:52<00:00, 33.75it/s, loss=0.0573]
Test Epoch 143 ==> 	accuracy: 0.9659, 	precision: 0.9795, 	recall: 0.8513, 	specificity: 0.9954, 	f1: 0.9109
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 144: 100%|██████████| 6507/6507 [07:56<00:00, 13.66it/s, loss=0.406]
Train Epoch 144 ==> 	accuracy: 0.9140, 	precision: 0.9998, 	recall: 0.8283, 	specificity: 0.9998, 	f1: 0.9060
Test Epoch 144: 100%|██████████| 1768/1768 [00:54<00:00, 32.69it/s, loss=0.0355]
Test Epoch 144 ==> 	accuracy: 0.9658, 	precision: 0.9807, 	recall: 0.8498, 	specificity: 0.9957, 	f1: 0.9105
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 145: 100%|██████████| 6507/6507 [07:49<00:00, 13.85it/s, loss=0.0076]
Train Epoch 145 ==> 	accuracy: 0.9138, 	precision: 0.9997, 	recall: 0.8277, 	specificity: 0.9998, 	f1: 0.9056
Test Epoch 145: 100%|██████████| 1768/1768 [00:58<00:00, 30.26it/s, loss=3]
Test Epoch 145 ==> 	accuracy: 0.9654, 	precision: 0.9811, 	recall: 0.8472, 	specificity: 0.9958, 	f1: 0.9093
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 146: 100%|██████████| 6507/6507 [07:53<00:00, 13.75it/s, loss=0.0092]
Train Epoch 146 ==> 	accuracy: 0.9143, 	precision: 0.9997, 	recall: 0.8289, 	specificity: 0.9998, 	f1: 0.9063
Test Epoch 146: 100%|██████████| 1768/1768 [00:50<00:00, 34.80it/s, loss=3.53]
Test Epoch 146 ==> 	accuracy: 0.9657, 	precision: 0.9804, 	recall: 0.8491, 	specificity: 0.9956, 	f1: 0.9100
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 147: 100%|██████████| 6507/6507 [08:07<00:00, 13.36it/s, loss=0.0499]
Train Epoch 147 ==> 	accuracy: 0.9152, 	precision: 0.9997, 	recall: 0.8307, 	specificity: 0.9998, 	f1: 0.9074
Test Epoch 147: 100%|██████████| 1768/1768 [00:53<00:00, 33.34it/s, loss=0.0992]
Test Epoch 147 ==> 	accuracy: 0.9658, 	precision: 0.9796, 	recall: 0.8505, 	specificity: 0.9955, 	f1: 0.9105
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 148: 100%|██████████| 6507/6507 [08:02<00:00, 13.49it/s, loss=0.0771]
Train Epoch 148 ==> 	accuracy: 0.9172, 	precision: 0.9998, 	recall: 0.8346, 	specificity: 0.9998, 	f1: 0.9098
Test Epoch 148: 100%|██████████| 1768/1768 [00:59<00:00, 29.64it/s, loss=0.538]
Test Epoch 148 ==> 	accuracy: 0.9659, 	precision: 0.9793, 	recall: 0.8511, 	specificity: 0.9954, 	f1: 0.9107
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 149: 100%|██████████| 6507/6507 [08:10<00:00, 13.27it/s, loss=0.0082]
Train Epoch 149 ==> 	accuracy: 0.9151, 	precision: 0.9998, 	recall: 0.8303, 	specificity: 0.9998, 	f1: 0.9072
Test Epoch 149: 100%|██████████| 1768/1768 [00:52<00:00, 33.89it/s, loss=0.115]
Test Epoch 149 ==> 	accuracy: 0.9657, 	precision: 0.9782, 	recall: 0.8511, 	specificity: 0.9951, 	f1: 0.9102
Adjusting learning rate of group 0 to 5.8150e-06.

进程已结束，退出代码为 0

'''

'''
ab seq 
'../model_save_sigBlock4_focalWithMs_deformable_7mer_ab_seq_multConv'
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6507/6507 [06:09<00:00, 17.61it/s, loss=0.0532]
Train Epoch 0 ==> 	accuracy: 0.5518, 	precision: 0.9934, 	recall: 0.1044, 	specificity: 0.9993, 	f1: 0.1889
Test Epoch 0: 100%|██████████| 1768/1768 [00:43<00:00, 40.57it/s, loss=0.328]
Test Epoch 0 ==> 	accuracy: 0.8496, 	precision: 0.9811, 	recall: 0.2697, 	specificity: 0.9987, 	f1: 0.4231
Train Epoch 1: 100%|██████████| 6507/6507 [07:01<00:00, 15.43it/s, loss=0.102]
Train Epoch 1 ==> 	accuracy: 0.6313, 	precision: 0.9962, 	recall: 0.2637, 	specificity: 0.9990, 	f1: 0.4170
Test Epoch 1: 100%|██████████| 1768/1768 [00:45<00:00, 38.88it/s, loss=0.268]
Test Epoch 1 ==> 	accuracy: 0.8722, 	precision: 0.9902, 	recall: 0.3790, 	specificity: 0.9990, 	f1: 0.5482
Train Epoch 2: 100%|██████████| 6507/6507 [07:23<00:00, 14.67it/s, loss=0.0931]
Train Epoch 2 ==> 	accuracy: 0.6690, 	precision: 0.9969, 	recall: 0.3391, 	specificity: 0.9989, 	f1: 0.5060
Test Epoch 2: 100%|██████████| 1768/1768 [00:46<00:00, 38.10it/s, loss=0.581]
Test Epoch 2 ==> 	accuracy: 0.8702, 	precision: 0.9949, 	recall: 0.3672, 	specificity: 0.9995, 	f1: 0.5364
Train Epoch 3: 100%|██████████| 6507/6507 [07:28<00:00, 14.51it/s, loss=0.0401]
Train Epoch 3 ==> 	accuracy: 0.6843, 	precision: 0.9974, 	recall: 0.3696, 	specificity: 0.9990, 	f1: 0.5394
Test Epoch 3: 100%|██████████| 1768/1768 [00:47<00:00, 37.20it/s, loss=0.302]
Test Epoch 3 ==> 	accuracy: 0.8873, 	precision: 0.9915, 	recall: 0.4529, 	specificity: 0.9990, 	f1: 0.6218
Train Epoch 4: 100%|██████████| 6507/6507 [07:29<00:00, 14.48it/s, loss=0.0883]
Train Epoch 4 ==> 	accuracy: 0.7031, 	precision: 0.9976, 	recall: 0.4072, 	specificity: 0.9990, 	f1: 0.5784
Test Epoch 4: 100%|██████████| 1768/1768 [00:47<00:00, 37.25it/s, loss=0.345]
Test Epoch 4 ==> 	accuracy: 0.8941, 	precision: 0.9912, 	recall: 0.4866, 	specificity: 0.9989, 	f1: 0.6528
Train Epoch 5: 100%|██████████| 6507/6507 [07:16<00:00, 14.90it/s, loss=0.0356]
Train Epoch 5 ==> 	accuracy: 0.7097, 	precision: 0.9977, 	recall: 0.4203, 	specificity: 0.9990, 	f1: 0.5915
Test Epoch 5: 100%|██████████| 1768/1768 [00:46<00:00, 38.26it/s, loss=0.199]
Test Epoch 5 ==> 	accuracy: 0.8875, 	precision: 0.9930, 	recall: 0.4532, 	specificity: 0.9992, 	f1: 0.6223
Train Epoch 6: 100%|██████████| 6507/6507 [07:15<00:00, 14.92it/s, loss=0.0581]
Train Epoch 6 ==> 	accuracy: 0.7142, 	precision: 0.9979, 	recall: 0.4292, 	specificity: 0.9991, 	f1: 0.6003
Test Epoch 6: 100%|██████████| 1768/1768 [00:45<00:00, 38.85it/s, loss=0.235]
Test Epoch 6 ==> 	accuracy: 0.8958, 	precision: 0.9934, 	recall: 0.4939, 	specificity: 0.9992, 	f1: 0.6598
Train Epoch 7: 100%|██████████| 6507/6507 [07:18<00:00, 14.85it/s, loss=0.0704]
Train Epoch 7 ==> 	accuracy: 0.7264, 	precision: 0.9981, 	recall: 0.4537, 	specificity: 0.9991, 	f1: 0.6238
Test Epoch 7: 100%|██████████| 1768/1768 [00:48<00:00, 36.43it/s, loss=0.181]
Test Epoch 7 ==> 	accuracy: 0.9023, 	precision: 0.9906, 	recall: 0.5273, 	specificity: 0.9987, 	f1: 0.6883
Train Epoch 8: 100%|██████████| 6507/6507 [07:22<00:00, 14.71it/s, loss=0.0968]
Train Epoch 8 ==> 	accuracy: 0.7238, 	precision: 0.9981, 	recall: 0.4484, 	specificity: 0.9992, 	f1: 0.6188
Test Epoch 8: 100%|██████████| 1768/1768 [00:46<00:00, 37.86it/s, loss=0.233]
Test Epoch 8 ==> 	accuracy: 0.8929, 	precision: 0.9944, 	recall: 0.4792, 	specificity: 0.9993, 	f1: 0.6467
Train Epoch 9: 100%|██████████| 6507/6507 [07:25<00:00, 14.61it/s, loss=0.117]
Train Epoch 9 ==> 	accuracy: 0.7326, 	precision: 0.9982, 	recall: 0.4659, 	specificity: 0.9992, 	f1: 0.6353
Test Epoch 9: 100%|██████████| 1768/1768 [00:48<00:00, 36.46it/s, loss=0.253]
Test Epoch 9 ==> 	accuracy: 0.8990, 	precision: 0.9928, 	recall: 0.5098, 	specificity: 0.9990, 	f1: 0.6737
Train Epoch 10: 100%|██████████| 6507/6507 [07:24<00:00, 14.63it/s, loss=0.0805]
Train Epoch 10 ==> 	accuracy: 0.7359, 	precision: 0.9983, 	recall: 0.4727, 	specificity: 0.9992, 	f1: 0.6416
Test Epoch 10: 100%|██████████| 1768/1768 [00:43<00:00, 40.35it/s, loss=0.308]
Test Epoch 10 ==> 	accuracy: 0.8898, 	precision: 0.9941, 	recall: 0.4640, 	specificity: 0.9993, 	f1: 0.6327
Train Epoch 11: 100%|██████████| 6507/6507 [07:25<00:00, 14.62it/s, loss=0.076]
Train Epoch 11 ==> 	accuracy: 0.7407, 	precision: 0.9984, 	recall: 0.4821, 	specificity: 0.9992, 	f1: 0.6502
Test Epoch 11: 100%|██████████| 1768/1768 [00:47<00:00, 37.38it/s, loss=0.244]
Test Epoch 11 ==> 	accuracy: 0.9053, 	precision: 0.9918, 	recall: 0.5413, 	specificity: 0.9989, 	f1: 0.7003
Train Epoch 12: 100%|██████████| 6507/6507 [07:09<00:00, 15.13it/s, loss=0.028]
Train Epoch 12 ==> 	accuracy: 0.7398, 	precision: 0.9985, 	recall: 0.4803, 	specificity: 0.9993, 	f1: 0.6486
Test Epoch 12: 100%|██████████| 1768/1768 [00:45<00:00, 38.77it/s, loss=0.227]
Test Epoch 12 ==> 	accuracy: 0.9008, 	precision: 0.9941, 	recall: 0.5181, 	specificity: 0.9992, 	f1: 0.6812
Train Epoch 13: 100%|██████████| 6507/6507 [07:06<00:00, 15.26it/s, loss=0.0829]
Train Epoch 13 ==> 	accuracy: 0.7445, 	precision: 0.9986, 	recall: 0.4897, 	specificity: 0.9993, 	f1: 0.6572
Test Epoch 13: 100%|██████████| 1768/1768 [00:47<00:00, 37.46it/s, loss=0.24]
Test Epoch 13 ==> 	accuracy: 0.9064, 	precision: 0.9920, 	recall: 0.5470, 	specificity: 0.9989, 	f1: 0.7052
Train Epoch 14: 100%|██████████| 6507/6507 [07:21<00:00, 14.73it/s, loss=0.0333]
Train Epoch 14 ==> 	accuracy: 0.7509, 	precision: 0.9986, 	recall: 0.5025, 	specificity: 0.9993, 	f1: 0.6686
Test Epoch 14: 100%|██████████| 1768/1768 [00:45<00:00, 38.76it/s, loss=0.228]
Test Epoch 14 ==> 	accuracy: 0.9064, 	precision: 0.9917, 	recall: 0.5470, 	specificity: 0.9988, 	f1: 0.7051
Train Epoch 15: 100%|██████████| 6507/6507 [07:19<00:00, 14.81it/s, loss=0.0712]
Train Epoch 15 ==> 	accuracy: 0.7494, 	precision: 0.9987, 	recall: 0.4995, 	specificity: 0.9993, 	f1: 0.6659
Test Epoch 15: 100%|██████████| 1768/1768 [00:48<00:00, 36.59it/s, loss=0.171]
Test Epoch 15 ==> 	accuracy: 0.9135, 	precision: 0.9884, 	recall: 0.5840, 	specificity: 0.9982, 	f1: 0.7342
Train Epoch 16: 100%|██████████| 6507/6507 [07:18<00:00, 14.85it/s, loss=0.0914]
Train Epoch 16 ==> 	accuracy: 0.7497, 	precision: 0.9987, 	recall: 0.5001, 	specificity: 0.9993, 	f1: 0.6665
Test Epoch 16: 100%|██████████| 1768/1768 [00:48<00:00, 36.59it/s, loss=0.36]
Test Epoch 16 ==> 	accuracy: 0.9085, 	precision: 0.9922, 	recall: 0.5571, 	specificity: 0.9989, 	f1: 0.7136
Train Epoch 17: 100%|██████████| 6507/6507 [07:16<00:00, 14.90it/s, loss=0.0925]
Train Epoch 17 ==> 	accuracy: 0.7543, 	precision: 0.9987, 	recall: 0.5093, 	specificity: 0.9993, 	f1: 0.6746
Test Epoch 17: 100%|██████████| 1768/1768 [00:48<00:00, 36.61it/s, loss=0.147]
Test Epoch 17 ==> 	accuracy: 0.9104, 	precision: 0.9903, 	recall: 0.5674, 	specificity: 0.9986, 	f1: 0.7215
Train Epoch 18: 100%|██████████| 6507/6507 [07:10<00:00, 15.10it/s, loss=0.077]
Train Epoch 18 ==> 	accuracy: 0.7601, 	precision: 0.9988, 	recall: 0.5209, 	specificity: 0.9994, 	f1: 0.6847
Test Epoch 18: 100%|██████████| 1768/1768 [00:45<00:00, 38.52it/s, loss=0.663]
Test Epoch 18 ==> 	accuracy: 0.9168, 	precision: 0.9877, 	recall: 0.6008, 	specificity: 0.9981, 	f1: 0.7471
Train Epoch 19: 100%|██████████| 6507/6507 [07:23<00:00, 14.66it/s, loss=0.0515]
Train Epoch 19 ==> 	accuracy: 0.7596, 	precision: 0.9988, 	recall: 0.5199, 	specificity: 0.9994, 	f1: 0.6839
Test Epoch 19: 100%|██████████| 1768/1768 [00:47<00:00, 37.31it/s, loss=0.173]
Test Epoch 19 ==> 	accuracy: 0.9134, 	precision: 0.9892, 	recall: 0.5829, 	specificity: 0.9984, 	f1: 0.7335
Train Epoch 20: 100%|██████████| 6507/6507 [07:23<00:00, 14.67it/s, loss=0.0742]
Train Epoch 20 ==> 	accuracy: 0.7591, 	precision: 0.9988, 	recall: 0.5188, 	specificity: 0.9994, 	f1: 0.6829
Test Epoch 20: 100%|██████████| 1768/1768 [00:47<00:00, 37.39it/s, loss=0.209]
Test Epoch 20 ==> 	accuracy: 0.9124, 	precision: 0.9885, 	recall: 0.5784, 	specificity: 0.9983, 	f1: 0.7298
Train Epoch 21: 100%|██████████| 6507/6507 [07:19<00:00, 14.81it/s, loss=0.0779]
Train Epoch 21 ==> 	accuracy: 0.7672, 	precision: 0.9988, 	recall: 0.5350, 	specificity: 0.9994, 	f1: 0.6967
Test Epoch 21: 100%|██████████| 1768/1768 [00:46<00:00, 38.08it/s, loss=0.209]
Test Epoch 21 ==> 	accuracy: 0.9104, 	precision: 0.9924, 	recall: 0.5663, 	specificity: 0.9989, 	f1: 0.7211
Train Epoch 22: 100%|██████████| 6507/6507 [07:16<00:00, 14.90it/s, loss=0.0347]
Train Epoch 22 ==> 	accuracy: 0.7667, 	precision: 0.9989, 	recall: 0.5339, 	specificity: 0.9994, 	f1: 0.6959
Test Epoch 22: 100%|██████████| 1768/1768 [00:45<00:00, 39.19it/s, loss=0.376]
Test Epoch 22 ==> 	accuracy: 0.9094, 	precision: 0.9919, 	recall: 0.5617, 	specificity: 0.9988, 	f1: 0.7172
Train Epoch 23: 100%|██████████| 6507/6507 [07:12<00:00, 15.05it/s, loss=0.0907]
Train Epoch 23 ==> 	accuracy: 0.7664, 	precision: 0.9990, 	recall: 0.5334, 	specificity: 0.9995, 	f1: 0.6955
Test Epoch 23: 100%|██████████| 1768/1768 [00:48<00:00, 36.72it/s, loss=0.166]
Test Epoch 23 ==> 	accuracy: 0.9154, 	precision: 0.9902, 	recall: 0.5925, 	specificity: 0.9985, 	f1: 0.7414
Train Epoch 24: 100%|██████████| 6507/6507 [07:13<00:00, 15.02it/s, loss=0.0967]
Train Epoch 24 ==> 	accuracy: 0.7701, 	precision: 0.9989, 	recall: 0.5408, 	specificity: 0.9994, 	f1: 0.7017
Test Epoch 24: 100%|██████████| 1768/1768 [00:46<00:00, 38.15it/s, loss=0.299]
Test Epoch 24 ==> 	accuracy: 0.9063, 	precision: 0.9932, 	recall: 0.5455, 	specificity: 0.9990, 	f1: 0.7042
Train Epoch 25: 100%|██████████| 6507/6507 [07:12<00:00, 15.06it/s, loss=0.0851]
Train Epoch 25 ==> 	accuracy: 0.7706, 	precision: 0.9990, 	recall: 0.5417, 	specificity: 0.9994, 	f1: 0.7025
Test Epoch 25: 100%|██████████| 1768/1768 [00:46<00:00, 38.30it/s, loss=1.21]
Test Epoch 25 ==> 	accuracy: 0.9170, 	precision: 0.9905, 	recall: 0.6002, 	specificity: 0.9985, 	f1: 0.7475
Train Epoch 26: 100%|██████████| 6507/6507 [07:15<00:00, 14.95it/s, loss=0.0796]
Train Epoch 26 ==> 	accuracy: 0.7726, 	precision: 0.9990, 	recall: 0.5458, 	specificity: 0.9994, 	f1: 0.7059
Test Epoch 26: 100%|██████████| 1768/1768 [00:46<00:00, 37.92it/s, loss=0.241]
Test Epoch 26 ==> 	accuracy: 0.9162, 	precision: 0.9898, 	recall: 0.5964, 	specificity: 0.9984, 	f1: 0.7443
Train Epoch 27: 100%|██████████| 6507/6507 [07:19<00:00, 14.80it/s, loss=0.0768]
Train Epoch 27 ==> 	accuracy: 0.7745, 	precision: 0.9991, 	recall: 0.5496, 	specificity: 0.9995, 	f1: 0.7091
Test Epoch 27: 100%|██████████| 1768/1768 [00:46<00:00, 38.29it/s, loss=0.139]
Test Epoch 27 ==> 	accuracy: 0.9166, 	precision: 0.9898, 	recall: 0.5984, 	specificity: 0.9984, 	f1: 0.7458
Train Epoch 28: 100%|██████████| 6507/6507 [07:17<00:00, 14.87it/s, loss=0.0713]
Train Epoch 28 ==> 	accuracy: 0.7754, 	precision: 0.9990, 	recall: 0.5514, 	specificity: 0.9995, 	f1: 0.7106
Test Epoch 28: 100%|██████████| 1768/1768 [00:47<00:00, 37.22it/s, loss=0.102]
Test Epoch 28 ==> 	accuracy: 0.9157, 	precision: 0.9894, 	recall: 0.5943, 	specificity: 0.9984, 	f1: 0.7426
Train Epoch 29: 100%|██████████| 6507/6507 [07:19<00:00, 14.80it/s, loss=0.0715]
Train Epoch 29 ==> 	accuracy: 0.7769, 	precision: 0.9991, 	recall: 0.5543, 	specificity: 0.9995, 	f1: 0.7130
Test Epoch 29: 100%|██████████| 1768/1768 [00:49<00:00, 35.51it/s, loss=0.255]
Test Epoch 29 ==> 	accuracy: 0.9176, 	precision: 0.9898, 	recall: 0.6036, 	specificity: 0.9984, 	f1: 0.7499
Train Epoch 30: 100%|██████████| 6507/6507 [07:14<00:00, 14.98it/s, loss=0.0742]
Train Epoch 30 ==> 	accuracy: 0.7757, 	precision: 0.9990, 	recall: 0.5520, 	specificity: 0.9995, 	f1: 0.7111
Test Epoch 30: 100%|██████████| 1768/1768 [00:46<00:00, 37.72it/s, loss=0.265]
Test Epoch 30 ==> 	accuracy: 0.9168, 	precision: 0.9884, 	recall: 0.6003, 	specificity: 0.9982, 	f1: 0.7470
Train Epoch 31: 100%|██████████| 6507/6507 [07:21<00:00, 14.73it/s, loss=0.0266]
Train Epoch 31 ==> 	accuracy: 0.7823, 	precision: 0.9991, 	recall: 0.5652, 	specificity: 0.9995, 	f1: 0.7220
Test Epoch 31: 100%|██████████| 1768/1768 [00:45<00:00, 38.90it/s, loss=0.17]
Test Epoch 31 ==> 	accuracy: 0.9151, 	precision: 0.9904, 	recall: 0.5908, 	specificity: 0.9985, 	f1: 0.7401
Train Epoch 32: 100%|██████████| 6507/6507 [07:19<00:00, 14.81it/s, loss=0.027]
Train Epoch 32 ==> 	accuracy: 0.7784, 	precision: 0.9991, 	recall: 0.5573, 	specificity: 0.9995, 	f1: 0.7155
Test Epoch 32: 100%|██████████| 1768/1768 [00:48<00:00, 36.39it/s, loss=0.201]
Test Epoch 32 ==> 	accuracy: 0.9124, 	precision: 0.9922, 	recall: 0.5762, 	specificity: 0.9988, 	f1: 0.7290
Train Epoch 33: 100%|██████████| 6507/6507 [07:20<00:00, 14.77it/s, loss=0.0159]
Train Epoch 33 ==> 	accuracy: 0.7809, 	precision: 0.9990, 	recall: 0.5623, 	specificity: 0.9995, 	f1: 0.7196
Test Epoch 33: 100%|██████████| 1768/1768 [00:46<00:00, 37.70it/s, loss=0.936]
Test Epoch 33 ==> 	accuracy: 0.9197, 	precision: 0.9891, 	recall: 0.6141, 	specificity: 0.9983, 	f1: 0.7578
Train Epoch 34: 100%|██████████| 6507/6507 [07:26<00:00, 14.57it/s, loss=0.0436]
Train Epoch 34 ==> 	accuracy: 0.7822, 	precision: 0.9991, 	recall: 0.5650, 	specificity: 0.9995, 	f1: 0.7218
Test Epoch 34: 100%|██████████| 1768/1768 [00:49<00:00, 35.70it/s, loss=0.181]
Test Epoch 34 ==> 	accuracy: 0.9192, 	precision: 0.9894, 	recall: 0.6115, 	specificity: 0.9983, 	f1: 0.7559
Train Epoch 35: 100%|██████████| 6507/6507 [07:30<00:00, 14.46it/s, loss=0.0277]
Train Epoch 35 ==> 	accuracy: 0.7844, 	precision: 0.9991, 	recall: 0.5694, 	specificity: 0.9995, 	f1: 0.7254
Test Epoch 35: 100%|██████████| 1768/1768 [00:47<00:00, 36.99it/s, loss=0.271]
Test Epoch 35 ==> 	accuracy: 0.9221, 	precision: 0.9872, 	recall: 0.6275, 	specificity: 0.9979, 	f1: 0.7673
Train Epoch 36: 100%|██████████| 6507/6507 [07:30<00:00, 14.45it/s, loss=0.0006]
Train Epoch 36 ==> 	accuracy: 0.7856, 	precision: 0.9991, 	recall: 0.5716, 	specificity: 0.9995, 	f1: 0.7272
Test Epoch 36: 100%|██████████| 1768/1768 [00:46<00:00, 38.11it/s, loss=0.261]
Test Epoch 36 ==> 	accuracy: 0.9212, 	precision: 0.9881, 	recall: 0.6223, 	specificity: 0.9981, 	f1: 0.7637
Train Epoch 37: 100%|██████████| 6507/6507 [07:27<00:00, 14.55it/s, loss=0.0338]
Train Epoch 37 ==> 	accuracy: 0.7860, 	precision: 0.9991, 	recall: 0.5724, 	specificity: 0.9995, 	f1: 0.7279
Test Epoch 37: 100%|██████████| 1768/1768 [00:47<00:00, 37.37it/s, loss=0.317]
Test Epoch 37 ==> 	accuracy: 0.9218, 	precision: 0.9881, 	recall: 0.6253, 	specificity: 0.9981, 	f1: 0.7659
Train Epoch 38: 100%|██████████| 6507/6507 [07:24<00:00, 14.63it/s, loss=0.0259]
Train Epoch 38 ==> 	accuracy: 0.7880, 	precision: 0.9991, 	recall: 0.5766, 	specificity: 0.9995, 	f1: 0.7312
Test Epoch 38: 100%|██████████| 1768/1768 [00:48<00:00, 36.49it/s, loss=0.177]
Test Epoch 38 ==> 	accuracy: 0.9210, 	precision: 0.9888, 	recall: 0.6209, 	specificity: 0.9982, 	f1: 0.7628
Train Epoch 39: 100%|██████████| 6507/6507 [07:30<00:00, 14.45it/s, loss=0.0803]
Train Epoch 39 ==> 	accuracy: 0.7901, 	precision: 0.9992, 	recall: 0.5806, 	specificity: 0.9995, 	f1: 0.7345
Test Epoch 39: 100%|██████████| 1768/1768 [00:48<00:00, 36.68it/s, loss=0.223]
Test Epoch 39 ==> 	accuracy: 0.9156, 	precision: 0.9912, 	recall: 0.5925, 	specificity: 0.9986, 	f1: 0.7417
Train Epoch 40: 100%|██████████| 6507/6507 [07:04<00:00, 15.34it/s, loss=0.0057]
Train Epoch 40 ==> 	accuracy: 0.7902, 	precision: 0.9992, 	recall: 0.5808, 	specificity: 0.9995, 	f1: 0.7346
Test Epoch 40: 100%|██████████| 1768/1768 [00:47<00:00, 36.99it/s, loss=0.235]
Test Epoch 40 ==> 	accuracy: 0.9235, 	precision: 0.9846, 	recall: 0.6359, 	specificity: 0.9974, 	f1: 0.7728
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 41: 100%|██████████| 6507/6507 [07:03<00:00, 15.38it/s, loss=0.0093]
Train Epoch 41 ==> 	accuracy: 0.7877, 	precision: 0.9992, 	recall: 0.5758, 	specificity: 0.9995, 	f1: 0.7306
Test Epoch 41: 100%|██████████| 1768/1768 [00:48<00:00, 36.66it/s, loss=0.183]
Test Epoch 41 ==> 	accuracy: 0.9216, 	precision: 0.9893, 	recall: 0.6235, 	specificity: 0.9983, 	f1: 0.7649
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 42: 100%|██████████| 6507/6507 [07:05<00:00, 15.30it/s, loss=0.0079]
Train Epoch 42 ==> 	accuracy: 0.7903, 	precision: 0.9992, 	recall: 0.5811, 	specificity: 0.9995, 	f1: 0.7348
Test Epoch 42: 100%|██████████| 1768/1768 [00:48<00:00, 36.43it/s, loss=0.154]
Test Epoch 42 ==> 	accuracy: 0.9165, 	precision: 0.9915, 	recall: 0.5971, 	specificity: 0.9987, 	f1: 0.7454
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 43: 100%|██████████| 6507/6507 [07:03<00:00, 15.37it/s, loss=0.0237]
Train Epoch 43 ==> 	accuracy: 0.7926, 	precision: 0.9992, 	recall: 0.5857, 	specificity: 0.9996, 	f1: 0.7385
Test Epoch 43: 100%|██████████| 1768/1768 [00:46<00:00, 37.86it/s, loss=0.208]
Test Epoch 43 ==> 	accuracy: 0.9257, 	precision: 0.9846, 	recall: 0.6467, 	specificity: 0.9974, 	f1: 0.7807
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 44: 100%|██████████| 6507/6507 [07:03<00:00, 15.37it/s, loss=0.0486]
Train Epoch 44 ==> 	accuracy: 0.7961, 	precision: 0.9993, 	recall: 0.5926, 	specificity: 0.9996, 	f1: 0.7440
Test Epoch 44: 100%|██████████| 1768/1768 [00:49<00:00, 35.87it/s, loss=1.01]
Test Epoch 44 ==> 	accuracy: 0.9278, 	precision: 0.9813, 	recall: 0.6599, 	specificity: 0.9968, 	f1: 0.7891
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 45: 100%|██████████| 6507/6507 [07:05<00:00, 15.29it/s, loss=0.0135]
Train Epoch 45 ==> 	accuracy: 0.7914, 	precision: 0.9992, 	recall: 0.5834, 	specificity: 0.9995, 	f1: 0.7366
Test Epoch 45: 100%|██████████| 1768/1768 [00:46<00:00, 37.79it/s, loss=0.671]
Test Epoch 45 ==> 	accuracy: 0.9273, 	precision: 0.9843, 	recall: 0.6549, 	specificity: 0.9973, 	f1: 0.7865
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 46: 100%|██████████| 6507/6507 [06:58<00:00, 15.56it/s, loss=0.0042]
Train Epoch 46 ==> 	accuracy: 0.7973, 	precision: 0.9993, 	recall: 0.5950, 	specificity: 0.9996, 	f1: 0.7459
Test Epoch 46: 100%|██████████| 1768/1768 [00:47<00:00, 37.53it/s, loss=0.127]
Test Epoch 46 ==> 	accuracy: 0.9198, 	precision: 0.9894, 	recall: 0.6143, 	specificity: 0.9983, 	f1: 0.7580
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 47: 100%|██████████| 6507/6507 [07:02<00:00, 15.41it/s, loss=0.0116]
Train Epoch 47 ==> 	accuracy: 0.7946, 	precision: 0.9993, 	recall: 0.5896, 	specificity: 0.9996, 	f1: 0.7416
Test Epoch 47: 100%|██████████| 1768/1768 [00:48<00:00, 36.46it/s, loss=0.232]
Test Epoch 47 ==> 	accuracy: 0.9243, 	precision: 0.9877, 	recall: 0.6378, 	specificity: 0.9980, 	f1: 0.7751
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 48: 100%|██████████| 6507/6507 [07:01<00:00, 15.42it/s, loss=0.0047]
Train Epoch 48 ==> 	accuracy: 0.7992, 	precision: 0.9993, 	recall: 0.5989, 	specificity: 0.9996, 	f1: 0.7489
Test Epoch 48: 100%|██████████| 1768/1768 [00:47<00:00, 37.59it/s, loss=0.468]
Test Epoch 48 ==> 	accuracy: 0.9248, 	precision: 0.9868, 	recall: 0.6410, 	specificity: 0.9978, 	f1: 0.7772
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 49: 100%|██████████| 6507/6507 [07:05<00:00, 15.29it/s, loss=0.0121]
Train Epoch 49 ==> 	accuracy: 0.7982, 	precision: 0.9993, 	recall: 0.5967, 	specificity: 0.9996, 	f1: 0.7472
Test Epoch 49: 100%|██████████| 1768/1768 [00:48<00:00, 36.75it/s, loss=0.361]
Test Epoch 49 ==> 	accuracy: 0.9275, 	precision: 0.9852, 	recall: 0.6554, 	specificity: 0.9975, 	f1: 0.7871
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 50: 100%|██████████| 6507/6507 [06:56<00:00, 15.61it/s, loss=0.0153]
Train Epoch 50 ==> 	accuracy: 0.8040, 	precision: 0.9993, 	recall: 0.6084, 	specificity: 0.9996, 	f1: 0.7563
Test Epoch 50: 100%|██████████| 1768/1768 [00:47<00:00, 37.31it/s, loss=0.386]
Test Epoch 50 ==> 	accuracy: 0.9216, 	precision: 0.9864, 	recall: 0.6251, 	specificity: 0.9978, 	f1: 0.7652
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 51: 100%|██████████| 6507/6507 [06:57<00:00, 15.60it/s, loss=0.0259]
Train Epoch 51 ==> 	accuracy: 0.8011, 	precision: 0.9993, 	recall: 0.6026, 	specificity: 0.9996, 	f1: 0.7518
Test Epoch 51: 100%|██████████| 1768/1768 [00:49<00:00, 35.98it/s, loss=0.653]
Test Epoch 51 ==> 	accuracy: 0.9250, 	precision: 0.9873, 	recall: 0.6414, 	specificity: 0.9979, 	f1: 0.7776
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 52: 100%|██████████| 6507/6507 [07:01<00:00, 15.42it/s, loss=0.0209]
Train Epoch 52 ==> 	accuracy: 0.8057, 	precision: 0.9994, 	recall: 0.6118, 	specificity: 0.9996, 	f1: 0.7589
Test Epoch 52: 100%|██████████| 1768/1768 [00:49<00:00, 35.85it/s, loss=0.166]
Test Epoch 52 ==> 	accuracy: 0.9236, 	precision: 0.9880, 	recall: 0.6342, 	specificity: 0.9980, 	f1: 0.7725
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 53: 100%|██████████| 6507/6507 [06:56<00:00, 15.64it/s, loss=0.0554]
Train Epoch 53 ==> 	accuracy: 0.8052, 	precision: 0.9993, 	recall: 0.6108, 	specificity: 0.9996, 	f1: 0.7582
Test Epoch 53: 100%|██████████| 1768/1768 [00:48<00:00, 36.11it/s, loss=0.142]
Test Epoch 53 ==> 	accuracy: 0.9244, 	precision: 0.9878, 	recall: 0.6382, 	specificity: 0.9980, 	f1: 0.7754
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 54: 100%|██████████| 6507/6507 [06:55<00:00, 15.66it/s, loss=0.0013]
Train Epoch 54 ==> 	accuracy: 0.8066, 	precision: 0.9993, 	recall: 0.6137, 	specificity: 0.9996, 	f1: 0.7604
Test Epoch 54: 100%|██████████| 1768/1768 [00:49<00:00, 35.65it/s, loss=0.223]
Test Epoch 54 ==> 	accuracy: 0.9254, 	precision: 0.9864, 	recall: 0.6441, 	specificity: 0.9977, 	f1: 0.7793
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 55: 100%|██████████| 6507/6507 [07:06<00:00, 15.26it/s, loss=0.0117]
Train Epoch 55 ==> 	accuracy: 0.8089, 	precision: 0.9993, 	recall: 0.6183, 	specificity: 0.9996, 	f1: 0.7639
Test Epoch 55: 100%|██████████| 1768/1768 [00:49<00:00, 35.72it/s, loss=0.212]
Test Epoch 55 ==> 	accuracy: 0.9255, 	precision: 0.9868, 	recall: 0.6443, 	specificity: 0.9978, 	f1: 0.7796
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 56: 100%|██████████| 6507/6507 [07:03<00:00, 15.35it/s, loss=0.0184]
Train Epoch 56 ==> 	accuracy: 0.8079, 	precision: 0.9994, 	recall: 0.6161, 	specificity: 0.9996, 	f1: 0.7623
Test Epoch 56: 100%|██████████| 1768/1768 [00:48<00:00, 36.41it/s, loss=0.132]
Test Epoch 56 ==> 	accuracy: 0.9232, 	precision: 0.9887, 	recall: 0.6317, 	specificity: 0.9981, 	f1: 0.7709
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 57: 100%|██████████| 6507/6507 [07:06<00:00, 15.26it/s, loss=0.0145]
Train Epoch 57 ==> 	accuracy: 0.8120, 	precision: 0.9994, 	recall: 0.6244, 	specificity: 0.9996, 	f1: 0.7686
Test Epoch 57: 100%|██████████| 1768/1768 [00:47<00:00, 37.10it/s, loss=2.42]
Test Epoch 57 ==> 	accuracy: 0.9271, 	precision: 0.9859, 	recall: 0.6530, 	specificity: 0.9976, 	f1: 0.7856
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 58: 100%|██████████| 6507/6507 [07:03<00:00, 15.35it/s, loss=0.0124]
Train Epoch 58 ==> 	accuracy: 0.8084, 	precision: 0.9994, 	recall: 0.6171, 	specificity: 0.9996, 	f1: 0.7631
Test Epoch 58: 100%|██████████| 1768/1768 [00:48<00:00, 36.18it/s, loss=0.491]
Test Epoch 58 ==> 	accuracy: 0.9309, 	precision: 0.9820, 	recall: 0.6747, 	specificity: 0.9968, 	f1: 0.7998
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 59: 100%|██████████| 6507/6507 [06:59<00:00, 15.51it/s, loss=0.0086]
Train Epoch 59 ==> 	accuracy: 0.8125, 	precision: 0.9994, 	recall: 0.6254, 	specificity: 0.9996, 	f1: 0.7694
Test Epoch 59: 100%|██████████| 1768/1768 [00:47<00:00, 37.11it/s, loss=0.334]
Test Epoch 59 ==> 	accuracy: 0.9300, 	precision: 0.9835, 	recall: 0.6692, 	specificity: 0.9971, 	f1: 0.7964
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 60: 100%|██████████| 6507/6507 [07:10<00:00, 15.11it/s, loss=0.0072]
Train Epoch 60 ==> 	accuracy: 0.8136, 	precision: 0.9995, 	recall: 0.6274, 	specificity: 0.9997, 	f1: 0.7709
Test Epoch 60: 100%|██████████| 1768/1768 [00:46<00:00, 37.75it/s, loss=0.25]
Test Epoch 60 ==> 	accuracy: 0.9303, 	precision: 0.9820, 	recall: 0.6714, 	specificity: 0.9968, 	f1: 0.7975
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 61: 100%|██████████| 6507/6507 [07:03<00:00, 15.35it/s, loss=0.0011]
Train Epoch 61 ==> 	accuracy: 0.8166, 	precision: 0.9994, 	recall: 0.6336, 	specificity: 0.9996, 	f1: 0.7755
Test Epoch 61: 100%|██████████| 1768/1768 [00:48<00:00, 36.13it/s, loss=0.315]
Test Epoch 61 ==> 	accuracy: 0.9315, 	precision: 0.9824, 	recall: 0.6773, 	specificity: 0.9969, 	f1: 0.8018
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 62: 100%|██████████| 6507/6507 [06:59<00:00, 15.53it/s, loss=0.0054]
Train Epoch 62 ==> 	accuracy: 0.8123, 	precision: 0.9994, 	recall: 0.6250, 	specificity: 0.9997, 	f1: 0.7691
Test Epoch 62: 100%|██████████| 1768/1768 [00:47<00:00, 37.00it/s, loss=0.217]
Test Epoch 62 ==> 	accuracy: 0.9330, 	precision: 0.9795, 	recall: 0.6868, 	specificity: 0.9963, 	f1: 0.8075
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 63: 100%|██████████| 6507/6507 [07:09<00:00, 15.13it/s, loss=0.0123]
Train Epoch 63 ==> 	accuracy: 0.8157, 	precision: 0.9994, 	recall: 0.6318, 	specificity: 0.9996, 	f1: 0.7742
Test Epoch 63: 100%|██████████| 1768/1768 [00:48<00:00, 36.50it/s, loss=0.162]
Test Epoch 63 ==> 	accuracy: 0.9287, 	precision: 0.9851, 	recall: 0.6612, 	specificity: 0.9974, 	f1: 0.7913
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 64: 100%|██████████| 6507/6507 [07:00<00:00, 15.47it/s, loss=0.0009]
Train Epoch 64 ==> 	accuracy: 0.8160, 	precision: 0.9994, 	recall: 0.6323, 	specificity: 0.9996, 	f1: 0.7745
Test Epoch 64: 100%|██████████| 1768/1768 [00:47<00:00, 37.48it/s, loss=0.2]
Test Epoch 64 ==> 	accuracy: 0.9322, 	precision: 0.9810, 	recall: 0.6817, 	specificity: 0.9966, 	f1: 0.8044
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 65: 100%|██████████| 6507/6507 [07:01<00:00, 15.44it/s, loss=0.0092]
Train Epoch 65 ==> 	accuracy: 0.8188, 	precision: 0.9994, 	recall: 0.6379, 	specificity: 0.9996, 	f1: 0.7788
Test Epoch 65: 100%|██████████| 1768/1768 [00:47<00:00, 37.42it/s, loss=0.138]
Test Epoch 65 ==> 	accuracy: 0.9296, 	precision: 0.9845, 	recall: 0.6664, 	specificity: 0.9973, 	f1: 0.7948
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 66: 100%|██████████| 6507/6507 [06:57<00:00, 15.57it/s, loss=0.0004]
Train Epoch 66 ==> 	accuracy: 0.8166, 	precision: 0.9995, 	recall: 0.6335, 	specificity: 0.9997, 	f1: 0.7755
Test Epoch 66: 100%|██████████| 1768/1768 [00:47<00:00, 37.12it/s, loss=0.2]
Test Epoch 66 ==> 	accuracy: 0.9296, 	precision: 0.9851, 	recall: 0.6658, 	specificity: 0.9974, 	f1: 0.7946
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 67: 100%|██████████| 6507/6507 [06:59<00:00, 15.52it/s, loss=0.0133]
Train Epoch 67 ==> 	accuracy: 0.8219, 	precision: 0.9995, 	recall: 0.6442, 	specificity: 0.9997, 	f1: 0.7834
Test Epoch 67: 100%|██████████| 1768/1768 [00:45<00:00, 38.62it/s, loss=0.336]
Test Epoch 67 ==> 	accuracy: 0.9295, 	precision: 0.9833, 	recall: 0.6665, 	specificity: 0.9971, 	f1: 0.7945
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 68: 100%|██████████| 6507/6507 [06:59<00:00, 15.50it/s, loss=0.011]
Train Epoch 68 ==> 	accuracy: 0.8215, 	precision: 0.9995, 	recall: 0.6434, 	specificity: 0.9996, 	f1: 0.7829
Test Epoch 68: 100%|██████████| 1768/1768 [00:48<00:00, 36.57it/s, loss=0.145]
Test Epoch 68 ==> 	accuracy: 0.9342, 	precision: 0.9790, 	recall: 0.6934, 	specificity: 0.9962, 	f1: 0.8118
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 69: 100%|██████████| 6507/6507 [07:09<00:00, 15.16it/s, loss=0.0228]
Train Epoch 69 ==> 	accuracy: 0.8217, 	precision: 0.9994, 	recall: 0.6438, 	specificity: 0.9996, 	f1: 0.7831
Test Epoch 69: 100%|██████████| 1768/1768 [00:47<00:00, 36.91it/s, loss=0.293]
Test Epoch 69 ==> 	accuracy: 0.9306, 	precision: 0.9840, 	recall: 0.6718, 	specificity: 0.9972, 	f1: 0.7985
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 70: 100%|██████████| 6507/6507 [07:06<00:00, 15.26it/s, loss=0.0054]
Train Epoch 70 ==> 	accuracy: 0.8219, 	precision: 0.9995, 	recall: 0.6441, 	specificity: 0.9997, 	f1: 0.7834
Test Epoch 70: 100%|██████████| 1768/1768 [00:48<00:00, 36.45it/s, loss=1.28]
Test Epoch 70 ==> 	accuracy: 0.9332, 	precision: 0.9806, 	recall: 0.6869, 	specificity: 0.9965, 	f1: 0.8078
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 71: 100%|██████████| 6507/6507 [07:05<00:00, 15.28it/s, loss=0.0237]
Train Epoch 71 ==> 	accuracy: 0.8234, 	precision: 0.9994, 	recall: 0.6472, 	specificity: 0.9996, 	f1: 0.7856
Test Epoch 71: 100%|██████████| 1768/1768 [00:50<00:00, 35.11it/s, loss=1.04]
Test Epoch 71 ==> 	accuracy: 0.9358, 	precision: 0.9769, 	recall: 0.7029, 	specificity: 0.9957, 	f1: 0.8175
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 72: 100%|██████████| 6507/6507 [07:07<00:00, 15.24it/s, loss=0.0237]
Train Epoch 72 ==> 	accuracy: 0.8263, 	precision: 0.9995, 	recall: 0.6528, 	specificity: 0.9997, 	f1: 0.7898
Test Epoch 72: 100%|██████████| 1768/1768 [00:49<00:00, 35.68it/s, loss=0.198]
Test Epoch 72 ==> 	accuracy: 0.9324, 	precision: 0.9816, 	recall: 0.6824, 	specificity: 0.9967, 	f1: 0.8051
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 73: 100%|██████████| 6507/6507 [07:07<00:00, 15.23it/s, loss=0.0143]
Train Epoch 73 ==> 	accuracy: 0.8234, 	precision: 0.9995, 	recall: 0.6471, 	specificity: 0.9997, 	f1: 0.7856
Test Epoch 73: 100%|██████████| 1768/1768 [00:47<00:00, 37.47it/s, loss=0.632]
Test Epoch 73 ==> 	accuracy: 0.9332, 	precision: 0.9820, 	recall: 0.6861, 	specificity: 0.9968, 	f1: 0.8079
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 74: 100%|██████████| 6507/6507 [07:06<00:00, 15.25it/s, loss=0.0024]
Train Epoch 74 ==> 	accuracy: 0.8261, 	precision: 0.9994, 	recall: 0.6526, 	specificity: 0.9996, 	f1: 0.7896
Test Epoch 74: 100%|██████████| 1768/1768 [00:47<00:00, 37.36it/s, loss=0.164]
Test Epoch 74 ==> 	accuracy: 0.9330, 	precision: 0.9814, 	recall: 0.6854, 	specificity: 0.9967, 	f1: 0.8071
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 75: 100%|██████████| 6507/6507 [07:04<00:00, 15.34it/s, loss=0.0028]
Train Epoch 75 ==> 	accuracy: 0.8247, 	precision: 0.9995, 	recall: 0.6496, 	specificity: 0.9997, 	f1: 0.7875
Test Epoch 75: 100%|██████████| 1768/1768 [00:48<00:00, 36.78it/s, loss=0.173]
Test Epoch 75 ==> 	accuracy: 0.9324, 	precision: 0.9827, 	recall: 0.6816, 	specificity: 0.9969, 	f1: 0.8049
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 76: 100%|██████████| 6507/6507 [07:06<00:00, 15.27it/s, loss=0.0163]
Train Epoch 76 ==> 	accuracy: 0.8300, 	precision: 0.9995, 	recall: 0.6603, 	specificity: 0.9997, 	f1: 0.7952
Test Epoch 76: 100%|██████████| 1768/1768 [00:47<00:00, 37.58it/s, loss=2.65]
Test Epoch 76 ==> 	accuracy: 0.9367, 	precision: 0.9757, 	recall: 0.7080, 	specificity: 0.9955, 	f1: 0.8206
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 77: 100%|██████████| 6507/6507 [07:05<00:00, 15.29it/s, loss=0.0082]
Train Epoch 77 ==> 	accuracy: 0.8279, 	precision: 0.9995, 	recall: 0.6561, 	specificity: 0.9997, 	f1: 0.7922
Test Epoch 77: 100%|██████████| 1768/1768 [00:48<00:00, 36.22it/s, loss=0.162]
Test Epoch 77 ==> 	accuracy: 0.9342, 	precision: 0.9801, 	recall: 0.6925, 	specificity: 0.9964, 	f1: 0.8116
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 78: 100%|██████████| 6507/6507 [06:56<00:00, 15.63it/s, loss=0.001]
Train Epoch 78 ==> 	accuracy: 0.8293, 	precision: 0.9995, 	recall: 0.6590, 	specificity: 0.9997, 	f1: 0.7943
Test Epoch 78: 100%|██████████| 1768/1768 [00:49<00:00, 36.07it/s, loss=0.279]
Test Epoch 78 ==> 	accuracy: 0.9316, 	precision: 0.9845, 	recall: 0.6761, 	specificity: 0.9973, 	f1: 0.8017
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 79: 100%|██████████| 6507/6507 [07:05<00:00, 15.29it/s, loss=0.014]
Train Epoch 79 ==> 	accuracy: 0.8278, 	precision: 0.9995, 	recall: 0.6559, 	specificity: 0.9997, 	f1: 0.7920
Test Epoch 79: 100%|██████████| 1768/1768 [00:48<00:00, 36.75it/s, loss=0.189]
Test Epoch 79 ==> 	accuracy: 0.9338, 	precision: 0.9819, 	recall: 0.6892, 	specificity: 0.9967, 	f1: 0.8100
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 80: 100%|██████████| 6507/6507 [07:04<00:00, 15.31it/s, loss=0.0022]
Train Epoch 80 ==> 	accuracy: 0.8280, 	precision: 0.9995, 	recall: 0.6564, 	specificity: 0.9997, 	f1: 0.7924
Test Epoch 80: 100%|██████████| 1768/1768 [00:48<00:00, 36.39it/s, loss=0.226]
Test Epoch 80 ==> 	accuracy: 0.9369, 	precision: 0.9771, 	recall: 0.7080, 	specificity: 0.9957, 	f1: 0.8210
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 81: 100%|██████████| 6507/6507 [07:07<00:00, 15.21it/s, loss=0.0116]
Train Epoch 81 ==> 	accuracy: 0.8302, 	precision: 0.9995, 	recall: 0.6608, 	specificity: 0.9997, 	f1: 0.7956
Test Epoch 81: 100%|██████████| 1768/1768 [00:46<00:00, 37.93it/s, loss=0.131]
Test Epoch 81 ==> 	accuracy: 0.9365, 	precision: 0.9774, 	recall: 0.7058, 	specificity: 0.9958, 	f1: 0.8197
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 82: 100%|██████████| 6507/6507 [07:07<00:00, 15.22it/s, loss=0]
Train Epoch 82 ==> 	accuracy: 0.8305, 	precision: 0.9995, 	recall: 0.6614, 	specificity: 0.9997, 	f1: 0.7961
Test Epoch 82: 100%|██████████| 1768/1768 [00:47<00:00, 36.86it/s, loss=0.178]
Test Epoch 82 ==> 	accuracy: 0.9361, 	precision: 0.9789, 	recall: 0.7026, 	specificity: 0.9961, 	f1: 0.8181
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 83: 100%|██████████| 6507/6507 [07:18<00:00, 14.83it/s, loss=0.0231]
Train Epoch 83 ==> 	accuracy: 0.8296, 	precision: 0.9995, 	recall: 0.6594, 	specificity: 0.9997, 	f1: 0.7946
Test Epoch 83: 100%|██████████| 1768/1768 [00:49<00:00, 35.47it/s, loss=0.221]
Test Epoch 83 ==> 	accuracy: 0.9362, 	precision: 0.9780, 	recall: 0.7039, 	specificity: 0.9959, 	f1: 0.8186
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 84: 100%|██████████| 6507/6507 [07:14<00:00, 14.96it/s, loss=0.0077]
Train Epoch 84 ==> 	accuracy: 0.8320, 	precision: 0.9995, 	recall: 0.6642, 	specificity: 0.9997, 	f1: 0.7981
Test Epoch 84: 100%|██████████| 1768/1768 [00:49<00:00, 36.01it/s, loss=0.134]
Test Epoch 84 ==> 	accuracy: 0.9327, 	precision: 0.9827, 	recall: 0.6828, 	specificity: 0.9969, 	f1: 0.8057
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 85: 100%|██████████| 6507/6507 [07:08<00:00, 15.18it/s, loss=0.0039]
Train Epoch 85 ==> 	accuracy: 0.8333, 	precision: 0.9995, 	recall: 0.6670, 	specificity: 0.9997, 	f1: 0.8001
Test Epoch 85: 100%|██████████| 1768/1768 [00:47<00:00, 37.28it/s, loss=0.62]
Test Epoch 85 ==> 	accuracy: 0.9369, 	precision: 0.9779, 	recall: 0.7076, 	specificity: 0.9959, 	f1: 0.8211
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 86: 100%|██████████| 6507/6507 [07:03<00:00, 15.37it/s, loss=0.0188]
Train Epoch 86 ==> 	accuracy: 0.8348, 	precision: 0.9995, 	recall: 0.6700, 	specificity: 0.9997, 	f1: 0.8022
Test Epoch 86: 100%|██████████| 1768/1768 [00:45<00:00, 38.71it/s, loss=0.993]
Test Epoch 86 ==> 	accuracy: 0.9362, 	precision: 0.9774, 	recall: 0.7045, 	specificity: 0.9958, 	f1: 0.8188
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 87: 100%|██████████| 6507/6507 [07:03<00:00, 15.38it/s, loss=0.0036]
Train Epoch 87 ==> 	accuracy: 0.8335, 	precision: 0.9995, 	recall: 0.6673, 	specificity: 0.9997, 	f1: 0.8003
Test Epoch 87: 100%|██████████| 1768/1768 [00:48<00:00, 36.63it/s, loss=0.205]
Test Epoch 87 ==> 	accuracy: 0.9326, 	precision: 0.9839, 	recall: 0.6817, 	specificity: 0.9971, 	f1: 0.8054
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 88: 100%|██████████| 6507/6507 [06:57<00:00, 15.59it/s, loss=0.0024]
Train Epoch 88 ==> 	accuracy: 0.8360, 	precision: 0.9996, 	recall: 0.6723, 	specificity: 0.9997, 	f1: 0.8039
Test Epoch 88: 100%|██████████| 1768/1768 [00:49<00:00, 36.00it/s, loss=0.162]
Test Epoch 88 ==> 	accuracy: 0.9364, 	precision: 0.9781, 	recall: 0.7049, 	specificity: 0.9959, 	f1: 0.8193
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 89: 100%|██████████| 6507/6507 [07:01<00:00, 15.44it/s, loss=0.0122]
Train Epoch 89 ==> 	accuracy: 0.8361, 	precision: 0.9996, 	recall: 0.6726, 	specificity: 0.9997, 	f1: 0.8041
Test Epoch 89: 100%|██████████| 1768/1768 [00:48<00:00, 36.37it/s, loss=0.177]
Test Epoch 89 ==> 	accuracy: 0.9387, 	precision: 0.9737, 	recall: 0.7195, 	specificity: 0.9950, 	f1: 0.8275
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 90: 100%|██████████| 6507/6507 [07:04<00:00, 15.34it/s, loss=0.0142]
Train Epoch 90 ==> 	accuracy: 0.8346, 	precision: 0.9996, 	recall: 0.6695, 	specificity: 0.9997, 	f1: 0.8019
Test Epoch 90: 100%|██████████| 1768/1768 [00:46<00:00, 37.96it/s, loss=0.135]
Test Epoch 90 ==> 	accuracy: 0.9386, 	precision: 0.9751, 	recall: 0.7182, 	specificity: 0.9953, 	f1: 0.8272
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 91: 100%|██████████| 6507/6507 [07:01<00:00, 15.44it/s, loss=0.0074]
Train Epoch 91 ==> 	accuracy: 0.8371, 	precision: 0.9995, 	recall: 0.6744, 	specificity: 0.9997, 	f1: 0.8054
Test Epoch 91: 100%|██████████| 1768/1768 [00:48<00:00, 36.30it/s, loss=0.522]
Test Epoch 91 ==> 	accuracy: 0.9353, 	precision: 0.9793, 	recall: 0.6984, 	specificity: 0.9962, 	f1: 0.8153
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 92: 100%|██████████| 6507/6507 [07:40<00:00, 14.14it/s, loss=0.0076]
Train Epoch 92 ==> 	accuracy: 0.8344, 	precision: 0.9996, 	recall: 0.6690, 	specificity: 0.9997, 	f1: 0.8016
Test Epoch 92: 100%|██████████| 1768/1768 [00:47<00:00, 37.22it/s, loss=0.169]
Test Epoch 92 ==> 	accuracy: 0.9360, 	precision: 0.9792, 	recall: 0.7019, 	specificity: 0.9962, 	f1: 0.8177
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 93: 100%|██████████| 6507/6507 [07:00<00:00, 15.49it/s, loss=0.0045]
Train Epoch 93 ==> 	accuracy: 0.8391, 	precision: 0.9996, 	recall: 0.6786, 	specificity: 0.9997, 	f1: 0.8084
Test Epoch 93: 100%|██████████| 1768/1768 [00:48<00:00, 36.53it/s, loss=1.38]
Test Epoch 93 ==> 	accuracy: 0.9345, 	precision: 0.9821, 	recall: 0.6924, 	specificity: 0.9968, 	f1: 0.8122
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 94: 100%|██████████| 6507/6507 [06:58<00:00, 15.55it/s, loss=0.0056]
Train Epoch 94 ==> 	accuracy: 0.8370, 	precision: 0.9996, 	recall: 0.6744, 	specificity: 0.9997, 	f1: 0.8054
Test Epoch 94: 100%|██████████| 1768/1768 [00:49<00:00, 36.05it/s, loss=0.159]
Test Epoch 94 ==> 	accuracy: 0.9364, 	precision: 0.9794, 	recall: 0.7037, 	specificity: 0.9962, 	f1: 0.8190
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 95: 100%|██████████| 6507/6507 [06:57<00:00, 15.58it/s, loss=0.0084]
Train Epoch 95 ==> 	accuracy: 0.8359, 	precision: 0.9995, 	recall: 0.6720, 	specificity: 0.9997, 	f1: 0.8037
Test Epoch 95: 100%|██████████| 1768/1768 [00:50<00:00, 35.03it/s, loss=1.98]
Test Epoch 95 ==> 	accuracy: 0.9367, 	precision: 0.9791, 	recall: 0.7058, 	specificity: 0.9961, 	f1: 0.8203
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 96: 100%|██████████| 6507/6507 [07:01<00:00, 15.42it/s, loss=0.0022]
Train Epoch 96 ==> 	accuracy: 0.8365, 	precision: 0.9995, 	recall: 0.6733, 	specificity: 0.9997, 	f1: 0.8046
Test Epoch 96: 100%|██████████| 1768/1768 [00:49<00:00, 35.78it/s, loss=2.25]
Test Epoch 96 ==> 	accuracy: 0.9349, 	precision: 0.9816, 	recall: 0.6948, 	specificity: 0.9967, 	f1: 0.8137
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 97: 100%|██████████| 6507/6507 [07:09<00:00, 15.16it/s, loss=0.0046]
Train Epoch 97 ==> 	accuracy: 0.8368, 	precision: 0.9996, 	recall: 0.6738, 	specificity: 0.9997, 	f1: 0.8050
Test Epoch 97: 100%|██████████| 1768/1768 [00:49<00:00, 35.64it/s, loss=0.722]
Test Epoch 97 ==> 	accuracy: 0.9371, 	precision: 0.9772, 	recall: 0.7090, 	specificity: 0.9957, 	f1: 0.8218
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 98: 100%|██████████| 6507/6507 [07:06<00:00, 15.27it/s, loss=0.0065]
Train Epoch 98 ==> 	accuracy: 0.8386, 	precision: 0.9996, 	recall: 0.6774, 	specificity: 0.9997, 	f1: 0.8076
Test Epoch 98: 100%|██████████| 1768/1768 [00:48<00:00, 36.14it/s, loss=0.245]
Test Epoch 98 ==> 	accuracy: 0.9373, 	precision: 0.9766, 	recall: 0.7105, 	specificity: 0.9956, 	f1: 0.8225
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 99: 100%|██████████| 6507/6507 [07:16<00:00, 14.91it/s, loss=0.0095]
Train Epoch 99 ==> 	accuracy: 0.8380, 	precision: 0.9996, 	recall: 0.6764, 	specificity: 0.9997, 	f1: 0.8068
Test Epoch 99: 100%|██████████| 1768/1768 [00:49<00:00, 36.05it/s, loss=0.158]
Test Epoch 99 ==> 	accuracy: 0.9366, 	precision: 0.9780, 	recall: 0.7061, 	specificity: 0.9959, 	f1: 0.8201
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 100: 100%|██████████| 6507/6507 [07:12<00:00, 15.03it/s, loss=0.0084]
Train Epoch 100 ==> 	accuracy: 0.8406, 	precision: 0.9996, 	recall: 0.6814, 	specificity: 0.9997, 	f1: 0.8104
Test Epoch 100: 100%|██████████| 1768/1768 [00:49<00:00, 35.69it/s, loss=0.147]
Test Epoch 100 ==> 	accuracy: 0.9389, 	precision: 0.9741, 	recall: 0.7204, 	specificity: 0.9951, 	f1: 0.8283
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 101: 100%|██████████| 6507/6507 [07:12<00:00, 15.05it/s, loss=0.0143]
Train Epoch 101 ==> 	accuracy: 0.8414, 	precision: 0.9996, 	recall: 0.6831, 	specificity: 0.9997, 	f1: 0.8116
Test Epoch 101: 100%|██████████| 1768/1768 [00:47<00:00, 37.00it/s, loss=1]
Test Epoch 101 ==> 	accuracy: 0.9380, 	precision: 0.9764, 	recall: 0.7140, 	specificity: 0.9956, 	f1: 0.8248
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 102: 100%|██████████| 6507/6507 [06:58<00:00, 15.54it/s, loss=0.0024]
Train Epoch 102 ==> 	accuracy: 0.8416, 	precision: 0.9996, 	recall: 0.6835, 	specificity: 0.9997, 	f1: 0.8119
Test Epoch 102: 100%|██████████| 1768/1768 [00:48<00:00, 36.52it/s, loss=0.696]
Test Epoch 102 ==> 	accuracy: 0.9377, 	precision: 0.9771, 	recall: 0.7123, 	specificity: 0.9957, 	f1: 0.8240
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 103: 100%|██████████| 6507/6507 [07:08<00:00, 15.20it/s, loss=0]
Train Epoch 103 ==> 	accuracy: 0.8406, 	precision: 0.9996, 	recall: 0.6815, 	specificity: 0.9997, 	f1: 0.8105
Test Epoch 103: 100%|██████████| 1768/1768 [00:48<00:00, 36.83it/s, loss=0.19]
Test Epoch 103 ==> 	accuracy: 0.9379, 	precision: 0.9765, 	recall: 0.7135, 	specificity: 0.9956, 	f1: 0.8246
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 104: 100%|██████████| 6507/6507 [07:06<00:00, 15.25it/s, loss=0.0073]
Train Epoch 104 ==> 	accuracy: 0.8411, 	precision: 0.9996, 	recall: 0.6824, 	specificity: 0.9997, 	f1: 0.8111
Test Epoch 104: 100%|██████████| 1768/1768 [00:46<00:00, 38.29it/s, loss=0.155]
Test Epoch 104 ==> 	accuracy: 0.9368, 	precision: 0.9791, 	recall: 0.7060, 	specificity: 0.9961, 	f1: 0.8204
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 105: 100%|██████████| 6507/6507 [07:04<00:00, 15.33it/s, loss=0.0051]
Train Epoch 105 ==> 	accuracy: 0.8419, 	precision: 0.9996, 	recall: 0.6841, 	specificity: 0.9997, 	f1: 0.8123
Test Epoch 105: 100%|██████████| 1768/1768 [00:47<00:00, 36.97it/s, loss=0.134]
Test Epoch 105 ==> 	accuracy: 0.9373, 	precision: 0.9783, 	recall: 0.7093, 	specificity: 0.9959, 	f1: 0.8223
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 106: 100%|██████████| 6507/6507 [07:05<00:00, 15.31it/s, loss=0.0085]
Train Epoch 106 ==> 	accuracy: 0.8415, 	precision: 0.9996, 	recall: 0.6833, 	specificity: 0.9997, 	f1: 0.8117
Test Epoch 106: 100%|██████████| 1768/1768 [00:47<00:00, 37.14it/s, loss=0.29]
Test Epoch 106 ==> 	accuracy: 0.9374, 	precision: 0.9781, 	recall: 0.7100, 	specificity: 0.9959, 	f1: 0.8228
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 107: 100%|██████████| 6507/6507 [07:01<00:00, 15.45it/s, loss=0.0078]
Train Epoch 107 ==> 	accuracy: 0.8420, 	precision: 0.9996, 	recall: 0.6843, 	specificity: 0.9997, 	f1: 0.8124
Test Epoch 107: 100%|██████████| 1768/1768 [00:47<00:00, 37.58it/s, loss=0.128]
Test Epoch 107 ==> 	accuracy: 0.9384, 	precision: 0.9766, 	recall: 0.7158, 	specificity: 0.9956, 	f1: 0.8261
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 108: 100%|██████████| 6507/6507 [06:53<00:00, 15.73it/s, loss=0.0558]
Train Epoch 108 ==> 	accuracy: 0.8416, 	precision: 0.9996, 	recall: 0.6835, 	specificity: 0.9997, 	f1: 0.8119
Test Epoch 108: 100%|██████████| 1768/1768 [00:48<00:00, 36.25it/s, loss=0.166]
Test Epoch 108 ==> 	accuracy: 0.9386, 	precision: 0.9761, 	recall: 0.7172, 	specificity: 0.9955, 	f1: 0.8269
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 109: 100%|██████████| 6507/6507 [07:03<00:00, 15.36it/s, loss=0.0202]
Train Epoch 109 ==> 	accuracy: 0.8428, 	precision: 0.9996, 	recall: 0.6860, 	specificity: 0.9997, 	f1: 0.8136
Test Epoch 109: 100%|██████████| 1768/1768 [00:47<00:00, 37.25it/s, loss=0.13]
Test Epoch 109 ==> 	accuracy: 0.9392, 	precision: 0.9737, 	recall: 0.7223, 	specificity: 0.9950, 	f1: 0.8293
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 110: 100%|██████████| 6507/6507 [07:00<00:00, 15.48it/s, loss=0.0104]
Train Epoch 110 ==> 	accuracy: 0.8438, 	precision: 0.9996, 	recall: 0.6880, 	specificity: 0.9997, 	f1: 0.8150
Test Epoch 110: 100%|██████████| 1768/1768 [00:48<00:00, 36.83it/s, loss=0.541]
Test Epoch 110 ==> 	accuracy: 0.9396, 	precision: 0.9741, 	recall: 0.7239, 	specificity: 0.9951, 	f1: 0.8306
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 111: 100%|██████████| 6507/6507 [07:04<00:00, 15.32it/s, loss=0.0015]
Train Epoch 111 ==> 	accuracy: 0.8440, 	precision: 0.9996, 	recall: 0.6884, 	specificity: 0.9997, 	f1: 0.8153
Test Epoch 111: 100%|██████████| 1768/1768 [00:48<00:00, 36.74it/s, loss=0.252]
Test Epoch 111 ==> 	accuracy: 0.9361, 	precision: 0.9796, 	recall: 0.7020, 	specificity: 0.9962, 	f1: 0.8179
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 112: 100%|██████████| 6507/6507 [07:04<00:00, 15.35it/s, loss=0.0128]
Train Epoch 112 ==> 	accuracy: 0.8414, 	precision: 0.9996, 	recall: 0.6831, 	specificity: 0.9997, 	f1: 0.8116
Test Epoch 112: 100%|██████████| 1768/1768 [00:47<00:00, 37.01it/s, loss=0.444]
Test Epoch 112 ==> 	accuracy: 0.9367, 	precision: 0.9788, 	recall: 0.7058, 	specificity: 0.9961, 	f1: 0.8201
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 113: 100%|██████████| 6507/6507 [07:04<00:00, 15.33it/s, loss=0.0004]
Train Epoch 113 ==> 	accuracy: 0.8429, 	precision: 0.9996, 	recall: 0.6861, 	specificity: 0.9997, 	f1: 0.8137
Test Epoch 113: 100%|██████████| 1768/1768 [00:47<00:00, 37.11it/s, loss=0.14]
Test Epoch 113 ==> 	accuracy: 0.9382, 	precision: 0.9769, 	recall: 0.7150, 	specificity: 0.9956, 	f1: 0.8257
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 114: 100%|██████████| 6507/6507 [07:01<00:00, 15.42it/s, loss=0.0167]
Train Epoch 114 ==> 	accuracy: 0.8435, 	precision: 0.9996, 	recall: 0.6874, 	specificity: 0.9997, 	f1: 0.8146
Test Epoch 114: 100%|██████████| 1768/1768 [00:47<00:00, 37.17it/s, loss=0.139]
Test Epoch 114 ==> 	accuracy: 0.9392, 	precision: 0.9739, 	recall: 0.7219, 	specificity: 0.9950, 	f1: 0.8292
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 115: 100%|██████████| 6507/6507 [07:05<00:00, 15.30it/s, loss=0.0018]
Train Epoch 115 ==> 	accuracy: 0.8453, 	precision: 0.9996, 	recall: 0.6908, 	specificity: 0.9997, 	f1: 0.8170
Test Epoch 115: 100%|██████████| 1768/1768 [00:49<00:00, 35.91it/s, loss=0.948]
Test Epoch 115 ==> 	accuracy: 0.9382, 	precision: 0.9762, 	recall: 0.7155, 	specificity: 0.9955, 	f1: 0.8258
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 116: 100%|██████████| 6507/6507 [07:02<00:00, 15.40it/s, loss=0.0044]
Train Epoch 116 ==> 	accuracy: 0.8432, 	precision: 0.9996, 	recall: 0.6866, 	specificity: 0.9997, 	f1: 0.8141
Test Epoch 116: 100%|██████████| 1768/1768 [00:48<00:00, 36.73it/s, loss=0.201]
Test Epoch 116 ==> 	accuracy: 0.9384, 	precision: 0.9769, 	recall: 0.7158, 	specificity: 0.9957, 	f1: 0.8262
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 117: 100%|██████████| 6507/6507 [07:06<00:00, 15.27it/s, loss=0.0187]
Train Epoch 117 ==> 	accuracy: 0.8457, 	precision: 0.9996, 	recall: 0.6917, 	specificity: 0.9997, 	f1: 0.8176
Test Epoch 117: 100%|██████████| 1768/1768 [00:48<00:00, 36.50it/s, loss=0.167]
Test Epoch 117 ==> 	accuracy: 0.9390, 	precision: 0.9747, 	recall: 0.7205, 	specificity: 0.9952, 	f1: 0.8285
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 118: 100%|██████████| 6507/6507 [06:58<00:00, 15.55it/s, loss=0.0089]
Train Epoch 118 ==> 	accuracy: 0.8445, 	precision: 0.9996, 	recall: 0.6892, 	specificity: 0.9997, 	f1: 0.8159
Test Epoch 118: 100%|██████████| 1768/1768 [00:47<00:00, 37.59it/s, loss=0.348]
Test Epoch 118 ==> 	accuracy: 0.9393, 	precision: 0.9758, 	recall: 0.7209, 	specificity: 0.9954, 	f1: 0.8292
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 119: 100%|██████████| 6507/6507 [07:03<00:00, 15.37it/s, loss=0.0042]
Train Epoch 119 ==> 	accuracy: 0.8447, 	precision: 0.9996, 	recall: 0.6897, 	specificity: 0.9997, 	f1: 0.8162
Test Epoch 119: 100%|██████████| 1768/1768 [00:47<00:00, 36.91it/s, loss=0.325]
Test Epoch 119 ==> 	accuracy: 0.9397, 	precision: 0.9754, 	recall: 0.7232, 	specificity: 0.9953, 	f1: 0.8306
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 120: 100%|██████████| 6507/6507 [06:55<00:00, 15.65it/s, loss=0.0087]
Train Epoch 120 ==> 	accuracy: 0.8440, 	precision: 0.9996, 	recall: 0.6883, 	specificity: 0.9997, 	f1: 0.8152
Test Epoch 120: 100%|██████████| 1768/1768 [00:49<00:00, 35.61it/s, loss=0.493]
Test Epoch 120 ==> 	accuracy: 0.9395, 	precision: 0.9754, 	recall: 0.7223, 	specificity: 0.9953, 	f1: 0.8300
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 121: 100%|██████████| 6507/6507 [07:03<00:00, 15.38it/s, loss=0.0164]
Train Epoch 121 ==> 	accuracy: 0.8460, 	precision: 0.9996, 	recall: 0.6923, 	specificity: 0.9997, 	f1: 0.8180
Test Epoch 121: 100%|██████████| 1768/1768 [00:49<00:00, 36.07it/s, loss=0.114]
Test Epoch 121 ==> 	accuracy: 0.9406, 	precision: 0.9726, 	recall: 0.7299, 	specificity: 0.9947, 	f1: 0.8340
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 122: 100%|██████████| 6507/6507 [06:59<00:00, 15.52it/s, loss=0.0151]
Train Epoch 122 ==> 	accuracy: 0.8474, 	precision: 0.9996, 	recall: 0.6952, 	specificity: 0.9997, 	f1: 0.8200
Test Epoch 122: 100%|██████████| 1768/1768 [00:48<00:00, 36.36it/s, loss=0.254]
Test Epoch 122 ==> 	accuracy: 0.9391, 	precision: 0.9755, 	recall: 0.7206, 	specificity: 0.9953, 	f1: 0.8289
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 123: 100%|██████████| 6507/6507 [06:59<00:00, 15.52it/s, loss=0.0073]
Train Epoch 123 ==> 	accuracy: 0.8438, 	precision: 0.9996, 	recall: 0.6878, 	specificity: 0.9997, 	f1: 0.8149
Test Epoch 123: 100%|██████████| 1768/1768 [00:47<00:00, 37.28it/s, loss=0.225]
Test Epoch 123 ==> 	accuracy: 0.9384, 	precision: 0.9769, 	recall: 0.7158, 	specificity: 0.9957, 	f1: 0.8262
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 124: 100%|██████████| 6507/6507 [06:59<00:00, 15.50it/s, loss=0.0116]
Train Epoch 124 ==> 	accuracy: 0.8474, 	precision: 0.9996, 	recall: 0.6951, 	specificity: 0.9997, 	f1: 0.8200
Test Epoch 124: 100%|██████████| 1768/1768 [00:49<00:00, 35.79it/s, loss=0.42]
Test Epoch 124 ==> 	accuracy: 0.9403, 	precision: 0.9732, 	recall: 0.7284, 	specificity: 0.9948, 	f1: 0.8332
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 125: 100%|██████████| 6507/6507 [07:07<00:00, 15.23it/s, loss=0.004]
Train Epoch 125 ==> 	accuracy: 0.8464, 	precision: 0.9996, 	recall: 0.6931, 	specificity: 0.9997, 	f1: 0.8186
Test Epoch 125: 100%|██████████| 1768/1768 [00:47<00:00, 37.60it/s, loss=0.695]
Test Epoch 125 ==> 	accuracy: 0.9401, 	precision: 0.9745, 	recall: 0.7261, 	specificity: 0.9951, 	f1: 0.8321
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 126: 100%|██████████| 6507/6507 [07:04<00:00, 15.33it/s, loss=0.0139]
Train Epoch 126 ==> 	accuracy: 0.8478, 	precision: 0.9996, 	recall: 0.6958, 	specificity: 0.9997, 	f1: 0.8205
Test Epoch 126: 100%|██████████| 1768/1768 [00:47<00:00, 36.84it/s, loss=0.405]
Test Epoch 126 ==> 	accuracy: 0.9392, 	precision: 0.9751, 	recall: 0.7214, 	specificity: 0.9953, 	f1: 0.8293
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 127: 100%|██████████| 6507/6507 [07:08<00:00, 15.20it/s, loss=0.0041]
Train Epoch 127 ==> 	accuracy: 0.8474, 	precision: 0.9996, 	recall: 0.6951, 	specificity: 0.9997, 	f1: 0.8200
Test Epoch 127: 100%|██████████| 1768/1768 [00:48<00:00, 36.37it/s, loss=0.276]
Test Epoch 127 ==> 	accuracy: 0.9396, 	precision: 0.9738, 	recall: 0.7240, 	specificity: 0.9950, 	f1: 0.8305
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 128: 100%|██████████| 6507/6507 [07:04<00:00, 15.34it/s, loss=0.021]
Train Epoch 128 ==> 	accuracy: 0.8460, 	precision: 0.9996, 	recall: 0.6923, 	specificity: 0.9997, 	f1: 0.8181
Test Epoch 128: 100%|██████████| 1768/1768 [00:49<00:00, 35.72it/s, loss=0.289]
Test Epoch 128 ==> 	accuracy: 0.9408, 	precision: 0.9720, 	recall: 0.7315, 	specificity: 0.9946, 	f1: 0.8348
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 129: 100%|██████████| 6507/6507 [07:05<00:00, 15.28it/s, loss=0.0006]
Train Epoch 129 ==> 	accuracy: 0.8449, 	precision: 0.9996, 	recall: 0.6900, 	specificity: 0.9997, 	f1: 0.8164
Test Epoch 129: 100%|██████████| 1768/1768 [00:47<00:00, 37.10it/s, loss=0.129]
Test Epoch 129 ==> 	accuracy: 0.9382, 	precision: 0.9781, 	recall: 0.7140, 	specificity: 0.9959, 	f1: 0.8254
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 130: 100%|██████████| 6507/6507 [06:58<00:00, 15.56it/s, loss=0.0095]
Train Epoch 130 ==> 	accuracy: 0.8454, 	precision: 0.9996, 	recall: 0.6911, 	specificity: 0.9997, 	f1: 0.8172
Test Epoch 130: 100%|██████████| 1768/1768 [00:49<00:00, 35.93it/s, loss=0.149]
Test Epoch 130 ==> 	accuracy: 0.9402, 	precision: 0.9742, 	recall: 0.7271, 	specificity: 0.9951, 	f1: 0.8327
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 131: 100%|██████████| 6507/6507 [07:06<00:00, 15.26it/s, loss=0.0016]
Train Epoch 131 ==> 	accuracy: 0.8464, 	precision: 0.9996, 	recall: 0.6931, 	specificity: 0.9997, 	f1: 0.8186
Test Epoch 131: 100%|██████████| 1768/1768 [00:48<00:00, 36.27it/s, loss=0.227]
Test Epoch 131 ==> 	accuracy: 0.9397, 	precision: 0.9740, 	recall: 0.7246, 	specificity: 0.9950, 	f1: 0.8310
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 132: 100%|██████████| 6507/6507 [07:08<00:00, 15.19it/s, loss=0.0011]
Train Epoch 132 ==> 	accuracy: 0.8480, 	precision: 0.9996, 	recall: 0.6963, 	specificity: 0.9997, 	f1: 0.8208
Test Epoch 132: 100%|██████████| 1768/1768 [00:50<00:00, 34.91it/s, loss=0.567]
Test Epoch 132 ==> 	accuracy: 0.9382, 	precision: 0.9776, 	recall: 0.7142, 	specificity: 0.9958, 	f1: 0.8254
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 133: 100%|██████████| 6507/6507 [07:12<00:00, 15.03it/s, loss=0.0115]
Train Epoch 133 ==> 	accuracy: 0.8474, 	precision: 0.9996, 	recall: 0.6950, 	specificity: 0.9997, 	f1: 0.8199
Test Epoch 133: 100%|██████████| 1768/1768 [00:47<00:00, 37.12it/s, loss=0.123]
Test Epoch 133 ==> 	accuracy: 0.9397, 	precision: 0.9752, 	recall: 0.7236, 	specificity: 0.9953, 	f1: 0.8308
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 134: 100%|██████████| 6507/6507 [07:03<00:00, 15.35it/s, loss=0.003]
Train Epoch 134 ==> 	accuracy: 0.8490, 	precision: 0.9996, 	recall: 0.6982, 	specificity: 0.9997, 	f1: 0.8222
Test Epoch 134: 100%|██████████| 1768/1768 [00:48<00:00, 36.67it/s, loss=0.575]
Test Epoch 134 ==> 	accuracy: 0.9395, 	precision: 0.9748, 	recall: 0.7231, 	specificity: 0.9952, 	f1: 0.8303
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 135: 100%|██████████| 6507/6507 [07:05<00:00, 15.28it/s, loss=0.02]
Train Epoch 135 ==> 	accuracy: 0.8488, 	precision: 0.9996, 	recall: 0.6979, 	specificity: 0.9997, 	f1: 0.8219
Test Epoch 135: 100%|██████████| 1768/1768 [00:46<00:00, 38.30it/s, loss=0.116]
Test Epoch 135 ==> 	accuracy: 0.9404, 	precision: 0.9733, 	recall: 0.7287, 	specificity: 0.9949, 	f1: 0.8334
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 136: 100%|██████████| 6507/6507 [07:02<00:00, 15.39it/s, loss=0.0118]
Train Epoch 136 ==> 	accuracy: 0.8484, 	precision: 0.9996, 	recall: 0.6971, 	specificity: 0.9997, 	f1: 0.8214
Test Epoch 136: 100%|██████████| 1768/1768 [00:48<00:00, 36.44it/s, loss=2.6]
Test Epoch 136 ==> 	accuracy: 0.9408, 	precision: 0.9731, 	recall: 0.7310, 	specificity: 0.9948, 	f1: 0.8349
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 137: 100%|██████████| 6507/6507 [07:01<00:00, 15.44it/s, loss=0.0048]
Train Epoch 137 ==> 	accuracy: 0.8490, 	precision: 0.9996, 	recall: 0.6983, 	specificity: 0.9997, 	f1: 0.8222
Test Epoch 137: 100%|██████████| 1768/1768 [00:47<00:00, 36.97it/s, loss=0.236]
Test Epoch 137 ==> 	accuracy: 0.9397, 	precision: 0.9747, 	recall: 0.7241, 	specificity: 0.9952, 	f1: 0.8309
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 138: 100%|██████████| 6507/6507 [07:06<00:00, 15.26it/s, loss=0.0028]
Train Epoch 138 ==> 	accuracy: 0.8487, 	precision: 0.9996, 	recall: 0.6977, 	specificity: 0.9997, 	f1: 0.8218
Test Epoch 138: 100%|██████████| 1768/1768 [00:49<00:00, 35.59it/s, loss=0.965]
Test Epoch 138 ==> 	accuracy: 0.9403, 	precision: 0.9732, 	recall: 0.7280, 	specificity: 0.9948, 	f1: 0.8329
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 139: 100%|██████████| 6507/6507 [07:07<00:00, 15.23it/s, loss=0.0226]
Train Epoch 139 ==> 	accuracy: 0.8495, 	precision: 0.9996, 	recall: 0.6993, 	specificity: 0.9997, 	f1: 0.8229
Test Epoch 139: 100%|██████████| 1768/1768 [00:48<00:00, 36.25it/s, loss=0.198]
Test Epoch 139 ==> 	accuracy: 0.9408, 	precision: 0.9714, 	recall: 0.7322, 	specificity: 0.9945, 	f1: 0.8350
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 140: 100%|██████████| 6507/6507 [07:03<00:00, 15.35it/s, loss=0.0096]
Train Epoch 140 ==> 	accuracy: 0.8478, 	precision: 0.9996, 	recall: 0.6959, 	specificity: 0.9997, 	f1: 0.8206
Test Epoch 140: 100%|██████████| 1768/1768 [00:49<00:00, 35.77it/s, loss=0.165]
Test Epoch 140 ==> 	accuracy: 0.9369, 	precision: 0.9789, 	recall: 0.7067, 	specificity: 0.9961, 	f1: 0.8208
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 141: 100%|██████████| 6507/6507 [07:03<00:00, 15.38it/s, loss=0.0125]
Train Epoch 141 ==> 	accuracy: 0.8497, 	precision: 0.9996, 	recall: 0.6997, 	specificity: 0.9997, 	f1: 0.8232
Test Epoch 141: 100%|██████████| 1768/1768 [00:47<00:00, 37.25it/s, loss=0.456]
Test Epoch 141 ==> 	accuracy: 0.9411, 	precision: 0.9713, 	recall: 0.7335, 	specificity: 0.9944, 	f1: 0.8358
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 142: 100%|██████████| 6507/6507 [07:08<00:00, 15.20it/s, loss=0.0234]
Train Epoch 142 ==> 	accuracy: 0.8483, 	precision: 0.9996, 	recall: 0.6968, 	specificity: 0.9997, 	f1: 0.8212
Test Epoch 142: 100%|██████████| 1768/1768 [00:49<00:00, 35.86it/s, loss=0.249]
Test Epoch 142 ==> 	accuracy: 0.9399, 	precision: 0.9751, 	recall: 0.7250, 	specificity: 0.9952, 	f1: 0.8316
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 143: 100%|██████████| 6507/6507 [07:11<00:00, 15.06it/s, loss=0.0012]
Train Epoch 143 ==> 	accuracy: 0.8510, 	precision: 0.9997, 	recall: 0.7022, 	specificity: 0.9998, 	f1: 0.8249
Test Epoch 143: 100%|██████████| 1768/1768 [00:48<00:00, 36.41it/s, loss=0.128]
Test Epoch 143 ==> 	accuracy: 0.9408, 	precision: 0.9722, 	recall: 0.7317, 	specificity: 0.9946, 	f1: 0.8350
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 144: 100%|██████████| 6507/6507 [07:20<00:00, 14.78it/s, loss=0.0047]
Train Epoch 144 ==> 	accuracy: 0.8485, 	precision: 0.9996, 	recall: 0.6973, 	specificity: 0.9997, 	f1: 0.8216
Test Epoch 144: 100%|██████████| 1768/1768 [00:49<00:00, 35.70it/s, loss=0.333]
Test Epoch 144 ==> 	accuracy: 0.9416, 	precision: 0.9714, 	recall: 0.7362, 	specificity: 0.9944, 	f1: 0.8376
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 145: 100%|██████████| 6507/6507 [07:17<00:00, 14.86it/s, loss=0.0154]
Train Epoch 145 ==> 	accuracy: 0.8482, 	precision: 0.9996, 	recall: 0.6967, 	specificity: 0.9997, 	f1: 0.8211
Test Epoch 145: 100%|██████████| 1768/1768 [00:47<00:00, 37.15it/s, loss=0.164]
Test Epoch 145 ==> 	accuracy: 0.9415, 	precision: 0.9719, 	recall: 0.7351, 	specificity: 0.9945, 	f1: 0.8371
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 146: 100%|██████████| 6507/6507 [07:05<00:00, 15.29it/s, loss=0.0006]
Train Epoch 146 ==> 	accuracy: 0.8495, 	precision: 0.9996, 	recall: 0.6993, 	specificity: 0.9997, 	f1: 0.8229
Test Epoch 146: 100%|██████████| 1768/1768 [00:50<00:00, 34.87it/s, loss=0.102]
Test Epoch 146 ==> 	accuracy: 0.9406, 	precision: 0.9724, 	recall: 0.7304, 	specificity: 0.9947, 	f1: 0.8342
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 147: 100%|██████████| 6507/6507 [07:08<00:00, 15.20it/s, loss=0.0187]
Train Epoch 147 ==> 	accuracy: 0.8484, 	precision: 0.9996, 	recall: 0.6970, 	specificity: 0.9997, 	f1: 0.8213
Test Epoch 147: 100%|██████████| 1768/1768 [00:48<00:00, 36.28it/s, loss=0.242]
Test Epoch 147 ==> 	accuracy: 0.9408, 	precision: 0.9729, 	recall: 0.7310, 	specificity: 0.9948, 	f1: 0.8348
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 148: 100%|██████████| 6507/6507 [07:11<00:00, 15.08it/s, loss=0.0032]
Train Epoch 148 ==> 	accuracy: 0.8523, 	precision: 0.9996, 	recall: 0.7048, 	specificity: 0.9997, 	f1: 0.8267
Test Epoch 148: 100%|██████████| 1768/1768 [00:47<00:00, 36.87it/s, loss=0.916]
Test Epoch 148 ==> 	accuracy: 0.9413, 	precision: 0.9714, 	recall: 0.7348, 	specificity: 0.9944, 	f1: 0.8367
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 149: 100%|██████████| 6507/6507 [07:14<00:00, 14.99it/s, loss=0.0121]
Train Epoch 149 ==> 	accuracy: 0.8489, 	precision: 0.9996, 	recall: 0.6981, 	specificity: 0.9997, 	f1: 0.8221
Test Epoch 149: 100%|██████████| 1768/1768 [00:50<00:00, 35.35it/s, loss=1.6]
Test Epoch 149 ==> 	accuracy: 0.9399, 	precision: 0.9750, 	recall: 0.7247, 	specificity: 0.9952, 	f1: 0.8314
Adjusting learning rate of group 0 to 5.8150e-06.

进程已结束，退出代码为 0

'''

'''
'../model_save_sigBlock4_focalWithMs_deformable_7mer_ab_seq'
/home/bio/anaconda3/bin/python /home/bio/bio_seq/oxog/oxog/script/train.py 
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 0: 100%|██████████| 6507/6507 [08:33<00:00, 12.67it/s, loss=0.121]
Train Epoch 0 ==> 	accuracy: 0.5664, 	precision: 0.9941, 	recall: 0.1336, 	specificity: 0.9992, 	f1: 0.2355
Test Epoch 0: 100%|██████████| 1768/1768 [00:59<00:00, 29.63it/s, loss=0.343]
Test Epoch 0 ==> 	accuracy: 0.8685, 	precision: 0.9793, 	recall: 0.3648, 	specificity: 0.9980, 	f1: 0.5316
Train Epoch 1: 100%|██████████| 6507/6507 [09:26<00:00, 11.48it/s, loss=0.0852]
Train Epoch 1 ==> 	accuracy: 0.6529, 	precision: 0.9964, 	recall: 0.3068, 	specificity: 0.9989, 	f1: 0.4692
Test Epoch 1: 100%|██████████| 1768/1768 [00:57<00:00, 30.72it/s, loss=0.283]
Test Epoch 1 ==> 	accuracy: 0.8853, 	precision: 0.9905, 	recall: 0.4436, 	specificity: 0.9989, 	f1: 0.6127
Train Epoch 2: 100%|██████████| 6507/6507 [09:33<00:00, 11.35it/s, loss=0.031]
Train Epoch 2 ==> 	accuracy: 0.6886, 	precision: 0.9972, 	recall: 0.3783, 	specificity: 0.9989, 	f1: 0.5485
Test Epoch 2: 100%|██████████| 1768/1768 [01:03<00:00, 27.99it/s, loss=0.351]
Test Epoch 2 ==> 	accuracy: 0.8977, 	precision: 0.9870, 	recall: 0.5066, 	specificity: 0.9983, 	f1: 0.6695
Train Epoch 3: 100%|██████████| 6507/6507 [09:50<00:00, 11.03it/s, loss=0.0786]
Train Epoch 3 ==> 	accuracy: 0.7020, 	precision: 0.9975, 	recall: 0.4051, 	specificity: 0.9990, 	f1: 0.5762
Test Epoch 3: 100%|██████████| 1768/1768 [00:59<00:00, 29.85it/s, loss=0.154]
Test Epoch 3 ==> 	accuracy: 0.8901, 	precision: 0.9940, 	recall: 0.4655, 	specificity: 0.9993, 	f1: 0.6340
Train Epoch 4: 100%|██████████| 6507/6507 [09:48<00:00, 11.06it/s, loss=0.0887]
Train Epoch 4 ==> 	accuracy: 0.7196, 	precision: 0.9978, 	recall: 0.4401, 	specificity: 0.9990, 	f1: 0.6108
Test Epoch 4: 100%|██████████| 1768/1768 [00:59<00:00, 29.59it/s, loss=0.244]
Test Epoch 4 ==> 	accuracy: 0.8869, 	precision: 0.9921, 	recall: 0.4506, 	specificity: 0.9991, 	f1: 0.6197
Train Epoch 5: 100%|██████████| 6507/6507 [09:39<00:00, 11.22it/s, loss=0.0905]
Train Epoch 5 ==> 	accuracy: 0.7264, 	precision: 0.9979, 	recall: 0.4538, 	specificity: 0.9991, 	f1: 0.6239
Test Epoch 5: 100%|██████████| 1768/1768 [01:00<00:00, 29.25it/s, loss=0.159]
Test Epoch 5 ==> 	accuracy: 0.8913, 	precision: 0.9936, 	recall: 0.4716, 	specificity: 0.9992, 	f1: 0.6396
Train Epoch 6: 100%|██████████| 6507/6507 [09:53<00:00, 10.96it/s, loss=0.0792]
Train Epoch 6 ==> 	accuracy: 0.7288, 	precision: 0.9981, 	recall: 0.4584, 	specificity: 0.9991, 	f1: 0.6283
Test Epoch 6: 100%|██████████| 1768/1768 [01:02<00:00, 28.38it/s, loss=0.3]
Test Epoch 6 ==> 	accuracy: 0.8934, 	precision: 0.9949, 	recall: 0.4813, 	specificity: 0.9994, 	f1: 0.6488
Train Epoch 7: 100%|██████████| 6507/6507 [09:41<00:00, 11.20it/s, loss=0.0916]
Train Epoch 7 ==> 	accuracy: 0.7408, 	precision: 0.9982, 	recall: 0.4824, 	specificity: 0.9991, 	f1: 0.6505
Test Epoch 7: 100%|██████████| 1768/1768 [01:00<00:00, 29.10it/s, loss=0.256]
Test Epoch 7 ==> 	accuracy: 0.9049, 	precision: 0.9895, 	recall: 0.5407, 	specificity: 0.9985, 	f1: 0.6993
Train Epoch 8: 100%|██████████| 6507/6507 [09:39<00:00, 11.23it/s, loss=0.031]
Train Epoch 8 ==> 	accuracy: 0.7372, 	precision: 0.9983, 	recall: 0.4752, 	specificity: 0.9992, 	f1: 0.6439
Test Epoch 8: 100%|██████████| 1768/1768 [00:57<00:00, 30.61it/s, loss=0.442]
Test Epoch 8 ==> 	accuracy: 0.8990, 	precision: 0.9940, 	recall: 0.5092, 	specificity: 0.9992, 	f1: 0.6734
Train Epoch 9: 100%|██████████| 6507/6507 [09:41<00:00, 11.18it/s, loss=0.0681]
Train Epoch 9 ==> 	accuracy: 0.7449, 	precision: 0.9985, 	recall: 0.4905, 	specificity: 0.9993, 	f1: 0.6579
Test Epoch 9: 100%|██████████| 1768/1768 [01:00<00:00, 29.43it/s, loss=0.283]
Test Epoch 9 ==> 	accuracy: 0.9138, 	precision: 0.9883, 	recall: 0.5857, 	specificity: 0.9982, 	f1: 0.7355
Train Epoch 10: 100%|██████████| 6507/6507 [09:44<00:00, 11.13it/s, loss=0.0734]
Train Epoch 10 ==> 	accuracy: 0.7493, 	precision: 0.9985, 	recall: 0.4993, 	specificity: 0.9992, 	f1: 0.6657
Test Epoch 10: 100%|██████████| 1768/1768 [01:00<00:00, 29.13it/s, loss=0.253]
Test Epoch 10 ==> 	accuracy: 0.8965, 	precision: 0.9942, 	recall: 0.4968, 	specificity: 0.9993, 	f1: 0.6625
Train Epoch 11: 100%|██████████| 6507/6507 [09:41<00:00, 11.20it/s, loss=0.0647]
Train Epoch 11 ==> 	accuracy: 0.7565, 	precision: 0.9986, 	recall: 0.5137, 	specificity: 0.9993, 	f1: 0.6785
Test Epoch 11: 100%|██████████| 1768/1768 [01:01<00:00, 28.76it/s, loss=0.209]
Test Epoch 11 ==> 	accuracy: 0.9151, 	precision: 0.9898, 	recall: 0.5909, 	specificity: 0.9984, 	f1: 0.7400
Train Epoch 12: 100%|██████████| 6507/6507 [09:38<00:00, 11.24it/s, loss=0.095]
Train Epoch 12 ==> 	accuracy: 0.7546, 	precision: 0.9986, 	recall: 0.5100, 	specificity: 0.9993, 	f1: 0.6751
Test Epoch 12: 100%|██████████| 1768/1768 [01:02<00:00, 28.35it/s, loss=0.147]
Test Epoch 12 ==> 	accuracy: 0.9076, 	precision: 0.9921, 	recall: 0.5528, 	specificity: 0.9989, 	f1: 0.7099
Train Epoch 13: 100%|██████████| 6507/6507 [09:38<00:00, 11.24it/s, loss=0.0413]
Train Epoch 13 ==> 	accuracy: 0.7591, 	precision: 0.9987, 	recall: 0.5188, 	specificity: 0.9993, 	f1: 0.6829
Test Epoch 13: 100%|██████████| 1768/1768 [01:00<00:00, 29.47it/s, loss=0.234]
Test Epoch 13 ==> 	accuracy: 0.9090, 	precision: 0.9923, 	recall: 0.5597, 	specificity: 0.9989, 	f1: 0.7157
Train Epoch 14: 100%|██████████| 6507/6507 [09:53<00:00, 10.96it/s, loss=0.105]
Train Epoch 14 ==> 	accuracy: 0.7661, 	precision: 0.9987, 	recall: 0.5330, 	specificity: 0.9993, 	f1: 0.6950
Test Epoch 14: 100%|██████████| 1768/1768 [01:01<00:00, 28.64it/s, loss=0.212]
Test Epoch 14 ==> 	accuracy: 0.9066, 	precision: 0.9932, 	recall: 0.5470, 	specificity: 0.9990, 	f1: 0.7055
Train Epoch 15: 100%|██████████| 6507/6507 [09:45<00:00, 11.11it/s, loss=0.0936]
Train Epoch 15 ==> 	accuracy: 0.7637, 	precision: 0.9987, 	recall: 0.5281, 	specificity: 0.9993, 	f1: 0.6909
Test Epoch 15: 100%|██████████| 1768/1768 [01:01<00:00, 28.74it/s, loss=0.159]
Test Epoch 15 ==> 	accuracy: 0.9115, 	precision: 0.9926, 	recall: 0.5718, 	specificity: 0.9989, 	f1: 0.7256
Train Epoch 16: 100%|██████████| 6507/6507 [09:47<00:00, 11.08it/s, loss=0.136]
Train Epoch 16 ==> 	accuracy: 0.7645, 	precision: 0.9989, 	recall: 0.5296, 	specificity: 0.9994, 	f1: 0.6922
Test Epoch 16: 100%|██████████| 1768/1768 [01:00<00:00, 29.19it/s, loss=0.27]
Test Epoch 16 ==> 	accuracy: 0.9183, 	precision: 0.9899, 	recall: 0.6068, 	specificity: 0.9984, 	f1: 0.7524
Train Epoch 17: 100%|██████████| 6507/6507 [09:44<00:00, 11.14it/s, loss=0.0457]
Train Epoch 17 ==> 	accuracy: 0.7706, 	precision: 0.9989, 	recall: 0.5418, 	specificity: 0.9994, 	f1: 0.7025
Test Epoch 17: 100%|██████████| 1768/1768 [01:02<00:00, 28.49it/s, loss=0.126]
Test Epoch 17 ==> 	accuracy: 0.9158, 	precision: 0.9902, 	recall: 0.5942, 	specificity: 0.9985, 	f1: 0.7427
Train Epoch 18: 100%|██████████| 6507/6507 [09:36<00:00, 11.28it/s, loss=0.0918]
Train Epoch 18 ==> 	accuracy: 0.7739, 	precision: 0.9989, 	recall: 0.5484, 	specificity: 0.9994, 	f1: 0.7081
Test Epoch 18: 100%|██████████| 1768/1768 [01:01<00:00, 28.61it/s, loss=0.202]
Test Epoch 18 ==> 	accuracy: 0.9072, 	precision: 0.9935, 	recall: 0.5501, 	specificity: 0.9991, 	f1: 0.7081
Train Epoch 19: 100%|██████████| 6507/6507 [09:23<00:00, 11.55it/s, loss=0.0137]
Train Epoch 19 ==> 	accuracy: 0.7724, 	precision: 0.9990, 	recall: 0.5453, 	specificity: 0.9994, 	f1: 0.7055
Test Epoch 19: 100%|██████████| 1768/1768 [00:59<00:00, 29.52it/s, loss=0.187]
Test Epoch 19 ==> 	accuracy: 0.9133, 	precision: 0.9916, 	recall: 0.5812, 	specificity: 0.9987, 	f1: 0.7329
Train Epoch 20: 100%|██████████| 6507/6507 [09:13<00:00, 11.76it/s, loss=0.0372]
Train Epoch 20 ==> 	accuracy: 0.7733, 	precision: 0.9989, 	recall: 0.5473, 	specificity: 0.9994, 	f1: 0.7071
Test Epoch 20: 100%|██████████| 1768/1768 [01:01<00:00, 28.98it/s, loss=0.283]
Test Epoch 20 ==> 	accuracy: 0.9134, 	precision: 0.9918, 	recall: 0.5814, 	specificity: 0.9988, 	f1: 0.7331
Train Epoch 21: 100%|██████████| 6507/6507 [09:19<00:00, 11.63it/s, loss=0.0061]
Train Epoch 21 ==> 	accuracy: 0.7778, 	precision: 0.9990, 	recall: 0.5562, 	specificity: 0.9994, 	f1: 0.7145
Test Epoch 21: 100%|██████████| 1768/1768 [00:59<00:00, 29.76it/s, loss=0.324]
Test Epoch 21 ==> 	accuracy: 0.9177, 	precision: 0.9906, 	recall: 0.6035, 	specificity: 0.9985, 	f1: 0.7500
Train Epoch 22: 100%|██████████| 6507/6507 [09:24<00:00, 11.53it/s, loss=0.0696]
Train Epoch 22 ==> 	accuracy: 0.7786, 	precision: 0.9990, 	recall: 0.5577, 	specificity: 0.9994, 	f1: 0.7158
Test Epoch 22: 100%|██████████| 1768/1768 [01:03<00:00, 27.80it/s, loss=0.197]
Test Epoch 22 ==> 	accuracy: 0.9147, 	precision: 0.9917, 	recall: 0.5878, 	specificity: 0.9987, 	f1: 0.7381
Train Epoch 23: 100%|██████████| 6507/6507 [09:27<00:00, 11.48it/s, loss=0.088]
Train Epoch 23 ==> 	accuracy: 0.7808, 	precision: 0.9990, 	recall: 0.5621, 	specificity: 0.9995, 	f1: 0.7194
Test Epoch 23: 100%|██████████| 1768/1768 [01:00<00:00, 29.14it/s, loss=0.222]
Test Epoch 23 ==> 	accuracy: 0.9147, 	precision: 0.9908, 	recall: 0.5882, 	specificity: 0.9986, 	f1: 0.7382
Train Epoch 24: 100%|██████████| 6507/6507 [09:31<00:00, 11.38it/s, loss=0.079]
Train Epoch 24 ==> 	accuracy: 0.7818, 	precision: 0.9990, 	recall: 0.5642, 	specificity: 0.9995, 	f1: 0.7211
Test Epoch 24: 100%|██████████| 1768/1768 [00:59<00:00, 29.92it/s, loss=0.208]
Test Epoch 24 ==> 	accuracy: 0.9174, 	precision: 0.9902, 	recall: 0.6019, 	specificity: 0.9985, 	f1: 0.7487
Train Epoch 25: 100%|██████████| 6507/6507 [09:28<00:00, 11.45it/s, loss=0.0771]
Train Epoch 25 ==> 	accuracy: 0.7845, 	precision: 0.9991, 	recall: 0.5696, 	specificity: 0.9995, 	f1: 0.7255
Test Epoch 25: 100%|██████████| 1768/1768 [01:00<00:00, 29.09it/s, loss=0.185]
Test Epoch 25 ==> 	accuracy: 0.9209, 	precision: 0.9895, 	recall: 0.6200, 	specificity: 0.9983, 	f1: 0.7624
Train Epoch 26: 100%|██████████| 6507/6507 [09:21<00:00, 11.59it/s, loss=0.163]
Train Epoch 26 ==> 	accuracy: 0.7871, 	precision: 0.9991, 	recall: 0.5746, 	specificity: 0.9995, 	f1: 0.7296
Test Epoch 26: 100%|██████████| 1768/1768 [00:58<00:00, 30.30it/s, loss=0.182]
Test Epoch 26 ==> 	accuracy: 0.9191, 	precision: 0.9888, 	recall: 0.6116, 	specificity: 0.9982, 	f1: 0.7557
Train Epoch 27: 100%|██████████| 6507/6507 [09:16<00:00, 11.70it/s, loss=0.0692]
Train Epoch 27 ==> 	accuracy: 0.7864, 	precision: 0.9990, 	recall: 0.5733, 	specificity: 0.9995, 	f1: 0.7285
Test Epoch 27: 100%|██████████| 1768/1768 [01:00<00:00, 29.40it/s, loss=0.291]
Test Epoch 27 ==> 	accuracy: 0.9195, 	precision: 0.9904, 	recall: 0.6125, 	specificity: 0.9985, 	f1: 0.7569
Train Epoch 28: 100%|██████████| 6507/6507 [09:12<00:00, 11.77it/s, loss=0.0421]
Train Epoch 28 ==> 	accuracy: 0.7886, 	precision: 0.9991, 	recall: 0.5777, 	specificity: 0.9995, 	f1: 0.7321
Test Epoch 28: 100%|██████████| 1768/1768 [01:00<00:00, 29.18it/s, loss=0.149]
Test Epoch 28 ==> 	accuracy: 0.9187, 	precision: 0.9901, 	recall: 0.6086, 	specificity: 0.9984, 	f1: 0.7538
Train Epoch 29: 100%|██████████| 6507/6507 [09:19<00:00, 11.63it/s, loss=0.0224]
Train Epoch 29 ==> 	accuracy: 0.7893, 	precision: 0.9991, 	recall: 0.5791, 	specificity: 0.9995, 	f1: 0.7332
Test Epoch 29: 100%|██████████| 1768/1768 [01:00<00:00, 29.45it/s, loss=0.204]
Test Epoch 29 ==> 	accuracy: 0.9241, 	precision: 0.9888, 	recall: 0.6363, 	specificity: 0.9981, 	f1: 0.7743
Train Epoch 30: 100%|██████████| 6507/6507 [09:15<00:00, 11.72it/s, loss=0.0685]
Train Epoch 30 ==> 	accuracy: 0.7896, 	precision: 0.9991, 	recall: 0.5798, 	specificity: 0.9995, 	f1: 0.7337
Test Epoch 30: 100%|██████████| 1768/1768 [01:01<00:00, 28.74it/s, loss=0.393]
Test Epoch 30 ==> 	accuracy: 0.9141, 	precision: 0.9925, 	recall: 0.5846, 	specificity: 0.9989, 	f1: 0.7358
Train Epoch 31: 100%|██████████| 6507/6507 [09:22<00:00, 11.58it/s, loss=0.0248]
Train Epoch 31 ==> 	accuracy: 0.7947, 	precision: 0.9991, 	recall: 0.5899, 	specificity: 0.9995, 	f1: 0.7418
Test Epoch 31: 100%|██████████| 1768/1768 [00:59<00:00, 29.70it/s, loss=0.278]
Test Epoch 31 ==> 	accuracy: 0.9051, 	precision: 0.9936, 	recall: 0.5395, 	specificity: 0.9991, 	f1: 0.6993
Train Epoch 32: 100%|██████████| 6507/6507 [09:03<00:00, 11.98it/s, loss=0.036]
Train Epoch 32 ==> 	accuracy: 0.7892, 	precision: 0.9991, 	recall: 0.5788, 	specificity: 0.9995, 	f1: 0.7330
Test Epoch 32: 100%|██████████| 1768/1768 [00:59<00:00, 29.70it/s, loss=0.179]
Test Epoch 32 ==> 	accuracy: 0.9227, 	precision: 0.9901, 	recall: 0.6285, 	specificity: 0.9984, 	f1: 0.7689
Train Epoch 33: 100%|██████████| 6507/6507 [09:13<00:00, 11.75it/s, loss=0.0399]
Train Epoch 33 ==> 	accuracy: 0.7932, 	precision: 0.9992, 	recall: 0.5869, 	specificity: 0.9995, 	f1: 0.7395
Test Epoch 33: 100%|██████████| 1768/1768 [00:58<00:00, 30.08it/s, loss=0.209]
Test Epoch 33 ==> 	accuracy: 0.9184, 	precision: 0.9921, 	recall: 0.6059, 	specificity: 0.9988, 	f1: 0.7523
Train Epoch 34: 100%|██████████| 6507/6507 [09:17<00:00, 11.67it/s, loss=0.119]
Train Epoch 34 ==> 	accuracy: 0.7958, 	precision: 0.9992, 	recall: 0.5921, 	specificity: 0.9995, 	f1: 0.7436
Test Epoch 34: 100%|██████████| 1768/1768 [00:58<00:00, 30.45it/s, loss=0.339]
Test Epoch 34 ==> 	accuracy: 0.9174, 	precision: 0.9872, 	recall: 0.6039, 	specificity: 0.9980, 	f1: 0.7494
Train Epoch 35: 100%|██████████| 6507/6507 [09:17<00:00, 11.68it/s, loss=0.0112]
Train Epoch 35 ==> 	accuracy: 0.7947, 	precision: 0.9992, 	recall: 0.5899, 	specificity: 0.9995, 	f1: 0.7418
Test Epoch 35: 100%|██████████| 1768/1768 [00:59<00:00, 29.86it/s, loss=0.218]
Test Epoch 35 ==> 	accuracy: 0.9239, 	precision: 0.9891, 	recall: 0.6352, 	specificity: 0.9982, 	f1: 0.7736
Train Epoch 36: 100%|██████████| 6507/6507 [09:06<00:00, 11.90it/s, loss=0.107]
Train Epoch 36 ==> 	accuracy: 0.7999, 	precision: 0.9992, 	recall: 0.6003, 	specificity: 0.9995, 	f1: 0.7501
Test Epoch 36: 100%|██████████| 1768/1768 [00:58<00:00, 30.12it/s, loss=0.187]
Test Epoch 36 ==> 	accuracy: 0.9233, 	precision: 0.9885, 	recall: 0.6325, 	specificity: 0.9981, 	f1: 0.7714
Train Epoch 37: 100%|██████████| 6507/6507 [09:09<00:00, 11.85it/s, loss=0.015]
Train Epoch 37 ==> 	accuracy: 0.7969, 	precision: 0.9992, 	recall: 0.5943, 	specificity: 0.9995, 	f1: 0.7453
Test Epoch 37: 100%|██████████| 1768/1768 [00:59<00:00, 29.91it/s, loss=0.507]
Test Epoch 37 ==> 	accuracy: 0.9259, 	precision: 0.9873, 	recall: 0.6462, 	specificity: 0.9979, 	f1: 0.7811
Train Epoch 38: 100%|██████████| 6507/6507 [09:12<00:00, 11.77it/s, loss=0.0808]
Train Epoch 38 ==> 	accuracy: 0.8001, 	precision: 0.9992, 	recall: 0.6006, 	specificity: 0.9995, 	f1: 0.7503
Test Epoch 38: 100%|██████████| 1768/1768 [00:57<00:00, 30.55it/s, loss=0.308]
Test Epoch 38 ==> 	accuracy: 0.9264, 	precision: 0.9877, 	recall: 0.6482, 	specificity: 0.9979, 	f1: 0.7827
Train Epoch 39: 100%|██████████| 6507/6507 [09:10<00:00, 11.82it/s, loss=0.0561]
Train Epoch 39 ==> 	accuracy: 0.8011, 	precision: 0.9992, 	recall: 0.6026, 	specificity: 0.9995, 	f1: 0.7518
Test Epoch 39: 100%|██████████| 1768/1768 [01:01<00:00, 28.80it/s, loss=0.334]
Test Epoch 39 ==> 	accuracy: 0.9257, 	precision: 0.9882, 	recall: 0.6447, 	specificity: 0.9980, 	f1: 0.7803
Train Epoch 40: 100%|██████████| 6507/6507 [08:47<00:00, 12.33it/s, loss=0.0099]
Train Epoch 40 ==> 	accuracy: 0.8046, 	precision: 0.9993, 	recall: 0.6097, 	specificity: 0.9995, 	f1: 0.7573
Test Epoch 40: 100%|██████████| 1768/1768 [00:58<00:00, 30.30it/s, loss=0.152]
Test Epoch 40 ==> 	accuracy: 0.9247, 	precision: 0.9849, 	recall: 0.6415, 	specificity: 0.9975, 	f1: 0.7769
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 41: 100%|██████████| 6507/6507 [08:52<00:00, 12.23it/s, loss=0.0081]
Train Epoch 41 ==> 	accuracy: 0.8028, 	precision: 0.9993, 	recall: 0.6060, 	specificity: 0.9996, 	f1: 0.7544
Test Epoch 41: 100%|██████████| 1768/1768 [00:59<00:00, 29.93it/s, loss=0.147]
Test Epoch 41 ==> 	accuracy: 0.9259, 	precision: 0.9870, 	recall: 0.6461, 	specificity: 0.9978, 	f1: 0.7810
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 42: 100%|██████████| 6507/6507 [08:46<00:00, 12.36it/s, loss=0.0056]
Train Epoch 42 ==> 	accuracy: 0.8034, 	precision: 0.9992, 	recall: 0.6073, 	specificity: 0.9995, 	f1: 0.7554
Test Epoch 42: 100%|██████████| 1768/1768 [00:57<00:00, 30.73it/s, loss=0.241]
Test Epoch 42 ==> 	accuracy: 0.9237, 	precision: 0.9891, 	recall: 0.6338, 	specificity: 0.9982, 	f1: 0.7726
Adjusting learning rate of group 0 to 1.0000e-04.
Train Epoch 43: 100%|██████████| 6507/6507 [08:43<00:00, 12.42it/s, loss=0.0114]
Train Epoch 43 ==> 	accuracy: 0.8055, 	precision: 0.9993, 	recall: 0.6115, 	specificity: 0.9996, 	f1: 0.7587
Test Epoch 43: 100%|██████████| 1768/1768 [01:01<00:00, 28.75it/s, loss=0.149]
Test Epoch 43 ==> 	accuracy: 0.9217, 	precision: 0.9889, 	recall: 0.6242, 	specificity: 0.9982, 	f1: 0.7654
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 44: 100%|██████████| 6507/6507 [08:58<00:00, 12.09it/s, loss=0.0144]
Train Epoch 44 ==> 	accuracy: 0.8088, 	precision: 0.9993, 	recall: 0.6181, 	specificity: 0.9996, 	f1: 0.7637
Test Epoch 44: 100%|██████████| 1768/1768 [01:01<00:00, 28.73it/s, loss=0.188]
Test Epoch 44 ==> 	accuracy: 0.9272, 	precision: 0.9867, 	recall: 0.6529, 	specificity: 0.9977, 	f1: 0.7858
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 45: 100%|██████████| 6507/6507 [09:15<00:00, 11.71it/s, loss=0.0261]
Train Epoch 45 ==> 	accuracy: 0.8055, 	precision: 0.9993, 	recall: 0.6114, 	specificity: 0.9996, 	f1: 0.7587
Test Epoch 45: 100%|██████████| 1768/1768 [01:00<00:00, 29.41it/s, loss=0.159]
Test Epoch 45 ==> 	accuracy: 0.9261, 	precision: 0.9878, 	recall: 0.6467, 	specificity: 0.9979, 	f1: 0.7817
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 46: 100%|██████████| 6507/6507 [09:06<00:00, 11.91it/s, loss=0.016]
Train Epoch 46 ==> 	accuracy: 0.8112, 	precision: 0.9993, 	recall: 0.6227, 	specificity: 0.9996, 	f1: 0.7673
Test Epoch 46: 100%|██████████| 1768/1768 [00:59<00:00, 29.71it/s, loss=0.184]
Test Epoch 46 ==> 	accuracy: 0.9275, 	precision: 0.9862, 	recall: 0.6546, 	specificity: 0.9977, 	f1: 0.7869
Adjusting learning rate of group 0 to 9.0000e-05.
Train Epoch 47: 100%|██████████| 6507/6507 [08:49<00:00, 12.28it/s, loss=0.0135]
Train Epoch 47 ==> 	accuracy: 0.8095, 	precision: 0.9994, 	recall: 0.6193, 	specificity: 0.9996, 	f1: 0.7647
Test Epoch 47: 100%|██████████| 1768/1768 [00:59<00:00, 29.68it/s, loss=0.33]
Test Epoch 47 ==> 	accuracy: 0.9269, 	precision: 0.9862, 	recall: 0.6520, 	specificity: 0.9976, 	f1: 0.7850
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 48: 100%|██████████| 6507/6507 [08:48<00:00, 12.30it/s, loss=0.007]
Train Epoch 48 ==> 	accuracy: 0.8131, 	precision: 0.9993, 	recall: 0.6266, 	specificity: 0.9996, 	f1: 0.7702
Test Epoch 48: 100%|██████████| 1768/1768 [01:00<00:00, 29.36it/s, loss=0.209]
Test Epoch 48 ==> 	accuracy: 0.9291, 	precision: 0.9866, 	recall: 0.6624, 	specificity: 0.9977, 	f1: 0.7926
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 49: 100%|██████████| 6507/6507 [08:59<00:00, 12.06it/s, loss=0.0102]
Train Epoch 49 ==> 	accuracy: 0.8123, 	precision: 0.9994, 	recall: 0.6250, 	specificity: 0.9996, 	f1: 0.7691
Test Epoch 49: 100%|██████████| 1768/1768 [01:01<00:00, 28.81it/s, loss=0.251]
Test Epoch 49 ==> 	accuracy: 0.9320, 	precision: 0.9843, 	recall: 0.6784, 	specificity: 0.9972, 	f1: 0.8032
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 50: 100%|██████████| 6507/6507 [09:00<00:00, 12.04it/s, loss=0.0046]
Train Epoch 50 ==> 	accuracy: 0.8178, 	precision: 0.9993, 	recall: 0.6359, 	specificity: 0.9996, 	f1: 0.7773
Test Epoch 50: 100%|██████████| 1768/1768 [01:01<00:00, 28.71it/s, loss=0.185]
Test Epoch 50 ==> 	accuracy: 0.9308, 	precision: 0.9817, 	recall: 0.6741, 	specificity: 0.9968, 	f1: 0.7993
Adjusting learning rate of group 0 to 8.1000e-05.
Train Epoch 51: 100%|██████████| 6507/6507 [09:06<00:00, 11.91it/s, loss=0.0023]
Train Epoch 51 ==> 	accuracy: 0.8165, 	precision: 0.9993, 	recall: 0.6334, 	specificity: 0.9996, 	f1: 0.7753
Test Epoch 51: 100%|██████████| 1768/1768 [01:01<00:00, 28.78it/s, loss=0.151]
Test Epoch 51 ==> 	accuracy: 0.9307, 	precision: 0.9852, 	recall: 0.6714, 	specificity: 0.9974, 	f1: 0.7986
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 52: 100%|██████████| 6507/6507 [09:07<00:00, 11.88it/s, loss=0.0012]
Train Epoch 52 ==> 	accuracy: 0.8193, 	precision: 0.9994, 	recall: 0.6391, 	specificity: 0.9996, 	f1: 0.7796
Test Epoch 52: 100%|██████████| 1768/1768 [01:01<00:00, 28.76it/s, loss=0.255]
Test Epoch 52 ==> 	accuracy: 0.9322, 	precision: 0.9847, 	recall: 0.6793, 	specificity: 0.9973, 	f1: 0.8040
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 53: 100%|██████████| 6507/6507 [09:07<00:00, 11.89it/s, loss=0.0052]
Train Epoch 53 ==> 	accuracy: 0.8201, 	precision: 0.9994, 	recall: 0.6406, 	specificity: 0.9996, 	f1: 0.7807
Test Epoch 53: 100%|██████████| 1768/1768 [01:00<00:00, 29.06it/s, loss=0.127]
Test Epoch 53 ==> 	accuracy: 0.9334, 	precision: 0.9825, 	recall: 0.6867, 	specificity: 0.9968, 	f1: 0.8084
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 54: 100%|██████████| 6507/6507 [09:10<00:00, 11.82it/s, loss=0.0124]
Train Epoch 54 ==> 	accuracy: 0.8214, 	precision: 0.9994, 	recall: 0.6432, 	specificity: 0.9996, 	f1: 0.7827
Test Epoch 54: 100%|██████████| 1768/1768 [01:01<00:00, 28.81it/s, loss=0.23]
Test Epoch 54 ==> 	accuracy: 0.9302, 	precision: 0.9852, 	recall: 0.6688, 	specificity: 0.9974, 	f1: 0.7967
Adjusting learning rate of group 0 to 7.2900e-05.
Train Epoch 55: 100%|██████████| 6507/6507 [09:03<00:00, 11.98it/s, loss=0.0012]
Train Epoch 55 ==> 	accuracy: 0.8226, 	precision: 0.9994, 	recall: 0.6456, 	specificity: 0.9996, 	f1: 0.7844
Test Epoch 55: 100%|██████████| 1768/1768 [01:00<00:00, 29.15it/s, loss=0.189]
Test Epoch 55 ==> 	accuracy: 0.9268, 	precision: 0.9880, 	recall: 0.6501, 	specificity: 0.9980, 	f1: 0.7842
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 56: 100%|██████████| 6507/6507 [09:05<00:00, 11.93it/s, loss=0.0139]
Train Epoch 56 ==> 	accuracy: 0.8234, 	precision: 0.9995, 	recall: 0.6472, 	specificity: 0.9997, 	f1: 0.7856
Test Epoch 56: 100%|██████████| 1768/1768 [01:00<00:00, 29.01it/s, loss=0.153]
Test Epoch 56 ==> 	accuracy: 0.9348, 	precision: 0.9833, 	recall: 0.6932, 	specificity: 0.9970, 	f1: 0.8131
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 57: 100%|██████████| 6507/6507 [08:56<00:00, 12.13it/s, loss=0.017]
Train Epoch 57 ==> 	accuracy: 0.8287, 	precision: 0.9994, 	recall: 0.6577, 	specificity: 0.9996, 	f1: 0.7933
Test Epoch 57: 100%|██████████| 1768/1768 [00:59<00:00, 29.75it/s, loss=0.3]
Test Epoch 57 ==> 	accuracy: 0.9318, 	precision: 0.9837, 	recall: 0.6779, 	specificity: 0.9971, 	f1: 0.8027
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 58: 100%|██████████| 6507/6507 [09:08<00:00, 11.87it/s, loss=0.0041]
Train Epoch 58 ==> 	accuracy: 0.8241, 	precision: 0.9994, 	recall: 0.6486, 	specificity: 0.9996, 	f1: 0.7867
Test Epoch 58: 100%|██████████| 1768/1768 [01:02<00:00, 28.10it/s, loss=0.109]
Test Epoch 58 ==> 	accuracy: 0.9314, 	precision: 0.9842, 	recall: 0.6753, 	specificity: 0.9972, 	f1: 0.8010
Adjusting learning rate of group 0 to 6.5610e-05.
Train Epoch 59: 100%|██████████| 6507/6507 [08:59<00:00, 12.07it/s, loss=0.0118]
Train Epoch 59 ==> 	accuracy: 0.8287, 	precision: 0.9994, 	recall: 0.6578, 	specificity: 0.9996, 	f1: 0.7934
Test Epoch 59: 100%|██████████| 1768/1768 [01:00<00:00, 29.42it/s, loss=0.184]
Test Epoch 59 ==> 	accuracy: 0.9320, 	precision: 0.9832, 	recall: 0.6791, 	specificity: 0.9970, 	f1: 0.8033
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 60: 100%|██████████| 6507/6507 [09:04<00:00, 11.95it/s, loss=0.0165]
Train Epoch 60 ==> 	accuracy: 0.8295, 	precision: 0.9995, 	recall: 0.6594, 	specificity: 0.9996, 	f1: 0.7946
Test Epoch 60: 100%|██████████| 1768/1768 [01:00<00:00, 29.10it/s, loss=0.22]
Test Epoch 60 ==> 	accuracy: 0.9346, 	precision: 0.9794, 	recall: 0.6949, 	specificity: 0.9962, 	f1: 0.8130
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 61: 100%|██████████| 6507/6507 [08:59<00:00, 12.06it/s, loss=0.0054]
Train Epoch 61 ==> 	accuracy: 0.8324, 	precision: 0.9995, 	recall: 0.6651, 	specificity: 0.9996, 	f1: 0.7987
Test Epoch 61: 100%|██████████| 1768/1768 [01:00<00:00, 29.36it/s, loss=0.19]
Test Epoch 61 ==> 	accuracy: 0.9365, 	precision: 0.9792, 	recall: 0.7044, 	specificity: 0.9961, 	f1: 0.8193
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 62: 100%|██████████| 6507/6507 [08:56<00:00, 12.12it/s, loss=0.035]
Train Epoch 62 ==> 	accuracy: 0.8295, 	precision: 0.9995, 	recall: 0.6594, 	specificity: 0.9997, 	f1: 0.7946
Test Epoch 62: 100%|██████████| 1768/1768 [01:01<00:00, 28.93it/s, loss=0.23]
Test Epoch 62 ==> 	accuracy: 0.9374, 	precision: 0.9779, 	recall: 0.7101, 	specificity: 0.9959, 	f1: 0.8227
Adjusting learning rate of group 0 to 5.9049e-05.
Train Epoch 63: 100%|██████████| 6507/6507 [08:52<00:00, 12.22it/s, loss=0.0078]
Train Epoch 63 ==> 	accuracy: 0.8337, 	precision: 0.9995, 	recall: 0.6677, 	specificity: 0.9996, 	f1: 0.8006
Test Epoch 63: 100%|██████████| 1768/1768 [01:01<00:00, 28.77it/s, loss=1.49]
Test Epoch 63 ==> 	accuracy: 0.9369, 	precision: 0.9795, 	recall: 0.7061, 	specificity: 0.9962, 	f1: 0.8207
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 64: 100%|██████████| 6507/6507 [08:51<00:00, 12.25it/s, loss=0.0054]
Train Epoch 64 ==> 	accuracy: 0.8326, 	precision: 0.9995, 	recall: 0.6655, 	specificity: 0.9997, 	f1: 0.7990
Test Epoch 64: 100%|██████████| 1768/1768 [00:59<00:00, 29.56it/s, loss=1.04]
Test Epoch 64 ==> 	accuracy: 0.9366, 	precision: 0.9805, 	recall: 0.7038, 	specificity: 0.9964, 	f1: 0.8194
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 65: 100%|██████████| 6507/6507 [08:56<00:00, 12.12it/s, loss=0.0073]
Train Epoch 65 ==> 	accuracy: 0.8352, 	precision: 0.9994, 	recall: 0.6708, 	specificity: 0.9996, 	f1: 0.8028
Test Epoch 65: 100%|██████████| 1768/1768 [01:00<00:00, 29.34it/s, loss=0.169]
Test Epoch 65 ==> 	accuracy: 0.9366, 	precision: 0.9803, 	recall: 0.7042, 	specificity: 0.9964, 	f1: 0.8196
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 66: 100%|██████████| 6507/6507 [09:12<00:00, 11.77it/s, loss=0.0069]
Train Epoch 66 ==> 	accuracy: 0.8332, 	precision: 0.9995, 	recall: 0.6668, 	specificity: 0.9996, 	f1: 0.8000
Test Epoch 66: 100%|██████████| 1768/1768 [01:01<00:00, 28.97it/s, loss=0.105]
Test Epoch 66 ==> 	accuracy: 0.9362, 	precision: 0.9805, 	recall: 0.7018, 	specificity: 0.9964, 	f1: 0.8181
Adjusting learning rate of group 0 to 5.3144e-05.
Train Epoch 67: 100%|██████████| 6507/6507 [09:01<00:00, 12.02it/s, loss=0.0073]
Train Epoch 67 ==> 	accuracy: 0.8380, 	precision: 0.9995, 	recall: 0.6763, 	specificity: 0.9997, 	f1: 0.8067
Test Epoch 67: 100%|██████████| 1768/1768 [01:01<00:00, 28.89it/s, loss=0.193]
Test Epoch 67 ==> 	accuracy: 0.9383, 	precision: 0.9763, 	recall: 0.7155, 	specificity: 0.9955, 	f1: 0.8258
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 68: 100%|██████████| 6507/6507 [08:50<00:00, 12.26it/s, loss=0.0264]
Train Epoch 68 ==> 	accuracy: 0.8383, 	precision: 0.9995, 	recall: 0.6769, 	specificity: 0.9997, 	f1: 0.8072
Test Epoch 68: 100%|██████████| 1768/1768 [00:59<00:00, 29.59it/s, loss=0.245]
Test Epoch 68 ==> 	accuracy: 0.9381, 	precision: 0.9783, 	recall: 0.7134, 	specificity: 0.9959, 	f1: 0.8251
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 69: 100%|██████████| 6507/6507 [08:50<00:00, 12.27it/s, loss=0.0004]
Train Epoch 69 ==> 	accuracy: 0.8392, 	precision: 0.9995, 	recall: 0.6788, 	specificity: 0.9997, 	f1: 0.8085
Test Epoch 69: 100%|██████████| 1768/1768 [01:00<00:00, 29.30it/s, loss=0.165]
Test Epoch 69 ==> 	accuracy: 0.9373, 	precision: 0.9795, 	recall: 0.7083, 	specificity: 0.9962, 	f1: 0.8221
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 70: 100%|██████████| 6507/6507 [08:48<00:00, 12.31it/s, loss=0.0077]
Train Epoch 70 ==> 	accuracy: 0.8396, 	precision: 0.9995, 	recall: 0.6795, 	specificity: 0.9997, 	f1: 0.8090
Test Epoch 70: 100%|██████████| 1768/1768 [00:59<00:00, 29.66it/s, loss=1.08]
Test Epoch 70 ==> 	accuracy: 0.9389, 	precision: 0.9761, 	recall: 0.7187, 	specificity: 0.9955, 	f1: 0.8278
Adjusting learning rate of group 0 to 4.7830e-05.
Train Epoch 71: 100%|██████████| 6507/6507 [08:53<00:00, 12.20it/s, loss=0.0116]
Train Epoch 71 ==> 	accuracy: 0.8406, 	precision: 0.9995, 	recall: 0.6816, 	specificity: 0.9997, 	f1: 0.8105
Test Epoch 71: 100%|██████████| 1768/1768 [01:01<00:00, 28.80it/s, loss=0.153]
Test Epoch 71 ==> 	accuracy: 0.9389, 	precision: 0.9757, 	recall: 0.7190, 	specificity: 0.9954, 	f1: 0.8279
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 72: 100%|██████████| 6507/6507 [08:58<00:00, 12.09it/s, loss=0.0087]
Train Epoch 72 ==> 	accuracy: 0.8433, 	precision: 0.9995, 	recall: 0.6869, 	specificity: 0.9997, 	f1: 0.8142
Test Epoch 72: 100%|██████████| 1768/1768 [01:00<00:00, 29.41it/s, loss=0.104]
Test Epoch 72 ==> 	accuracy: 0.9412, 	precision: 0.9733, 	recall: 0.7329, 	specificity: 0.9948, 	f1: 0.8362
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 73: 100%|██████████| 6507/6507 [08:47<00:00, 12.33it/s, loss=0.0558]
Train Epoch 73 ==> 	accuracy: 0.8421, 	precision: 0.9995, 	recall: 0.6844, 	specificity: 0.9997, 	f1: 0.8125
Test Epoch 73: 100%|██████████| 1768/1768 [01:01<00:00, 28.91it/s, loss=0.188]
Test Epoch 73 ==> 	accuracy: 0.9391, 	precision: 0.9774, 	recall: 0.7189, 	specificity: 0.9957, 	f1: 0.8285
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 74: 100%|██████████| 6507/6507 [08:43<00:00, 12.43it/s, loss=0.0025]
Train Epoch 74 ==> 	accuracy: 0.8446, 	precision: 0.9995, 	recall: 0.6896, 	specificity: 0.9997, 	f1: 0.8161
Test Epoch 74: 100%|██████████| 1768/1768 [00:58<00:00, 30.10it/s, loss=0.208]
Test Epoch 74 ==> 	accuracy: 0.9402, 	precision: 0.9745, 	recall: 0.7266, 	specificity: 0.9951, 	f1: 0.8325
Adjusting learning rate of group 0 to 4.3047e-05.
Train Epoch 75: 100%|██████████| 6507/6507 [08:43<00:00, 12.44it/s, loss=0.0079]
Train Epoch 75 ==> 	accuracy: 0.8432, 	precision: 0.9995, 	recall: 0.6867, 	specificity: 0.9997, 	f1: 0.8141
Test Epoch 75: 100%|██████████| 1768/1768 [00:59<00:00, 29.88it/s, loss=2.71]
Test Epoch 75 ==> 	accuracy: 0.9389, 	precision: 0.9782, 	recall: 0.7175, 	specificity: 0.9959, 	f1: 0.8278
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 76: 100%|██████████| 6507/6507 [08:51<00:00, 12.24it/s, loss=0.0124]
Train Epoch 76 ==> 	accuracy: 0.8469, 	precision: 0.9995, 	recall: 0.6941, 	specificity: 0.9997, 	f1: 0.8193
Test Epoch 76: 100%|██████████| 1768/1768 [01:00<00:00, 29.25it/s, loss=0.239]
Test Epoch 76 ==> 	accuracy: 0.9381, 	precision: 0.9792, 	recall: 0.7127, 	specificity: 0.9961, 	f1: 0.8249
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 77: 100%|██████████| 6507/6507 [08:45<00:00, 12.38it/s, loss=0.0075]
Train Epoch 77 ==> 	accuracy: 0.8470, 	precision: 0.9996, 	recall: 0.6942, 	specificity: 0.9997, 	f1: 0.8194
Test Epoch 77: 100%|██████████| 1768/1768 [00:57<00:00, 30.49it/s, loss=0.103]
Test Epoch 77 ==> 	accuracy: 0.9389, 	precision: 0.9769, 	recall: 0.7185, 	specificity: 0.9956, 	f1: 0.8280
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 78: 100%|██████████| 6507/6507 [08:39<00:00, 12.51it/s, loss=0.0147]
Train Epoch 78 ==> 	accuracy: 0.8492, 	precision: 0.9996, 	recall: 0.6987, 	specificity: 0.9997, 	f1: 0.8225
Test Epoch 78: 100%|██████████| 1768/1768 [01:01<00:00, 28.80it/s, loss=0.171]
Test Epoch 78 ==> 	accuracy: 0.9428, 	precision: 0.9689, 	recall: 0.7443, 	specificity: 0.9939, 	f1: 0.8419
Adjusting learning rate of group 0 to 3.8742e-05.
Train Epoch 79: 100%|██████████| 6507/6507 [08:49<00:00, 12.29it/s, loss=0.0055]
Train Epoch 79 ==> 	accuracy: 0.8458, 	precision: 0.9995, 	recall: 0.6919, 	specificity: 0.9997, 	f1: 0.8178
Test Epoch 79: 100%|██████████| 1768/1768 [00:59<00:00, 29.55it/s, loss=0.114]
Test Epoch 79 ==> 	accuracy: 0.9375, 	precision: 0.9811, 	recall: 0.7080, 	specificity: 0.9965, 	f1: 0.8225
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 80: 100%|██████████| 6507/6507 [08:41<00:00, 12.47it/s, loss=0.0127]
Train Epoch 80 ==> 	accuracy: 0.8478, 	precision: 0.9995, 	recall: 0.6959, 	specificity: 0.9997, 	f1: 0.8206
Test Epoch 80: 100%|██████████| 1768/1768 [00:59<00:00, 29.81it/s, loss=0.29]
Test Epoch 80 ==> 	accuracy: 0.9408, 	precision: 0.9743, 	recall: 0.7300, 	specificity: 0.9950, 	f1: 0.8346
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 81: 100%|██████████| 6507/6507 [08:59<00:00, 12.07it/s, loss=0.0186]
Train Epoch 81 ==> 	accuracy: 0.8484, 	precision: 0.9995, 	recall: 0.6971, 	specificity: 0.9997, 	f1: 0.8213
Test Epoch 81: 100%|██████████| 1768/1768 [01:00<00:00, 29.14it/s, loss=2.34]
Test Epoch 81 ==> 	accuracy: 0.9416, 	precision: 0.9706, 	recall: 0.7368, 	specificity: 0.9943, 	f1: 0.8377
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 82: 100%|██████████| 6507/6507 [08:55<00:00, 12.15it/s, loss=0.0292]
Train Epoch 82 ==> 	accuracy: 0.8491, 	precision: 0.9996, 	recall: 0.6985, 	specificity: 0.9997, 	f1: 0.8224
Test Epoch 82: 100%|██████████| 1768/1768 [01:00<00:00, 29.22it/s, loss=1.84]
Test Epoch 82 ==> 	accuracy: 0.9415, 	precision: 0.9748, 	recall: 0.7328, 	specificity: 0.9951, 	f1: 0.8366
Adjusting learning rate of group 0 to 3.4868e-05.
Train Epoch 83: 100%|██████████| 6507/6507 [08:46<00:00, 12.36it/s, loss=0.0005]
Train Epoch 83 ==> 	accuracy: 0.8498, 	precision: 0.9996, 	recall: 0.6999, 	specificity: 0.9997, 	f1: 0.8233
Test Epoch 83: 100%|██████████| 1768/1768 [01:00<00:00, 29.36it/s, loss=0.197]
Test Epoch 83 ==> 	accuracy: 0.9396, 	precision: 0.9785, 	recall: 0.7208, 	specificity: 0.9959, 	f1: 0.8301
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 84: 100%|██████████| 6507/6507 [08:45<00:00, 12.38it/s, loss=0.0469]
Train Epoch 84 ==> 	accuracy: 0.8512, 	precision: 0.9996, 	recall: 0.7027, 	specificity: 0.9997, 	f1: 0.8252
Test Epoch 84: 100%|██████████| 1768/1768 [01:00<00:00, 29.06it/s, loss=0.182]
Test Epoch 84 ==> 	accuracy: 0.9419, 	precision: 0.9744, 	recall: 0.7354, 	specificity: 0.9950, 	f1: 0.8382
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 85: 100%|██████████| 6507/6507 [08:52<00:00, 12.22it/s, loss=0.0135]
Train Epoch 85 ==> 	accuracy: 0.8532, 	precision: 0.9996, 	recall: 0.7066, 	specificity: 0.9997, 	f1: 0.8280
Test Epoch 85: 100%|██████████| 1768/1768 [00:59<00:00, 29.83it/s, loss=0.186]
Test Epoch 85 ==> 	accuracy: 0.9411, 	precision: 0.9743, 	recall: 0.7315, 	specificity: 0.9950, 	f1: 0.8356
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 86: 100%|██████████| 6507/6507 [09:04<00:00, 11.95it/s, loss=0.0095]
Train Epoch 86 ==> 	accuracy: 0.8536, 	precision: 0.9996, 	recall: 0.7075, 	specificity: 0.9997, 	f1: 0.8285
Test Epoch 86: 100%|██████████| 1768/1768 [01:01<00:00, 28.89it/s, loss=0.116]
Test Epoch 86 ==> 	accuracy: 0.9426, 	precision: 0.9717, 	recall: 0.7408, 	specificity: 0.9944, 	f1: 0.8406
Adjusting learning rate of group 0 to 3.1381e-05.
Train Epoch 87: 100%|██████████| 6507/6507 [08:59<00:00, 12.05it/s, loss=0.0246]
Train Epoch 87 ==> 	accuracy: 0.8520, 	precision: 0.9996, 	recall: 0.7042, 	specificity: 0.9997, 	f1: 0.8263
Test Epoch 87: 100%|██████████| 1768/1768 [01:02<00:00, 28.26it/s, loss=0.0891]
Test Epoch 87 ==> 	accuracy: 0.9413, 	precision: 0.9759, 	recall: 0.7310, 	specificity: 0.9954, 	f1: 0.8359
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 88: 100%|██████████| 6507/6507 [09:07<00:00, 11.88it/s, loss=0.0073]
Train Epoch 88 ==> 	accuracy: 0.8553, 	precision: 0.9996, 	recall: 0.7109, 	specificity: 0.9997, 	f1: 0.8309
Test Epoch 88: 100%|██████████| 1768/1768 [01:00<00:00, 29.01it/s, loss=0.318]
Test Epoch 88 ==> 	accuracy: 0.9442, 	precision: 0.9657, 	recall: 0.7542, 	specificity: 0.9931, 	f1: 0.8470
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 89: 100%|██████████| 6507/6507 [08:54<00:00, 12.17it/s, loss=0.0318]
Train Epoch 89 ==> 	accuracy: 0.8569, 	precision: 0.9996, 	recall: 0.7141, 	specificity: 0.9997, 	f1: 0.8331
Test Epoch 89: 100%|██████████| 1768/1768 [01:02<00:00, 28.42it/s, loss=0.172]
Test Epoch 89 ==> 	accuracy: 0.9433, 	precision: 0.9704, 	recall: 0.7457, 	specificity: 0.9941, 	f1: 0.8434
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 90: 100%|██████████| 6507/6507 [09:01<00:00, 12.01it/s, loss=0.0315]
Train Epoch 90 ==> 	accuracy: 0.8552, 	precision: 0.9996, 	recall: 0.7107, 	specificity: 0.9997, 	f1: 0.8307
Test Epoch 90: 100%|██████████| 1768/1768 [01:02<00:00, 28.20it/s, loss=0.744]
Test Epoch 90 ==> 	accuracy: 0.9435, 	precision: 0.9690, 	recall: 0.7478, 	specificity: 0.9939, 	f1: 0.8442
Adjusting learning rate of group 0 to 2.8243e-05.
Train Epoch 91: 100%|██████████| 6507/6507 [08:55<00:00, 12.16it/s, loss=0]
Train Epoch 91 ==> 	accuracy: 0.8570, 	precision: 0.9996, 	recall: 0.7143, 	specificity: 0.9997, 	f1: 0.8332
Test Epoch 91: 100%|██████████| 1768/1768 [01:01<00:00, 28.61it/s, loss=0.201]
Test Epoch 91 ==> 	accuracy: 0.9437, 	precision: 0.9695, 	recall: 0.7484, 	specificity: 0.9939, 	f1: 0.8447
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 92: 100%|██████████| 6507/6507 [08:57<00:00, 12.12it/s, loss=0.0034]
Train Epoch 92 ==> 	accuracy: 0.8556, 	precision: 0.9996, 	recall: 0.7114, 	specificity: 0.9997, 	f1: 0.8312
Test Epoch 92: 100%|██████████| 1768/1768 [01:01<00:00, 28.87it/s, loss=0.389]
Test Epoch 92 ==> 	accuracy: 0.9420, 	precision: 0.9738, 	recall: 0.7361, 	specificity: 0.9949, 	f1: 0.8384
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 93: 100%|██████████| 6507/6507 [08:51<00:00, 12.25it/s, loss=0.0048]
Train Epoch 93 ==> 	accuracy: 0.8583, 	precision: 0.9996, 	recall: 0.7168, 	specificity: 0.9997, 	f1: 0.8349
Test Epoch 93: 100%|██████████| 1768/1768 [01:01<00:00, 28.79it/s, loss=0.58]
Test Epoch 93 ==> 	accuracy: 0.9444, 	precision: 0.9674, 	recall: 0.7536, 	specificity: 0.9935, 	f1: 0.8472
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 94: 100%|██████████| 6507/6507 [09:07<00:00, 11.88it/s, loss=0.0019]
Train Epoch 94 ==> 	accuracy: 0.8571, 	precision: 0.9996, 	recall: 0.7145, 	specificity: 0.9997, 	f1: 0.8334
Test Epoch 94: 100%|██████████| 1768/1768 [01:01<00:00, 28.65it/s, loss=0.219]
Test Epoch 94 ==> 	accuracy: 0.9412, 	precision: 0.9772, 	recall: 0.7295, 	specificity: 0.9956, 	f1: 0.8354
Adjusting learning rate of group 0 to 2.5419e-05.
Train Epoch 95: 100%|██████████| 6507/6507 [09:05<00:00, 11.92it/s, loss=0.0069]
Train Epoch 95 ==> 	accuracy: 0.8558, 	precision: 0.9996, 	recall: 0.7120, 	specificity: 0.9997, 	f1: 0.8316
Test Epoch 95: 100%|██████████| 1768/1768 [01:01<00:00, 28.91it/s, loss=0.155]
Test Epoch 95 ==> 	accuracy: 0.9416, 	precision: 0.9748, 	recall: 0.7333, 	specificity: 0.9951, 	f1: 0.8369
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 96: 100%|██████████| 6507/6507 [08:56<00:00, 12.13it/s, loss=0.0161]
Train Epoch 96 ==> 	accuracy: 0.8574, 	precision: 0.9996, 	recall: 0.7151, 	specificity: 0.9997, 	f1: 0.8337
Test Epoch 96: 100%|██████████| 1768/1768 [01:01<00:00, 28.98it/s, loss=0.783]
Test Epoch 96 ==> 	accuracy: 0.9432, 	precision: 0.9705, 	recall: 0.7451, 	specificity: 0.9942, 	f1: 0.8430
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 97: 100%|██████████| 6507/6507 [09:02<00:00, 12.00it/s, loss=0.0005]
Train Epoch 97 ==> 	accuracy: 0.8581, 	precision: 0.9996, 	recall: 0.7164, 	specificity: 0.9997, 	f1: 0.8347
Test Epoch 97: 100%|██████████| 1768/1768 [01:01<00:00, 28.80it/s, loss=3.04]
Test Epoch 97 ==> 	accuracy: 0.9427, 	precision: 0.9722, 	recall: 0.7410, 	specificity: 0.9946, 	f1: 0.8410
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 98: 100%|██████████| 6507/6507 [08:46<00:00, 12.37it/s, loss=0.0079]
Train Epoch 98 ==> 	accuracy: 0.8593, 	precision: 0.9996, 	recall: 0.7189, 	specificity: 0.9997, 	f1: 0.8363
Test Epoch 98: 100%|██████████| 1768/1768 [01:01<00:00, 28.78it/s, loss=0.239]
Test Epoch 98 ==> 	accuracy: 0.9444, 	precision: 0.9682, 	recall: 0.7532, 	specificity: 0.9936, 	f1: 0.8472
Adjusting learning rate of group 0 to 2.2877e-05.
Train Epoch 99: 100%|██████████| 6507/6507 [08:50<00:00, 12.27it/s, loss=0.0032]
Train Epoch 99 ==> 	accuracy: 0.8582, 	precision: 0.9996, 	recall: 0.7167, 	specificity: 0.9997, 	f1: 0.8349
Test Epoch 99: 100%|██████████| 1768/1768 [01:00<00:00, 29.44it/s, loss=0.189]
Test Epoch 99 ==> 	accuracy: 0.9418, 	precision: 0.9749, 	recall: 0.7344, 	specificity: 0.9951, 	f1: 0.8377
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 100: 100%|██████████| 6507/6507 [08:47<00:00, 12.33it/s, loss=0.0021]
Train Epoch 100 ==> 	accuracy: 0.8616, 	precision: 0.9997, 	recall: 0.7234, 	specificity: 0.9998, 	f1: 0.8393
Test Epoch 100: 100%|██████████| 1768/1768 [00:59<00:00, 29.53it/s, loss=3.59]
Test Epoch 100 ==> 	accuracy: 0.9448, 	precision: 0.9681, 	recall: 0.7549, 	specificity: 0.9936, 	f1: 0.8484
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 101: 100%|██████████| 6507/6507 [08:48<00:00, 12.31it/s, loss=0.0106]
Train Epoch 101 ==> 	accuracy: 0.8629, 	precision: 0.9996, 	recall: 0.7260, 	specificity: 0.9997, 	f1: 0.8411
Test Epoch 101: 100%|██████████| 1768/1768 [00:59<00:00, 29.47it/s, loss=0.0887]
Test Epoch 101 ==> 	accuracy: 0.9435, 	precision: 0.9708, 	recall: 0.7460, 	specificity: 0.9942, 	f1: 0.8437
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 102: 100%|██████████| 6507/6507 [09:00<00:00, 12.04it/s, loss=0.012]
Train Epoch 102 ==> 	accuracy: 0.8621, 	precision: 0.9996, 	recall: 0.7245, 	specificity: 0.9997, 	f1: 0.8401
Test Epoch 102: 100%|██████████| 1768/1768 [00:58<00:00, 29.98it/s, loss=2.86]
Test Epoch 102 ==> 	accuracy: 0.9448, 	precision: 0.9681, 	recall: 0.7550, 	specificity: 0.9936, 	f1: 0.8483
Adjusting learning rate of group 0 to 2.0589e-05.
Train Epoch 103: 100%|██████████| 6507/6507 [08:47<00:00, 12.33it/s, loss=0.0003]
Train Epoch 103 ==> 	accuracy: 0.8612, 	precision: 0.9996, 	recall: 0.7227, 	specificity: 0.9997, 	f1: 0.8389
Test Epoch 103: 100%|██████████| 1768/1768 [01:01<00:00, 28.68it/s, loss=0.876]
Test Epoch 103 ==> 	accuracy: 0.9439, 	precision: 0.9707, 	recall: 0.7485, 	specificity: 0.9942, 	f1: 0.8452
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 104: 100%|██████████| 6507/6507 [08:57<00:00, 12.12it/s, loss=0.0165]
Train Epoch 104 ==> 	accuracy: 0.8611, 	precision: 0.9996, 	recall: 0.7225, 	specificity: 0.9997, 	f1: 0.8388
Test Epoch 104: 100%|██████████| 1768/1768 [01:00<00:00, 29.41it/s, loss=0.318]
Test Epoch 104 ==> 	accuracy: 0.9437, 	precision: 0.9723, 	recall: 0.7461, 	specificity: 0.9945, 	f1: 0.8443
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 105: 100%|██████████| 6507/6507 [09:03<00:00, 11.96it/s, loss=0.0084]
Train Epoch 105 ==> 	accuracy: 0.8623, 	precision: 0.9996, 	recall: 0.7249, 	specificity: 0.9997, 	f1: 0.8404
Test Epoch 105: 100%|██████████| 1768/1768 [01:02<00:00, 28.23it/s, loss=0.237]
Test Epoch 105 ==> 	accuracy: 0.9449, 	precision: 0.9681, 	recall: 0.7557, 	specificity: 0.9936, 	f1: 0.8488
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 106: 100%|██████████| 6507/6507 [08:57<00:00, 12.11it/s, loss=0.0297]
Train Epoch 106 ==> 	accuracy: 0.8627, 	precision: 0.9996, 	recall: 0.7256, 	specificity: 0.9997, 	f1: 0.8408
Test Epoch 106: 100%|██████████| 1768/1768 [01:00<00:00, 29.28it/s, loss=0.166]
Test Epoch 106 ==> 	accuracy: 0.9447, 	precision: 0.9681, 	recall: 0.7543, 	specificity: 0.9936, 	f1: 0.8479
Adjusting learning rate of group 0 to 1.8530e-05.
Train Epoch 107: 100%|██████████| 6507/6507 [09:05<00:00, 11.93it/s, loss=0.0045]
Train Epoch 107 ==> 	accuracy: 0.8631, 	precision: 0.9996, 	recall: 0.7266, 	specificity: 0.9997, 	f1: 0.8415
Test Epoch 107: 100%|██████████| 1768/1768 [01:04<00:00, 27.25it/s, loss=0.349]
Test Epoch 107 ==> 	accuracy: 0.9447, 	precision: 0.9692, 	recall: 0.7535, 	specificity: 0.9938, 	f1: 0.8478
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 108: 100%|██████████| 6507/6507 [09:05<00:00, 11.93it/s, loss=0.0001]
Train Epoch 108 ==> 	accuracy: 0.8630, 	precision: 0.9996, 	recall: 0.7262, 	specificity: 0.9997, 	f1: 0.8413
Test Epoch 108: 100%|██████████| 1768/1768 [01:00<00:00, 29.23it/s, loss=0.101]
Test Epoch 108 ==> 	accuracy: 0.9449, 	precision: 0.9690, 	recall: 0.7547, 	specificity: 0.9938, 	f1: 0.8485
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 109: 100%|██████████| 6507/6507 [08:59<00:00, 12.05it/s, loss=0.0055]
Train Epoch 109 ==> 	accuracy: 0.8643, 	precision: 0.9996, 	recall: 0.7288, 	specificity: 0.9997, 	f1: 0.8430
Test Epoch 109: 100%|██████████| 1768/1768 [01:00<00:00, 29.40it/s, loss=0.338]
Test Epoch 109 ==> 	accuracy: 0.9459, 	precision: 0.9646, 	recall: 0.7634, 	specificity: 0.9928, 	f1: 0.8523
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 110: 100%|██████████| 6507/6507 [08:53<00:00, 12.20it/s, loss=0.0002]
Train Epoch 110 ==> 	accuracy: 0.8647, 	precision: 0.9996, 	recall: 0.7297, 	specificity: 0.9997, 	f1: 0.8436
Test Epoch 110: 100%|██████████| 1768/1768 [01:02<00:00, 28.51it/s, loss=0.165]
Test Epoch 110 ==> 	accuracy: 0.9457, 	precision: 0.9663, 	recall: 0.7610, 	specificity: 0.9932, 	f1: 0.8514
Adjusting learning rate of group 0 to 1.6677e-05.
Train Epoch 111: 100%|██████████| 6507/6507 [09:02<00:00, 11.99it/s, loss=0.0106]
Train Epoch 111 ==> 	accuracy: 0.8652, 	precision: 0.9996, 	recall: 0.7306, 	specificity: 0.9997, 	f1: 0.8442
Test Epoch 111: 100%|██████████| 1768/1768 [01:03<00:00, 27.77it/s, loss=0.578]
Test Epoch 111 ==> 	accuracy: 0.9444, 	precision: 0.9676, 	recall: 0.7537, 	specificity: 0.9935, 	f1: 0.8473
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 112: 100%|██████████| 6507/6507 [08:47<00:00, 12.33it/s, loss=0.0257]
Train Epoch 112 ==> 	accuracy: 0.8635, 	precision: 0.9996, 	recall: 0.7273, 	specificity: 0.9997, 	f1: 0.8420
Test Epoch 112: 100%|██████████| 1768/1768 [01:01<00:00, 28.86it/s, loss=2.88]
Test Epoch 112 ==> 	accuracy: 0.9439, 	precision: 0.9713, 	recall: 0.7478, 	specificity: 0.9943, 	f1: 0.8450
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 113: 100%|██████████| 6507/6507 [08:56<00:00, 12.12it/s, loss=0.0043]
Train Epoch 113 ==> 	accuracy: 0.8637, 	precision: 0.9996, 	recall: 0.7277, 	specificity: 0.9997, 	f1: 0.8423
Test Epoch 113: 100%|██████████| 1768/1768 [01:00<00:00, 29.02it/s, loss=3.11]
Test Epoch 113 ==> 	accuracy: 0.9440, 	precision: 0.9696, 	recall: 0.7497, 	specificity: 0.9940, 	f1: 0.8456
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 114: 100%|██████████| 6507/6507 [08:45<00:00, 12.38it/s, loss=0.0176]
Train Epoch 114 ==> 	accuracy: 0.8652, 	precision: 0.9997, 	recall: 0.7307, 	specificity: 0.9997, 	f1: 0.8443
Test Epoch 114: 100%|██████████| 1768/1768 [01:02<00:00, 28.27it/s, loss=0.351]
Test Epoch 114 ==> 	accuracy: 0.9465, 	precision: 0.9626, 	recall: 0.7681, 	specificity: 0.9923, 	f1: 0.8544
Adjusting learning rate of group 0 to 1.5009e-05.
Train Epoch 115: 100%|██████████| 6507/6507 [08:46<00:00, 12.35it/s, loss=0.011]
Train Epoch 115 ==> 	accuracy: 0.8676, 	precision: 0.9996, 	recall: 0.7355, 	specificity: 0.9997, 	f1: 0.8474
Test Epoch 115: 100%|██████████| 1768/1768 [00:59<00:00, 29.87it/s, loss=0.316]
Test Epoch 115 ==> 	accuracy: 0.9458, 	precision: 0.9628, 	recall: 0.7644, 	specificity: 0.9924, 	f1: 0.8522
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 116: 100%|██████████| 6507/6507 [08:48<00:00, 12.31it/s, loss=0.0177]
Train Epoch 116 ==> 	accuracy: 0.8650, 	precision: 0.9997, 	recall: 0.7302, 	specificity: 0.9998, 	f1: 0.8440
Test Epoch 116: 100%|██████████| 1768/1768 [00:59<00:00, 29.88it/s, loss=0.827]
Test Epoch 116 ==> 	accuracy: 0.9463, 	precision: 0.9640, 	recall: 0.7659, 	specificity: 0.9926, 	f1: 0.8536
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 117: 100%|██████████| 6507/6507 [08:39<00:00, 12.53it/s, loss=0.0071]
Train Epoch 117 ==> 	accuracy: 0.8675, 	precision: 0.9996, 	recall: 0.7353, 	specificity: 0.9997, 	f1: 0.8474
Test Epoch 117: 100%|██████████| 1768/1768 [00:59<00:00, 29.69it/s, loss=0.417]
Test Epoch 117 ==> 	accuracy: 0.9462, 	precision: 0.9637, 	recall: 0.7658, 	specificity: 0.9926, 	f1: 0.8534
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 118: 100%|██████████| 6507/6507 [08:44<00:00, 12.41it/s, loss=0.0024]
Train Epoch 118 ==> 	accuracy: 0.8658, 	precision: 0.9996, 	recall: 0.7318, 	specificity: 0.9997, 	f1: 0.8450
Test Epoch 118: 100%|██████████| 1768/1768 [01:01<00:00, 28.86it/s, loss=0.186]
Test Epoch 118 ==> 	accuracy: 0.9447, 	precision: 0.9684, 	recall: 0.7543, 	specificity: 0.9937, 	f1: 0.8481
Adjusting learning rate of group 0 to 1.3509e-05.
Train Epoch 119: 100%|██████████| 6507/6507 [08:45<00:00, 12.37it/s, loss=0.0192]
Train Epoch 119 ==> 	accuracy: 0.8659, 	precision: 0.9996, 	recall: 0.7320, 	specificity: 0.9997, 	f1: 0.8451
Test Epoch 119: 100%|██████████| 1768/1768 [00:59<00:00, 29.57it/s, loss=0.265]
Test Epoch 119 ==> 	accuracy: 0.9463, 	precision: 0.9641, 	recall: 0.7662, 	specificity: 0.9927, 	f1: 0.8538
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 120: 100%|██████████| 6507/6507 [08:59<00:00, 12.06it/s, loss=0.0145]
Train Epoch 120 ==> 	accuracy: 0.8653, 	precision: 0.9996, 	recall: 0.7310, 	specificity: 0.9997, 	f1: 0.8444
Test Epoch 120: 100%|██████████| 1768/1768 [01:01<00:00, 28.94it/s, loss=0.443]
Test Epoch 120 ==> 	accuracy: 0.9459, 	precision: 0.9667, 	recall: 0.7615, 	specificity: 0.9933, 	f1: 0.8519
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 121: 100%|██████████| 6507/6507 [09:07<00:00, 11.88it/s, loss=0.0034]
Train Epoch 121 ==> 	accuracy: 0.8672, 	precision: 0.9996, 	recall: 0.7346, 	specificity: 0.9997, 	f1: 0.8469
Test Epoch 121: 100%|██████████| 1768/1768 [01:00<00:00, 29.32it/s, loss=0.262]
Test Epoch 121 ==> 	accuracy: 0.9453, 	precision: 0.9692, 	recall: 0.7565, 	specificity: 0.9938, 	f1: 0.8497
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 122: 100%|██████████| 6507/6507 [08:50<00:00, 12.27it/s, loss=0.0002]
Train Epoch 122 ==> 	accuracy: 0.8685, 	precision: 0.9996, 	recall: 0.7374, 	specificity: 0.9997, 	f1: 0.8487
Test Epoch 122: 100%|██████████| 1768/1768 [00:59<00:00, 29.75it/s, loss=1.56]
Test Epoch 122 ==> 	accuracy: 0.9466, 	precision: 0.9635, 	recall: 0.7678, 	specificity: 0.9925, 	f1: 0.8546
Adjusting learning rate of group 0 to 1.2158e-05.
Train Epoch 123: 100%|██████████| 6507/6507 [08:48<00:00, 12.31it/s, loss=0.0058]
Train Epoch 123 ==> 	accuracy: 0.8667, 	precision: 0.9997, 	recall: 0.7336, 	specificity: 0.9997, 	f1: 0.8462
Test Epoch 123: 100%|██████████| 1768/1768 [01:01<00:00, 28.90it/s, loss=0.155]
Test Epoch 123 ==> 	accuracy: 0.9466, 	precision: 0.9639, 	recall: 0.7676, 	specificity: 0.9926, 	f1: 0.8546
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 124: 100%|██████████| 6507/6507 [08:50<00:00, 12.28it/s, loss=0]
Train Epoch 124 ==> 	accuracy: 0.8689, 	precision: 0.9996, 	recall: 0.7380, 	specificity: 0.9997, 	f1: 0.8491
Test Epoch 124: 100%|██████████| 1768/1768 [01:01<00:00, 28.70it/s, loss=2.05]
Test Epoch 124 ==> 	accuracy: 0.9466, 	precision: 0.9646, 	recall: 0.7673, 	specificity: 0.9928, 	f1: 0.8547
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 125: 100%|██████████| 6507/6507 [08:59<00:00, 12.07it/s, loss=0.0107]
Train Epoch 125 ==> 	accuracy: 0.8678, 	precision: 0.9996, 	recall: 0.7358, 	specificity: 0.9997, 	f1: 0.8477
Test Epoch 125: 100%|██████████| 1768/1768 [01:00<00:00, 29.25it/s, loss=0.104]
Test Epoch 125 ==> 	accuracy: 0.9459, 	precision: 0.9665, 	recall: 0.7619, 	specificity: 0.9932, 	f1: 0.8521
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 126: 100%|██████████| 6507/6507 [08:45<00:00, 12.38it/s, loss=0.0017]
Train Epoch 126 ==> 	accuracy: 0.8686, 	precision: 0.9997, 	recall: 0.7375, 	specificity: 0.9998, 	f1: 0.8488
Test Epoch 126: 100%|██████████| 1768/1768 [01:01<00:00, 28.65it/s, loss=0.0387]
Test Epoch 126 ==> 	accuracy: 0.9470, 	precision: 0.9614, 	recall: 0.7716, 	specificity: 0.9920, 	f1: 0.8561
Adjusting learning rate of group 0 to 1.0942e-05.
Train Epoch 127: 100%|██████████| 6507/6507 [08:51<00:00, 12.25it/s, loss=0.198]
Train Epoch 127 ==> 	accuracy: 0.8686, 	precision: 0.9996, 	recall: 0.7375, 	specificity: 0.9997, 	f1: 0.8488
Test Epoch 127: 100%|██████████| 1768/1768 [00:59<00:00, 29.49it/s, loss=1.36]
Test Epoch 127 ==> 	accuracy: 0.9462, 	precision: 0.9633, 	recall: 0.7663, 	specificity: 0.9925, 	f1: 0.8536
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 128: 100%|██████████| 6507/6507 [08:51<00:00, 12.23it/s, loss=0.0062]
Train Epoch 128 ==> 	accuracy: 0.8687, 	precision: 0.9996, 	recall: 0.7377, 	specificity: 0.9997, 	f1: 0.8489
Test Epoch 128: 100%|██████████| 1768/1768 [00:57<00:00, 30.58it/s, loss=0.145]
Test Epoch 128 ==> 	accuracy: 0.9465, 	precision: 0.9654, 	recall: 0.7661, 	specificity: 0.9929, 	f1: 0.8542
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 129: 100%|██████████| 6507/6507 [08:48<00:00, 12.32it/s, loss=0.0088]
Train Epoch 129 ==> 	accuracy: 0.8665, 	precision: 0.9996, 	recall: 0.7334, 	specificity: 0.9997, 	f1: 0.8460
Test Epoch 129: 100%|██████████| 1768/1768 [01:01<00:00, 28.83it/s, loss=0.656]
Test Epoch 129 ==> 	accuracy: 0.9468, 	precision: 0.9638, 	recall: 0.7686, 	specificity: 0.9926, 	f1: 0.8552
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 130: 100%|██████████| 6507/6507 [08:58<00:00, 12.09it/s, loss=0.0065]
Train Epoch 130 ==> 	accuracy: 0.8678, 	precision: 0.9996, 	recall: 0.7359, 	specificity: 0.9997, 	f1: 0.8477
Test Epoch 130: 100%|██████████| 1768/1768 [01:00<00:00, 29.39it/s, loss=0.39]
Test Epoch 130 ==> 	accuracy: 0.9466, 	precision: 0.9632, 	recall: 0.7686, 	specificity: 0.9924, 	f1: 0.8549
Adjusting learning rate of group 0 to 9.8477e-06.
Train Epoch 131: 100%|██████████| 6507/6507 [08:59<00:00, 12.06it/s, loss=0.0304]
Train Epoch 131 ==> 	accuracy: 0.8680, 	precision: 0.9996, 	recall: 0.7362, 	specificity: 0.9997, 	f1: 0.8479
Test Epoch 131: 100%|██████████| 1768/1768 [00:59<00:00, 29.49it/s, loss=1.97]
Test Epoch 131 ==> 	accuracy: 0.9465, 	precision: 0.9630, 	recall: 0.7682, 	specificity: 0.9924, 	f1: 0.8546
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 132: 100%|██████████| 6507/6507 [08:59<00:00, 12.07it/s, loss=0.0101]
Train Epoch 132 ==> 	accuracy: 0.8701, 	precision: 0.9996, 	recall: 0.7404, 	specificity: 0.9997, 	f1: 0.8507
Test Epoch 132: 100%|██████████| 1768/1768 [00:59<00:00, 29.61it/s, loss=0.108]
Test Epoch 132 ==> 	accuracy: 0.9461, 	precision: 0.9653, 	recall: 0.7640, 	specificity: 0.9929, 	f1: 0.8529
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 133:  46%|████▌     | 2974/6507 [03:49<05:01, 11.73it/s, loss=0.418]
Train Epoch 133: 100%|██████████| 6507/6507 [08:58<00:00, 12.07it/s, loss=0.0001]
Train Epoch 133 ==> 	accuracy: 0.8685, 	precision: 0.9997, 	recall: 0.7372, 	specificity: 0.9998, 	f1: 0.8486
Test Epoch 133: 100%|██████████| 1768/1768 [01:00<00:00, 29.38it/s, loss=0.327]
Test Epoch 133 ==> 	accuracy: 0.9474, 	precision: 0.9614, 	recall: 0.7738, 	specificity: 0.9920, 	f1: 0.8575
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 134: 100%|██████████| 6507/6507 [09:12<00:00, 11.77it/s, loss=0.0016]
Train Epoch 134 ==> 	accuracy: 0.8714, 	precision: 0.9996, 	recall: 0.7431, 	specificity: 0.9997, 	f1: 0.8525
Test Epoch 134: 100%|██████████| 1768/1768 [00:59<00:00, 29.52it/s, loss=0.711]
Test Epoch 134 ==> 	accuracy: 0.9461, 	precision: 0.9659, 	recall: 0.7633, 	specificity: 0.9931, 	f1: 0.8527
Adjusting learning rate of group 0 to 8.8629e-06.
Train Epoch 135: 100%|██████████| 6507/6507 [09:09<00:00, 11.83it/s, loss=0.0034]
Train Epoch 135 ==> 	accuracy: 0.8716, 	precision: 0.9997, 	recall: 0.7434, 	specificity: 0.9998, 	f1: 0.8527
Test Epoch 135: 100%|██████████| 1768/1768 [00:59<00:00, 29.74it/s, loss=0.272]
Test Epoch 135 ==> 	accuracy: 0.9471, 	precision: 0.9626, 	recall: 0.7715, 	specificity: 0.9923, 	f1: 0.8565
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 136: 100%|██████████| 6507/6507 [09:07<00:00, 11.88it/s, loss=0.0257]
Train Epoch 136 ==> 	accuracy: 0.8706, 	precision: 0.9996, 	recall: 0.7415, 	specificity: 0.9997, 	f1: 0.8514
Test Epoch 136: 100%|██████████| 1768/1768 [01:02<00:00, 28.08it/s, loss=4.47]
Test Epoch 136 ==> 	accuracy: 0.9469, 	precision: 0.9621, 	recall: 0.7709, 	specificity: 0.9922, 	f1: 0.8559
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 137: 100%|██████████| 6507/6507 [09:07<00:00, 11.88it/s, loss=0.0061]
Train Epoch 137 ==> 	accuracy: 0.8716, 	precision: 0.9997, 	recall: 0.7434, 	specificity: 0.9998, 	f1: 0.8527
Test Epoch 137: 100%|██████████| 1768/1768 [01:02<00:00, 28.51it/s, loss=0.266]
Test Epoch 137 ==> 	accuracy: 0.9469, 	precision: 0.9629, 	recall: 0.7699, 	specificity: 0.9924, 	f1: 0.8556
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 138: 100%|██████████| 6507/6507 [09:04<00:00, 11.94it/s, loss=0.0106]
Train Epoch 138 ==> 	accuracy: 0.8720, 	precision: 0.9997, 	recall: 0.7443, 	specificity: 0.9997, 	f1: 0.8533
Test Epoch 138: 100%|██████████| 1768/1768 [01:01<00:00, 28.85it/s, loss=0.569]
Test Epoch 138 ==> 	accuracy: 0.9466, 	precision: 0.9625, 	recall: 0.7691, 	specificity: 0.9923, 	f1: 0.8550
Adjusting learning rate of group 0 to 7.9766e-06.
Train Epoch 139: 100%|██████████| 6507/6507 [09:01<00:00, 12.02it/s, loss=0.0002]
Train Epoch 139 ==> 	accuracy: 0.8719, 	precision: 0.9996, 	recall: 0.7441, 	specificity: 0.9997, 	f1: 0.8531
Test Epoch 139: 100%|██████████| 1768/1768 [00:59<00:00, 29.74it/s, loss=0.117]
Test Epoch 139 ==> 	accuracy: 0.9462, 	precision: 0.9652, 	recall: 0.7643, 	specificity: 0.9929, 	f1: 0.8531
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 140: 100%|██████████| 6507/6507 [08:54<00:00, 12.17it/s, loss=0.0407]
Train Epoch 140 ==> 	accuracy: 0.8693, 	precision: 0.9996, 	recall: 0.7389, 	specificity: 0.9997, 	f1: 0.8497
Test Epoch 140: 100%|██████████| 1768/1768 [00:59<00:00, 29.91it/s, loss=0.132]
Test Epoch 140 ==> 	accuracy: 0.9470, 	precision: 0.9636, 	recall: 0.7698, 	specificity: 0.9925, 	f1: 0.8559
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 141: 100%|██████████| 6507/6507 [09:01<00:00, 12.02it/s, loss=0.0001]
Train Epoch 141 ==> 	accuracy: 0.8719, 	precision: 0.9996, 	recall: 0.7441, 	specificity: 0.9997, 	f1: 0.8532
Test Epoch 141: 100%|██████████| 1768/1768 [00:59<00:00, 29.91it/s, loss=0.269]
Test Epoch 141 ==> 	accuracy: 0.9478, 	precision: 0.9596, 	recall: 0.7776, 	specificity: 0.9916, 	f1: 0.8591
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 142: 100%|██████████| 6507/6507 [08:58<00:00, 12.09it/s, loss=0.0181]
Train Epoch 142 ==> 	accuracy: 0.8704, 	precision: 0.9997, 	recall: 0.7411, 	specificity: 0.9998, 	f1: 0.8512
Test Epoch 142: 100%|██████████| 1768/1768 [01:00<00:00, 29.18it/s, loss=0.0957]
Test Epoch 142 ==> 	accuracy: 0.9467, 	precision: 0.9647, 	recall: 0.7674, 	specificity: 0.9928, 	f1: 0.8548
Adjusting learning rate of group 0 to 7.1790e-06.
Train Epoch 143: 100%|██████████| 6507/6507 [08:53<00:00, 12.20it/s, loss=0.0018]
Train Epoch 143 ==> 	accuracy: 0.8735, 	precision: 0.9997, 	recall: 0.7473, 	specificity: 0.9998, 	f1: 0.8553
Test Epoch 143: 100%|██████████| 1768/1768 [00:58<00:00, 30.13it/s, loss=0.607]
Test Epoch 143 ==> 	accuracy: 0.9469, 	precision: 0.9600, 	recall: 0.7727, 	specificity: 0.9917, 	f1: 0.8562
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 144: 100%|██████████| 6507/6507 [08:52<00:00, 12.23it/s, loss=0.008]
Train Epoch 144 ==> 	accuracy: 0.8713, 	precision: 0.9997, 	recall: 0.7429, 	specificity: 0.9997, 	f1: 0.8524
Test Epoch 144: 100%|██████████| 1768/1768 [01:00<00:00, 29.18it/s, loss=3.45]
Test Epoch 144 ==> 	accuracy: 0.9483, 	precision: 0.9575, 	recall: 0.7820, 	specificity: 0.9911, 	f1: 0.8609
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 145: 100%|██████████| 6507/6507 [08:49<00:00, 12.29it/s, loss=0.004]
Train Epoch 145 ==> 	accuracy: 0.8705, 	precision: 0.9996, 	recall: 0.7413, 	specificity: 0.9997, 	f1: 0.8513
Test Epoch 145: 100%|██████████| 1768/1768 [01:00<00:00, 29.07it/s, loss=0.409]
Test Epoch 145 ==> 	accuracy: 0.9470, 	precision: 0.9621, 	recall: 0.7714, 	specificity: 0.9922, 	f1: 0.8563
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 146: 100%|██████████| 6507/6507 [08:51<00:00, 12.24it/s, loss=0.0126]
Train Epoch 146 ==> 	accuracy: 0.8720, 	precision: 0.9996, 	recall: 0.7442, 	specificity: 0.9997, 	f1: 0.8532
Test Epoch 146: 100%|██████████| 1768/1768 [00:59<00:00, 29.86it/s, loss=0.0696]
Test Epoch 146 ==> 	accuracy: 0.9464, 	precision: 0.9637, 	recall: 0.7668, 	specificity: 0.9926, 	f1: 0.8540
Adjusting learning rate of group 0 to 6.4611e-06.
Train Epoch 147: 100%|██████████| 6507/6507 [08:44<00:00, 12.40it/s, loss=0.0005]
Train Epoch 147 ==> 	accuracy: 0.8713, 	precision: 0.9996, 	recall: 0.7429, 	specificity: 0.9997, 	f1: 0.8524
Test Epoch 147: 100%|██████████| 1768/1768 [00:59<00:00, 29.68it/s, loss=0.152]
Test Epoch 147 ==> 	accuracy: 0.9468, 	precision: 0.9630, 	recall: 0.7697, 	specificity: 0.9924, 	f1: 0.8556
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 148: 100%|██████████| 6507/6507 [08:50<00:00, 12.26it/s, loss=0.0039]
Train Epoch 148 ==> 	accuracy: 0.8748, 	precision: 0.9997, 	recall: 0.7498, 	specificity: 0.9998, 	f1: 0.8569
Test Epoch 148: 100%|██████████| 1768/1768 [01:02<00:00, 28.20it/s, loss=0.607]
Test Epoch 148 ==> 	accuracy: 0.9476, 	precision: 0.9593, 	recall: 0.7769, 	specificity: 0.9915, 	f1: 0.8585
Adjusting learning rate of group 0 to 5.8150e-06.
Train Epoch 149: 100%|██████████| 6507/6507 [08:44<00:00, 12.40it/s, loss=0.0058]
Train Epoch 149 ==> 	accuracy: 0.8726, 	precision: 0.9997, 	recall: 0.7454, 	specificity: 0.9997, 	f1: 0.8540
Test Epoch 149: 100%|██████████| 1768/1768 [01:02<00:00, 28.35it/s, loss=1.32]
Test Epoch 149 ==> 	accuracy: 0.9477, 	precision: 0.9615, 	recall: 0.7755, 	specificity: 0.9920, 	f1: 0.8585
Adjusting learning rate of group 0 to 5.8150e-06.

进程已结束，退出代码为 0

'''