import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import tqdm
from dataset.dataset import PairDataset, MultiSimSampler, collate_fn, get_dataset, get_test_dataset, DEBASE_DICT
from model.loss import FocalLoss
from model.model import DetectModel
from pytorch_metric_learning import losses, miners, samplers
from torch import nn
import pandas as pd


def find_best_threshold(df, spec_target=0.999, start=0.7):
    """
    从指定起点 (start) 开始遍历阈值，找出 spec>=spec_target 且 recall 最大的阈值。
    """
    labels = df['label'].values
    probs = df['prob'].values

    # 取从start开始的阈值集合
    thresholds = np.sort(np.unique(probs))
    thresholds = thresholds[thresholds >= start]

    best_t, best_recall, best_spec = start, 0, 0

    for t in thresholds:
        pred = (probs >= t).astype(int)
        TP = np.sum((pred == 1) & (labels == 1))
        TN = np.sum((pred == 0) & (labels == 0))
        FP = np.sum((pred == 1) & (labels == 0))
        FN = np.sum((pred == 0) & (labels == 1))

        spec = TN / (TN + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)

        if spec >= spec_target and recall > best_recall:
            best_t, best_recall, best_spec = t, recall, spec
    return best_t, best_recall, best_spec
    # return {"threshold": best_t, "recall": best_recall, "specificity": best_spec}


# 定义颜色常量
BLUE = '\033[94m'
GREEN = '\033[92m'


def parse_data(batch: dict, device, kmers=5):
    signal = batch['signals'].to(device)
    original_kmer = batch['kmers'].to(device)
    kmer_len = original_kmer[0].shape[0] - kmers
    kmer = original_kmer[:, kmer_len // 2: -kmer_len // 2] if kmer_len > 0 else original_kmer
    original_kmer = [''.join(DEBASE_DICT.get(base, 'N') for base in seq) for seq in original_kmer.tolist()]
    mean = batch['mean']
    std = batch['std']
    dwell = batch['dwell']
    label = batch['labels'].to(device)
    sig_l = batch['sig_l'].to(device)
    read_id = batch['read_id']
    position = batch['position']
    return signal, kmer, mean, std, dwell, sig_l, label, read_id, position, original_kmer


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


def prediction(args: argparse.Namespace, resume, generate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "")
    model = DetectModel(sig_blocks=4, sig_l=175, seq_l=5).to(device)
    if args.resume is not None:
        model.load_state_dict(torch.load(resume, map_location=device),strict=False)
    # os.makedirs(args.save_dir, exist_ok=True)
    model.eval()
    test_dataset = get_test_dataset(args.test_pos_dir, args.test_neg_dir, args.test_work_dir,
                                    generate)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers)
    pbar = tqdm.tqdm(total=len(test_dataloader), desc="Test :")
    info = {
        'read_id': [],
        'position': [],
        'kmer_seq': [],
        'label': [],
        'prob': [],
    }
    info_static = {
        'mean': [],
        'std': [],
        'dwell': [],
    }
    embeds = []

    TP_total = TN_total = FP_total = FN_total = 0
    for batch in test_dataloader:
        signal, kmer, mean, std, dwell, sig_l, label, read_id, position, original_kmer = parse_data(batch, device)
        embed, pred = model(signal, kmer, sig_l)
        embeds.append(embed.cpu().detach().numpy())
        TP, TN, FP, FN = calculate_confusion(pred.squeeze(-1), label, args.threshold)
        TP_total += TP
        TN_total += TN
        FP_total += FP
        FN_total += FN
        pbar.update(1)
        info["read_id"].extend(read_id)
        info['kmer_seq'].extend(original_kmer)
        info["position"].extend(np.concatenate(position).tolist())
        info["label"].extend(label.cpu().detach().tolist())
        info["prob"].extend(pred.squeeze(-1).cpu().detach().numpy().tolist())
        info_static["mean"].append(pd.DataFrame(mean.squeeze(-1).cpu().detach().numpy()))
        info_static["std"].append(pd.DataFrame(std.squeeze(-1).cpu().detach().numpy()))
        info_static["dwell"].append(pd.DataFrame(dwell.squeeze(-1).cpu().detach().numpy()))

    metrics = compute_metrics_from_confusion(TP_total, TN_total, FP_total, FN_total)
    pbar.close()
    print(GREEN + 'Test ==> \t'
                  'accuracy: {:.4f}, \t'
                  'precision: {:.4f}, \t'
                  'recall: {:.4f}, \t'
                  'specificity: {:.4f}, \t'
                  'f1: {:.4f}'.format(
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['specificity'],
        metrics['f1']

    ))
    info_df = pd.DataFrame(info)
    info_static_df = dict()
    static = []
    embeds = pd.DataFrame(np.concatenate(embeds, axis=0))
    embeds.columns = ["feature_{}".format(i) for i in range(embeds.shape[1])]
    for key, v in info_static.items():
        seq_l = v[0].shape[1]
        info_static_df[key] = pd.concat(info_static[key])
        info_static_df[key].columns = [key + '_{}'.format(i - seq_l // 2) for i in range(seq_l)]
        static.append(info_static_df[key])
    # df.append(info_df)
    static = pd.concat(static, axis=1).reset_index(drop=True)
    df = pd.concat([info_df, static, embeds], axis=1)
    return find_best_threshold(df)


def main(args):
    threshold = []
    recall = []
    spec = []
    epoch = []
    with torch.no_grad():
        for i in range(30):
            resume = os.path.join(args.resume, 'model_{}.pth'.format(i + 120))
            info = prediction(args, resume, True if i == 0 else False)
            epoch.append(i + 120)
            threshold.append(info[0])
            recall.append(info[1])
            spec.append(info[2])

    df = pd.DataFrame({'epoch': epoch, "threshold": threshold, 'recall': recall, 'specific': spec})
    df.to_csv(args.save_dir, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str,
                        default='../random42_ab_seq')
    parser.add_argument('--generate', default=True)
    parser.add_argument('--test-pos-dir', type=str, default='/home/bio/8oxog/data/feature/8oxog_valid_test/valid')
    parser.add_argument('--test-neg-dir', type=str, default='/home/bio/8oxog/data/feature/g_valid_test/valid')
    # parser.add_argument('--test-pos-dir', type=str, default='/home/bio/8oxog/data/7mer_feature/8oxog_train')
    # parser.add_argument('--test-neg-dir', type=str, default='/home/bio/8oxog/data/7mer_feature/g_train')
    parser.add_argument('--test-work-dir', type=str, default='/home/bio/8oxog/data/feature/test_workspace')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--save-dir', type=str, default='../random42_ab_seq/7mer.csv')
    args = parser.parse_args()
    main(args)
