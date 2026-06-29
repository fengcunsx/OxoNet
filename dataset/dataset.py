import os
import random

from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
import torch

from dataset.generate_paire import generate_pair, combine_data

BASE_DICT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
DEBASE_DICT = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


class PairDataset(Dataset):
    def __init__(self, pos_file, neg_file):
        super().__init__()
        # 只在主进程加载
        pos_data = np.load(pos_file, allow_pickle=False, mmap_mode=None)
        neg_data = np.load(neg_file, allow_pickle=False, mmap_mode=None)

        # 转 torch tensor 方便后续
        self.pos_signal = torch.from_numpy(pos_data["signal"]).float()
        self.pos_mean = torch.from_numpy(pos_data["mean"]).float()
        self.pos_std = torch.from_numpy(pos_data["std"]).float()
        self.pos_dwell = torch.from_numpy(pos_data["dwell"]).float()
        self.pos_kmer = self.encode_kmers(pos_data["kmers"])
        self.pos_readid = pos_data["read_id"]
        self.pos_pos = pos_data["basecall_pos"]

        self.neg_signal = torch.from_numpy(neg_data["signal"]).float()
        self.neg_mean = torch.from_numpy(neg_data["mean"]).float()
        self.neg_std = torch.from_numpy(neg_data["std"]).float()
        self.neg_dwell = torch.from_numpy(neg_data["dwell"]).float()
        self.neg_kmer = self.encode_kmers(neg_data["kmers"])
        self.neg_readid = neg_data["read_id"]
        self.neg_pos = neg_data["basecall_pos"]

        self.n = self.pos_signal.shape[0]

    def encode_kmers(self, kmer_array):
        return np.stack([
            np.fromiter((BASE_DICT[ch] for ch in s), dtype=np.int64, count=len(s))
            for s in kmer_array
        ])

    def __len__(self):
        return self.n * 2

    def __getitem__(self, idx):
        real_idx = idx // 2
        is_pos = (idx % 2 == 0)
        if is_pos:
            sig_l = self.pos_dwell[real_idx].sum().item()
            return {
                "signal": self.pos_signal[real_idx].reshape(-1, 1),
                "kmer": torch.tensor(self.pos_kmer[real_idx], dtype=torch.long),
                "mean": self.pos_mean[real_idx].reshape(-1, 1),
                "std": self.pos_std[real_idx].reshape(-1, 1),
                "dwell": self.pos_dwell[real_idx].reshape(-1, 1),
                "sig_l": torch.tensor(sig_l),
                "read_id": self.pos_readid[real_idx],
                "position": self.pos_pos[real_idx],
                "label": True,
            }
        else:
            sig_l = self.neg_dwell[real_idx].sum().item()
            return {
                "signal": self.neg_signal[real_idx].reshape(-1, 1),
                "kmer": torch.tensor(self.neg_kmer[real_idx], dtype=torch.long),
                "mean": self.neg_mean[real_idx].reshape(-1, 1),
                "std": self.neg_std[real_idx].reshape(-1, 1),
                "dwell": self.neg_dwell[real_idx].reshape(-1, 1),
                "sig_l": torch.tensor(sig_l),
                "read_id": self.neg_readid[real_idx],
                "position": self.neg_pos[real_idx],
                "label": False,
            }


def collate_fn(batch):
    return {
        "signals": torch.stack([b["signal"] for b in batch]),
        "kmers": torch.stack([b["kmer"] for b in batch]),
        "mean": torch.stack([b["mean"] for b in batch]),
        "std": torch.stack([b["std"] for b in batch]),
        "dwell": torch.stack([b["dwell"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch]),  # 用 7mer id 作为类别
        'sig_l': torch.stack([b['sig_l'] for b in batch]),
        "read_id": [b["read_id"] for b in batch],
        "position": [b["position"] for b in batch],
    }


class MultiSimSampler(Sampler):
    def __init__(self, n, n_sample, shuffle=True, seed=42):
        super().__init__()
        # data must be a pair
        self.n = n
        assert n_sample % 2 == 0
        self.batch_size = n_sample // 2
        self.shuffle = shuffle
        self.epoch = 0
        # total data count is 2n
        self.all_classes = list(range(n))
        self.seed = seed

    def set_epoch(self, epoch):  # ← 加这个方法
        self.epoch = epoch

    def __iter__(self):
        classes = self.all_classes[:]
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)  # ← 用独立的 Random 对象
            rng.shuffle(classes)
        # if self.shuffle:
        #     random.shuffle(classes)

        for i in range(0, len(classes), self.batch_size):
            selected = classes[i:min(i + self.batch_size, len(classes))]
            for c in selected:
                yield c * 2
                yield c * 2 + 1

    def __len__(self):
        return self.n * 2


def get_dataset(pos_dir, neg_dir, save_dir, epoch, kmer=5, n_threads=8, generate=False):
    if generate:
        generate_pair(pos_dir, neg_dir, save_dir, kmer, epoch=epoch, n_threads=n_threads)
    pos_file = os.path.join(save_dir, "pos.npz")
    neg_file = os.path.join(save_dir, "neg.npz")
    return PairDataset(pos_file, neg_file)


class TestDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        # 只在主进程加载
        data = np.load(file, allow_pickle=False, mmap_mode=None)

        # 转 torch tensor 方便后续
        self.signal = torch.from_numpy(data["signal"]).float()
        self.mean = torch.from_numpy(data["mean"]).float()
        self.std = torch.from_numpy(data["std"]).float()
        self.dwell = torch.from_numpy(data["dwell"]).float()
        self.kmer = self.encode_kmers(data["kmers"])
        self.readid = data["read_id"]
        self.pos = data["basecall_pos"]
        self.label = data['label']

        self.n = self.signal.shape[0]

    def encode_kmers(self, kmer_array):
        return np.stack([
            np.fromiter((BASE_DICT[ch] for ch in s), dtype=np.int64, count=len(s))
            for s in kmer_array
        ])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sig_l = self.dwell[idx].sum().item()
        return {
            "signal": self.signal[idx].reshape(-1, 1),
            "kmer": torch.tensor(self.kmer[idx], dtype=torch.long),
            "mean": self.mean[idx].reshape(-1, 1),
            "std": self.std[idx].reshape(-1, 1),
            "dwell": self.dwell[idx].reshape(-1, 1),
            "sig_l": torch.tensor(sig_l),
            "read_id": self.readid[idx],
            "position": self.pos[idx],
            "label": self.label[idx],
        }


def get_test_dataset(pos_dir, neg_dir, save_dir, generate=False):
    file_pth = os.path.join(save_dir, 'test.npz')
    if generate or os.path.exists(file_pth):
        os.makedirs(save_dir, exist_ok=True)
        combine_data(pos_dir, neg_dir, file_pth)
    return TestDataset(file_pth)


if __name__ == "__main__":
    # pos_file = '../exp/samples/pos.npz'
    # neg_file = '../exp/samples/neg.npz'
    # batch_size = 32
    # dataset = PairDataset(pos_file, neg_file)
    # dataloader = DataLoader(dataset, batch_size=batch_size,
    #                         sampler=MultiSimSampler(len(dataset) // 2, n_sample=batch_size, shuffle=True),
    #                         collate_fn=collate_fn, num_workers=8)
    # print(len(dataloader))
    # for batch in dataloader:
    #     print(type(batch))

    pos_dir = '/home/bio/8oxog/data/feature/8oxog_test'
    neg_dir = '/home/bio/8oxog/data/feature/g_test'
    save_dir = '/home/bio/8oxog/data/feature'
    get_test_dataset(pos_dir, neg_dir, save_dir, generate=True)
