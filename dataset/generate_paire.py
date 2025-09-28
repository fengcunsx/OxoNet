import argparse
import os
import shutil
import tempfile

from concurrent.futures import ThreadPoolExecutor
import numpy as np


def process_file(file, pos_dir, neg_dir, pos_workspace, neg_workspace, kmer, generate_pos=False, epoch=0):
    all_pos = np.load(os.path.join(pos_dir, file))
    keys = list(all_pos.files)
    l = all_pos[keys[0]].shape[0]
    if l < 2:
        print(f"Warning: {file} has less than 2 samples, skipping")
        return

    # 找到对应的负样本文件
    half_window = kmer // 2
    cur_neg_file = file[:half_window] + 'G' + file[half_window + 1:]
    # neg = pd.read_csv(os.path.join(neg_dir, cur_neg_file), sep='\t', header=None)
    neg = np.load(os.path.join(neg_dir, cur_neg_file))

    # idx
    neg_l = neg[keys[0]].shape[0]
    sample_start = l * epoch % neg_l
    idx = [i % neg_l for i in range(sample_start, sample_start + l)]

    # neg = neg.iloc[idx, :]
    # neg = neg.sample(frac=1).reset_index(drop=True)
    # neg = neg.sample(n=min(all_pos.shape[0], len(neg))).reset_index(drop=True)
    sampled = dict()
    for key in keys:
        sampled[key] = neg[key][idx]
    # # 多线程安全写入
    # with pos_lock:
    #     all_pos.to_csv(pos_file, sep='\t', mode='a', header=False, index=False)
    # with neg_lock:
    #     neg.to_csv(neg_file, sep='\t', mode='a', header=False, index=False)
    filename = file.split('.')[0]
    if generate_pos:
        np.savez_compressed(pos_workspace + f'/{filename}', **all_pos)
    np.savez_compressed(neg_workspace + f'/{filename}', **sampled)


def combine_part(dir, target):
    files = os.listdir(dir)
    all_data = dict()
    for file in files:
        if file.endswith('.npz'):
            sub = np.load(os.path.join(dir, file))
            for key in sub.files:
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(sub[key])

    for key, v in all_data.items():
        all_data[key] = np.concatenate(v, axis=0)

    np.savez_compressed(target, **all_data)
    # for file in files:
    #     if file.endswith('.npz'):
    #         os.remove(os.path.join(dir, file))


def generate_pair(pos_dir, neg_dir, save_dir, kmer, epoch=0, n_threads=8):
    workdir = tempfile.mkdtemp()
    os.makedirs(save_dir, exist_ok=True)
    pos_workspace = os.path.join(workdir, 'pos')
    neg_workspace = os.path.join(workdir, 'neg')
    os.makedirs(pos_workspace, exist_ok=True)
    os.makedirs(neg_workspace, exist_ok=True)

    pos_file = os.path.join(save_dir, 'pos.npz')
    neg_file = os.path.join(save_dir, 'neg.npz')
    all_pos_exist = os.path.exists(pos_file)
    try:

        files = os.listdir(pos_dir)

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for file in files:
                futures.append(
                    executor.submit(process_file, file, pos_dir, neg_dir, pos_workspace, neg_workspace, kmer,
                                    not all_pos_exist,
                                    epoch)
                )
            # 等待所有线程完成
            for f in futures:
                f.result()

        combine_part(neg_workspace, neg_file)
        if not all_pos_exist:
            combine_part(pos_workspace, pos_file)
    finally:
        shutil.rmtree(pos_workspace)
        shutil.rmtree(neg_workspace)


def combine_data(pos_dir, neg_dir, save_file):
    files = []
    for file in os.listdir(pos_dir):
        files.append(os.path.join(pos_dir, file))
        files.append(os.path.join(neg_dir, file))

    data = np.load(files[0])
    keys = data.files

    merged = {k: [] for k in keys}

    # 依次读取并存储
    for f in files:
        npz = np.load(f)
        for k in keys:
            merged[k].append(npz[k])

    # 拼接并保存
    merged = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
    np.savez(save_file, **merged)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_dir', type=str, default='../exp/split/8oxog_train')
    parser.add_argument('--neg_dir', type=str, default='../exp/split/g_train')
    parser.add_argument('--save_dir', type=str, default='../exp/samples')
    parser.add_argument('--n_threads', type=int, default=8)
    parser.add_argument('--kmer', type=int, default=5)
    args = parser.parse_args()

    generate_pair(args.pos_dir, args.neg_dir, args.save_dir, args.kmer, n_threads=args.n_threads)
