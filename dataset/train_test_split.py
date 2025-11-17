import argparse
import os
import shutil

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def parse_esox(info: str):
    arr = np.fromstring(info, sep='*')
    return arr


ENCODING_DICT_CRF = {
    "_": 0,
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "o": 5,
    "a": 1,
    "c": 2,
    "g": 3,
    "t": 4,
}


def parse_seq(seq: str):
    arr = np.array(seq.split('*'))
    return np.vectorize(ENCODING_DICT_CRF.get)(arr)


def dataframe2npz_esox(data: pd.DataFrame):
    return {
        'x1': np.array(data['x1'].apply(parse_esox).tolist()),
        'e1': np.array(data['e1'].apply(parse_esox).tolist()),
        's1': np.array(data['s1'].apply(parse_seq).tolist()),
        'read_id': np.array(data['read_id'].tolist()),
        'label': data['label'].values,
        'basecall_pos': data[['position']].values,
        'kmers': np.array(data['kmer'].tolist())
    }


def dataframe2npz(df: pd.DataFrame, windows=5, sig_len=125):
    half_window = windows // 2

    def parse_raw(info: str):
        l = sig_len
        arr = np.fromstring(info, sep='*')

        if len(arr) < l:
            return np.pad(arr, (0, l - len(arr)), mode='constant', constant_values=0)
        else:
            return arr[:l]

    signal = np.array(df['signal'].apply(parse_raw).tolist())
    label = df['label'].values
    read_id = np.array(df['read_id'].tolist())
    kmers = np.array(df['kmer'].tolist())
    mean = df[['mean_{}'.format(i - half_window) for i in range(windows)]].values
    std = df[['std_{}'.format(i - half_window) for i in range(windows)]].values
    dwell = df[['dwell_{}'.format(i - half_window) for i in range(windows)]].values
    basecall_pos = df[['position']].values
    return {
        'signal': signal,
        'label': label,
        'read_id': read_id,
        'kmers': kmers,
        'mean': mean,
        'std': std,
        'dwell': dwell,
        'basecall_pos': basecall_pos
    }


def split_dataset(data_pth, windows=5, sig_len=125, test_size=0.1, random_state=66):
    data = pd.read_csv(data_pth, sep='\t', header=None)
    half_window = windows // 2
    head = ['kmer'] + ['mean_{}'.format(i - half_window) for i in range(windows)] \
           + ['std_{}'.format(i - half_window) for i in range(windows)] \
           + ['dwell_{}'.format(i - half_window) for i in range(windows)] + ['signal', 'label', 'read_id', 'position']
    data.columns = head
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    test_npz = dataframe2npz(test, windows=windows, sig_len=sig_len)
    train_npz = dataframe2npz(train, windows=windows, sig_len=sig_len)
    return train_npz, test_npz


def split_dataset_esox(data_pth, test_size=0.1, random_state=66):
    data = pd.read_csv(data_pth, sep='\t', header=None)
    head = ['kmer', 'read_id', 'position', 'x1', 'e1', 's1', 'label']
    data.columns = head
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    test_npz = dataframe2npz_esox(test)
    train_npz = dataframe2npz_esox(train)
    return train_npz, test_npz


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sig_len', type=int, default=175)
    parser.add_argument('--window_size', type=int, default=7)
    args = parser.parse_args()

    # file = '../exp/data/data.tsv'
    # save_file = '../exp/data/info'
    # train, test = split_dataset(file, args.window_size, args.sig_len)
    # np.savez_compressed(save_file + 'train.npz', **train)
    #
    # file = '../exp/data/data_esox.tsv'
    # save_file = '../exp/data/info_esox'
    # train, test = split_dataset_esox(file)
    # np.savez_compressed(save_file + '_train.npz', **train)

    data_dir = '/home/bio/8oxog/data/batch4/feature'
    save_dir = '/home/bio/8oxog/data/batch4/split'
    if os.path.exists(save_dir):
        print('Directory already exists,dataset splitted')
        exit(1)
    os.makedirs(save_dir, exist_ok=True)
    pos_dir = os.path.join(data_dir, '8oxog')
    neg_dir = os.path.join(data_dir, 'g')

    # save_pth
    train_pos_save_path = os.path.join(save_dir, '8oxog_train')
    test_pos_save_path = os.path.join(save_dir, '8oxog_test')

    train_neg_save_path = os.path.join(save_dir, 'g_train')
    test_neg_save_path = os.path.join(save_dir, 'g_test')

    os.makedirs(train_pos_save_path, exist_ok=True)
    os.makedirs(train_neg_save_path, exist_ok=True)
    os.makedirs(test_pos_save_path, exist_ok=True)
    os.makedirs(test_neg_save_path, exist_ok=True)

    # for file in os.listdir(pos_dir):
    #     file_name = file.split('.')[0] + '.npz'
    #     file_path = os.path.join(pos_dir, file)
    #     train, test = split_dataset(file_path, args.window_size, args.sig_len)
    #     np.savez(os.path.join(train_pos_save_path, file_name), **train)
    #     np.savez(os.path.join(test_pos_save_path, file_name), **test)
    #
    #     file_path = os.path.join(neg_dir, file)
    #     train, test = split_dataset(file_path, args.window_size, args.sig_len)
    #     np.savez(os.path.join(train_neg_save_path, file_name), **train)
    #     np.savez(os.path.join(test_neg_save_path, file_name), **test)

    for file in os.listdir(pos_dir):
        file_name = file.split('.')[0] + '.npz'
        file_path = os.path.join(pos_dir, file)
        train, test = split_dataset(file_path, windows=args.window_size, sig_len=args.sig_len)
        np.savez(os.path.join(train_pos_save_path, file_name), **train)
        np.savez(os.path.join(test_pos_save_path, file_name), **test)

        file_path = os.path.join(neg_dir, file)
        train, test = split_dataset(file_path, windows=args.window_size, sig_len=args.sig_len)
        np.savez(os.path.join(train_neg_save_path, file_name), **train)
        np.savez(os.path.join(test_neg_save_path, file_name), **test)

    print('success')
