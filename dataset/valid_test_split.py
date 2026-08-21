import os
import numpy as np


def split_npz_by_attr(dir_path: str, attr_key: str, save_dir: str, random_state=66):
    """
    将 dir_path 下的每个 npz 文件，按 attr_key 对应的属性分组，
    每个取值组内随机划分 50% 为 test，其余为 valid，
    并保存到 save_dir 下。

    参数：
        dir_path: 原始 npz 文件目录
        attr_key: 属性字段名，例如 'label' 或 'type'
        save_dir: 输出目录
    """
    np.random.seed(random_state)
    valid_dir = os.path.join(save_dir, 'valid')
    test_dir = os.path.join(save_dir, 'test')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    npz_files = [f for f in os.listdir(dir_path) if f.endswith('.npz')]

    for fname in npz_files:
        path = os.path.join(dir_path, fname)
        data = np.load(path, allow_pickle=True)

        if attr_key not in data:
            print(f"[WARN] 文件 {fname} 中未找到属性 {attr_key}，跳过。")
            continue

        attr = data[attr_key]
        unique_values = np.unique(attr)

        test_indices = []
        valid_indices = []

        for val in unique_values:
            idx = np.where(attr == val)[0]
            np.random.shuffle(idx)

            split_point = len(idx) // 2
            valid_indices.extend(idx[:split_point])
            test_indices.extend(idx[split_point:])

        test_indices = np.array(test_indices)
        valid_indices = np.array(valid_indices)

        # 构造划分后的字典
        test_dict = {k: v[test_indices] for k, v in data.items()}
        valid_dict = {k: v[valid_indices] for k, v in data.items()}

        # 保存文件
        base = os.path.splitext(fname)[0]
        np.savez_compressed(os.path.join(test_dir, f"{base}.npz"), **test_dict)
        np.savez_compressed(os.path.join(valid_dir, f"{base}.npz"), **valid_dict)

        print(f"[OK] 文件 {fname} 已划分并保存至 {save_dir}")


if __name__ == '__main__':
    dir_path = '/home/bio/8oxog/data/7mer_feature/8oxog_test'
    save_dir = '/home/bio/8oxog/data/7mer_feature/8oxog_valid_test'
    split_npz_by_attr(dir_path, 'kmers', save_dir)

# 示例调用
# split_npz_by_attr("/path/to/your/dir", attr_key="label", save_dir="/path/to/output")
