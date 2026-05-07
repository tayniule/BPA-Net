import os
import pickle
import numpy as np
from load_data_test import preprocess_subject_data, calculate_subject_correlations_all_samples, stepwise_clean


def get_73_subject_split(base_path, train_ratio=0.7, val_ratio=0.1):
    # 1. 扫描实际存在的 73 名被试
    available_subs = []
    sub_avg_features = []

    print("正在扫描被试文件夹并提取特征...")
    for sub_id in range(1, 81):
        sub_dir = os.path.join(base_path, str(sub_id))
        if not os.path.exists(sub_dir):
            continue

        available_subs.append(sub_id)
        trial_feats = []
        # 读取每个 Trial 的 pkl 提取 PSD 特征
        for pkl_file in os.listdir(sub_dir):
            if not pkl_file.endswith('.pkl'): continue
            with open(os.path.join(sub_dir, pkl_file), 'rb') as f:
                content = pickle.load(f)
                # 使用 preprocess_subject_data 计算 PSD
                # 假设 content['data'] 维度符合要求
                f_psd = preprocess_subject_data(np.expand_dims(content['data'], axis=0))
                trial_feats.append(np.mean(f_psd, axis=0))

        sub_avg_features.append(np.mean(trial_feats, axis=0))

    # 2. 计算 73x73 相关性矩阵
    # 注意：需要将被试特征列表包装成 calculate_subject_correlations_all_samples 预期的格式
    corr_matrix = calculate_subject_correlations_all_samples(sub_avg_features)

    # 3. 层次化分组
    range_groups = {"≥0.7": [], "0.5~0.7": [], "0.3~0.5": [], "<0.3": []}
    n_subs = len(available_subs)

    for i in range(n_subs):
        non_diag = [j for j in range(n_subs) if j != i]
        max_val = np.max(corr_matrix[i, non_diag])

        if max_val >= 0.7:
            indices = np.where(corr_matrix[i, :] >= 0.7)[0].tolist()
            range_groups["≥0.7"].extend(indices)
        elif max_val >= 0.5:
            indices = np.where(corr_matrix[i, :] >= 0.5)[0].tolist()
            range_groups["0.5~0.7"].extend(indices)
        elif max_val >= 0.3:
            indices = np.where(corr_matrix[i, :] >= 0.3)[0].tolist()
            range_groups["0.3~0.5"].extend(indices)
        else:
            range_groups["<0.3"].append(i)

    # 4. 去重并清理
    for k in range_groups: range_groups[k] = sorted(set(range_groups[k]))
    clean_groups = stepwise_clean(range_groups)

    # 5. 映射回真实被试 ID 并划分
    train_subs, val_subs, test_subs = [], [], []
    for group_name, member_indices in clean_groups.items():
        real_ids = [available_subs[idx] for idx in member_indices]
        np.random.shuffle(real_ids)

        n = len(real_ids)
        tr_cut = int(n * train_ratio)
        val_cut = int(n * (train_ratio + val_ratio))

        train_subs.extend(real_ids[:tr_cut])
        val_subs.extend(real_ids[tr_cut:val_cut])
        test_subs.extend(real_ids[val_cut:])

    return train_subs, val_subs, test_subs