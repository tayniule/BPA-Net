import numpy as np


def z_score_normalize_by_subject_for_train_data(Xs, ds):
    """
    根据被试ID对脑电数据进行 Z-score 归一化。

    参数:
        Xs (np.ndarray): 脑电数据，形状为 (n_samples, 1, n_channels, n_timepoints)。
        ds (np.ndarray): 被试ID，形状为 (n_samples,)。

    返回:
        np.ndarray: 归一化后的脑电数据，形状与 Xs 相同。
    """
    # 获取唯一的被试ID
    unique_subjects = np.unique(ds)

    # 遍历每个被试
    for subject_id in unique_subjects:
        # 找到当前被试的索引
        subject_indices = np.where(ds == subject_id)[0]

        # 提取当前被试的数据
        subject_data = Xs[subject_indices]

        # 计算均值和标准差
        mean = np.mean(subject_data, axis=(0, 2, 3), keepdims=True)  # 沿样本、通道和时间维度计算均值
        std = np.std(subject_data, axis=(0, 2, 3), keepdims=True)  # 沿样本、通道和时间维度计算标准差

        # 避免除零错误，添加一个极小值
        std = std + 1e-8

        # 进行 Z-score 归一化
        normalized_data = (subject_data - mean) / std

        # 将归一化后的数据放回原位置
        Xs[subject_indices] = normalized_data

    return Xs


def z_score_normalize_for_test_data(Xs, ds):
    """
    根据被试ID对脑电数据进行 Z-score 归一化。

    参数:
        Xs (np.ndarray): 脑电数据，形状为 (n_samples, 1, n_channels, n_timepoints)。
        ds (np.ndarray): 被试ID，形状为 (n_samples,)。

    返回:
        np.ndarray: 归一化后的脑电数据，形状与 Xs 相同。
    """
    # 获取唯一的被试ID
    unique_subjects = np.unique(ds)

    # 遍历每个被试
    for subject_id in unique_subjects:
        # 找到当前被试的索引
        subject_indices = np.where(ds == subject_id)[0]

        # 提取当前被试的数据
        subject_data = Xs[subject_indices]

        # 计算均值和标准差
        mean = np.mean(subject_data, axis=(0, 2, 3), keepdims=True)  # 沿样本、通道和时间维度计算均值
        std = np.std(subject_data, axis=(0, 2, 3), keepdims=True)  # 沿样本、通道和时间维度计算标准差

        # 避免除零错误，添加一个极小值
        std = std + 1e-8

        # 进行 Z-score 归一化
        normalized_data = (subject_data - mean) / std

        # 将归一化后的数据放回原位置
        Xs[subject_indices] = normalized_data

    return Xs