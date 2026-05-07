#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Saliency Map 可视化脚本 - 适配十折交叉验证
根据 solver_groupnum.py 的划分逻辑，使用相同的 fold 划分
"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import pickle
import numpy as np
import json
import random
import re
import itertools
import time
import gc
import contextlib
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import torch.nn.init as init
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import models
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ==================== 配置类 ====================

class Config:
    """MDNet 需要的配置类"""

    def __init__(self):
        self.dataset_name = 'MER'
        self.hidden_size = 64
        self.batch_size = 64
        self.num_classes = 2
        self.dropout = 0.5
        self.subject_num = 73

    def activation(self):
        return nn.ReLU()


class DataConfig:
    """数据路径配置类 - 与 solver_groupnum.py 保持一致"""
    BASE_DATA_PATH = '/eds-storage/scw/MER/data'
    dataset_name = 'MER'
    name = 'MER'
    hidden_size = 64
    num_classes = 2
    dropout = 0.5
    weight_decay = 1e-4

    @staticmethod
    def activation():
        return nn.ReLU()

    subject_num = 73
    batch_size = 64
    SUBJECT_RANGE = range(1, 81)
    EEG_CHANNELS = 18
    GSR_CHANNELS = 1
    PPG_CHANNELS = 1
    TOTAL_CHANNELS = 20
    optimizer = torch.optim.Adam
    SAMPLE_RATE = 200
    WINDOW_SIZE = 4
    TIME_POINTS = 800
    OUTPUT_ROOT = './output_mer'
    LOG_DIR = os.path.join(OUTPUT_ROOT, 'logs')
    CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, 'checkpoints')
    RESULT_DIR = os.path.join(OUTPUT_ROOT, 'results')
    CURVE_DIR = os.path.join(OUTPUT_ROOT, 'curves')
    PKL_PATTERN = r'(\d+)-(\d+)-(\d+)\.pkl'


def set_seed(seed=42):
    """设置随机种子 - 与 solver_groupnum.py 一致"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_10_folds_subjects(config=DataConfig, seed=42):
    """
    扫描数据路径，获取所有有效被试，并随机均分为 10 个 Fold
    与 solver_groupnum.py 完全一致
    """
    np.random.seed(seed)
    base_path = config.BASE_DATA_PATH
    available_subs = []

    print(f"扫描数据路径以获取有效被试: {base_path}")
    for sub_id in config.SUBJECT_RANGE:
        sub_dir = os.path.join(base_path, str(sub_id))
        if os.path.exists(sub_dir) and os.path.isdir(sub_dir):
            pkl_files = [f for f in os.listdir(sub_dir) if f.endswith('.pkl')]
            if pkl_files:
                available_subs.append(sub_id)

    if not available_subs:
        raise ValueError(f"在 {base_path} 中没有找到任何有效被试数据！")

    print(f"共找到 {len(available_subs)} 名有效被试。")

    # 打乱被试并划分为10份
    np.random.shuffle(available_subs)
    folds = [list(arr) for arr in np.array_split(available_subs, 10)]

    return folds


def get_split_for_fold(folds, fold_idx, seed=42):
    """
    根据当前 fold 索引，生成 train, val, test 集合。
    与 solver_groupnum.py 完全一致
    - Test: 当前 fold
    - Rest: 剩余的 9 个 fold
    - Val: Rest 中的 10%
    - Train: Rest 中的 90%
    """
    np.random.seed(seed + fold_idx)

    test_subs = folds[fold_idx]

    rest_subs = []
    for i, f in enumerate(folds):
        if i != fold_idx:
            rest_subs.extend(f)

    np.random.shuffle(rest_subs)

    # 划分 10% 作为验证集
    val_size = max(1, int(len(rest_subs) * 0.10))
    val_subs = rest_subs[:val_size]
    train_subs = rest_subs[val_size:]

    # 排序使输出更美观
    return sorted(train_subs), sorted(val_subs), sorted(test_subs)


# ==================== 数据集类（与 solver_groupnum.py 一致）====================

class MERMultimodalDataset(Dataset):
    """MER多模态数据集类 - 与 solver_groupnum.py 完全一致"""

    def __init__(self, subject_ids, config=DataConfig, task='valence', transform=None):
        self.subject_ids = set(str(s) for s in subject_ids)
        self.config = config
        self.task = task
        self.transform = transform
        self.task_idx = 1 if task == 'arousal' else 0
        self.samples = self._scan_samples()

        print(f"Dataset [{task}] 初始化完成:")
        print(f"  被试数: {len(subject_ids)}")
        print(f"  样本数: {len(self.samples)}")

    def _scan_samples(self):
        """扫描所有被试的pkl文件"""
        samples = []
        pattern = re.compile(self.config.PKL_PATTERN)

        for sub_id in self.subject_ids:
            sub_dir = os.path.join(self.config.BASE_DATA_PATH, sub_id)
            if not os.path.exists(sub_dir):
                continue

            for filename in os.listdir(sub_dir):
                if not filename.endswith('.pkl'):
                    continue

                match = pattern.match(filename)
                if match:
                    file_sub_id, stimulus_id, window_id = match.groups()
                    if file_sub_id == sub_id:
                        file_path = os.path.join(sub_dir, filename)
                        samples.append({
                            'path': file_path,
                            'subject_id': int(sub_id),
                            'stimulus_id': int(stimulus_id),
                            'window_id': int(window_id)
                        })
        return samples

    def _load_sample(self, sample_info):
        """加载单个样本"""
        try:
            with open(sample_info['path'], 'rb') as f:
                data_dict = pickle.load(f)

            sample = data_dict['sample']  # (20, 800)
            label = data_dict['label']  # [valence, arousal]

            if sample.shape != (self.config.TOTAL_CHANNELS, self.config.TIME_POINTS):
                if sample.shape == (self.config.TIME_POINTS, self.config.TOTAL_CHANNELS):
                    sample = sample.T
                else:
                    raise ValueError(f"错误的shape: {sample.shape}")

            target = int(label[self.task_idx])
            return sample, target, sample_info['subject_id']

        except Exception as e:
            print(f"加载失败 {sample_info['path']}: {e}")
            return np.zeros((self.config.TOTAL_CHANNELS, self.config.TIME_POINTS)), 0, sample_info['subject_id']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sample, target, subject_id = self._load_sample(sample_info)

        if self.transform:
            sample = self.transform(sample)

        # 分割模态
        eeg = sample[:self.config.EEG_CHANNELS, :]  # (18, 800)
        gsr = sample[self.config.EEG_CHANNELS:self.config.EEG_CHANNELS + self.config.GSR_CHANNELS, :]  # (1, 800)
        ppg = sample[self.config.EEG_CHANNELS + self.config.GSR_CHANNELS:, :]  # (1, 800)

        # 转换为Tensor
        eeg = torch.from_numpy(eeg).float()
        gsr = torch.from_numpy(gsr).float()
        ppg = torch.from_numpy(ppg).float()

        length = torch.tensor(self.config.TIME_POINTS)
        domain = torch.tensor(subject_id, dtype=torch.long)

        return eeg, gsr, ppg, torch.tensor(target).long(), length, domain


# ==================== 分组加载（从已保存的聚类结果加载）====================

def load_groups_from_checkpoint(groups_file):
    """从已保存的 checkpoint 加载分组结果"""
    if not os.path.exists(groups_file):
        print(f"警告: 分组文件不存在 {groups_file}")
        return None

    with open(groups_file, 'rb') as f:
        groups = pickle.load(f)

    print(f"已加载分组: {groups_file}")
    if isinstance(groups, dict):
        actual_groups = [subs for subs in groups.values() if len(subs) > 0]
    else:
        actual_groups = [g for g in groups if len(g) > 0]

    print(f"  非空组数: {len(actual_groups)}, 各组大小: {[len(g) for g in actual_groups]}")
    return groups


# ==================== Saliency Map 计算与绘制 ====================

def compute_saliency_maps(model, dataloader, device, groups, max_samples=None):
    """
    计算 Saliency Maps

    Args:
        model: 已加载权重的 MDNet 模型
        dataloader: 数据加载器
        device: 计算设备
        groups: subject 分组（只在训练集上计算的分组）
        max_samples: 最多处理的样本数（None=全部）

    Returns:
        saliency_maps: list of (20, 800) arrays
        labels: list of true labels
        preds: list of predicted labels
        subject_ids: list of subject ids
    """
    model.eval()

    all_saliency = []
    all_labels = []
    all_preds = []
    all_subjects = []

    sample_count = 0
    print(f"Computing saliency maps for {max_samples if max_samples else 'all'} samples...")

    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        eeg, gsr, ppg, labels, lengths, subject_ids = batch_data
        batch_size = eeg.size(0)

        # 移到设备
        eeg = eeg.to(device).float()
        gsr = gsr.to(device).float()
        ppg = ppg.to(device).float()
        lengths = lengths.to(device)
        subject_ids = subject_ids.to(device)

        # 获取预测（不开梯度）
        with torch.no_grad():
            output = model(eeg, gsr, ppg, lengths, subject_ids, groups)
            pred_classes = output.argmax(dim=1)

        # 为每个样本计算 saliency
        for b in range(batch_size):
            if max_samples and sample_count >= max_samples:
                break

            # 创建带梯度的单样本输入
            eeg_b = eeg[b:b + 1].clone().detach().requires_grad_(True)
            gsr_b = gsr[b:b + 1].clone().detach().requires_grad_(True)
            ppg_b = ppg[b:b + 1].clone().detach().requires_grad_(True)

            # 前向传播
            out_b = model(eeg_b, gsr_b, ppg_b,
                          lengths[b:b + 1], subject_ids[b:b + 1], groups)

            # 获取预测类别分数并反向传播
            pred_class = pred_classes[b].item()
            target_score = out_b[0, pred_class]

            model.zero_grad()
            target_score.backward()

            # 提取梯度并计算 saliency (绝对值)
            grad_eeg = eeg_b.grad[0].abs().cpu().numpy()  # (18, 800)
            grad_gsr = gsr_b.grad[0].abs().cpu().numpy()  # (1, 800) or (800,)
            grad_ppg = ppg_b.grad[0].abs().cpu().numpy()  # (1, 800) or (800,)

            # 确保维度正确
            if grad_gsr.ndim == 1:
                grad_gsr = grad_gsr[np.newaxis, :]
            if grad_ppg.ndim == 1:
                grad_ppg = grad_ppg[np.newaxis, :]

            # 拼接: (20, 800)
            saliency = np.concatenate([grad_eeg, grad_gsr, grad_ppg], axis=0)

            # 样本内归一化到 [0, 1]
            smin, smax = saliency.min(), saliency.max()
            if smax > smin:
                saliency = (saliency - smin) / (smax - smin)

            all_saliency.append(saliency)
            all_labels.append(labels[b].item())
            all_preds.append(pred_class)
            all_subjects.append(subject_ids[b].item())

            sample_count += 1

            # 清理
            del eeg_b, gsr_b, ppg_b, out_b
            if sample_count % 50 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if max_samples and sample_count >= max_samples:
            break

    print(f"Processed {sample_count} samples")
    return all_saliency, all_labels, all_preds, all_subjects


def plot_saliency_heatmap(saliency_matrix, channel_names, save_path, title="Saliency Map"):
    """绘制单张 Saliency Map 热力图"""
    fig, ax = plt.subplots(figsize=(16, 6))

    im = ax.imshow(
        saliency_matrix,
        aspect='auto',
        cmap='magma',
        interpolation='nearest',
        vmin=0, vmax=1
    )

    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Channels', fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.set_yticks(range(len(channel_names)))
    ax.set_yticklabels(channel_names, fontsize=8)
    ax.set_xticks(range(0, 801, 100))

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label('Attribution Intensity', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def analyze_and_plot_saliency(saliency_maps, labels, preds, subjects, save_dir, dataset_name="MER"):
    """
    分析 saliency maps 并绘制多种可视化
    """
    os.makedirs(save_dir, exist_ok=True)

    saliency_array = np.stack(saliency_maps, axis=0)  # (N, 20, 800)
    labels = np.array(labels)
    preds = np.array(preds)
    subjects = np.array(subjects)

    channel_names = [f"EEG_{i + 1:02d}" for i in range(18)] + ["GSR", "PPG"]

    # 1. 整体平均 Saliency Map
    mean_saliency = np.mean(saliency_array, axis=0)
    plot_saliency_heatmap(
        mean_saliency, channel_names,
        os.path.join(save_dir, "mean_saliency_all.png"),
        title=f"Mean Saliency - All Samples ({dataset_name}, n={len(saliency_maps)})"
    )

    # 2. 按类别分组的 Saliency Map
    unique_classes = np.unique(labels)
    for c in unique_classes:
        mask = labels == c
        class_mean = np.mean(saliency_array[mask], axis=0)
        plot_saliency_heatmap(
            class_mean, channel_names,
            os.path.join(save_dir, f"mean_saliency_class_{c}.png"),
            title=f"Mean Saliency - Class {c} (n={mask.sum()})"
        )

    # 3. 预测正确 vs 错误对比
    correct_mask = labels == preds
    if correct_mask.any():
        correct_mean = np.mean(saliency_array[correct_mask], axis=0)
        plot_saliency_heatmap(
            correct_mean, channel_names,
            os.path.join(save_dir, "mean_saliency_correct.png"),
            title=f"Mean Saliency - Correct Predictions (n={correct_mask.sum()})"
        )

    if (~correct_mask).any():
        wrong_mean = np.mean(saliency_array[~correct_mask], axis=0)
        plot_saliency_heatmap(
            wrong_mean, channel_names,
            os.path.join(save_dir, "mean_saliency_wrong.png"),
            title=f"Mean Saliency - Wrong Predictions (n={(~correct_mask).sum()})"
        )

    # 4. 类别差异图
    if len(unique_classes) == 2:
        class_0 = np.mean(saliency_array[labels == 0], axis=0)
        class_1 = np.mean(saliency_array[labels == 1], axis=0)
        diff = np.abs(class_0 - class_1)
        diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        plot_saliency_heatmap(
            diff_norm, channel_names,
            os.path.join(save_dir, "saliency_class_difference.png"),
            title="Saliency Difference - |Class 0 - Class 1|"
        )

    # 5. 通道重要性排序
    channel_importance = np.mean(mean_saliency, axis=1)  # (20,)
    sorted_idx = np.argsort(channel_importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#d62728' if 'EEG' in channel_names[i] else '#2ca02c' if 'GSR' in channel_names[i] else '#1f77b4'
              for i in sorted_idx]
    bars = ax.barh(range(len(sorted_idx)), channel_importance[sorted_idx], color=colors)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([channel_names[i] for i in sorted_idx])
    ax.set_xlabel('Mean Attribution Intensity', fontsize=12)
    ax.set_title('Channel Importance Ranking', fontsize=14)
    ax.invert_yaxis()

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, channel_importance[sorted_idx])):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "channel_importance_ranking.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 时间重要性曲线
    temporal_importance = np.mean(mean_saliency, axis=0)  # (800,)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(range(800), temporal_importance, 'b-', linewidth=1.5)
    ax.fill_between(range(800), 0, temporal_importance, alpha=0.3, color='blue')
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Mean Attribution', fontsize=12)
    ax.set_title('Temporal Importance Profile', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 标记峰值
    peak_idx = np.argmax(temporal_importance)
    ax.axvline(x=peak_idx, color='r', linestyle='--', alpha=0.5)
    ax.text(peak_idx, temporal_importance[peak_idx], f'Peak: {peak_idx}', color='r')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temporal_importance_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存数据
    np.savez(
        os.path.join(save_dir, "saliency_data.npz"),
        saliency_maps=saliency_array,
        labels=labels,
        preds=preds,
        subjects=subjects,
        mean_saliency=mean_saliency
    )

    print(f"\nAll results saved to {save_dir}/")
    print(f"Top 5 most important channels:")
    for i in range(5):
        idx = sorted_idx[i]
        print(f"  {i + 1}. {channel_names[idx]}: {channel_importance[idx]:.4f}")

    return mean_saliency


# ==================== 主函数 ====================

def main():
    """
    主函数：使用与 solver_groupnum.py 相同的十折划分逻辑
    """
    # ========== 配置参数（根据你的实际情况修改）==========
    BASE_OUTPUT_DIR = '/code/scw/MER/output_group_num_search_arousal'  # 训练输出目录
    GROUP_NUM = 5  # 使用的 group_num
    FOLD_IDX = 4  # 要可视化的 fold (0-9，对应 fold_1 到 fold_10)
    TASK = 'arousal'  # 'arousal' 或 'valence'
    MAX_SAMPLES = None  # None=全部样本，或设为100测试
    # ===================================================

    print(f"\n{'=' * 60}")
    print(f"Saliency Map Visualization")
    print(f"Group Num: {GROUP_NUM}, Fold: {FOLD_IDX + 1}, Task: {TASK}")
    print(f"{'=' * 60}\n")

    # 1. 设置随机种子（与训练时一致）
    set_seed(42)

    # 2. 获取十折划分（与训练时完全一致）
    print("[Step 1] Generating 10-fold splits...")
    folds = get_10_folds_subjects(DataConfig, seed=42)

    # 加载已保存的 fold 划分（如果存在）
    fold_split_path = os.path.join(BASE_OUTPUT_DIR, 'fold_split.json')
    if os.path.exists(fold_split_path):
        print(f"[*] Loading saved fold split: {fold_split_path}")
        with open(fold_split_path, 'r') as f:
            fold_data = json.load(f)
            folds = [fold_data[f"fold_{i}"] for i in range(10)]

    # 3. 获取当前 fold 的数据划分
    print(f"\n[Step 2] Getting split for fold {FOLD_IDX + 1}...")
    train_subs, val_subs, test_subs = get_split_for_fold(folds, FOLD_IDX, seed=42)

    print(f"Train: {len(train_subs)} subjects | Val: {len(val_subs)} | Test: {len(test_subs)}")
    print(f"Test subjects: {test_subs}")

    # 4. 加载分组（从训练时保存的分组文件）
    print(f"\n[Step 3] Loading subject groups...")
    fold_dir = os.path.join(BASE_OUTPUT_DIR, f'group_num_{GROUP_NUM}', f'fold_{FOLD_IDX + 1}')
    groups_file = os.path.join(fold_dir, 'correlation_analysis', f'groups_ng{GROUP_NUM}.pkl')

    groups = load_groups_from_checkpoint(groups_file)
    if groups is None:
        print(f"[Error] Cannot load groups from {groups_file}")
        print("Please check if the training was completed successfully.")
        return 1

    # 5. 创建设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Step 4] Using device: {device}")

    # 6. 创建测试数据集（使用测试集被试）
    print(f"\n[Step 5] Creating test dataset for fold {FOLD_IDX + 1}...")
    test_dataset = MERMultimodalDataset(test_subs, config=DataConfig, task=TASK)

    if len(test_dataset) == 0:
        print("❌ 测试集为空，请检查数据路径")
        return 1

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 7. 加载模型
    print(f"\n[Step 6] Loading model...")
    model_path = os.path.join(fold_dir, 'checkpoints', f'model_g{GROUP_NUM}_fold{FOLD_IDX + 1}.pt')

    if not os.path.exists(model_path):
        # 尝试找任何可用的模型文件
        import glob
        checkpoint_dir = os.path.dirname(model_path)
        available_models = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
        if available_models:
            model_path = available_models[0]
            print(f"[!] Specified model not found, using: {model_path}")
        else:
            print(f"❌ No model found in {checkpoint_dir}")
            return 1

    print(f"[*] Model path: {model_path}")

    # 创建配置和模型
    config = Config()
    config.dataset_name = 'MER'
    config.hidden_size = 64
    config.batch_size = 4
    config.num_classes = 2
    config.dropout = 0.5
    config.subject_num = 73

    params = {
        'dataset_name': 'MER',
        'hidden_size': 64,
        'batch_size': 4,
        'num_classes': 2,
        'dropout': 0.5,
        'weight_decay': 1e-4,
        "group_num": GROUP_NUM,
        'activation': nn.ReLU(),
        'subject_num': 73,
        "diff_weight": 0.05,
        "sim_weight": 0.001,
        "learning_rate": 1e-4
    }

    model = models.MDNet(config, params)
    model.to(device)

    # 初始化模型（dummy forward）
    print("[*] Initializing model...")
    model.eval()
    with torch.no_grad():
        dummy_eeg = torch.randn(2, 18, 800).to(device)
        dummy_gsr = torch.randn(2, 1, 800).to(device)
        dummy_ppg = torch.randn(2, 1, 800).to(device)
        dummy_lengths = torch.tensor([800, 800]).to(device)
        dummy_subject_ids = torch.tensor([1, 2]).to(device)
        _ = model(dummy_eeg, dummy_gsr, dummy_ppg, dummy_lengths, dummy_subject_ids, groups)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    # 处理可能的键名差异
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    else:
        state_dict = checkpoint

    # 移除 'module.' 前缀（如果是分布式训练保存的）
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    print("[✓] Model loaded successfully")

    # 8. 计算 Saliency Maps
    print(f"\n[Step 7] Computing saliency maps...")
    saliency_maps, labels, preds, subjects = compute_saliency_maps(
        model, test_loader, device, groups, max_samples=MAX_SAMPLES
    )

    # 9. 分析和可视化
    print(f"\n[Step 8] Generating visualizations...")
    save_dir = os.path.join(fold_dir, 'saliency_maps')
    mean_saliency = analyze_and_plot_saliency(
        saliency_maps, labels, preds, subjects, save_dir,
        dataset_name=f"MER-{TASK}-Fold{FOLD_IDX + 1}"
    )

    print(f"\n{'=' * 60}")
    print(f"[✓] Success! All results saved to:")
    print(f"    {save_dir}")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())