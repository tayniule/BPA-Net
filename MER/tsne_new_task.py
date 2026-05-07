#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
t-SNE可视化脚本 - 方法2：真实标签 vs 模型预测对比
修复版：处理 leaner 层维度不匹配问题
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict

# 导入自定义模块
from solver_groupnum import Solver, DataConfig, MERMultimodalDataset, get_split_for_fold, set_seed, to_gpu
from AgglomerativeClusteringCorrection import get_subject_groups
import models

# ========== 配置 ==========
BASE_DIR = '/code/scw/MER/output_group_num_search_arousal'
GROUP_NUM = 5
FOLD_IDX = 4
TASK = 'arousal'
MODE = 'test'
MAX_SAMPLES = None


# =========================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=BASE_DIR)
    parser.add_argument('--group_num', type=int, default=GROUP_NUM)
    parser.add_argument('--fold', type=int, default=FOLD_IDX)
    parser.add_argument('--mode', type=str, default=MODE, choices=['train', 'val', 'test'])
    parser.add_argument('--task', type=str, default=TASK)
    parser.add_argument('--max_samples', type=int, default=MAX_SAMPLES)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_model_with_dimension_fix(model_path, device, config, params, groups):
    """
    加载 MDNet 模型，自动处理 leaner 层的维度不匹配问题
    """
    # 1. 先创建模型
    model = models.MDNet(config, params)
    model.to(device)
    model.eval()

    # 2. 关键步骤：dummy forward 来初始化 leaner 层
    print("[*] Running dummy forward to initialize leaner layers...")
    with torch.no_grad():
        dummy_eeg = torch.randn(2, 18, 800).to(device)
        dummy_gsr = torch.randn(2, 1, 800).to(device)
        dummy_ppg = torch.randn(2, 1, 800).to(device)
        dummy_lengths = torch.tensor([800, 800]).to(device)
        dummy_subject_ids = torch.tensor([1, 2]).to(device)

        # 这会自动创建 leaner 层，并设置正确的输入维度
        _ = model(dummy_eeg, dummy_gsr, dummy_ppg, dummy_lengths, dummy_subject_ids, groups)

    # 3. 现在加载权重
    print(f"[*] Loading weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # 处理不同格式的 checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict',
                                    checkpoint.get('model_state_dict',
                                                   checkpoint))
    else:
        state_dict = checkpoint

    # 移除 'module.' 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 检查维度匹配情况
    model_dict = model.state_dict()
    matched = 0
    mismatched = []

    for k, v in list(new_state_dict.items()):
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                matched += 1
            else:
                mismatched.append((k, model_dict[k].shape, v.shape))
                del new_state_dict[k]
        else:
            del new_state_dict[k]

    if mismatched:
        print(f"[!] Found {len(mismatched)} mismatched layers (will use initialized values):")
        for name, model_shape, ckpt_shape in mismatched:
            print(f"    {name}: model={model_shape}, checkpoint={ckpt_shape}")

    print(f"[*] Loading {len(new_state_dict)}/{len(model_dict)} layers...")
    model.load_state_dict(new_state_dict, strict=False)

    print(f"[✓] Model loaded successfully")
    return model


def extract_features_with_hook(model, dataloader, device, groups, max_samples=None):
    """
    提取特征和模型预测
    """
    model.eval()

    all_features = []
    all_labels = []
    all_preds = []
    all_probs = []
    all_subjects = []

    sample_count = 0

    print(f"[*] Extracting features from {max_samples if max_samples else 'all'} samples...")

    with torch.no_grad():
        for batch in dataloader:
            eeg, gsr, ppg, y, l, d = batch
            batch_size = eeg.size(0)

            eeg = eeg.to(device).float()
            gsr = gsr.to(device).float()
            ppg = ppg.to(device).float()
            l = l.to(device)
            d = d.to(device)

            # 前向传播
            output = model(eeg, gsr, ppg, l, d, groups)

            # 获取预测
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            # 获取 shared 特征
            if hasattr(model, 'utt_shared_subject'):
                features = model.utt_shared_subject
            elif hasattr(model, 'utt_shared_eeg_1'):
                features = model.utt_shared_eeg_1
            else:
                features = output  # 退而求其次

            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_subjects.append(d.cpu().numpy())

            sample_count += batch_size
            if max_samples and sample_count >= max_samples:
                break

    # 合并
    features_array = np.concatenate(all_features, axis=0)
    true_labels = np.concatenate(all_labels)
    pred_labels = np.concatenate(all_preds)
    pred_probs = np.concatenate(all_probs)
    subject_ids = np.concatenate(all_subjects)

    print(f"[*] Feature shape: {features_array.shape}")
    print(f"[*] Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")

    return features_array, true_labels, pred_labels, pred_probs, subject_ids


def compute_tsne(features, perplexity=30, random_state=42):
    """计算t-SNE降维"""
    print(f"[*] Computing t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    features_2d = MinMaxScaler().fit_transform(features_2d)
    print(f"[*] t-SNE done")
    return features_2d


def plot_tsne_true_vs_pred(data_2d, true_labels, pred_labels, save_path, title="t-SNE: True vs Predicted"):
    """绘制真实标签 vs 预测对比图"""
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(true_labels, pred_labels)
    wrong_mask = true_labels != pred_labels

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    colors = ['#1f77b4', '#d62728']
    class_names = ['Low', 'High'] if TASK == 'arousal' else ['Negative', 'Positive']

    # 左图：真实标签
    for i in [0, 1]:
        mask = true_labels == i
        axes[0].scatter(data_2d[mask, 0], data_2d[mask, 1],
                        c=colors[i], s=40, alpha=0.6, edgecolors='none',
                        label=f'True {class_names[i]} (n={mask.sum()})')

    axes[0].set_title('Ground Truth Labels', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 右图：模型预测
    for i in [0, 1]:
        mask = pred_labels == i
        axes[1].scatter(data_2d[mask, 0], data_2d[mask, 1],
                        c=colors[i], s=40, alpha=0.6, edgecolors='none',
                        label=f'Predicted {class_names[i]} (n={mask.sum()})')

    if wrong_mask.any():
        axes[1].scatter(data_2d[wrong_mask, 0], data_2d[wrong_mask, 1],
                        facecolors='none', edgecolors='yellow', s=120, linewidths=2.5,
                        label=f'Misclassified ({wrong_mask.sum()}, {100 * wrong_mask.sum() / len(true_labels):.1f}%)')

    axes[1].set_title(f'Model Predictions (Accuracy: {acc:.3f})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] Saved: {save_path}")
    return acc


def main():
    args = parse_args()

    print(f"\n{'=' * 70}")
    print(f"t-SNE Visualization (Method 2: True vs Predicted) - FIXED")
    print(f"Base: {args.base_dir}")
    print(f"Group: {args.group_num}, Fold: {args.fold + 1}, Mode: {args.mode}, Task: {args.task}")
    print(f"{'=' * 70}\n")

    # 1. 设置随机种子
    set_seed(args.seed)

    # 2. 获取fold划分
    fold_split_path = os.path.join(args.base_dir, 'fold_split.json')
    with open(fold_split_path, 'r') as f:
        fold_data = json.load(f)
        folds = [fold_data[f"fold_{i}"] for i in range(10)]

    train_subs, val_subs, test_subs = get_split_for_fold(folds, args.fold, seed=args.seed)

    if args.mode == 'train':
        target_subs = train_subs
    elif args.mode == 'val':
        target_subs = val_subs
    else:
        target_subs = test_subs

    print(f"[Data] {args.mode}: {len(target_subs)} subjects")

    # 3. 加载分组
    fold_dir = os.path.join(args.base_dir, f'group_num_{args.group_num}', f'fold_{args.fold + 1}')
    groups_file = os.path.join(fold_dir, 'correlation_analysis', f'groups_ng{args.group_num}.pkl')

    with open(groups_file, 'rb') as f:
        groups = pickle.load(f)

    actual_groups = [g for g in groups if len(g) > 0] if isinstance(groups, list) else [v for v in groups.values() if
                                                                                        len(v) > 0]
    print(f"[Groups] {len(actual_groups)} groups: {[len(g) for g in actual_groups]}")

    # 4. 创建数据集
    DataConfig.OUTPUT_ROOT = fold_dir
    dataset = MERMultimodalDataset(target_subs, DataConfig, task=args.task)

    if len(dataset) == 0:
        print("[Error] Empty dataset!")
        return 1

    if args.max_samples and len(dataset) > args.max_samples:
        indices = np.random.choice(len(dataset), args.max_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"[Limit] Using {args.max_samples} random samples")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0
    )

    # 5. 加载模型（使用修复后的函数）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")

    model_path = os.path.join(fold_dir, 'checkpoints', f'model_g{args.group_num}_fold{args.fold + 1}.pt')
    if not os.path.exists(model_path):
        import glob
        available = glob.glob(os.path.join(os.path.dirname(model_path), '*.pt'))
        if available:
            model_path = available[0]
            print(f"[!] Using: {model_path}")
        else:
            print(f"[Error] No model found!")
            return 1

    # 创建配置
    class TempConfig:
        dataset_name = 'MER'
        hidden_size = 64
        num_classes = 2
        dropout = 0.5
        subject_num = 73
        batch_size = 32
        def activation(self):
            return nn.ReLU()

    params = {
        'dataset_name': 'MER',
        'hidden_size': 64,
        'batch_size': 32,
        'num_classes': 2,
        'dropout': 0.5,
        'weight_decay': 1e-4,
        "group_num": args.group_num,
        'activation': nn.ReLU(),
        'subject_num': 73,
        "diff_weight": 0.05,
        "sim_weight": 0.001,
        "learning_rate": 1e-4
    }

    # 使用修复后的加载函数
    model = load_model_with_dimension_fix(model_path, device, TempConfig(), params, groups)

    # 6. 提取特征和预测
    print(f"\n[Extracting features...]")
    features, true_labels, pred_labels, pred_probs, subject_ids = extract_features_with_hook(
        model, dataloader, device, groups, max_samples=None
    )

    # 7. t-SNE降维
    features_2d = compute_tsne(features, perplexity=min(30, len(features) - 1))

    # 8. 绘制可视化
    print(f"\n[Generating plots...]")
    save_dir = os.path.join(fold_dir, 'tsne_results')
    os.makedirs(save_dir, exist_ok=True)

    plot_tsne_true_vs_pred(
        features_2d, true_labels, pred_labels,
        os.path.join(save_dir, f'tsne_true_vs_pred_{args.mode}.png'),
        title=f"t-SNE: {args.task.capitalize()} - Fold {args.fold + 1} ({args.mode} set)"
    )

    # 保存数据
    np.savez(
        os.path.join(save_dir, f'tsne_data_{args.mode}.npz'),
        features_2d=features_2d,
        true_labels=true_labels,
        pred_labels=pred_labels,
        pred_probs=pred_probs,
        subject_ids=subject_ids
    )

    print(f"\n{'=' * 70}")
    print(f"[✓] All results saved to: {save_dir}")
    print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())