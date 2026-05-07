#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立t-SNE可视化脚本 - 不依赖Solver类的方法
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# 强制重新加载
import importlib
import solver_groupnum

importlib.reload(solver_groupnum)

from solver_groupnum import Solver, DataConfig, MERMultimodalDataset, get_split_for_fold, set_seed, to_gpu
from AgglomerativeClusteringCorrection import get_subject_groups
import models

# ========== 配置 ==========
BASE_DIR = '/code/scw/MER/output_group_num_search_arousal'
GROUP_NUM = 6  # 根据实际修改
FOLD = 8  # 根据实际修改
MODE = 'train'


# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=BASE_DIR)
    parser.add_argument('--group_num', type=int, default=GROUP_NUM)
    parser.add_argument('--fold', type=int, default=FOLD)
    parser.add_argument('--mode', type=str, default=MODE, choices=['train', 'test'])
    parser.add_argument('--task', type=str, default='arousal')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()


def run_tsne_visualization(solver, model_path, data_loader, groups, mode, output_root):
    """
    完全独立的t-SNE可视化实现，不依赖Solver类的方法
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print(f"[*] Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # 处理维度不匹配
    for name, param in checkpoint.items():
        if 'leaner' in name and 'weight' in name:
            layer_name = name.replace('.weight', '')
            input_dim = param.shape[1]
            output_dim = param.shape[0]
            setattr(solver.model, layer_name, nn.Linear(input_dim, output_dim).to(device))
            print(f"[INIT] Creating {layer_name} with input_size={input_dim}")

    solver.model.load_state_dict(checkpoint, strict=False)
    solver.model.eval()

    # 提取特征
    print("[*] Extracting features...")
    shared_list, private_list = [], []
    subject_labels, task_labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            eeg, mod2, mod3, y, l, d = batch
            eeg, mod2, mod3 = to_gpu(eeg), to_gpu(mod2), to_gpu(mod3)
            l, d = to_gpu(l), to_gpu(d)

            solver.model(eeg, mod2, mod3, l, d, groups)

            shared_list.append(solver.model.utt_shared_subject.cpu().numpy())
            private_list.append(solver.model.utt_private_subject.cpu().numpy())
            subject_labels.append(d.cpu().numpy())
            task_labels.append(y.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(data_loader)} batches")

    # 准备标签
    all_subs = np.concatenate(subject_labels, axis=0)
    all_tasks = np.concatenate(task_labels, axis=0)

    # 建立subject到group的映射
    sub_to_group = {}
    if isinstance(groups, dict):
        for g_id, subs in groups.items():
            for s in subs:
                sub_to_group[int(s)] = int(g_id)
    else:
        for g_idx, group in enumerate(groups):
            for s in group:
                sub_to_group[int(s)] = g_idx

    # 根据mode过滤
    if mode == "train":
        mask = np.array([int(s) in sub_to_group for s in all_subs])
        all_subs = all_subs[mask]
        all_tasks = all_tasks[mask]

        # 过滤特征列表
        filtered_shared, filtered_private = [], []
        idx = 0
        for s, p in zip(shared_list, private_list):
            batch_len = len(s)
            batch_mask = mask[idx:idx + batch_len]
            if batch_mask.any():
                filtered_shared.append(s[batch_mask])
                filtered_private.append(p[batch_mask])
            idx += batch_len

        shared_list = filtered_shared
        private_list = filtered_private
        group_labels = np.array([sub_to_group[int(s)] for s in all_subs])
        title_suffix = f"Train Set (n={len(all_subs)})"
        print(f"[*] Train mode: {len(all_subs)} samples, {len(np.unique(group_labels))} groups")
    else:
        group_labels = np.array([sub_to_group.get(int(s), -1) for s in all_subs])
        title_suffix = f"Test Set (n={len(all_subs)})"
        print(f"[*] Test mode: {len(all_subs)} samples")

    # 绘制函数
    def draw_plot(data_list, color_data, title, filename, is_task=False):
        if len(data_list) == 0:
            print(f"[Warning] No data for {title}")
            return

        data = np.concatenate(data_list, axis=0)
        print(f"  Running t-SNE on {len(data)} samples...")

        perplexity = min(30, len(data) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        data_2d = tsne.fit_transform(data)
        data_2d = MinMaxScaler().fit_transform(data_2d)

        plt.figure(figsize=(10, 8))

        if is_task:
            cmap = 'coolwarm'
            scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=color_data,
                                  cmap=cmap, s=20, alpha=0.6, edgecolors='none')
            cbar = plt.colorbar(scatter)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Low', 'High'])
        else:
            unique_groups = np.unique(color_data)
            plot_groups = unique_groups[unique_groups >= 0]
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(plot_groups), 1)))

            for i, g in enumerate(plot_groups):
                mask = color_data == g
                plt.scatter(data_2d[mask, 0], data_2d[mask, 1],
                            c=[colors[i]], s=20, alpha=0.6,
                            label=f'Group {int(g)} (n={mask.sum()})', edgecolors='none')

            if -1 in unique_groups:
                mask = color_data == -1
                plt.scatter(data_2d[mask, 0], data_2d[mask, 1],
                            c='gray', s=10, alpha=0.3,
                            label=f'Ungrouped (n={mask.sum()})', edgecolors='none', marker='x')

            plt.legend(loc='best', fontsize=9)

        plt.title(f"{title}\n({title_suffix})", fontsize=12)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        save_dir = os.path.join(output_root, 'tsne_results')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{filename}_{mode}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved: {save_path}")

    # 绘制三张图
    print("[*] Plotting...")
    draw_plot(shared_list, group_labels, "Shared Features (by Group)", "shared_by_group", is_task=False)
    draw_plot(private_list, group_labels, "Private Features (by Group)", "private_by_group", is_task=False)
    draw_plot(shared_list, all_tasks, "Shared Features (by Task Label)", "shared_by_label", is_task=True)

    print("[*] Done!")


def main():
    args = parse_args()

    print(f"\n{'=' * 60}")
    print(f"t-SNE Visualization")
    print(f"Base: {args.base_dir}, Group: {args.group_num}, Fold: {args.fold}, Mode: {args.mode}")
    print(f"{'=' * 60}\n")

    # 路径检查
    fold_dir = os.path.join(args.base_dir, f'group_num_{args.group_num}', f'fold_{args.fold}')
    checkpoint_dir = os.path.join(fold_dir, 'checkpoints')
    model_path = os.path.join(checkpoint_dir, f'model_g{args.group_num}_fold{args.fold}.pt')

    if not os.path.exists(model_path):
        import glob
        available = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
        if available:
            model_path = available[0]
            print(f"[!] Using: {model_path}")
        else:
            print(f"[Error] No model in {checkpoint_dir}")
            return 1

    # 初始化
    set_seed(args.seed)
    DataConfig.OUTPUT_ROOT = fold_dir
    DataConfig.batch_size = args.batch_size

    # 加载fold划分
    with open(os.path.join(args.base_dir, 'fold_split.json'), 'r') as f:
        fold_data = json.load(f)
        folds = [fold_data[f'fold_{i}'] for i in range(10)]

    train_subs, val_subs, test_subs = get_split_for_fold(folds, args.fold - 1, seed=args.seed)
    print(f"[Split] Train: {len(train_subs)} | Val: {len(val_subs)} | Test: {len(test_subs)}")

    # 加载groups
    correlation_dir = os.path.join(fold_dir, 'correlation_analysis')
    groups = get_subject_groups(train_subs, DataConfig, args.group_num, correlation_dir)

    actual_groups = [g for g in groups if len(g) > 0]
    print(f"[Groups] {len(actual_groups)} groups: {[len(g) for g in actual_groups]}")

    # 选择数据集
    target_subs = train_subs if args.mode == 'train' else test_subs
    print(f"\n[*] Using {args.mode.upper()} set ({len(target_subs)} subjects)")

    # 创建DataLoader
    dataset = MERMultimodalDataset(target_subs, DataConfig, task=args.task)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)
    print(f"[Dataset] {len(dataset)} samples")

    # 创建Solver
    cfg = DataConfig()
    cfg.name = f"g{args.group_num}_fold{args.fold}"
    params = {"group_num": args.group_num, "diff_weight": 0.05,
              "sim_weight": 0.001, "learning_rate": 1e-4}

    dummy = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [0]), batch_size=1)
    solver = Solver(cfg, dummy, dummy, dummy, groups, params, is_train=False,
                    seed=args.seed, config=DataConfig)
    solver.build(cuda=torch.cuda.is_available())
    print("[*] Model built")

    # 执行可视化（使用独立函数，不依赖Solver的方法）
    run_tsne_visualization(solver, model_path, data_loader, groups, args.mode, fold_dir)

    print(f"\n[✓] Results: {os.path.join(fold_dir, 'tsne_results')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())