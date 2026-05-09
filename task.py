import matplotlib.pyplot as plt
import os
import sys
import pickle
import numpy as np
import json
import random
import re
import math
import itertools
import time
import gc
import contextlib
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import torch.nn.init as init
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models
from utils.misc import softmax
from torch.utils.data import Dataset


# ==================== 随机种子固定函数 ====================
def set_seed(seed=42):
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


# ==================== 2. 为当前 Fold 生成 Train/Val/Test 划分 ====================
def get_split_for_fold(folds, fold_idx, seed=42):
    """
    根据当前 fold 索引，生成 train, val, test 集合。
    - Test: 当前 fold
    - Rest: 剩余的 9 个 fold
    - Val: Rest 中的 10%
    - Train: Rest 中的 90%
    """
    np.random.seed(seed + fold_idx)  # 保证每次实验的随机性一致但不同fold有变化

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

