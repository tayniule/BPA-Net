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
import get_distillation_kernel, get_distillation_kernel_homo
from utils.misc import softmax
from torch.utils.data import Dataset
from AgglomerativeClusteringCorrection import SubjectCorrelationAnalyzer, get_subject_groups


class DataConfig:
    """
    数据路径配置类 - 根据你的实际情况修改
    """
    # 原始数据根目录（被试文件夹所在位置）
    BASE_DATA_PATH = '/eds-storage/scw/MER/data'  # 例如：'/home/user/eeg_data/processed'
    dataset_name = 'MER'  # <-- 添加这一行，可选值: 'DEAP', 'HCI', 'MER'
    hidden_size = 64  # <-- 添加这一行
    num_classes = 2  # <-- 添加这一行（二分类：高/低唤醒或效价）
    dropout = 0.5  # <-- 添加这一行
    weight_decay = 1e-4

    @staticmethod
    def activation():
        return nn.ReLU()
        # <-- 添加这一行

    subject_num = 73
    batch_size = 64
    # 被试ID范围（根据实际情况修改，1-80是当前代码的默认值）
    SUBJECT_RANGE = range(1, 81)  # 或者 [1,2,3,4...] 明确指定
    # 通道配置（根据你的预处理代码：18 EEG + 1 GSR + 1 PPG = 20通道）
    EEG_CHANNELS = 18
    GSR_CHANNELS = 1
    PPG_CHANNELS = 1
    TOTAL_CHANNELS = 20  # EEG_CHANNELS + GSR_CHANNELS + PPG_CHANNELS

    # 数据shape配置
    SAMPLE_RATE = 200
    WINDOW_SIZE = 4  # seconds
    TIME_POINTS = 800  # SAMPLE_RATE * WINDOW_SIZE
    # 结果保存根目录
    OUTPUT_ROOT = './output_mer'  # 所有结果会保存在这里
    LOG_DIR = os.path.join(OUTPUT_ROOT, 'logs')  # <-- 添加这一行
    CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, 'checkpoints')
    RESULT_DIR = os.path.join(OUTPUT_ROOT, 'results')
    CURVE_DIR = os.path.join(OUTPUT_ROOT, 'curves')
    PKL_PATTERN = r'(\d+)-(\d+)-(\d+)\.pkl'  # 匹配 {sub}-{stimulus}-{window}.pkl

    # ==================== 自动创建输出目录 ====================


def setup_directories(config_class=DataConfig):
    """创建所有必要的输出目录"""
    dirs = [
        os.path.join(config_class.OUTPUT_ROOT, 'checkpoints'),
        os.path.join(config_class.OUTPUT_ROOT, 'results'),
        os.path.join(config_class.OUTPUT_ROOT, 'logs'),
        os.path.join(config_class.OUTPUT_ROOT, 'curves')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"输出目录已创建: {config_class.OUTPUT_ROOT}")


# ==================== 新增：随机分组函数（消融实验） ====================
# ==================== 新增：随机分组函数（消融实验）====================
def get_random_subject_groups(train_subjects, test_subjects, num_groups, fold_idx, seed=42,
                              save_dir='./ablation_random_groups'):
    """
    在交叉验证划分后，对训练集受试者进行随机分组（消融实验用）。
    严格防止数据泄露：仅在训练集上随机打乱分组，测试集随机分配组标签仅用于推理流程。

    Args:
        train_subjects (list): 当前折的训练集受试者 ID 列表。
        test_subjects (list): 当前折的测试集受试者 ID 列表。
        num_groups (int): 分组数量 (K)。
        fold_idx (int): 当前交叉验证的折数索引（从0开始）。
        seed (int): 随机种子，保证可复现。
        save_dir (str): 随机分组结果的保存路径。

    Returns:
        list: groups，格式为 [[sub1, sub2, ...], [sub3, sub4, ...], ...]，兼容原有代码
    """
    os.makedirs(save_dir, exist_ok=True)

    # 设置随机种子（根据fold_idx变化，确保不同fold分组不同但可复现）
    random.seed(seed + fold_idx)
    np.random.seed(seed + fold_idx)

    # 1. 复制并打乱训练集受试者，保证真正的随机
    shuffled_train = list(train_subjects)
    random.shuffle(shuffled_train)

    # 2. 将训练集受试者尽可能平均地分配到 num_groups 个组中
    # 返回 list 格式，兼容原有代码
    groups = [[] for _ in range(num_groups)]
    for idx, sub in enumerate(shuffled_train):
        group_idx = idx % num_groups
        groups[group_idx].append(int(sub))

    # 3. 处理测试集受试者：在随机分组的 Baseline 中，测试集也应当被随机分配到一个组中进行推理
    # 注意：这里的分配不影响训练，仅为了在推理时跑通 Group Encoder 流程
    test_assignments = {}
    for sub in test_subjects:
        random_g = random.randint(0, num_groups - 1)
        test_assignments[int(sub)] = int(random_g)
        # 将测试集也加入对应的组，确保模型能处理测试集
        groups[random_g].append(int(sub))

    # 4. 构建 subject_to_group 映射字典（用于保存）
    subject_to_group_map = {}
    for g_id, subs in enumerate(groups):
        for sub in subs:
            subject_to_group_map[int(sub)] = int(g_id)

    # 5. 保存当前折、当前 K 值的随机分组结果
    save_path = os.path.join(save_dir, f'random_groups_k{num_groups}_fold{fold_idx + 1}.json')
    save_data = {
        'num_groups': num_groups,
        'fold': fold_idx + 1,
        'train_distribution': {str(i): [int(s) for s in groups[i] if int(s) in [int(x) for x in train_subjects]]
                               for i in range(num_groups)},
        'test_assignments': test_assignments,
        'full_map': {int(k): int(v) for k, v in subject_to_group_map.items()},
        'groups_list': groups,  # 保存list格式便于查看
        'note': 'Random grouping baseline for ablation study. Train subjects are randomly shuffled.'
    }

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=4)

    print(f"[Random Group] Saved grouping result to: {save_path}")
    print(f"[Random Group] Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
    train_counts = [len([s for s in g if int(s) in [int(x) for x in train_subjects]]) for g in groups]
    test_counts = [len([s for s in g if int(s) in [int(x) for x in test_subjects]]) for g in groups]
    print(f"[Random Group] Train distribution per group: {train_counts}")
    print(f"[Random Group] Test distribution per group: {test_counts}")

    return groups  # 返回 list 格式，兼容原有代码


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


# ==================== 优化的Dataset类 ====================
class MERMultimodalDataset(Dataset):
    """
    MER多模态数据集类

    适配你的预处理数据结构：
    - 文件: {subject_id}/{subject_id}-{stimulus_id}-{window_id}.pkl
    - 内容: {'sample': (20, 800), 'label': [valence_label, arousal_label]}
    - 通道: 0-17 EEG, 18 GSR, 19 PPG
    """

    def __init__(self, subject_ids, config=DataConfig, task='valence',
                 transform=None):
        """
        Args:
            subject_ids: 被试ID列表，例如 [1, 2, 3, 5, 8]
            config: 数据配置
            task: 'arousal' 或 'valence'，决定使用label的哪个维度
            transform: 可选的数据增强
        """
        self.subject_ids = set(str(s) for s in subject_ids)
        self.config = config
        self.task = task
        self.transform = transform

        # 任务索引：0=valence, 1=arousal
        self.task_idx = 1 if task == 'arousal' else 0

        # 扫描所有样本
        self.samples = self._scan_samples()

        print(f"Dataset [{task}] 初始化完成:")
        print(f"  被试数: {len(subject_ids)}")
        print(f"  样本数: {len(self.samples)}")

        if len(self.samples) > 0:
            # 验证第一个样本
            sample_data, label, sid = self._load_sample(self.samples[0])
            print(f"  数据shape: {sample_data.shape}")
            print(f"  标签示例: {label} (task={task})")

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

                # 解析文件名: {sub}-{stimulus}-{window}.pkl
                match = pattern.match(filename)
                if match:
                    file_sub_id, stimulus_id, window_id = match.groups()
                    # 只添加属于当前被试的文件（防止误匹配）
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
            label = data_dict['label']  # [valence_label, arousal_label]

            # 数据验证
            if sample.shape != (self.config.TOTAL_CHANNELS, self.config.TIME_POINTS):
                # 尝试转置
                if sample.shape == (self.config.TIME_POINTS, self.config.TOTAL_CHANNELS):
                    sample = sample.T
                else:
                    raise ValueError(f"错误的shape: {sample.shape}")

            # 选择任务对应的标签
            target = int(label[self.task_idx])

            return sample, target, sample_info['subject_id']

        except Exception as e:
            print(f"加载失败 {sample_info['path']}: {e}")
            # 返回零数据（实际应用中应该过滤）
            return np.zeros((self.config.TOTAL_CHANNELS, self.config.TIME_POINTS)), 0, sample_info['subject_id']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sample, target, subject_id = self._load_sample(sample_info)

        # 数据增强（可选）
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

        # 返回格式兼容原始solver代码: (eeg, mod2, mod3, y, length, domain)
        # 这里: eeg=EEG, mod2=GSR, mod3=PPG
        length = torch.tensor(self.config.TIME_POINTS)
        domain = torch.tensor(subject_id)

        return eeg, gsr, ppg, torch.tensor(target).long(), length, domain

    def get_subject_distribution(self):
        """获取被试分布统计（用于验证）"""
        from collections import defaultdict
        sub_counts = defaultdict(int)
        label_counts = defaultdict(int)

        for i in range(len(self)):
            sample_info = self.samples[i]
            sub_counts[sample_info['subject_id']] += 1

            _, target, _ = self._load_sample(sample_info)
            label_counts[target] += 1

        return dict(sub_counts), dict(label_counts)


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.ConvTranspose2d):
            init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        elif isinstance(module, nn.LayerNorm):
            init.constant_(module.bias, 0)
            init.constant_(module.weight, 1.0)


def kappa(confusion_mat):
    pe_rows = np.sum(confusion_mat, axis=0)
    pe_cols = np.sum(confusion_mat, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_mat) / float(sum_total)
    return (po - pe) / (1 - pe)


class Solver(object):
    def __init__(self, train_config, train_data_loader, dev_data_loader,
                 test_data_loader, groups, params, is_train=True,
                 model=None, seed=42, config=DataConfig):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.groups = groups
        self.is_train = is_train
        self.model = model
        self.params = params
        self.seed = seed
        self.config = config

        set_seed(seed)

        # 路径配置
        self.checkpoint_dir = os.path.join(config.OUTPUT_ROOT, 'checkpoints')
        self.result_dir = os.path.join(config.OUTPUT_ROOT, 'results')
        self.curve_dir = os.path.join(config.OUTPUT_ROOT, 'curves')

        # [修改点 1] 确保 log_dir 使用最新的 config.OUTPUT_ROOT 动态生成
        # 原代码直接使用 config.LOG_DIR 可能导致使用的是旧的类属性值
        self.log_dir = os.path.join(config.OUTPUT_ROOT, 'logs', train_config.name)

        # 创建目录
        for d in [self.checkpoint_dir, self.result_dir, self.log_dir, self.curve_dir]:
            os.makedirs(d, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard log dir: {self.log_dir}")  # 打印确认路径

        self._save_config()

    def visualize_tsne_from_checkpoint(self, model_path, data_loader, mode="train"):
        """
        从checkpoint加载模型并绘制t-SNE
        mode="train": 在训练集上绘制（推荐，因为有分组标签）
        mode="test": 在测试集上绘制（被试可能无分组）
        """
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import MinMaxScaler

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)

        # 处理可能的维度不匹配
        for name, param in checkpoint.items():
            if 'leaner' in name and 'weight' in name:
                layer_name = name.replace('.weight', '')
                input_dim = param.shape[1]
                output_dim = param.shape[0]
                setattr(self.model, layer_name, nn.Linear(input_dim, output_dim).to(device))

        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        print(f"[*] Loaded model from: {model_path}")

        shared_list, private_list = [], []
        subject_labels, task_labels = [], []

        # 提取特征
        with torch.no_grad():
            for batch in data_loader:
                eeg, mod2, mod3, y, l, d = batch
                eeg, mod2, mod3 = to_gpu(eeg), to_gpu(mod2), to_gpu(mod3)
                l, d = to_gpu(l), to_gpu(d)

                # 前向传播
                self.model(eeg, mod2, mod3, l, d, self.groups)

                shared_list.append(self.model.utt_shared_subject.cpu().numpy())
                private_list.append(self.model.utt_private_subject.cpu().numpy())
                subject_labels.append(d.cpu().numpy())
                task_labels.append(y.cpu().numpy())

        # 调用绘图函数 - 确保传了mode参数
        self._plot_tsne_by_group(shared_list, private_list, subject_labels, task_labels, mode=mode)
        print("[*] t-SNE visualization completed!")

    def _plot_tsne_by_group(self, shared_list, private_list, subject_labels, task_labels, mode="train"):
        """
        绘制t-SNE图，支持train和test模式
        """
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import MinMaxScaler
        import matplotlib.pyplot as plt

        all_subs = np.concatenate(subject_labels, axis=0)
        all_tasks = np.concatenate(task_labels, axis=0)

        # 建立 Subject 到 Group 的映射
        sub_to_group = {}
        if isinstance(self.groups, dict):
            for g_id, subs in self.groups.items():
                for s in subs:
                    sub_to_group[int(s)] = int(g_id)
        else:
            # list 格式：[[sub1, sub2], [sub3, sub4], ...]
            for g_idx, group in enumerate(self.groups):
                for s in group:
                    sub_to_group[int(s)] = g_idx

        # 关键修改：对于训练集，只保留有分组的被试
        if mode == "train":
            # 训练集：过滤掉未分组的被试
            mask = np.array([int(s) in sub_to_group for s in all_subs])
            all_subs = all_subs[mask]
            all_tasks = all_tasks[mask]

            # 过滤特征
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

            if len(all_subs) == 0:
                print("[Warning] No samples with valid group labels found!")
                return

            group_color_labels = np.array([sub_to_group[int(s)] for s in all_subs])
            title_suffix = f"Train Set (n={len(all_subs)})"
            print(f"[*] Train mode: {len(all_subs)} samples with valid group labels")
        else:
            # 测试集：未分组的标记为 -1
            group_color_labels = np.array([sub_to_group.get(int(s), -1) for s in all_subs])
            title_suffix = f"Test Set (n={len(all_subs)})"
            ungrouped_count = np.sum(group_color_labels == -1)
            print(f"[*] Test mode: {len(all_subs)} samples, {ungrouped_count} ungrouped")

        def draw_plot(data_list, color_data, title, filename, is_task=False):
            if len(data_list) == 0:
                print(f"[Warning] No data for {title}")
                return

            data = np.concatenate(data_list, axis=0)

            # t-SNE降维
            perplexity = min(30, len(data) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            data_2d = tsne.fit_transform(data)
            data_2d = MinMaxScaler().fit_transform(data_2d)

            plt.figure(figsize=(10, 8))

            if is_task:
                # 按任务标签着色（二分类）
                cmap = 'coolwarm'
                scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=color_data,
                                      cmap=cmap, s=20, alpha=0.6, edgecolors='none')
                cbar = plt.colorbar(scatter)
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(['Low', 'High'])
            else:
                # 按 group 着色
                unique_groups = np.unique(color_data)
                plot_groups = unique_groups[unique_groups >= 0]

                colors = plt.cm.tab10(np.linspace(0, 1, max(len(plot_groups), 1)))

                for i, g in enumerate(plot_groups):
                    mask = color_data == g
                    plt.scatter(data_2d[mask, 0], data_2d[mask, 1],
                                c=[colors[i]], s=20, alpha=0.6,
                                label=f'Group {int(g)} (n={mask.sum()})', edgecolors='none')

                # 未分组的用灰色显示
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

            # 保存
            save_dir = os.path.join(self.config.OUTPUT_ROOT, 'tsne_results')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{filename}_{mode}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[+] Saved: {save_path}")

        # 绘制三张图
        print("[*] Plotting shared features by group...")
        draw_plot(shared_list, group_color_labels,
                  "Shared Features (by Group)", "shared_by_group", is_task=False)

        print("[*] Plotting private features by group...")
        draw_plot(private_list, group_color_labels,
                  "Private Features (by Group)", "private_by_group", is_task=False)

        print("[*] Plotting shared features by task label...")
        draw_plot(shared_list, all_tasks,
                  "Shared Features (by Task Label)", "shared_by_label", is_task=True)

    def visualize_tsne(self, model_path=None):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import MinMaxScaler

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)

        # 1. 维度修复逻辑 (保持之前的成功经验)
        for name, param in checkpoint.items():
            if 'leaner' in name and 'weight' in name:
                layer_name = name.replace('.weight', '')
                input_dim = param.shape[1]
                output_dim = param.shape[0]
                setattr(self.model, layer_name, nn.Linear(input_dim, output_dim).to(device))

        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

        shared_list, private_list = [], []
        subject_labels, task_labels = [], []

        # 2. 特征提取
        with torch.no_grad():
            for batch in self.test_data_loader:
                eeg, eog, emg, y, l, d = batch
                self.model(to_gpu(eeg), to_gpu(eog), to_gpu(emg), to_gpu(l), to_gpu(d), self.groups)

                shared_list.append(self.model.utt_shared_subject.cpu().numpy())
                private_list.append(self.model.utt_private_subject.cpu().numpy())
                subject_labels.append(d.cpu().numpy())
                task_labels.append(y.cpu().numpy())

        # 3. 执行基于 Group 的增强绘图
        self._plot_tsne_by_group(shared_list, private_list, subject_labels, task_labels)

    def _plot_tsne_by_group(self, shared_list, private_list, subject_labels, task_labels):
        all_subs = np.concatenate(subject_labels, axis=0)
        all_tasks = np.concatenate(task_labels, axis=0)

        # --- 核心逻辑：建立 Subject 到 Group 的映射 ---
        # 假设 self.groups 格式为: {group_id: [sub1, sub2, ...]}
        sub_to_group = {}
        for g_id, subs in self.groups.items():
            for s in subs:
                sub_to_group[int(s)] = int(g_id)

        # 将测试集中的每个点映射到其对应的 Group ID
        # 如果某个被试不在分组里，默认设为 -1
        group_color_labels = np.array([sub_to_group.get(int(s), -1) for s in all_subs])

        def draw_plot(data_list, color_data, title, filename, is_task=False):
            data = np.concatenate(data_list, axis=0)
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            data_2d = tsne.fit_transform(data)
            data_2d = MinMaxScaler().fit_transform(data_2d)

            plt.figure(figsize=(12, 11))

            # 使用 tab10 配色，因为 group_num 通常较小 (2, 3, 5 等)
            cmap = 'coolwarm' if is_task else 'tab10'
            scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=color_data,
                                  cmap=cmap, s=80, alpha=0.7, edgecolors='none')

            plt.title(title, fontsize=42, fontweight='bold', pad=30)
            plt.xticks([]);
            plt.yticks([])

            cbar = plt.colorbar(scatter)
            # 动态设置颜色条标签
            if is_task:
                cbar.set_ticks([0, 1]);
                cbar.set_ticklabels(['Low', 'High'])
            else:
                # 显示 Group 0, Group 1...
                unique_groups = np.unique(color_data)
                cbar.set_ticks(unique_groups)
                cbar.set_ticklabels([f'Group {int(g)}' for g in unique_groups])

            cbar.ax.tick_params(labelsize=26)

            os.makedirs('tsne_results', exist_ok=True)
            plt.savefig(f'tsne_results/{filename}.png', dpi=300)
            plt.close()

        # 4. 生成你要求的图
        # Shared 特征：按 Group 染色（预期：不同 Group 的点混在一起，证明去除了组间差异）
        draw_plot(shared_list, group_color_labels, "Shared Features (by Group)", "shared_by_group")

        # Private 特征：按 Group 染色（预期：形成 group_num 个明显的簇）
        draw_plot(private_list, group_color_labels, "Private Features (by Group)", "private_by_group")

        # 附加：Shared 特征按 Label 染色（验证性能）
        draw_plot(shared_list, all_tasks, "Shared Features (by Label)", "shared_by_label", is_task=True)

    def _save_config(self):
        """保存训练配置"""
        config_path = os.path.join(self.config.OUTPUT_ROOT,
                                   f'{self.train_config.name}_config.txt')
        with open(config_path, 'w') as f:
            f.write(f"Experiment: {self.train_config.name}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Task: {getattr(self.train_config, 'task', 'valence')}\n")
            f.write(f"Data path: {self.config.BASE_DATA_PATH}\n")
            f.write(f"Train samples: {len(self.train_data_loader.dataset)}\n")
            f.write(f"Valid samples: {len(self.dev_data_loader.dataset)}\n")
            f.write(f"Test samples: {len(self.test_data_loader.dataset)}\n")
            f.write(f"Model params: {self.params}\n")

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):
        if self.model is None:
            self.model = models.MDNet(self.train_config, self.params)

        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.params["learning_rate"],
                weight_decay=self.train_config.weight_decay)

        initialize_weights(self.model)
        print(f"Model built on {'CUDA' if cuda and torch.cuda.is_available() else 'CPU'}")

    @time_desc_decorator('Training Start!')
    def train(self, index=1):
        curr_patience = patience = self.train_config.patience
        num_trials = 1
        best_valid_acc = 0
        best_acc_f1 = 0

        # 初始化保存路径变量，防止未赋值引用
        model_path = os.path.join(self.checkpoint_dir, f'model_{self.train_config.name}.pt')
        optim_path = os.path.join(self.checkpoint_dir, f'optim_{self.train_config.name}.pt')

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")

        self.loss_diff = DiffLoss()
        self.loss_cmd = CMD()

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

        history = {
            'train_loss': [], 'valid_loss': [],
            'train_acc': [], 'valid_acc': [],
            'valid_f1': []
        }
        best_metrics = {'epoch': 0, 'valid_acc': 0, 'valid_f1': 0}

        for e in range(self.train_config.n_epoch):
            self.epoch_i = e + 1
            self.model.train()

            train_loss, train_acc, train_cnt = 0.0, 0, 0

            for batch_idx, batch in enumerate(self.train_data_loader):
                eeg, mod2, mod3, y, l, d = batch
                eeg, mod2, mod3, y, l, d = to_gpu(eeg), to_gpu(mod2), to_gpu(mod3), to_gpu(y), to_gpu(l), to_gpu(d)

                self.model.zero_grad()

                y_tilde = self.model(eeg, mod2, mod3, l, d, self.groups)
                y = y.view(-1)

                _, predicted = torch.max(y_tilde, 1)
                train_acc += predicted.eq(y).sum().item()
                train_cnt += y.shape[0]

                cls_loss = self.criterion(y_tilde, y)
                diff_loss = self.get_diff_loss()
                cmd_loss = self.get_cmd_loss()

                logits_homo, reprs_homo, logits_hetero, reprs_hetero = self._prepare_distillation_features()
                graph_loss_homo, graph_loss_hetero = self._compute_graph_distillation_loss(
                    logits_homo, reprs_homo, logits_hetero, reprs_hetero
                )

                loss = (self.train_config.cls_weight * cls_loss +
                        self.params["diff_weight"] * diff_loss +
                        self.params["sim_weight"] * cmd_loss +
                        graph_loss_homo + graph_loss_hetero)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                # 减少打印频率，每20个batch打印一次
                if (batch_idx + 1) % 20 == 0:
                    print(f'Epoch {e + 1} [{batch_idx + 1}/{len(self.train_data_loader)}] Loss: {loss.item():.4f}')

            epoch_train_loss = train_loss / len(self.train_data_loader)
            epoch_train_acc = train_acc / train_cnt

            # 验证
            valid_loss, valid_acc, valid_f1, valid_precision, valid_recall = self._eval_epoch(mode="dev")

            history['train_loss'].append(epoch_train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_acc'].append(epoch_train_acc)
            history['valid_acc'].append(valid_acc)
            history['valid_f1'].append(valid_f1)

            # TensorBoard 写入
            self.writer.add_scalars('Loss', {'train': epoch_train_loss, 'valid': valid_loss}, e)
            self.writer.add_scalars('Accuracy', {'train': epoch_train_acc, 'valid': valid_acc}, e)
            self.writer.add_scalar('F1/valid', valid_f1, e)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], e)

            print(f"Epoch {e + 1}/{self.train_config.n_epoch} | "
                  f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
                  f"Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f} F1: {valid_f1:.4f}")

            # 保存最佳模型策略
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_acc_f1 = valid_f1
                best_metrics = {'epoch': e + 1, 'valid_acc': valid_acc, 'valid_f1': valid_f1}

                torch.save(self.model.state_dict(), model_path)
                torch.save(self.optimizer.state_dict(), optim_path)
                print(f"  [*] Best model saved (Acc: {valid_acc:.4f})")
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    num_trials -= 1
                    curr_patience = patience
                    # 读取最佳模型进行学习率衰减或继续训练
                    if os.path.exists(model_path):
                        self.model.load_state_dict(torch.load(model_path))
                        self.optimizer.load_state_dict(torch.load(optim_path))
                        lr_scheduler.step()
                        print(
                            f"  [!] Patience exhausted. Loading best model and decaying LR to {self.optimizer.param_groups[0]['lr']:.6f}")
                    else:
                        print("  [!] Patience exhausted but no model saved yet.")

            if num_trials <= 0:
                print("Early stopping triggered.")
                break

        self._plot_history(history, best_metrics)
        self.writer.close()

        # 最终使用最佳模型进行测试
        test_acc, test_f1, test_kappa = self._final_evaluation(index)
        return test_acc, test_f1, test_kappa, best_metrics

    def _prepare_distillation_features(self):
        """准备图蒸馏的特征（复用原始代码逻辑）"""
        # 从模型获取中间特征
        logits_homo = [
            self.model.logits_eeg_low,
            self.model.logits_mod2_low,
            self.model.logits_mod3_low
        ]
        reprs_homo = [
            self.model.repr_eeg_low,
            self.model.repr_mod2_low,
            self.model.repr_mod3_low
        ]
        logits_hetero = [
            self.model.logits_eeg_high,
            self.model.logits_mod2_high,
            self.model.logits_mod3_high
        ]
        reprs_hetero = [
            self.model.repr_eeg_high,
            self.model.repr_mod2_high,
            self.model.repr_mod3_high
        ]

        return (torch.stack(logits_homo), torch.stack(reprs_homo),
                torch.stack(logits_hetero), torch.stack(reprs_hetero))

    def _compute_graph_distillation_loss(self, logits_homo, reprs_homo, logits_hetero, reprs_hetero):
        """计算图蒸馏损失（复用原始逻辑，简化版）"""
        batch_size = logits_homo.size(1)

        # 初始化蒸馏核（每次前向传播都新建，保持与原始代码一致）
        model_distill_homo = get_distillation_kernel_homo.DistillationKernel(
            n_classes=2, hidden_size=50, gd_size=64,
            to_idx=[0, 1, 2], from_idx=[0, 1, 2],
            gd_prior=softmax([0, 0, 1, 0, 1, 0], 0.25),
            gd_reg=10, w_losses=[1, 10], metric='l1', alpha=1 / 8,
            hyp_params=self.train_config, batch_size=batch_size
        ).cuda()

        model_distill_hetero = get_distillation_kernel.DistillationKernel(
            n_classes=2, hidden_size=100, gd_size=64,
            to_idx=[0, 1, 2], from_idx=[0, 1, 2],
            gd_prior=softmax([0, 0, 1, 0, 1, 1], 0.25),
            gd_reg=10, w_losses=[1, 10], metric='l1', alpha=1 / 8,
            hyp_params=self.train_config, batch_size=batch_size
        ).cuda()

        # 前向计算边权重
        edges_homo, _ = model_distill_homo(logits_homo, reprs_homo)
        edges_hetero, _ = model_distill_hetero(logits_hetero, reprs_hetero)

        # 计算损失
        loss_reg_homo, loss_logit_homo, _ = model_distill_homo.distillation_loss(
            logits_homo, reprs_homo, edges_homo)
        loss_homo = 0.05 * (loss_logit_homo + loss_reg_homo)

        loss_reg_hetero, loss_logit_hetero, _ = model_distill_hetero.distillation_loss(
            logits_hetero, reprs_hetero, edges_hetero)
        loss_hetero = 0.05 * (loss_logit_hetero + loss_reg_hetero)

        return loss_homo, loss_hetero

    def _eval_epoch(self, mode="dev"):
        """评估一个epoch"""
        self.model.eval()
        dataloader = {
            "train": self.train_data_loader,
            "dev": self.dev_data_loader,
            "test": self.test_data_loader
        }[mode]

        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                eeg, mod2, mod3, y, l, d = batch
                eeg = to_gpu(eeg)
                mod2 = to_gpu(mod2)
                mod3 = to_gpu(mod3)
                y = to_gpu(y)
                l = to_gpu(l)
                d = to_gpu(d)

                y_tilde = self.model(eeg, mod2, mod3, l, d, self.groups)
                y = y.view(-1)

                loss = self.criterion(y_tilde, y)
                total_loss += loss.item()

                _, predicted = torch.max(y_tilde, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)

        avg_loss = total_loss / len(dataloader)

        return avg_loss, accuracy, f1, precision, recall

    def _final_evaluation(self, index=1):
        """最终测试评估"""
        print("\n" + "=" * 50)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 50)

        # 加载最佳模型
        model_path = os.path.join(self.checkpoint_dir, f'model_{self.train_config.name}.pt')
        self.model.load_state_dict(torch.load(model_path))

        test_loss, test_acc, test_f1, precision, recall = self._eval_epoch(mode="test")

        # 详细评估（混淆矩阵等）
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.test_data_loader:
                eeg, mod2, mod3, y, l, d = batch
                eeg, mod2, mod3 = to_gpu(eeg), to_gpu(mod2), to_gpu(mod3)
                y_tilde = self.model(eeg, mod2, mod3, l, d, self.groups)
                _, predicted = torch.max(y_tilde, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.view(-1).cpu().numpy())

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        kappa_score = kappa(cm)

        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Test Kappa: {kappa_score:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, digits=4))
        print("Confusion Matrix:")
        print(cm)

        # 保存混淆矩阵图
        self._plot_confusion_matrix(cm, test_acc, test_f1, kappa_score, index)

        # 记录到TensorBoard
        self.writer.add_scalar('Test/Accuracy', test_acc, 0)
        self.writer.add_scalar('Test/F1', test_f1, 0)
        self.writer.add_scalar('Test/Kappa', kappa_score, 0)

        # 保存结果到文本
        result_path = os.path.join(self.result_dir, f'{self.train_config.name}_test_results.txt')
        with open(result_path, 'w') as f:
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test F1: {test_f1:.4f}\n")
            f.write(f"Test Kappa: {kappa_score:.4f}\n")
            f.write(f"Test Precision: {precision:.4f}\n")
            f.write(f"Test Recall: {recall:.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))

        return test_acc, test_f1, kappa_score

    def _plot_history(self, history, best_metrics):
        """绘制训练历史曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失
        axes[0, 0].plot(history['train_loss'], label='Train')
        axes[0, 0].plot(history['valid_loss'], label='Valid')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 准确率
        axes[0, 1].plot(history['train_acc'], label='Train')
        axes[0, 1].plot(history['valid_acc'], label='Valid')
        axes[0, 1].axvline(best_metrics['epoch'] - 1, color='r', linestyle='--', label='Best')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1
        axes[1, 0].plot(history['valid_f1'], label='Valid F1', color='purple')
        axes[1, 0].axvline(best_metrics['epoch'] - 1, color='r', linestyle='--')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 损失组成（最后一个epoch）
        axes[1, 1].bar(['Total', 'Cls', 'Diff', 'Sim'],
                       [history['train_loss'][-1], 0, 0, 0])  # 简化版
        axes[1, 1].set_title('Loss Components (Last Epoch)')

        plt.tight_layout()

        # 保存
        curve_path = os.path.join(self.curve_dir, f'{self.train_config.name}_curves.png')
        fig.savefig(curve_path, dpi=300, bbox_inches='tight')
        self.writer.add_figure('Training/Curves', fig, 0)
        plt.close(fig)
        print(f"Training curves saved to: {curve_path}")

    def _plot_confusion_matrix(self, cm, acc, f1, kappa, index):
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')

        # 添加数值
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=12)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix\nAcc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}')
        plt.colorbar(im, ax=ax)

        # 保存
        cm_path = os.path.join(self.result_dir, f'{index}_cm_{self.train_config.name}.png')
        fig.savefig(cm_path, dpi=300, bbox_inches='tight')
        self.writer.add_figure('Test/ConfusionMatrix', fig, 0)
        plt.close(fig)
        print(f"Confusion matrix saved to: {cm_path}")

    # 损失计算方法（保持原有逻辑）
    def get_cmd_loss(self):
        loss = self.loss_cmd(self.model.utt_shared_eeg_1, self.model.utt_shared_eog_1, 5)
        loss += self.loss_cmd(self.model.utt_shared_eeg_1, self.model.utt_shared_emg_1, 5)
        loss += self.loss_cmd(self.model.utt_shared_eog_1, self.model.utt_shared_emg_1, 5)
        return loss / 3.0

    def get_diff_loss(self):
        shared_eeg = self.model.utt_shared_eeg_1
        shared_eog = self.model.utt_shared_eog_1
        shared_emg = self.model.utt_shared_emg_1
        private_eeg = self.model.utt_private_eeg_1
        private_eog = self.model.utt_private_eog_1
        private_emg = self.model.utt_private_emg_1

        loss = (self.loss_diff(private_eeg, shared_eeg) +
                self.loss_diff(private_eog, shared_eog) +
                self.loss_diff(private_emg, shared_emg) +
                self.loss_diff(private_emg, private_eeg) +
                self.loss_diff(private_eog, private_emg) +
                self.loss_diff(private_eeg, private_eog))
        return loss


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # ========== 1. 全局基础配置 ==========
    DataConfig.BASE_DATA_PATH = '/code/clisa/clisa/Downstream_dataset/AdaBrain-Bench-LaBraM-fusion/preprocessing/MER/data'
    base_output_dir = '/code/scw/MER/output_random_group_ablation_valence'  # 改为随机分组实验的输出目录

    # 降低 Batch Size 以避免 OOM
    DataConfig.batch_size = 32

    set_seed(42)

    print("=" * 80)
    print("MER Dataset - Random Grouping Ablation Study (10-Fold Cross Validation)")
    print("=" * 80)

    # ========== 2. 定义要测试的 group_num 参数范围 ==========
    GROUP_NUM_OPTIONS = [2, 3, 4, 5, 6]  # 你要测试的 group_num 值

    # ========== 3. 获取 10 份 Folds（所有 group_num 共享相同的 fold 划分）==========
    print("\n[Step 1] 生成 10-Fold 划分（所有 group_num 共享）")
    folds = get_10_folds_subjects(DataConfig, seed=42)

    # 保存 fold 划分，确保可复现
    fold_save_path = os.path.join(base_output_dir, 'fold_split.json')
    os.makedirs(base_output_dir, exist_ok=True)
    with open(fold_save_path, 'w') as f:
        json.dump({f"fold_{i}": [int(x) for x in fold] for i, fold in enumerate(folds)}, f, indent=2)
    print(f"Fold 划分已保存: {fold_save_path}")

    # 基础超参数（group_num 会被动态覆盖）
    base_params = {
        "group_num": None,  # 会被动态设置
        "diff_weight": 0.05,
        "sim_weight": 0.001,
        "learning_rate": 1e-4
    }

    # ========== 4. 外层循环：遍历不同的 group_num ==========
    all_group_results = {}  # 记录每个 group_num 的结果
    start_time_total = time.time()

    for group_num in GROUP_NUM_OPTIONS:
        print(f"\n{'#' * 80}")
        print(f'# 开始测试 Random Grouping Baseline: group_num = {group_num}')
        print(f'{"#" * 80}')

        group_output_dir = os.path.join(base_output_dir, f'group_num_{group_num}')
        os.makedirs(group_output_dir, exist_ok=True)

        # 当前 group_num 的所有 fold 结果
        fold_results_for_this_group = []
        start_time_group = time.time()

        # ========== 5. 内层循环：10-Fold CV ==========
        for fold_idx in range(10):
            print(f"\n{'=' * 60}")
            print(f'Group={group_num} | Fold {fold_idx + 1}/10')
            print(f"{'=' * 60}")

            # 强制清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- A. 设置当前 Fold 的输出路径 ---
            fold_output_dir = os.path.join(group_output_dir, f'fold_{fold_idx + 1}')
            DataConfig.OUTPUT_ROOT = fold_output_dir
            DataConfig.LOG_DIR = os.path.join(fold_output_dir, 'logs')
            DataConfig.CHECKPOINT_DIR = os.path.join(fold_output_dir, 'checkpoints')
            DataConfig.RESULT_DIR = os.path.join(fold_output_dir, 'results')
            DataConfig.CURVE_DIR = os.path.join(fold_output_dir, 'curves')
            setup_directories(DataConfig)

            # --- B. 数据集划分 ---
            train_subs, val_subs, test_subs = get_split_for_fold(folds, fold_idx, seed=42)
            print(f"[Split] Train: {len(train_subs)} | Val: {len(val_subs)} | Test: {len(test_subs)}")

            # --- C. 关键修改：使用随机分组替代相关性聚类 ---
            print(f"\n[Random Grouping] 对训练集进行随机分组 (k={group_num})...")
            try:
                # 使用随机分组函数（消融实验）
                groups = get_random_subject_groups(
                    train_subjects=train_subs,
                    test_subjects=test_subs,
                    num_groups=group_num,
                    fold_idx=fold_idx,
                    seed=42,  # 保证可复现
                    save_dir=os.path.join(group_output_dir, 'random_groupings')  # 保存分组结果
                )

                # 验证分组数（修复：groups现在是list，不是dict）
                actual_groups = len([g for g in groups if len(g) > 0])
                print(f"[Random Grouping] 实际分组数: {actual_groups}")
                print(f"[Random Grouping] 各组大小: {[len([s for s in g if s in train_subs]) for g in groups]}")

            except Exception as e:
                print(f"[Error] 随机分组失败: {e}")
                import traceback

                traceback.print_exc()
                continue  # 跳过这个 fold

            # --- D. 构建 DataLoader ---
            try:
                train_dataset = MERMultimodalDataset(train_subs, config=DataConfig, task='valence')
                val_dataset = MERMultimodalDataset(val_subs, config=DataConfig, task='valence')
                test_dataset = MERMultimodalDataset(test_subs, config=DataConfig, task='valence')

                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=DataConfig.batch_size,
                    shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=DataConfig.batch_size,
                    shuffle=False, num_workers=2, pin_memory=True)
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=DataConfig.batch_size,
                    shuffle=False, num_workers=2, pin_memory=True)
            except Exception as e:
                print(f"[Error] DataLoader 构建失败: {e}")
                continue

            # --- E. 配置并训练模型 ---
            current_params = base_params.copy()
            current_params["group_num"] = group_num  # 设置当前 group_num

            train_config = DataConfig()
            train_config.name = f"random_g{group_num}_fold{fold_idx + 1}"
            train_config.n_epoch = 60
            train_config.patience = 8
            train_config.cls_weight = 1.0
            train_config.weight_decay = 1e-4
            train_config.optimizer = torch.optim.Adam

            solver = None
            try:
                solver = Solver(
                    train_config,
                    train_loader,
                    val_loader,
                    test_loader,
                    groups,  # 使用随机分组得到的分组
                    current_params,
                    seed=42,
                    config=DataConfig
                )
                solver.build()

                # 训练并获取测试结果
                test_acc, test_f1, test_kappa, best_metrics = solver.train(index=fold_idx + 1)


                # 记录结果
                result_entry = {
                    "group_num": group_num,
                    "fold": fold_idx + 1,
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "test_kappa": test_kappa,
                    "best_val_acc": best_metrics['valid_acc'],
                    "best_epoch": best_metrics['epoch'],
                    "actual_groups": actual_groups,
                    "group_sizes": [len([s for s in g if s in train_subs]) for g in groups],  # 修复：去掉.values()
                    "grouping_method": "random"  # 标记为随机分组
                }
                fold_results_for_this_group.append(result_entry)

                print(f"\n[Fold {fold_idx + 1} Success] Acc={test_acc:.4f}, F1={test_f1:.4f}")

            except Exception as e:
                print(f"\n[Error] Fold {fold_idx + 1} 训练失败!")
                import traceback

                traceback.print_exc()

            finally:
                if solver is not None:
                    del solver
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ========== 6. 当前 group_num 的总结 ==========
        group_time = (time.time() - start_time_group) / 60  # 分钟

        print(f"\n{'=' * 80}")
        print(f'Random Group Num = {group_num} 完成 | 耗时: {group_time:.1f} min')
        print(f"{'=' * 80}")

        if fold_results_for_this_group:
            # 计算统计指标
            accs = [r['test_acc'] for r in fold_results_for_this_group]
            f1s = [r['test_f1'] for r in fold_results_for_this_group]
            kappas = [r['test_kappa'] for r in fold_results_for_this_group]

            group_summary = {
                'group_num': group_num,
                'grouping_method': 'random',
                'n_folds_completed': len(fold_results_for_this_group),
                'test_acc_mean': np.mean(accs),
                'test_acc_std': np.std(accs),
                'test_f1_mean': np.mean(f1s),
                'test_f1_std': np.std(f1s),
                'test_kappa_mean': np.mean(kappas),
                'test_kappa_std': np.std(kappas),
                'fold_results': fold_results_for_this_group
            }

            all_group_results[group_num] = group_summary

            # 打印当前 group_num 的统计
            print(f"\n[Random Group={group_num} 统计结果]")
            print(f"  完成 Fold 数: {len(fold_results_for_this_group)}/10")
            print(f"  Test Acc:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(f"  Test F1:   {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(f"  Test Kappa: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")

            # 保存当前 group_num 的详细结果
            with open(os.path.join(group_output_dir, 'summary.json'), 'w') as f:
                json.dump(group_summary, f, indent=2)
        else:
            print(f"[Warning] Group={group_num} 没有成功完成的 Fold")

    # ========== 7. 所有 group_num 的最终总结 ==========
    print(f"\n{'=' * 80}")
    print("随机分组消融实验完成 - 最终总结报告")
    print(f"{'=' * 80}")

    if all_group_results:
        # 创建对比表格
        print(
            f"\n{'Group Num':<10} | {'Acc Mean':<10} | {'Acc Std':<10} | {'F1 Mean':<10} | {'Kappa Mean':<10} | {'Folds':<6}")
        print("-" * 80)

        # 按准确率排序找出最佳
        sorted_results = sorted(all_group_results.items(),
                                key=lambda x: x[1]['test_acc_mean'],
                                reverse=True)

        for gnum, res in sorted_results:
            print(f"{gnum:<10} | {res['test_acc_mean']:<10.4f} | {res['test_acc_std']:<10.4f} | "
                  f"{res['test_f1_mean']:<10.4f} | {res['test_kappa_mean']:<10.4f} | {res['n_folds_completed']:<6}")

        # 最佳结果
        best_group_num = sorted_results[0][0]
        best_result = sorted_results[0][1]

        print(f"\n{'=' * 80}")
        print(f"最佳 Random Group Num: {best_group_num}")
        print(f"  Test Acc:  {best_result['test_acc_mean']:.4f} ± {best_result['test_acc_std']:.4f}")
        print(f"  Test F1:   {best_result['test_f1_mean']:.4f} ± {best_result['test_f1_std']:.4f}")
        print(f"  Test Kappa: {best_result['test_kappa_mean']:.4f} ± {best_result['test_kappa_std']:.4f}")
        print(f"{'=' * 80}")

        # 保存完整总结
        final_summary = {
            'best_group_num': int(best_group_num),
            'all_results': {int(k): v for k, v in all_group_results.items()},
            'search_space': GROUP_NUM_OPTIONS,
            'grouping_method': 'random',
            'note': 'Ablation study: random grouping vs PSD-based clustering'
        }

        with open(os.path.join(base_output_dir, 'final_summary.json'), 'w') as f:
            json.dump(final_summary, f, indent=2)

        # 绘制对比图
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            group_nums = list(all_group_results.keys())
            acc_means = [all_group_results[g]['test_acc_mean'] for g in group_nums]
            acc_stds = [all_group_results[g]['test_acc_std'] for g in group_nums]
            f1_means = [all_group_results[g]['test_f1_mean'] for g in group_nums]
            kappa_means = [all_group_results[g]['test_kappa_mean'] for g in group_nums]

            # Accuracy
            axes[0].errorbar(group_nums, acc_means, yerr=acc_stds, marker='o', capsize=5)
            axes[0].set_xlabel('Group Num')
            axes[0].set_ylabel('Test Accuracy')
            axes[0].set_title('Random Grouping: Accuracy vs Group Num')
            axes[0].grid(True, alpha=0.3)

            # F1
            axes[1].plot(group_nums, f1_means, marker='o')
            axes[1].set_xlabel('Group Num')
            axes[1].set_ylabel('Test F1')
            axes[1].set_title('Random Grouping: F1 vs Group Num')
            axes[1].grid(True, alpha=0.3)

            # Kappa
            axes[2].plot(group_nums, kappa_means, marker='o')
            axes[2].set_xlabel('Group Num')
            axes[2].set_ylabel('Test Kappa')
            axes[2].set_title('Random Grouping: Kappa vs Group Num')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(os.path.join(base_output_dir, 'random_group_comparison.png'), dpi=300)
            print(f"\n对比图已保存: {os.path.join(base_output_dir, 'random_group_comparison.png')}")
            plt.close(fig)

        except Exception as e:
            print(f"绘图失败: {e}")

    else:
        print("[Error] 没有成功完成的实验")

    total_time = (time.time() - start_time_total) / 3600
    print(f"\n总耗时: {total_time:.2f} 小时")
    print(f"所有结果保存在: {base_output_dir}")
    print(f"提示：此结果为随机分组消融实验，用于与PSD聚类分组对比")