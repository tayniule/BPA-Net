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
#from load_data_test import preprocess_subject_data, calculate_subject_correlations_all_samples, stepwise_clean
from torch.utils.data import Dataset
import seaborn as sns

class DataConfig:
    """
    数据路径配置类 - 根据你的实际情况修改
    """
    # 原始数据根目录（被试文件夹所在位置）
    BASE_DATA_PATH = '/eds-storage/scw/MER/data'  # 例如：'/home/user/eeg_data/processed'
    dataset_name = 'MER'  # <-- 添加这一行，可选值: 'DEAP', 'HCI', 'MER'
    name = 'MER'
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
    optimizer = torch.optim.Adam
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


# ==================== 随机种子固定函数 ====================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


# ==================== 新增：从相关性分析结果加载分组 ====================
def load_subject_groups(groups_file='/code/scw/MER/output_mer_experiment/correlation_analysis/subject_groups.pkl'):
    """
    加载预计算的相关性分组结果

    返回:
        groups: 分组列表，每个元素是一组被试ID
        range_groups: 按相关性范围的分组字典
        all_subjects: 所有被试ID列表
    """
    if not os.path.exists(groups_file):
        print(f"警告: 分组文件不存在 {groups_file}，将使用原始划分方法")
        return None, None, None

    with open(groups_file, 'rb') as f:
        data = pickle.load(f)

    groups = data.get('groups', [])
    range_groups = data.get('range_groups', {})
    all_subjects = data.get('all_subjects', [])

    print(f"已加载分组配置: {groups_file}")
    print(f"  总被试数: {len(all_subjects)}")
    print(f"  分组数: {len(groups)}")
    for i, g in enumerate(groups):
        range_name = [k for k, v in range_groups.items() if set(v) == set(g)]
        name = range_name[0] if range_name else f"Group{i}"
        print(f"    {name}: {len(g)}人 - {sorted(g)[:10]}{'...' if len(g) > 10 else ''}")

    return groups, range_groups, all_subjects


def stratified_split_from_groups(groups, range_groups, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    基于相关性分组进行分层划分，确保每个组都按比例分配到训练/验证/测试集

    划分比例: 7:1:2 (train:val:test)
    """
    np.random.seed(seed)

    train_subs, val_subs, test_subs = [], [], []

    # 按优先级处理各组（高相关性优先）
    priority_order = ["≥0.7", "0.5~0.7", "0.3~0.5", "<0.3"]

    print(f"\n分层划分 (比例 train:val:test = {train_ratio}:{val_ratio}:{test_ratio}):")

    for range_key in priority_order:
        if range_key not in range_groups or not range_groups[range_key]:
            continue

        subjects = list(range_groups[range_key])
        np.random.shuffle(subjects)

        n = len(subjects)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # 测试集拿剩下的，避免舍入误差
        n_test = n - n_train - n_val

        train_subs.extend(subjects[:n_train])
        val_subs.extend(subjects[n_train:n_train + n_val])
        test_subs.extend(subjects[n_train + n_val:])

        print(f"  {range_key}: 总数={n}, 训练={n_train}, 验证={n_val}, 测试={n_test}")

    # 去重并排序
    train_subs = sorted(set(train_subs))
    val_subs = sorted(set(val_subs))
    test_subs = sorted(set(test_subs))

    # 验证无重叠
    assert len(set(train_subs) & set(val_subs)) == 0, "训练集和验证集重叠!"
    assert len(set(train_subs) & set(test_subs)) == 0, "训练集和测试集重叠!"
    assert len(set(val_subs) & set(test_subs)) == 0, "验证集和测试集重叠!"

    # 验证比例
    total = len(train_subs) + len(val_subs) + len(test_subs)
    print(f"\n最终划分:")
    print(f"  训练集: {len(train_subs)}人 ({len(train_subs) / total:.1%})")
    print(f"  验证集: {len(val_subs)}人 ({len(val_subs) / total:.1%})")
    print(f"  测试集: {len(test_subs)}人 ({len(test_subs) / total:.1%})")
    print(f"  总计: {total}人")

    return train_subs, val_subs, test_subs

def extract_psd_for_subject(subject_id, config=DataConfig, max_trials=50):
    """
    为单个被试提取PSD特征（用于被试间相关性计算）

    从你的预处理代码可见，数据shape是(20, 800)，我们提取EEG部分(前18通道)的PSD
    """
    subject_dir = os.path.join(config.BASE_DATA_PATH, str(subject_id))
    if not os.path.exists(subject_dir):
        return None

    pkl_files = [f for f in os.listdir(subject_dir) if f.endswith('.pkl')]
    if not pkl_files:
        return None

    psd_features = []

    # 限制处理的trial数量，避免内存溢出
    for pkl_file in pkl_files[:max_trials]:
        try:
            with open(os.path.join(subject_dir, pkl_file), 'rb') as f:
                data_dict = pickle.load(f)

            sample = data_dict['sample']  # shape: (20, 800)

            # 只使用EEG通道（前18通道）计算PSD
            eeg_data = sample[:config.EEG_CHANNELS, :]  # (18, 800)

            # 简单的PSD估计（使用FFT的幅值平方）
            fft_vals = np.fft.rfft(eeg_data, axis=1)  # (18, 401)
            psd = np.abs(fft_vals) ** 2  # 功率谱密度

            # 在频率维度上平均，然后在通道维度上平均，得到标量特征
            psd_mean = np.mean(psd, axis=1)  # (18,)
            psd_feature = np.mean(psd_mean)  # 标量

            psd_features.append(psd_feature)

        except Exception as e:
            continue

    if psd_features:
        return np.mean(psd_features)  # 该被试的平均PSD特征
    return None


def calculate_subject_correlations_simple(subject_ids, config=DataConfig):
    """
    简化的被试间相关性计算

    由于预处理代码没有提供复杂的相关性计算函数，
    这里使用基于PSD特征的简单相关性估计
    """
    print("提取被试PSD特征...")
    subject_features = {}

    for sub_id in subject_ids:
        feat = extract_psd_for_subject(sub_id, config)
        if feat is not None:
            subject_features[sub_id] = feat

    if len(subject_features) < 2:
        print("警告：被试数量不足，使用随机分组")
        return None, list(subject_features.keys())

    # 构建特征向量（这里简化为1维特征，实际可以扩展）
    sub_list = sorted(subject_features.keys())
    feat_array = np.array([subject_features[s] for s in sub_list]).reshape(-1, 1)

    # 计算相关性矩阵（由于特征是1维，这里使用特征差异的倒数作为相似度）
    n = len(sub_list)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # 使用特征相似度（越接近1表示越相似）
                diff = abs(feat_array[i] - feat_array[j])
                max_val = max(abs(feat_array[i]), abs(feat_array[j]))
                if max_val > 0:
                    sim = 1.0 / (1.0 + diff / max_val)
                else:
                    sim = 1.0
                corr_matrix[i, j] = sim

    return corr_matrix, sub_list

# ==================== 修改后的数据划分函数 ====================
def get_73_subject_split(config=DataConfig, train_ratio=0.7, val_ratio=0.1, seed=42):
    """
    扫描被试文件夹，基于数据分布划分为训练/验证/测试集

    适配你的预处理数据结构：被试id/{sub}-{stimulus}-{window}.pkl
    """
    np.random.seed(seed)

    base_path = config.BASE_DATA_PATH
    available_subs = []

    print(f"扫描数据路径: {base_path}")

    # 1. 扫描实际存在的被试
    for sub_id in config.SUBJECT_RANGE:
        sub_dir = os.path.join(base_path, str(sub_id))
        if os.path.exists(sub_dir) and os.path.isdir(sub_dir):
            # 检查是否有pkl文件
            pkl_files = [f for f in os.listdir(sub_dir) if f.endswith('.pkl')]
            if pkl_files:
                available_subs.append(sub_id)

    if not available_subs:
        raise ValueError(f"在 {base_path} 中没有找到任何有效被试数据！")

    print(f"找到 {len(available_subs)} 名有效被试")

    # 2. 尝试计算被试间相关性进行分层划分
    corr_matrix, valid_subs = calculate_subject_correlations_simple(available_subs, config)

    # 3. 分层划分逻辑
    if corr_matrix is not None and len(valid_subs) >= 4:
        # 基于相关性分层
        n_subs = len(valid_subs)
        range_groups = {"high": [], "medium": [], "low": []}

        for i in range(n_subs):
            # 计算该被试与其他被试的平均相关性
            non_diag = [j for j in range(n_subs) if j != i]
            avg_corr = np.mean(corr_matrix[i, non_diag])

            if avg_corr >= 0.7:
                range_groups["high"].append(i)
            elif avg_corr >= 0.4:
                range_groups["medium"].append(i)
            else:
                range_groups["low"].append(i)

        # 清理分组
        for k in range_groups:
            range_groups[k] = sorted(set(range_groups[k]))

        print(f"分层结果: High={len(range_groups['high'])}, "
              f"Medium={len(range_groups['medium'])}, "
              f"Low={len(range_groups['low'])}")
    else:
        # 随机分组
        print("使用随机分层")
        range_groups = {"all": list(range(len(valid_subs)))}

    # 4. 按比例划分
    train_subs, val_subs, test_subs = [], [], []

    for group_name, member_indices in range_groups.items():
        if not member_indices:
            continue

        real_ids = [valid_subs[idx] for idx in member_indices]
        np.random.shuffle(real_ids)

        n = len(real_ids)
        tr_cut = int(n * train_ratio)
        val_cut = int(n * (train_ratio + val_ratio))

        train_subs.extend(real_ids[:tr_cut])
        val_subs.extend(real_ids[tr_cut:val_cut])
        test_subs.extend(real_ids[val_cut:])

    print(f"\n划分完成: 训练集 {len(train_subs)}人, "
          f"验证集 {len(val_subs)}人, 测试集 {len(test_subs)}人")

    return sorted(train_subs), sorted(val_subs), sorted(test_subs)

def get_10_folds_subjects(config, seed=42):
    np.random.seed(seed)
    base_path = config.BASE_DATA_PATH
    available_subs = []
    for sub_id in config.SUBJECT_RANGE:
        sub_dir = os.path.join(base_path, str(sub_id))
        if os.path.exists(sub_dir) and os.path.isdir(sub_dir):
            pkl_files = [f for f in os.listdir(sub_dir) if f.endswith('.pkl')]
            if pkl_files:
                available_subs.append(sub_id)
    np.random.shuffle(available_subs)
    return [list(arr) for arr in np.array_split(available_subs, 10)]

def get_split_for_fold(folds, fold_idx, seed=42):
    np.random.seed(seed + fold_idx)
    test_subs = folds[fold_idx]
    rest_subs = []
    for i, f in enumerate(folds):
        if i != fold_idx:
            rest_subs.extend(f)
    np.random.shuffle(rest_subs)
    val_size = max(1, int(len(rest_subs) * 0.10))
    val_subs = rest_subs[:val_size]
    train_subs = rest_subs[val_size:]
    return sorted(train_subs), sorted(val_subs), sorted(test_subs)

def create_train_groups_dynamic(train_subs, num_groups=3, seed=42):
    np.random.seed(seed)
    subs_copy = list(train_subs)
    np.random.shuffle(subs_copy)
    return [list(x) for x in np.array_split(subs_copy, num_groups)]

# ==================== 优化的Dataset类 ====================
class MERMultimodalDataset(Dataset):
    """
    MER多模态数据集类

    适配你的预处理数据结构：
    - 文件: {subject_id}/{subject_id}-{stimulus_id}-{window_id}.pkl
    - 内容: {'sample': (20, 800), 'label': [valence_label, arousal_label]}
    - 通道: 0-17 EEG, 18 GSR, 19 PPG
    """

    def __init__(self, subject_ids, config=DataConfig, task='arousal',
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # def visualize_tsne(self, model_path=None):
    #
    #     # 1. 自动定义设备，解决 AttributeError: 'Solver' object has no attribute 'device'
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    #     print(f"Loading model from {model_path} for t-SNE...")
    #     checkpoint = torch.load(model_path, map_location=device)
    #
    #     # 2. 强行修复 leaner 层的形状 [64, 1] -> [64, 528/16896]
    #     # 这是解决 RuntimeError 的核心
    #     for name, param in checkpoint.items():
    #         if 'leaner' in name and 'weight' in name:
    #             layer_name = name.replace('.weight', '')
    #             input_dim = param.shape[1]
    #             output_dim = param.shape[0]
    #             print(f"Force re-initializing layer {layer_name} to shape ({output_dim}, {input_dim})")
    #             setattr(self.model, layer_name, nn.Linear(input_dim, output_dim).to(device))
    #
    #     # 3. 加载权重
    #     self.model.load_state_dict(checkpoint, strict=False)
    #     self.model.eval()
    #
    #     shared_list = []
    #     private_list = []
    #     subject_labels = []
    #
    #     # 4. 提取特征
    #     with torch.no_grad():
    #         for batch in self.test_data_loader:
    #             eeg, eog, emg, y, l, d = batch
    #             # 前向传播 (MDNet 内部会自动更新特征到 self.model.utt_shared_subject 等属性)
    #             self.model(to_gpu(eeg), to_gpu(eog), to_gpu(emg), to_gpu(l), to_gpu(d), self.groups)
    #
    #             shared_list.append(self.model.utt_shared_subject.cpu().numpy())
    #             private_list.append(self.model.utt_private_subject.cpu().numpy())
    #             subject_labels.append(d.cpu().numpy())
    #
    #     # 5. 调用绘图方法
    #     self._plot_tsne_combined(shared_list, private_list, subject_labels)
    #
    # def _plot_tsne_combined(self, shared_list, private_list, subject_labels):
    #
    #     all_subjects = np.concatenate(subject_labels, axis=0)
    #
    #     # 将不连续的 Subject ID (如 101, 105) 映射为连续颜色索引
    #     unique_subs = np.unique(all_subjects)
    #     sub_map = {val: i for i, val in enumerate(unique_subs)}
    #     mapped_subjects = np.array([sub_map[s] for s in all_subjects])
    #
    #     def draw_single(data_list, title, filename):
    #         data = np.concatenate(data_list, axis=0)
    #         # t-SNE 降维
    #         tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    #         data_2d = tsne.fit_transform(data)
    #         data_2d = MinMaxScaler().fit_transform(data_2d)
    #
    #         plt.figure(figsize=(12, 11))
    #
    #         # 增大散点大小 s=60, edgecolors='none' 让颜色更纯净
    #         scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=mapped_subjects,
    #                               cmap='tab20', s=60, alpha=0.8, edgecolors='none')
    #
    #         # 【重点】设置超大标题字体
    #         plt.title(title, fontsize=42, fontweight='bold', pad=30)
    #
    #         # 移除刻度，保留坐标轴边框
    #         plt.xticks([])
    #         plt.yticks([])
    #
    #         # 【重点】设置颜色条刻度字体
    #         cbar = plt.colorbar(scatter)
    #         cbar.ax.tick_params(labelsize=26)
    #
    #         plt.tight_layout()
    #         os.makedirs('tsne_results', exist_ok=True)
    #         plt.savefig(f'tsne_results/{filename}.png', dpi=300)
    #         print(f"Success: Image saved to tsne_results/{filename}.png")
    #
    #     # 分别绘制
    #     draw_single(shared_list, "Shared Features", "shared_features_tsne")
    #     draw_single(private_list, "Private Features", "private_features_tsne")
    def visualize_tsne(self, checkpoint_path, output_tsne_dir):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import MinMaxScaler

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 强行修复 leaner 层的形状 (保持之前的逻辑)
        for name, param in checkpoint.items():
            if 'leaner' in name and 'weight' in name:
                layer_name = name.replace('.weight', '')
                input_dim = param.shape[1]
                output_dim = param.shape[0]
                setattr(self.model, layer_name, nn.Linear(input_dim, output_dim).to(device))

        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

        shared_list, private_list = [], []
        subject_labels, task_labels = [], []  # 增加任务标签列表

        with torch.no_grad():
            for batch in self.test_data_loader:
                eeg, eog, emg, y, l, d = batch  # y 是真实标签 (Valence/Arousal)

                self.model(to_gpu(eeg), to_gpu(eog), to_gpu(emg), to_gpu(l), to_gpu(d), self.groups)

                shared_list.append(self.model.utt_shared_subject.cpu().numpy())
                private_list.append(self.model.utt_private_subject.cpu().numpy())
                subject_labels.append(d.cpu().numpy())  # 受试者 ID
                task_labels.append(y.cpu().numpy())  # 情绪标签 (0 或 1)

        # 调用改进后的绘图逻辑
        self._plot_tsne_enhanced(shared_list, private_list, subject_labels, task_labels)

    def _plot_tsne_enhanced(self, shared_list, private_list, sub_labels, task_labels):
        all_subs = np.concatenate(sub_labels, axis=0)
        all_tasks = np.concatenate(task_labels, axis=0)

        # 受试者 ID 映射
        u_subs = np.unique(all_subs)
        sub_map = {val: i for i, val in enumerate(u_subs)}
        mapped_subs = np.array([sub_map[s] for s in all_subs])

        def draw(data_list, color_data, title, filename, cmap_name):
            data = np.concatenate(data_list, axis=0)
            data_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(data)
            data_2d = MinMaxScaler().fit_transform(data_2d)

            plt.figure(figsize=(12, 11))
            # 设置点的大小和样式
            scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=color_data,
                                  cmap=cmap_name, s=70, alpha=0.7, edgecolors='none')

            plt.title(title, fontsize=42, fontweight='bold', pad=30)
            plt.xticks([]);
            plt.yticks([])

            cbar = plt.colorbar(scatter)
            # 如果是任务标签，设置离散刻度
            if cmap_name != 'tab20':
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(['Low', 'High'])
            cbar.ax.tick_params(labelsize=26)

            os.makedirs('tsne_results_new', exist_ok=True)
            plt.savefig(f'tsne_results_new/{filename}.png', dpi=300)
            plt.close()

        # 绘制 4 张图，进行全方位对比
        # 1. 共享特征 - 按受试者分 (预期：混杂)
        draw(shared_list, mapped_subs, "Shared by Subject", "shared_sub", 'tab20')
        # 2. 共享特征 - 按标签分 (预期：聚类)
        draw(shared_list, all_tasks, "Shared by Label", "shared_task", 'coolwarm')
        # 3. 私有特征 - 按受试者分 (预期：聚类)
        draw(private_list, mapped_subs, "Private by Subject", "private_sub", 'tab20')
        # 4. 私有特征 - 按标签分 (预期：混杂或无序)
        draw(private_list, all_tasks, "Private by Label", "private_task", 'coolwarm')

    def _save_config(self):
        """保存训练配置"""
        config_path = os.path.join(self.config.OUTPUT_ROOT,
                                   f'{self.train_config.name}_config.txt')
        with open(config_path, 'w') as f:
            f.write(f"Experiment: {self.train_config.name}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Task: {getattr(self.train_config, 'task', 'arousal')}\n")
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
                y = y.squeeze()

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
                y = y.squeeze()

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


if __name__ == "__main__":
    # ================= 1. 基础配置 =================
    DataConfig.BASE_DATA_PATH = '/eds-storage/scw/MER/data'

    # ⚠️ 修改这里：指定你要可视化的模型所在的根目录，以及表现最好的是第几折
    base_output_dir = '/code/scw/MER/output_10fold_cv_arousal'
    TARGET_FOLD = 10  # <--- 在这里填入你 Test Acc 最高的那一折的数字 (1-10)

    fold_idx = TARGET_FOLD - 1

    print("=" * 60)
    print(f"准备绘制 t-SNE: 正在复刻 FOLD {TARGET_FOLD} 的数据环境...")
    print("=" * 60)

    # ================= 2. 完美复刻数据划分与分组 =================
    folds = get_10_folds_subjects(DataConfig, seed=42)
    train_subs, val_subs, test_subs = get_split_for_fold(folds, fold_idx, seed=42)

    # 这里的参数必须和训练时保持一致！
    best_params = {"group_num": 2, "diff_weight": 0.05, "sim_weight": 0.001, "learning_rate": 1e-4}
    groups = create_train_groups_dynamic(train_subs, num_groups=best_params["group_num"], seed=42 + fold_idx)

    print(f"🎯 成功还原测试集被试: {test_subs}")

    # ================= 3. 构建 DataLoader =================
    test_dataset = MERMultimodalDataset(test_subs, config=DataConfig, task='arousal')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=DataConfig.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    empty_loader = torch.utils.data.DataLoader([], batch_size=1)

    # ================= 4. 初始化 Solver =================
    solver = Solver(
        train_config=DataConfig(),
        train_data_loader=empty_loader,
        dev_data_loader=empty_loader,
        test_data_loader=test_loader,
        groups=groups,
        params=best_params,
        seed=42,
        config=DataConfig
    )
    solver.build()

    # ================= 5. 定位模型并执行可视化 =================
    # 自动拼接该折的 checkpoint 路径
    checkpoint_path = os.path.join(base_output_dir, f'fold_{TARGET_FOLD}', 'checkpoints', 'model_cv_fold_10.pt')

    # 自动将图片保存在该折的 curves 文件夹下
    output_tsne_dir = os.path.join(base_output_dir, f'fold_{TARGET_FOLD}', 'curves')
    os.makedirs(output_tsne_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print(f"❌ 找不到模型权重文件: {checkpoint_path}")
    else:
        # 防雷机制：修正 visualizer 里可能存在的 squeeze 崩溃问题
        # （可选检查：如果在 visualize_tsne 里有 y.squeeze()，请手动改为 y.view(-1)）
        solver.visualize_tsne(checkpoint_path, output_tsne_dir)
        print(f"✅ t-SNE 绘制完成！图片已保存至: {output_tsne_dir}")