# ==================== 路径配置（必须放在最前面）====================
import sys
import os

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# ==================== 标准库导入 ====================
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器运行
import matplotlib.pyplot as plt
import pickle
import random
import re
import json
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import torch.nn.init as init
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import numpy as np
from scipy import signal

# ==================== 本地模块导入 ====================
try:
    from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
    import models
    import get_distillation_kernel, get_distillation_kernel_homo
    from utils.misc import softmax
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保utils、models等模块在当前目录或Python路径中")
    raise


# ==================== 被试间相关性分析与分组模块（基于load_data_test.py优化）====================

class SubjectCorrelationAnalyzer:
    """
    被试间相关性分析器
    适配MER数据集结构：被试文件夹下包含多个trial的pkl文件
    """

    def __init__(self, config, output_dir='./correlation_analysis'):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设置日志文件
        self.log_file = os.path.join(output_dir, f'correlation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    def log(self, message, print_console=True):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        if print_console:
            print(log_message)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def extract_psd_features(self, subject_data, sampling_rate=200, segment_length=4):
        """
        提取PSD特征（基于load_data_test.py的preprocess_subject_data优化）

        Args:
            subject_data: list of arrays, each (20, 800) - trials × channels × timepoints
                         或者单个array (n_trials, 20, 800)
            sampling_rate: 采样率，默认200Hz
            segment_length: 段长（秒），默认4秒（即800点）

        Returns:
            psd_features: (n_segments, n_features) PSD特征矩阵
        """
        # 统一处理为list格式
        if isinstance(subject_data, np.ndarray):
            if subject_data.ndim == 2:  # 单个trial (20, 800)
                subject_data = [subject_data]
            elif subject_data.ndim == 3:  # 多个trial (n_trials, 20, 800)
                subject_data = [subject_data[i] for i in range(subject_data.shape[0])]

        all_psd_features = []

        for trial_idx, trial_data in enumerate(subject_data):
            # trial_data shape: (20, 800)
            n_channels, n_timepoints = trial_data.shape

            # 只使用EEG通道（前18通道）
            eeg_data = trial_data[:self.config.EEG_CHANNELS, :]

            # 计算每个通道的PSD
            segment_psd = []
            for channel in range(eeg_data.shape[0]):
                # 使用Welch方法计算PSD
                freqs, psd = signal.welch(eeg_data[channel], fs=sampling_rate, nperseg=256)
                segment_psd.append(psd)

            # 拉平所有通道的PSD特征
            flattened_psd = np.concatenate(segment_psd)
            all_psd_features.append(flattened_psd)

        return np.array(all_psd_features)

    def load_subject_data(self, subject_id):
        """
        加载单个被试的所有trial数据

        Returns:
            list of arrays: 每个元素是一个trial的 (20, 800) 数据
        """
        subject_dir = os.path.join(self.config.BASE_DATA_PATH, str(subject_id))
        if not os.path.exists(subject_dir):
            self.log(f"警告: 被试 {subject_id} 的目录不存在: {subject_dir}", print_console=False)
            return None

        # 扫描所有pkl文件
        pkl_files = []
        for f in os.listdir(subject_dir):
            if f.endswith('.pkl'):
                # 解析文件名: {sub}-{stimulus}-{window}.pkl
                match = re.match(self.config.PKL_PATTERN, f)
                if match:
                    pkl_files.append(os.path.join(subject_dir, f))

        if not pkl_files:
            self.log(f"警告: 被试 {subject_id} 没有找到有效的pkl文件", print_console=False)
            return None

        # 加载所有trial
        all_trials = []
        for pkl_path in sorted(pkl_files):
            try:
                with open(pkl_path, 'rb') as f:
                    data_dict = pickle.load(f)

                sample = data_dict['sample']  # (20, 800)

                # 验证shape
                if sample.shape != (self.config.TOTAL_CHANNELS, self.config.TIME_POINTS):
                    if sample.shape == (self.config.TIME_POINTS, self.config.TOTAL_CHANNELS):
                        sample = sample.T
                    else:
                        self.log(f"警告: {pkl_path} 形状异常: {sample.shape}", print_console=False)
                        continue

                all_trials.append(sample)

            except Exception as e:
                self.log(f"错误加载 {pkl_path}: {e}", print_console=False)
                continue

        if not all_trials:
            return None

        self.log(f"被试 {subject_id}: 成功加载 {len(all_trials)} 个trials")
        return all_trials

    def calculate_subject_correlations(self, subject_ids):
        """
        计算所有被试间的相关性矩阵（基于load_data_test.py的calculate_subject_correlations_all_samples）

        Args:
            subject_ids: 被试ID列表

        Returns:
            correlation_matrix: (n_subjects, n_subjects) 相关性矩阵
            valid_subject_ids: 实际有效的被试ID列表
            subject_psd_features: 每个被试的PSD特征列表
        """
        self.log("="*60)
        self.log("开始计算被试间相关性")
        self.log("="*60)

        # 1. 加载所有被试数据并提取PSD特征
        subject_psd_features = []
        valid_subject_ids = []

        for sub_id in subject_ids:
            self.log(f"\n处理被试 {sub_id}...")
            trials_data = self.load_subject_data(sub_id)

            if trials_data is None:
                self.log(f"  跳过被试 {sub_id} (无有效数据)")
                continue

            # 提取PSD特征
            psd_features = self.extract_psd_features(trials_data,
                                                     sampling_rate=self.config.SAMPLE_RATE,
                                                     segment_length=self.config.WINDOW_SIZE)

            self.log(f"  提取PSD特征: {psd_features.shape}")
            subject_psd_features.append(psd_features)
            valid_subject_ids.append(sub_id)

        n_subjects = len(valid_subject_ids)
        self.log(f"\n有效被试数: {n_subjects}")

        if n_subjects < 2:
            self.log("错误: 有效被试数不足2，无法计算相关性")
            return None, valid_subject_ids, subject_psd_features

        # 2. 计算被试间相关性矩阵
        self.log("\n计算相关性矩阵...")
        correlation_matrix = np.zeros((n_subjects, n_subjects))

        for i in range(n_subjects):
            for j in range(i, n_subjects):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # 合并两个被试的所有样本
                    combined_features = np.vstack([subject_psd_features[i], subject_psd_features[j]])

                    # 计算相关矩阵
                    corr_matrix = np.corrcoef(combined_features)
                    n_i = subject_psd_features[i].shape[0]

                    # 提取组间相关部分（被试i的样本 vs 被试j的样本）
                    inter_correlation = corr_matrix[:n_i, n_i:].mean()

                    correlation_matrix[i, j] = inter_correlation
                    correlation_matrix[j, i] = inter_correlation

        self.log("相关性矩阵计算完成")
        return correlation_matrix, valid_subject_ids, subject_psd_features

    def analyze_correlation_matrix(self, correlation_matrix, subject_ids):
        """
        分析相关性矩阵，生成分组建议
        """
        self.log("\n" + "="*60)
        self.log("相关性矩阵分析")
        self.log("="*60)

        n = len(subject_ids)

        # 1. 统计信息
        # 提取上三角（排除对角线）
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        upper_tri_values = correlation_matrix[mask]

        self.log(f"\n相关性统计:")
        self.log(f"  均值: {np.mean(upper_tri_values):.4f}")
        self.log(f"  标准差: {np.std(upper_tri_values):.4f}")
        self.log(f"  最小值: {np.min(upper_tri_values):.4f}")
        self.log(f"  最大值: {np.max(upper_tri_values):.4f}")
        self.log(f"  中位数: {np.median(upper_tri_values):.4f}")

        # 2. 按阈值分布统计
        thresholds = [0.7, 0.5, 0.3, 0.0]
        ranges = ["≥0.7", "0.5~0.7", "0.3~0.5", "<0.3"]

        self.log(f"\n相关系数分布:")
        for i, (thresh, range_name) in enumerate(zip(thresholds, ranges)):
            if i == 0:
                count = np.sum(upper_tri_values >= thresh)
            elif i == len(thresholds) - 1:
                count = np.sum(upper_tri_values < thresholds[i-1])
            else:
                count = np.sum((upper_tri_values >= thresh) & (upper_tri_values < thresholds[i-1]))

            percentage = count / len(upper_tri_values) * 100
            self.log(f"  {range_name}: {count} ({percentage:.1f}%)")

        # 3. 每个被试的最大相关性分析
        self.log(f"\n各被试最大相关性分析:")
        max_corr_per_subject = []

        for i in range(n):
            non_diag = [j for j in range(n) if j != i]
            max_val = np.max(correlation_matrix[i, non_diag])
            max_idx = non_diag[np.argmax(correlation_matrix[i, non_diag])]
            max_corr_per_subject.append({
                'subject_id': subject_ids[i],
                'max_correlation': max_val,
                'most_similar_subject': subject_ids[max_idx]
            })
            self.log(f"  被试 {subject_ids[i]}: max={max_val:.4f} (与被试 {subject_ids[max_idx]})")

        # 4. 生成分组建议（基于stepwise_clean逻辑）
        range_groups = self._generate_groups(correlation_matrix, subject_ids)

        return {
            'statistics': {
                'mean': float(np.mean(upper_tri_values)),
                'std': float(np.std(upper_tri_values)),
                'min': float(np.min(upper_tri_values)),
                'max': float(np.max(upper_tri_values)),
                'median': float(np.median(upper_tri_values))
            },
            'distribution': {range_name: int(np.sum(upper_tri_values >= thresh if i == 0 else
                                                    (upper_tri_values >= thresh) & (upper_tri_values < thresholds[i-1]) if i < len(thresholds)-1 else
                                                    upper_tri_values < thresholds[i-1]))
                             for i, (thresh, range_name) in enumerate(zip(thresholds, ranges))},
            'max_correlations': max_corr_per_subject,
            'range_groups': range_groups
        }

    def _generate_groups(self, correlation_matrix, subject_ids):
        """
        基于相关性矩阵生成分组（使用stepwise_clean逻辑）
        """
        n = len(subject_ids)

        range_groups = {
            "≥0.7": [],
            "0.5~0.7": [],
            "0.3~0.5": [],
            "<0.3": []
        }

        for i in range(n):
            non_diag_indices = [j for j in range(n) if j != i]
            non_diag_values = correlation_matrix[i, non_diag_indices]
            max_val = np.max(non_diag_values)

            if max_val >= 0.7:
                indices = np.where(correlation_matrix[i, :] >= 0.7)[0]
                range_groups["≥0.7"].extend(indices.tolist())
            elif max_val >= 0.5:
                indices = np.where(correlation_matrix[i, :] >= 0.5)[0]
                range_groups["0.5~0.7"].extend(indices.tolist())
            elif max_val >= 0.3:
                indices = np.where(correlation_matrix[i, :] >= 0.3)[0]
                range_groups["0.3~0.5"].extend(indices.tolist())
            else:
                indices = np.where(correlation_matrix[i, :] < 0.3)[0]
                range_groups["<0.3"].extend(indices.tolist())
                range_groups["<0.3"].append(i)

        # 去重
        for key in range_groups:
            range_groups[key] = sorted(set(range_groups[key]))

        # 应用stepwise_clean
        cleaned_groups = self._stepwise_clean(range_groups)

        # 映射回被试ID
        result = {}
        for key, indices in cleaned_groups.items():
            if indices:
                result[key] = [subject_ids[i] for i in indices if i < len(subject_ids)]
            else:
                result[key] = []

        self.log(f"\n分组结果 (stepwise_clean后):")
        for key, subs in result.items():
            self.log(f"  {key}: {len(subs)}人被试 - {subs}")

        return result

    def _stepwise_clean(self, range_groups):
        """
        分步清理分组（直接从load_data_test.py复制）
        """
        # 第一步：从"0.5~0.7"中删除出现在"≥0.7"中的元素
        if "0.5~0.7" in range_groups and "≥0.7" in range_groups:
            high_set = set(range_groups["≥0.7"])
            mid_high_set = set(range_groups["0.5~0.7"])
            cleaned_mid_high = mid_high_set - high_set
            range_groups["0.5~0.7"] = sorted(cleaned_mid_high)

        # 第二步：从"0.3~0.5"中删除出现在"≥0.7"和"0.5~0.7"中的元素
        if "0.3~0.5" in range_groups:
            higher_elements = set()
            for key in ["≥0.7", "0.5~0.7"]:
                if key in range_groups:
                    higher_elements.update(range_groups[key])

            mid_low_set = set(range_groups["0.3~0.5"])
            cleaned_mid_low = mid_low_set - higher_elements
            range_groups["0.3~0.5"] = sorted(cleaned_mid_low)

        # 第三步：从"<0.3"中删除出现在所有其他组中的元素
        if "<0.3" in range_groups:
            higher_elements = set()
            for key in ["≥0.7", "0.5~0.7", "0.3~0.5"]:
                if key in range_groups:
                    higher_elements.update(range_groups[key])

            low_set = set(range_groups["<0.3"])
            cleaned_low = low_set - higher_elements
            range_groups["<0.3"] = sorted(cleaned_low)

        return range_groups

    def visualize_and_save(self, correlation_matrix, subject_ids, analysis_results):
        """
        可视化相关性矩阵并保存所有结果
        """
        self.log("\n" + "=" * 60)
        self.log("保存结果和可视化")
        self.log("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存相关性矩阵为numpy文件
        matrix_path = os.path.join(self.output_dir, f'correlation_matrix_{timestamp}.npy')
        np.save(matrix_path, correlation_matrix)
        self.log(f"相关性矩阵已保存: {matrix_path}")

        # 2. 保存为文本格式（便于查看）
        txt_path = os.path.join(self.output_dir, f'correlation_matrix_{timestamp}.txt')
        with open(txt_path, 'w') as f:
            f.write("Subject IDs: " + str(subject_ids) + "\n\n")
            f.write("Correlation Matrix:\n")
            f.write(np.array2string(correlation_matrix, precision=4, separator=', '))
        self.log(f"文本格式已保存: {txt_path}")

        # 3. 保存完整分析结果为JSON
        json_path = os.path.join(self.output_dir, f'analysis_results_{timestamp}.json')

        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_to_native(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        # 转换所有数据为JSON安全格式
        json_safe_results = convert_to_native({
            'subject_ids': subject_ids,
            'correlation_matrix': correlation_matrix,
            'statistics': analysis_results['statistics'],
            'distribution': analysis_results['distribution'],
            'max_correlations': analysis_results['max_correlations'],
            'range_groups': analysis_results['range_groups']
        })

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
        self.log(f"JSON分析结果已保存: {json_path}")

        # 4. 绘制热力图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 左图：完整热力图
        im1 = axes[0].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[0].set_title('Subject-to-Subject Correlation Matrix\n(MER Dataset)', fontsize=14)
        axes[0].set_xlabel('Subject Index')
        axes[0].set_ylabel('Subject Index')

        # 设置刻度标签为被试ID
        axes[0].set_xticks(range(len(subject_ids)))
        axes[0].set_yticks(range(len(subject_ids)))
        axes[0].set_xticklabels(subject_ids, rotation=90, fontsize=8)
        axes[0].set_yticklabels(subject_ids, fontsize=8)

        # 添加颜色条
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Correlation Coefficient', rotation=270, labelpad=15)

        # 右图：带数值标注的热力图（显示部分）
        display_size = min(20, len(subject_ids))  # 最多显示20x20
        display_matrix = correlation_matrix[:display_size, :display_size]

        im2 = axes[1].imshow(display_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[1].set_title(f'Detailed View (First {display_size} Subjects)', fontsize=14)
        axes[1].set_xlabel('Subject Index')
        axes[1].set_ylabel('Subject Index')

        # 添加数值标注
        for i in range(display_size):
            for j in range(display_size):
                text = axes[1].text(j, i, f'{display_matrix[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontsize=6)

        axes[1].set_xticks(range(display_size))
        axes[1].set_yticks(range(display_size))
        axes[1].set_xticklabels(subject_ids[:display_size], rotation=90, fontsize=8)
        axes[1].set_yticklabels(subject_ids[:display_size], fontsize=8)

        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('Correlation Coefficient', rotation=270, labelpad=15)

        plt.tight_layout()

        # 保存图片
        heatmap_path = os.path.join(self.output_dir, f'correlation_heatmap_{timestamp}.png')
        fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        self.log(f"热力图已保存: {heatmap_path}")
        plt.close(fig)

        # 5. 绘制分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        upper_tri_values = correlation_matrix[mask]

        ax.hist(upper_tri_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(upper_tri_values), color='red', linestyle='--',
                   label=f'Mean: {np.mean(upper_tri_values):.3f}')
        ax.axvline(np.median(upper_tri_values), color='green', linestyle='--',
                   label=f'Median: {np.median(upper_tri_values):.3f}')

        # 标记分组阈值
        for thresh in [0.3, 0.5, 0.7]:
            ax.axvline(thresh, color='orange', linestyle=':', alpha=0.5)
            ax.text(thresh, ax.get_ylim()[1] * 0.9, f'{thresh}', rotation=90, va='top')

        ax.set_xlabel('Correlation Coefficient', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Inter-Subject Correlations', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        hist_path = os.path.join(self.output_dir, f'correlation_distribution_{timestamp}.png')
        fig.savefig(hist_path, dpi=300, bbox_inches='tight')
        self.log(f"分布图已保存: {hist_path}")
        plt.close(fig)

        return {
            'matrix_npy': matrix_path,
            'matrix_txt': txt_path,
            'json': json_path,
            'heatmap': heatmap_path,
            'histogram': hist_path
        }

    def run_full_analysis(self, subject_ids, save_groups_for_solver=True):
        """
        运行完整的分析流程
        """
        self.log("="*60)
        self.log("被试间相关性分析 - 完整流程")
        self.log("="*60)
        self.log(f"数据路径: {self.config.BASE_DATA_PATH}")
        self.log(f"被试范围: {list(subject_ids)}")

        # 1. 计算相关性矩阵
        corr_matrix, valid_ids, psd_features = self.calculate_subject_correlations(subject_ids)

        if corr_matrix is None:
            self.log("分析失败: 无法计算相关性矩阵")
            return None

        # 2. 分析矩阵
        analysis = self.analyze_correlation_matrix(corr_matrix, valid_ids)

        # 3. 可视化和保存
        paths = self.visualize_and_save(corr_matrix, valid_ids, analysis)

        # 4. 为solver保存分组配置（可选）
        if save_groups_for_solver:
            groups_config = self._save_groups_for_solver(analysis['range_groups'], valid_ids)
            paths['groups_config'] = groups_config

        self.log("\n" + "="*60)
        self.log("分析完成!")
        self.log(f"所有结果保存在: {self.output_dir}")
        self.log(f"日志文件: {self.log_file}")
        self.log("="*60)

        return {
            'correlation_matrix': corr_matrix,
            'valid_subject_ids': valid_ids,
            'psd_features': psd_features,
            'analysis': analysis,
            'saved_paths': paths
        }

    def _save_groups_for_solver(self, range_groups, subject_ids):
        """
        保存分组配置供solver使用
        """
        # 转换为solver可用的格式（被试ID列表的列表）
        groups = []
        for key in ["≥0.7", "0.5~0.7", "0.3~0.5", "<0.3"]:
            if key in range_groups and range_groups[key]:
                groups.append(range_groups[key])

        # 保存为pickle（solver可以直接加载）
        import pickle
        groups_path = os.path.join(self.output_dir, 'subject_groups.pkl')
        with open(groups_path, 'wb') as f:
            pickle.dump({
                'groups': groups,
                'range_groups': range_groups,
                'all_subjects': subject_ids
            }, f)

        self.log(f"分组配置已保存: {groups_path}")
        return groups_path


# ==================== 数据路径配置类 ====================
class DataConfig:
    """
    数据路径配置类 - 根据你的预处理代码结构配置
    """
    # 修改为你的实际数据根目录（包含所有被试文件夹的目录）
    BASE_DATA_PATH = '/code/clisa/clisa/Downstream_dataset/AdaBrain-Bench-LaBraM-fusion/preprocessing/MER/data'

    # 结果保存根目录
    OUTPUT_ROOT = './output_mer_experiment'

    # 通道配置（根据你的预处理代码：18 EEG + 1 GSR + 1 PPG = 20通道）
    EEG_CHANNELS = 18
    GSR_CHANNELS = 1
    PPG_CHANNELS = 1
    TOTAL_CHANNELS = 20  # EEG_CHANNELS + GSR_CHANNELS + PPG_CHANNELS

    # 数据shape配置
    SAMPLE_RATE = 200
    WINDOW_SIZE = 4  # seconds
    TIME_POINTS = 800  # SAMPLE_RATE * WINDOW_SIZE

    # 被试ID范围（明确命名为 SUBJECT_RANGE，与代码中使用的一致）
    SUBJECT_RANGE = range(1, 81)  # 默认1-81，根据你的实际被试数修改

    # 文件模式配置
    PKL_PATTERN = r'(\d+)-(\d+)-(\d+)\.pkl'  # 匹配 {sub}-{stimulus}-{window}.pkl


# ==================== 自动创建输出目录 ====================
def setup_directories(config_class=DataConfig):
    """创建所有必要的输出目录"""
    dirs = [
        os.path.join(config_class.OUTPUT_ROOT, 'checkpoints'),
        os.path.join(config_class.OUTPUT_ROOT, 'results'),
        os.path.join(config_class.OUTPUT_ROOT, 'logs'),
        os.path.join(config_class.OUTPUT_ROOT, 'curves'),
        os.path.join(config_class.OUTPUT_ROOT, 'correlation_analysis')  # 新增相关性分析目录
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


# ==================== 优化的数据划分函数（使用相关性分析）====================
def get_73_subject_split(config=DataConfig, train_ratio=0.7, val_ratio=0.1,
                         seed=42, use_correlation=True, correlation_results=None):
    """
    扫描被试文件夹，基于相关性划分为训练/验证/测试集

    Args:
        use_correlation: 是否使用预计算的相关性结果
        correlation_results: 预计算的相关性分析结果（如果use_correlation=True）
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

    # 2. 使用相关性结果进行分层划分
    if use_correlation and correlation_results is not None:
        print("使用预计算的相关性结果进行分层划分...")

        # 确保只使用有效被试
        valid_ids = correlation_results['valid_subject_ids']
        range_groups = correlation_results['analysis']['range_groups']

        # 过滤只包含可用被试的分组
        filtered_groups = {}
        for key, subs in range_groups.items():
            filtered = [s for s in subs if s in available_subs]
            if filtered:
                filtered_groups[key] = filtered

        print(f"分层分组: { {k: len(v) for k, v in filtered_groups.items()} }")
    else:
        # 随机分组
        print("使用随机划分...")
        filtered_groups = {"all": available_subs}

    # 3. 按比例划分
    train_subs, val_subs, test_subs = [], [], []

    # 按相关性从高到低优先分配（确保高相关性被试分布在各集合中）
    group_priority = ["≥0.7", "0.5~0.7", "0.3~0.5", "<0.3", "all"]

    for group_key in group_priority:
        if group_key not in filtered_groups:
            continue

        real_ids = filtered_groups[group_key].copy()
        np.random.shuffle(real_ids)

        # 确保每个组都按比例分配
        n = len(real_ids)
        tr_cut = int(n * train_ratio)
        val_cut = int(n * (train_ratio + val_ratio))

        train_subs.extend(real_ids[:tr_cut])
        val_subs.extend(real_ids[tr_cut:val_cut])
        test_subs.extend(real_ids[val_cut:])

    # 去重（防止一个被试出现在多个集合中）
    train_subs = sorted(set(train_subs))
    val_subs = sorted(set(val_subs))
    test_subs = sorted(set(test_subs))

    # 确保没有重叠
    all_assigned = set(train_subs) | set(val_subs) | set(test_subs)
    assert len(all_assigned) == len(train_subs) + len(val_subs) + len(test_subs), "集合有重叠！"

    print(f"\n划分完成: 训练集 {len(train_subs)}人, 验证集 {len(val_subs)}人, 测试集 {len(test_subs)}人")
    print(f"  训练集: {train_subs}")
    print(f"  验证集: {val_subs}")
    print(f"  测试集: {test_subs}")

    return train_subs, val_subs, test_subs


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 设置配置
    DataConfig.BASE_DATA_PATH = '/code/clisa/clisa/Downstream_dataset/AdaBrain-Bench-LaBraM-fusion/preprocessing/MER/data'
    DataConfig.OUTPUT_ROOT = './output_mer_experiment'

    # 创建输出目录
    setup_directories(DataConfig)

    # 步骤1：运行被试间相关性分析
    print("="*60)
    print("步骤1: 被试间相关性分析")
    print("="*60)

    analyzer = SubjectCorrelationAnalyzer(
        config=DataConfig,
        output_dir=os.path.join(DataConfig.OUTPUT_ROOT, 'correlation_analysis')
    )

    # 运行完整分析
    correlation_results = analyzer.run_full_analysis(
        subject_ids=DataConfig.SUBJECT_RANGE,
        save_groups_for_solver=True
    )

    if correlation_results is None:
        print("错误: 相关性分析失败，请检查数据路径")
        sys.exit(1)

    # 步骤2：基于相关性结果划分数据集
    print("\n" + "="*60)
    print("步骤2: 数据集划分")
    print("="*60)

    train_subs, val_subs, test_subs = get_73_subject_split(
        DataConfig,
        seed=42,
        use_correlation=True,
        correlation_results=correlation_results
    )

    print("\n" + "="*60)
    print("准备就绪！可以开始训练")
    print("="*60)
    print(f"运行TensorBoard查看结果:")
    print(f"  tensorboard --logdir={DataConfig.OUTPUT_ROOT}/logs")