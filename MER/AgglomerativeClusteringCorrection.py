# ==================== 路径配置（必须放在最前面）====================
import sys
import os

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# ==================== 标准库导入 ====================
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import random
import re
import json
from collections import defaultdict
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
from scipy import signal


# ==================== 被试间相关性分析与分组模块 ====================

class SubjectCorrelationAnalyzer:
    """
    被试间相关性分析器 - 基于AgglomerativeClustering聚类分组
    适配MER数据集：18 EEG + 1 GSR + 1 PPG = 20通道，200Hz，4s窗口
    """

    def __init__(self, config, output_dir='./correlation_analysis'):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, f'correlation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    def log(self, message, print_console=True):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        if print_console:
            print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def extract_psd_features(self, subject_data, sampling_rate=200):
        """提取EEG的PSD特征"""
        # 统一处理为list格式
        if isinstance(subject_data, np.ndarray):
            if subject_data.ndim == 2:
                subject_data = [subject_data]
            elif subject_data.ndim == 3:
                subject_data = [subject_data[i] for i in range(subject_data.shape[0])]

        all_psd_features = []
        for trial_data in subject_data:
            # 只使用EEG通道（前18通道）
            eeg_data = trial_data[:self.config.EEG_CHANNELS, :]
            segment_psd = []
            for channel in range(eeg_data.shape[0]):
                freqs, psd = signal.welch(eeg_data[channel], fs=sampling_rate, nperseg=256)
                segment_psd.append(psd)
            flattened_psd = np.concatenate(segment_psd)
            all_psd_features.append(flattened_psd)

        return np.array(all_psd_features)

    def load_subject_data(self, subject_id):
        """加载单个被试的所有trial数据"""
        subject_dir = os.path.join(self.config.BASE_DATA_PATH, str(subject_id))
        if not os.path.exists(subject_dir):
            return None

        pkl_files = []
        for f in os.listdir(subject_dir):
            if f.endswith('.pkl'):
                match = re.match(self.config.PKL_PATTERN, f)
                if match:
                    pkl_files.append(os.path.join(subject_dir, f))

        if not pkl_files:
            return None

        all_trials = []
        for pkl_path in sorted(pkl_files):
            try:
                with open(pkl_path, 'rb') as f:
                    data_dict = pickle.load(f)
                sample = data_dict['sample']
                # 验证shape并处理转置
                if sample.shape != (self.config.TOTAL_CHANNELS, self.config.TIME_POINTS):
                    if sample.shape == (self.config.TIME_POINTS, self.config.TOTAL_CHANNELS):
                        sample = sample.T
                    else:
                        continue
                all_trials.append(sample)
            except Exception:
                continue

        return all_trials if all_trials else None

    def calculate_subject_correlations(self, subject_ids):
        """计算被试间相关性矩阵"""
        self.log("=" * 60)
        self.log("开始计算被试间相关性")
        self.log("=" * 60)

        subject_psd_features = []
        valid_subject_ids = []

        for sub_id in subject_ids:
            trials_data = self.load_subject_data(sub_id)
            if trials_data is None:
                continue

            psd_features = self.extract_psd_features(trials_data, sampling_rate=self.config.SAMPLE_RATE)
            subject_psd_features.append(psd_features)
            valid_subject_ids.append(sub_id)
            self.log(f"被试 {sub_id}: 提取PSD特征 {psd_features.shape}")

        n_subjects = len(valid_subject_ids)
        self.log(f"\n有效被试数: {n_subjects}")

        if n_subjects < 2:
            self.log("错误: 有效被试数不足2")
            return None, valid_subject_ids, subject_psd_features

        # 计算相关性矩阵
        correlation_matrix = np.zeros((n_subjects, n_subjects))
        for i in range(n_subjects):
            for j in range(i, n_subjects):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    combined_features = np.vstack([subject_psd_features[i], subject_psd_features[j]])
                    corr_matrix = np.corrcoef(combined_features)
                    n_i = subject_psd_features[i].shape[0]
                    inter_correlation = corr_matrix[:n_i, n_i:].mean()
                    correlation_matrix[i, j] = inter_correlation
                    correlation_matrix[j, i] = inter_correlation

        self.log("相关性矩阵计算完成")
        return correlation_matrix, valid_subject_ids, subject_psd_features

    def perform_clustering(self, correlation_matrix, subject_ids, n_groups):
        """
        使用AgglomerativeClustering进行分组（核心方法）

        Args:
            correlation_matrix: (n_subjects, n_subjects) 相关性矩阵
            subject_ids: 被试ID列表
            n_groups: 目标组数

        Returns:
            groups: list of lists, 每个子列表包含该组的被试ID
            cluster_labels: 每个被试的聚类标签
            silhouette: 轮廓系数
        """
        self.log(f"\n执行层次聚类: n_groups={n_groups}")

        n_subjects = len(subject_ids)
        if n_groups > n_subjects:
            self.log(f"警告: 调整 n_groups 从 {n_groups} 到 {n_subjects}")
            n_groups = n_subjects

        # 相关性转距离矩阵
        distance_matrix = 1 - correlation_matrix
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        # 层次聚类
        clusterer = AgglomerativeClustering(
            n_clusters=n_groups,
            metric='precomputed',
            linkage='average'
        )
        cluster_labels = clusterer.fit_predict(distance_matrix)

        # 计算轮廓系数
        try:
            silhouette = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
            self.log(f"轮廓系数: {silhouette:.4f}")
        except Exception:
            silhouette = None

        # 整理分组结果
        groups = [[] for _ in range(n_groups)]
        for idx, label in enumerate(cluster_labels):
            groups[label].append(subject_ids[idx])

        # 记录结果
        for i, group in enumerate(groups):
            self.log(f"  Group {i + 1} (n={len(group)}): {group}")

        return groups, cluster_labels, silhouette

    def run_full_analysis(self, subject_ids, n_groups, save_results=True):
        """
        运行完整分析流程 - solver直接调用接口

        Args:
            subject_ids: 被试ID列表
            n_groups: 目标组数（必需参数）
            save_results: 是否保存结果文件

        Returns:
            dict: {
                'groups': [[sub1, sub2], [sub3, sub4], ...],  # 直接可用的分组
                'correlation_matrix': np.array,
                'valid_subject_ids': list,
                'silhouette_score': float,
                'cluster_labels': np.array
            }
        """
        self.log("=" * 60)
        self.log(f"相关性分析 - n_groups={n_groups}")
        self.log(f"被试数: {len(subject_ids)}")
        self.log("=" * 60)

        # 1. 计算相关性矩阵
        corr_matrix, valid_ids, _ = self.calculate_subject_correlations(subject_ids)
        if corr_matrix is None:
            return None

        # 2. 执行聚类
        groups, cluster_labels, silhouette = self.perform_clustering(corr_matrix, valid_ids, n_groups)

        # 3. 保存结果（可选）
        if save_results:
            self._save_results(corr_matrix, valid_ids, groups, cluster_labels, silhouette, n_groups)

        return {
            'groups': groups,  # 关键：直接可用的分组列表
            'correlation_matrix': corr_matrix,
            'valid_subject_ids': valid_ids,
            'silhouette_score': silhouette,
            'cluster_labels': cluster_labels
        }

    def _save_results(self, corr_matrix, subject_ids, groups, cluster_labels, silhouette, n_groups):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存相关性矩阵
        np.save(os.path.join(self.output_dir, f'corr_matrix_k{n_groups}_{timestamp}.npy'), corr_matrix)

        # 保存分组结果JSON - 转换numpy类型为Python原生类型
        result = {
            'n_groups': int(n_groups),
            'silhouette_score': float(silhouette) if silhouette else None,
            'groups': [[int(s) for s in group] for group in groups],
            'subject_ids': [int(s) for s in subject_ids],
            'cluster_labels': [int(x) for x in cluster_labels.tolist()]
        }
        with open(os.path.join(self.output_dir, f'groups_k{n_groups}_{timestamp}.json'), 'w') as f:
            json.dump(result, f, indent=2)

        # 绘制热力图
        self._plot_heatmap(corr_matrix, subject_ids, groups, n_groups, silhouette, timestamp)

    def _plot_heatmap(self, corr_matrix, subject_ids, groups, n_groups, silhouette, timestamp):
        """绘制相关性热力图"""
        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        title = f'Subject Correlation Matrix (k={n_groups})'
        if silhouette:
            title += f'\nSilhouette={silhouette:.3f}'
        ax.set_title(title, fontsize=14)

        ax.set_xticks(range(len(subject_ids)))
        ax.set_yticks(range(len(subject_ids)))
        ax.set_xticklabels(subject_ids, rotation=90, fontsize=8)
        ax.set_yticklabels(subject_ids, fontsize=8)

        plt.colorbar(im, ax=ax, label='Correlation')
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f'heatmap_k{n_groups}_{timestamp}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.log(f"热力图保存: {save_path}")


# ==================== 便捷函数：solver直接调用 ====================

def get_subject_groups(subject_ids, config, n_groups, output_dir=None):
    """
    solver直接调用的便捷函数

    Args:
        subject_ids: 被试ID列表
        config: DataConfig配置对象
        n_groups: 目标组数
        output_dir: 可选，结果保存目录

    Returns:
        groups: list of lists, 分组结果，可直接传入Solver
    """
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_ROOT, f'correlation_k{n_groups}')

    analyzer = SubjectCorrelationAnalyzer(config, output_dir)
    result = analyzer.run_full_analysis(subject_ids, n_groups, save_results=True)

    if result is None:
        raise ValueError("相关性分析失败")

    return result['groups']


# ==================== 数据配置类 ====================

class DataConfig:
    """数据路径配置类"""
    BASE_DATA_PATH = '/eds-storage/scw/MER/data'
    OUTPUT_ROOT = './output_mer_experiment'

    EEG_CHANNELS = 18
    GSR_CHANNELS = 1
    PPG_CHANNELS = 1
    TOTAL_CHANNELS = 20

    SAMPLE_RATE = 200
    WINDOW_SIZE = 4
    TIME_POINTS = 800

    SUBJECT_RANGE = range(1, 81)
    PKL_PATTERN = r'(\d+)-(\d+)-(\d+)\.pkl'


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例：solver中的使用方式
    DataConfig.BASE_DATA_PATH = '/eds-storage/scw/MER/data'

    # 方式1：直接使用便捷函数（推荐）
    groups = get_subject_groups(
        subject_ids=list(range(1, 81)),
        config=DataConfig,
        n_groups=3
    )
    print(f"分组结果: {groups}")

    # 方式2：使用类接口（需要更多控制时）
    analyzer = SubjectCorrelationAnalyzer(
        config=DataConfig,
        output_dir='./analysis_k4'
    )
    result = analyzer.run_full_analysis(
        subject_ids=list(range(1, 81)),
        n_groups=4
    )
    print(f"轮廓系数: {result['silhouette_score']:.4f}")
    print(f"分组: {result['groups']}")