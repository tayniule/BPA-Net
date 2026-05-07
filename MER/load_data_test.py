import os
import _pickle as cPickle
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None
    print("Warning: seaborn not found, visualization functions disabled")

def load_data_per_subject(sub):
    """
    This function loads the target subject's original file
    Parameters
    ----------
    sub: which subject to load

    Returns
    -------
    data: (40, 32, 7680) label: (40, 4)
    """
    data_path = '/data/Running/Emotion/DEAP/data_preprocessed_python/data_preprocessed_python'
    sub += 1
    if (sub < 10):
        sub_code = str('s0' + str(sub) + '.dat')
    else:
        sub_code = str('s' + str(sub) + '.dat')

    subject_path = os.path.join(data_path, sub_code)
    subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
    label = subject['labels']
    data = subject['data'][:, 0:32, 3 * 128:]  # Excluding the first 3s of baseline  ###40multi-modal
    #   data: 40 x 32 x 7680
    #   label: 40 x 4
    sub_label = np.full((40, 1), sub)
    print('data:' + str(data.shape) + ' label:' + str(label.shape))
    return data, label, sub_label


def preprocess_subject_data(subject_data, segment_length=4, sampling_rate=128):
    """
    处理单个被试的数据
    subject_data: (40, 32, 7680) - trials × channels × timepoints
    返回: PSD特征矩阵 (n_segments, n_features)
    """
    n_trials, n_channels, n_timepoints = subject_data.shape
    segment_samples = segment_length * sampling_rate  # 4 * 128 = 512个采样点

    all_psd_features = []

    for trial in range(n_trials):
        for start_idx in range(0, n_timepoints - segment_samples + 1, segment_samples):
            segment = subject_data[trial, :, start_idx:start_idx + segment_samples]

            # 计算每个通道的PSD
            segment_psd = []
            for channel in range(n_channels):
                freqs, psd = signal.welch(segment[channel], fs=sampling_rate, nperseg=256)
                segment_psd.append(psd)

            # 拉平所有通道的PSD特征
            flattened_psd = np.concatenate(segment_psd)
            all_psd_features.append(flattened_psd)

    return np.array(all_psd_features)


def process_all_subjects(all_subject_data):
    """
    处理所有32个被试的数据
    all_subject_data: list of 32 arrays, each (40, 32, 7680)
    返回: 所有被试的PSD特征列表
    """
    subject_psd_features = []

    for i, subject_data in enumerate(all_subject_data):
        print(f"处理被试 {i + 1}/32...")
        psd_features = preprocess_subject_data(subject_data)
        subject_psd_features.append(psd_features)

        print(f"被试 {i + 1} 生成 {psd_features.shape[0]} 个样本, 特征维度: {psd_features.shape[1]}")

    return subject_psd_features


def calculate_subject_correlations_all_samples(subject_psd_features):
    """
    使用所有样本计算被试间相关性
    """
    n_subjects = len(subject_psd_features)
    correlation_matrix = np.zeros((n_subjects, n_subjects))

    for i in range(n_subjects):
        for j in range(i, n_subjects):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                # 合并两个被试的所有样本
                combined_features = np.vstack([subject_psd_features[i], subject_psd_features[j]])

                # 计算相关矩阵并提取组间相关性
                corr_matrix = np.corrcoef(combined_features)
                n_i = subject_psd_features[i].shape[0]
                n_j = subject_psd_features[j].shape[0]

                # 提取组间相关部分
                inter_correlation = corr_matrix[:n_i, n_i:].mean()
                correlation_matrix[i, j] = inter_correlation
                correlation_matrix[j, i] = inter_correlation

    return correlation_matrix


# 主处理函数
def main_analysis(test_sub):
    sub_to_run = np.arange(32)
    all_subject_data = []
    for sub in sub_to_run:
        if sub == test_sub:
            continue
        data_, label_, sub_label_ = load_data_per_subject(sub)
        all_subject_data.append(data_)
    # test_sub_data, _, _ = load_data_per_subject(test_sub)

    # 1. 预处理所有被试数据
    print("步骤1: 预处理被试数据...")
    subject_psd_features = process_all_subjects(all_subject_data)

    # 2. 计算被试间相关性
    print("步骤2: 计算被试间相关性...")
    corr_matrix_all = calculate_subject_correlations_all_samples(subject_psd_features)

    return {
        'subject_psd_features': subject_psd_features,
        'correlation_matrix_all': corr_matrix_all
    }

def get_groups(test_sub):
    groups = []
    x = main_analysis(test_sub)
    co = x['correlation_matrix_all']
    features = x['subject_psd_features']

    range_groups = {
        "≥0.7": [],
        "0.5~0.7": [],
        "0.3~0.5": [],
        "<0.3": []
    }

    for i in range(31):
        non_diag_indices = [j for j in range(31) if j != i]

        non_diag_values = co[i, non_diag_indices]

        max_val = np.max(non_diag_values)

        if max_val >= 0.7:
            indices = np.where(co[i, :] >= 0.7)[0]
            range_groups["≥0.7"].extend(indices.tolist())  # 改为 extend
        elif max_val >= 0.5:
            indices = np.where(co[i, :] >= 0.5)[0]
            range_groups["0.5~0.7"].extend(indices.tolist())
        elif max_val >= 0.3:
            indices = np.where(co[i, :] >= 0.3)[0]
            range_groups["0.3~0.5"].extend(indices.tolist())
        else:  # max_val < 0.3
            indices = np.where(co[i, :] < 0.3)[0]
            range_groups["<0.3"].extend(indices.tolist())
            range_groups["<0.3"].append(i)

        for key in range_groups:
            range_groups[key] = sorted(set(range_groups[key]))

    result = stepwise_clean(range_groups)
    for key, values in result.items():
        if values != None:
            groups.append(values)
    for i in range(len(groups)):
        for j in range(len(groups[i])):
            if groups[i][j] >= test_sub:
                groups[i][j] += 1

    return groups, features



def stepwise_clean(range_groups):
    """
    分步清理：先清理"0.5~0.7"，再清理"0.3~0.5"，最后清理"<0.3"
    """
    # 第一步：从"0.5~0.7"中删除出现在"≥0.7"中的元素
    if "0.5~0.7" in range_groups and "≥0.7" in range_groups:
        high_set = set(range_groups["≥0.7"])
        mid_high_set = set(range_groups["0.5~0.7"])
        cleaned_mid_high = mid_high_set - high_set
        range_groups["0.5~0.7"] = sorted(cleaned_mid_high)

    # 第二步：从"0.3~0.5"中删除出现在"≥0.7"和"0.5~0.7"中的元素
    if "0.3~0.5" in range_groups:
        # 收集所有更高分组的元素
        higher_elements = set()
        for key in ["≥0.7", "0.5~0.7"]:
            if key in range_groups:
                higher_elements.update(range_groups[key])

        mid_low_set = set(range_groups["0.3~0.5"])
        cleaned_mid_low = mid_low_set - higher_elements
        range_groups["0.3~0.5"] = sorted(cleaned_mid_low)

    # 第三步：从"<0.3"中删除出现在所有其他组中的元素
    if "<0.3" in range_groups:
        # 收集所有更高分组的元素
        higher_elements = set()
        for key in ["≥0.7", "0.5~0.7", "0.3~0.5"]:
            if key in range_groups:
                higher_elements.update(range_groups[key])

        low_set = set(range_groups["<0.3"])
        cleaned_low = low_set - higher_elements
        range_groups["<0.3"] = sorted(cleaned_low)

    return range_groups


def get_test_group(test_sub, features):
    groups = []
    all_subject_data = []
    data_, label_, sub_label_ = load_data_per_subject(test_sub)
    # all_subject_data.append(data_)
    psd_features = preprocess_subject_data(data_)
    features.append(psd_features)
    co = calculate_subject_correlations_all_samples(features)
    range_groups = {
        "≥0.7": [],
        "0.5~0.7": [],
        "0.3~0.5": [],
        "<0.3": []
    }

    for i in range(31):
        non_diag_indices = [j for j in range(31) if j != i]

        non_diag_values = co[i, non_diag_indices]

        max_val = np.max(non_diag_values)

        if max_val >= 0.7:
            indices = np.where(co[i, :] >= 0.7)[0]
            range_groups["≥0.7"].extend(indices.tolist())  # 改为 extend
        elif max_val >= 0.5:
            indices = np.where(co[i, :] >= 0.5)[0]
            range_groups["0.5~0.7"].extend(indices.tolist())
        elif max_val >= 0.3:
            indices = np.where(co[i, :] >= 0.3)[0]
            range_groups["0.3~0.5"].extend(indices.tolist())
        else:  # max_val < 0.3
            indices = np.where(co[i, :] < 0.3)[0]
            range_groups["<0.3"].extend(indices.tolist())
            range_groups["<0.3"].append(i)

        for key in range_groups:
            range_groups[key] = sorted(set(range_groups[key]))

    result = stepwise_clean(range_groups)
    for key, values in result.items():
        if values != None:
            groups.append(values)
    return groups


# group = get_groups()
# print(group)

# groups = []
# result = stepwise_clean(range_groups)
# print("层级清理后的分组:")
# for key, values in result.items():
#     if values != None:
#         groups.append(values)
#     print(f"{key} ({len(values)}个元素): {values}")

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
# plt.figure(figsize=(10, 8))
# sns.heatmap(co,
#             cmap='RdBu_r',  # 红蓝配色，中间是白色
#             center=0,
#             square=True,
#             cbar_kws={'label': '数值大小'})
# plt.title('32×32矩阵热力图', fontsize=14)
# plt.xlabel('列索引')
# plt.ylabel('行索引')
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(12, 10))
# sns.heatmap(co,
#             annot=True,  # 显示数值
#             fmt='.2f',   # 数值格式
#             annot_kws={'size': 6},
#             cmap='coolwarm',
#             square=True,
#             cbar_kws={'shrink': 0.8})
# plt.title('带数值标注的32×32矩阵', fontsize=14)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.tight_layout()
# plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
# plt.show()
# row_max_values = np.zeros(32)
# row_max_indices = np.zeros(32, dtype=int)
# for i in range(32):
#     # 获取第 i 行的所有非对角线元素的索引
#     non_diag_indices = [j for j in range(32) if j != i]
#
#     # 提取非对角线元素
#     non_diag_values = co[i, non_diag_indices]
#
#     # 找到最大值
#     max_val = np.max(non_diag_values)
#     row_max_values[i] = max_val
#
#     # 找到最大值在原始列中的索引
#     # 注意：non_diag_indices 中最大值对应的索引
#     max_val_idx_in_non_diag = np.argmax(non_diag_values)
#     original_col = non_diag_indices[max_val_idx_in_non_diag]
#     row_max_indices[i] = original_col
#
# # 输出结果
# print("行号 | 最大值(排除对角线) | 最大值所在的列")
# print("-" * 50)
# for i in range(32):
#     print(f"{i:3d} | {row_max_values[i]:.6f}             | {row_max_indices[i]}")
#
# # 也可以直接输出数组
# print("\n所有行的最大值（排除对角线）：")
# print(row_max_values)

# 存储结果


#     # 找出满足条件的元素
#     filtered_items = []
#     for idx, val in enumerate(non_diag_values):
#         col_idx = non_diag_indices[idx]
#         if condition(val):
#             filtered_items.append((col_idx, val))
#
#     # 按数值从大到小排序
#     filtered_items.sort(key=lambda x: x[1], reverse=True)
#
#     # 存储结果
#     results.append({
#         'row': i,
#         'max_value': max_val,
#         'range': range_desc,
#         'filtered_items': filtered_items,
#         'count': len(filtered_items)
#     })
#
# # 输出结果
# print("行号 | 最大值(排除对角线) | 范围     | 符合条件的个数 | 符合条件的列和数值")
# print("-" * 90)
# for res in results:  # 只显示前15行
#     items_str = ', '.join(f"{col}:{val:.4f}" for col, val in res['filtered_items'])  # 最多显示5个
#     if res['count'] > 5:
#         items_str += f" ... (+{res['count'] - 5}个)"
#     print(
#         f"{res['row']:3d} | {res['max_value']:.6f}              | {res['range']:7s} | {res['count']:5d}         | {items_str}")
#
# # 统计各范围的行数
# range_stats = {}
# for res in results:
#     r = res['range']
#     range_stats[r] = range_stats.get(r, 0) + 1
#
# print("\n=== 统计汇总 ===")
# print("范围分布:")
# for r, count in sorted(range_stats.items()):
#     print(f"  {r}: {count}行")
#
# # 如果需要详细查看某一行
# print("\n=== 示例：查看第0行的详细结果 ===")
# res0 = results[0]
# print(f"行 {res0['row']}: 最大值 = {res0['max_value']:.6f}, 范围 = {res0['range']}")
# print("符合条件的元素 (列:数值):")
# for col, val in res0['filtered_items']:
#     print(f"  列 {col}: {val:.6f}")

# 基于所有样本的相关矩阵
# sns.heatmap(x['correlation_matrix_all'],
#             annot=True, fmt='.2f', cmap='coolwarm',
#             center=0, ax=ax2, square=True)
# ax2.set_title('基于所有样本的被试间相关性')
# ax2.set_xlabel('被试编号')
# ax2.set_ylabel('被试编号')
#
# plt.tight_layout()
# plt.show()

# 定义分组边界
# bins = [0, 0.1, 0.2, 0.3, 1.0]
# group_labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3以上']

# 方法1：统计所有相关系数的分布
# flat_correlations = co[np.triu_indices(31, k=1)]  # 取上三角（排除对角线）
# group_counts, _ = np.histogram(flat_correlations, bins=bins)

# print("=== 所有相关系数的分组统计 ===")
# for label, count in zip(group_labels, group_counts):
#     print(f"{label}: {count}个相关系数 ({count/len(flat_correlations)*100:.1f}%)")

# 方法2：按被试平均相关系数分组
# subject_means = np.mean(co, axis=1)  # 每个被试的平均相关系数
# subject_groups = np.digitize(subject_means, bins[:-1]) - 1  # 分组索引
# groups = []
# print("\n=== 被试分组结果 ===")
# for i, label in enumerate(group_labels):
#     subjects_in_group = np.where(subject_groups == i)[0] + 1
#     groups.append(subjects_in_group)