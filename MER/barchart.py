import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 0. 字体大小配置（核心） ----------------------
# 论文正文字体大小（根据你的论文实际字号调整，如12pt）
paper_font_size = 15
# 图表字体：比论文正文小1号
plot_font_size = paper_font_size

# 可选：配置中文字体（如SimSun/宋体，解决中文显示乱码）
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体
# plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 全局字体大小基准
plt.rcParams['font.size'] = plot_font_size

# ---------------------- 1. 数据准备 ----------------------
models = ['DCAG-Net(w/o DG)', 'DCAG-Net(w/o GE)', 'DCAG-Net']
# Arousal 指标
arousal_acc = [0.738, 0.745, 0.779]
arousal_f1 = [0.627, 0.637, 0.684]
# Valence 指标
valence_acc = [0.561, 0.565, 0.626]
valence_f1 = [0.406, 0.488, 0.598]

# 渐变配色（同一色系从浅到深，突出DCAG-Net）
colors = ['#CCF0E1', '#81C784', '#4CAF50']  # 浅绿 -> 中绿 -> 深绿


# ---------------------- 2. 绘图函数（调整字体大小） ----------------------
def plot_ablation_bar(arousal_data, valence_data, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 6))

    # 横坐标位置：Arousal(0), Valence(1)
    x = np.arange(2)
    width = 0.25  # 单根柱子宽度
    gap = 0.02    # 组内间隙

    # 计算Y轴范围（最小值向下微调5%，最大值向上微调5%，突出差异）
    all_data = arousal_data + valence_data
    y_min = min(all_data) * 0.95  # 最小值下探5%
    y_max = max(all_data) * 1.05  # 最大值上提5%

    # 绘制3组柱子（每组对应一个模型）
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [arousal_data[i], valence_data[i]]
        # 让3组柱子以刻度为中心对称分布
        offset = (i - 1) * (width + gap)
        bars = ax.bar(x + offset, values, width, label=model, color=color,
                      edgecolor='white', linewidth=1.2)  # 加粗边框更清晰

        # 柱子顶部数值标签：比基准小0.5，保证紧凑
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=plot_font_size - 0.5, fontweight='bold')

    # 坐标轴与标签设置
    ax.set_ylabel(ylabel, fontsize=plot_font_size, fontweight='bold')
    #ax.set_title(title, fontsize=plot_font_size + 1, fontweight='bold', pad=15)  # 标题略大
    ax.set_xticks(x)
    ax.set_xticklabels(['Arousal', 'Valence'], fontsize=plot_font_size, fontweight='bold')

    # Y轴范围（动态调整）+ 网格
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    legend = ax.legend(title='MER', loc='upper left', bbox_to_anchor=(1.01, 1),
                       frameon=False, fontsize=plot_font_size - 0.5,
                       title_fontsize=plot_font_size)
    # 2. 获取图例标题对象，设置加粗（兼容所有matplotlib版本）
    legend.get_title().set_fontweight('bold')

    # 边框美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------- 3. 绘制两张图 ----------------------
# 绘制 ACC 图
plot_ablation_bar(arousal_acc, valence_acc, 'Accuracy', 'ablation_module_acc.png')

# 绘制 F1 图
plot_ablation_bar(arousal_f1, valence_f1, 'F1 Score', 'ablation_module_f1.png')