# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
#
# # ====================== 绘图参数（可按需调整） ======================
# fs = 200          # 采样率
# draw_seconds = 5  # 绘制时长
# num_points = int(fs * draw_seconds)
# time = np.linspace(0, draw_seconds, num_points)
# num_channels = 18  # 通道数，和你参考图一致
#
# # 科研绘图风格设置（匹配参考图的柔和质感）
# rcParams.update({
#     'figure.dpi': 300,
#     'savefig.dpi': 300,
#     'font.size': 16,
#     'font.family': 'Arial',
#     'font.weight': 'bold',
#     'axes.spines.top': False,
#     'axes.spines.right': False,
#     'axes.spines.left': False,
#     'axes.spines.bottom': False,
#     'xtick.bottom': False,
#     'ytick.left': False,
# })
#
# # ====================== 生成【真实脑电特征】的模拟信号 ======================
# # 定义柔和马卡龙配色（完美匹配参考图的彩色风格）
# colors = [
#     '#4A90E2', '#C27BA0', '#82CA9D', '#9B7BBD', '#F7D794',
#     '#50A7C6', '#B86B7D', '#76D7EA', '#A3A3A3', '#F2A65A',
#     '#6C5CE7', '#00B894', '#E17055', '#74B9FF', '#FD79A8',
#     '#FDCB6E', '#00CEC9', '#D6A2E8'
# ]
#
# # 生成18通道真实脑电信号（δ+θ+α+β+噪声，符合生理特征）
# eeg = np.zeros((num_points, num_channels))
# for ch in range(num_channels):
#     # 脑电节律叠加，通道间相位偏移，避免完全同步
#     delta = 2.5 * np.sin(2 * np.pi * 1.2 * time + ch * 0.4)    # δ慢波（基础）
#     theta = 1.2 * np.sin(2 * np.pi * 6 * time + ch * 0.6)      # θ波
#     alpha = 0.8 * np.sin(2 * np.pi * 10 * time + ch * 0.8)     # α波
#     beta = 0.5 * np.sin(2 * np.pi * 20 * time + ch * 1.0)      # β波
#     noise = np.random.randn(num_points) * 0.25                 # 真实EEG噪声
#     eeg[:, ch] = delta + theta + alpha + beta + noise
#
# # ====================== 绘图（完美复刻参考图风格） ======================
# fig, ax = plt.subplots(figsize=(10, 7))
#
# # 通道偏移量（控制波形间距，参考图的舒展效果）
# offset = 6
# for ch in range(num_channels):
#     ax.plot(time, eeg[:, ch] + ch * offset,
#             color=colors[ch], linewidth=1.1, alpha=0.9)
#
# # 标题（和参考图完全一致的样式）
# ax.set_title('EEG Signals', fontweight='bold', fontsize=22, pad=20)
#
# # 隐藏所有坐标轴，只保留波形（参考图的极简风格）
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xlim(0, draw_seconds)
# ax.set_ylim(-2, num_channels * offset + 2)
#
# # 保存高清图（PNG预览 + PDF矢量图，可直接用于论文）
# plt.tight_layout()
# plt.savefig('/code/scw/MER/eeg_signals_color_stacked.pdf', bbox_inches='tight')
# plt.savefig('/code/scw/MER/eeg_signals_color_stacked.png', bbox_inches='tight')
#
# print("✅ 完美复刻的彩色EEG堆叠图已生成！")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib import rcParams

# ====================== 【顶刊绘图风格设置】 ======================
rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'Arial',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.bottom': False,
    'ytick.left': False,
})

# ====================== 【1. 生成真实感 FFT 频谱图】 ======================
# 模拟真实EEG信号的FFT频谱（低频高、高频衰减，符合生理信号特征）
freq = np.linspace(0, 50, 100)
# 构造真实PSD曲线：α波(8-13Hz)主峰+β波次峰+高频衰减
psd = np.zeros_like(freq)
# α波主峰（10Hz）
alpha_peak = 5 * np.exp(-((freq - 10)/3)**2)
# β波次峰（20Hz）
beta_peak = 2 * np.exp(-((freq - 20)/5)**2)
# 高频衰减
high_decay = 3 * np.exp(-freq/15)
# 加微小噪声，更真实
noise = np.random.randn(len(freq)) * 0.15
psd = alpha_peak + beta_peak + high_decay + noise

# ====================== 【2. 生成真实感 Conv2d 卷积核】 ======================
# 3个真实卷积核（不同深浅蓝色，符合参考图的堆叠效果）
conv_kernels = [
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * 0.3,  # 浅蓝核
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) * 0.6,  # 中蓝核
    np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) * 0.9  # 深蓝核（拉普拉斯边缘检测）
]
kernel_colors = ['#E3F2FD', '#90CAF9', '#1976D2']  # 浅→深蓝，匹配参考图

# ====================== 【3. 绘制完整模块图】 ======================
fig = plt.figure(figsize=(10, 4))

# --- 模块标题 ---
plt.figtext(0.5, 0.95, 'Coarse-grained Alignment Block',
            fontsize=20, fontweight='bold', ha='center', va='top')

# --------------------- 左图：FFT 频谱（真实感增强） ---------------------
ax1 = fig.add_axes([0.1, 0.1, 0.35, 0.7])
# 绘制真实频谱曲线
ax1.plot(freq, psd, color='#1A237E', linewidth=2.5, zorder=3)
# 绘制坐标轴箭头（参考图风格）
ax1.arrow(0, 0, 52, 0, head_width=0.3, head_length=1.5, fc='#1A237E', ec='#1A237E', zorder=2)
ax1.arrow(0, 0, 0, 7, head_width=1.5, head_length=0.3, fc='#1A237E', ec='#1A237E', zorder=2)
# 美化坐标轴
ax1.set_xlim(-2, 52)
ax1.set_ylim(-0.5, 7)
ax1.set_xticks([])
ax1.set_yticks([])
# 加圆角外框（参考图样式）
fft_box = FancyBboxPatch((-2, -0.5), 54, 7.5, boxstyle="round,pad=0.3",
                         ec='#1A237E', fc='#E3F2FD', linewidth=2, zorder=1)
ax1.add_patch(fft_box)
# FFT 标注
ax1.text(25, -1.8, 'FFT', fontsize=22, fontweight='bold', ha='center', color='#000000')

# --------------------- 中间箭头（数据流） ---------------------
arrow = plt.Arrow(0.48, 0.45, 0.03, 0, width=0.08, color='#000000', transform=fig.transFigure)
fig.patches.append(arrow)

# --------------------- 右图：Conv2d 卷积核（真实感增强） ---------------------
ax2 = fig.add_axes([0.55, 0.1, 0.35, 0.7])
# 绘制3个堆叠的卷积核（透视效果，匹配参考图的3D感）
for i, (kernel, color) in enumerate(zip(conv_kernels, kernel_colors)):
    # 透视偏移：越往后的核越靠右靠上，营造3D堆叠感
    dx = i * 0.8
    dy = i * 0.3
    # 绘制卷积核矩形
    kernel_rect = Rectangle((dx, dy), 3, 3, fc=color, ec='#0D47A1', linewidth=1.5, zorder=3-i)
    ax2.add_patch(kernel_rect)
    # 绘制卷积核内部网格（真实感）
    for x in range(4):
        ax2.plot([dx+x, dx+x], [dy, dy+3], color='#0D47A1', linewidth=0.8, alpha=0.7, zorder=4-i)
    for y in range(4):
        ax2.plot([dx, dx+3], [dy+y, dy+y], color='#0D47A1', linewidth=0.8, alpha=0.7, zorder=4-i)

# 绘制输出箭头（参考图样式）
ax2.arrow(3.5, 1.5, 2, 0, head_width=0.3, head_length=0.5, fc='#0D47A1', ec='#0D47A1', zorder=10)
# 加蓝色高亮外框（参考图的蓝色框）
conv_box = FancyBboxPatch((-0.5, -0.5), 6.5, 4.5, boxstyle="round,pad=0.3",
                          ec='#2196F3', fc='#E3F2FD', linewidth=3, zorder=1)
ax2.add_patch(conv_box)
# Conv2d 标注
ax2.text(2.5, -1.8, 'Conv2d', fontsize=22, fontweight='bold', ha='center', color='#000000')

# 统一坐标轴范围
ax2.set_xlim(-1, 7)
ax2.set_ylim(-2, 4.5)
ax2.set_xticks([])
ax2.set_yticks([])

# ====================== 【4. 保存高清矢量图】 ======================
plt.savefig('coarse_grained_alignment_block_realistic.pdf', bbox_inches='tight')
plt.savefig('coarse_grained_alignment_block_realistic.png', bbox_inches='tight')

print("✅ 真实感拉满的 FFT + Conv2d 模块图绘制完成！")
plt.show()