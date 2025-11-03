import matplotlib.pyplot as plt
import numpy as np

# 手动指定中文字体（seaborn 的 annot 和 tick labels 需要单独处理）
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# ==================== 生成数据 ====================
iterations = np.arange(0, 51)  # 0 到 50

# BiGRU Layer 2 的神经元个数变化（分段常数）
neuron_counts = [
    16.5] * 3 + [22.5] * 18 + [16.5] * 30  # 3+18+30=51

neuron_counts = np.array(neuron_counts)

# ==================== 绘图 ====================
fig, ax = plt.subplots(figsize=(6, 4))

# 绘制阶梯线（where='mid' 表示中间跳变）
ax.step(iterations, neuron_counts, where='mid',
        linewidth=2, color='#2E8B57', label='BiGRULayer_2')  # 深绿色

# 添加图例
ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)

# 设置坐标轴标签
ax.set_xlabel('迭代轮数')
ax.set_ylabel('BiGRU层隐藏神经元个数')

# 设置刻度
ax.set_xticks(range(0, 51, 10))
ax.set_yticks([16.5, 17, 18, 19, 20, 21, 22, 22.5])
ax.set_xlim(0, 50)
ax.set_ylim(16.2, 23)

# 添加网格（浅灰色）
ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)

# 设置边框
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 对于坐标轴上的数字，我们需要单独设置字体
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')  # 设置为 Times New Roman

# 设置坐标轴范围
ax.set_xlim(-1, 50)
ax.set_ylim(16, 24)

# 紧凑布局
plt.tight_layout()

plt.savefig('plots/best_param_BiGRU_neuron_counts2.png', dpi=300)

# 显示图形
plt.show()