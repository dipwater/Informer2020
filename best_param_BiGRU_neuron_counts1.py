import matplotlib.pyplot as plt
import numpy as np


# 手动指定中文字体（seaborn 的 annot 和 tick labels 需要单独处理）
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# ==================== 生成数据 ====================
# 迭代轮数（0 到 50）
iterations = np.arange(0, 51)

# BiGRU 层神经元个数变化（分段常数）
neuron_counts = [
    26] * 3 + [40] * 18 + [10] * 30  # 3+18+30=51

# 转换为数组
neuron_counts = np.array(neuron_counts)

# ==================== 绘图 ====================
fig, ax = plt.subplots(figsize=(6, 4))

# 绘制阶梯线（使用 step='pre' 或 'mid'）
ax.step(iterations, neuron_counts, where='mid',
        linewidth=2, color='#9ACD32', label='BiGRULayer_1')

# 添加图例
ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)

# 设置坐标轴标签
ax.set_xlabel('迭代轮数')
ax.set_ylabel('BiGRU层隐藏神经元个数')

# 设置刻度
ax.set_xticks(range(0, 51, 10))
ax.set_yticks(range(10, 42, 2))  # 从10到40，步长2

# 添加网格（细密网格）
ax.grid(True, alpha=0.7, linestyle='-', linewidth=0.5)

# 设置边框
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 对于坐标轴上的数字，我们需要单独设置字体
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')  # 设置为 Times New Roman

# 设置坐标轴范围
ax.set_xlim(-1, 50)
ax.set_ylim(5, 45)

# 紧凑布局
plt.tight_layout()
plt.savefig('plots/best_param_BiGRU_neuron_counts1.png', dpi=300)

# 显示图形
plt.show()