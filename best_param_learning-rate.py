import matplotlib.pyplot as plt
import numpy as np

# 手动指定中文字体（seaborn 的 annot 和 tick labels 需要单独处理）
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# ==================== 生成数据 ====================
iterations = np.arange(0, 51)  # 0 到 50

# 学习率变化（分段常数）
learning_rates = [
    0.00059] * 3 + [0.0009] * 18 + [0.001] * 30  # 3+18+30=51

learning_rates = np.array(learning_rates)

# ==================== 绘图 ====================
fig, ax = plt.subplots(figsize=(6, 4))

# 绘制阶梯线（where='mid' 表示中间跳变）
ax.step(iterations, learning_rates, where='mid',
        linewidth=2, color='#A020F0', label='Learning Rate')  # 紫色

# 添加图例
ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)

# 设置坐标轴标签
ax.set_xlabel('迭代轮数')
ax.set_ylabel('学习率')

# 设置刻度
ax.set_xticks(range(0, 51, 10))
ax.set_yticks([0.0008, 0.0009, 0.0010])
ax.set_xlim(0, 50)
ax.set_ylim(0.0005, 0.0011)

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

# 紧凑布局
plt.tight_layout()
plt.savefig('plots/best_param_learning_rate.png', dpi=300)

# 显示图形
plt.show()