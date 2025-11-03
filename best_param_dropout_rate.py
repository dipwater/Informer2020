import matplotlib.pyplot as plt
import numpy as np

# 手动指定中文字体（seaborn 的 annot 和 tick labels 需要单独处理）
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# ==================== 生成数据 ====================
iterations = np.arange(0, 51)  # 0 到 50

# Dropout 比率变化（分段常数）
dropout_rates = [
    0.22] * 3 + [0.01] * 18 + [0.07] * 14 + [0.13] * 16  # 3+18+14+16=51

dropout_rates = np.array(dropout_rates)

# ==================== 绘图 ====================
fig, ax = plt.subplots(figsize=(6, 4))

# 绘制阶梯线（where='mid' 表示中间跳变）
ax.step(iterations, dropout_rates, where='mid',
        linewidth=2, color='#A020F0', label='Dropout')  # 紫色

# 添加图例
ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)

# 设置坐标轴标签
ax.set_xlabel('迭代轮数')
ax.set_ylabel('Dropout')

# 设置刻度
ax.set_xticks(range(0, 51, 10))
ax.set_yticks([0.01, 0.05, 0.10, 0.15, 0.20, 0.22])
ax.set_xlim(0, 50)
ax.set_ylim(0.005, 0.23)

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
ax.set_ylim(0, 0.25)

# 紧凑布局
plt.tight_layout()
plt.savefig('plots/best_param_dropout_rate.png', dpi=300)

# 显示图形
plt.show()