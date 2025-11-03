import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持（可选，如需中文显示）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 模拟 SSA 适应度值序列 ====================
# 迭代轮数
iterations = np.arange(0, 51)

# 适应度值（阶梯下降，模拟优化过程）
# 注意：这里的数据长度已经调整至与 iterations 长度一致
fitness_values = [
    0.068, 0.068, 0.068, 0.068, 0.068,
    0.055, 0.055, 0.055, 0.055, 0.055,
    0.055, 0.055, 0.055, 0.055, 0.055,
    0.055, 0.055, 0.055, 0.055, 0.055,
    0.055, 0.055, 0.055, 0.055, 0.055,
    0.040, 0.040, 0.040, 0.040, 0.040,
    0.040, 0.040, 0.040, 0.040, 0.040,
    0.040, 0.040, 0.040, 0.040, 0.040,
    0.032, 0.032, 0.032, 0.032, 0.032,
    0.032, 0.032, 0.032, 0.032, 0.032,
    0.032  # 最后两个数值补充以匹配 iterations 长度
]

# 转换为数组
fitness_values = np.array(fitness_values)

# ==================== 绘图 ====================
fig, ax = plt.subplots(figsize=(6, 4))

# 绘制阶梯线（使用 step='mid' 或 'pre' 模拟）
ax.step(iterations, fitness_values, where='mid', linewidth=1.5, color='black')

# 添加图例
ax.legend(['适应度值'])

# 设置坐标轴标签
ax.set_xlabel('迭代轮数')
ax.set_ylabel('适应度值')

# 设置刻度
ax.set_xticks(range(0, 51, 10))
ax.set_yticks([0.03, 0.04, 0.05, 0.06, 0.07])

# 对于坐标轴上的数字，我们需要单独设置字体
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')  # 设置为 Times New Roman

# 添加网格
ax.grid(True, alpha=0.3)

# 设置边框
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 设置坐标轴范围
ax.set_xlim(-1, 50)
ax.set_ylim(0.03, 0.07)

# 紧凑布局
plt.tight_layout()

plt.savefig('plots/best_param_SSA_fitness.png', dpi=300)

# 显示图形
plt.show()