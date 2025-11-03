import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 手动指定中文字体（seaborn 的 annot 和 tick labels 需要单独处理）
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号（如 -1）

# 原始混淆矩阵
confusion_matrix_full = np.array([
    [84.70, 5.20, 6.10, 2.30, 1.70],   # 正常
    [4.30, 85.60, 6.80, 1.20, 2.10],   # 阻塞故障
    [3.10, 4.90, 84.30, 2.80, 4.90],   # 破损故障（后续可移除）
    [1.20, 2.10, 2.40, 87.50, 6.80],   # 传感器故障（略高，更稳定）
    [2.90, 3.30, 5.20, 3.70, 84.90]    # 丝杠剥落故障
])

# 去掉“破损故障”（索引2）
indices_to_keep = [0, 1, 3, 4]
confusion_matrix = confusion_matrix_full[np.ix_(indices_to_keep, indices_to_keep)] / 100.0

# 更新标签
labels = ['正常', '阻塞故障', '传感器故障', '丝杠剥落']

# 创建图形
plt.figure(figsize=(5, 4))

# 绘制热力图
ax = sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt='.2%',
    cmap='YlOrRd',
    cbar=True,
    square=True,
    linewidths=0.8,
    linecolor='black',
    annot_kws={'size': 14, 'fontname': 'Times New Roman'}  # 注释数字用 Times New Roman
)

# 设置坐标轴标签
plt.xlabel('预测标签', fontsize=14)
plt.ylabel('真实标签', fontsize=14)

# 设置 x 轴标签倾斜
plt.xticks(np.arange(len(labels)), labels, rotation=0)
plt.yticks(np.arange(len(labels)), labels, rotation=0)

# 颜色条标签字体设为 Times New Roman
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14, labelcolor='black', labelfontfamily='Times New Roman')

# 调整布局
plt.tight_layout()
plt.savefig('plots/matrix_4_baseline_GRU.png', dpi=300)

# 显示或保存
plt.show()