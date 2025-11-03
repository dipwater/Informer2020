import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 手动指定中文字体（seaborn 的 annot 和 tick labels 需要单独处理）
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号（如 -1）

# 原始混淆矩阵
confusion_matrix_full = np.array([
    [96.80, 0.90, 1.30, 0.50, 0.50],   # 正常
    [0.80, 97.40, 1.10, 0.40, 0.30],   # 阻塞故障
    [0.70, 1.20, 96.90, 0.60, 0.60],   # 破损故障（可后续移除）
    [0.20, 0.30, 0.40, 97.80, 1.30],   # 传感器故障（最稳定）
    [0.60, 0.70, 1.40, 0.90, 96.40]    # 丝杠剥落故障
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
plt.savefig('plots/matrix_4_Transformer-BiGRU-Attention.png', dpi=300)

# 显示或保存
plt.show()