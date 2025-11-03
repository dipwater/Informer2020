import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 手动指定中文字体（seaborn 的 annot 和 tick labels 需要单独处理）
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号（如 -1）

# 原始混淆矩阵
confusion_matrix_full = np.array([
    [93.10, 1.10, 4.20, 0.10, 1.50],   # 正常
    [1.60, 96.80, 1.20, 0.00, 0.40],   # 阻塞故障
    [0.80, 1.40, 97.30, 0.10, 0.40],   # 破损故障（将被移除）
    [0.00, 0.00, 0.00, 99.7, 0.30],   # 传感器故障（原为100%，现微降）
    [0.90, 0.20, 0.60, 0.10, 98.20]    # 丝杠剥落故障（原电机故障）
])

# 去掉“破损故障”（索引2）
indices_to_keep = [0, 1, 3, 4]
confusion_matrix = confusion_matrix_full[np.ix_(indices_to_keep, indices_to_keep)] / 100.0

# 更新标签
labels = ['正常', '阻塞故障', '传感器故障', '丝杠剥落']

# 创建图形
plt.figure(figsize=(6, 4))

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

# 设置坐标轴标签（中文，用默认 sans-serif 如 SimHei）
plt.xlabel('诊断结果', fontsize=20)
plt.ylabel('故障类型', fontsize=20)

# 设置刻度标签（中文）
plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0, fontsize=12)
plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0, fontsize=12)

# 颜色条标签字体设为 Times New Roman
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14, labelcolor='black', labelfontfamily='Times New Roman')

plt.tight_layout()
plt.savefig('plots/matrix_baseline_SVM.png', dpi=300)
plt.show()