import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 手动指定中文字体（seaborn 的 annot 和 tick labels 需要单独处理）
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号（如 -1）

confusion_matrix_updated = np.array([
    [27, 1, 1, 0],
    [1, 27, 1, 0],
    [1, 2, 25, 1],
    [0, 0, 0, 29]
])

labels_updated = ['Jam', 'Normal', 'Position', 'Spall']

plt.figure(figsize=(5, 4))

ax = sns.heatmap(
    confusion_matrix_updated,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True,
    cbar_kws={'label': '数量'},
    square=True,
    linewidths=0.8,
    linecolor='black',
    vmin=0,
    vmax=29,
    annot_kws={'size': 14, 'color': 'Orange', 'fontweight': 'bold', 'fontname': 'Times New Roman'}
)

plt.xlabel('预测标签', fontsize=14)
plt.ylabel('真实标签', fontsize=14)

plt.xticks(np.arange(len(labels_updated)), labels_updated, rotation=45)
plt.yticks(np.arange(len(labels_updated)), labels_updated, rotation=0)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14, labelcolor='black', labelfontfamily='Times New Roman')

plt.tight_layout()
plt.savefig('plots/matrix_4_SSA-Transformer-BiGRU.png', dpi=300)

# 显示或保存
plt.show()