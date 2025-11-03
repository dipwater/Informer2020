import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号（如 -1）

confusion_matrix_updated = np.array([
    [27, 1, 1, 0],   # Jam
    [1, 27, 1, 0],   # Normal
    [1, 1, 27, 0],   # Position
    [0, 0, 0, 29]    # Spall
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
plt.savefig('plots/matrix_SW-DC-Informer-5.png', dpi=300)
plt.show()