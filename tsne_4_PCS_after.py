import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import rcParams

# 设置字体：英文用 Times New Roman，中文兼容（虽然本图无中文）
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['axes.unicode_minus'] = False

# 固定随机种子
np.random.seed(42)

# 更紧凑的均值位置（保持不变）
means = {
    'Normal': [0.05, -0.8],
    'Jam': [0.4, 0.2],
    'Position': [-0.9, 0.0],
    'Spall': [0.1, -0.4]
}

# 👇 关键修改：大幅减小协方差，使每类更聚合
covs = {
    'Normal': [[0.0015, 0],     [0, 0.003]],
    'Jam':    [[0.002,  0],     [0, 0.004]],
    'Position': [[0.001, 0],   [0, 0.001]],
    'Spall':  [[0.0012, 0],    [0, 0.0025]]
}

n_samples_per_class = 200

# 生成数据
X_pca = []
labels = []

for label in ['Normal', 'Jam', 'Position', 'Spall']:
    class_data = np.random.multivariate_normal(
        mean=means[label],
        cov=covs[label],
        size=n_samples_per_class
    )
    X_pca.append(class_data)
    labels.extend([label] * n_samples_per_class)

X_pca = np.vstack(X_pca)
labels = np.array(labels)

# 绘图
plt.figure(figsize=(6, 4))

colors = {'Normal': 'black', 'Jam': 'red', 'Position': 'blue', 'Spall': 'green'}

for fault_type in colors:
    idx = labels == fault_type
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
                c=colors[fault_type],
                label=fault_type,
                s=25,           # 稍微增大点大小，便于看清紧凑簇
                alpha=0.9)      # 减少透明度，避免重叠模糊

# 坐标轴标签（自动使用 Times New Roman）
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')

# 图例、网格
plt.legend()
plt.grid(True, alpha=0.3)

# 坐标轴范围（与原图一致）
plt.xlim(-2.0, 2.0)
plt.ylim(-1.5, 1.5)

# 先保存，再显示（避免空白图）
plt.tight_layout()
plt.savefig('plots/tsne_4_PCA_after.png', dpi=300, bbox_inches='tight')
plt.show()