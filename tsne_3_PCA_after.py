import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from matplotlib import rcParams

# 设置中文字体和英文字体
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子以便复现结果
np.random.seed(42)

# 定义每类的数据分布参数（均值和协方差矩阵）
# 根据图中位置估计
means = {
    'Normal': [0.05, -0.8],
    'Jam': [0.4, 0.2],
    'Position': [-0.9, 0.0],
    'Spall': [0.1, -0.4]
}

covs = {
    'Normal': [[0.01, 0], [0, 0.02]],
    'Jam': [[0.02, 0], [0, 0.03]],
    'Position': [[0.01, 0], [0, 0.01]],
    'Spall': [[0.01, 0], [0, 0.02]]
}

# 每类样本数量
n_samples_per_class = 200

# 生成数据
X_pca = []
labels = []

for label in ['Normal', 'Jam', 'Position', 'Spall']:
    # 从多变量正态分布采样
    class_data = np.random.multivariate_normal(
        mean=means[label],
        cov=covs[label],
        size=n_samples_per_class
    )
    X_pca.append(class_data)
    labels.extend([label] * n_samples_per_class)

# 合并所有数据
X_pca = np.vstack(X_pca)
labels = np.array(labels)

# 绘图
plt.figure(figsize=(6, 4))

colors = {'Normal': 'black', 'Jam': 'red', 'Position': 'blue', 'Spall': 'green'}
markers = {'Normal': 'o', 'Jam': 'o', 'Position': 'o', 'Spall': 'o'}

for fault_type in colors.keys():
    idx = labels == fault_type
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
                c=colors[fault_type], marker=markers[fault_type],
                label=fault_type, s=20, alpha=0.8)

plt.rcParams['font.family'] = 'Times New Roman'

# 设置标签和标题
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')

plt.legend()
# 网格
plt.grid(True, alpha=0.3)

# 设置坐标轴范围（匹配原图）
plt.xlim(-2.0, 2)
plt.ylim(-1.5, 1.5)

# 布局优化
plt.tight_layout()
plt.savefig('plots/tsne_3_PCA_after.png', dpi=300, bbox_inches='tight')
# 显示图形
plt.show()