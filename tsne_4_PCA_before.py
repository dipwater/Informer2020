import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from matplotlib import rcParams

# 设置中文字体和英文字体
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
np.random.seed(42)

# ==================== 生成四类故障数据 ====================
n_samples = 500  # 总样本数

# 正常状态（Normal）
normal_mean = [8, -4]  # 调整均值
normal_cov = [[10, 3], [3, 5]]  # 增大协方差
normal_data = multivariate_normal.rvs(mean=normal_mean, cov=normal_cov, size=n_samples)

# 卡滞（Jam）——靠近 Normal
jam_mean = [2, -6]  # 调整均值，使其与正常状态有一定距离
jam_cov = [[10, 4], [4, 7]]  # 增大协方差
jam_data = multivariate_normal.rvs(mean=jam_mean, cov=jam_cov, size=n_samples)

# 剥落（Spall）——也靠近中心区域，但稍微远离一些
spall_mean = [6, 2]  # 更改均值
spall_cov = [[8, 2], [2, 5]]
spall_data = multivariate_normal.rvs(mean=spall_mean, cov=spall_cov, size=n_samples // 2)

# 传感器故障（Position）——保持独立，在上方且远离其他数据
sensor_mean = [-5, 8]  # 进一步远离其他类别
sensor_cov = [[2, 0.2], [0.2, 2]]
sensor_data = multivariate_normal.rvs(mean=sensor_mean, cov=sensor_cov, size=n_samples // 2)

# 合并所有数据
X = np.vstack([normal_data, jam_data, sensor_data, spall_data])
labels = ['Normal'] * n_samples + ['Jam'] * n_samples + ['Position'] * (n_samples // 2) + ['Spall'] * (n_samples // 2)


# ==================== 数据标准化（模拟 PCA 前处理）====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 为了模拟 PCA 的效果，我们手动设置“主成分”方向（不实际做 PCA）
# 这里直接使用原始数据作为“主成分”投影结果
pc1 = X_scaled[:, 0]  # 第一主成分
pc2 = X_scaled[:, 1]  # 第二主成分

# ==================== 绘图 ====================
plt.figure(figsize=(6, 4))

# 定义颜色和标记
colors = {'Normal': 'black', 'Jam': 'red', 'Position': 'blue', 'Spall': 'green'}
markers = {'Normal': 'o', 'Jam': 'o', 'Position': 'o', 'Spall': 'o'}

# 分类绘制
for label in colors:
    mask = [l == label for l in labels]
    plt.scatter(pc1[mask], pc2[mask], c=colors[label], marker=markers[label], s=10, alpha=0.7, label=label)

plt.rcParams['font.family'] = 'Times New Roman'

# 设置标签和标题
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')

plt.legend()
# 网格
plt.grid(True, alpha=0.3)

# 设置坐标轴范围（模拟原图）
plt.xlim(-5, 5)
plt.ylim(-3, 5)

# 布局优化
plt.tight_layout()
plt.savefig('plots/tsne_4_PCA_before.png', dpi=300, bbox_inches='tight')
# 显示图形
plt.show()