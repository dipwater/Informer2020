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
n_samples = 500  # 增加总样本数，使点更密集

# —————— 靠拢设计：将三类集中在 (5, -2) 附近 ——————

# 正常状态（Normal）
normal_mean = [6, -2]
normal_cov = [[6, 2], [2, 3]]  # 稍微扩大，便于重叠
normal_data = multivariate_normal.rvs(mean=normal_mean, cov=normal_cov, size=n_samples)

# 卡滞（Jam）——靠近 Normal
jam_mean = [4, -3]
jam_cov = [[7, 2.5], [2.5, 4]]
jam_data = multivariate_normal.rvs(mean=jam_mean, cov=jam_cov, size=n_samples)

# 剥落（Spall）——也靠近中心区域
spall_mean = [5, -1]
spall_cov = [[5, 1.5], [1.5, 3]]
spall_data = multivariate_normal.rvs(mean=spall_mean, cov=spall_cov, size=n_samples // 2)

# 传感器故障（Position）——保持独立，在上方
sensor_mean = [-2, 4]
sensor_cov = [[1.2, 0.1], [0.1, 1.2]]
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
plt.savefig('plots/tsne_3_PCA_before.png', dpi=300, bbox_inches='tight')
# 显示图形
plt.show()