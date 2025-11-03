import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = 'Times New Roman'

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 生成100步（对应于你的需求）
steps = np.linspace(0, 100, 100)

# ==================== 模拟训练损失（快速下降 → 收敛到极低值）====================
train_loss_raw = 0.05 * np.exp(-0.15 * steps) + 0.002 * np.exp(-0.02 * steps)
# 添加小幅度噪声（模拟训练波动）
train_noise = 0.0005 * np.random.randn(len(steps))
train_loss = train_loss_raw + train_noise

# 使用Savitzky-Golay滤波器进行平滑（保留趋势，去除剧烈抖动）
train_loss_smooth = savgol_filter(train_loss, window_length=11, polyorder=3)

# ==================== 模拟测试损失（初始低，缓慢下降，稳定在 ~0.008）====================
test_loss_raw = 0.012 * np.exp(-0.05 * steps) + 0.008 * (1 - np.exp(-0.01 * steps))
test_noise = 0.0008 * np.random.randn(len(steps))
test_loss = test_loss_raw + test_noise
test_loss_smooth = savgol_filter(test_loss, window_length=11, polyorder=3)


# 绘图
plt.figure(figsize=(6, 4))
plt.plot(steps, train_loss, 'b-', label='train_MSE-loss', linewidth=2)
plt.plot(steps, test_loss, color='orange', linestyle='-', label='test_MSE-loss', linewidth=2)

# 添加标签和标题
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.01, 0.06)
plt.xlim(-5, 100)
plt.tight_layout()

plt.savefig('plots/Loss-Transformer-BiLSTM.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
