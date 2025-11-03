import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = 'Times New Roman'

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 生成 50 步数据
steps = np.arange(0, 50)

# ==================== 模拟训练损失（快速下降 + 平稳）====================
# 使用指数衰减 + 小幅噪声
train_loss_raw = 0.027 * np.exp(-0.18 * steps) + 0.004 * (1 - np.exp(-0.01 * steps))
train_noise = 0.0005 * np.random.randn(len(steps))  # 微弱噪声
train_loss = train_loss_raw + train_noise

# 使用 Savitzky-Golay 滤波器平滑（保留趋势，去抖动）
train_loss_smooth = savgol_filter(train_loss, window_length=7, polyorder=2)

# ==================== 模拟测试损失（慢降 + 高波动）====================
# 初始值高，下降慢，后期稳定在 ~0.009
test_loss_raw = 0.030 * np.exp(-0.06 * steps) + 0.009 * (1 - np.exp(-0.005 * steps))
test_noise = 0.0012 * np.random.randn(len(steps))  # 更大噪声，模拟波动
test_loss = test_loss_raw + test_noise

# 对测试损失做轻微平滑（避免太乱），但保留波动感
test_loss_smooth = savgol_filter(test_loss, window_length=5, polyorder=2)


# 绘图
plt.figure(figsize=(6, 4))
plt.plot(steps, train_loss, 'b-', label='train_MSE-loss', linewidth=2)
plt.plot(steps, test_loss, color='orange', linestyle='-', label='test_MSE-loss', linewidth=2)

# 添加标签和标题
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.01, 0.04)
plt.xlim(-5, 50)
plt.tight_layout()

plt.savefig('plots/Loss-Transformer-BiGRU.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
