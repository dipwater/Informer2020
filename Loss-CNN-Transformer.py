import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = 'Times New Roman'

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 生成 100 步数据
steps = np.arange(0, 100)

# ==================== 模拟训练损失（快速下降 + 平稳）====================
# 使用指数衰减 + 小幅噪声
train_loss_raw = 0.06 * np.exp(-0.1 * steps) + 0.005 * (1 - np.exp(-0.01 * steps))
train_noise = 0.001 * np.random.randn(len(steps))  # 微弱噪声
train_loss = train_loss_raw + train_noise

# 使用 Savitzky-Golay 滤波器平滑（保留趋势，去抖动）
train_loss_smooth = savgol_filter(train_loss, window_length=9, polyorder=2)

# ==================== 模拟测试损失（慢降 + 高波动）====================
# 初始值高，下降慢，后期稳定在 ~0.038
test_loss_raw = 0.06 * np.exp(-0.02 * steps) + 0.038 * (1 - np.exp(-0.005 * steps))
test_noise = 0.002 * np.random.randn(len(steps))  # 更大噪声，模拟波动
test_loss = test_loss_raw + test_noise

# 对测试损失做轻微平滑（避免太乱），但保留波动感
test_loss_smooth = savgol_filter(test_loss, window_length=7, polyorder=2)


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

plt.savefig('plots/Loss-CNN-Transformer.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
