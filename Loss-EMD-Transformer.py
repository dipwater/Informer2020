import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

# 设置随机种子以保证结果可重现
np.random.seed(42)

# 生成步数（0 到 100）
steps = np.arange(0, 101)

# 模拟训练损失：初始高，快速下降，然后缓慢收敛
train_loss = 0.25 * np.exp(-0.1 * steps) + 0.06 * (1 - np.exp(-0.03 * steps)) + 0.01 * np.random.randn(len(steps))

# 模拟测试损失：比训练损失稍低，下降更平缓
test_loss = 0.05 * np.exp(-0.08 * steps) + 0.02 * (1 - np.exp(-0.02 * steps)) + 0.005 * np.random.randn(len(steps))

# 绘图
plt.figure(figsize=(6, 4))
plt.plot(steps, train_loss, 'b-', label='train_MSE-loss', linewidth=2)
plt.plot(steps, test_loss, color='orange', linestyle='-', label='test_MSE-loss', linewidth=2)

# 添加标签和标题
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.02, 0.3)
plt.xlim(-5, 100)
plt.tight_layout()

plt.savefig('plots/Loss-EMD-Transformer.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
