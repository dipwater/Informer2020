import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据准备（参考你的表格）
models = ['RNN', 'LSTM', 'GRU', 'Transformer', 'Informer', 'The proposed method']
mae = [0.1461, 0.1480, 0.1294, 0.0560, 0.0592, 0.0420]
mse = [0.0300, 0.0331, 0.0239, 0.0089, 0.0059, 0.0040]
rmse = [0.1732, 0.1819, 0.1545, 0.0943, 0.0768, 0.0632]

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 创建图形
fig, ax = plt.subplots(figsize=(12, 6))

# 设置柱子宽度和位置
x = np.arange(len(models))
width = 0.25  # 柱子宽度

# 绘制柱状图
bars1 = ax.bar(x - width, mae, width, label='MAE', color='#b5cf6b', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, mse, width, label='MSE', color='#7f8d6c', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, rmse, width, label='RMSE', color='#324e69', edgecolor='black', linewidth=0.5)

# 设置坐标轴
ax.set_xlabel('模型')
ax.set_ylabel('误差值')
ax.set_title('不同模型预测结果对比')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 0.20)  # Y轴范围
ax.grid(axis='y', alpha=0.3)  # 添加水平网格线

# 添加图例
ax.legend(loc='upper right')

# 调整布局防止标签被截断
plt.tight_layout()

# 保存图片（可选）
plt.savefig('plots/model_comparison_bar.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()