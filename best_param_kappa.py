import matplotlib.pyplot as plt

# 设置中文字体（macOS 或 Windows 兼容）
plt.rcParams['font.sans-serif'] = ['STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 数据
A = [1, 3, 5, 7, 10, 15, 20, 30]  # 窗口滑动长度
accuracy = [96.5, 98.5, 99.5, 99.7, 99.9, 100.0, 100.0, 100.0]  # 准确率 (%)
kappa = [0.97, 0.98, 0.99, 0.995, 0.998, 1.00, 1.00, 1.00]     # Kappa 系数

# 创建图形
fig, ax1 = plt.subplots(figsize=(6, 4))

# 左侧 Y 轴：准确率
color = 'red'
ax1.plot(A, accuracy, color=color, marker='*', markersize=10, linewidth=2, label='准确率')
ax1.set_xlabel('窗口滑动长度A', fontsize=14)
ax1.set_ylabel('准确率 (%)', fontsize=14, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 右侧 Y 轴：Kappa 系数
ax2 = ax1.twinx()
color = 'blue'
ax2.plot(A, kappa, color=color, marker='^', markersize=10, linewidth=2, label='kappa系数')
ax2.set_ylabel('kappa系数', fontsize=14, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# 设置坐标轴范围
ax1.set_ylim(95, 103)
ax2.set_ylim(0.94, 1.02)

# 设置 X 轴刻度
ax1.set_xticks(A)
ax1.set_xticklabels(A)

# 图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.grid(True, color='#cccccc', linestyle='-', alpha=0.5)

# 对于坐标轴上的数字，我们需要单独设置字体
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontname('Times New Roman')  # 设置为 Times New Roman

# 对于坐标轴上的数字，我们需要单独设置字体
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontname('Times New Roman')  # 设置为 Times New Roman

# 标题与图注
plt.tight_layout()

plt.savefig('plots/best_param_kappa_accuracy.png', dpi=300)

# 显示或保存
plt.show()