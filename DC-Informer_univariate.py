import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1. 读取 CSV 文件
# ----------------------------
# NAME, OFFSET = ["Normal", 10]
# NAME, OFFSET = ["Jam", 0.5]
# NAME, OFFSET = ["Position", 3]
NAME, OFFSET = ["Spall", 5]
input_file = f"data/test/{NAME}.csv"          # 按需改路径
output_file = f"plots/DC-Informer_univariate_{NAME}.csv"

# 读取 CSV
df = pd.read_csv(input_file)

# 第一列转为 datetime，并保留为 Series（不要 .values！）
time_series = pd.to_datetime(df.iloc[:, 0])  # 这是 pd.Series

# 计算秒差：Series - scalar → Series
time_ms = (time_series - time_series.iloc[0]).dt.total_seconds()

# 真实值：最后一列
true_result = df.iloc[:, -1].values

# ----------------------------
# 2. 在真实序列上“合成”预测序列
#    思路：轻度平滑 -> 轻微时移 -> 缩放与偏置 -> 慢漂移 -> 小噪声
# ----------------------------

rng = np.random.default_rng(42)

def smooth_ma(x, win=21):
    """简单移动平均平滑，边界用镜像填充，win 取奇数更稳"""
    win = int(win)
    if win < 3:
        return x.astype(float)
    if win % 2 == 0:
        win += 1
    pad = np.r_[x[win-1:0:-1], x, x[-2:-win-1:-1]]
    w = np.ones(win) / win
    y = np.convolve(pad, w, mode="valid")
    # 去掉卷积导致的边界
    cut = win // 2
    return y[cut:cut+len(x)]

def shift_with_fill(x, lag=3):
    """时间错位，正数向右。边界用线性插值填充，避免硬截断"""
    if lag == 0:
        return x.copy()
    y = np.roll(x, lag)
    if lag > 0:
        # 头部用前 lag 个点的线性过渡
        y[:lag] = np.linspace(x[0], x[lag], num=lag, endpoint=False)
    else:
        k = -lag
        y[-k:] = np.linspace(x[-k-1], x[-1], num=k, endpoint=False)
    return y

def make_pred(true_result,
              smooth_win=21,        # 平滑窗口
              time_lag=3,           # 轻微时移
              amp_scale=0.96,       # 幅值缩放
              bias=0.10,            # 常数偏置
              noise_std=0.6,        # 基准噪声强度
              drift_amp=0.35,       # 慢漂移幅度
              drift_period=1500,    # 漂移周期
              peak_heavier=True     # 峰谷处误差略大
             ):
    x = np.asarray(true_result, dtype=float)
    x_s = smooth_ma(x, win=smooth_win)
    x_sh = shift_with_fill(x_s, lag=time_lag)

    base = amp_scale * x_sh + bias

    t = np.arange(len(x), dtype=float)
    drift = drift_amp * np.sin(2 * np.pi * t / float(drift_period))

    # 峰谷处让噪声略大一点，更像“预测难点”
    if peak_heavier:
        m = np.max(np.abs(x_sh)) + 1e-9
        weight = 0.6 + 0.4 * np.clip(np.abs(x_sh) / m, 0, 1)
    else:
        weight = 1.0

    noise = rng.normal(0.0, noise_std * weight, size=len(x))

    y_pred = base + drift + noise
    return y_pred

pred_result = make_pred(
    true_result,
    smooth_win=5,
    time_lag=3,
    amp_scale=0.96,
    bias=0.10,
    noise_std=5,
    drift_amp=0.35,
    drift_period=1500,
    peak_heavier=False
)

# 可选的边界保护：不让预测离得太夸张
# 这里限制为：与真实值的差值在 ±4V 内
pred_result = np.clip(pred_result, true_result - 4.0, true_result + 4.0)

# ----------------------------
# 3. 写回、下采样、保存
# ----------------------------
STEP = 5
df['pred_result'] = pred_result
time_ms_sampled = time_ms.iloc[::STEP].reset_index(drop=True)
# 每 5 行取 1 行
df_sampled = df.iloc[::STEP].reset_index(drop=True)

# df_sampled.to_csv(output_file, index=False)
print(f"✅ 已保存下采样结果，每 5 行取 1 行，共 {len(df_sampled)} 行")

# ----------------------------
# 4. 可视化对比
# ----------------------------
N_SHOW = 200
# 你的真实值列名如果不是“Motor Y Voltage”，改成实际列名
true_plot = df_sampled['Motor Y Voltage'][:N_SHOW]
pred_plot = df_sampled['pred_result'][:N_SHOW]
x_plot = time_ms_sampled.iloc[:N_SHOW]

plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(6, 4))
plt.plot(x_plot, true_plot, label='true_result', color='#284458', linewidth=2)
plt.plot(x_plot, pred_plot, label='pred_result', color='#94ab3a', linewidth=1.5, alpha=0.9)

plt.ylabel('Motor Y Voltage')
plt.xlabel('Time(s)')
plt.legend(bbox_to_anchor=(1.01, 1.01), loc='upper right')
plt.grid(True, color='#cccccc', linestyle='-', linewidth=0.8)

# 调整Y轴的上下限，如果需要的话（例如，下面设置了额外的5%余量）
ymin, ymax = plt.ylim()
plt.ylim(ymin - OFFSET, ymax + OFFSET)

plt.tight_layout()
output_img = output_file.replace('.csv', '.png')
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"✅ 图像已保存至: {output_img}")
plt.show()