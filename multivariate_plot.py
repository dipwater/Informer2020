import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# ----------------------------
# 1. 配置参数
# ----------------------------
SEQ_LEN = 200      # 历史输入长度
LABEL_LEN = 30     # 解码器引导长度（通常 = pred_len）
PRED_LEN = 30      # 预测长度
MODEL_PATH = './checkpoints/informer_custom_ftS_sl200_ll30_pl30_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_2/checkpoint.pth'  # 替换为你的模型路径
DATA_PATH = './data/test/Normal.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# 1. 加载数据
# ----------------------------
df = pd.read_csv(DATA_PATH)

# 解析时间戳
df['date'] = pd.to_datetime(df['date'])

# 提取数值特征（第1~7列，共7列）
feature_cols = [
    'Actuator Z Position',
    'Motor Z Current',
    'Motor Y Temperature',
    'Motor Z Temperature',
    'Nut Y Temperature',
    'Ambient Temperature',
    'Motor Y Voltage'  # 包含目标的历史值作为输入
]
data_vals = df[feature_cols].values.astype(np.float32)  # (N, 7)

# 提取时间特征（6维，freq='s'）
def get_time_features(dt_series):
    return np.stack([
        dt_series.dt.second.values,
        dt_series.dt.minute.values,
        dt_series.dt.hour.values,
        dt_series.dt.weekday.values,
        dt_series.dt.day.values,
        dt_series.dt.month.values
    ], axis=1).astype(np.float32)

data_time = get_time_features(df['date'])  # (N, 6)

print(f"数值特征形状: {data_vals.shape}")
print(f"时间特征形状: {data_time.shape}")

# ----------------------------
# 2. 构建测试集
# ----------------------------
def create_dataset(val, time, seq_len, label_len, pred_len):
    X_val, X_time, Y = [], [], []
    L = len(val)
    step = pred_len  # non-overlapping
    for i in range(0, L - seq_len - pred_len + 1, step):
        X_val.append(val[i:i + seq_len])                     # (200, 7)
        X_time.append(time[i:i + seq_len])                   # (200, 6)
        Y.append(val[i + seq_len:i + seq_len + pred_len, -1])  # (30,) ← 最后一列
    return np.array(X_val), np.array(X_time), np.array(Y)

X_val, X_time, Y_true = create_dataset(data_vals, data_time, SEQ_LEN, LABEL_LEN, PRED_LEN)
print(f"X_val: {X_val.shape}, X_time: {X_time.shape}, Y_true: {Y_true.shape}")

# 转为 Tensor
X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
X_time = torch.tensor(X_time, dtype=torch.float32).to(DEVICE)
Y_true = torch.tensor(Y_true, dtype=torch.float32).to(DEVICE)

# ----------------------------
# 3. 构造 x_dec 和 x_mark_dec
# ----------------------------
B = X_val.shape[0]

# x_dec: [B, label_len + pred_len, 7]
dec_inp = torch.zeros(B, PRED_LEN, 7).to(DEVICE)
x_dec = torch.cat([X_val[:, -LABEL_LEN:, :], dec_inp], dim=1)  # (B, 60, 7)

# x_mark_dec: 需要未来时间特征
# 推断采样间隔（假设等间隔）
dt = df['date'].iloc[1] - df['date'].iloc[0]  # 如 10ms
x_mark_dec_list = []

for i in range(0, len(data_vals) - SEQ_LEN - PRED_LEN + 1, PRED_LEN):
    # 当前窗口结束时间
    end_time = df['date'].iloc[i + SEQ_LEN - 1]
    # 生成未来 PRED_LEN 个时间点
    future_dates = [end_time + (j + 1) * dt for j in range(PRED_LEN)]
    future_dates = pd.Series(future_dates)
    future_time_feats = get_time_features(future_dates)  # (30, 6)
    # 过去 label_len 个时间特征
    past_time = data_time[i + SEQ_LEN - LABEL_LEN : i + SEQ_LEN]  # (30, 6)
    dec_time = np.concatenate([past_time, future_time_feats], axis=0)  # (60, 6)
    x_mark_dec_list.append(dec_time)

x_mark_dec = torch.tensor(np.array(x_mark_dec_list), dtype=torch.float32).to(DEVICE)
print(f"x_mark_dec shape: {x_mark_dec.shape}")

# ----------------------------
# 4. 加载模型
# ----------------------------
from models.model import Informer

model = Informer(
    enc_in=7,        # ✅ 7个输入特征
    dec_in=7,
    c_out=1,         # 预测单变量（Motor Y Voltage）
    seq_len=SEQ_LEN,
    label_len=LABEL_LEN,
    pred_len=PRED_LEN,
    factor=5,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=2048,
    dropout=0.05,
    attn='prob',
    embed='timeF',   # 必须与训练一致
    freq='s',        # 秒级 → 6维时间特征
    activation='gelu'
).to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ 模型加载成功！")
else:
    raise FileNotFoundError(MODEL_PATH)

# ----------------------------
# 5. 推理
# ----------------------------
model.eval()
with torch.no_grad():
    pred = model(X_val, X_time, x_dec, x_mark_dec)  # 输出: (B, 30, 1)

pred_flat = pred.cpu().numpy().reshape(-1)
true_flat = Y_true.cpu().numpy().reshape(-1)

# ----------------------------
# 6. 绘图
# ----------------------------
plt.figure(figsize=(14, 6))
plt.plot(true_flat, label='True Result', color='darkblue')
plt.plot(pred_flat, label='Predicted Result', color='olive', alpha=0.8)
plt.xlabel('Time Step')
plt.ylabel('Motor Y Voltage')
plt.title('Informer Prediction (7-var input, 1-var output)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./prediction.png', dpi=300)
plt.show()

print("✅ 完成！")
