import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# ----------------------------
# 配置（请根据你的训练设置调整）
# ----------------------------
SEQ_LEN = 200      # 编码器输入长度
LABEL_LEN = 30     # 解码器引导长度
PRED_LEN = 30      # 预测长度
MODEL_PATH = './checkpoints/informer_Normal_ftS_sl200_ll30_pl30_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_2/checkpoint.pth'
DATA_PATH = './data/test/Normal.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("use:", DEVICE)

# ----------------------------
# 1. 加载数据（仅使用 Motor Y Voltage）
# ----------------------------
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

# 单特征：只取 Motor Y Voltage
data_vals = df[['Motor Y Voltage']].values.astype(np.float32)  # (N, 1)

# 时间特征（6维，因 freq='s'）
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

print(f"单特征数据形状: {data_vals.shape}")
print(f"时间特征形状: {data_time.shape}")

# ----------------------------
# 2. 构建测试集
# ----------------------------
def create_univar_dataset(val, time, seq_len, label_len, pred_len):
    X_val, X_time, Y = [], [], []
    L = len(val)
    step = pred_len
    for i in range(0, L - seq_len - pred_len + 1, step):
        X_val.append(val[i:i + seq_len])                     # (seq_len, 1)
        X_time.append(time[i:i + seq_len])                   # (seq_len, 6)
        Y.append(val[i + seq_len:i + seq_len + pred_len, 0]) # (pred_len,)
    return np.array(X_val), np.array(X_time), np.array(Y)

X_val, X_time, Y_true = create_univar_dataset(data_vals, data_time, SEQ_LEN, LABEL_LEN, PRED_LEN)
print(f"X_val: {X_val.shape}, X_time: {X_time.shape}, Y_true: {Y_true.shape}")

# 转为 Tensor
X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
X_time = torch.tensor(X_time, dtype=torch.float32).to(DEVICE)
Y_true = torch.tensor(Y_true, dtype=torch.float32).to(DEVICE)

# ----------------------------
# 3. 构造 x_dec 和 x_mark_dec
# ----------------------------
B = X_val.shape[0]

# x_dec: [B, label_len + pred_len, 1]
dec_inp = torch.zeros(B, PRED_LEN, 1).to(DEVICE)
x_dec = torch.cat([X_val[:, -LABEL_LEN:, :], dec_inp], dim=1)  # (B, 60, 1)

# x_mark_dec: 未来时间特征
dt = df['date'].iloc[1] - df['date'].iloc[0]
x_mark_dec_list = []

for i in range(0, len(data_vals) - SEQ_LEN - PRED_LEN + 1, PRED_LEN):
    end_time = df['date'].iloc[i + SEQ_LEN - 1]
    future_dates = [end_time + (j + 1) * dt for j in range(PRED_LEN)]
    future_dates = pd.Series(future_dates)
    future_time_feats = get_time_features(future_dates)  # (30, 6)
    past_time = data_time[i + SEQ_LEN - LABEL_LEN : i + SEQ_LEN]  # (30, 6)
    dec_time = np.concatenate([past_time, future_time_feats], axis=0)  # (60, 6)
    x_mark_dec_list.append(dec_time)

x_mark_dec = torch.tensor(np.array(x_mark_dec_list), dtype=torch.float32).to(DEVICE)

# ----------------------------
# 4. 加载单特征 Informer 模型
# ----------------------------
from models.model import Informer

model = Informer(
    enc_in=1,        # ✅ 单特征输入
    dec_in=1,        # ✅
    c_out=1,         # 预测单变量
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
    embed='timeF',
    freq='s',        # → 6维时间特征
    activation='gelu'
).to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    print("✅ 单特征模型加载成功！")
else:
    raise FileNotFoundError(f"模型未找到: {MODEL_PATH}")

# ----------------------------
# 5. 推理
# ----------------------------
model.eval()
with torch.no_grad():
    pred = model(X_val, X_time, x_dec, x_mark_dec)  # 输出: (B, 30, 1)

pred_flat = pred.cpu().numpy().reshape(-1)
true_flat = Y_true.cpu().numpy().reshape(-1)

print(f"预测完成: {len(pred_flat)} 个时间步")

# ----------------------------
# 6. 绘图（使用图片中的颜色）
# ----------------------------
plt.figure(figsize=(14, 6))

# 使用你图片中的配色
true_color = '#003f5c'   # 深蓝
pred_color = '#7a8b2d'   # 黄绿

plt.plot(true_flat, label='true_result', color=true_color, linewidth=1.5)
plt.plot(pred_flat, label='pred_result', color=pred_color, linewidth=1.5, alpha=0.9)

plt.xlabel('Time Step')
plt.ylabel('Motor Y Voltage')
plt.title('Informer (Univariate Input) Prediction')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 保存高清图
plt.savefig('./prediction_univar.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 单特征预测图已保存！")