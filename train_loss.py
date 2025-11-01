import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# -------------------------------
# 1. 配置路径与设备
# -------------------------------
MODEL_PATH = './checkpoints/informer_FLEA_ftMS_sl500_ll50_pl50_dm512_nh8_el2_dl1_df2048_atprob_ebtimeF_dtTrue_Exp_fixed_0/checkpoint.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 2. 加载模型（需与训练时一致）
# -------------------------------
from models.model import Informer  # 确保路径正确

# 根据你的训练命令重建模型
model = Informer(
    enc_in=7,          # 特征数（['Actuator Z Position', ..., 'Motor Y Voltage']）
    dec_in=7,
    c_out=1,           # 预测单变量
    seq_len=500,
    label_len=50,
    pred_len=50,
    factor=5,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=2048,
    dropout=0.05,
    attn='prob',
    embed='fixed',
    freq='t',
    activation='gelu',
    output_attention=False,
    distil=True,
    mix=True,
    device=DEVICE
)

# 加载权重
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
model.to(DEVICE)
model.eval()
print("✅ 模型加载成功！")

# -------------------------------
# 3. 构建数据加载器（需与训练时一致）
# -------------------------------
from data.data_loader import DataLoader

# 假设你使用的是 Informer2020 的 data_loader
root_path = './data/FLEA/'
data_parser = {
    'FLEA': {'root_path': root_path, 'data_path': 'Normal.csv', 'target': 'Motor Y Voltage'}
}

# 共同参数
data_args = {
    'root_path': root_path,
    'data_path': 'Normal.csv',
    'target': 'Motor Y Voltage',
    'features': 'MS',  # 或 'S'
    'scale': True,
    'inverse': False,
    'timeenc': 0,
    'freq': 't',
    'cols': None
}

# 训练集（通常前 70%）
train_dataset = Dataset_FLEA(
    **data_args,
    flag='train',
    size=[500, 50, 50],
    batch_size=64
)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

# 验证集（中间 10%）
val_dataset = Dataset_FLEA(
    **data_args,
    flag='val',
    size=[500, 50, 50],
    batch_size=64
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

print(f"训练集 batch 数: {len(train_loader)}")
print(f"验证集 batch 数: {len(val_loader)}")

# -------------------------------
# 4. 推理并收集预测 & 真实值
# -------------------------------
def evaluate_model(model, data_loader, device):
    preds = []
    trues = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -50:, :]).float()
            dec_inp = torch.cat([batch_y[:, :50, :], dec_inp], dim=1).float().to(device)

            # forward
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # 只取预测部分（最后 50 步）
            f_dim = -1 if data_args['features'] == 'MS' else 0
            outputs = outputs[:, -50:, f_dim:]
            batch_y = batch_y[:, -50:, f_dim:]

            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues

print("\n🔍 在训练集上推理...")
train_preds, train_trues = evaluate_model(model, train_loader, DEVICE)

print("🔍 在验证集上推理...")
val_preds, val_trues = evaluate_model(model, val_loader, DEVICE)

# -------------------------------
# 5. 计算 MSE Loss
# -------------------------------
train_mse = mean_squared_error(train_trues.flatten(), train_preds.flatten())
val_mse = mean_squared_error(val_trues.flatten(), val_preds.flatten())

print(f"\n📊 最终损失:")
print(f"  Train MSE: {train_mse:.6f}")
print(f"  Val   MSE: {val_mse:.6f}")

# -------------------------------
# 6. 绘制对比图
# -------------------------------
plt.figure(figsize=(8, 5))
x = ['Train', 'Validation']
y = [train_mse, val_mse]
colors = ['steelblue', 'orange']

bars = plt.bar(x, y, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
plt.title('Final Model Performance (MSE Loss)', fontsize=14)
plt.ylabel('MSE Loss', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱子上显示数值
for bar, loss in zip(bars, y):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + loss*0.01,
             f'{loss:.4f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
output_dir = './plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/final_model_loss.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 图像已保存: {output_dir}/final_model_loss.png")