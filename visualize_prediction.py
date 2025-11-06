import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# --- 请确保以下模块路径正确 ---
from data.data_loader import Dataset_FLEA  # 你的 Dataset_FLEA
from models.custom_model import TCCT                    # 你的 TCCT 模型
from data.config import Config

def load_model(model_path, config):
    model = TCCT(
        enc_in=config.enc_in,
        dec_in=config.dec_in,
        c_out=config.c_out,
        seq_len=config.seq_len,
        label_len=config.label_len,
        pred_len=config.pred_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_layers=config.d_layers,
        dropout=config.dropout
    ).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()
    return model

def visualize_predictions(config, model_path, num_samples=3, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    # 加载测试集（不 shuffle）
    test_dataset = Dataset_FLEA(
        root_path=config.root_path,
        flag='test',
        size=[config.seq_len, config.label_len, config.pred_len],
        data_path=config.data_path,
        target=config.target,
        scale=True
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = load_model(model_path, config)

    # 存储所有预测和真实值用于整体绘图（可选）
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for idx, (batch_x, batch_y, _, _) in enumerate(test_loader):
            if idx >= num_samples:
                break

            batch_x = batch_x.float().to(config.device)
            batch_y = batch_y.float().to(config.device)

            # 构造 decoder 输入
            dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).to(config.device)

            # 预测
            pred = model(batch_x, None, dec_inp, None)  # [1, pred_len, 1]

            # 还原为原始电压值
            pred_original = test_dataset.inverse_transform(pred.cpu().numpy())
            true_original = test_dataset.inverse_transform(batch_y[:, -config.pred_len:, :].cpu().numpy())

            all_preds.append(pred_original.flatten())
            all_trues.append(true_original.flatten())

            # 绘制单个样本
            plt.figure(figsize=(12, 4))
            x_axis = np.arange(config.pred_len)
            plt.plot(x_axis, true_original.flatten(), 'o-', label='Ground Truth', linewidth=2)
            plt.plot(x_axis, pred_original.flatten(), 's--', label='Prediction', linewidth=2)
            plt.title(f'Test Sample {idx + 1}: {config.target} Prediction (Voltage)')
            plt.xlabel('Time Step (0.5s per step)')
            plt.ylabel('Voltage (V)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_sample_{idx+1}.png'), dpi=300)
            plt.close()

    # 绘制所有样本叠加图（可选）
    if all_preds:
        plt.figure(figsize=(14, 5))
        for i, (pred, true) in enumerate(zip(all_preds, all_trues)):
            x = np.arange(i * config.pred_len, (i + 1) * config.pred_len)
            plt.plot(x, true, 'o-', color='C0', alpha=0.7, markersize=3)
            plt.plot(x, pred, 's-', color='C1', alpha=0.7, markersize=3)
        plt.title(f'All Test Predictions ({len(all_preds)} samples)')
        plt.xlabel('Time Step (concatenated)')
        plt.ylabel('Voltage (V)')
        plt.legend(['Ground Truth', 'Prediction'], loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'all_predictions.png'), dpi=300)
        plt.show()
        plt.close()

    print(f"✅ 可视化完成！图像已保存至 {save_dir}/")

if __name__ == '__main__':
    config = Config()
    config.root_path = 'data/test'
    config.data_path = 'Normal.csv'
    model_path = 'checkpoints/tcct_csp_flea.pth'
    visualize_predictions(config, model_path, num_samples=1)