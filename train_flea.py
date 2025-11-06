# train_flea.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import json  # 用于保存结构化日志（可选）
from data.data_loader import Dataset_FLEA
from models.custom_model import TCCT
from data.config import Config


# ----------------------------
# 训练主函数
# ----------------------------
def main():
    config = Config()
    print(f"Using device: {config.device}")

    # === 日志目录 ===
    log_dir = "results/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_loss_log.txt")
    csv_file = os.path.join(log_dir, "train_loss_log.csv")

    # 写入 CSV 头部（首次运行）
    with open(csv_file, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")

    # Data
    train_dataset = Dataset_FLEA(
        root_path=config.root_path,
        flag='train',
        size=[config.seq_len, config.label_len, config.pred_len],
        data_path=config.data_path,
        target=config.target,
        scale=True
    )
    val_dataset = Dataset_FLEA(
        root_path=config.root_path,
        flag='val',
        size=[config.seq_len, config.label_len, config.pred_len],
        data_path=config.data_path,
        target=config.target,
        scale=True
    )
    test_dataset = Dataset_FLEA(
        root_path=config.root_path,
        flag='test',
        size=[config.seq_len, config.label_len, config.pred_len],
        data_path=config.data_path,
        target=config.target,
        scale=True
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Model
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

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Train
    for epoch in range(config.epochs):
        model.train()
        train_loss = []
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float().to(config.device)
            batch_y = batch_y.float().to(config.device)

            dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).to(config.device)

            outputs = model(batch_x, None, dec_inp, None)
            true = batch_y[:, -config.pred_len:, :].to(config.device)
            loss = criterion(outputs, true)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y, _, _ in val_loader:
                batch_x = batch_x.float().to(config.device)
                batch_y = batch_y.float().to(config.device)
                dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).to(config.device)
                outputs = model(batch_x, None, dec_inp, None)
                true = batch_y[:, -config.pred_len:, :].to(config.device)
                loss = criterion(outputs, true)
                val_loss.append(loss.item())

        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)

        # === 打印日志 ===
        log_msg = f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        print(log_msg)

        # === 保存到文本日志 ===
        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")

        # === 保存到 CSV ===
        with open(csv_file, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss:.8f},{avg_val_loss:.8f}\n")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/tcct_csp_flea.pth')
    print("✅ Model saved to checkpoints/tcct_csp_flea.pth")

    # Test prediction example
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
            if i >= 1: break
            batch_x = batch_x.float().to(config.device)
            batch_y = batch_y.float().to(config.device)
            dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).to(config.device)
            pred = model(batch_x, None, dec_inp, None)
            pred_original = test_dataset.inverse_transform(pred.cpu().numpy())
            true_original = test_dataset.inverse_transform(batch_y[:, -config.pred_len:, :].cpu().numpy())
            print("\n🔍 示例预测（原始电压值，单位：V）:")
            print("预测值:", np.round(pred_original.flatten()[:5], 3), "...")
            print("真实值:", np.round(true_original.flatten()[:5], 3), "...")

if __name__ == '__main__':
    main()