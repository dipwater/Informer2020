import re
import matplotlib.pyplot as plt


def calc_losses(loss_file):
    train_losses = []
    val_losses = []

    with open(loss_file, 'r') as f:
        for line in f:
            match = re.search(r'Train Loss:\s*([\d.]+).*?Vali.*?Loss:\s*([\d.]+)', line)
            if match:
                train_losses.append(float(match.group(1)))
                val_losses.append(float(match.group(2)))

    if train_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='train_MSE-loss', color='blue')
        plt.plot(val_losses, label='test_MSE-loss', color='orange')
        plt.title('Recovered Loss Curve from Console Output')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.savefig(loss_file.replace('.txt', '.png'), dpi=300)
        print("✅ 从终端日志恢复 loss 曲线成功！")
    else:
        print("❌ 未找到 loss 信息，请检查文件格式。")


if __name__ == '__main__':
    calc_losses('./results/FLEA_S_loss2.txt')
