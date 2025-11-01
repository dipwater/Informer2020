import numpy as np
import torch
from exp.exp_informer import Exp_Informer


def load_model(checkpoint_path, args):
    exp = Exp_Informer(args)
    exp.model.load_state_dict(torch.load(checkpoint_path))
    exp.model.eval()
    return exp


def compute_mae(exp_model, test_data, seq_len=96, pred_len=24):
    # 将 test_data 切分为多个 (seq_len + pred_len) 片段
    mae_list = []
    for i in range(0, len(test_data) - seq_len - pred_len, pred_len):
        batch = test_data[i:i + seq_len + pred_len]
        x = batch[:seq_len].reshape(1, -1, 1)  # [B, L, D]
        y_true = batch[seq_len:seq_len + pred_len]

        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            pred = exp_model.model(x).cpu().numpy().flatten()

        mae = np.mean(np.abs(pred - y_true))
        mae_list.append(mae)
    return np.mean(mae_list)


# 加载四个模型（需构建相同的 args）
states = ["Normal", "Position", "Jam", "Spall"]
models = {}
for state in states:
    args = get_informer_args(state)  # 你需定义此函数，返回训练时的参数
    model = load_model(f"./checkpoints/{state}/model.pth", args)
    models[state] = model

# 诊断
test_sequence = np.loadtxt("test_MotorY.csv")  # 仅 Motor Y Voltage 数值
maes = {}
for state, model in models.items():
    mae = compute_mae(model, test_sequence)
    maes[state] = mae

predicted_state = min(maes, key=maes.get)
print(f"诊断结果: {predicted_state}, MAE: {maes[predicted_state]:.4f}")