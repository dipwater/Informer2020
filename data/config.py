
import torch

# ----------------------------
# 配置参数
# ----------------------------
class Config:
    seq_len = 120      # 60 秒历史
    label_len = 30     # 15 秒已知前缀
    pred_len = 20      # 预测未来 10 秒
    batch_size = 32
    learning_rate = 1e-4
    epochs = 10
    root_path = './data/FLEA/'
    data_path = 'Normal.csv'
    target = 'Motor Y Voltage'
    enc_in = 7
    dec_in = 1
    c_out = 1
    d_model = 128
    n_heads = 4
    e_layers = 2
    d_layers = 1
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'