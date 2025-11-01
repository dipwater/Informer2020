# 文件: exp/exp_ema.py

from data_provider.data_factory import data_provider
from data_provider.data_loader import EMADataset  # ← 新增导入
from models import Informer
import torch

class Exp_EMA_Informer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_model(self):
        model = Informer(
            enc_in=1,          # 单变量输入
            dec_in=1,
            c_out=1,
            seq_len=self.args.seq_len,
            label_len=self.args.label_len,
            out_len=self.args.pred_len,
            factor=self.args.factor,
            d_model=self.args.d_model,
            n_heads=self.args.n_heads,
            e_layers=self.args.e_layers,
            d_layers=self.args.d_layers,
            d_ff=self.args.d_ff,
            dropout=self.args.dropout,
            attn=self.args.attn,      # 'prob' for ProbSparse
            embed=self.args.embed,
            freq=self.args.freq,
            activation=self.args.activation,
            output_attention=self.args.output_attention,
            distil=self.args.distil,
            mix=self.args.mix,
            device=self.device
        ).float()
        return model.to(self.device)

    def _get_data(self, flag):
        # 使用自定义 EMADataset
        Data = EMADataset
        timeenc = 0  # 不需要时间编码（无日期）

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = 1
        else:
            shuffle_flag = True; drop_last = True; batch_size = self.args.batch_size

        data_set = Data(
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            scale=True
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader

    def train(self):
        model = self._build_model()
        train_data, train_loader = self._get_data(flag='train')
        # ... 标准训练循环（可复用原代码）
        # 为节省篇幅，此处省略，实际可保留原 train() 逻辑

    def test(self, model=None):
        test_data, test_loader = self._get_data(flag='test')
        if model is None:
            model = self._build_model()
            # 加载已训练权重
            model.load_state_dict(torch.load(self.args.checkpoint_path))
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                dec_inp = torch.zeros_like(y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs = model(x, dec_inp)
                pred = outputs.detach().cpu().numpy()
                true = y[:, -self.args.pred_len:, :].detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae = np.mean(np.abs(preds - trues))
        return mae, preds, trues