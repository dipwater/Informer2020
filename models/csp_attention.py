# models/csp_attention.py
import torch
import torch.nn as nn
import numpy as np

# ----------------------------
# CSPAttention 模块
# ----------------------------
class CSPAttention(nn.Module):
    def __init__(self, d_model, n_heads, conv_kernel_size=3, split_ratio=0.5, dropout=0.1):
        super(CSPAttention, self).__init__()
        assert 0 < split_ratio <= 1.0
        self.d_model = d_model
        self.n_heads = n_heads
        self.split_ratio = split_ratio

        self.d_attn = int(d_model * split_ratio)
        self.d_conv = d_model - self.d_attn

        if self.d_attn % n_heads != 0:
            self.d_attn = (self.d_attn // n_heads) * n_heads
            self.d_conv = d_model - self.d_attn
            if self.d_conv < 0:
                raise ValueError("d_model too small for given n_heads and split_ratio")

        # Attention branch
        if self.d_attn > 0:
            self.q_proj = nn.Linear(self.d_attn, self.d_attn)
            self.k_proj = nn.Linear(self.d_attn, self.d_attn)
            self.v_proj = nn.Linear(self.d_attn, self.d_attn)
            self.attn_dropout = nn.Dropout(dropout)
            self.out_proj_attn = nn.Linear(self.d_attn, self.d_attn)

        # Conv branch
        if self.d_conv > 0:
            padding = conv_kernel_size // 2
            self.conv = nn.Conv1d(
                in_channels=self.d_conv,
                out_channels=self.d_conv,
                kernel_size=conv_kernel_size,
                padding=padding,
                groups=self.d_conv
            )
            self.conv_dropout = nn.Dropout(dropout)
            self.out_proj_conv = nn.Linear(self.d_conv, self.d_conv)

        self.norm = nn.LayerNorm(d_model)
        self.final_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        assert D == self.d_model

        outputs = []

        if self.d_attn > 0:
            q_attn = queries[:, :, :self.d_attn]
            k_attn = keys[:, :, :self.d_attn]
            v_attn = values[:, :, :self.d_attn]

            Q = self.q_proj(q_attn).view(B, L, self.n_heads, -1).transpose(1, 2)
            K = self.k_proj(k_attn).view(B, L, self.n_heads, -1).transpose(1, 2)
            V = self.v_proj(v_attn).view(B, L, self.n_heads, -1).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e9)
            attn = self.attn_dropout(torch.softmax(scores, dim=-1))
            attn_out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, -1)
            attn_out = self.out_proj_attn(attn_out)
            outputs.append(attn_out)

        if self.d_conv > 0:
            x_conv = queries[:, :, self.d_attn:]
            conv_in = x_conv.transpose(1, 2)
            conv_out = self.conv(conv_in)
            conv_out = conv_out.transpose(1, 2)
            conv_out = self.conv_dropout(conv_out)
            conv_out = self.out_proj_conv(conv_out)
            outputs.append(conv_out)

        if len(outputs) == 2:
            out = torch.cat(outputs, dim=-1)
        else:
            out = outputs[0]

        out = self.dropout(self.final_proj(out))
        out = self.norm(queries + out)
        return out
