import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.csp_attention import CSPAttention

# ----------------------------
# 辅助模块
# ----------------------------
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=padding, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.transpose(1, 2)

class EncoderLayer(nn.Module):
    def __init__(self, csp_attn, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.csp_attn = csp_attn
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, attn_mask=None):
        new_x = self.csp_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        y = x.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y)).transpose(1, 2)
        return self.norm2(x + y)

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x, attn_mask)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

# ----------------------------
# Decoder（使用标准注意力，支持因果掩码）
# ----------------------------
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / np.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.tril(torch.ones(L, S)).to(queries.device)
            scores.masked_fill_(attn_mask == 0, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out)

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask))
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask))
        x = self.norm2(x)
        y = x.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y)).transpose(1, 2)
        return self.norm3(x + y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

# ----------------------------
# TCCT 模型（集成 CSPAttention）
# ----------------------------
class TCCT(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 d_model=128, n_heads=4, e_layers=2, d_layers=1, dropout=0.1):
        super(TCCT, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.dec_embedding = nn.Linear(dec_in, d_model)

        # Encoder with CSPAttention
        csp_attn = CSPAttention(d_model, n_heads, conv_kernel_size=3, split_ratio=0.5, dropout=dropout)
        conv_layers = [ConvLayer(d_model) for _ in range(e_layers - 1)]
        self.encoder = Encoder(
            attn_layers=[EncoderLayer(csp_attn, d_model, dropout=dropout) for _ in range(e_layers)],
            conv_layers=conv_layers,
            norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder with standard attention
        dec_self_attn = AttentionLayer(FullAttention(mask_flag=True), d_model, n_heads)
        dec_cross_attn = AttentionLayer(FullAttention(mask_flag=False), d_model, n_heads)
        self.decoder = Decoder(
            layers=[DecoderLayer(dec_self_attn, dec_cross_attn, d_model, dropout=dropout)
                    for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out)
        dec_out = self.projection(dec_out)

        return dec_out[:, -self.pred_len:, :]
