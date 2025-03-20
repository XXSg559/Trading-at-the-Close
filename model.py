import torch
import torch.nn as nn
import torch.nn.functional as F
from bit import RMSNorm
import math


def custom_mask(seq_len, feature_dim, past_days, per_day=55):
    """
    生成一个自定义掩码:
    - 默认全 1
    - 最后一列(target 特征):
        - 过去时间步可见
        - 当前时间步和未来时间步屏蔽
    """
    past = past_days * per_day
    mask = torch.ones(seq_len - past, seq_len, feature_dim)  # 默认全 1

    for t in range(seq_len - past):
        mask[t, t + past, -1] = 0
        mask[t, t + past + 1 :, :] = 0  # 当前时间步及未来时间步的 target 置 0

    return mask


def causal_mask(seq_len):
    """
    生成一个因果掩码，防止attention关注未来位置。

    参数：
    - seq_len (int): 当前序列的长度。

    返回：
    - mask (torch.Tensor): (seq_len, seq_len) 的布尔掩码，`True` 表示被屏蔽的位置。
    """

    mask = torch.ones(seq_len, seq_len)
    mask = torch.tril(mask)

    return mask


def combined_rotary_embedding(
    max_len_day, max_len_minute, d_model, base_day=500, base_minute=20000
):
    """
    生成结合了日度和分钟级 RoPE 的 sin 和 cos 值。

    :param max_len_day: 日度位置编码的最大长度（例如 10 或 30）
    :param max_len_minute: 分钟级位置编码的最大长度（例如 55）
    :param d_model: 词向量维度（必须为偶数）
    :param base_day: 日度编码用的 base 值（较小，产生较大角度）
    :param base_minute: 分钟级编码用的 base 值（较大，产生较小角度）
    :return: sin_combined, cos_combined，形状为 (max_len_day, max_len_minute, d_model//2)
    """
    # 生成位置索引
    positions_day = torch.arange(max_len_day, dtype=torch.float32).unsqueeze(
        1
    )  # (max_len_day, 1)
    positions_minute = torch.arange(max_len_minute, dtype=torch.float32).unsqueeze(
        1
    )  # (max_len_minute, 1)

    # 计算 div_term 对于日度和分钟级别
    # 每两个维度共享一个旋转角度
    div_term_day = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(base_day) / d_model)
    )
    div_term_minute = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(base_minute) / d_model)
    )

    # 计算每个位置对应的角度
    theta_day = positions_day * div_term_day  # (max_len_day, d_model//2)
    theta_minute = positions_minute * div_term_minute  # (max_len_minute, d_model//2)

    # 组合两个时间尺度的角度，使用笛卡尔积（广播加法）：
    # theta_combined 的形状为 (max_len_day, max_len_minute, d_model//2)
    theta_combined = theta_day.unsqueeze(1) + theta_minute.unsqueeze(0)

    # 计算合成后的 sin 和 cos 值
    sin_combined = torch.sin(theta_combined)
    cos_combined = torch.cos(theta_combined)

    # reshape
    sin_combined = sin_combined.view(max_len_day * max_len_minute, d_model // 2)
    cos_combined = cos_combined.view(max_len_day * max_len_minute, d_model // 2)

    return sin_combined, cos_combined


# copy from transformers
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_d = d_model // self.n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=1)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, position_ids, mask=None):
        batch, time, dim = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch, time, self.n_heads, self.n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_heads, self.n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_heads, self.n_d).permute(0, 2, 1, 3)

        cos, sin = position_ids
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.n_d)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch, time, dim)

        return output


class LightSwiGLU(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super().__init__()
        # 中间维度设为 d_ffn * 2/3（与标准 SwiGLU 一致）
        hidden_size = int(2 * d_ffn / 3)
        self.w = nn.Linear(d_model, hidden_size)
        self.v = nn.Linear(d_model, hidden_size)
        self.w2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        swiglu = F.silu(self.w(x)) * self.v(x)
        x = self.dropout(swiglu)
        return self.w2(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.attn1 = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = LightSwiGLU(d_model, dim_feedforward, dropout)
        self.norm2 = RMSNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        positions_ids,
        tgt_mask=None,
    ):
        _x = tgt
        x = self.attn1(tgt, positions_ids, tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x


# @torch.compile
class Decoder(nn.Module):
    r"""
    Decoder 的简单实现，由多个 DecoderLayer 组成。
    args:
        d_model,
        n_head,
        dim_feedforward,
        num_layers,
        dropout
    """

    def __init__(self, d_model, n_head, dim_feedforward, num_layers, dropout=0.1):

        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_head, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, positions_ids, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, positions_ids, tgt_mask)
        return x
