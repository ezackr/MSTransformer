import torch
from torch import nn

from mstransformer.src.transformer.position import PositionalEncoding


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.0):
        super(EncoderBlock, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads,
                                               dropout=dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x_attn = self.norm1(x + self.dropout1(self.attention(x, x, x)))
        x_out = self.norm2(x_attn + self.dropout2(self.feedforward(x_attn)))
        return x_out


class Encoder(nn.Module):
    def __init__(
            self,
            num_channels,
            num_heads=8,
            dropout=0.0,
            max_len=44100*5,
            num_layers=6
    ):
        super(Encoder, self).__init__()

        self.positional_encoding = PositionalEncoding(max_len, num_channels)
        self.blocks = [
            EncoderBlock(num_channels, num_heads, dropout)
            for _ in range(num_layers)
        ]

    def forward(self, x):
        seq_len, d_model = x.size(1), x.size(2)
        for block in self.blocks:
            pass
        pass
