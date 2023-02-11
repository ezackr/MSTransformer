import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Applies Positional Encoding to each position in a sequence.
    Uses encoding proposed in "Attention Is All You Need".
    """

    def __init__(self, seq_len=512, d_model=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(seq_len, d_model, dtype=torch.float)
        pe.require_grad = False

        pos = torch.arange(start=0, end=seq_len, dtype=torch.float).unsqueeze(1)
        dim = torch.arange(start=0, end=d_model, step=2, dtype=torch.float)
        dim = torch.exp(dim * (-math.log(10e4) / d_model))

        pe[:, 0::2] = torch.sin(pos * dim)
        pe[:, 1::2] = torch.cos(pos * dim)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, :]
