import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Applies Positional Encoding to each position in a sequence, where
    `dim_model = num_channels = 2`. Uses encoding proposed in
    "Attention Is All You Need".
    """

    def __init__(self, seq_len=441000, d_model=2):
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
        # uses default shape from MUSDBDataset.
        # x.shape == (batch_size, num_channels, seq_length)
        x = x.permute(0, 2, 1)
        # x.shape == (batch_size, seq_length, num_channels)
        x_enc = x + self.pe[:, :x.size(1), :]
        x_enc = x_enc.permute(0, 2, 1)
        # x.shape == (batch_size, num_channels, seq_length)
        return x_enc
