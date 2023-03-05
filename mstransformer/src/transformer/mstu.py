from tqdm.auto import tqdm

import torch
from torch import nn
from torch.nn import functional as F

from mstransformer.src.transformer.encoder import Encoder
from mstransformer.src.transformer.sampling import ConvLayer, DownSample, UpSample


def _get_next_power_of_two(n):
    k = 1
    while k < n:
        k *= 2
    return k


def pad_input(x):
    pad = _get_next_power_of_two(x.shape[-1]) - x.shape[-1]
    x_padded = F.pad(x, (0, pad))
    return x_padded


def crop_output(x, length):
    x_crop = x[..., :length]
    return x_crop


class MSTU(nn.Module):
    def __init__(
            self,
            num_channels=2,
            hidden_dim=4,
            num_sample_layers=8,
            num_trans_layers=6,
            num_heads=8,
            max_len=2048,
            dropout=0.0,
    ):
        super(MSTU, self).__init__()

        # down-sampling.
        self.down_sample_blocks = [DownSample(num_channels, hidden_dim)]
        in_channels = hidden_dim
        for _ in range(num_sample_layers - 1):
            self.down_sample_blocks.append(DownSample(in_channels, in_channels * 2))
            in_channels *= 2

        # transformer encoder.
        self.encoder = Encoder(
            num_layers=num_trans_layers,
            d_model=in_channels,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout
        )
        self.bottleneck = ConvLayer(in_channels, in_channels * 2)
        in_channels *= 2

        # up-sampling.
        self.up_sample_blocks = []
        for _ in range(num_sample_layers):
            self.up_sample_blocks.append(UpSample(in_channels, in_channels // 2))
            in_channels //= 2

        self.out_conv = ConvLayer(in_channels, num_channels)

    def forward(self, x):
        length = x.shape[-1]
        x_in = pad_input(x)

        # down-sample.
        context_list = []
        for block in self.down_sample_blocks:
            context, x_in = block(x_in)
            context_list.append(context)
        context_list.reverse()

        # transformer encoder.
        x_in = x_in.permute(0, 2, 1)
        x_out = self.encoder(x_in)
        x_out = x_out.permute(0, 2, 1)
        x_out = self.bottleneck(x_out)

        # up-sample.
        for i in range(len(context_list)):
            context = context_list[i]
            block = self.up_sample_blocks[i]
            x_out = block(x_out, context)

        # output convolution.
        x_out = self.out_conv(x_out)
        x_out = crop_output(x_out, length)
        return x_out
