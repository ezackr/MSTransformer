import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()

    def forward(self):
        pass


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

    def forward(self):
        pass


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
