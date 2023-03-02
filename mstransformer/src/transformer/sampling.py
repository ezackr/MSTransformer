import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=(3,), padding=1, bias=False)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=(3,), padding=1, bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_out = self.relu(self.norm1(self.conv1(x)))
        x_out = self.relu(self.norm2(self.conv2(x_out)))
        return x_out


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

    def forward(self):
        pass


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

    def forward(self):
        pass
