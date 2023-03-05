import torch
from torch import nn


class ConvLayer(nn.Module):
    """
    Double convolutional layer using BatchNorm and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=(3,), padding=1, bias=False)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=(3,), padding=1, bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_out = self.dropout1(self.relu(self.norm1(self.conv1(x))))
        x_out = self.dropout2(self.relu(self.norm2(self.conv2(x_out))))
        return x_out


class DownSample(nn.Module):
    """
    Uses double convolution and max pooling to down-sample a waveform.
    Returns both `x_conv` and `x_pool`. `x_conv` is used as context for the
    decoder. `x_pool` is used as input for the next layer.
    """
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels)
        self.max_pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.max_pool(x_conv)
        return x_conv, x_pool


class UpSample(nn.Module):
    """
    Uses both transpose convolution and double convolution to up-sample a
    waveform. Has two inputs. `x` has shape (batch_size, in_channels, L) and
    `context` of shape (batch_size, out_channels, 2*L). Output has shape
    (batch_size, out_channels, 2*L).
    """
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2,),
            stride=(2,),
        )
        self.conv = ConvLayer(out_channels + out_channels, out_channels)

    def forward(self, x, context):
        x_out = self.conv_transpose(x)
        x_out = torch.cat([x_out, context], dim=1)
        x_out = self.conv(x_out)
        return x_out
