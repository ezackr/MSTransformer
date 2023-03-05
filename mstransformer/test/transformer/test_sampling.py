import torch

from mstransformer.src.transformer.sampling import ConvLayer, DownSample, UpSample


def test_convlayer():
    x_in = torch.rand(size=(32, 2, 44100))
    conv = ConvLayer(in_channels=2, out_channels=4)
    x_out = conv(x_in)
    assert x_out.shape == (32, 4, 44100)


def test_downsample():
    x_in = torch.rand(size=(32, 2, 44100))
    down = DownSample(in_channels=2, out_channels=4)
    x_out = down(x_in)
    assert x_out[0].shape == (32, 4, 44100)
    assert x_out[1].shape == (32, 4, 44100 // 2)


def test_upsample():
    x_in = torch.rand(32, 4, 22050)
    context = torch.rand(32, 2, 44100)
    up = UpSample(in_channels=4, out_channels=2)
    x_out = up(x_in, context)
    assert x_out.shape == (32, 2, 44100)



