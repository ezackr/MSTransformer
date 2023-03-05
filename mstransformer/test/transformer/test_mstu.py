import torch

from mstransformer.src.transformer.mstu import MSTU, pad_input, crop_output


def test_pad_input():
    batch_size = 32
    num_channels = 2
    length = 44100
    x_in = torch.rand(size=(batch_size, num_channels, length))
    x_pad = pad_input(x_in)
    assert x_pad.shape == (32, 2, 65536)


def test_pad_input_power_of_two():
    batch_size = 32
    num_channels = 2
    length = 64
    x_in = torch.rand(size=(batch_size, num_channels, length))
    x_pad = pad_input(x_in)
    assert x_pad.shape == x_in.shape


def test_crop_output():
    batch_size = 32
    num_channels = 2
    length = 44100
    x_in = torch.rand(size=(batch_size, num_channels, length))
    x_pad = pad_input(x_in)
    x_crop = crop_output(x_pad, length)
    assert x_crop.shape == x_in.shape


def test_mstu():
    batch_size = 32
    num_channels = 2
    length = 44100
    x_in = torch.rand(size=(batch_size, num_channels, length))
    model = MSTU()
    x_out = model(x_in)
    assert x_out.shape == x_in.shape
