import torch

from mstransformer.src.utils.fourier import STFT, ISTFT


def test_stft():
    num_samples = 32
    num_channels = 2
    seq_len = 44100
    x = torch.randn(size=(num_samples, num_channels, seq_len))

    stft = STFT()
    x_stft = stft(x)
    assert x_stft.shape == (num_samples, num_channels, 2049, 40, 2)


def test_istft():
    num_samples = 32
    num_channels = 2
    seq_len = 44100
    x = torch.randn(size=(num_samples, num_channels, seq_len))

    stft = STFT(center=True)
    x_stft = stft(x).detach()

    istft = ISTFT(center=True)
    x_out = istft(x_stft, length=x.shape[-1])
    assert x_out.shape == (num_samples, num_channels, seq_len)
