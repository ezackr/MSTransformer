import torch
from torch import nn


class STFT(nn.Module):
    def __init__(
            self,
            n_fft: int = 4096,
            hop_length: int = 1024,
            center: bool = False
    ):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    def forward(self, x):
        # x.shape == (num_samples, seq_len)
        x_stft = x.view(-1, x.shape[-1])
        x_stft = torch.stft(
            x_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        x_stft = torch.view_as_real(x_stft)
        x_stft = x_stft.view(x.shape[:-1] + x_stft.shape[-3:])
        # x_stft.shape == (num_samples, num_bins, num_frames, 2)
        return x_stft


class ISTFT(nn.Module):
    def __init__(
            self,
            n_fft: int = 4096,
            hop_length: int = 1024,
            center: bool = False
    ):
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    def forward(self, x, length=None):
        # x.shape == (num_samples, num_bind, num_frames, 2)
        x_out = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        x_out = torch.istft(
            torch.view_as_complex(x_out),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length
        )
        x_out = x_out.view(x.shape[:-3] + x_out.shape[-1:])
        # x_out.shape == (num_samples, seq_len)
        return x_out


class ComplexNorm(nn.Module):
    def __init__(self, mono: bool = True):
        super(ComplexNorm, self).__init__()
        self.mono = mono

    def forward(self, x):
        spec = torch.abs(torch.view_as_complex(x))
        if self.mono:
            spec = torch.mean(spec, dim=1, keepdim=True)
        return spec


def get_fourier_transforms(
        n_fft: int = 4096,
        hop_length: int = 1024,
        center=False
):
    encoder = STFT(n_fft=n_fft, hop_length=hop_length, center=center)
    decoder = ISTFT(n_fft=n_fft, hop_length=hop_length, center=center)
    return encoder, decoder
