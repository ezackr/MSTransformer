import torch
from torch import nn

from mstransformer.src.utils import get_fourier_transforms


class PreprocessLayer(nn.Module):
    """
    Preprocesses data into transformer format. Input is default MUSDB
    tensor of shape (batch_size, sample_rate * duration). Output is modified
    STFT transformation of shape (batch_size, num_frames, d_model).
    """
    def __init__(
            self,
            n_fft: int = 2048,
            hop_length: int = 512,
            center: bool = True,
            d_model: int = 512,
    ):
        super(PreprocessLayer, self).__init__()
        self.stft, _ = get_fourier_transforms(
            n_fft=n_fft,
            hop_length=hop_length,
            center=center)
        input_dim = (n_fft - 2) // 2 + 2
        self.linear1 = nn.Linear(2, 1)
        self.linear2 = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # x.shape == (batch_size, sample_rate * duration)
        x_spec = self.stft(x)
        # x_spec.shape == (batch_size, num_bins, num_frames, 2)
        x_spec = self.linear1(x_spec).squeeze().permute(0, 2, 1)
        # x_spec.shape == (batch_size, num_frames, num_bins)
        x_spec = self.linear2(x_spec)
        # x_spec.shape == (batch_size, num_frames, d_model)
        return x_spec


class PostprocessLayer(nn.Module):
    """
    Converts transformer output to original audio format. Input is
    transformer output of shape (batch_size, num_frames, d_model). Output
    is tensor of shape (batch_size, sample_rate * duration).
    """
    def __init__(
            self,
            n_fft: int = 2048,
            hop_length: int = 512,
            center: bool = True,
            d_model: int = 512
    ):
        super(PostprocessLayer, self).__init__()
        stft_dim = (n_fft - 2) // 2 + 2
        self.linear1 = nn.Linear(d_model, stft_dim)
        self.linear2 = nn.Linear(1, 2)
        _, self.istft = get_fourier_transforms(
            n_fft=n_fft,
            hop_length=hop_length,
            center=center
        )

    def forward(self, x, length=None):
        # x.shape == (batch_size, num_frames, d_model)
        x_audio = self.linear1(x).permute(0, 2, 1)
        x_audio = x_audio.unsqueeze(1).permute(0, 2, 3, 1)
        x_audio = self.linear2(x_audio)
        x_audio = self.istft(x_audio, length=length)
        return x_audio
