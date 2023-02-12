import torch
from torch import nn

from mstransformer.src.utils import get_fourier_transforms


def complex_norm(spec):
    return torch.abs(torch.view_as_complex(spec))


class PreprocessLayer(nn.Module):
    """
    Preprocesses data into transformer format. Input is default MUSDB
    tensor of shape (batch_size, sample_rate * duration). Output is STFT
    of shape (batch_size, num_frames, num_bins).
    """

    def __init__(
            self,
            n_fft: int = 2048,
            hop_length: int = 512,
            center: bool = False
    ):
        super(PreprocessLayer, self).__init__()

        self.stft, _ = get_fourier_transforms(
            n_fft=n_fft,
            hop_length=hop_length,
            center=center)
        self.input_dim = (n_fft - 2) / 2 + 2

    def forward(self, x):
        x_spec = complex_norm(self.stft(x))
        x_spec = x_spec.permute(0, 2, 1)
        return x_spec
