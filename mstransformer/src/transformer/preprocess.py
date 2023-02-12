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
            center: bool = False,
            d_model: int = 512,
    ):
        super(PreprocessLayer, self).__init__()

        # apply STFT to input data.
        self.stft, _ = get_fourier_transforms(
            n_fft=n_fft,
            hop_length=hop_length,
            center=center)
        input_dim = (n_fft - 2) // 2 + 2
        print(input_dim)
        # reduce complex to single dimension.
        self.linear1 = nn.Linear(2, 1)
        # reduce default dimension to d_model
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
