import torch
from torch import nn
from torch.utils.data import DataLoader

from mstransformer.src.utils import load_dataset, get_fourier_transforms


def get_dataloader(
        target: str = 'vocals',
        root: str = None,
        subsets: str = 'train',
        is_wav: bool = False,
        download: bool = True,
        duration: float = 1.0,
        samples_per_track: int = 16,
        batch_size: int = 32,
):
    train_dataset, val_dataset = load_dataset(
        target, root, subsets, is_wav,
        download, duration, samples_per_track
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=32)
    return train_loader, val_loader


def complex_norm(spec):
    return torch.abs(torch.view_as_complex(spec))


def train(
    target: str = 'vocals',
    root: str = None,
    subsets: str = 'train',
    is_wav: bool = False,
    download: bool = True,
    duration: float = 2.0,
    samples_per_track: int = 16,
    batch_size: int = 32,
    sample_rate: float = 44100.0,
):
    num_epochs = 5

    print(f'Loading Dataset:')
    train_loader, val_loader = get_dataloader(
        target, root, subsets, is_wav, download,
        duration, samples_per_track, batch_size
    )
    stft, _ = get_fourier_transforms(n_fft=2048, hop_length=512)

    count = 0
    for x, y in train_loader:
        if count >= 3:
            break
        x_spec = stft(x)
        x_spec = complex_norm(x_spec)
        print(x_spec.shape)

        count += 1


if __name__ == '__main__':
    train(

    )
