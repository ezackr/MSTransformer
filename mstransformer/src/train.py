import torch
from torch import nn
from torch.utils.data import DataLoader

from mstransformer.src.utils import ComplexNorm
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


def get_spec_enc(device=None):
    stft, _ = get_fourier_transforms()
    spec_enc = nn.Sequential(
        stft,
        ComplexNorm()
    ).to(device)
    return spec_enc


def train():
    print(f'----- loading dataset:')
    train_loader, val_loader = get_dataloader()
    spec_enc = get_spec_enc()


if __name__ == '__main__':
    train()
