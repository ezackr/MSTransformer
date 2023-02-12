import time
from tqdm.auto import tqdm

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch.optim import AdamW

from mstransformer.src.transformer import MSTransformer
from mstransformer.src.utils import load_dataset


def get_dataloader(
        target: str = 'vocals',
        duration: float = 1.0,
        samples_per_track: int = 8,
        batch_size: int = 32
):
    train_dataset, val_dataset = load_dataset(
        target=target,
        download=True,
        duration=duration,
        samples_per_track=samples_per_track
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def train():
    save_artifact = True

    train_loader, val_loader = get_dataloader(
        target='vocals',
        duration=1.0,
        samples_per_track=8,
        batch_size=16
    )

    model = MSTransformer(dropout=0.1)
    optimizer = AdamW(model.parameters())

    num_epochs = 10
    train_losses = []
    val_losses = []
    for i in tqdm(range(num_epochs)):
        start_time = time.time()

        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            x_hat, t_hat = model(x, y)
            loss = mse_loss(x_hat, t_hat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x_hat, t_hat = model(x, y)
                loss = mse_loss(x_hat, t_hat)
                total_loss += loss.item()
        val_losses.append(total_loss / len(val_loader))

        print(f'\nepoch {i} summary: '
              f'train_loss={train_losses[-1]} '
              f'val_loss={val_losses[-1]} '
              f'time={(time.time() - start_time) / 60}m\n')

    if save_artifact:
        artifact_name = 'mstransformer_10epoch.pt'
        path = F'/Users/elliottzackrone/PycharmProjects/artifacts/{artifact_name}'
        torch.save(model.state_dict(), path)


if __name__ == '__main__':
    train()
