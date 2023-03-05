import time
from tqdm.auto import tqdm

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch.optim import AdamW

from mstransformer.src.transformer import MSTU
from mstransformer.src.utils import sdr, get_number_of_parameters, load_dataset


def get_dataloader(
        target: str = 'vocals',
        duration: float = 1.0,
        samples_per_track: int = 8,
        batch_size: int = 64
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
        samples_per_track=4,
        batch_size=32
    )

    model = MSTU(
        num_channels=2,
        hidden_dim=4,
        num_sample_layers=8,
        num_trans_layers=6,
        num_bottleneck_layers=4,
        num_heads=8,
        max_len=1024,
        dropout=0.1
    )
    optimizer = AdamW(model.parameters())

    num_epochs = 4
    train_losses = []
    val_losses = []
    for i in tqdm(range(num_epochs)):
        start_time = time.time()

        model.train()
        total_loss = 0
        # TRAINING LOOP:
        for x, y in tqdm(train_loader):
            # zero gradients.
            optimizer.zero_grad()
            # get source estimate.
            y_hat = model(x)
            # calculate loss and update parameters.
            loss = mse_loss(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        total_loss = 0
        # VALIDATION LOOP:
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                y_hat = model(x)
                loss = mse_loss(y_hat, y)
                total_loss += loss.item()
        val_losses.append(total_loss / len(val_loader))

        print(f'\nepoch {i + 1} summary: '
              f'train_loss={train_losses[-1]} '
              f'val_loss={val_losses[-1]} '
              f'time={(time.time() - start_time) / 60}m\n')
    print(f'(FINAL) train loss={train_losses}')
    print(f'(FINAL) val loss={val_losses}')

    if save_artifact:
        artifact_name = 'mstu_4sample_4epoch_exp2.pt'
        path = F'/Users/elliottzackrone/PycharmProjects/artifacts/{artifact_name}'
        torch.save(model.state_dict(), path)


def evaluate():
    # load model artifact.
    model = MSTU(dropout=0.1)
    artifact_name = 'mstu_128sample_20epoch.pt'
    path = F'/Users/elliottzackrone/PycharmProjects/artifacts/{artifact_name}'
    model.load_state_dict(torch.load(path))
    model.eval()
    # get validation dataset.
    _, val_loader = get_dataloader(
        target='vocals',
        duration=1.0,
        batch_size=32
    )

    # evaluate model.
    total_score = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            y_hat = model(x)
            total_score += sdr(target=y, estimate=y_hat).item()
        total_score = total_score / len(val_loader)
    print(f'MSTU model = \"{path}\", \n'
          f'number of parameters = {get_number_of_parameters(model)}\n'
          f'evaluation SDR = {total_score}')


if __name__ == '__main__':
    mode = 'train'

    if mode == 'train':
        print(f'Training...')
        train()
    else:
        print(f'Evaluating...')
        evaluate()
