import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

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
    avg, std = train_dataset.get_statistics()
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, avg, std


def train():
    save_artifact = True

    train_loader, val_loader, avg, std = get_dataloader(
        target='vocals',
        duration=1.0,
        samples_per_track=4,
        batch_size=32
    )

    model = MSTU(
        num_channels=2,
        hidden_dim=64,
        num_sample_layers=4,
        num_trans_layers=12,
        num_bottleneck_layers=4,
        num_heads=8,
        max_len=4096,
        dropout=0.5
    )
    optimizer = AdamW(model.parameters(), lr=1e-4)

    num_epochs = 4
    train_losses = []
    val_losses = []
    for i in tqdm(range(num_epochs)):
        start_time = time.time()

        model.train()
        total_loss = 0
        # TRAINING LOOP:
        for x, y in tqdm(train_loader):
            x = (x - avg) / std
            y = (y - avg) / std
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
                x = (x - avg) / std
                y = (y - avg) / std
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
        artifact_name = 'mstu_4sample_4epoch_exp8.pt'
        path = F'/Users/elliottzackrone/PycharmProjects/artifacts/{artifact_name}'
        torch.save(model.state_dict(), path)


def evaluate():
    # load model artifact.
    model = MSTU(
        num_channels=2,
        hidden_dim=64,
        num_sample_layers=4,
        num_trans_layers=12,
        num_bottleneck_layers=4,
        num_heads=8,
        max_len=4096,
        dropout=0.5
    )
    artifact_name = 'mstu_actually_good_low_complexity.pt'
    path = F'/Users/elliottzackrone/PycharmProjects/artifacts/{artifact_name}'
    model.load_state_dict(torch.load(path))
    model.eval()
    # get validation dataset.
    _, val_loader, avg, std = get_dataloader(
        target='vocals',
        duration=1.0,
        batch_size=32
    )

    # evaluate model.
    total_score = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = (x - avg) / std
            y = (y - avg) / std
            y_hat = model(x)
            total_score += sdr(target=y, estimate=y_hat).item()
        total_score = total_score / len(val_loader)
    print(f'MSTU model = \"{path}\", \n'
          f'number of parameters = {get_number_of_parameters(model)}\n'
          f'evaluation SDR = {total_score}')


def analysis():
    # load model artifact.
    model = MSTU(
        num_channels=2,
        hidden_dim=64,
        num_sample_layers=4,
        num_trans_layers=12,
        num_bottleneck_layers=4,
        num_heads=8,
        max_len=4096,
        dropout=0.5
    )
    artifact_name = 'mstu_actually_good_low_complexity.pt'
    path = F'/Users/elliottzackrone/PycharmProjects/artifacts/{artifact_name}'
    model.load_state_dict(torch.load(path))
    model.eval()
    # get validation dataset.
    train_dataset, val_dataset = load_dataset(
        target='vocals',
        download=True,
        duration=1.0,
        samples_per_track=4
    )
    avg, std = train_dataset.get_statistics()
    # get standardized estimates.
    x = ((val_dataset[0][0] - avg) / std).unsqueeze(0)
    y = ((val_dataset[0][1] - avg) / std).unsqueeze(0)
    y_hat = model(x).detach()
    # convert to mono channel.
    y = torch.mean(y, dim=1).squeeze()
    y_hat = torch.mean(y_hat, dim=1).squeeze()
    # plot results.
    length = 44100
    # plot true source.
    plt.plot(range(length), y, color='blue')
    plt.title('True Source')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.ylim(top=2.5, bottom=-2.0)
    plt.show()
    # plot estimated source.
    plt.plot(range(length), y_hat, color='orange')
    plt.title('MSTU Estimate Source')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.ylim(top=2.5, bottom=-2.0)
    plt.show()


if __name__ == '__main__':
    mode = 'analyze'

    if mode == 'train':
        print(f'Training...')
        train()
    elif mode == 'evaluate':
        print(f'Evaluating...')
        evaluate()
    elif mode == 'analyze':
        print(f'Analyzing...')
        analysis()
