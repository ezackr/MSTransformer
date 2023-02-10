from typing import Optional, Tuple

import musdb
import numpy as np
import torch
from torch.utils.data import Dataset


class MUSDBDataset(Dataset):
    """
    Description.

    Parameters:
        - target:
        - root:
        - subsets:
        - is_wav:
        - download:
        - split:
        - duration:
        - samples_per_track:
    """
    def __init__(
            self,
            target: str = 'vocals',
            root: str = None,
            subsets: str = 'train',
            is_wav: bool = False,
            download: bool = False,
            split: str = 'train',
            duration: Optional[float] = 6.0,
            samples_per_track: int = 64
    ) -> None:
        self.mus = musdb.DB(
            root=root,
            subsets=subsets,
            is_wav=is_wav,
            download=download,
            split=split,
        )
        self.target = target
        self.is_wav = is_wav
        self.subsets = subsets
        self.split = split
        self.duration = duration
        self.samples_per_track = samples_per_track

    def _get_train_item(self, track):
        sources = []
        target_idx = None

        # randomly sample from each source to create a new mixture.
        for k, source in enumerate(self.mus.setup['sources']):
            if source == self.target:
                target_idx = k
            track.chunk_duration = self.duration
            track.chunk_start = np.random.uniform(low=0, high=track.duration - self.duration)
            audio = torch.as_tensor(track.sources[source].audio.T, dtype=torch.float32)
            sources.append(audio)

        # stems tensor of shape (num_sources, num_channels, 44100).
        stems = torch.stack(sources, dim=0)
        # let x be the mixture of the stems.
        x = stems.sum(0)
        if target_idx is not None:
            # y is the stem of the target source.
            y = stems[target_idx]
        else:
            # default target source is vocals.
            vocal_idx = list(self.mus.setup['sources'].keys()).index('vocals')
            y = x - stems[vocal_idx]
        return x, y

    def _get_val_item(self, track):
        x = torch.as_tensor(track.audio.T, dtype=torch.float32)
        y = torch.as_tensor(track.targets[self.target].audio.T, dtype=torch.float32)
        return x, y

    def __getitem__(self, idx):
        track = self.mus.tracks[idx // self.samples_per_track]
        if self.split == 'train':
            return self._get_train_item(track)
        else:
            return self._get_val_item(track)

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track


def load_dataset(
        target: str = 'vocals',
        root: str = None,
        subsets: str = 'train',
        is_wav: bool = False,
        download: bool = False,
        duration: Optional[float] = 6.0,
        samples_per_track: int = 64
) -> Tuple[MUSDBDataset, MUSDBDataset]:
    train_dataset = MUSDBDataset(
        target=target,
        root=root,
        subsets=subsets,
        is_wav=is_wav,
        download=download,
        split='train',
        duration=duration,
        samples_per_track=samples_per_track
    )
    validation_dataset = MUSDBDataset(
        target=target,
        root=root,
        subsets=subsets,
        is_wav=is_wav,
        download=download,
        split='valid',
        duration=duration,
        samples_per_track=samples_per_track
    )
    return train_dataset, validation_dataset
