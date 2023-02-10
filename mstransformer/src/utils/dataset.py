from typing import Optional, Tuple

import musdb
import numpy as np
import torch
from torch.utils.data import Dataset


class MUSDBDataset(Dataset):
    def __init__(
            self,
            target: str = 'vocals',
            root: str = None,
            download: bool = False,
            is_wav: bool = False,
            subsets: str = 'train',
            split: str = 'train',
            duration: Optional[float] = 6.0,
            samples_per_track: int = 64,
            *args,
            **kwargs
    ) -> None:
        self.mus = musdb.DB(
            root=root,
            is_wav=is_wav,
            split=split,
            subsets=subsets,
            download=download,
            *args,
            **kwargs
        )
        self.target = target
        self.is_wav = is_wav
        self.subsets = subsets
        self.split = split
        self.duration = duration
        self.samples_per_track = samples_per_track
        self.sample_rate = 44100.0

    def __getitem__(self, idx):
        sources = []
        target_idx = None

        target_track = self.mus.tracks[idx // self.samples_per_track]

        if self.split == 'train' and self.duration:
            # load training data
            for k, source in enumerate(self.mus.setup['sources']):
                if source == self.target:
                    target_idx = k

                target_track.chunk_duration = self.duration
                target_track.chunk_start = np.random.uniform(low=0, high=target_track.duration - self.duration)
                audio = torch.as_tensor(target_track.sources[source].audio.T, dtype=torch.float32)
                sources.append(audio)

            stems = torch.stack(sources, dim=0)
            x = stems.sum(0)
            if target_idx:
                y = stems[target_idx]
            else:
                vocal_idx = list(self.mus.setup['sources'].keys()).index('vocals')
                y = x - stems[vocal_idx]
        else:
            # load validation data
            x = torch.as_tensor(target_track.audio.T, dtype=torch.float32)
            y = torch.as_tensor(target_track[self.target].audio.T, dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track


def load_dataset(
        target: str = 'vocals',
        root: str = None,
        download: bool = False,
        is_wav: bool = False,
        subsets: str = 'train',
        duration: Optional[float] = 6.0,
        samples_per_track: int = 64,
        *args,
        **kwargs
) -> Tuple[MUSDBDataset, MUSDBDataset]:
    train_dataset = MUSDBDataset(
        target=target,
        root=root,
        download=download,
        is_wav=is_wav,
        subsets=subsets,
        split='train',
        duration=duration,
        samples_per_track=samples_per_track,
        *args,
        **kwargs
    )
    validation_dataset = MUSDBDataset(
        target=target,
        root=root,
        download=download,
        is_wav=is_wav,
        subsets=subsets,
        split='valid',
        duration=duration,
        samples_per_track=samples_per_track
    )
    return train_dataset, validation_dataset
