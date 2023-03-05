from mstransformer.src.utils import MUSDBDataset, load_dataset


def test_musdb():
    # NOTE: First test run will take roughly 10 minutes to load the dataset.
    # Subsequent tests should run in < 45 secs.
    musdb = MUSDBDataset(download=True, samples_per_track=1, duration=1.0, mono=True)
    for x, y in musdb:
        assert x.shape == y.shape
        # default sampling frequency for MUSDB is 44.1 kHz.
        assert x.shape[-1] == 44100


def test_load_dataset_mono():
    num_samples = 32
    train_dataset, val_dataset = load_dataset(download=True, samples_per_track=num_samples, mono=True)
    assert len(train_dataset) == 80 * num_samples
    assert len(val_dataset) == 14 * num_samples


def test_load_dataset_stereo():
    num_samples = 16
    train_dataset, val_dataset = load_dataset(download=True, samples_per_track=num_samples, mono=False)
    assert len(train_dataset) == 80 * num_samples
    assert len(train_dataset[0]) == 2
    assert train_dataset[0][0].shape == (2, 220500)
