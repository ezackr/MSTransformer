from mstransformer.src.utils import MUSDBDataset, load_dataset


def test_musdb():
    # NOTE: First test run will take roughly 10 minutes to load the dataset.
    # Subsequent tests should run in < 45 secs.
    musdb = MUSDBDataset(download=True, samples_per_track=1, duration=1.0)
    for x, y in musdb:
        assert x.shape[-1] == 44100


def test_load_dataset():
    _, _ = load_dataset(download=True)
