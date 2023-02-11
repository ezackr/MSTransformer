import torch

from mstransformer.src.transformer.preprocess import PreprocessLayer


def test_preprocesslayer():
    batch_size = 32
    rate = 44100.0
    duration = 2.0
    audio = torch.rand(size=(batch_size, int(rate * duration)))

    preprocess = PreprocessLayer()
    spec = preprocess(audio)

    assert spec.shape[0] == audio.shape[0]
    assert spec.shape[-1] == preprocess.input_dim
