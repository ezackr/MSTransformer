import torch

from mstransformer.src.transformer.preprocess import PreprocessLayer, PostprocessLayer


def test_preprocesslayer():
    batch_size = 32
    rate = 44100.0
    duration = 2.0
    audio = torch.rand(size=(batch_size, int(rate * duration)))

    d_model = 512
    preprocess = PreprocessLayer(d_model=d_model)
    spec = preprocess(audio)

    assert spec.shape[0] == audio.shape[0]
    assert spec.shape[-1] == d_model


def test_postprocesslayer():
    batch_size = 32
    seq_len = 173
    d_model = 512
    spec = torch.rand(size=(batch_size, seq_len, d_model))
    total_length = 44100 * 2

    postprocess = PostprocessLayer(d_model=d_model)
    audio = postprocess(spec, length=total_length)

    assert audio.shape[0] == spec.shape[0]
    assert audio.shape[-1] == total_length
