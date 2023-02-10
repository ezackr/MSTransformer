import torch

from mstransformer.src.transformer.position import PositionalEncoding


def test_positional_encoding():
    batch_size = 32
    d_model = 2
    seq_len = 44100
    x = torch.rand(size=(batch_size, d_model, seq_len))

    position_encoding = PositionalEncoding(seq_len, d_model)
    x_enc = position_encoding(x)
    assert x_enc.shape == x.shape
    assert x_enc[0][0][0] == x[0][0][0]
    assert x_enc[0][0][1] != x[0][0][1]
