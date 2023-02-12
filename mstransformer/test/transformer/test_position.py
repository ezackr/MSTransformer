import torch

from mstransformer.src.transformer.position import PositionalEncoding


def test_positional_encoding():
    batch_size = 32
    d_model = 512
    seq_len = 256
    x = torch.rand(size=(batch_size, seq_len, d_model))

    position_encoding = PositionalEncoding(seq_len, d_model)
    x_enc = position_encoding(x)
    assert x_enc.shape == x.shape
    assert x_enc[0][0][0] == x[0][0][0]
    assert x_enc[0][0][1] != x[0][0][1]
