import torch

from mstransformer.src.transformer.mstransformer import MSTransformer


def test_mstransformer():
    batch_size = 32
    total_length = 44100
    x = torch.rand(size=(batch_size, total_length))
    target = torch.rand(size=(batch_size, total_length))

    transformer = MSTransformer(d_model=512, max_len=512, dropout=0.0)
    x_hat, t_hat = transformer(x, target)
    assert x_hat.shape == t_hat.shape == (32, 44100)
