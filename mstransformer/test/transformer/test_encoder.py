import torch

from mstransformer.src.transformer.encoder import Encoder, EncoderBlock


def test_encoder_block():
    batch_size = 32
    d_model = 512
    seq_len = 256
    x = torch.rand(size=(batch_size, seq_len, d_model))

    block = EncoderBlock(d_model=d_model, num_heads=8, dropout=0.0)
    x_enc = block(x)
    assert x.shape == x_enc.shape


def test_encoder():
    batch_size = 32
    d_model = 512
    seq_len = 256
    x = torch.rand(size=(batch_size, seq_len, d_model))

    encoder = Encoder(
        num_layers=6,
        d_model=d_model,
        num_heads=8,
        max_len=seq_len,
        dropout=0.0
    )
    x_enc = encoder(x)
    assert x.shape == x_enc.shape
