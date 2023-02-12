import torch
from torch import nn

from mstransformer.src.transformer.decoder import Decoder, DecoderBlock


def test_decoder_block():
    batch_size = 32
    d_model = 512
    seq_len = 256
    x = torch.rand(size=(batch_size, seq_len, d_model))
    tgt = torch.rand(size=(batch_size, seq_len, d_model))

    mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=1)
    mask[mask.bool()] = -float('inf')

    decoder = DecoderBlock(
        d_model=d_model,
        num_heads=8,
        dropout=0.0
    )
    x_dec = decoder(tgt, x, mask)
    assert x_dec.shape == x.shape


def test_decoder():
    batch_size = 32
    d_model = 512
    seq_len = 256
    x = torch.rand(size=(batch_size, seq_len, d_model))
    tgt = torch.rand(size=(batch_size, seq_len, d_model))

    mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=1)
    mask[mask.bool()] = -float('inf')

    decoder = Decoder(
        num_layers=6,
        d_model=d_model,
        num_heads=8,
        max_len=512,
        dropout=0.0
    )
    x_dec = decoder(tgt, x, mask)
    assert x_dec.shape == x.shape
