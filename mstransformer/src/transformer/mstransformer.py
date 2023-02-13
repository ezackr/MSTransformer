import torch
from torch import nn

from mstransformer.src.transformer.decoder import Decoder
from mstransformer.src.transformer.encoder import Encoder
from mstransformer.src.transformer.preprocess import PreprocessLayer, PostprocessLayer


def _get_target_mask(batch_size):
    mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=0)
    mask[mask.bool()] = -1e9
    return mask


class MSTransformer(nn.Module):
    def __init__(
            self,
            num_layers=6,
            d_model=512,
            max_len=512,
            dropout=0.0,
    ):
        super(MSTransformer, self).__init__()
        self.preprocess = PreprocessLayer(d_model=d_model)
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )
        self.postprocess = PostprocessLayer(d_model=d_model)

    def forward(self, x, tgt):
        # preprocess data.
        x_spec = self.preprocess(x)
        t_spec = self.preprocess(tgt)
        # encode.
        x_enc = self.encoder(x_spec)
        # decode.
        t_mask = _get_target_mask(batch_size=len(x))
        x_hat = self.decoder(t_spec, x_enc, mask=t_mask)
        # reformat data.
        x_hat = self.postprocess(x_hat, length=tgt.shape[-1])
        t_hat = self.postprocess(t_spec, length=tgt.shape[-1])
        return x_hat, t_hat
