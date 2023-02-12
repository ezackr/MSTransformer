import torch
from torch import nn

from mstransformer.src.transformer.decoder import Decoder
from mstransformer.src.transformer.encoder import Encoder
from mstransformer.src.transformer.preprocess import PreprocessLayer


def _get_target_mask(batch_size):
    mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=0)
    mask[mask.bool()] = -float('inf')
    return mask


class MSTransformer(nn.Module):
    def __init__(
            self,
            d_model=512,
            max_len=512,
            dropout=0.0,
    ):
        super(MSTransformer, self).__init__()

        # preprocess layer converts raw audio to STFT spectrogram.
        self.preprocess = PreprocessLayer()
        input_dim = self.preprocess.input_dim

        # set data to be of size `d_model`.
        self.in_linear = nn.Linear(input_dim, d_model)
        # encode spectrogram.
        self.encoder = Encoder(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )

        # decode spectrogram.
        self.decoder = Decoder(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )
        # reset data to input size.
        self.out_linear = nn.Linear(d_model, input_dim)

    def forward(self, x, tgt):
        # preprocess data.
        x_spec = self.preprocess(x)
        x_spec = self.in_linear(x_spec)
        # encode.
        x_enc = self.encoder(x_spec)
        # decode.
        tgt_mask = _get_target_mask(batch_size=len(x))
        y_hat = self.decoder(tgt, x_enc, mask=tgt_mask)
        # reformat data.
        y_hat = self.out_linear(y_hat)
        return y_hat
