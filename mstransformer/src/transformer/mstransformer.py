import torch
from torch import nn

from mstransformer.src.transformer.encoder import Encoder
from mstransformer.src.transformer.preprocess import PreprocessLayer


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
        self.linear = nn.Linear(input_dim, d_model)
        # encode spectrogram.
        self.encoder = Encoder(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )

        # decode spectrogram.
        self.decoder = lambda x: x

    def forward(self, x, target):
        x_spec = self.preprocess(x)
        x_spec = self.linear(x_spec)
        x_enc = self.encoder(x_spec)
        y_hat = self.decoder(x_enc)
        return y_hat
