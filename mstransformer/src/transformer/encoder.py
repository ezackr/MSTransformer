from torch import nn

from mstransformer.src.transformer.position import PositionalEncoding


class EncoderBlock(nn.Module):
    """
    A single encoder block for the transformer encoder. Includes a
    MultiHeadAttention layer followed by a FeedForward layer.

    Parameters:
        - d_model (int): dimension of input.
        - num_heads (int): number of attention heads.
        - dropout (float): dropout probability.
    """
    def __init__(self, d_model, num_heads=8, dropout=0.0):
        super(EncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads,
                                               dropout=dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 8),
            nn.ReLU(),
            nn.Linear(d_model * 8, d_model)
        )
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x_attn = self.norm1(x + self.dropout1(self.attention(x, x, x)))
        x_out = self.norm2(x_attn + self.dropout2(self.feedforward(x_attn)))
        return x_out


class Encoder(nn.Module):
    """
    Lays separate EncoderBlocks in sequence to create the encoder.

    Parameters:
        - num_layers (int): number of encoder blocks.
        - num_channels (int): dimension of input.
        - num_heads (int): number of attention heads.
        - max_len (int): length of the input sequence.
        - dropout (float): dropout probability.
    """
    def __init__(
            self,
            num_layers=6,
            num_channels=2,
            num_heads=8,
            max_len=220500,
            dropout=0.0
    ):
        super(Encoder, self).__init__()
        self.positional_encoding = PositionalEncoding(max_len, num_channels)
        self.blocks = [
            EncoderBlock(num_channels, num_heads, dropout)
            for _ in range(num_layers)
        ]

    def forward(self, x):
        x_enc = self.positional_encoding(x)
        for block in self.blocks:
            x_enc = block(x_enc)
        return x_enc
