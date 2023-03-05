from torch import nn

from mstransformer.src.transformer.position import PositionalEncoding


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(DecoderBlock, self).__init__()
        # self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        # encoder-decoder attention
        self.enc_dec_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                  num_heads=num_heads)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        # feed-forward
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        self.norm3 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, tgt, memory, mask):
        tgt = self.dropout1(self.norm1(tgt + self.self_attn(tgt, tgt, tgt, attn_mask=mask)[0]))
        out = self.dropout2(self.norm2(tgt + self.enc_dec_attn(tgt, memory, memory)[0]))
        out = self.dropout3(self.norm3(out + self.feedforward(out)))
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            num_layers=6,
            d_model=512,
            num_heads=8,
            max_len=512,
            dropout=0.0
    ):
        super(Decoder, self).__init__()
        self.positional_encoding = PositionalEncoding(max_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, target, memory, mask):
        tgt = self.positional_encoding(target)
        for block in self.blocks:
            tgt = block(tgt, memory, mask)
        return tgt
