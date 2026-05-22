import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding           = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer         = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=2048, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids):
        src_pad_mask = (src_ids == 0)
        tgt_pad_mask = (tgt_ids == 0)
        src = self.positional_encoding(self.embedding(src_ids))
        tgt = self.positional_encoding(self.embedding(tgt_ids))
        T   = tgt.size(1)
        tgt_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=src.device), diagonal=1
        )
        out = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        return self.fc_out(out)