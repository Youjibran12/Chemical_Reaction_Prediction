class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src_ids):
        embedded = self.embedding(src_ids)
        outputs, hidden = self.gru(embedded)
        fwd = hidden[0::2]
        bwd = hidden[1::2]
        hidden = torch.tanh(self.fc_hidden(torch.cat([fwd, bwd], dim=2)))
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_hidden_dim * 2, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_hidden_dim,     attn_dim, bias=False)
        self.v     = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden, src_mask=None):
        energy = torch.tanh(
            self.W_enc(encoder_outputs) +
            self.W_dec(decoder_hidden).unsqueeze(1)
        )
        scores = self.v(energy).squeeze(-1)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim,
                 num_layers, dropout, attn_dim, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim, attn_dim)
        self.gru = nn.GRU(
            embed_dim + enc_hidden_dim * 2, dec_hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out  = nn.Linear(dec_hidden_dim + enc_hidden_dim * 2 + embed_dim, vocab_size)

    def forward_train(self, tgt_ids, hidden, encoder_outputs, src_mask=None):
        """Step-by-step Bahdanau attention — identical to inference path."""
        B, tgt_len = tgt_ids.shape
        embedded   = self.dropout(self.embedding(tgt_ids))
        outputs    = []
        for t in range(tgt_len):
            context, _ = self.attention(encoder_outputs, hidden[-1], src_mask)
            gru_input  = torch.cat([embedded[:, t:t+1, :], context.unsqueeze(1)], dim=2)
            dec_out, hidden = self.gru(gru_input, hidden)
            emb_t  = embedded[:, t, :]
            logit  = self.fc_out(torch.cat([dec_out.squeeze(1), context, emb_t], dim=1))
            outputs.append(logit)
        return torch.stack(outputs, dim=1), hidden

    def forward_step(self, tgt_token, hidden, encoder_outputs, src_mask=None):
        """Single-step inference."""
        embedded         = self.dropout(self.embedding(tgt_token.unsqueeze(1)))
        context, weights = self.attention(encoder_outputs, hidden[-1], src_mask)
        gru_input        = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        dec_output, hidden = self.gru(gru_input, hidden)
        dec_output       = dec_output.squeeze(1)
        embedded         = embedded.squeeze(1)
        logits = self.fc_out(torch.cat([dec_output, context, embedded], dim=1))
        return logits, hidden, weights


class GRUSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=2,
                 dropout=0.1, attn_dim=128, padding_idx=0):
        super().__init__()
        self.encoder     = GRUEncoder(vocab_size, embed_dim, hidden_dim,
                                      num_layers, dropout, padding_idx)
        self.decoder     = GRUDecoder(vocab_size, embed_dim, hidden_dim, hidden_dim,
                                      num_layers, dropout, attn_dim, padding_idx)
        self.padding_idx = padding_idx

    def forward(self, src_ids, tgt_ids):
        src_mask = (src_ids == self.padding_idx)
        encoder_outputs, hidden = self.encoder(src_ids)
        logits, _ = self.decoder.forward_train(tgt_ids[:, :-1], hidden, encoder_outputs, src_mask)
        return logits

    def encode(self, src_ids):
        return self.encoder(src_ids)

    def decode_step(self, token, hidden, encoder_outputs, src_mask=None):
        return self.decoder.forward_step(token, hidden, encoder_outputs, src_mask)