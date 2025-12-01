import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=2048,
        pe=None,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding module (e.g., SinusoidalPositionalEncoding)
        self.positional_encoding = pe

        # Core Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, dim)
        )

        # Output projection to vocab
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (batch, src_seq_len)
        tgt: (batch, tgt_seq_len)
        """

        # Token embeddings + scale
        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        tgt_emb = self.tgt_embedding(tgt) * (self.d_model ** 0.5)

        # Positional encoding
        if self.positional_encoding is not None:
            src_emb = self.positional_encoding(src_emb)
            tgt_emb = self.positional_encoding(tgt_emb)

        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )

        # (batch, tgt_seq_len, vocab_size)
        logits = self.fc_out(out)
        return logits