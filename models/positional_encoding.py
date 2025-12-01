import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements fixed sinusoidal positional encoding
    as in 'Attention Is All You Need'.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class LearnedPositionalEncoding(nn.Module):
    """
    Learned (trainable) positional embeddings.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(
            0, seq_len, device=x.device
        ).unsqueeze(0).expand(batch_size, seq_len)

        pos_emb = self.position_embeddings(positions)
        return x + pos_emb

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE).
    Applies rotary transform to last dimension.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        )

        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        self.register_buffer("cos", torch.cos(freqs))
        self.register_buffer("sin", torch.sin(freqs))

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)

        cos = self.cos[:seq_len].unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # rotary transformation
        x_rotated = torch.cat(
            [x1 * cos - x2 * sin,
             x1 * sin + x2 * cos],
            dim=-1,
        )

        return x_rotated
