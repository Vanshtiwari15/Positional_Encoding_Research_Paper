import sys
print(sys.path)
import torch
from models.positional_encoding import SinusoidalPositionalEncoding
from models.transformer import TransformerModel

vocab_size = 1000
batch = 2
seq_len = 10

pe = SinusoidalPositionalEncoding(d_model=512)
model = TransformerModel(
    vocab_size=vocab_size,
    pe=pe
)

src = torch.randint(0, vocab_size, (batch, seq_len))
tgt = torch.randint(0, vocab_size, (batch, seq_len))

out = model(src, tgt)

print("Output shape:", out.shape)