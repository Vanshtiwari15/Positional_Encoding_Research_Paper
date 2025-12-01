import sys
import os

# Ensure project root is on path (important for Colab)
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from data.load_dataset import load_opus_de_en
from data.preprocess import preprocess_example
from models.positional_encoding import SinusoidalPositionalEncoding
from models.positional_encoding import LearnedPositionalEncoding
from models.positional_encoding import RotaryPositionalEncoding
from models.transformer import TransformerModel
from training.train import train_one_epoch, evaluate_loss


# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIM = 512
BATCH_SIZE = 32
EPOCHS = 3
LR = 3e-4
MAX_LEN = 64

# low-resource: limit training samples
MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 500

def main():
    print("Using device:", DEVICE)

    # ---------- Load dataset ----------
    train_raw, val_raw, test_raw = load_opus_de_en()

    # Low-resource simulation
    train_raw = train_raw.select(
        range(min(MAX_TRAIN_SAMPLES, len(train_raw)))
    )
    val_raw = val_raw.select(
        range(min(MAX_VAL_SAMPLES, len(val_raw)))
    )

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id

    # ---------- Preprocess ----------
    train_processed = [
        preprocess_example(ex, tokenizer, MAX_LEN)
        for ex in train_raw
    ]

    val_processed = [
        preprocess_example(ex, tokenizer, MAX_LEN)
        for ex in val_raw
    ]

    train_loader = DataLoader(
        train_processed,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_processed,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ---------- Model ----------
    # pe = SinusoidalPositionalEncoding(d_model=MODEL_DIM)
    # pe = LearnedPositionalEncoding(d_model=MODEL_DIM)
    pe = RotaryPositionalEncoding(d_model=MODEL_DIM)


    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=MODEL_DIM,
        pe=pe,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # ---------- Training + Validation ----------
    os.makedirs("results/tables", exist_ok=True)

    # Store loss per epoch (for plotting)
    train_losses = []
    val_losses = []


    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
        )

        val_loss = evaluate_loss(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    # Save final losses for this PE type
    final_train_loss = train_loss
    final_val_loss = val_loss

    # with open("results/tables/sinusoidal_loss.txt", "w") as f:
    #     f.write(f"Final Train Loss: {final_train_loss:.4f}\n")
    #     f.write(f"Final Val Loss: {final_val_loss:.4f}\n")

    # print("✅ Baseline (Sinusoidal PE) training + loss evaluation completed")

    # import json
    # os.makedirs("results/logs", exist_ok=True)

    # with open("results/logs/sinusoidal_losses.json", "w") as f:
    #     json.dump(
    #       {
    #           "train": train_losses,
    #           "val": val_losses,
    #       },
    #       f,
    #       indent=2
    #     )

    # with open("results/tables/learned_loss.txt", "w") as f:
    #     f.write(f"Final Train Loss: {final_train_loss:.4f}\n")
    #     f.write(f"Final Val Loss: {final_val_loss:.4f}\n")

    # print("✅ Baseline (Learned PE) training + loss evaluation completed")

    # import json
    # os.makedirs("results/logs", exist_ok=True)

    # with open("results/logs/learned_losses.json", "w") as f:
    #     json.dump(
    #       {
    #           "train": train_losses,
    #           "val": val_losses,
    #       },
    #       f,
    #       indent=2
    #     )
    

    with open("results/tables/rotary_loss.txt", "w") as f:
        f.write(f"Final Train Loss: {final_train_loss:.4f}\n")
        f.write(f"Final Val Loss: {final_val_loss:.4f}\n")

    print("✅ Baseline (Rotary PE) training + loss evaluation completed")

    import json
    os.makedirs("results/logs", exist_ok=True)
    
    with open("results/logs/rotary_losses.json", "w") as f:
        json.dump(
          {
              "train": train_losses,
              "val": val_losses,
          },
          f,
          indent=2
        )


if __name__ == "__main__":
    main()