import sys
import os

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.load_dataset import load_opus_de_en
from data.preprocess import preprocess_example
from models.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
)
from models.transformer import TransformerModel
from training.train import train_one_epoch, evaluate_loss


# CONFIG 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIM = 512
BATCH_SIZE = 32
EPOCHS = 3
LR = 3e-4
MAX_LEN = 64

# low-resource: limit training samples
MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 500

DATASET_NAME = "de-en"   # change to dataset2 / dataset3 if needed


def main():
    print("Using device:", DEVICE)

    #Load dataset
    train_raw, val_raw, test_raw = load_opus_de_en()

    train_raw = train_raw.select(
        range(min(MAX_TRAIN_SAMPLES, len(train_raw)))
    )
    val_raw = val_raw.select(
        range(min(MAX_VAL_SAMPLES, len(val_raw)))
    )

    #Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id

    #Preprocess
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

    #Positional Encoding Variants
    pe_variants = [
        ("sinusoidal", SinusoidalPositionalEncoding),
        ("learned", LearnedPositionalEncoding),
        ("rotary", RotaryPositionalEncoding),
    ]

    #Run Experiments
    for pe_name, pe_class in pe_variants:

        print(f"\n===== Running {pe_name.upper()} Positional Encoding =====")

        pe = pe_class(d_model=MODEL_DIM)

        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=MODEL_DIM,
            pe=pe,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

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

        # Save Results

        save_dir = f"results/{DATASET_NAME}/{pe_name}"
        os.makedirs(save_dir, exist_ok=True)

        # Save losses
        import json
        with open(f"{save_dir}/losses.json", "w") as f:
            json.dump(
                {
                    "train": train_losses,
                    "val": val_losses,
                },
                f,
                indent=2
            )

        # Save model checkpoint
        torch.save(
            model.state_dict(),
            f"{save_dir}/model.pth"
        )

        print(f"Saved results and model in {save_dir}")


if __name__ == "__main__":
    main()