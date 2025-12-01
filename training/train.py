import torch
from torch.optim import AdamW
from tqdm import tqdm


def generate_square_subsequent_mask(sz, device):
    """
    Causal mask so decoder cannot see future tokens.
    Size: (sz, sz)
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask.to(device)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):
    model.train()
    total_loss = 0.0
    count = 0

    for batch in tqdm(dataloader, desc="Training"):
        src = batch["src"].to(device)  # (batch, src_len)
        tgt = batch["tgt"].to(device)  # (batch, tgt_len)

        # Teacher forcing: shift target
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_mask = generate_square_subsequent_mask(
            tgt_input.size(1), device
        )

        optimizer.zero_grad()

        logits = model(
            src=src,
            tgt=tgt_input,
            tgt_mask=tgt_mask,
        )  # (batch, seq_len, vocab)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


def evaluate_loss(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Computes average loss over a validation set.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)