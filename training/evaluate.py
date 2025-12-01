import torch
import sacrebleu
from tqdm import tqdm


def greedy_decode(
    model,
    src,
    max_len,
    sos_id,
    eos_id,
    device
):
    model.eval()
    src = src.to(device)

    batch_size = src.size(0)

    ys = torch.full(
        (batch_size, 1),
        sos_id,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_mask = torch.triu(
                torch.ones(ys.size(1), ys.size(1), device=device),
                diagonal=1
            ).bool()

            out = model(src, ys, tgt_mask=tgt_mask)
            next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)

            ys = torch.cat([ys, next_token], dim=1)

    # cut after EOS
    results = []
    for seq in ys:
        seq = seq.tolist()
        if eos_id in seq:
            seq = seq[: seq.index(eos_id)]
        results.append(seq)

    return results


def evaluate_bleu(
    model,
    dataset,
    tokenizer,
    device,
    max_len=64,
    max_samples=200
):
    predictions = []
    references = []

    sos_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    for i, example in enumerate(tqdm(dataset, desc="Evaluating BLEU")):
        if i >= max_samples:
            break

        src_ids = tokenizer(
          example["translation"]["de"],
          return_tensors="pt",
          truncation=True,
          max_length=max_len,
        )["input_ids"]


        pred_tokens = greedy_decode(
            model,
            src_ids,
            max_len,
            sos_id,
            eos_id,
            device,
        )[0]

        pred_text = tokenizer.decode(
            pred_tokens, skip_special_tokens=True
        )
        ref_text = example["translation"]["en"]

        predictions.append(pred_text)
        references.append(ref_text)

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score