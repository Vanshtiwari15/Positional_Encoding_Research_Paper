def preprocess_example(example, tokenizer, max_len=64):
    """
    Preprocess a single OPUS Books example (de -> en).
    Returns: dict with 'src' and 'tgt' tensors (1D each).
    """
    # source: German, target: English
    src_text = example["translation"]["de"]
    tgt_text = example["translation"]["en"]

    src_enc = tokenizer(
        src_text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    tgt_enc = tokenizer(
        tgt_text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    # tokenizer returns shape (1, seq_len) → squeeze to (seq_len,)
    return {
        "src": src_enc["input_ids"].squeeze(0),
        "tgt": tgt_enc["input_ids"].squeeze(0),
    }