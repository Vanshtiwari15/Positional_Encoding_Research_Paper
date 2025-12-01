from datasets import load_dataset


def load_opus_de_en():
    """
    Loads OPUS Books (de->en) and creates
    train / validation / test splits manually.
    """
    dataset = load_dataset("opus_books", "de-en")["train"]

    # Shuffle for reproducibility
    dataset = dataset.shuffle(seed=42)

    n = len(dataset)
    train_size = int(0.9 * n)
    val_size = int(0.05 * n)

    train_data = dataset.select(range(0, train_size))
    val_data = dataset.select(range(train_size, train_size + val_size))
    test_data = dataset.select(range(train_size + val_size, n))

    return train_data, val_data, test_data