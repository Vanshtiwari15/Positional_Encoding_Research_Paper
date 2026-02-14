Positional Encoding in Low-Resource Transformer Training

This repository contains the implementation and experiments for the paper:

How Do Positional Encodings Affect Transformer Training in Low-Resource Settings? 

positional_encoding_low_resource

Overview

Transformers require positional encodings to model token order. While most prior studies evaluate positional encoding strategies in large-scale training regimes, this project investigates their behavior under extreme low-resource conditions.

We train a standard encoder–decoder Transformer from scratch on a small German–English dataset while varying only the positional encoding mechanism.

The goal is to study optimization behavior, not final translation quality.

Positional Encoding Variants Compared

Sinusoidal Positional Encoding (fixed, non-trainable)

Learned Absolute Positional Embeddings

Rotary Positional Encoding (RoPE)

All other architectural and training settings are kept identical.

Experimental Setup
Dataset

OPUS Books (German–English)

Restricted to 2,000 sentence pairs

Maximum sequence length: 64 tokens

Marian tokenizer (Helsinki-NLP/opus-mt-de-en)

Model Configuration

Encoder–Decoder Transformer

4 layers (encoder & decoder)

Model dimension: 512

8 attention heads

Feedforward dimension: 2048

Dropout: 0.1

Optimizer: AdamW

Learning rate: 3e-4

Batch size: 32

Training epochs: 3

Evaluation

Cross-entropy loss (training & validation)

BLEU not used due to instability in low-resource regime

Results

Final loss after 3 epochs:

Positional Encoding	Train Loss	Validation Loss
Sinusoidal	5.8132	5.8166
Learned	5.8284	5.8113
Rotary	5.8420	5.8576
Key Findings

Learned positional embeddings show no advantage over sinusoidal encoding.

Rotary encoding converges more slowly and results in higher validation loss.

Under extreme data constraints, simpler positional encodings are sufficient.

Repository Structure
├── main.py
├── plots_losses.py
├── results/
│   ├── logs/
│   ├── plots/
│   └── tables/
├── .gitignore
└── README.md

How to Run
1. Install dependencies
pip install torch transformers matplotlib

2. Train model
python main.py

3. Generate plots
python plots_losses.py


Results will be saved in the results/ directory.

Limitations

Single dataset (German–English)

Very small training size (2,000 pairs)

Only 3 epochs

Evaluation based solely on cross-entropy loss

Conclusion

Under extreme low-resource training, positional encoding complexity does not improve early optimization. Simpler absolute positional encodings are sufficient, and more expressive methods may require larger datasets or longer training.

Citation

If you use this work, please cite:

Vansh Tiwari. How Do Positional Encodings Affect Transformer Training in Low-Resource Settings?
