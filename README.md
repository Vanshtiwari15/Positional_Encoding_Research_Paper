# 📐 Positional Encoding in Low-Resource Transformer Training

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![Transformer](https://img.shields.io/badge/Model-Encoder--Decoder-orange.svg)]()
[![Research](https://img.shields.io/badge/Type-Empirical%20Study-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> A controlled empirical study analyzing how different positional encoding strategies affect Transformer optimization under extreme low-resource training conditions.

---

## 🌟 Overview

Transformers require positional encodings to model token order. Most research evaluates these mechanisms in large-scale settings.  

This project studies their behavior under **extreme low-resource conditions (2,000 training examples)**.

We implement a standard encoder–decoder Transformer from scratch and compare:

- Sinusoidal Positional Encoding
- Learned Absolute Positional Embeddings
- Rotary Positional Encoding (RoPE)

All experiments keep architecture and hyperparameters identical.

---

## 🔬 Research Question

Does increasing positional encoding complexity improve optimization when data is severely limited?

---

## 🧠 Experimental Setup

### 📚 Dataset
- OPUS Books (German–English)
- 2,000 sentence pairs
- Max sequence length: 64
- Marian tokenizer (Helsinki-NLP/opus-mt-de-en)

### ⚙️ Model Configuration
- Encoder–Decoder Transformer
- 4 layers (encoder & decoder)
- Model dimension: 512
- 8 attention heads
- Feedforward dimension: 2048
- Dropout: 0.1
- Optimizer: AdamW
- Learning rate: 3e-4
- Batch size: 32
- Epochs: 3

### 📊 Evaluation
- Training Cross-Entropy Loss
- Validation Cross-Entropy Loss
- BLEU not used due to instability in low-resource regime

---

## 📈 Results

| Positional Encoding | Train Loss | Validation Loss |
|---------------------|------------|-----------------|
| Sinusoidal          | 5.8132     | 5.8166          |
| Learned             | 5.8284     | 5.8113          |
| Rotary              | 5.8420     | 5.8576          |

### Key Findings

- Learned embeddings show no advantage over sinusoidal encoding.
- Rotary encoding converges more slowly in low-resource settings.
- Simpler encodings are sufficient under extreme data constraints.

---

## 🏗️ Architecture

The model follows a standard encoder–decoder Transformer architecture.

### 🔹 Input Processing
Input Sentence (German)
↓
Tokenization
↓
Token Embeddings
↓

Positional Encoding (Sinusoidal / Learned / Rotary)


Maximum sequence length (context window): **64 tokens**

All tokens within this context window attend to each other via self-attention.

---

### 🔹 Encoder
[ Self-Attention ]
↓
[ Add & LayerNorm ]
↓
[ Feedforward Network ]
↓
[ Add & LayerNorm ]


- 4 stacked encoder layers  
- Model dimension: 512  
- 8 attention heads  
- Full attention within the 64-token context window  

Each token attends to all other tokens in the input sequence.

---

### 🔹 Decoder
Target Tokens (shifted right)
↓
Masked Self-Attention (causal)
↓
Cross-Attention (attends to encoder outputs)
↓
Feedforward Network


- 4 stacked decoder layers  
- Causal masking ensures tokens only attend to previous positions  
- Cross-attention connects decoder to encoder representations  

---

### 🔹 Context Window Behavior

- Maximum sequence length: **64**
- Attention complexity: **O(n²)** within the window
- No sparse or sliding-window attention
- Rotary encoding (when used) modifies query/key vectors inside attention

---

### 🔹 Output



Decoder Hidden States
↓
Linear Projection
↓
Softmax
↓
Next-Token Probability Distribution


Training objective: **Cross-Entropy Loss**

---

### 🔬 Positional Encoding Injection

- Sinusoidal / Learned → added to token embeddings
- Rotary → applied directly inside attention mechanism

All other architectural components remain identical across experiments.
