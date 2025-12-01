Title:
How do different positional encodings affect Transformer performance when training data is limited?

Goal:
To study how sinusoidal, learned, and relative positional encodings influence
 learning stability and performance of Transformers under limited training data.

Method:
Controlled experiments where only positional encoding is changed.

Task:
Machine Translation (English → German)

Dataset:
IWSLT En-De (low-resource setting)


Research Question:
How do different positional encodings affect Transformer performance
when training data is limited?

Dataset:
IWSLT En-De

Task:
Machine Translation

Positional Encodings:
1. Sinusoidal
2. Learned
3. Relative

Hypothesis:
With limited data, fixed or relative positional encodings
generalize better than learned positional encodings.