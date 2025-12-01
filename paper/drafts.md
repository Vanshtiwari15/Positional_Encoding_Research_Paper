Intro idea:
Transformers need position info.
Most studies use big data.
We test low data.

Method idea:
Fix everything.
Change PE only.

Expected result:
Learned PE overfits.
Relative PE better.

Transformers rely on positional encodings to capture word order information,
as self-attention alone is permutation-invariant. While several positional
encoding methods have been proposed, most prior work evaluates them under
large-scale training regimes. In practical scenarios, especially for students
and low-resource languages, training data is often limited. 

In this work, we study how different positional encodings affect Transformer
performance when training data is limited. We conduct a controlled comparison
of sinusoidal, learned, and relative positional encodings, keeping all other
factors fixed.


Methodology:
We use a standard encoder–decoder Transformer architecture.
To ensure a fair comparison, all experiments use the same model
size, tokenizer, optimizer, and training schedule. Positional
encoding is the only component that differs across models.
