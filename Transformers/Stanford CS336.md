# Stanford CS336: Language Modeling from Scratch | Spring 2026

https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV

## 01 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 1: Overview, Tokenization](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

### pre NNs
- entropy based
- n gram models

### NNs
- LSTMs
- Sequence to sequence models
- Attention
- Transformers
- MoE

### foundational models
- ELMo, BERT: pre-training + fine tune
- T5: text to text

### scaling
- GPT2 -> GPT3 (> 10x size increase)
- Meta OPT: hw issues
- Open weight models: leading with Llama series

### progress
- 2018: fine tune (bert)
- 2020: prompt (gpt3)
- 2022: talk (chatgpt)
- 2026: agents (autonomous)

### tokenization
- BPE

### model arch
- originally transformer
- improvements
  - activation functions
  - positional encoding
  - normalization
  - attention (expensive $\mathcal{O}(n^2)$)
  - linear attention (?)
  - MLP with MoE
  - architecture

### training
- loss function (next v/s multi)
- optimizer
- initializer
- lr schedule
- batch size
- MoE load balancing

### interesting stuff
- roofline analysis

### kernel
- function running on gpu
### - target: minimize data movement

## 02 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 2: PyTorch (einops)](https://www.youtube.com/watch?v=kuYAsz7zspQ&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 03 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 3: Architectures](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 04 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 4: Attention Alternatives](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 05 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 5: GPUs, TPUs](https://www.youtube.com/watch?v=izZba4UA7iY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 06 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 6: Kernels, Triton, XLA](https://www.youtube.com/watch?v=xnDHaNUvHBg&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 07 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 7: Parallelism](https://www.youtube.com/watch?v=SzpOcwdIL0Y&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 08 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 8: Parallelism](https://www.youtube.com/watch?v=6-cXp-aOmdg&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 09 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 9: Scaling Laws](https://www.youtube.com/watch?v=Q15rhEWZPQ4&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 10 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 10: Inference](https://www.youtube.com/watch?v=EfM546A79aM&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 11 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 11:  Scaling Laws](https://www.youtube.com/watch?v=vTfEyOyzV9E&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 12 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 12: Evaluation](https://www.youtube.com/watch?v=JpAxdTWQJxM&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 13 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 13: Data (Sources, Datasets)](https://www.youtube.com/watch?v=-qm0ln33G24&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 14 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 14: Data](https://www.youtube.com/watch?v=5sxHosTLPF8&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 15 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 15: Mid/Post-Training](https://www.youtube.com/watch?v=2oH6PWPrYFo&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 16 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 16: Post-Training - RLVR](https://www.youtube.com/watch?v=dIFAi87Ws4E&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
