# Stanford CS336: Language Modeling from Scratch | Spring 2026

https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV

## Index

- [01 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 1: Overview, Tokenization](#01---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-1-overview-tokenization)
- [02 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 2: PyTorch (einops)](#02---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-2-pytorch-einops)
- [03 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 3: Architectures](#03---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-3-architectures)
- [04 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 4: Attention Alternatives](#04---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-4-attention-alternatives)
- [05 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 5: GPUs, TPUs](#05---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-5-gpus-tpus)
- [06 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 6: Kernels, Triton, XLA](#06---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-6-kernels-triton-xla)
- [07 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 7: Parallelism](#07---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-7-parallelism)
- [08 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 8: Parallelism](#08---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-8-parallelism)
- [09 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 9: Scaling Laws](#09---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-9-scaling-laws)
- [10 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 10: Inference](#10---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-10-inference)
- [11 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 11:  Scaling Laws](#11---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-11--scaling-laws)
- [12 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 12: Evaluation](#12---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-12-evaluation)
- [13 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 13: Data (Sources, Datasets)](#13---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-13-data-sources-datasets)
- [14 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 14: Data](#14---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-14-data)
- [15 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 15: Mid/Post-Training](#15---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-15-midpost-training)
- [16 - Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 16: Post-Training - RLVR](#16---stanford-cs336-language-modeling-from-scratch--spring-2026--lecture-16-post-training---rlvr)

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
  - attention (expensive $\mathcal{O}(n^2)$ )
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
- target: minimize data movement

### parallelism
- sharding
- splits

### inference
- RL-HF
- eval
- phases: prefill -> decode
  - prefill: feed tokens to fill kv cache
  - decode: generate 1 token at a time (memory bound)
- faster: small model, speculative decoding, fused kernel, continuous batching

### scaling laws
- what to focus on?
- can't do hyperparameter tuning due to scale
- scaling recipe (FLOPs -> hyperparameter budget)
- D (model params) = 20N (token count)

### data
- better data = better model
- metrics
  - perplexity (private documents, not on the internet, avoids contamination)
  - benchmark (mostly public)

### data curation
- scraping + crawling

### data processing
- transformation
- filtering
- deduplication
- weighted data
- synthetic data
- types: pre-training, mid-training, post-training (SFT).

### alignment
- generate response then grade
- RL algo: PPO, GRPO, DPO (preference data)
  - unstable, hard to tune
- lots of infra requirements

---

### tokenization
- encode string -> token, decode token -> string
- [tokenization demo](https://tiktokenizer.vercel.app/?encoder=gpt2)
- compression ratio (high better, attention is quadratic but higher leads to sparsity)
- types
  - character level (massive vocab size)
  - byte level (compression ratio = 1, vocab size is small)
  - word level (compression ratio is better, vocab is unbounded)
  - byte pair level (train tokenizer, common sequences have shorter tokens, byte pair and merge recursively)
  - unk token messes up perplexity

## 02 - [Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 2: PyTorch (einops)](https://www.youtube.com/watch?v=kuYAsz7zspQ&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

### resource accounting

#### tensors
- building blocks
- FP32 (1 sign, 8 exp, 23 fraction)
  - IEEE 754
  - "single precision"
  - FP32 is still heavy for ML
  - torch default is float 32
  - 4 bytes per tensor
- FP16 (1 sign, 5 exp, 10 fraction)
  - low dynamic range
  - overflow and underflow!
- bfloat16 (1 sign, 8 exp, 7 fration)
  - brain floating point (2018)
  - same dynamic range as FP32 but less precise
  - 2 bytes per tensor
- fp32 works but requires lots of memory
- fp8, fp16 and even bfloat16 is risky (instability)
- mixed precision training
  - bf16 for parameters, activation, gradients
  - fp32 for optimizer states (?)
  - pytorch AMP library (automatically casts when safe)
- fp8
  - E4M3 (higher dynamic range)
  - E5M2 (higher precision)
- fp4
  - Nemotron 3 used this (NVFP4)
  - blocks can be scaled up and down to increase range

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
