# DATA266 Lab 1 — Task 1 Results

## Student

Praveen

## Model

Character-level GPT-style decoder-only language model implemented from scratch in PyTorch.

## Architecture

- Vocabulary size: 118
- Context length: 128
- Embedding dimension: 192
- Attention heads: 6
- Dimension per head: 32
- Transformer blocks: 3
- FFN expansion multiplier: 3
- FFN hidden dimension: 576
- Dropout: 0.15
- Fused Q/K/V projection
- Learnable token embeddings
- Learnable positional embeddings
- Pre-LayerNorm Transformer blocks
- GELU feed-forward network
- Residual connections
- Linear language-model head
- Parameter count: 1,183,104

No nn.MultiheadAttention, nn.Transformer, or pretrained Transformer model was used.

## Architecture Rationale

A 128-character context was chosen to reduce the quadratic cost of self-attention while still providing enough short-range context for TinyStories.

The 192-dimensional embedding is divided across 6 attention heads, giving 32 dimensions per head.

Three Transformer blocks and a 3x feed-forward expansion were selected to create a smaller and computationally efficient model while still retaining multiple stages of self-attention and nonlinear processing.

The fused Q/K/V projection provides a different attention implementation from the teammate model while still computing scaled dot-product multi-head causal self-attention manually.

## Dataset

Dataset: TinyStories

Independent deterministic split:

- Training stories: 100,000
- Validation stories: 10,000
- Training characters: 89,954,584
- Validation characters: 8,971,053
- Training character vocabulary: 117
- Final vocabulary including <UNK>: 118
- Validation-only unseen characters: 2

## Training

- Epochs: 10
- Batch size: 192
- Optimizer: AdamW
- Initial learning rate: 0.0004
- Minimum learning rate: 4e-05
- Warm-up steps: 400
- Scheduler: linear warm-up followed by cosine decay
- Weight decay: 0.01
- Gradient clipping: 1.0
- Mixed precision: BF16
- Seed: 2026

## Final Evaluation

| Metric | Result |
|---|---:|
| Training CE loss | 0.7603 |
| Validation CE loss | 0.7626 |
| Generalization gap | 0.0022 |
| Validation perplexity | 2.1438 |
| Bits per character | 1.1001 |
| Train Top-1 accuracy | 75.93% |
| Validation Top-1 accuracy | 75.89% |
| Character Distinct-1 | 0.0093 |
| Character Distinct-2 | 0.0820 |
| Character Distinct-3 | 0.2463 |
| Character repeated 4-gram rate | 0.5765 |
| Word Distinct-1 | 0.2838 |
| Word Distinct-2 | 0.7414 |
| Word Distinct-3 | 0.9069 |
| Word repeated 4-gram rate | 0.0304 |
| Parameter count | 1,183,104 |
| Average training throughput | 2,237,338 tokens/sec |
| Average generation throughput | 702.83 tokens/sec |
| Peak GPU memory | 1.00 GB |
| Total training time | 6.94 min |
| NaN count | 0 |

## Generation

Both greedy decoding and temperature-based sampling were evaluated.

Greedy decoding generated more deterministic but substantially more repetitive outputs.

Temperature sampling generated more diverse text, but several semantic and world-logic inconsistencies appeared.

Generated samples:

`outputs/generated_samples.csv`

## Failure Analysis

Three representative failure cases are documented in:

`failure_analysis.md`

## Hardware

GPU: NVIDIA GeForce RTX 5090
