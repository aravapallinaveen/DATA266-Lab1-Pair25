# DATA266 Lab 1 — Task 1 Results

## Model

GPT-style decoder-only character language model implemented from scratch in PyTorch.

### Architecture

- Character vocabulary size: 110
- Context length: 256
- Embedding dimension: 256
- Attention heads: 8
- Dimension per head: 32
- Transformer blocks: 4
- Feed-forward hidden dimension: 1024
- Dropout: 0.1
- Learnable token embeddings
- Learnable positional embeddings
- Pre-LayerNorm Transformer blocks
- Causal multi-head self-attention implemented manually
- Residual connections
- GELU feed-forward networks
- Linear language-model head
- Parameters: 3,278,336

No nn.MultiheadAttention, nn.Transformer, or pretrained Transformer model was used.

## Architecture Rationale

A context length of 256 characters was selected to provide multiple sentences of context while keeping the quadratic computational cost of self-attention manageable.

The 256-dimensional embedding is divided across 8 attention heads, giving 32 dimensions per head. Four Transformer blocks provide multiple stages of attention and nonlinear transformation while keeping the model small enough for efficient experimentation.

## Dataset

Dataset: TinyStories

Individual deterministic split:

- Training stories: 100,000
- Validation stories: 10,000
- Training characters: 89,733,070
- Validation characters: 8,954,303
- Final vocabulary size: 110
- Validation-only unseen characters: 3

Validation-only unseen characters were mapped to a reserved <UNK> token.

## Training Configuration

- Epochs: 10
- Batch size: 128
- Optimizer: AdamW
- Initial learning rate: 0.0003
- Minimum learning rate: 3e-05
- Warm-up steps: 500
- Scheduler: linear warm-up followed by cosine decay
- Weight decay: 0.01
- Gradient clipping: 1.0
- Mixed precision: BF16
- Seed: 171

## Final Evaluation

| Metric | Result |
|---|---:|
| Training cross-entropy loss | 0.6190 |
| Validation cross-entropy loss | 0.6285 |
| Generalization gap | 0.0095 |
| Validation perplexity | 1.8748 |
| Bits per character | 0.9067 |
| Train Top-1 accuracy | 80.22% |
| Validation Top-1 accuracy | 79.98% |
| Character Distinct-1 | 0.0097 |
| Character Distinct-2 | 0.0824 |
| Character Distinct-3 | 0.2340 |
| Character repeated 4-gram rate | 0.6072 |
| Word Distinct-1 | 0.2449 |
| Word Distinct-2 | 0.6848 |
| Word Distinct-3 | 0.8685 |
| Word repeated 4-gram rate | 0.0712 |
| Parameter count | 3,278,336 |
| Average training throughput | 909,419 tokens/sec |
| Average generation throughput | 404.56 tokens/sec |
| Peak GPU memory | 3.57 GB |
| Training time | 17.00 min |
| NaN count | 0 |

## Training Stability

Training and validation loss decreased across all 10 epochs. No NaN values were observed during training. Gradient norms remained finite throughout the run.

## Generation

Both greedy decoding and temperature sampling were evaluated.

Greedy decoding produced more deterministic and template-like outputs. Temperature sampling produced greater variety but occasionally reduced semantic coherence.

Generated examples are stored in:

`outputs/generated_samples.csv`

The detailed three-case analysis is stored in:

`failure_analysis.md`

## Hardware

- GPU: NVIDIA GeForce RTX 5090
- Peak allocated GPU memory: 3.57 GB
