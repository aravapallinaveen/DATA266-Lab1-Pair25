# Task 2 Results — Yelp Polarity Sentiment Classification

**Member:** Naveen Aravapalli  
**Dataset:** `fancyzhx/yelp_polarity`  
**Seed:** 171  
**Hardware:** AMD Ryzen 9 7950X 16-Core Processor; NVIDIA GeForce RTX 4090 (23.99 GB)  
**Software:** Python 3.12.14, PyTorch 2.14.0+cu130, CUDA runtime 13.0

## Data and Preprocessing

The official Yelp Polarity training split was deterministically divided into 504,000 training and 56,000 validation examples; the official 38,000-example test split was kept unchanged. All three splits were exactly class-balanced.

Text was lowercased, newline artifacts were normalized, common contractions were expanded, tokens containing digits were removed, non-alphabetic/special characters were removed, and stopwords were filtered while preserving sentiment-critical negations. The vocabulary was learned from the training split only.

Final preprocessing settings:

| Setting | Value |
|---|---:|
| MAX_LEN | 256 |
| MIN_FREQ | 5 |
| Vocabulary size | 60,713 |
| Train / validation / test | 504,000 / 56,000 / 38,000 |

Reviews that became empty after preprocessing were represented by one `<UNK>` token so no examples were removed.

## Model Lineup

### Baseline C — Single-Layer GRU
- Learned embedding from scratch: 64 dimensions
- One unidirectional GRU, hidden size 64
- Dropout: 0.2
- Batch size: 256
- Learning rate: 1e-3
- Epochs: 3
- Parameters: 3,910,657

### Experimental E — Bidirectional GRU
- Learned embedding from scratch: 128 dimensions
- One bidirectional GRU, hidden size 128
- Dropout: 0.3
- Batch size: 256
- Learning rate: 5e-4
- Epochs: 5
- Parameters: 7,969,665

### Experimental I — Two-Layer BiGRU + MeanMax Pooling
- Learned embedding from scratch: 192 dimensions
- Two-layer bidirectional GRU, hidden size 192
- Mean + max sequence pooling followed by an MLP head
- Dropout: 0.4
- Batch size: 512
- Learning rate: 3e-4
- Epochs: 8
- Parameters: 12,915,265

No pretrained embeddings or pretrained language models were used.

## Test-Set Metrics

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline GRU | 0.94276 | 0.94276 | 0.98660 | 0.98696 | 0.88557 | 0.04345 | 0.01146 |
| Bidirectional GRU | 0.94461 | 0.94460 | 0.98642 | 0.98614 | 0.88929 | 0.04274 | 0.01895 |
| 2-Layer BiGRU + MeanMax | 0.94613 | 0.94613 | 0.98814 | 0.98852 | 0.89229 | 0.04172 | 0.02112 |

Macro, micro, and weighted precision/recall/F1 are stored in `metrics_report.csv` and `outputs/core_test_metrics.csv`.

## Confusion Matrices

| Model | TN | FP | FN | TP | Total Errors |
|---|---:|---:|---:|---:|---:|
| Baseline GRU | 17,818 | 1,182 | 993 | 18,007 | 2,175 |
| Bidirectional GRU | 17,822 | 1,178 | 927 | 18,073 | 2,105 |
| 2-Layer BiGRU + MeanMax | 18,045 | 955 | 1,092 | 17,908 | 2,047 |

## 95% Bootstrap Confidence Intervals

| Model | Accuracy 95% CI | Macro-F1 95% CI | MCC 95% CI |
|---|---|---|---|
| Baseline GRU | [0.94029, 0.94524] | [0.94029, 0.94523] | [0.88061, 0.89052] |
| Bidirectional GRU | [0.94242, 0.94713] | [0.94242, 0.94713] | [0.88494, 0.89434] |
| 2-Layer BiGRU + MeanMax | [0.94397, 0.94850] | [0.94396, 0.94850] | [0.88796, 0.89701] |

Bootstrap confidence intervals used 1,000 test-set resamples with seed 171.

## Paired McNemar Tests

| Comparison | Baseline correct / comparison wrong | Baseline wrong / comparison correct | Discordant pairs | Exact p-value |
|---|---:|---:|---:|---:|
| Baseline vs Bidirectional GRU | 564 | 634 | 1,198 | 0.046159 |
| Baseline vs 2-Layer BiGRU + MeanMax | 553 | 681 | 1,234 | 0.000297 |

Both exact p-values are below 0.05; the saved test predictions and McNemar table provide the reproducible evidence for these paired comparisons.

## Robustness by Data Slice

### Macro-F1

| Slice | Baseline GRU | Bidirectional GRU | 2-Layer BiGRU + MeanMax |
|---|---:|---:|---:|
| Short (0–64 tokens) | 0.94461 | 0.94668 | 0.94734 |
| Medium (65–128) | 0.93748 | 0.94029 | 0.94239 |
| Long (129+) | 0.93451 | 0.93155 | 0.93858 |
| Has negation | 0.93853 | 0.94148 | 0.94318 |
| No negation | 0.93156 | 0.93017 | 0.93188 |

### Error Rate

| Slice | Baseline GRU | Bidirectional GRU | 2-Layer BiGRU + MeanMax |
|---|---:|---:|---:|
| Short (0–64 tokens) | 0.05502 | 0.05293 | 0.05237 |
| Medium (65–128) | 0.06189 | 0.05913 | 0.05692 |
| Long (129+) | 0.06098 | 0.06314 | 0.05664 |
| Has negation | 0.05956 | 0.05673 | 0.05487 |
| No negation | 0.05079 | 0.05168 | 0.05108 |

The 2-layer BiGRU + MeanMax model produced the highest overall test accuracy, macro-F1, ROC-AUC, PR-AUC, and MCC among the three runs. It also had the lowest error rate on the short, medium, long, and negation slices. The baseline remained substantially more efficient and had the lowest ECE of the three models.

## Training Efficiency

| Model | Best Epoch | Training Time | Train Examples/s | Peak GPU Memory |
|---|---:|---:|---:|---:|
| Baseline GRU | 3 | 2.83 min | 9,302.2 | 0.14 GB |
| Bidirectional GRU | 5 | 8.42 min | 5,195.1 | 0.39 GB |
| 2-Layer BiGRU + MeanMax | 7 | 26.80 min | 2,566.1 | 1.43 GB |

The larger recurrent models increased parameter count and training cost. The baseline provided the fastest training and inference, while the deeper stacked model used the most computation and memory.

## Manual Error Review

Twenty errors from the 2-layer BiGRU + MeanMax test predictions were manually reviewed:

- 5 confident false positives
- 5 confident false negatives
- 5 near-threshold errors
- 5 slice-specific failures

The completed worksheet is stored in `outputs/manual_20_error_review_completed.csv`, and the detailed case-by-case summary is in `failure_analysis.md`.

Recurring cases in the completed review include mixed/aspect-conflicting sentiment, temporal EDIT/UPDATE reversals, target/entity confusion, domain-specific wording, complex negation, long narrative dilution, explicit rating/recommendation overrides, and reviewer-language effects.

## Reproducibility Note

The official Experimental I run completed all 8 epochs and preserved its best/last checkpoints plus the complete 8-row training history. After completion, the training cell was accidentally launched a second time, which opened the same raw-log filename in write mode. The accidental rerun was interrupted before completing epoch 1. The surviving official checkpoints and history were verified, and the event is documented transparently in `experimental_stacked_bigru_recovery_note.md`.

## Key Artifacts

- `metrics_report.csv`
- `failure_analysis.md`
- `outputs/manual_20_error_review_completed.csv`
- `outputs/core_test_metrics.csv`
- `outputs/bootstrap_95ci.csv`
- `outputs/mcnemar_tests.csv`
- `outputs/slice_robustness_metrics.csv`
- `data_processed/preprocessing_config.json`
- `data_processed/model_configs.json`
- `data_processed/word_to_idx.json`
- `checkpoints/*_best.pt`
- `checkpoints/*_last.pt`
