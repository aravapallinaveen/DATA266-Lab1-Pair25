# Task 2 — Praveen SID 8511

## Overview

Independent Yelp Polarity sentiment-classification experiments using three models:

- Baseline CNN
- Experimental BiLSTM with attention
- Experimental BiGRU with masked max pooling

No pretrained embeddings or pretrained language models were used.

## Dataset and preprocessing

- Dataset: Yelp Polarity
- Deterministic 90/10 stratified training-validation split
- Official Yelp test split
- Random seed: 8511
- Maximum sequence length: 256
- `<PAD>` and `<UNK>` tokens
- Negation words such as `no`, `not`, and `never` were preserved

## Model architectures

### Baseline CNN

Embedding size 128, two Conv1D layers with 128 channels and kernel sizes 5 and 3, ReLU activations, dropout 0.25, global max pooling, and a linear classifier.

### BiLSTM with attention

Embedding size 160, bidirectional LSTM with hidden size 128, learned attention pooling, dropout 0.30, and a linear classifier.

### BiGRU with masked max pooling

Embedding size 128, bidirectional GRU with hidden size 96, LayerNorm, dropout 0.20, masked max pooling, and a linear classifier.

## Training configuration

- Optimizer: AdamW
- Learning rate: 0.0002
- Betas: (0.9, 0.999)
- Weight decay: 0.0001
- Loss: BCEWithLogitsLoss
- Batch size: 256
- Epochs: 8
- Seed: 8511
- Best validation checkpoint retained for each model

## Final test metrics

| accuracy | precision_macro | recall_macro | f1_macro | precision_micro | recall_micro | f1_micro | precision_weighted | recall_weighted | f1_weighted | roc_auc | pr_auc | mcc | brier_score | ece_15_bins | model | training_seconds | examples_per_sec | peak_memory_gb | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9506842105263158 | 0.9508026729772004 | 0.9506842105263158 | 0.9506809704983278 | 0.9506842105263158 | 0.9506842105263158 | 0.9506842105263158 | 0.9508026729772004 | 0.9506842105263158 | 0.9506809704983278 | 0.989138771468144 | 0.9893030667426324 | 0.9014868757200682 | 0.0382846221327781 | 0.0131122911579563 | baseline_cnn | 520.7521563998889 | 7742.646766696826 | 0.2134289741516113 | 8545537 |
| 0.951657894736842 | 0.9516649324522024 | 0.951657894736842 | 0.9516577064228522 | 0.951657894736842 | 0.951657894736842 | 0.951657894736842 | 0.9516649324522024 | 0.951657894736842 | 0.9516577064228524 | 0.9901092853185596 | 0.9904367305811986 | 0.9033228271616294 | 0.0368274264037609 | 0.0109351286603578 | experimental_bilstm_attention | 1425.0234403999057 | 2829.427141821959 | 1.0486550331115725 | 10815074 |
| 0.945921052631579 | 0.9459373892575428 | 0.9459210526315788 | 0.9459205573402844 | 0.945921052631579 | 0.945921052631579 | 0.945921052631579 | 0.9459373892575428 | 0.945921052631579 | 0.9459205573402846 | 0.9880187063711912 | 0.988398401438436 | 0.8918584417394985 | 0.0412436090409755 | 0.0187860749623934 | experimental_bigru_pool | 730.8540958000813 | 5516.833008353173 | 0.9012126922607422 | 8544833 |

## Hardware and reproducibility

The original training log is preserved in `training_raw.log`. Runtime, throughput, peak memory, and parameter counts are recorded in `metrics_report.csv`.

The implementation is in `src/train_praveen.py`. This report and the executed notebook use the existing checkpoints and outputs; no retraining was performed.

## Evaluation artifacts

- `outputs/bootstrap_95ci.csv`
- `outputs/mcnemar_tests.csv`
- `outputs/slice_robustness_metrics.csv`
- `outputs/manual_20_error_review.csv`
- `outputs/*_confusion_matrix.csv`
- `outputs/data_quality.csv`
- `outputs/class_distribution.csv`
