# Praveen - Task 3 CycleGAN Results

SID: 8511  
Domains: Monet (A) and Photo (B)  
Model: Independent Praveen SID 8511 CycleGAN configuration documented in `src/CONFIG_COMPARISON.md`.

## Reproducibility status

The implementation, notebook, RTX 4090 training log, and metrics report are committed. The full training run completed on a RunPod NVIDIA GeForce RTX 4090 using CUDA.

## RTX 4090 training run

| Item | Result |
|---|---:|
| GPU | NVIDIA GeForce RTX 4090 |
| Device | CUDA |
| Epochs | 200 |
| Training duration | 48.08 minutes |
| Monet images used for A2B inference | 300 |
| Photo images used for B2A inference | 7,038 |

## Local-validation metrics

The following FID values were computed by the local evaluator and are labeled local validation. They should be replaced with official class-evaluator values if the class evaluator is required.

| Direction | FID | Status |
|---|---:|---|
| Monet -> Photo (A2B) | 98.329445 | local validation |
| Photo -> Monet (B2A) | 92.867265 | local validation |
| Mean | 95.598355 | local validation |

The observed mean content-cosine values were:

| Direction | Content cosine | Status |
|---|---:|---|
| Monet -> Photo (A2B) | 0.8229490374 | local validation |
| Photo -> Monet (B2A) | 0.7893867940 | local validation |

## Metrics still pending

The following metrics were not produced by the current evaluator and remain pending:

- KID
- Generative precision
- Generative recall
- LPIPS
- Cycle L1
- Human style, content, and artifact scores
- Human inter-rater kappa
- Kaggle score

These values must come from the class-approved evaluator, a documented two-rater human audit, or the Kaggle result, as applicable. No values are fabricated in `full_metrics_report.csv`.

## Saved run evidence

- `training_4090.log` records the completed 200-epoch training run.
- `full_metrics_report.csv` records the observed local-validation metrics and labels unavailable metrics as pending.
- `outputs/checkpoints/latest.pt` and generated inference archives are retained with the local run artifacts because they exceed normal GitHub file-size limits.

