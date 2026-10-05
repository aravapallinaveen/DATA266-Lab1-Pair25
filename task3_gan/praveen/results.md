# Praveen - Task 3 CycleGAN Results

SID: 8511  
Domains: Monet (A) and Photo (B)  
Model: the independent configuration documented in `src/CONFIG_COMPARISON.md`.

## Reproducibility status

The implementation and notebook are committed. The official GPU-lab run must still be executed and its raw log, checkpoint, generated predictions, and complete metrics report must be copied into this folder. This file intentionally does not claim pending metrics.

## Local validation run

The code passed a synthetic smoke test and a local CUDA training/inference validation on an NVIDIA GeForce RTX 2060. The local validation produced 300 A2B and 7,038 B2A translations. The reported local FID values were:

| Direction | FID | Status |
|---|---:|---|
| Monet -> Photo | 98.329445 | local validation |
| Photo -> Monet | 92.867265 | local validation |
| Mean | 95.598355 | local validation |

KID, generative precision/recall, LPIPS, cycle-L1, content cosine, human audit, and Kaggle score must be filled from the official run/evaluator.

## Required evidence to add after the official run

- `checkpoints/latest.pt`
- `outputs/pred_A2B/` and `outputs/pred_B2A/`
- `full_metrics_report.csv`
- raw training log under `reproducibility/raw_logs/`
- exact GPU, elapsed time, images/sec, peak VRAM, and NaN/Inf counts
- 30 fixed-sample audit completed by two raters with agreement statistic
- Kaggle submission and public/private score
