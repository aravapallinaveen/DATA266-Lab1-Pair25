# Praveen - Task 3 CycleGAN Results

SID: 8511  
Domains: Monet (A) and Photo (B)  
Model: the independent configuration documented in `src/CONFIG_COMPARISON.md`.

## Reproducibility status

The implementation, training code, final multiscale configuration, and evaluator results are documented here. The final run was executed on RunPod using an NVIDIA GeForce RTX 4090.

## Baseline RTX 4090 local validation run

The baseline code passed a local CUDA training/inference validation on an NVIDIA GeForce RTX 4090. The local validation produced 300 A2B and 7,038 B2A translations. The reported local FID values were:

| Direction | FID | Status |
|---|---:|---|
| Monet -> Photo | 98.329445 | local validation |
| Photo -> Monet | 92.867265 | local validation |
| Mean | 95.598355 | local validation |

KID, LPIPS, precision/recall, human-audit metrics, and the official Kaggle leaderboard score remain pending for the final multiscale run.

## Required evidence to add after the official run

- `checkpoints/latest.pt`
- `outputs/pred_A2B/` and `outputs/pred_B2A/`
- `full_metrics_report.csv`
- raw training log under `reproducibility/raw_logs/`
- exact GPU, elapsed time, images/sec, peak VRAM, and NaN/Inf counts
- 30 fixed-sample audit completed by two raters with agreement statistic
- Kaggle submission and public/private score

## Final multiscale RTX 4090 run

The final multiscale CycleGAN run used an NVIDIA GeForce RTX 4090.

| Item | Result |
|---|---:|
| Epochs | 300 |
| B2A FID | 109.365 |
| A2B FID | 116.184 |
| Overall FID | 112.774391 |
| B2A MiFID | 0.4129 |
| A2B MiFID | 0.4335 |
| Overall MiFID | 0.423197 |
| Kaggle submission | `submission.csv` |

The final evaluator generated 300 A2B images and 7,038 B2A images. The reported training time of 72.77 minutes refers to the resumed final training segment, not the complete run.

KID, LPIPS, precision/recall, human-audit metrics, and Kaggle leaderboard score remain pending unless officially evaluated.



## Human audit of final CycleGAN outputs

Thirty deterministic final outputs were independently reviewed by two human raters:
15 A2B and 15 B2A samples. Selection used seed 8511 with sorted filenames followed
by deterministic seeded sampling. Ratings used a 1–5 scale.

| Measure | Result |
|---|---:|
| Mean style score | 4.016667 |
| Mean content score | 4.25 |
| Mean artifact score | 3.933333 |
| Weighted kappa — style | 0.72 |
| Weighted kappa — content | 0.367816 |
| Weighted kappa — artifacts | 0.769231 |
| Exact agreement — style | 76.67% |
| Exact agreement — content | 63.33% |
| Exact agreement — artifacts | 80.0% |

Detailed ratings are stored in `outputs/human_audit_30_samples.csv`.
Aggregate results are stored in `outputs/human_audit_summary.csv`.
