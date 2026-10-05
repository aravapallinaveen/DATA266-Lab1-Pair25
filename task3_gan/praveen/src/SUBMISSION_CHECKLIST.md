# Praveen SID 8511 submission checklist

- [ ] Execute `run_smoke_test.sh` successfully.
- [ ] Run full training on the GPU lab and preserve `outputs/logs/train_raw.jsonl` unchanged.
- [ ] Copy `outputs/checkpoints/latest.pt` and epoch checkpoints into the team repository.
- [ ] Save A2B and B2A generated samples.
- [ ] Run both-direction evaluation.
- [ ] Compute FID, KID, precision/recall, LPIPS, cycle L1, and content cosine similarity.
- [ ] Record generator/discriminator/cycle/identity loss curves.
- [ ] Record gradient norms and NaN/Inf counts.
- [ ] Complete the blinded 30-sample audit with two raters.
- [ ] Calculate Cohen's kappa or percentage agreement.
- [ ] Generate Kaggle predictions directly from the trained checkpoint.
- [ ] Record public/private leaderboard scores and rank.
- [ ] Fill `results.md`, `failure_analysis.md`, and `full_metrics_report.csv`.
- [ ] Record GPU, PyTorch, CUDA, package versions, training time, images/second, and peak VRAM.
- [ ] Remove personal paths and credentials before committing.

