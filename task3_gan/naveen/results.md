# Task 3 CycleGAN — Naveen

## Final checkpoint

Epoch 150  
Global step 45000  
Checkpoint: checkpoints/epoch_150.pt  
SHA256: 9cc42d24ca52fd53d3ba1b690c92a5fb0fd4eb8a27dfd087f86e030a3ae24152

## Architecture

Two ResNet-9 generators; two 70x70 PatchGAN discriminators.
InstanceNorm (affine=False, track_running_stats=False); Tanh generator output.
LSGAN with MSE adversarial loss. Four-model parameter count: 28,285,832.
No image replay buffer.

## Hyperparameters

SEED = 171; SID4 = int("0171") = 171.
Adam; initial lr = 2e-4; betas = (0.5, 0.999); batch size = 1.
Cycle L1 loss weight = 10; identity L1 enabled, weight = 5.
Normal weight initialization (mean 0, std 0.02); float32.

## Dataset / domain definition

A = Monet (300 images). B = Photo (7038 images).
A2B = Monet -> Photo. B2A = Photo -> Monet.
Raw datasets excluded; see ../data/DATASET_README.txt.

## Training configuration

Completed epochs 1–150: 300 steps/epoch; 45,000 total steps.
Each Monet used once per epoch; Photo independently sampled uniformly with replacement.
Training augmentation: Resize 286x286, RandomCrop 256, RandomHorizontalFlip,
ToTensor and Normalize(mean/std 0.5) to [-1,1].
Deterministic inference: RGB, Resize 256x256, ToTensor, Normalize(mean/std 0.5).
LR constant 2e-4 through epoch 100; 2e-4*(200-epoch)/100 for epochs 101–200.
Recorded epoch-150 LR: 0.0001. This package contains the completed epoch-150 result.
Logged training images/sec counts both input domains (600 images per epoch).

## Final metrics

| Metric | Recorded value |
|---|---:|
| FID A2B | 109.48913912902457 |
| MiFID A2B | 0.43110403418540955 |
| FID B2A | 105.86786360611433 |
| MiFID B2A | 0.41312262415885925 |
| Submission FID | 107.67850136756945 |
| Submission MiFID | 0.4221133291721344 |
| KID A2B mean | 0.025088945133143494 |
| KID A2B std | 0.002984295586468944 |
| KID B2A mean | 0.010222349777397022 |
| KID B2A std | 0.0015632233408134943 |
| Generative precision A2B | 0.6433333333333333 |
| Generative recall A2B | 0.3 |
| Generative precision B2A | 0.35 |
| Generative recall B2A | 0.6133333333333333 |
| Cycle L1 A [0,1] | 0.05189370443423589 |
| Cycle L1 B [0,1] | 0.06651143689950308 |
| LPIPS A mean | 0.3531525456905365 |
| LPIPS A std | 0.10359809903306247 |
| LPIPS B mean | 0.3055188757677873 |
| LPIPS B std | 0.07937863311323239 |
| Content cosine A2B mean | 0.8284416557570147 |
| Content cosine A2B std | 0.07511244998941267 |
| Content cosine B2A mean | 0.7828227874722511 |
| Content cosine B2A std | 0.10520040518911024 |
| Parameters, four models | 28285832 |
| Logged training time, minutes | 91.63137033666717 |
| Training images/sec, epoch 150 | 16.807604405639427 |
| Peak allocated VRAM GiB, all epochs | 2.4495863914489746 |
| NaN, all epochs | 0.0 |
| Inf, all epochs | 0.0 |

Source files: metrics_report.csv and full_metrics_report.csv.
Official evaluator: N_EVAL=300, BATCH_SIZE=32; 300 images evaluated per set.
Both complete prediction folders are included.
Protocols: ../reproducibility/manifests/naveen_task3/final_metrics/evaluation_protocol.md.
Training cycle/identity losses in the CSV are weighted.
The reported cycle L1 [0,1] is an evaluation measurement.
Peak VRAM above is training allocated memory, maximum across all epochs.

## Hardware

NVIDIA GeForce RTX 4090; recorded total memory 23.98779296875 GiB.
CUDA 13.0; NVIDIA driver 610.60.
Python 3.12.15; Windows 11 build 22631.

| Package | Recorded version |
|---|---|
| torch | 2.14.1+cu130 |
| torchvision | 0.29.1+cu130 |
| numpy | 2.5.2 |
| pandas | 3.0.6 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| Pillow | 12.3.0 |
| lpips | 0.1.4 |
| pypdf | 6.19.0 |
| ipython | 9.17.1 |

Official FID execution used isolated SciPy 1.16.3 for sqrtm(..., disp=False)
compatibility. Recorded global/final supplemental SciPy was 1.18.1.
Full environment evidence is preserved in the manifests.

## Kaggle

PENDING final leaderboard score/rank

## Human audit

A blinded human evaluation was conducted on 30 fixed CycleGAN translations:
15 Monet→Photo (A2B) and 15 Photo→Monet (B2A).

Two independent human raters evaluated each translation using a 1–5 scale
for target-style quality, content preservation, and artifact quality.
Higher scores indicate better performance.

| Human evaluation metric | Result |
|---|---:|
| Style quality | 3.667 / 5 |
| Content preservation | 4.033 / 5 |
| Artifact quality | 3.433 / 5 |
| Overall human audit score | 3.711 / 5 |
| Style weighted Cohen's kappa | 0.671 |
| Content weighted Cohen's kappa | 0.742 |
| Artifact weighted Cohen's kappa | 0.610 |
| Mean weighted Cohen's kappa | 0.674 |
| Mean exact agreement | 57.78% |

The human evaluation showed that content preservation was the strongest
dimension, while artifact quality received the lowest mean score.
Monet→Photo translations were rated higher than Photo→Monet translations
across style, content preservation, and artifact quality.

Evidence:
`outputs/human_audit/human_audit_30_samples_completed.csv`,
`outputs/human_audit/human_audit_summary.csv`,
`outputs/human_audit/human_audit_by_direction.csv`.

## Analysis

The epoch-150 CycleGAN achieved better distribution-level performance in the
B2A direction by FID, while the human audit favored A2B translations across
all three qualitative dimensions.

A2B achieved higher generative precision but lower recall, suggesting that
its outputs were concentrated in a narrower but relatively realistic region
of the target Photo distribution. B2A showed the opposite pattern, with
greater coverage but lower precision.

Content preservation was the strongest human-rated dimension at 4.033/5,
while artifact quality was the weakest at 3.433/5. The overall human audit
score was 3.711/5. Inter-rater reliability was reasonably consistent, with
a mean quadratic-weighted Cohen's kappa of 0.674.

The main limitation is maintaining clean target-style appearance without
introducing visible translation artifacts. Future work could test replay
buffers, alternative upsampling strategies, discriminator regularization,
and checkpoint selection using both quantitative metrics and fixed-sample
human evaluation.
