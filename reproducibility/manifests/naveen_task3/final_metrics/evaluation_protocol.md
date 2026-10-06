# Epoch-150 evaluation protocol

Checkpoint: naveen/checkpoints_repro/repro_20261002T030447909094Z_57cacd34/epoch_150.pt. Epoch 150; global step 45000; seed 171.
All paths in this implementation are project-relative.

## Search and exact reuse
Searched project source/notebook cells, reference folder, instructor notebooks, the team course archive, DATA266_Lab1_Fall_2026.pdf (Task 3, page 9), and the CycleGAN training guide. The assignment lists metrics and defines content cosine as input versus translation; it supplies no further implementation or LPIPS pairing.
Classifier precision/recall in Task 2 is not reused as a generative metric.
Reused verbatim cell 5 of the preserved official epoch-150 evaluator: get_inception_model, INCEPTION_TF, load_batch, and get_activations. Cell 4 supplies the same sorted-prefix image selection.
Reused src/models.py Generator and src/dataset.py load_image(training=False) for in-memory cycles.
Parameter counts come from the existing executed step-18 notebook output; training figures come directly from the complete 1–150 epoch CSV history.
Course-search excerpts and reuse manifests are under manifests/.

## Shared samples and features
Each feature set contains 300 images: all sorted real Monet; first 300 sorted real Photo; corresponding same-stem existing A2B/B2A outputs. No hand-picking or filtering. All 300/7038 stored prediction files are preserved.
Exactly the instructor preprocessing: convert RGB, Resize(299), CenterCrop(299), ToTensor, ImageNet normalization.
Torchvision Inception V3 IMAGENET1K_V1, transform_input=False, fc=Identity, eval mode and no_grad; raw 2048-D pooled features before the classifier. Package versions and model SHA256 are recorded in evaluation_environment.txt.
Features are extracted once into inception_features.npz and reused across KID, generative precision/recall, and content cosine. Previously computed FID/MiFID are loaded from their existing CSV, never recomputed.

## KID A2B and B2A
A2B compares generated Photos with real Photos. B2A compares generated Monet with real Monet.
Unbiased polynomial MMD squared: k(x,y)=(x dot y / 2048 + 1)^3.
Sum off-diagonal within-real and within-generated kernels divided by m(m-1), minus twice the mean cross kernel.
100 subsets, each containing 100 independently sampled real/generated items without replacement; NumPy PCG64 seed 171 reset per direction. Mean and population standard deviation (ddof=0) of subset estimates, unscaled raw KID. Estimator values are not clipped.
Fallback formula follows Binkowski et al.'s reference polynomial-MMD procedure:
https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py
Subset indices and all estimates are retained.

## Generative precision and recall
Raw, unnormalized instructor Inception features; 300 real and 300 generated per direction.
Closed Euclidean balls centered on each real/generated feature, radius to its third other nearest neighbor (k=3; self excluded).
Precision = fraction of generated samples inside the union of real balls.
Recall = fraction of real samples inside the union of generated balls.
Deterministic full pairwise distances; seed 171, no extra sampling.
Protocol: https://arxiv.org/abs/1904.06991
These are feature-space generative metrics, not classifier scores.

## Cycle reconstruction L1
300 Monet and the first 300 sorted Photos, same seed/sample manifest.
Convert RGB, Resize((256,256)), ToTensor, Normalize(mean/std=0.5); no augmentation/crop/flip.
Epoch-150 generators loaded strictly, eval/no_grad, float32. A->B->A and B->A->B tensors exist only in memory.
Per-image unweighted mean absolute error; average across 300 images. Primary L1 uses [0,1]. Also record [-1,1] tensor L1 separately (twice [0,1] L1).
No generated images are rewritten or exported.

## LPIPS A and B
No supplied paired-target protocol exists for this unpaired task. Use real input versus its cycle reconstruction in the same domain.
Official lpips 0.1.4; LPIPS net='alex', version='0.1', pretrained=True, pnet_rand=False, spatial=False, eval mode, no_grad.
Inputs are RGB 256x256 tensors in [-1,1], normalize=False. Mean and population standard deviation (ddof=0) over 300/domain. Per-image values retained.
Official implementation: https://github.com/richzhang/PerceptualSimilarity
Metric-only AlexNet and LPIPS calibration weights never generate/modify submission images.
If the metric dependency fails, its fields stay empty and the exact error is recorded; no proxy is substituted.

## Content preservation cosine
Input versus its existing same-filename translation: real Monet/A2B and real Photo/B2A, 300 pairs per direction.
Cosine = dot(source feature, translation feature)/(L2 norm(source) * L2 norm(translation)).
Same instructor Inception preprocessing/features reused. Mean and population std (ddof=0), with per-pair scores retained. No cross-domain target pairing is asserted.

## Existing training and efficiency records
Epoch-150 losses, LR, gradient norms, NaN/Inf and efficiency come from the existing epoch CSV.
Training cycle losses are weighted by 10; identity losses by 5. Evaluation cycle L1 is unweighted and separately labeled.
training_time_minutes is the exact epoch-150 logged elapsed_minutes; images/sec is epoch-150 training input throughput; peak_gpu_memory_GB is that epoch's recorded allocated VRAM divided by 1024^3 (GiB).
Also record complete-history NaN/Inf totals and maximum logged VRAM.
Parameter counts: existing executed notebook output, not an architecture change.
Six plots use the complete existing epoch history through 150, with no smoothing or edits to logs.

## Pending
human_audit_score, inter_rater_agreement, Kaggle_public_score, Kaggle_private_score, and Kaggle_rank are literal PENDING. No human/Kaggle result is estimated.
