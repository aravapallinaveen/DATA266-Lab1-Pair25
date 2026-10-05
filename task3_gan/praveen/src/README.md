# Praveen SID 8511 - DATA266 Task 3 CycleGAN

This package implements Praveen's independent CycleGAN configuration for Monet-to-Photo and Photo-to-Monet translation.

## Distinct design

- 48/96/192 channel ResNet generators with 6 residual blocks
- Nearest-neighbor upsampling followed by convolution
- 48/96/192/384 channel PatchGAN discriminators
- Spectral normalization in discriminator convolutions
- BCE-with-logits adversarial loss instead of LSGAN/MSE
- AdamW, learning rate 1e-4, betas (0.5, 0.99)
- Batch size 2
- Cycle weight 12 and identity weight 2
- 200 epochs, 150 updates per epoch, 30,000 total updates
- Seed 8511

## Dataset layout

The script accepts either of these layouts:

```text
data/
  monet_jpg/
  photo_jpg/
```

or:

```text
data/
  trainA/
  trainB/
```

Images must be RGB or convertible to RGB. The training transform resizes to 286, randomly crops to 256, randomly flips horizontally, converts to a tensor, and normalizes to [-1, 1].

## Install

Use the Python environment supplied by the GPU lab when possible:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch, install the lab-approved PyTorch wheel first, then run the remaining requirements command without replacing PyTorch.

## CPU smoke test

This checks imports, dataset loading, forward/backward passes, checkpoint writing, and sample generation. It does not produce meaningful image quality.

```bash
python train.py --config config.json --smoke-test --data-root ./data
```

If the dataset is not available locally, the smoke test creates a tiny synthetic RGB dataset when `--synthetic-smoke-data` is supplied:

```bash
python train.py --config config.json --smoke-test --synthetic-smoke-data
```

## Full GPU training

```bash
python train.py --config config.json --data-root /path/to/data --device cuda
```

Resume from the latest checkpoint:

```bash
python train.py --config config.json --data-root /path/to/data --device cuda --resume outputs/checkpoints/latest.pt
```

## Evaluation and inference

Generate translations from a folder:

```bash
python evaluate.py --checkpoint outputs/checkpoints/latest.pt --input-dir /path/to/images --direction A2B --output-dir outputs/inference_A2B
```

The evaluation script also writes a CSV with cycle L1 and content cosine similarity for generated samples. FID, KID, LPIPS, precision/recall, and human-audit fields are included in the report template and can be populated on the GPU lab environment using the class-approved evaluator.

## Output structure

```text
outputs/
  checkpoints/
  samples/
  logs/
  metrics_report.csv
  config_used.json
```

The raw JSONL log is never rewritten by the script. Copy it unchanged into the team's raw-log folder.


