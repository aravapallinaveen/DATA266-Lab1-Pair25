"""User-locked CycleGAN configuration. Paths are project-relative."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CONFIG = dict(domain_A="Monet", domain_B="Photo", data_A="data/monet_jpg",
              data_B="data/photo_jpg", output_root="naveen", image_size=256,
              resize_size=286, residual_blocks=9, generator_channels=64,
              discriminator_channels=64, normalization="InstanceNorm",
              output_activation="Tanh", discriminator="70x70 PatchGAN",
              adversarial="LSGAN MSE", lambda_cycle=10.0, lambda_identity=5.0,
              optimizer="Adam", learning_rate=2e-4, betas=(0.5, 0.999),
              batch_size=1, epochs=200, steps_per_epoch=300, constant_epochs=100,
              seed=266, fixed_sample_count=4, precision="float32",
              initialization="normal(0, 0.02)", replay_buffer=False,
              sampling="Monet permutation; independent uniform Photo with replacement",
              augmentation=["Resize 286", "RandomCrop 256", "RandomHorizontalFlip",
                            "ToTensor", "Normalize mean/std 0.5"],
              lr_schedule="epochs 1-100: 2e-4; epochs 101-200: 2e-4*(200-epoch)/100")

def epoch_lr(epoch):
    if not 1 <= epoch <= 200:
        raise ValueError("Epoch must be in 1..200")
    return CONFIG["learning_rate"] * (1.0 if epoch <= 100 else (200-epoch)/100)
