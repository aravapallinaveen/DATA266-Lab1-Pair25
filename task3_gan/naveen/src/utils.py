import json
import platform
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from config import CONFIG

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

def environment(device):
    info = dict(python=sys.version, pytorch=torch.__version__, torchvision=torchvision.__version__,
                cuda_runtime=torch.version.cuda, os=platform.platform(), seed=CONFIG["seed"],
                configuration=CONFIG, device=str(device), gpu_name=None, gpu_vram_GB=None,
                nvidia_driver=None)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu_name=p.name, gpu_vram_GB=p.total_memory/1024**3)
    try:
        info["nvidia_driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info

def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = ((tensor.detach().cpu().squeeze(0).clamp(-1, 1)+1)*127.5).round().to(torch.uint8)
    Image.fromarray(arr.permute(1, 2, 0).numpy()).convert("RGB").save(path)
    with Image.open(path) as image:
        assert image.mode == "RGB" and image.size == (256, 256)
        image.verify()

def random_states():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore_random(states):
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"].cpu())
    if states["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in states["cuda"]])

def save_checkpoint(path, models, optimizers, schedulers, epoch, global_step, fixed, elapsed, metadata=None):
    payload = dict(epoch=epoch, global_step=global_step, configuration=CONFIG,
                   models={k: m.state_dict() for k, m in models.items()},
                   optimizers={k: o.state_dict() for k, o in optimizers.items()},
                   schedulers={k: s.state_dict() for k, s in schedulers.items()},
                   random_states=random_states(), fixed_samples=fixed, elapsed_seconds=elapsed)
    if metadata:
        payload.update(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

def load_checkpoint(path, device="cpu"):
    # Resume includes Python/NumPy RNG objects; only load your own trusted checkpoints.
    return torch.load(path, map_location=device, weights_only=False)

def gradient_stats(parameters):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    nan = sum(int(torch.isnan(g).sum()) for g in grads)
    inf = sum(int(torch.isinf(g).sum()) for g in grads)
    norm = float(torch.sqrt(sum(g.float().square().sum() for g in grads))) if grads else 0.
    return norm, nan, inf
