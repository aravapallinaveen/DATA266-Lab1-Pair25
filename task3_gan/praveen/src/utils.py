import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize(x):
    return ((x.detach().cpu().clamp(-1, 1) + 1) / 2 * 255).byte()


def save_grid(tensor, path, nrow=4):
    from torchvision.utils import make_grid
    grid = make_grid(denormalize(tensor).float() / 255.0, nrow=nrow)
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def append_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_metrics_template(path):
    fields = ["direction", "fid", "kid", "precision", "recall", "cycle_l1", "lpips", "content_cosine", "human_style", "human_content", "human_artifacts", "human_kappa", "kaggle_score"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(path).exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

