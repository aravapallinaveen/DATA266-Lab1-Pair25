"""Unpaired Monet/Photo dataset utilities."""
from pathlib import Path
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_domain(root, names):
    root = Path(root)
    for name in names:
        candidate = root / name
        if candidate.exists():
            files = sorted(p for p in candidate.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
            if files:
                return files
    raise FileNotFoundError(f"Could not find a populated domain under {root}; tried {names}")


def make_transform(image_size=256, resize_size=286, train=True):
    ops = [transforms.Resize((resize_size, resize_size), antialias=True)]
    if train:
        ops += [transforms.RandomCrop(image_size), transforms.RandomHorizontalFlip()]
    else:
        ops += [transforms.CenterCrop(image_size)]
    ops += [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    return transforms.Compose(ops)


class UnpairedDataset(Dataset):
    def __init__(self, root, image_size=256, resize_size=286, train=True, synthetic=False, synthetic_size=8):
        self.transform = make_transform(image_size, resize_size, train)
        if synthetic:
            self.a = [None] * synthetic_size
            self.b = [None] * synthetic_size
            self.synthetic = True
        else:
            self.a = find_domain(root, ["monet_jpg", "trainA", "A", "monet"])
            self.b = find_domain(root, ["photo_jpg", "trainB", "B", "photo"])
            self.synthetic = False

    def __len__(self):
        return len(self.a)

    def _load(self, path, domain):
        if self.synthetic:
            base = torch.zeros(3, 286, 286)
            if domain == "A":
                base[0].fill_(0.2)
                base[1, 80:210, 80:210] = 0.8
            else:
                base[2].fill_(0.2)
                base[1, 60:220, 60:220] = 0.8
            return base[:, :256, :256] * 2 - 1
        with Image.open(path) as im:
            return self.transform(im.convert("RGB"))

    def __getitem__(self, index):
        b_index = random.randrange(len(self.b))
        return {"A": self._load(self.a[index], "A"), "B": self._load(self.b[b_index], "B")}

