"""Exactly one Monet permutation and independent Photo replacement draws per epoch."""
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import ROOT, CONFIG

def transform(training=False):
    ops = ([transforms.Resize((286, 286)), transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip()] if training else
           [transforms.Resize((256, 256))])
    return transforms.Compose(ops + [transforms.ToTensor(),
                                    transforms.Normalize((.5,)*3, (.5,)*3)])

def image_paths(domain):
    path = ROOT / CONFIG["data_" + domain]
    result = sorted(p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    expected = 300 if domain == "A" else 7038
    if len(result) != expected:
        raise ValueError(f"Domain {domain}: expected {expected} JPEGs, found {len(result)}")
    return result

def load_image(path, training=False):
    with Image.open(path) as im:
        return transform(training)(im.convert("RGB"))

class EpochDataset(Dataset):
    def __init__(self, paths_A, paths_B, tiny=False):
        self.paths_A, self.paths_B = paths_A, paths_B
        indices = random.sample(range(len(paths_A)), len(paths_A))
        if tiny:
            indices = indices[:2]
        self.pairs = [(i, random.randrange(len(paths_B))) for i in indices]
        assert len(set(i for i, _ in self.pairs)) == len(self.pairs)
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, index):
        a, b = self.pairs[index]
        return load_image(self.paths_A[a], True), load_image(self.paths_B[b], True)
    def manifest(self):
        return [{"step": n+1, "A": self.paths_A[a].relative_to(ROOT).as_posix(),
                 "B": self.paths_B[b].relative_to(ROOT).as_posix()}
                for n, (a, b) in enumerate(self.pairs)]
