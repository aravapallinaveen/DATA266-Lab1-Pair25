"""Generate translations and basic traceable metrics."""
import argparse, csv
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from models import Generator
from utils import denormalize


def cosine(a, b):
    a, b = a.flatten(1), b.flatten(1)
    return torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--input-dir", required=True); ap.add_argument("--direction", choices=["A2B", "B2A"], default="A2B"); ap.add_argument("--output-dir", required=True); ap.add_argument("--device", default="auto")
    args = ap.parse_args(); device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    state = torch.load(args.checkpoint, map_location=device); model = Generator().to(device); model.load_state_dict(state["models"]["G_A2B" if args.direction == "A2B" else "G_B2A"]); model.eval()
    tfm = transforms.Compose([transforms.Resize((286, 286), antialias=True), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); rows = []
    files = sorted(p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    with torch.no_grad():
        for p in files:
            with Image.open(p) as im: x = tfm(im.convert("RGB")).unsqueeze(0).to(device)
            y = model(x); Image.fromarray(denormalize(y)[0].permute(1,2,0).numpy()).save(out / p.name)
            rows.append({"file": p.name, "content_cosine_input_translation": cosine(x, y)})
    with open(out / "inference_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "content_cosine_input_translation"]); w.writeheader(); w.writerows(rows)
    print(f"Generated {len(rows)} images in {out}")


if __name__ == "__main__": main()

