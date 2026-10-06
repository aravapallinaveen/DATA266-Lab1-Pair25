"""File integrity audit only; instructor-required scientific metrics are deferred."""
from PIL import Image
from config import ROOT
from dataset import image_paths
from utils import write_json

def main():
    report = {}
    for direction, domain in (("A2B", "A"), ("B2A", "B")):
        folder = ROOT / "naveen/outputs" / ("pred_"+direction)
        expected = {p.stem+".png" for p in image_paths(domain)}
        actual = {p.name for p in folder.glob("*.png")}
        errors = []
        for name in sorted(actual):
            try:
                with Image.open(folder / name) as image:
                    if image.mode != "RGB" or image.size != (256, 256):
                        errors.append(name+": incorrect mode or dimensions")
                    image.verify()
            except (OSError, ValueError) as error:
                errors.append(f"{name}: {error}")
        report[direction] = dict(expected=len(expected), found=len(actual),
                                 missing=sorted(expected-actual), extra=sorted(actual-expected), errors=errors)
    write_json(ROOT / "naveen/outputs/audit/output_integrity.json", report)
    print(report)
    if any(v["missing"] or v["extra"] or v["errors"] for v in report.values()):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
