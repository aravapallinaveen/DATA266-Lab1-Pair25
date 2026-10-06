"""Portable entry point for the unchanged official instructor notebook.
Default: verify package only. Metric execution requires an explicit --run.
"""
import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reproducibility/manifests/naveen_task3/official_evaluation"
OFFICIAL = EVIDENCE / "Part3_Evaluation_Script.official.ipynb"
OFFICIAL_SHA256 = "702a1265433bf2f15c7900c83442c626d10ac0918094d093dde8ef82069d4cef"
SCORE_KEYS = ["FID_A2B", "MiFID_A2B", "FID_B2A", "MiFID_B2A", "submission_FID", "submission_MiFID"]
DOMAINS = {
    "REAL_MONET": ROOT / "data/monet_jpg",
    "REAL_PHOTO": ROOT / "data/photo_jpg",
    "GEN_A2B": ROOT / "naveen/outputs/pred_A2B",
    "GEN_B2A": ROOT / "naveen/outputs/pred_B2A",
}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def list_images(folder):
    # Match instructor extension collection, sorting and deduplication.
    paths = []
    for ext in (".jpg", ".jpeg", ".png"):
        paths += list(folder.glob("*" + ext))
        paths += list(folder.glob("*" + ext.upper()))
    return sorted(set(paths))

def configured_notebook():
    if sha256(OFFICIAL) != OFFICIAL_SHA256:
        raise ValueError("Original official notebook SHA256 mismatch")
    nb = json.loads(OFFICIAL.read_text(encoding="utf-8"))
    originals = [copy.deepcopy(c["source"]) for c in nb["cells"]]
    source = "".join(nb["cells"][3]["source"])
    lines = source.splitlines(keepends=True)
    updates = {"BASE": str(ROOT / "data"),
               "GEN_A2B": str(DOMAINS["GEN_A2B"]), "GEN_B2A": str(DOMAINS["GEN_B2A"])}
    tree = ast.parse(source)
    seen = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in updates:
                if node.lineno != node.end_lineno:
                    raise ValueError("Unexpected multiline instructor path assignment")
                line = lines[node.lineno - 1]
                lines[node.lineno - 1] = line[:node.value.col_offset] + repr(updates[name]) + line[node.value.end_col_offset:]
                seen.add(name)
    if seen != set(updates):
        raise ValueError("Required instructor path assignments missing")
    updated = "".join(lines)
    # AST equality after restoring ONLY path value nodes.
    restored = ast.parse(updated)
    for old, new in zip(tree.body, restored.body):
        if isinstance(old, ast.Assign) and isinstance(old.targets[0], ast.Name) and old.targets[0].id in updates:
            new.value = copy.deepcopy(old.value)
    if ast.dump(tree) != ast.dump(restored):
        raise ValueError("Unexpected non-path change in instructor configuration")
    nb["cells"][3]["source"] = updated.splitlines(keepends=True)
    for index, cell in enumerate(nb["cells"]):
        if index != 3 and cell["source"] != originals[index]:
            raise ValueError("Instructor cell source changed")
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return nb

def verify_package():
    nb = configured_notebook()
    config = ast.parse("".join(nb["cells"][3]["source"]))
    constants = {n.targets[0].id: n.value.value for n in config.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
    assert constants["N_EVAL"] == 300 and constants["BATCH_SIZE"] == 32
    counts = {name: len(list_images(path)) if path.is_dir() else None for name, path in DOMAINS.items()}
    assert counts["GEN_A2B"] == 300 and counts["GEN_B2A"] == 7038
    with (ROOT / "naveen/submission.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["ID", "FID", "MiFID"]
        rows = list(reader)
    assert len(rows) == 1 and rows[0] == {"ID": "1", "FID": "107.67850136756945", "MiFID": "0.4221133291721344"}
    print(json.dumps({"package_verified": True, "official_sha256": OFFICIAL_SHA256,
                      "input_counts": counts, "N_EVAL": 300, "BATCH_SIZE": 32,
                      "formulas_and_non_path_sources_unchanged": True,
                      "metrics_executed": False}, indent=2))
    if counts["REAL_MONET"] is None or counts["REAL_PHOTO"] is None:
        print("Raw datasets are excluded from this archive. Restore data/monet_jpg and data/photo_jpg before --run.")
    return nb, counts

def execute(nb, counts, output):
    if counts["REAL_MONET"] != 300 or counts["REAL_PHOTO"] != 7038:
        raise ValueError("Restore the exact real datasets (300 Monet, 7038 Photo) before execution")
    import scipy
    import scipy.linalg
    if scipy.__version__ != "1.16.3" or "disp" not in inspect.signature(scipy.linalg.sqrtm).parameters:
        raise RuntimeError("Use an isolated environment with scipy==1.16.3; do not patch the instructor formula or global environment")
    # Every new result stays in a new directory below local_evaluation.
    allowed = (ROOT / "naveen/outputs/local_evaluation").resolve()
    output = output.resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("New evaluation directory must be a child of naveen/outputs/local_evaluation")
    if output.exists():
        raise FileExistsError("Refusing to overwrite any existing evaluation")
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.io import capture_output
    from traitlets.config import Config
    # Protect the supplied package, including all generated image bytes.
    protected_paths = [ROOT / "naveen/submission.csv", ROOT / "naveen/checkpoints/epoch_150.pt", OFFICIAL]
    protected_paths += [p for name, folder in DOMAINS.items() for p in list_images(folder)]
    protected = {str(p): sha256(p) for p in protected_paths}
    output.mkdir(parents=True, exist_ok=False)
    def write_json(name, data):
        with (output / name).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    write_json("Part3_Evaluation_Script.ipynb", nb)
    write_json("instructor_paths.json", {"inputs": {k: str(v) for k, v in DOMAINS.items()},
               "official_sha256": OFFICIAL_SHA256, "modified_config_values": ["BASE", "GEN_A2B", "GEN_B2A"],
               "N_EVAL": 300, "BATCH_SIZE": 32, "scipy": scipy.__version__})
    before_cwd = Path.cwd()
    config = Config()
    config.HistoryManager.hist_file = str(output / "ipython_history.sqlite")
    shell = InteractiveShell.instance(config=config)
    start = time.perf_counter()
    error = None
    try:
        os.chdir(output)
        with (output / "evaluation.log").open("x", encoding="utf-8", buffering=1) as log:
            for index, cell in enumerate(nb["cells"]):
                if cell["cell_type"] != "code":
                    continue
                log.write(f"Official instructor cell {index}\n")
                with capture_output(stdout=True, stderr=True, display=True) as captured:
                    result = shell.run_cell("".join(cell["source"]), store_history=True)
                cell["execution_count"] = shell.execution_count - 1
                cell["outputs"] = []
                for stream in ("stdout", "stderr"):
                    value = getattr(captured, stream)
                    if value:
                        cell["outputs"].append({"output_type": "stream", "name": stream, "text": value})
                        log.write(value)
                        print(value, end="", flush=True)
                for rich in captured.outputs:
                    cell["outputs"].append({"output_type": "display_data", "data": rich.data, "metadata": rich.metadata})
                failure = result.error_before_exec or result.error_in_exec
                if failure:
                    cell["outputs"].append({"output_type": "error", "ename": type(failure).__name__,
                        "evalue": str(failure), "traceback": traceback.format_exception(type(failure), failure, failure.__traceback__)})
                write_json("Part3_Evaluation_Script.executed.ipynb", nb)
                if failure:
                    raise RuntimeError(f"Instructor cell {index} failed") from failure
            ns = shell.user_ns
            scores = dict(zip(SCORE_KEYS, [float(ns[k]) for k in
                          ["fid_A2B", "mifid_A2B", "fid_B2A", "mifid_B2A", "sub_fid", "sub_mifid"]]))
            assert all(math.isfinite(v) for v in scores.values())
            assert ns["N_EVAL"] == 300 and ns["BATCH_SIZE"] == 32
            sets = {k: ns[v] for k, v in {"real_monet": "real_monet", "real_photo": "real_photo",
                    "gen_A2B": "gen_a2b", "gen_B2A": "gen_b2a"}.items()}
            assert all(len(v) == 300 for v in sets.values())
            write_json("instructor_input_manifest.json", sets)
            with (output / "evaluation_results.csv").open("x", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["checkpoint_epoch"] + SCORE_KEYS)
                writer.writeheader()
                writer.writerow({"checkpoint_epoch": 150, **scores})
            nbformat.validate(nbformat.from_dict(nb))
            print(json.dumps(scores, indent=2))
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        os.chdir(before_cwd)
        unchanged = all(sha256(Path(p)) == digest for p, digest in protected.items())
        write_json("runtime.json", {"seconds": time.perf_counter() - start, "error": error,
                   "protected_inputs_unchanged": unchanged, "training_started": False,
                   "images_generated": False, "scipy": scipy.__version__})
        if not unchanged:
            raise RuntimeError("Protected input hash changed")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly execute the official evaluator after restoring datasets")
    parser.add_argument("--output", type=Path, help="New child directory below naveen/outputs/local_evaluation")
    args = parser.parse_args()
    nb, counts = verify_package()
    if args.run:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        output = args.output or ROOT / "naveen/outputs/local_evaluation" / suffix
        execute(nb, counts, output)
    elif args.output:
        parser.error("--output requires --run")

if __name__ == "__main__":
    main()
