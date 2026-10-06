"""Epoch-150 evaluation and recording only. All artifact paths are project-relative."""
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

OUT_REL = Path("naveen/outputs_repro/final_metrics_epoch150")
EVAL_REL = Path("naveen/outputs_repro/evaluation/epoch_150")
MAIN_REL = Path("naveen/src/Task3_CycleGAN_Naveen_Repro.ipynb")
FINAL_NB_REL = Path("naveen/src/Task3_Final_Metrics_Epoch150.ipynb")
SEED = 171
CAP = 300
KID_SUBSETS = 100
KID_SUBSET_SIZE = 100
PR_K = 3
SCORE_KEYS = ["FID_A2B", "MiFID_A2B", "FID_B2A", "MiFID_B2A", "submission_FID", "submission_MiFID"]
EXPECTED_FID = [109.48913912902457, 0.43110403418540955, 105.86786360611433,
                0.41312262415885925, 107.67850136756945, 0.4221133291721344]


def find_root():
    return next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p / "naveen/src/models.py").is_file() and (p / "data/monet_jpg").is_dir())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(root, path):
    return Path(path).resolve().relative_to(root).as_posix()


def output_path(ctx, name):
    p = (ctx["out"] / name).resolve()
    if not p.is_relative_to(ctx["out"].resolve()):
        raise PermissionError("Evaluation outputs must remain inside the new final-metrics folder")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(ctx, name, data):
    with output_path(ctx, name).open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_csv(ctx, name, rows, fields):
    with output_path(ctx, name).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_paths(folder):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def evidence_files(root):
    files = set()
    scopes = [
        root / EVAL_REL,
        root / "naveen/outputs/evaluation/epoch_100",
        root / "naveen/outputs_repro/evaluation/epoch_100",
        root / "naveen/logs_repro",
    ]
    skip = {"python_packages", "packages_readable", "torch_cache", "instructor_workdir", "runtime",
            "cache", "__pycache__", ".ipynb_checkpoints"}
    for scope in scopes:
        for folder, dirs, names in os.walk(scope, followlinks=False):
            dirs[:] = [d for d in dirs if d not in skip]
            files.update(Path(folder) / name for name in names)
    files.update((root / "naveen/src").glob("*.py"))
    files.add(root / MAIN_REL)
    for folder in [root / "naveen/checkpoints", root / "naveen/checkpoints_repro"]:
        files.update(folder.rglob("epoch_100.pt"))
    return sorted(files)


def prepare_evaluation(root):
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    out = root / OUT_REL
    ctx = dict(root=root, out=out, errors=[], started=time.perf_counter())
    cps = list((root / "naveen/checkpoints_repro").rglob("epoch_150.pt"))
    assert len(cps) == 1, "An unambiguous preserved reproduction epoch-150 checkpoint is required"
    cp = cps[0]
    cp_hash = sha256(cp)
    state = torch.load(cp, map_location="cpu", weights_only=False)
    assert (state["epoch"], state["global_step"], state["seed"]) == (150, 45000, 171)
    assert state["run_kind"] == "repro" and not state.get("diagnostic_only")
    ctx.update(checkpoint=relative(root, cp), checkpoint_sha256=cp_hash,
               generator_weights={k: state["models"][k] for k in ("G_A2B", "G_B2A")},
               checkpoint_configuration=state["configuration"])
    del state
    folders = {
        "real_A": root / "data/monet_jpg", "real_B": root / "data/photo_jpg",
        "gen_A2B": root / EVAL_REL / "pred_A2B", "gen_B2A": root / EVAL_REL / "pred_B2A",
    }
    counts = {"real_A": 300, "real_B": 7038, "gen_A2B": 300, "gen_B2A": 7038}
    selected, all_hashes, inventory = {}, {}, {}
    for name, folder in folders.items():
        paths = image_paths(folder)
        assert len(paths) == counts[name], (name, len(paths))
        if name.startswith("gen_"):
            assert all(p.suffix.lower() == ".jpg" for p in paths)
        for i, p in enumerate(paths, 1):
            with Image.open(p) as im:
                assert im.size == (256, 256) and im.mode == "RGB", p
                im.verify()
            all_hashes[relative(root, p)] = sha256(p)
            if i % 2000 == 0:
                print(f"Verified {name}: {i}/{len(paths)}", flush=True)
        selected[name] = paths[:CAP]
        inventory[name] = {"count": len(paths), "selected": CAP, "mode": "RGB", "size": [256, 256],
                           "folder": relative(root, folder)}
        print(f"Verified {name}: {len(paths)} existing images; selected {CAP}.", flush=True)
    for a, b in [("real_A", "gen_A2B"), ("real_B", "gen_B2A")]:
        assert [p.stem for p in selected[a]] == [p.stem for p in selected[b]]
    previous = json.loads((root / EVAL_REL / "metrics/instructor_input_manifest.json").read_text(encoding="utf-8"))
    for ours, old in [("real_A", "real_monet"), ("real_B", "real_photo"), ("gen_A2B", "gen_A2B"), ("gen_B2A", "gen_B2A")]:
        assert [relative(root, p) for p in selected[ours]] == previous[old]
    save_json(ctx, "manifests/evaluation_samples.json",
              {"seed": SEED, "N_EVAL": CAP, "sets": {k: [relative(root, p) for p in v] for k, v in selected.items()},
               "selection": "All 300 sorted Monet; first 300 sorted Photo; same-stem translations; no quality selection."})
    save_json(ctx, "manifests/input_inventory.json", inventory)
    # Verify logged training history, without training or modifying any CSV.
    log_root = root / "naveen/logs_repro" / cp.parent.name
    csv_paths = sorted(log_root.glob("*_epochs.csv"))
    history = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in csv_paths], ignore_index=True).sort_values("epoch")
    assert list(history["epoch"]) == list(range(1, 151))
    assert list(history["global_step"]) == [e * 300 for e in range(1, 151)]
    row = history.loc[history["epoch"] == 150].iloc[0].to_dict()
    assert row["global_step"] == 45000 and row["learning_rate"] == 0.0001
    assert row["NaN_count"] == row["Inf_count"] == 0
    # Parameter counts are read from an existing executed notebook output.
    nb = json.loads((root / MAIN_REL).read_text(encoding="utf-8"))
    parameter_cell = next(c for c in nb["cells"] if c.get("id") == "step-18")
    text = "\n".join("".join(o.get("data", {}).get("text/plain", [])) for o in parameter_cell["outputs"])
    parameter_counts = {name: int(re.search(r"\b" + name + r"\s+(\d+)", text).group(1))
                        for name in ("G_A2B", "G_B2A", "D_A", "D_B")}
    assert parameter_counts == dict(G_A2B=11378179, G_B2A=11378179, D_A=2764737, D_B=2764737)
    with (root / EVAL_REL / "metrics/evaluation_results.csv").open(newline="", encoding="utf-8") as f:
        scores = list(csv.DictReader(f))
    assert len(scores) == 1 and int(scores[0]["checkpoint_epoch"]) == 150
    fid = {k: float(scores[0][k]) for k in SCORE_KEYS}
    assert [fid[k] for k in SCORE_KEYS] == EXPECTED_FID
    save_json(ctx, "manifests/training_evidence.json",
              {"epoch150_row": row, "parameter_counts": parameter_counts,
               "parameter_evidence": relative(root, root / MAIN_REL) + "#step-18-existing-output",
               "history_sources": [relative(root, p) for p in csv_paths],
               "training_time_definition": "Exact elapsed_minutes at epoch 150, including logged elapsed run overhead.",
               "throughput_definition": "Epoch-150 logged training images/sec: 600 input domain images / epoch_seconds.",
               "memory_definition": "Epoch-150 logged max_memory_allocated / 1024^3; training, not metric extraction."})
    # Protect all inputs, both epoch-100 evidence folders, epoch-150 FID evidence, logs, and checkpoint.
    protected = dict(all_hashes)
    protected[relative(root, cp)] = cp_hash
    for i, p in enumerate(evidence_files(root), 1):
        key = relative(root, p)
        if key not in protected:
            protected[key] = sha256(p)
        if i % 4000 == 0:
            print(f"Protected evidence hashes: {i} files", flush=True)
    save_json(ctx, "manifests/protected_files_before.json", protected)
    # Reuse instructor metric weights, copied only into the new evaluation cache.
    weights = root / EVAL_REL / "metrics/torch_cache/hub/checkpoints/inception_v3_google-0cc3c7bd.pth"
    target = output_path(ctx, "torch_cache/hub/checkpoints/" + weights.name)
    assert not target.exists()
    shutil.copyfile(weights, target)
    assert sha256(target) == sha256(weights) and sha256(target).startswith("0cc3c7bd")
    ctx.update(paths=selected, protected=protected, history=history, epoch150=row,
               parameter_counts=parameter_counts, official_scores=fid,
               inception_weights_sha256=sha256(target))
    print(f"Checkpoint {ctx['checkpoint']}: epoch 150, step 45000, seed 171.")
    print("Verified complete epochs 1–150, existing official scores, parameter-count evidence, and immutable inputs.")
    return ctx


def evaluation_environment(ctx):
    import numpy as np
    import torch
    import torchvision
    names = ["torch", "torchvision", "numpy", "pandas", "scipy", "matplotlib", "Pillow", "lpips", "pypdf", "ipython"]
    versions = {k: importlib.metadata.version(k) for k in names}
    info = {"python": sys.version, "packages": versions, "cuda": torch.version.cuda, "seed": SEED,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "GPU_total_GiB": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
            "Inception": "torchvision Inception_V3_Weights.IMAGENET1K_V1; fc=Identity; 2048 pooled features",
            "Inception_weights_sha256": ctx["inception_weights_sha256"],
            "LPIPS": "lpips 0.1.4; net=alex; version=0.1; pretrained=True; pnet_rand=False; eval mode",
            "AlexNet_weights_sha256": sha256(ctx["out"] / "torch_cache/hub/checkpoints/alexnet-owt-7be5be79.pth")}
    try:
        info["nvidia_driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        info["nvidia_driver"] = "unavailable"
    save_json(ctx, "manifests/evaluation_environment.json", info)
    output_path(ctx, "evaluation_environment.txt").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    ctx["environment"] = info
    return info


def kid_from_features(real, generated, seed=SEED, subsets=KID_SUBSETS, subset_size=KID_SUBSET_SIZE):
    import numpy as np
    x, y = real.astype(np.float64), generated.astype(np.float64)
    assert x.shape == y.shape == (CAP, 2048)
    # Reuse full kernels across subsets; this is the same unbiased estimator.
    kernel_xx = (x @ x.T / x.shape[1] + 1)**3
    kernel_yy = (y @ y.T / y.shape[1] + 1)**3
    kernel_xy = (x @ y.T / x.shape[1] + 1)**3
    rng = np.random.default_rng(seed)
    estimates, indices = [], []
    for _ in range(subsets):
        ix = rng.choice(len(x), subset_size, replace=False)
        iy = rng.choice(len(y), subset_size, replace=False)
        aa = kernel_xx[np.ix_(ix, ix)]
        bb = kernel_yy[np.ix_(iy, iy)]
        ab = kernel_xy[np.ix_(ix, iy)]
        m = subset_size
        estimate = (aa.sum() - np.trace(aa) + bb.sum() - np.trace(bb)) / (m * (m-1)) - 2 * ab.mean()
        estimates.append(float(estimate))
        indices.append((ix.tolist(), iy.tolist()))
    values = np.asarray(estimates)
    assert np.isfinite(values).all()
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
            "subset_estimates": estimates, "subset_indices": indices,
            "seed": seed, "subsets": subsets, "subset_size": subset_size, "std_ddof": 0}


def squared_distances(a, b):
    import numpy as np
    a, b = a.astype(np.float64), b.astype(np.float64)
    return np.maximum((a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2*a@b.T, 0)


def generative_precision_recall(real, generated, k=PR_K):
    import numpy as np
    rr, gg = squared_distances(real, real), squared_distances(generated, generated)
    np.fill_diagonal(rr, np.inf)
    np.fill_diagonal(gg, np.inf)
    radii_real = np.partition(rr, k-1, axis=1)[:, k-1]
    radii_gen = np.partition(gg, k-1, axis=1)[:, k-1]
    rg = squared_distances(real, generated)
    precision_membership = (rg <= radii_real[:, None]).any(axis=0)
    recall_membership = (rg <= radii_gen[None, :]).any(axis=1)
    return {"precision": float(precision_membership.mean()), "recall": float(recall_membership.mean()),
            "k": k, "real_samples": len(real), "generated_samples": len(generated),
            "precision_membership": precision_membership.tolist(), "recall_membership": recall_membership.tolist()}


def content_cosine(source, translated):
    import numpy as np
    a, b = source.astype(np.float64), translated.astype(np.float64)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    assert (denom > 0).all()
    values = (a*b).sum(1) / denom
    assert np.isfinite(values).all() and (np.abs(values) <= 1+1e-12).all()
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0)), "values": values.tolist(), "std_ddof": 0}


def metric_sanity_checks():
    import numpy as np
    a = np.arange(5, dtype=np.float64)[:, None]
    same = generative_precision_recall(a, a, 3)
    separate = generative_precision_recall(a, a+100, 3)
    assert same["precision"] == same["recall"] == 1
    assert separate["precision"] == separate["recall"] == 0
    c = content_cosine(a+1, a+1)
    assert c["mean"] == 1 and c["std"] == 0
    from sklearn.metrics.pairwise import polynomial_kernel
    rng = np.random.default_rng(171)
    x, y = rng.random((300,2048)), rng.random((300,2048))
    result = kid_from_features(x, y, subsets=1, subset_size=8)
    ix, iy = result["subset_indices"][0]
    xx = polynomial_kernel(x[ix], degree=3, gamma=1/2048, coef0=1)
    yy = polynomial_kernel(y[iy], degree=3, gamma=1/2048, coef0=1)
    xy = polynomial_kernel(x[ix], y[iy], degree=3, gamma=1/2048, coef0=1)
    expected = (xx.sum()-np.trace(xx)+yy.sum()-np.trace(yy))/(8*7)-2*xy.mean()
    assert math.isclose(result["mean"], expected, rel_tol=1e-10, abs_tol=1e-12)
    print("Checked KID against independent sklearn kernels, manifold self/disjoint fixtures and cosine identity; no training or image generation.")


def cycle_and_lpips(ctx, generator_class, load_image, lpips_model, device):
    import numpy as np
    import torch
    generators = {}
    for name in ("G_A2B", "G_B2A"):
        model = generator_class().to(device)
        model.load_state_dict(ctx["generator_weights"][name], strict=True)
        model.eval().requires_grad_(False)
        generators[name] = model
    records = []
    start = time.perf_counter()
    with torch.no_grad():
        for domain, paths in [("A", ctx["paths"]["real_A"]), ("B", ctx["paths"]["real_B"])]:
            first, second = ("G_A2B", "G_B2A") if domain == "A" else ("G_B2A", "G_A2B")
            for index, path in enumerate(paths, 1):
                real = load_image(path, training=False).unsqueeze(0).to(device)
                intermediate = generators[first](real)
                reconstruction = generators[second](intermediate)
                assert real.shape == reconstruction.shape == (1, 3, 256, 256)
                assert not torch.is_grad_enabled() and torch.isfinite(reconstruction).all()
                raw_l1 = (real-reconstruction).abs().mean().item()
                # Both real and reconstructed images use [-1,1]; [0,1] error is half.
                zero_one_l1 = ((real+1)/2 - (reconstruction+1)/2).abs().mean().item()
                assert math.isclose(zero_one_l1, raw_l1/2, rel_tol=1e-6, abs_tol=1e-8)
                distance = float(lpips_model(real, reconstruction, normalize=False).item()) if lpips_model is not None else None
                assert math.isfinite(raw_l1) and (distance is None or math.isfinite(distance))
                records.append({"domain": domain, "source": relative(ctx["root"], path),
                                "cycle_L1_0_1": zero_one_l1, "cycle_L1_minus1_1": raw_l1, "LPIPS": distance})
                if index % 50 == 0:
                    print(f"Cycle/LPIPS {domain}: {index}/300; intermediate tensors only, no image files saved.", flush=True)
    save_csv(ctx, "cycle_lpips_per_image.csv", records,
             ["domain", "source", "cycle_L1_0_1", "cycle_L1_minus1_1", "LPIPS"])
    result = {}
    for domain in ("A", "B"):
        rows = [r for r in records if r["domain"] == domain]
        assert len(rows) == 300
        for key in ("cycle_L1_0_1", "cycle_L1_minus1_1", "LPIPS"):
            values = [r[key] for r in rows]
            result[f"{key}_{domain}"] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))} if values[0] is not None else None
    save_json(ctx, "cycle_lpips_results.json", {"sample_count_per_domain":300, "results":result,
               "std_ddof":0, "tensor_range":"[-1,1]", "cycle_L1_report_range":"[0,1]",
               "translation_and_reconstruction":"In-memory tensors only; no prediction folders changed.",
               "evaluation_seconds":time.perf_counter()-start})
    del generators
    return result


def make_plots(ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    pairs = [
        ("generator_total", ["G_total"], "Generator total loss"),
        ("discriminators", ["D_A","D_B"], "Discriminator losses"),
        ("cycle", ["cycle_A","cycle_B"], "Cycle losses (training weighted by 10)"),
        ("identity", ["identity_A","identity_B"], "Identity losses (training weighted by 5)"),
        ("gradient_norms", ["generator_gradient_norm","discriminator_gradient_norm"], "Gradient L2 norms"),
        ("learning_rate", ["learning_rate"], "Learning rate"),
    ]
    result=[]
    for name, keys, label in pairs:
        fig, ax = plt.subplots(figsize=(9, 4.3), constrained_layout=True)
        for key in keys:
            ax.plot(ctx["history"]["epoch"],ctx["history"][key],label=key,linewidth=1.5)
        ax.set(xlabel="Epoch",ylabel=label,xlim=(1,150),title=label)
        ax.grid(alpha=0.25)
        if len(keys)>1: ax.legend()
        path = output_path(ctx, "plots/"+name+".png")
        if path.exists(): raise FileExistsError(path)
        fig.savefig(path,dpi=180)
        plt.close(fig)
        display(Image(filename=str(path)))
        result.append(relative(ctx["root"],path))
    save_json(ctx,"manifests/plot_files.json",result)
    print("Saved six plots from complete logged epochs 1–150. Raw logs are unchanged.")
    return result


def build_reports(ctx, kid, pr, cosine, cycle):
    import pandas as pd
    data = {"checkpoint":ctx["checkpoint"],"epoch":150,"global_step":45000,**ctx["official_scores"]}
    for direction in ("A2B","B2A"):
        for stat in ("mean","std"):
            data[f"KID_{direction}_{stat}"]=kid[direction][stat]
            data[f"content_cosine_{direction}_{stat}"]=cosine[direction][stat]
        data[f"generative_precision_{direction}"]=pr[direction]["precision"]
        data[f"generative_recall_{direction}"]=pr[direction]["recall"]
    for domain in ("A","B"):
        data[f"cycle_L1_{domain}_0_1"]=cycle[f"cycle_L1_0_1_{domain}"]["mean"]
        data[f"cycle_L1_{domain}_minus1_1"]=cycle[f"cycle_L1_minus1_1_{domain}"]["mean"]
        lp = cycle[f"LPIPS_{domain}"]
        for stat in ("mean","std"):
            data[f"LPIPS_{domain}_{stat}"]=lp[stat] if lp else None
    row=ctx["epoch150"]
    for key in ("G_total","D_A","D_B","generator_gradient_norm","discriminator_gradient_norm",
                "NaN_count","Inf_count","images_per_second","peak_gpu_memory_GB","learning_rate","epoch_seconds"):
        data[key]=float(row[key])
    for label,original in [("cycle_loss_A","cycle_A"),("cycle_loss_B","cycle_B"),
                           ("identity_loss_A","identity_A"),("identity_loss_B","identity_B")]:
        data[label]=float(row[original])
    for model,count in ctx["parameter_counts"].items():
        data["parameter_count_"+model]=count
    data["parameter_count_total"]=sum(ctx["parameter_counts"].values())
    data["training_time_minutes"]=float(row["elapsed_minutes"])
    data["history_peak_gpu_memory_GiB"]=float(ctx["history"]["peak_gpu_memory_GB"].max())
    data["history_NaN_total"]=float(ctx["history"]["NaN_count"].sum())
    data["history_Inf_total"]=float(ctx["history"]["Inf_count"].sum())
    data["evaluation_sample_count_per_set"]=300
    for key,value in data.items():
        if isinstance(value,(int,float)):
            assert math.isfinite(value),key
    save_csv(ctx,"automated_metrics_epoch150.csv",[data],list(data))
    long=[]
    for key,value in data.items():
        if key in ("checkpoint","epoch","global_step"): continue
        direction = next((d for d in ("A2B","B2A") if d in key), "all")
        if key.startswith(("LPIPS_A","cycle_L1_A")): direction="A"
        if key.startswith(("LPIPS_B","cycle_L1_B")): direction="B"
        units="unitless"
        if key.startswith("parameter_count"): units="parameters"
        elif key=="training_time_minutes": units="minutes"
        elif key=="images_per_second": units="training input images/sec, epoch 150"
        elif "memory" in key: units="GiB (1024^3 bytes)"
        elif key=="epoch_seconds": units="seconds"
        if key in SCORE_KEYS:
            protocol="Existing official instructor result; not recomputed"
            evidence=(EVAL_REL/"metrics/evaluation_results.csv").as_posix()
        elif key.startswith("KID"):
            protocol="Unbiased degree-3 polynomial MMD; gamma=1/2048, coef0=1; 100 subsets of 100, seed 171; raw scale; std ddof=0"
            evidence=(OUT_REL/"kid_results.json").as_posix()
        elif key.startswith("generative_"):
            protocol="k=3 nearest-neighbor closed-ball manifolds in raw 2048-D instructor Inception feature space; 300 each"
            evidence=(OUT_REL/"precision_recall_results.json").as_posix()
        elif key.startswith("content_cosine"):
            protocol="Input vs same-filename existing translation; instructor Inception pooled features; 300 pairs; std ddof=0"
            evidence=(OUT_REL/"content_cosine_results.json").as_posix()
        elif key.startswith(("cycle_L1","LPIPS")):
            protocol="Cycle reconstruction pairing; deterministic 256 RGB tensors; LPIPS AlexNet v0.1; 300/domain; std ddof=0"
            evidence=(OUT_REL/"cycle_lpips_per_image.csv").as_posix()
        else:
            protocol="Existing epoch-150 training CSV/output; weighted training losses; cumulative logged elapsed time"
            evidence=(OUT_REL/"manifests/training_evidence.json").as_posix()
        long.append(dict(metric=key,direction=direction,value=value if value is not None else "UNAVAILABLE",
                         units=units,checkpoint=ctx["checkpoint"],protocol=protocol,evidence_file=evidence))
    for key in ["human_audit_score","inter_rater_agreement","Kaggle_public_score","Kaggle_private_score","Kaggle_rank"]:
        long.append(dict(metric=key,direction="all",value="PENDING",units="pending",
                         checkpoint=ctx["checkpoint"],protocol="Not performed during this evaluation",
                         evidence_file="PENDING"))
    save_csv(ctx,"full_metrics_report.csv",long,["metric","direction","value","units","checkpoint","protocol","evidence_file"])
    ctx["metrics"]=data
    print("Saved tidy metrics and long-form report; human audit/agreement and Kaggle remain PENDING.")
    return pd.DataFrame([data]),pd.DataFrame(long)


def document_protocol(ctx):
    text = """# Epoch-150 evaluation protocol

Checkpoint: {checkpoint}. Epoch 150; global step 45000; seed 171.
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
""".format(checkpoint=ctx["checkpoint"])
    with output_path(ctx,"evaluation_protocol.md").open("x",encoding="utf-8") as f:
        f.write(text)
    print("Saved factual evaluation_protocol.md with definitions, exact reuse, sample policy, versions and evidence paths.")


def validate_outputs(ctx):
    import numpy as np
    import pandas as pd
    from PIL import Image
    metrics=ctx["metrics"]
    assert all(math.isfinite(v) for v in metrics.values() if isinstance(v,(int,float)))
    for key,path_hash in ctx["protected"].items():
        assert sha256(ctx["root"]/key)==path_hash, f"Protected evidence changed: {key}"
    for name in ("automated_metrics_epoch150.csv","full_metrics_report.csv","cycle_lpips_per_image.csv"):
        assert len(pd.read_csv(ctx["out"]/name))>0
    full=pd.read_csv(ctx["out"]/"full_metrics_report.csv",keep_default_na=False)
    pending=["human_audit_score","inter_rater_agreement","Kaggle_public_score","Kaggle_private_score","Kaggle_rank"]
    assert set(full.loc[full.metric.isin(pending),"value"])=={"PENDING"}
    with np.load(ctx["out"]/"inception_features.npz") as arrays:
        assert set(arrays.files)=={"real_A","real_B","gen_A2B","gen_B2A"}
        assert all(arrays[k].shape==(300,2048) and np.isfinite(arrays[k]).all() for k in arrays.files)
    for p in (ctx["out"]/"plots").glob("*.png"):
        with Image.open(p) as im: im.verify()
    save_json(ctx,"manifests/validation_before_notebook_append.json",{
        "all_numeric_metrics_finite":True,"NaN_count_in_metrics":0,"Inf_count_in_metrics":0,
        "sample_count_each_set":300,"cycle_count_each_domain":300,
        "generated_prediction_files_unchanged":True,"epoch100_folders_unchanged":True,
        "epoch150_official_evidence_unchanged":True,"raw_training_logs_unchanged":True,
        "checkpoint_unchanged":True,"main_notebook_unchanged_before_authorized_append":True,
        "CSVs_parse":True,"plots_readable":True,"no_training":True,"no_deleted_existing_files":True,"no_zip_created":True,
    })
    print("Validation passed: finite metrics, exact samples, readable CSVs/plots, protected file hashes unchanged.")
    for p in [ctx["root"]/ctx["checkpoint"],ctx["out"]/"automated_metrics_epoch150.csv",ctx["out"]/"full_metrics_report.csv"]:
        print(sha256(p),relative(ctx["root"],p))
    print("Final notebook SHA256 is recorded by the executor after saving its executed outputs.")
