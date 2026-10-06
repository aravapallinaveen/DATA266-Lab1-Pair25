"""Run --smoke-test before explicitly starting official training."""
import argparse
import csv
from datetime import datetime, timezone
import math
import random
import tempfile
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config import CONFIG, ROOT, epoch_lr
from dataset import EpochDataset, image_paths, load_image
from models import build_models
from losses import generator_losses, discriminator_loss
from utils import (seed_all, write_json, environment, save_image, save_checkpoint,
                   load_checkpoint, restore_random, gradient_stats)

LOSS_KEYS = ["G_A2B_adv", "G_B2A_adv", "cycle_A", "cycle_B", "identity_A", "identity_B",
             "G_total", "D_A", "D_B"]
FIELDS = ["epoch", "global_step", "learning_rate"] + LOSS_KEYS + [
    "generator_gradient_norm", "discriminator_gradient_norm", "NaN_count", "Inf_count",
    "images_per_second", "epoch_seconds", "elapsed_minutes", "peak_gpu_memory_GB"]

def setup(device):
    models = build_models(device)
    gp = list(models["G_A2B"].parameters()) + list(models["G_B2A"].parameters())
    dp = list(models["D_A"].parameters()) + list(models["D_B"].parameters())
    optimizers = {"G": torch.optim.Adam(gp, lr=CONFIG["learning_rate"], betas=CONFIG["betas"]),
                  "D_A": torch.optim.Adam(models["D_A"].parameters(), lr=CONFIG["learning_rate"], betas=CONFIG["betas"]),
                  "D_B": torch.optim.Adam(models["D_B"].parameters(), lr=CONFIG["learning_rate"], betas=CONFIG["betas"])}
    # Scheduler index 0 corresponds to epoch 1. Step after each completed epoch.
    schedulers = {k: torch.optim.lr_scheduler.LambdaLR(o, lambda i:
                  1.0 if i < 100 else max(0., (199-i)/100)) for k, o in optimizers.items()}
    return models, optimizers, schedulers, gp, dp

class NonfiniteTrainingError(FloatingPointError):
    def __init__(self, phase, losses, tensors, nan, inf, generator_updated=False):
        super().__init__(f"Nonfinite training values during {phase}: NaN={nan}, Inf={inf}")
        self.diagnostics = dict(phase=phase, losses=losses, NaN_count=nan, Inf_count=inf,
                                generator_updated=generator_updated,
                                tensors={k: v.detach().cpu() for k, v in tensors.items()})


def train_step(models, optimizers, gp, dp, a, b):
    for p in dp:
        p.requires_grad_(False)
    optimizers["G"].zero_grad(set_to_none=True)
    losses, tensors = generator_losses(models, a, b)
    scalars = {k: float(v.detach()) for k, v in losses.items()}
    nan_l = sum(math.isnan(v) for v in scalars.values())
    inf_l = sum(math.isinf(v) for v in scalars.values())
    if nan_l or inf_l:
        raise NonfiniteTrainingError("generator losses", scalars, tensors, nan_l, inf_l)
    losses["G_total"].backward()
    gn, nan_g, inf_g = gradient_stats(gp)
    if nan_g or inf_g or not math.isfinite(gn):
        raise NonfiniteTrainingError("generator gradients", scalars, tensors,
                                     nan_g+int(math.isnan(gn)), inf_g+int(math.isinf(gn)))
    optimizers["G"].step()
    for p in dp:
        p.requires_grad_(True)
    for domain in ("A", "B"):
        key = "D_" + domain
        optimizers[key].zero_grad(set_to_none=True)
        losses[key] = discriminator_loss(models[key], tensors["real_"+domain], tensors["fake_"+domain])
        value = float(losses[key].detach())
        scalars[key] = value
        if not math.isfinite(value):
            raise NonfiniteTrainingError(key+" loss", scalars, tensors,
                                         int(math.isnan(value)), int(math.isinf(value)), True)
        losses[key].backward()
    dn, nan_d, inf_d = gradient_stats(dp)
    if nan_d or inf_d or not math.isfinite(dn):
        raise NonfiniteTrainingError("discriminator gradients", scalars, tensors,
                                     nan_d+int(math.isnan(dn)), inf_d+int(math.isinf(dn)), True)
    optimizers["D_A"].step()
    optimizers["D_B"].step()
    scalars.update(generator_gradient_norm=gn, discriminator_gradient_norm=dn,
                   NaN_count=nan_g+nan_d+nan_l, Inf_count=inf_g+inf_d+inf_l)
    return scalars, tensors

@torch.no_grad()
def fixed_outputs(models, fixed, folder, device):
    for name in ("G_A2B", "G_B2A"):
        models[name].eval()
    for domain, direction in (("A", "A2B"), ("B", "B2A")):
        for identifier in fixed[domain]:
            p = ROOT / identifier
            image = load_image(p).unsqueeze(0).to(device)
            save_image(models["G_"+direction](image), folder / direction / (p.stem+".png"))
    for model in models.values():
        model.train()

def smoke(device, paths_a, paths_b):
    audit = ROOT / "naveen/outputs/audit"
    audit.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="smoke_", dir=audit))
    write_json(folder / "environment.json", environment(device))
    models, optimizers, schedulers, gp, dp = setup(device)
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    print("Parameter counts:", counts, flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dataset = EpochDataset(paths_a[:2], paths_b[:2], tiny=True)
    results, shapes = [], {}
    for step, (a, b) in enumerate(DataLoader(dataset, batch_size=1, num_workers=0), 1):
        values, tensors = train_step(models, optimizers, gp, dp, a.to(device), b.to(device))
        shapes = {k: list(v.shape) for k, v in tensors.items()}
        with torch.no_grad():
            shapes["D_A"] = list(models["D_A"](tensors["real_A"]).shape)
            shapes["D_B"] = list(models["D_B"](tensors["real_B"]).shape)
        assert all(s == [1, 3, 256, 256] for k, s in shapes.items() if not k.startswith("D_"))
        assert shapes["D_A"] == shapes["D_B"] == [1, 1, 30, 30]
        results.append(values)
        print(f"Step {step}: {values}", flush=True)
        if step == 1:
            save_image(tensors["fake_A"], folder / "temporary_B2A.png")
            save_image(tensors["fake_B"], folder / "temporary_A2B.png")
    assert len(results) == 2
    print("Shapes:", shapes, flush=True)
    for scheduler in schedulers.values():
        scheduler.step()
    save_checkpoint(folder / "temporary.pt", models, optimizers, schedulers, 1, 2, {}, 0.)
    expected_rng = (random.random(), torch.rand(3))
    checkpoint = load_checkpoint(folder / "temporary.pt")
    for k, model in models.items():
        with torch.no_grad():
            next(model.parameters()).add_(1.)
        model.load_state_dict(checkpoint["models"][k])
        assert all(torch.equal(v.detach().cpu(), checkpoint["models"][k][n]) for n, v in model.state_dict().items())
    for k, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint["optimizers"][k])
        schedulers[k].load_state_dict(checkpoint["schedulers"][k])
    restore_random(checkpoint["random_states"])
    assert random.random() == expected_rng[0] and torch.equal(torch.rand(3), expected_rng[1])
    assert checkpoint["epoch"] == 1 and checkpoint["global_step"] == 2
    peak = torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else 0.
    report = dict(training_steps=2, shapes=shapes, parameter_counts=counts, losses=results,
                  peak_gpu_memory_GB=peak, checkpoint_reload_succeeded=True)
    write_json(folder / "smoke_report.json", report)
    print(f"Checkpoint reload succeeded; peak GPU allocated memory: {peak:.3f} GiB", flush=True)
    print("Smoke report:", folder / "smoke_report.json", flush=True)

def official(device, paths_a, paths_b, resume, stop_epoch=100):
    from reporting import finalize_report
    import hashlib
    import traceback
    begin = time.perf_counter()
    output = ROOT / "naveen"
    latest = output / "checkpoints/latest.pt"
    if not 1 <= stop_epoch <= CONFIG["epochs"]:
        raise ValueError("Stop epoch outside configured schedule")
    if not resume:
        existing = (list((output / "checkpoints").glob("*.pt")) +
                    list((output / "logs").glob("training_*")) +
                    list((output / "outputs/fixed_samples").glob("epoch_*")))
        if existing:
            raise FileExistsError("Official artifacts already exist. Use --resume; existing artifacts are preserved.")
    models, optimizers, schedulers, gp, dp = setup(device)
    start_epoch, global_step, elapsed_before = 1, 0, 0.
    rng = random.Random(CONFIG["seed"])
    fixed = {d: [p.relative_to(ROOT).as_posix() for p in rng.sample(paths, CONFIG["fixed_sample_count"])]
             for d, paths in (("A", paths_a), ("B", paths_b))}
    if resume:
        if "pilot" in resume.resolve().parts:
            raise ValueError("Pilot artifacts cannot be used for official training")
        ck = load_checkpoint(resume)
        if ck.get("run_kind", "official") != "official" or ck.get("diagnostic_only"):
            raise ValueError("Only official complete-epoch checkpoints may resume")
        if ck["configuration"] != CONFIG:
            raise ValueError("Resume configuration differs from locked configuration")
        if ck["global_step"] != ck["epoch"]*300:
            raise ValueError("Not an official complete-epoch checkpoint")
        for k, m in models.items():
            m.load_state_dict(ck["models"][k])
        for k, o in optimizers.items():
            o.load_state_dict(ck["optimizers"][k])
            schedulers[k].load_state_dict(ck["schedulers"][k])
            if ck.get("scheduler_pending_step", False):
                o._opt_called = True
                schedulers[k].step()
        restore_random(ck["random_states"])
        start_epoch, global_step = ck["epoch"]+1, ck["global_step"]
        fixed, elapsed_before = ck["fixed_samples"], ck["elapsed_seconds"]
    if start_epoch > stop_epoch:
        print(f"Checkpoint already reached stop epoch {stop_epoch}; no training started.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest_root = output / "reproducibility/manifests" / stamp
    env = environment(device)
    write_json(manifest_root / "environment.json", env)
    write_json(manifest_root / "fixed_samples.json", fixed)
    write_json(manifest_root / "dataset.json", {d: [p.relative_to(ROOT).as_posix() for p in paths]
               for d, paths in (("A", paths_a), ("B", paths_b))})
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    write_json(manifest_root / "source_hashes.json", sources)
    write_json(manifest_root / "run_plan.json", dict(run_kind="official", start_epoch=start_epoch,
               stop_epoch=stop_epoch, steps_per_epoch=300, configuration=CONFIG, pilot_used=False))
    csv_path = logs / f"training_{stamp}.csv"
    raw_path = logs / f"training_{stamp}.log"
    steps_path = logs / f"training_{stamp}_steps.csv"
    gradients, rows = [], []
    status_path = output / f"outputs/metrics/status_{stamp}.json"
    write_json(status_path, dict(status="running", run_kind="official", epoch=0, global_step=global_step,
               start_epoch=start_epoch, stop_epoch=stop_epoch, csv=csv_path.relative_to(ROOT).as_posix()))
    with csv_path.open("x", newline="", encoding="utf-8") as cf, \
         raw_path.open("x", encoding="utf-8", buffering=1) as raw, \
         steps_path.open("x", newline="", encoding="utf-8") as sf:
        writer = csv.DictWriter(cf, fieldnames=FIELDS)
        writer.writeheader()
        step_writer = csv.DictWriter(sf, fieldnames=["epoch", "step", "global_step", "learning_rate"] +
                                     LOSS_KEYS + ["generator_gradient_norm", "discriminator_gradient_norm", "NaN_count", "Inf_count"])
        step_writer.writeheader()
        raw.write(f"OFFICIAL start_epoch={start_epoch} stop_epoch={stop_epoch}\nConfiguration: {CONFIG}\nResume: {resume}\n")
        raw.write(f"Environment: {env}\nFixed samples: {fixed}\n")
        print(f"OFFICIAL fresh={not bool(resume)} epochs={start_epoch}..{stop_epoch}; 300 steps/epoch", flush=True)
        print(f"Raw log: {raw_path}\nEpoch CSV: {csv_path}", flush=True)
        epoch, index = start_epoch, 0
        try:
            for epoch in range(start_epoch, stop_epoch+1):
                index = 0
                assert all(abs(o.param_groups[0]["lr"]-epoch_lr(epoch)) < 1e-12 for o in optimizers.values())
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                epoch_begin = time.perf_counter()
                dataset = EpochDataset(paths_a, paths_b)
                assert len(dataset) == 300 and sorted(a for a, _ in dataset.pairs) == list(range(300))
                write_json(manifest_root / f"epoch_{epoch:03d}_pairs.json", dataset.manifest())
                sums = {k: 0. for k in LOSS_KEYS + ["generator_gradient_norm", "discriminator_gradient_norm", "NaN_count", "Inf_count"]}
                lr = optimizers["G"].param_groups[0]["lr"]
                for index, (a, b) in enumerate(DataLoader(dataset, batch_size=1, num_workers=0), 1):
                    values, tensors = train_step(models, optimizers, gp, dp, a.to(device), b.to(device))
                    del tensors
                    global_step += 1
                    for k, v in values.items():
                        sums[k] += v
                    gradients.append((values["generator_gradient_norm"], values["discriminator_gradient_norm"]))
                    step_writer.writerow(dict(epoch=epoch, step=index, global_step=global_step, learning_rate=lr, **values))
                    raw.write(f"epoch={epoch} step={index} global_step={global_step} learning_rate={lr} losses={values}\n")
                    if index % 50 == 0:
                        sf.flush()
                        print(f"Epoch {epoch}/{stop_epoch} step {index}/300 G={values['G_total']:.4f}", flush=True)
                assert index == 300 and global_step == epoch*300
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter()-epoch_begin
                elapsed = elapsed_before + time.perf_counter()-begin
                row = {k: v if k in ("NaN_count", "Inf_count") else v/300 for k, v in sums.items()}
                row.update(epoch=epoch, global_step=global_step, learning_rate=lr,
                           images_per_second=600/seconds, epoch_seconds=seconds, elapsed_minutes=elapsed/60,
                           peak_gpu_memory_GB=torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else 0.)
                writer.writerow(row)
                cf.flush()
                sf.flush()
                rows.append(row)
                raw.write(f"EPOCH SUMMARY {row}\n")
                if epoch % 5 == 0:
                    fixed_outputs(models, fixed, output / f"outputs/fixed_samples/epoch_{epoch:03d}", device)
                # Preserve current LR in the final checkpoint. Resume explicitly advances once.
                pending = epoch == stop_epoch
                if not pending:
                    for scheduler in schedulers.values():
                        scheduler.step()
                elapsed = elapsed_before + time.perf_counter()-begin
                metadata = dict(run_kind="official", scheduler_pending_step=pending, stop_epoch=stop_epoch)
                save_checkpoint(latest, models, optimizers, schedulers, epoch, global_step, fixed, elapsed, metadata=metadata)
                if epoch % 10 == 0:
                    permanent = output / f"checkpoints/epoch_{epoch:03d}.pt"
                    if permanent.exists():
                        raise FileExistsError(f"Permanent checkpoint already exists: {permanent}")
                    save_checkpoint(permanent, models, optimizers, schedulers, epoch, global_step,
                                    fixed, elapsed, metadata=metadata)
                write_json(status_path, dict(status="running", run_kind="official", epoch=epoch,
                           global_step=global_step, stop_epoch=stop_epoch, final_epoch_metrics=row,
                           elapsed_seconds=time.perf_counter()-begin))
                print(f"Completed epoch {epoch}; LR={lr:.8f}; G={row['G_total']:.4f}; "
                      f"D_A={row['D_A']:.4f}; D_B={row['D_B']:.4f}; "
                      f"NaN={int(row['NaN_count'])}; Inf={int(row['Inf_count'])}", flush=True)
            summary = finalize_report(rows, gradients, time.perf_counter()-begin, output, stamp,
                                      start_epoch, stop_epoch, csv_path, raw_path, manifest_root)
            raw.write(f"COMPLETION SUMMARY {summary}\n")
            write_json(status_path, dict(status="completed", epoch=stop_epoch, global_step=global_step,
                       stop_epoch=stop_epoch, summary=f"naveen/outputs/metrics/summary_{stamp}.json"))
        except BaseException as error:
            raw.write(f"STOPPED epoch={epoch} step={index} global_step={global_step}: {error!r}\n")
            raw.write(traceback.format_exc())
            cf.flush()
            sf.flush()
            diagnostic_path = output / f"checkpoints/diagnostic_{stamp}_epoch_{epoch:03d}_step_{index:03d}.pt"
            diagnostics = getattr(error, "diagnostics", {})
            metadata = dict(run_kind="official", diagnostic_only=True, failed_epoch=epoch,
                            failed_step=index, error=repr(error), diagnostics=diagnostics,
                            gradients={k: {n: p.grad.detach().cpu() for n, p in m.named_parameters() if p.grad is not None}
                                       for k, m in models.items()})
            save_checkpoint(diagnostic_path, models, optimizers, schedulers, epoch-1,
                            global_step, fixed, elapsed_before+time.perf_counter()-begin, metadata=metadata)
            write_json(status_path, dict(status="stopped", epoch=epoch, step=index,
                       global_step=global_step, error=repr(error),
                       NaN_count=diagnostics.get("NaN_count", 0), Inf_count=diagnostics.get("Inf_count", 0),
                       diagnostic=diagnostic_path.relative_to(ROOT).as_posix(), automatic_resume=False))
            raw.write(f"Diagnostic saved: {diagnostic_path}\nNo automatic continuation.\n")
            print(f"TRAINING STOPPED. Diagnostic: {diagnostic_path}. No automatic continuation.", flush=True)
            raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-epoch", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.smoke_test and args.resume:
        parser.error("Smoke test cannot resume official training")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu explicitly")
    seed_all(CONFIG["seed"])
    device = torch.device(args.device)
    paths_a, paths_b = image_paths("A"), image_paths("B")
    if args.smoke_test:
        smoke(device, paths_a, paths_b)
    else:
        official(device, paths_a, paths_b, args.resume, args.stop_epoch)

if __name__ == "__main__":
    main()
