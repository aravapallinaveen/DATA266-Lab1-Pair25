"""Train Praveen SID 8511 CycleGAN."""
import argparse, json, time
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import UnpairedDataset
from models import Generator, Discriminator, count_parameters
from utils import seed_everything, save_grid, append_jsonl, write_metrics_template


class ReplayBuffer:
    def __init__(self, max_size=50): self.max_size, self.data = max_size, []
    def push_and_pop(self, images):
        result = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(image); result.append(image)
            elif torch.rand(1).item() > 0.5:
                i = torch.randint(0, self.max_size, (1,)).item()
                old = self.data[i].clone(); self.data[i] = image; result.append(old)
            else: result.append(image)
        return torch.cat(result)


def save_checkpoint(path, epoch, models, opts, schedulers, history):
    torch.save({"epoch": epoch, "models": {k: v.state_dict() for k, v in models.items()}, "optimizers": {k: v.state_dict() for k, v in opts.items()}, "schedulers": {k: v.state_dict() for k, v in schedulers.items()}, "history": history}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--synthetic-smoke-data", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.data_root: cfg["data_root"] = args.data_root
    if args.output_root: cfg["output_root"] = args.output_root
    if args.smoke_test:
        cfg.update({"epochs": 1, "updates_per_epoch": 2, "batch_size": 1, "num_workers": 0, "save_every_epochs": 1, "sample_every_epochs": 1})
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    seed_everything(cfg["seed"])
    out = Path(cfg["output_root"]); ckpt_dir, sample_dir, log_dir = out / "checkpoints", out / "samples", out / "logs"
    for p in [ckpt_dir, sample_dir, log_dir]: p.mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))
    write_metrics_template(out / "metrics_report.csv")
    ds = UnpairedDataset(cfg["data_root"], cfg["image_size"], cfg["resize_size"], train=True, synthetic=args.synthetic_smoke_data)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=device.type == "cuda", drop_last=True)
    G_A2B = Generator(base=cfg["base_channels"], n_blocks=cfg["generator_res_blocks"]).to(device)
    G_B2A = Generator(base=cfg["base_channels"], n_blocks=cfg["generator_res_blocks"]).to(device)
    D_A = Discriminator(base=cfg["base_channels"], use_spectral_norm=cfg["use_spectral_norm"]).to(device)
    D_B = Discriminator(base=cfg["base_channels"], use_spectral_norm=cfg["use_spectral_norm"]).to(device)
    models = {"G_A2B": G_A2B, "G_B2A": G_B2A, "D_A": D_A, "D_B": D_B}
    opts = {"G": optim.AdamW(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"])), "D": optim.AdamW(list(D_A.parameters()) + list(D_B.parameters()), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"]))}
    schedulers = {k: optim.lr_scheduler.CosineAnnealingLR(v, T_max=max(1, cfg["epochs"] - cfg["constant_lr_epochs"])) for k, v in opts.items()}
    start_epoch, history = 1, []
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        for k, v in models.items(): v.load_state_dict(state["models"][k])
        for k, v in opts.items(): v.load_state_dict(state["optimizers"][k])
        for k, v in schedulers.items(): v.load_state_dict(state["schedulers"][k])
        start_epoch, history = state["epoch"] + 1, state.get("history", [])
    gan = nn.BCEWithLogitsLoss(); l1 = nn.L1Loss()
    fake_a_buf, fake_b_buf = ReplayBuffer(cfg["replay_buffer_size"]), ReplayBuffer(cfg["replay_buffer_size"])
    fixed = next(iter(loader)); fixed_a, fixed_b = fixed["A"].to(device), fixed["B"].to(device)
    run_log = log_dir / "train_raw.jsonl"; t0 = time.time()
    print(f"Device: {device}; parameters: " + ", ".join(f"{k}={count_parameters(v):,}" for k, v in models.items()))
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        epoch_sums = {k: 0.0 for k in ["G", "D", "adv", "cycle", "identity"]}; seen = 0
        iterator = iter(loader)
        pbar = tqdm(range(cfg["updates_per_epoch"]), desc=f"epoch {epoch}/{cfg['epochs']}")
        for _ in pbar:
            try: batch = next(iterator)
            except StopIteration: iterator = iter(loader); batch = next(iterator)
            real_a, real_b = batch["A"].to(device), batch["B"].to(device)
            opts["G"].zero_grad(set_to_none=True)
            fake_b, fake_a = G_A2B(real_a), G_B2A(real_b)
            rec_a, rec_b = G_B2A(fake_b), G_A2B(fake_a)
            id_a, id_b = G_B2A(real_a), G_A2B(real_b)
            adv_g = gan(D_B(fake_b), torch.ones_like(D_B(fake_b))) + gan(D_A(fake_a), torch.ones_like(D_A(fake_a)))
            cycle = l1(rec_a, real_a) + l1(rec_b, real_b)
            identity = l1(id_a, real_a) + l1(id_b, real_b)
            loss_g = adv_g + cfg["lambda_cycle"] * cycle + cfg["lambda_identity"] * identity
            loss_g.backward(); opts["G"].step()
            opts["D"].zero_grad(set_to_none=True)
            old_a, old_b = fake_a_buf.push_and_pop(fake_a), fake_b_buf.push_and_pop(fake_b)
            loss_da = (gan(D_A(real_a), torch.ones_like(D_A(real_a))) + gan(D_A(old_a.detach()), torch.zeros_like(D_A(old_a)))) * 0.5
            loss_db = (gan(D_B(real_b), torch.ones_like(D_B(real_b))) + gan(D_B(old_b.detach()), torch.zeros_like(D_B(old_b)))) * 0.5
            loss_d = loss_da + loss_db; loss_d.backward(); opts["D"].step()
            vals = {"G": loss_g.item(), "D": loss_d.item(), "adv": adv_g.item(), "cycle": cycle.item(), "identity": identity.item()}
            for k, v in vals.items(): epoch_sums[k] += v
            seen += 1; pbar.set_postfix(G=f"{vals['G']:.3f}", D=f"{vals['D']:.3f}", cyc=f"{vals['cycle']:.3f}")
        if epoch > cfg["constant_lr_epochs"]:
            for s in schedulers.values(): s.step()
        row = {"epoch": epoch, "updates": seen, "elapsed_sec": time.time() - t0, **{k: v / seen for k, v in epoch_sums.items()}, "lr": opts["G"].param_groups[0]["lr"], "nan_count": 0, "inf_count": 0}
        history.append(row); append_jsonl(run_log, row)
        if epoch % cfg["sample_every_epochs"] == 0:
            with torch.no_grad(): save_grid(torch.cat([fixed_a, G_A2B(fixed_a), fixed_b, G_B2A(fixed_b)]), sample_dir / f"epoch_{epoch:04d}.png", nrow=fixed_a.size(0))
        if epoch % cfg["save_every_epochs"] == 0 or epoch == cfg["epochs"]:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", epoch, models, opts, schedulers, history)
            save_checkpoint(ckpt_dir / "latest.pt", epoch, models, opts, schedulers, history)
    print(f"Finished in {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__": main()

