"""Official training summaries and descriptive plots; no scientific conclusions."""
import statistics
import time
from pathlib import Path
from config import ROOT
from utils import write_json


def finalize_report(rows, gradients, seconds, output, stamp, first, last, csv_path, raw_path, manifest_root):
    reporting_begin = time.perf_counter()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["G_A2B_adv", "G_B2A_adv", "cycle_A", "cycle_B", "identity_A", "identity_B", "G_total", "D_A", "D_B"]
    best = {key: dict(epoch=min(rows, key=lambda r:r[key])["epoch"],
                      value=min(r[key] for r in rows)) for key in keys}
    norms = {key: dict(mean=statistics.fmean(pair[i] for pair in gradients),
                      median=statistics.median(pair[i] for pair in gradients),
                      min=min(pair[i] for pair in gradients), max=max(pair[i] for pair in gradients),
                      std=statistics.pstdev(pair[i] for pair in gradients))
             for i, key in enumerate(("generator", "discriminator"))}
    summary = dict(run_kind="official", start_epoch=first, stop_epoch=last, completed_epochs=len(rows),
                   completed_steps=len(gradients), total_wall_clock_seconds=seconds,
                   lowest_training_losses=best, final_epoch_metrics=rows[-1], gradient_norm_summary=norms,
                   NaN_total=int(sum(r["NaN_count"] for r in rows)), Inf_total=int(sum(r["Inf_count"] for r in rows)),
                   peak_vram_GiB=max(r["peak_gpu_memory_GB"] for r in rows),
                   average_images_per_second=600*len(rows)/sum(r["epoch_seconds"] for r in rows),
                   throughput_definition="Both real domains; total training inputs / sum of epoch training seconds",
                   epoch_050_path=(output/"checkpoints/epoch_050.pt").relative_to(ROOT).as_posix(),
                   epoch_100_path=(output/"checkpoints/epoch_100.pt").relative_to(ROOT).as_posix(),
                   artifacts=dict(epoch_csv=csv_path.relative_to(ROOT).as_posix(),
                                  raw_log=raw_path.relative_to(ROOT).as_posix(),
                                  manifests=manifest_root.relative_to(ROOT).as_posix()),
                   pilot_used=False, automatic_continuation=False)
    plot_dir=output/"outputs/plots"/stamp
    plot_dir.mkdir(parents=True, exist_ok=False)
    x=[r["epoch"] for r in rows]
    groups=[("generator_losses", ["G_total", "G_A2B_adv", "G_B2A_adv"]),
            ("cycle_identity_losses", ["cycle_A", "cycle_B", "identity_A", "identity_B"]),
            ("discriminator_losses", ["D_A", "D_B"]),
            ("gradient_norms", ["generator_gradient_norm", "discriminator_gradient_norm"]),
            ("learning_rate", ["learning_rate"]),
            ("throughput", ["images_per_second"]),
            ("peak_gpu_memory", ["peak_gpu_memory_GB"])]
    for name, group in groups:
        fig, ax=plt.subplots(figsize=(9,5))
        for key in group:
            ax.plot(x,[r[key] for r in rows],label=key)
        ax.set_xlabel("Official epoch")
        ax.set_ylabel({"learning_rate":"Learning rate", "throughput":"Real images / second",
                       "peak_gpu_memory":"Peak allocated GPU memory (GiB)"}.get(name,"Epoch mean"))
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(plot_dir/(name+".png"),dpi=160)
        plt.close(fig)
    seconds += time.perf_counter() - reporting_begin
    summary["total_wall_clock_seconds"] = seconds
    write_json(output/f"outputs/metrics/summary_{stamp}.json",summary)
    lines=[f"Official training completed epochs {first} through {last}; {len(gradients)} steps.",
           f"Total wall-clock time including final checkpoint and plots: {seconds:.3f} s ({seconds/3600:.3f} hours)",
           "Lowest observed training losses by epoch (descriptive; no checkpoint ranking):"]
    lines += [f"  {key}: {value['value']:.6f} at epoch {value['epoch']}" for key,value in best.items()]
    lines += ["Final epoch losses:"]+[f"  {key}: {rows[-1][key]:.6f}" for key in keys]
    lines += [f"Gradient norm summary: {norms}",f"NaN total: {summary['NaN_total']}; Inf total: {summary['Inf_total']}",
              f"Peak allocated VRAM: {summary['peak_vram_GiB']:.6f} GiB",
              f"Average real images/sec: {summary['average_images_per_second']:.6f}",
              f"Epoch 50 checkpoint: {ROOT/summary['epoch_050_path']}",
              f"Epoch 100 checkpoint: {ROOT/summary['epoch_100_path']}",
              "Stopped cleanly. No next epoch or submission was started."]
    text="\n".join(lines)
    (output/f"outputs/metrics/summary_{stamp}.txt").write_text(text+"\n",encoding="utf-8")
    print(text,flush=True)
    return summary
