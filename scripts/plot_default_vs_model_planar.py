import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

from source.metrics.parser import parse_metrics_file, get_vun

args = sys.argv

repo_root = Path(__file__).resolve().parents[1]
outputs_dir = repo_root / "outputs" / f"default_vs_model_{args[1]}"
results_dir = repo_root / "results" / "plots"
results_dir.mkdir(exist_ok=True)

configs = {
    "defog_default": "DeFoG Default Parameters",
    "defog_model": "DeFoG with Learned Model",
}

num_steps_list = np.arange(5, 51, 5).tolist()

vun_results = {cfg: {"steps": [], "mean": [], "std": []} for cfg in configs}

for cfg, label in configs.items():
    for steps in num_steps_list:
        exp_dir = outputs_dir / cfg / f"{args[1]}_{cfg}_{steps}_steps"

        metric_files = list(exp_dir.glob("test_epoch*"))
        if not metric_files:
            print(f" No metrics file found in {exp_dir}")
            continue

        metrics_file = metric_files[0]

        metrics = parse_metrics_file(metrics_file)
        vun_mean, vun_std = get_vun(metrics)

        # convert from % to [0,1] scale
        vun_mean /= 100
        vun_std /= 100

        vun_results[cfg]["steps"].append(steps)
        vun_results[cfg]["mean"].append(vun_mean)
        vun_results[cfg]["std"].append(vun_std)

plt.figure(figsize=(8, 5))

x_positions = np.arange(len(num_steps_list))

for cfg, label in configs.items():
    data = vun_results[cfg]
    if data["steps"]:
        # ensure consistent ordering
        sorted_indices = sorted(range(len(data["steps"])), key=lambda i: data["steps"][i])
        means = [data["mean"][i] for i in sorted_indices]
        stds = [data["std"][i] for i in sorted_indices]

        plt.errorbar(
            x_positions, means, yerr=stds,
            marker="o", capsize=3, label=label
        )

plt.xticks(x_positions, num_steps_list)
plt.yticks([0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.95])
plt.xlabel("Number of Sampling Steps")
plt.ylabel("VUN")
plt.title("Planar Experiment: DeFoG Default vs Learned Model")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

out_path = results_dir / "default_vs_model_planar.png"
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Figure saved at {out_path}")

