import sys
from pathlib import Path
from source.figure_plot_helper import collect_metrics
import matplotlib.pyplot as plt

# read command line argument
if len(sys.argv) < 2:
    print("Usage: python plot_figure2a.py <dataset>")
    sys.exit(1)

dataset = sys.argv[1].lower()  # 'planar' or 'qm9'

# paths
repo_root = Path(__file__).resolve().parents[1]
outputs_dir = repo_root / "outputs"
results_dir = repo_root / "results" / "plots"
results_dir.mkdir(exist_ok=True, parents=True)

# configurations and steps
configs = ["vanilla", "+distortion", "++target_guidance", "+++stochasticity"]
num_steps_list = [5, 10, 50, 100, 1000]

# collect results
results_dict = {}
for cfg in configs:
    for steps in num_steps_list:
        exp_prefix = f"2_a_{cfg}_{dataset}" if dataset=="planar" else f"2_a_woh_{cfg}_{dataset}"
        _, values = collect_metrics(outputs_dir, exp_prefix, steps, dataset=dataset)
        results_dict.setdefault(cfg, []).append(values)

# plotting
plt.figure(figsize=(8,5))
x_positions = range(len(num_steps_list))
for cfg in configs:
    cleaned = []
    for v in results_dict[cfg]:
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                cleaned.append(float("nan"))
            else:
                cleaned.append(float(v[0]))
        elif isinstance(v, (int, float)):
            cleaned.append(float(v))
        else:
            cleaned.append(float("nan"))
    plt.plot(x_positions, cleaned, marker="o", label=cfg)


plt.xticks(x_positions, num_steps_list)
plt.xlabel("Number of Sampling Steps")
plt.ylabel("VUN" if dataset=="planar" else "Validity")
plt.title(f"Figure 2a – {dataset.capitalize()} Dataset")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

out_path = results_dir / f"figure2a_{dataset}.png"
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Figure saved at {out_path}")
