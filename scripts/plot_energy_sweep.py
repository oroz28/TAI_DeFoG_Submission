import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
csv_path = repo_root / "results" / "tables" / "energy_sweep_results_with_h.csv"
df = pd.read_csv(csv_path)

# extract mean and std from string columns
df["energy_mean"] = df["energy_mean"].str.extract(r'\(([-0-9\.eE]+),')[0].astype(float)

# for std extract the first value
df["energy_std"] = df["energy_std"].str.extract(r'\(([-0-9\.eE]+),')[0].astype(float)


# sort by condition_value descending
df = df.sort_values("condition_value", ascending=False)

cond_vals = df["condition_value"].tolist()
means = df["energy_mean"].tolist()
stds = df["energy_std"].tolist()

print("Condition Values:", cond_vals)
print("Means:", means)
print("Stds:", stds)

plt.figure(figsize=(8, 5))
x_positions = np.arange(len(cond_vals))

plt.errorbar(
    x_positions,
    means,
    yerr=stds,
    marker="o",
    capsize=3,
    label="Energy"
)

plt.xticks(x_positions, [f"{v:.0f}" for v in cond_vals])
plt.xlabel("Condition Value (Energy)")
plt.ylabel("Generated Molecule Energy")
plt.title("Energy vs Condition Value (num_steps=100)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# save figure
out_path = repo_root / "results" / "plots" / "figure_energy_sweep_energy_with_h.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Figure saved at {out_path}")

