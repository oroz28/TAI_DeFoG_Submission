import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = "/home/group-2/Submission_Code/outputs/qm9_con_grid_output/grid_summary_results.csv"

results_dir = Path("/home/group-2/Submission_Code/results/plots/")
out_path = results_dir / 'qm9_cond_validity.png'
results_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

exp_name = "qm9_conditional.ckpt"
df = df[df["exp_name"] == exp_name]

df = df.sort_values("num_steps")

num_steps_list = df["num_steps"].tolist()
means = df["validity_mean"].tolist()
stds = df["validity_std"].tolist()

plt.figure(figsize=(8, 5))

x_positions = np.arange(len(num_steps_list))
all_values = means.copy()

plt.errorbar(
    x_positions,
    means,
    yerr=stds,
    marker="o",
    capsize=3,
    label="QM9 Conditional"
)
plt.xticks(x_positions, num_steps_list)
plt.yticks([60, 80, 90, 100])
plt.ylim(bottom=min(all_values) - 5)
plt.xlabel("Number of Sampling Steps")
plt.ylabel("Validity")
plt.title("QM9 conditioned u0=-400 (Validity vs Sampling Steps)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Figure saved at {out_path}")

