import pandas as pd
import matplotlib.pyplot as plt
import os


csv_base_path = "/home/group-2/Submission_Code/qm9_with_h_baseline/summary_results_qm9_with_h.csv"
csv_second_path = "/home/group-2/Submission_Code/qm9_with_h_grid/summary_results_qm9_with_h.csv"
output_dir = "/home/group-2/Submission_Code/results/plots/"

steps_to_use = [10, 20, 30, 40, 50]


df_base = pd.read_csv(csv_base_path)
df_base = df_base[df_base["num_steps"].isin(steps_to_use)]

base_steps = df_base["num_steps"]
base_validity = df_base["validity_mean"]

df_second = pd.read_csv(csv_second_path)
df_second = df_second[df_second["num_steps"].isin(steps_to_use)]

df_second_best = df_second.groupby("num_steps")["validity_mean"].max().reset_index()

second_steps = df_second_best["num_steps"]
second_validity = df_second_best["validity_mean"]

plt.figure(figsize=(10, 6))

plt.plot(base_steps, base_validity, marker="o", label="qm9 with h")
plt.plot(second_steps, second_validity, marker="o", label="qm9 with h optimized")

plt.xlabel("num_steps")
plt.ylabel("validity_mean")
plt.title("Optimized qm9 results")
plt.grid(True)
plt.legend()
plt.tight_layout()

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "qm9_with_h.png")

plt.savefig(output_path)
plt.close()

print(f"Plot saved in: {output_path}")
