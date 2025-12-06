import pandas as pd
import matplotlib.pyplot as plt
import sys

args = sys.argv

# read CSV based on argument
if args[1] == "planar":
    csv_path = "/home/group-2/Submission_Code/results/tables/validity_table_def_planar_cv.csv"
    baseline_path = "planar_cv_outputs"
    best_path = "subgraph_graph_002"
    worst_path = "subgraph_graph_003"
    plot_title = "Planar - 15 Fixed Nodes"
elif args[1] == "qm9_with_h":
    csv_path = "/home/group-2/Submission_Code/results/tables/validity_table_def_qm9_with_h_cv.csv"
    baseline_path = "qm9_with_h_cv_outputs"
    best_path = "subgraphs_graph_4077"
    worst_path = "subgraphs_graph_19858"
    plot_title = "QM9 with H - 2 Fixed Nodes"
df = pd.read_csv(csv_path)

print(df['experiment'].tolist())


# filter data for baseline, best, and worst
baseline_values = df[df['experiment'].str.contains(baseline_path, regex=False)]['Validity_mean'].tolist()
best_values = df[df['experiment'].str.contains(best_path, regex=False)]['Validity_mean'].tolist()
worst_values = df[df['experiment'].str.contains(worst_path, regex=False)]['Validity_mean'].tolist()


print("Baseline:", baseline_values)
print("Best:", best_values)
print("Worst:", worst_values)

# create boxplot
data = [baseline_values, best_values, worst_values]
labels = ["Baseline", "Best", "Worst"]

plt.figure(figsize=(7, 4))
plt.boxplot(
    data,
    labels=labels,
    patch_artist=True,
    medianprops=dict(color="black"),
    boxprops=dict(facecolor="lightblue", color="blue"),
    showfliers=False,
)
plt.ylabel("Validity_mean")
plt.title(f"Baseline vs Best vs Worst (Boxplot) - {plot_title}")
plt.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()

save_path = f"/home/group-2/Submission_Code/results/plots/validity_boxplot_{args[1]}_cv.png"
plt.savefig(save_path)
plt.show()
print(f"Boxplot saved to {save_path}")