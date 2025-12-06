from pathlib import Path
import pandas as pd
import re
from source.figure_plot_helper import plot_line_with_baseline, read_baseline
import sys

repo_root = Path(__file__).resolve().parents[1]
output_dir = repo_root / "results/plots"
output_dir.mkdir(exist_ok=True, parents=True)

args = sys.argv

# load data and baseline
if args[1] == "planar":
    planar_df = pd.read_csv(repo_root / "results/tables/validity_table_def_planar.csv")
    baseline_path = repo_root / "outputs/t1_planar_50_steps/test_epoch0_res_50_general.txt"
else:
    planar_df = pd.read_csv(repo_root / "results/tables/validity_table_def_qm9_with_h.csv")
    baseline_path = repo_root / "outputs/qm9_with_h_baseline/test_epoch0_res_0_general.txt"


# process dataframe
def process_df(df):
    col_mean = "Validity_mean"

    # extract length from experiment name
    def extract_len(exp_name, dataset):
        if dataset == "planar":
            m = re.search(r"len_(\d+)", exp_name)
        else:  # qm9
            m = re.search(r"chain_(\d+)", exp_name)

        return int(m.group(1)) if m else None


    df["length"] = df["experiment"].apply(lambda x: extract_len(x, args[1]))

    # join by length and calculate mean and std
    grouped = df.groupby("length")[col_mean]
    summary = grouped.agg(
        mean="mean",
        std="std"
    ).reset_index()

    return summary.sort_values("length")


planar_summary = process_df(planar_df)
print(planar_summary)

# read baseline
baseline_mean, baseline_std = read_baseline(baseline_path, dataset=args[1])

plot_line_with_baseline(
    planar_summary,
    x_col="length",
    mean_col="mean",
    std_col="std",
    xlabel="Number of fixed elements",
    ylabel="Mean validity (%)",
    title=f"Validity vs. Number of fixed elements ({args[1]})",
    out_path=output_dir / f"validity_{args[1]}_with_baseline_def.png",
    dataset=args[1],
    baseline=baseline_mean,
    baseline_std=baseline_std
)


print(f"[OK] Plot saved to {output_dir / f'validity_{args[1]} _with_baseline_def.png'}")