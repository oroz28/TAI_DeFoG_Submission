import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "/home/group-2/Submission_Code/outputs/qm9_with_h_final_gridsearch/config_results.csv"
OUTDIR = "/home/group-2/Submission_Code/results/plots/"
PLOT_NAME = "final_gridsearch_qm9_with_h.png"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    required_cols = {"num_steps", "eta", "omega", "validity_mean"}
    best_rows = df.loc[df.groupby("num_steps")["validity_mean"].idxmax()]

    print("\n=== Best configuration ===")
    for _, row in best_rows.iterrows():
        print(
            f"num_steps={int(row['num_steps'])}: "
            f"eta={row['eta']}, omega={row['omega']}, validity_mean={row['validity_mean']:.4f}"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(best_rows["num_steps"], best_rows["validity_mean"], marker="o")
    plt.xlabel("num_steps")
    plt.ylabel("Best validity mean")
    plt.title("Best validity mean per num_steps")
    plt.grid(True)

    output_path = os.path.join(OUTDIR, PLOT_NAME)
    plt.savefig(output_path, dpi=300)

    print(f"\nPlot saved in: {output_path}")


if __name__ == "__main__":
    main()
