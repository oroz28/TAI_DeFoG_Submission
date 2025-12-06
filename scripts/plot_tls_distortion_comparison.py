import sys
from pathlib import Path
from itertools import product
from source.figure_plot_helper import collect_metrics, plot_curves

# read dataset from command line arguments
if len(sys.argv) < 2:
    print("Usage: python plot_fig3.py <dataset>")
    print("Example: python plot_fig3.py planar")
    sys.exit(1)

dataset = sys.argv[1]  # 'planar', 'qm9' or 'tls

# paths
repo_root = Path(__file__).resolve().parents[1]
outputs_dir = repo_root / ("outputs" if dataset == "planar" else "outputs_tls" if dataset == "tls" else "outputs_qm9")
results_dir = repo_root / "results" / "plots" / dataset
results_dir.mkdir(exist_ok=True, parents=True)

# parameters
num_steps_list = [100]
distortions = ["polyinc", "revcos", "identity", "cos", "polydec"]

# y_ticks_config for different datasets and experiment types
y_ticks_config = {
    # (50, "distortion"): {"vun": [0, 0.3, 0.6], "ratio": [10, 15]},
    # (50, "omega"):       {"vun": [0.3, 0.15], "ratio": [5, 7.5, 10, 15]},
    # (50, "eta"):         {"vun": [0.1, 0.05, 0.0], "ratio": [0, 100, 200]},
    # (1000, "distortion"):{"vun": [0.8, 0.9], "ratio": [1.2, 1.4]},
    # (1000, "omega"):     {"vun": [0.8, 0.9], "ratio": [5, 10, 15]},
    # (1000, "eta"):       {"vun": [0.84, 0.88, 0.96], "ratio": [2, 3, 5]},
}

# distortion plots
for steps in num_steps_list:
    ratio_means, vun_means = collect_metrics(outputs_dir, "distortion", distortions, steps)
    plot_curves(distortions, ratio_means, vun_means,
                title=f"Distortion Effects ({dataset}, {steps} steps)",
                out_path=results_dir / f"figure1_distortion_{steps}_steps.png",
                xlabel="Time Distortion Type",
                steps=steps,
                exp_type="distortion",
                y_ticks_config=y_ticks_config)

print("\nAll plots generated successfully for dataset:", dataset)
