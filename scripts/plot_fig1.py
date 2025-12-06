import sys
from pathlib import Path
import numpy as np
from source.figure_plot_helper import collect_metrics, plot_curves

# read dataset from command line arguments
if len(sys.argv) < 2:
    print("Usage: python plot_figure1.py <dataset>")
    print("Example: python plot_figure1.py planar")
    sys.exit(1)

dataset = sys.argv[1].lower()  # 'planar' or 'qm9'

# paths
repo_root = Path(__file__).resolve().parents[1]
outputs_dir = repo_root / ("outputs" if dataset == "planar" else "outputs_qm9")
results_dir = repo_root / "results" / "plots" / dataset
results_dir.mkdir(exist_ok=True, parents=True)

# parameters
num_steps_list = [50]  # , 1000
distortions = ["polyinc", "revcos", "cos", "polydec"]
omegas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
etas = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

# plotting Figure 1
# distortion
for steps in num_steps_list:
    ratio_means, vun_means = collect_metrics(outputs_dir, "distortion", distortions, steps, dataset=dataset)
    plot_curves(
        distortions,
        ratio_means,
        vun_means,
        title=f"Figure 1: Distortion Effects ({dataset}, {steps} steps)",
        out_path=results_dir / f"figure1_distortion_{steps}_steps.png",
        xlabel="Time Distortion Type",
        dataset=dataset,
    )

# omega
for steps in num_steps_list:
    ratio_means, vun_means = collect_metrics(outputs_dir, "omega", omegas, steps, dataset=dataset)
    labels = [str(v) for v in omegas]
    plot_curves(
        labels,
        ratio_means,
        vun_means,
        title=f"Figure 1: Ω sweep ({dataset}, {steps} steps)",
        out_path=results_dir / f"figure1_omega_{steps}_steps.png",
        xlabel="Omega (ω)",
        dataset=dataset,
    )

# eta
for steps in num_steps_list:
    ratio_means, vun_means = collect_metrics(outputs_dir, "eta", etas, steps, dataset=dataset)
    labels = [str(v) for v in etas]
    plot_curves(
        labels,
        ratio_means,
        vun_means,
        title=f"Figure 1: η sweep ({dataset}, {steps} steps)",
        out_path=results_dir / f"figure1_eta_{steps}_steps.png",
        xlabel="Eta (η)",
        dataset=dataset,
    )

print("\nAll Figure 1 plots generated successfully.")
