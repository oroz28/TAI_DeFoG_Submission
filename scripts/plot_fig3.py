import sys
from pathlib import Path
from itertools import product
from source.figure_plot_helper import collect_metrics, plot_curves

# read command line argument

dataset = sys.argv[1] if len(sys.argv) > 1 else "planar"  # 'planar' or 'qm9'

# paths
repo_root = Path(__file__).resolve().parents[1]
outputs_dir = repo_root / "outputs"
results_dir = repo_root / "results" / "plots"
results_dir.mkdir(exist_ok=True, parents=True)

# configurations and steps
num_steps_list = [50] #[50, 1000]
distortions = ["polyinc", "revcos", "identity", "cos", "polydec"]
omegas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
etas = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

# y-axis ticks configuration for specific plots
y_ticks_config = {
    # (50, "distortion"): {"vun": [0, 0.3, 0.6], "ratio": [10, 15]},
    # (50, "omega"):       {"vun": [0.3, 0.15], "ratio": [5, 7.5, 10, 15]},
    # (50, "eta"):         {"vun": [0.1, 0.05, 0.0], "ratio": [0, 100, 200]},
    # (1000, "distortion"):{"vun": [0.8, 0.9], "ratio": [1.2, 1.4]},
    # (1000, "omega"):     {"vun": [0.8, 0.9], "ratio": [5, 10, 15]},
    # (1000, "eta"):       {"vun": [0.84, 0.88, 0.96], "ratio": [2, 3, 5]},
}

# define and generate plots
# distortion
ratio_means, vun_means = [], []
for steps in num_steps_list:
    for distortion in distortions:
        ratio_mean, vun_mean = collect_metrics(outputs_dir, f"f3_distortion_{distortion}", steps)
        ratio_means.append(ratio_mean)
        vun_means.append(vun_mean)
        print(f"Distortion: {distortion}, Steps: {steps}, Ratio Means: {ratio_means}, VUN Means: {vun_means}")
    plot_curves(distortions, ratio_means, vun_means,
                title=f"Figure 3: Distortion Effects ({dataset}, {steps} steps)",
                out_path=results_dir / f"figure3_distortion_{steps}_steps.png",
                xlabel="Time Distortion Type",
                steps=steps,
                exp_type="distortion",
                y_ticks_config=y_ticks_config)

# omega
ratio_means, vun_means = [], []
for steps in num_steps_list:
    for omega in omegas:
        ratio_mean, vun_mean = collect_metrics(outputs_dir, f"f3_omega_{omega}", steps)
        ratio_means.append(ratio_mean)
        vun_means.append(vun_mean)
        print(f"Omega: {omega}, Steps: {steps}, Ratio Means: {ratio_means}, VUN Means: {vun_means}")
    plot_curves(omegas, ratio_means, vun_means,
                title=f"Figure 3: Omega Effects ({dataset}, {steps} steps)",
                out_path=results_dir / f"figure3_omega_{steps}_steps.png",
                xlabel="Omega Values",
                steps=steps,
                exp_type="omega",
                y_ticks_config=y_ticks_config)

# eta
ratio_means, vun_means = [], []
for steps in num_steps_list:
    for eta in etas:
        ratio_mean, vun_mean = collect_metrics(outputs_dir, f"f3_eta_{eta}", steps)
        ratio_means.append(ratio_mean)
        vun_means.append(vun_mean)
        print(f"Eta: {eta}, Steps: {steps}, Ratio Means: {ratio_means}, VUN Means: {vun_means}")
    plot_curves(etas, ratio_means, vun_means,
                title=f"Figure 3: Eta Effects ({dataset}, {steps} steps)",
                out_path=results_dir / f"figure3_eta_{steps}_steps.png",
                xlabel="Eta Values",
                steps=steps,
                exp_type="eta",
                y_ticks_config=y_ticks_config)

print("\nAll plots generated successfully for dataset:", dataset)
