from pathlib import Path
from source.metrics.parser import parse_metrics_file, get_vun
from source.figure_plot_helper import plot_vun_comparison

# paths
repo_root = Path(__file__).resolve().parents[1]
outputs_dir = repo_root / "outputs"
results_dir = repo_root / "results" / "plots"
results_dir.mkdir(exist_ok=True)

# experiment configurations
configs = {
    "opt": "Optimized Hyperparameters per num_steps",
    "def": "Default Optimized Hyperparameters for all num_steps",
}
num_steps_list = list(range(20, 51, 5))
vun_results = {cfg: {"steps": [], "mean": [], "std": []} for cfg in configs}

# collect results
for cfg in configs:
    for steps in num_steps_list:
        exp_dir = outputs_dir / f"hyper_comp_steps_{steps}_{cfg}"
        metric_files = list(exp_dir.glob("test_epoch*"))
        if not metric_files:
            print(f"No metrics file found in {exp_dir}")
            continue

        metrics = parse_metrics_file(metric_files[0])
        mean, std = get_vun(metrics)
        mean /= 100
        std /= 100

        vun_results[cfg]["steps"].append(steps)
        vun_results[cfg]["mean"].append(mean)
        vun_results[cfg]["std"].append(std)

# plot results
plot_vun_comparison(vun_results, num_steps_list, results_dir,
                    y_limit=None,
                    plot_title="Planar Dataset Comparison (Full Range)",
                    file_suffix="_full_range")

plot_vun_comparison(vun_results, num_steps_list, results_dir,
                    y_limit=[0.55, 1.0],
                    plot_title="Planar Dataset Comparison (Zoomed: 0.55 to 1.0)",
                    file_suffix="_zoom_07_10")
