from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .metrics.parser import parse_metrics_file, get_ratio, get_vun, get_validity, get_validity_planar

def collect_metrics(outputs_dir: Path, exp_prefix: str, steps: int, dataset="planar"):
    """
    Collects metrics (Ratio and VUN/Validity) from experiment directories.
    
    Args:
        outputs_dir: Path to the outputs directory
        exp_prefix: experiment prefix string
        steps: number of steps used in the experiment
        dataset: "planar" or "qm9" to determine which metric to collect
    Returns:
        ratio_means: list of ratio means corresponding to values
        vun_means: list of VUN/Validity means corresponding to values
    """
    
    ratio_means, vun_means = [], []
    exp_dir = outputs_dir / f"{exp_prefix}_{steps}_steps"
    metric_files = list(exp_dir.glob("test_epoch*"))
    if not metric_files:
        print(f"No metrics found in {exp_dir}")
        return [], []
    metrics = parse_metrics_file(metric_files[0])
    ratio_mean, _ = get_ratio(metrics)
    vun_mean, _ = get_vun(metrics) if dataset=="planar" else get_validity(metrics)
    if dataset=="planar":
        vun_mean /= 100
    ratio_means.append(ratio_mean)
    vun_means.append(vun_mean)
    return ratio_means, vun_means

def plot_curves(x_labels, ratio_means, vun_means, title, out_path: Path,
                xlabel: str, steps: int, exp_type: str, y_ticks_config: dict,
                dataset="planar"):
    """ 
    Plots two curves: VUN/Validity and Ratio (avg. ratio) on the same plot with dual y-axes.
    
    Args:
        x_labels: list of labels for x-axis
        ratio_means: list of ratio means corresponding to x_labels
        vun_means: list of VUN/Validity means corresponding to x_labels
        title: title of the plot
        out_path: Path to save the plot
        xlabel: label for x-axis
        steps: number of steps used in the experiment (for y-ticks configuration)
        exp_type: type of experiment (for y-ticks configuration)
        y_ticks_config: dict with y-ticks configurations
        dataset: "planar" or "qm9" to adjust labels
    """
    
    x_positions = np.arange(len(x_labels))
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(x_positions, vun_means, marker="s", color="tab:blue", 
             label="VUN" if dataset=="planar" else "Validity")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("VUN" if dataset=="planar" else "Validity", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=30 if len(x_labels) > 6 else 0)
    ax1.grid(True, linestyle="--", alpha=0.5)

    vun_ticks = y_ticks_config.get((steps, exp_type), {}).get("vun", [])
    if vun_ticks:
        y_min, y_max = ax1.get_ylim()
        valid_vun_ticks = [t for t in vun_ticks if y_min <= t <= y_max]
        if valid_vun_ticks:
            ax1.set_yticks(valid_vun_ticks)

    ax2 = ax1.twinx()
    ax2.plot(x_positions, ratio_means, marker="o", color="tab:orange", label="Ratio (avg. ratio)")
    ax2.set_ylabel("Ratio (avg. ratio)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ratio_ticks = y_ticks_config.get((steps, exp_type), {}).get("ratio", [])
    if ratio_ticks:
        y_min, y_max = ax2.get_ylim()
        valid_ratio_ticks = [t for t in ratio_ticks if y_min <= t <= y_max]
        if valid_ratio_ticks:
            ax2.set_yticks(valid_ratio_ticks)

    ax2.set_ylim(ax2.get_ylim()[1], ax2.get_ylim()[0])

    plt.title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_line_with_baseline(df, x_col, mean_col, std_col, xlabel, ylabel, title, out_path: Path, 
                            baseline=None, baseline_std=None, dataset="planar", color="blue", marker="o"):
    """
    Plots a line with error bars and an optional baseline.
    
    Args:
        df: DataFrame containing the data
        x_col: column name for x-axis
        mean_col: column name for mean values
        std_col: column name for std deviation  
        xlabel: label for x-axis
        ylabel: label for y-axis
        title: plot title
        out_path: Path to save the plot
        baseline: optional baseline value to plot
        baseline_std: optional std deviation for the baseline
        dataset: "planar" or "qm9" to adjust colors
        color: color for the main line
        marker: marker style for the main line
    """
    plt.figure(figsize=(7,5))
    plt.errorbar(df[x_col], df[mean_col], yerr=df[std_col],
                 fmt=marker, color=color, markersize=6, capsize=3, label=ylabel, linewidth=2, linestyle='-')
    
    # Área sombreada de std
    plt.fill_between(df[x_col],
                     df[mean_col] - df[std_col],
                     df[mean_col] + df[std_col],
                     color=color, alpha=0.15)
    
    # Línea de baseline
    if baseline is not None:
        plt.hlines(baseline, df[x_col].min(), df[x_col].max(),
                   colors="orange" if dataset=="planar" else "lime",
                   linestyles="--", linewidth=2, label=f"{ylabel} Baseline")
        if baseline_std is not None:
            plt.fill_between([df[x_col].min(), df[x_col].max()],
                             [baseline - baseline_std, baseline - baseline_std],
                             [baseline + baseline_std, baseline + baseline_std],
                             color="orange" if dataset=="planar" else "lime",
                             alpha=0.2)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
    
def plot_vun_comparison(results_dict: dict, steps_list: list, results_dir: Path,
                        y_limit=None, plot_title="", file_suffix=""):
    """
    Plots comparison of VUN across multiple configurations.

    Args:
        results_dict: dict of the form {cfg: {"steps":[], "mean":[], "std":[]}}
        steps_list: list of steps corresponding to the experiments
        results_dir: Path to save the plots
        y_limit: optional tuple (ymin, ymax) to set y-axis limits
        plot_title: title for the plot
        file_suffix: string to append to saved file
    """
    plt.figure(figsize=(8, 5))
    x_positions = np.arange(len(steps_list))

    for cfg, data in results_dict.items():
        if data["steps"]:
            sorted_indices = sorted(range(len(data["steps"])), key=lambda i: data["steps"][i])
            means = [data["mean"][i] for i in sorted_indices]
            stds = [data["std"][i] for i in sorted_indices]

            plt.errorbar(x_positions, means, yerr=stds, marker="o", capsize=3, label=cfg)

    plt.xticks(x_positions, steps_list)
    plt.xlabel("Number of Sampling Steps")
    plt.ylabel("VUN")

    if y_limit:
        plt.ylim(y_limit[0], y_limit[1])
    else:
        plt.yticks([0, 0.3, 0.6, 0.9, 1.0])

    plt.title(plot_title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = results_dir / f"vun_comparison{file_suffix}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {out_path}")
    
    
def read_baseline(file_path: Path, dataset="planar"):
    """
    Reads baseline metrics from a given file.
    
    Args:
        file_path: Path to the baseline metrics file
        dataset: "planar" or "qm9" to determine which metric to read
    
    Returns:
        mean: baseline mean value
        std: baseline std deviation
    """
    if not file_path.exists():
        print(f"Baseline file not found: {file_path}")
        return None, None

    metrics = parse_metrics_file(file_path)
    mean, std = get_validity_planar(metrics) if dataset=="planar" else get_validity(metrics)
    print(f"Baseline file: {file_path}, Validity mean: {mean}, std: {std}")

    return float(mean), float(std)

