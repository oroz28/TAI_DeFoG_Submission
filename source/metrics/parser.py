import re
from pathlib import Path
import numpy as np



def parse_metrics_file(path: Path):
    """
    Parse a metrics file and return a dictionary of metrics.
    The file is expected to have lines in the format:
    key: (mean, std)
    or  key: value  (for single values).
    
    Args:
        path (Path): Path to the metrics file.
    Returns:
        dict: A dictionary with metric names as keys and (mean, std) tuples as values.
    """
    
    tuple_pattern = re.compile(r"^([\w /]+): \(([-\d.eE]+), ([-\d.eE]+)\)$")
    single_pattern = re.compile(r"^([\w /]+): ([-\d.eE]+)$")
    
    metrics = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            
            if match := tuple_pattern.match(line):
                key, mean, std = match.groups()
                metrics[key] = (float(mean), float(std))
            
            elif match := single_pattern.match(line):
                key, value = match.groups()
                metrics[key] = (float(value), 0.0)
    
    return metrics


def get_vun(metrics: dict):
    """
    Get VUN from metrics dict.
    
    Args:
        metrics (dict): Dictionary of metrics.
    Returns:
        tuple: (mean VUN, std VUN) in percentage.
    """
    vun_mean, vun_std = metrics["sampling/frac_unic_non_iso_valid"]

    return vun_mean * 100, vun_std * 100


def get_ratio(metrics: dict):
    """
    Get ratio from metrics dict.
    
    Args:
        metrics (dict): Dictionary of metrics.
    Returns:
        tuple: (mean ratio, std ratio).
    """
    ratio_mean, ratio_std = metrics["average_ratio"]
    return ratio_mean, ratio_std,


def get_validity(metrics: dict):
    """
    Get validity from metrics dict.
    
    Args:
        metrics (dict): Dictionary of metrics.
    Returns:
        tuple: (mean validity, std validity) in percentage.
    """
    print(metrics)
    validity_mean, validity_std = metrics["Validity"]
    return validity_mean * 100, validity_std * 100,

def get_validity_planar(metrics: dict):
    """
    Get validity of planar dataset from metrics dict.
    
    Args:
        metrics (dict): Dictionary of metrics.
    Returns:
        tuple: (mean validity, std validity) in percentage.
    """
    validity_mean, validity_std = metrics["planar_acc"]
    return validity_mean * 100, validity_std * 100,

def get_mmd(metrics: dict):
    """
    Compute Mean Absolute Normalized (MAE) and normalized std
    from all entries in the metrics dict containing '_dist'.
    
    Args:
        metrics (dict): Dictionary of metrics.
    Returns:
        tuple: (MAE normalized, std normalized).
    """
    dist_metrics = {k: v for k, v in metrics.items() if k.endswith("_dist")}
    
    print("Distance metrics found:", dist_metrics.keys())
    
    if not dist_metrics:
        raise ValueError("No metrics with '_dist' found in dictionary.")
    
    values = np.array([v[0] for v in dist_metrics.values()])
    stds = np.array([v[1] for v in dist_metrics.values()])
    
    max_abs = np.max(np.abs(values))
    if max_abs == 0:
        max_abs = 1.0
    
    values_norm = np.abs(values) / max_abs
    stds_norm = stds / max_abs
    
    mae_norm = np.mean(values_norm)
    std_norm = np.mean(stds_norm)
    
    return mae_norm, std_norm

def get_time(metrics: dict):
    """
    Get time metrics from metrics dict.
    
    Args:
        metrics (dict): Dictionary of metrics.
    Returns:
        tuple: (total sampling time, number of samples).
    """
    time = metrics["total_sampling_time"]
    n_samples = metrics["n_samples"]
    return time, n_samples

def get_energy(metrics: dict):
    """
    Get energy metrics from metrics dict.
    
    Args:
        metrics (dict): Dictionary of metrics.
    Returns:
        tuple: (mean energy, std energy).
    """
    energy_mean = metrics["energy_mean"]
    energy_std = metrics["energy_std"]
    return energy_mean, energy_std

def get_mae(metrics: dict):
    """
    Get energy metrics from metrics dict.
    
    Args:
        metrics (dict): Dictionary of metrics.
    Returns:
        float: Conditional MAE.
    """
    cond_mae = metrics["cond_mae"][0]
    return cond_mae

