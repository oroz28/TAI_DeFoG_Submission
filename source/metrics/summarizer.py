import pandas as pd
from pathlib import Path
from .parser import parse_metrics_file, get_vun, get_ratio, get_validity, get_validity_planar, get_mmd


def build_results_table_from_dict(file_map, output_file):
    """ 
    Build results DataFrame from a dictionary mapping experiment names and steps to file paths.
    
    Args:
        file_map (dict): Dictionary with keys as (experiment_name, steps) and values as file paths.
        output_file (str): Path to save the resulting DataFrame as CSV.
    Returns:
        pd.DataFrame: DataFrame containing VUN and Ratio metrics.
    """
    rows = []
    for (exp_name, steps), file_path in file_map.items():
        file = Path(file_path)
        if not file.exists():
            print(f"Warning: file not found {file}")
            continue

        metrics = parse_metrics_file(file)
        vun = get_vun(metrics)
        ratio = get_ratio(metrics)

        rows.append(
            {
                "experiment": exp_name,
                "steps": steps,
                "VUN_mean": vun[0],
                "VUN_std": vun[1],
                "Ratio_mean": ratio[0],
                "Ratio_std": ratio[1],
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values(by=["experiment", "steps"], inplace=True)

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    return df

def build_validity_table_from_dict(file_map, output_file, dataset="qm9"):
    """
    Build validity results DataFrame from a dictionary mapping experiment names and steps to file paths.
    
    Args:
        file_map (dict): Dictionary with keys as (experiment_name, steps) and values as file paths.
        output_file (str): Path to save the resulting DataFrame as CSV.
        dataset (str): Dataset type, either 'qm9' or 'planar'.  
    Returns:
        pd.DataFrame: DataFrame containing Validity and MMD/Ratio metrics.
    """
    rows = []
    for (exp_name, steps), file_path in file_map.items():
        file = Path(file_path)
        if not file.exists():
            print(f"Warning: file not found {file}")
            continue
        metrics = parse_metrics_file(file)
        validity = get_validity_planar(metrics) if dataset.startswith("planar") else get_validity(metrics)
        mmd = get_mmd(metrics) if dataset.startswith("qm9") else get_ratio(metrics)
        

        rows.append(
            {
                "experiment": exp_name,
                "steps": steps,
                "Validity_mean": validity[0],
                "Validity_std": validity[1],
                "MMD_mean": mmd[0],
                "MMD_std": mmd[1],
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values(by=["experiment", "steps"], inplace=True)

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

    return df

