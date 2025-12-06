from pathlib import Path
import pandas as pd
from .metrics.summarizer import build_results_table_from_dict, build_validity_table_from_dict

def make_results_table(file_map: dict, output_file: Path, dataset: str = None, validity: bool = False) -> pd.DataFrame:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    """ 
    Create a results table (CSV) from experiment metrics files.
    
    Args:
        file_map: dict mapping experiment names to their metrics file paths
        output_file: Path to save the resulting CSV table
        dataset: "planar" or "qm9" (required if validity=True)
        validity: whether to build a validity table instead of results table
    Returns:
        DataFrame containing the results table
    """
    
    if validity:
        if dataset is None:
            raise ValueError("Dataset must be specified for validity table.")
        df = build_validity_table_from_dict(file_map, output_file, dataset=dataset)
    else:
        df = build_results_table_from_dict(file_map, output_file)
    
    df.to_csv(output_file, index=False)
    print(f"[OK] Saved table to: {output_file}")
    return df
