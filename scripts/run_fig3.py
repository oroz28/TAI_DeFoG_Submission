from pathlib import Path
from source.figure_run_helper import run_figure3
import sys

# script to run distortion experiments for DeFoG and generate plots for Figure 3
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = repo_root / "outputs"
    defog_src_path = repo_root / "external" / "DeFoG" / "src"
    
    args = sys.argv
    dataset = args[1] if len(args) > 1 else "planar"

    run_figure3(outputs_dir=outputs_dir, defog_src_path=defog_src_path, dataset=dataset)
