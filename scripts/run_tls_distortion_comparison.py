from pathlib import Path
from source.figure_run_helper import run_figure3

# script to run distortion experiments for DeFoG and generate plots for Distortion Comparison Figure
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = repo_root / "outputs"
    defog_src_path = repo_root / "external" / "DeFoG" / "src"

    run_figure3(outputs_dir=outputs_dir, defog_src_path=defog_src_path, num_steps_list=[100], dataset="tls")
    
    print("Figure 3 experiments for TLS dataset completed successfully.")
