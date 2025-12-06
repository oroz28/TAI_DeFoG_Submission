from pathlib import Path
from source.figure_run_helper import run_distortions

# run distortion experiments for DeFoG
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = repo_root / "outputs"
    defog_src_path = repo_root / "external" / "DeFoG" / "src"

    run_distortions(outputs_dir=outputs_dir, defog_src_path=defog_src_path)
