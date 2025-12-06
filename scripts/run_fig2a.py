import argparse
from pathlib import Path
from source.figure_run_helper import run_figure_2a 

# script to run Figure 2a experiments for QM9 or Planar datasets
def main():
    parser = argparse.ArgumentParser(description="Run Figure 2a experiments for QM9 or Planar.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["qm9", "planar"],
        help="Dataset to run experiments on: 'qm9' or 'planar'."
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default=None,
        help="Directory where outputs will be saved. Default is ./outputs"
    )
    parser.add_argument(
        "--defoq_src",
        type=str,
        default=None,
        help="Path to DeFoG src folder. Default is ../external/DeFoG/src"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        nargs="+",
        default=[5, 10, 50, 100, 1000],
        help="List of sample steps to run (default [5, 10, 50, 100, 1000])"
    )
    args = parser.parse_args()
    
    print(f"Running Figure 2a experiments on dataset: {args.dataset} with steps: {args.num_steps}")

    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else repo_root / "outputs"
    defoq_src_path = Path(args.defoq_src) if args.defoq_src else repo_root / "external" / "DeFoG" / "src"

    run_figure_2a(
        dataset=args.dataset,
        outputs_dir=outputs_dir,
        defog_src_path=defoq_src_path,
        num_steps_list=args.num_steps
    )

if __name__ == "__main__":
    main()
