#!/usr/bin/env python3
import argparse
from pathlib import Path
from source.model_helper import VUNPredictorMLP, run_dataset_with_model
import pandas as pd
import numpy as np

# main function to run experiments with default vs model parameters for planar and qm9 datasets
def main():
    parser = argparse.ArgumentParser(description="Run Figure of base model vs MLP optimization.")
    parser.add_argument("--dataset", type=str, required=True, choices=["planar", "qm9"],
                        help="Dataset to run: 'planar' or 'qm9'")
    parser.add_argument("--use_model", action="store_true",
                        help="Use MLP model to predict parameters (only for planar).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    defog_src = repo_root / "external" / "DeFoG" / "src"
    outputs_dir = repo_root / "outputs" / f"default_vs_model_{args.dataset}"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # num_steps list
    num_steps_list = np.arange(5, 51, 5).tolist()
    mlp_model = None
    
    # checkpoint paths and model loading
    if args.dataset == "planar":
        checkpoint = repo_root / "checkpoints" / "planar.ckpt"
        if args.use_model:
            csv_path = repo_root / "results" / "tables" / "grid_summary_results_planar.csv"
            df_existing = pd.read_csv(csv_path)
            mlp_model = VUNPredictorMLP(df_existing=df_existing)
    else:
        checkpoint = repo_root / "checkpoints" / "qm9_with_h_conditional.ckpt"
        if args.use_model:
            print("Warning: MLP model usage is only implemented for 'planar' dataset. Ignoring --use_model flag.")
            mlp_model = 1

    # run experiments
    run_dataset_with_model(
        dataset=args.dataset,
        outputs_dir=outputs_dir,
        defog_src_path=defog_src,
        num_steps_list=num_steps_list,
        checkpoint=checkpoint,
        mlp_model=mlp_model,
        use_model=args.use_model,
    )

if __name__ == "__main__":
    main()
