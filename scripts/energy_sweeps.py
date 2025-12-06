#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from source.hyperparameter_search_helper import evaluate_convo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="qm9_with_h", choices=["qm9", "tls"])
    parser.add_argument("--exp_name", type=str, default="qm9_with_h_conditional.ckpt")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--distortion", type=str, default="polydec")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--omega", type=float, default=0.0)
    parser.add_argument("--num_folds", type=int, default=1)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--cond_start", type=float, default=-350)
    parser.add_argument("--cond_end", type=float, default=-475)
    parser.add_argument("--num_cond", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    defoq_src = repo_root / "external" / "DeFoG" / "src"
    outputs_dir = repo_root / "outputs" / f"energy_sweeps_{args.dataset}"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cond_values = np.linspace(args.cond_start, args.cond_end, args.num_cond)
    all_results = []

    for cond_value in cond_values:
        df_row = evaluate_convo(
            exp_name=args.exp_name,
            num_steps=args.num_steps,
            distortion=args.distortion,
            eta=args.eta,
            omega=args.omega,
            defoq_src=defoq_src,
            outputs_dir=outputs_dir,
            num_folds=args.num_folds,
            samples_to_generate=args.samples,
            condition_value=cond_value,
            output_folder_name="search_qm9_with_h_conditional.ckpt_100_steps"
        )
        all_results.append(df_row)

    summary_df = pd.concat(all_results, ignore_index=True)
    out_path = outputs_dir / "energy_sweep_results_with_h.csv"
    summary_df.to_csv(out_path, index=False)
    print("Saved results to:", out_path)
    print(summary_df.head())

if __name__ == "__main__":
    main()
