import argparse
from pathlib import Path
from source.run_defog_helper import run_defog_experiments

# script parameters and execution for running DeFoG experiments and generating Table 1
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs="?",
        default="new",
        choices=["new", "old"],
        help="Choose 'new' for new checkpoints or 'old' for old checkpoints.",)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    defog_src = repo_root / "external" / "DeFoG" / "src"
    outputs_dir = repo_root / "outputs"

    # choose experiments based on argument
    if args.experiment == "old":
        experiments = [
            "planar-59999-14h.ckpt",
            "tree-19999-5h.ckpt",
            "sbm-31999-40h.ckpt",
        ]
        prefix = "old_ckpts"
    else:  # "new"
        experiments = [
            "planar.ckpt",
            "tree.ckpt",
            "sbm.ckpt",
        ]
        prefix = "new_ckpts"

    print(f" Ejecutando con experiment='{args.experiment}'")
    print("Checkpoints:", experiments)

    run_defog_experiments(
        experiments=experiments,
        num_steps_list=[50, 1000],
        outputs_dir=outputs_dir,
        defog_src_path=defog_src,
        extra_args=["visualization=False", "general.num_sample_fold=5"],
        name_prefix=f"t1_{prefix}",
    )

if __name__ == "__main__":
    main()
