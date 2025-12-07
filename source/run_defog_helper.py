from itertools import product
import subprocess
import os
from pathlib import Path

def run_defog_experiments(
    experiments,
    num_steps_list,
    outputs_dir: Path,
    defog_src_path: Path,
    extra_args=None,
    pythonpath_extra="",
    name_prefix="run",
    verbose=True,
    overrides_per_experiment=None,
):
    """
    Runs Defog experiments with specified overrides.
    
    Args:
        experiments: list of experiment checkpoint names
        num_steps_list: list of number of steps to run experiments with
        outputs_dir: Path to the outputs directory
        defog_src_path: Path to the Defog source code
        extra_args: list of extra command line arguments to pass
        pythonpath_extra: extra paths to add to PYTHONPATH
        name_prefix: prefix for naming experiment directories
        verbose: whether to print verbose output
        overrides_per_experiment: dict mapping experiment names to their specific overrides
    """

    print(f"Outputs directory: {outputs_dir}")
    outputs_dir.mkdir(exist_ok=True, parents=True)
    pythonpath = f":{defog_src_path.parent.resolve()}{pythonpath_extra}"

    for exp_name, steps in product(experiments, num_steps_list):

        exp_tag = exp_name.split("-")[0].split(".")[0]
        run_dir = outputs_dir / f"{name_prefix}_{exp_name}_{steps}_steps"
        # run_dir.mkdir(exist_ok=True)

        cmd = [
            "python",
            f"{defog_src_path}/main.py",
            f"+experiment={exp_tag}",
            f"dataset={exp_tag.split('_')[0]}",
            f"sample.sample_steps={steps}",
            f"general.test_only=/home/group-2/Submission_Code/checkpoints/{exp_name}",
            f"hydra.run.dir={str(run_dir.resolve())}",
        ]

        # specific overrides per experiment
        if overrides_per_experiment and exp_name in overrides_per_experiment:
            overrides = overrides_per_experiment[exp_name]
            if any(k.startswith("hydra") for k in overrides.keys()):
                cmd = [c for c in cmd if not c.startswith("hydra")]
            
            cmd.extend([f"{k}={v}" for k, v in overrides.items()])
            print(cmd)

        # extra args
        if extra_args:
            cmd.extend(extra_args)

        if verbose:
            print("Running:", " ".join(cmd))

        env = os.environ.copy()
        env["PYTHONPATH"] = pythonpath
        subprocess.run(cmd, check=True, env=env)

    os.environ["PYTHONPATH"] = ""
    if verbose:
        print("All experiments completed.")
