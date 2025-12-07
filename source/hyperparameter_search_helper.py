from itertools import product
import os
import subprocess
import pandas as pd
from pathlib import Path
from source.metrics.parser import (
    parse_metrics_file,
    get_validity_planar,
    get_ratio,
    get_time,
    get_vun,
)
import random
import numpy as np
from .run_defog_helper import run_defog_experiments
from .metrics.parser import parse_metrics_file, get_validity_planar, get_ratio, get_time, get_vun, get_validity, get_mmd, get_mae, get_energy

def evaluate_convo(
    exp_name,
    num_steps,
    distortion,
    eta,
    omega,
    defoq_src,
    outputs_dir,
    num_folds=5,
    output_folder_name=None,
    csv_name="config_results.csv",
    output_prefix="search",
    samples_to_generate=None,
    condition_value=None,
):
    """ 
    Evaluates a single configuration of hyperparameters for a given experiment.
    
    Args:
        exp_name: Name of the experiment/checkpoint
        num_steps: Number of sampling steps
        distortion: Type of distortion to use
        eta: Eta parameter for distortion
        omega: Omega parameter for distortion
        defoq_src: Path to the Defog source code
        outputs_dir: Path to the outputs directory
        num_folds: Number of folds for sampling
        output_folder_name: Custom output folder name (if None, a name is generated)
        csv_name: Name of the CSV file to store results
        output_prefix: Prefix for the output folder name
        samples_to_generate: Number of samples to generate (if None, default is used)
        condition_value: Conditional value for conditional generation (if any)
    Returns:
        df_row: DataFrame row with the evaluation results
    """
    # output directory
    if output_folder_name is None:
        base = exp_name.split(".")[0]
        output_dir = outputs_dir / f"{output_prefix}_{base}_{num_steps}_steps_distortion:{distortion}_eta:{eta}_omega:{omega}"
    else:
        output_dir = outputs_dir / output_folder_name

    # overrides for the experiment
    overrides = {
        "sample.time_distortion": distortion,
        "sample.eta": eta,
        "sample.omega": omega,
        "general.num_sample_fold": num_folds,
        "visualization": False,
        "general.test_only": f"/home/group-2/TAI_defog/tai_defog/checkpoints/{exp_name}",
        "hydra.run.dir": str(output_dir.resolve()),
    }
    if samples_to_generate is not None:
        overrides["general.final_model_samples_to_generate"] = samples_to_generate
    if condition_value is not None:
        overrides["general.condition_values"] = condition_value
        
    print(output_dir)

    # run experiment
    run_defog_experiments(
        experiments=[exp_name],
        num_steps_list=[num_steps],
        outputs_dir=output_dir,
        defog_src_path=defoq_src,
        overrides_per_experiment={exp_name: overrides},
        name_prefix=output_prefix,
        verbose=True
    )

    # parse metrics
    print(output_dir / f"test_epoch0_res_{eta}_general.txt")
    metrics_file = output_dir / f"test_epoch0_res_{eta}_general.txt"
    metrics = parse_metrics_file(metrics_file)

    validity_mean, validity_std = get_validity_planar(metrics) if exp_name.split(".")[0].startswith("planar") else get_validity(metrics)
    mmd_mean, mmd_std = get_ratio(metrics) if exp_name.split(".")[0].startswith("planar") else get_mmd(metrics)
    time, _ = get_time(metrics)
    vun_mean, vun_std = get_vun(metrics) if exp_name.split(".")[0].startswith("planar") else (None, None)
    mae_mean = None if exp_name.split(".")[0].startswith("planar") else get_mae(metrics)
    energy_mean, energy_std = get_energy(metrics)

    df_row = pd.DataFrame([{
        "exp_name": exp_name,
        "num_steps": num_steps,
        "distortion": distortion,
        "eta": eta,
        "omega": omega,
        "validity_mean": validity_mean,
        "validity_std": validity_std,
        "vun_mean": vun_mean,
        "vun_std": vun_std,
        "mmd_mean": mmd_mean,
        "mmd_std": mmd_std,
        "mae_mean": mae_mean,
        "time": time,
        "samples_to_generate": samples_to_generate,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
    }])
    
    if condition_value is not None:
        df_row["condition_value"] = condition_value

    csv_path = outputs_dir / csv_name
    df_row.to_csv(csv_path, mode="a", index=False, header=not csv_path.exists())
    
    print(f"Resultado: \n{df_row}")

    return df_row


def generate_unique_random_configs(n_samples, csv_path):
    """
    Generates unique random configurations for hyperparameter search,
    ensuring no duplicates with existing configurations in the provided CSV.
    
    Args:
        n_samples: Number of unique configurations to generate
        csv_path: Path to the existing CSV file with previous configurations
    Returns:
        DataFrame with new unique configurations
    """
    df_existing = pd.read_csv(csv_path)

    # set of used configurations
    used = set(
        (row.num_steps, row.distortion, row.eta, row.omega)
        for _, row in df_existing.iterrows()
    )

    new_configs = []

    while len(new_configs) < n_samples:
        # generate random configuration in continuous/discrete spaces
        eta = round(random.uniform(0, 250), 4)
        omega = round(random.uniform(0.0, 1.0), 4)
        num_steps = random.choice(np.arange(10, 51, 1).tolist())
        distortion = random.choice(["polydec", "cos"])

        combo = (num_steps, distortion, eta, omega)

        if combo in used:
            continue

        # save new configuration
        used.add(combo)
        new_configs.append({
            "num_steps": num_steps,
            "distortion": distortion,
            "eta": eta,
            "omega": omega
        })

    return pd.DataFrame(new_configs)


def make_objective(
    checkpoint_name,
    defoq_src,
    outputs_dir,
    distortions,
    eta_bounds,
    omega_bounds,
    num_steps,
    num_folds=5,
    **eval_kwargs
):
    """
    Creates an objective function for hyperparameter optimization.
    
    Args:
        checkpoint_name: Name of the experiment/checkpoint
        defoq_src: Path to the Defog source code
        outputs_dir: Path to the outputs directory
        distortions: List of distortion types to choose from
        eta_bounds: Tuple with (min, max) bounds for eta
        omega_bounds: Tuple with (min, max) bounds for omega
        num_steps: Number of sampling steps
        num_folds: Number of folds for sampling
        eval_kwargs: Additional keyword arguments for evaluation
    Returns:
        objective function to be used in hyperparameter optimization  
    """

    def objective(trial):
        """ 
        Objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
        Returns:
            score: Validity mean score from evaluation
        """
        phi_id = trial.suggest_categorical("distortion", distortions)
        eta = trial.suggest_float("eta", *eta_bounds)
        omega = trial.suggest_float("omega", *omega_bounds)

        score_df = evaluate_convo(
            checkpoint_name,
            num_steps,
            phi_id,
            eta,
            omega,
            defoq_src,
            outputs_dir,
            num_folds=num_folds,
            **eval_kwargs
        )

        score = score_df["validity_mean"].iloc[0]
        print("Score:", score)
        return score

    return objective



def run_hyperparameter_search(
    mode,                         # "grid" oR "random"
    dataset,
    exp_name,
    distortions,
    num_steps_list,
    eta_list,
    omega_list,
    conditional_list=None,        # just for QM9
    num_random_samples=100,
    num_folds=1,
    samples_to_generate=100,
    outputs_root=None,
    defog_src=None,
):
    """
    Runs hyperparameter search (grid or random) for given experiment.
    
    Args:
        mode: "grid" or "random" to specify search type
        dataset: "planar" or "qm9" to determine experiment type
        exp_name: Name of the experiment/checkpoint
        distortions: List of distortion types to use
        num_steps_list: List of number of steps to run experiments with
        eta_list: List of eta values to use
        omega_list: List of omega values to use
        conditional_list: List of conditional values (for QM9)
        num_random_samples: Number of random samples (for random search)
        num_folds: Number of folds for sampling
        samples_to_generate: Number of samples to generate per experiment
        outputs_root: Path to the outputs root directory
        defog_src: Path to the Defog source code
    Returns:
        df_results: DataFrame with all evaluation results
    """
    if outputs_root is None:
        repo_root = Path(__file__).resolve().parents[1]
        outputs_root = repo_root / "outputs" / f"{dataset}_final_gridsearch"

    outputs_root.mkdir(exist_ok=True, parents=True)

    if conditional_list is None:
        conditional_list = [None]


    csv_path = outputs_root / f"summary_results_{dataset}.csv"

    # load existing results if any
    df_existing = pd.read_csv(csv_path) if csv_path.exists() else None

    results = []

    # grid search
    if mode == "grid":
        for num_steps in num_steps_list:

            grid = product(distortions, eta_list, omega_list, conditional_list)

            for distortion, eta, omega, cond in grid:
                print("--------------------------------------------------")
                print(f"Evaluating: steps={num_steps}, distortion={distortion}, eta={eta}, omega={omega}, condition={cond}")
                print("--------------------------------------------------")
                df_row = evaluate_convo(
                    exp_name,
                    num_steps,
                    distortion,
                    float(eta),
                    float(omega),
                    defog_src,
                    outputs_root,
                    num_folds=num_folds,
                    output_folder_name=outputs_root / f"search_{exp_name}_{num_steps}_steps",
                    samples_to_generate=samples_to_generate,
                    condition_value=cond,
                )
                df_row["search_type"] = "grid"
                results.append(df_row)
                
                print(f"Output: \n{df_row}")

    # random search
    elif mode == "random":

        df_random = generate_unique_random_configs(
            num_random_samples,
            csv_path if csv_path.exists() else None,
        )

        for _, row in df_random.iterrows():

            df_row = evaluate_convo(
                exp_name,
                int(row.num_steps),
                row.distortion,
                float(row.eta),
                float(row.omega),
                defog_src,
                outputs_root,
                num_folds=num_folds,
                output_folder_name=outputs_root / f"{dataset}_random_numsteps_{num_steps}",
                samples_to_generate=samples_to_generate,
                condition_value=row.condition_value if "condition_value" in row else None,
            )
            df_row["search_type"] = "random"
            results.append(df_row)

    # concatenate all results
    df_results = pd.concat(results, ignore_index=True)

    # add to existing results
    if df_existing is not None:
        df_results = pd.concat([df_existing, df_results], ignore_index=True)

    df_results.to_csv(csv_path, index=False)

    print(f"[OK] Results saved at {csv_path}")
    return df_results

