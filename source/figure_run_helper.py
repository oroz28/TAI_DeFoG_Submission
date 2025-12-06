from pathlib import Path
from .run_defog_helper import run_defog_experiments
from itertools import product


def run_figure_2a(dataset: str, outputs_dir: Path, defog_src_path: Path, num_steps_list=None):
    """
    Run experiments for Figure 2a varying configurations like distortion, eta, and omega.
    
    Args:
        dataset: "planar" or "qm9"
        outputs_dir: Path to the outputs directory
        defog_src_path: Path to the Defog source code
        num_steps_list: list of number of steps to run experiments with
    """
    if num_steps_list is None:
        num_steps_list = [5, 10, 50, 100, 1000]

    if dataset == "qm9":
        experiment = ["qm9_no_h.ckpt"]
        checkpoint = "/home/group-2/Submission_Code/checkpoints/qm9_no_h.ckpt"
        configs = {
            "vanilla": {"sample.eta": "0.0", "sample.omega": "0.0", "sample.time_distortion": "identity"},
            "+distortion": {"sample.eta": "0.0", "sample.omega": "0.0"},
            "++target_guidance": {"sample.eta": "0.0", "sample.omega": "0.05"},
            "+++stochasticity": {"sample.omega": "0.05", "sample.eta": "50"},
        }
    elif dataset == "planar":
        experiment = ["planar.ckpt"]
        checkpoint = "/home/group-2/Submission_Code/checkpoints/planar.ckpt"
        configs = {
            "vanilla": {"sample.eta": "0.0", "sample.omega": "0.0", "sample.time_distortion": "identity"},
            "+distortion": {"sample.eta": "0.0", "sample.omega": "0.0"},
            "++target_guidance": {"sample.eta": "0.0"},
            "+++stochasticity": {},
        }
    else:
        raise ValueError("Dataset must be 'planar' or 'qm9'.")

    overrides_per_experiment = {}
    experiments_list = []

    for config_name, config_flags in configs.items():
        for steps in num_steps_list:
            run_dir_name = f"2_a_{config_name}_{dataset}_{steps}_steps"
            print(f"Preparing experiment: {run_dir_name}")
            overrides = {
                "general.test_only": checkpoint,
                "sample.sample_steps": steps,
                "hydra.run.dir": str((outputs_dir / run_dir_name).resolve()),
            }
            overrides.update(config_flags)
            key = f"{config_name}_{steps}"
            overrides_per_experiment[experiment[0]] = overrides
            experiments_list.append(key)

            run_defog_experiments(
                experiments=experiment,
                num_steps_list=[steps],
                outputs_dir=(outputs_dir / run_dir_name),
                defog_src_path=defog_src_path,
                overrides_per_experiment=overrides_per_experiment,
                name_prefix=f"2_a_{dataset}",
                verbose=True,
                extra_args=["visualization=False"],
            )
    
def run_figure3(
    outputs_dir: Path,
    defog_src_path: Path,
    num_steps_list=None,
    dataset="planar",
):
    """
    Run experiments for Figure 3 varying distortion types, omega, and eta.
    
    Args:
        outputs_dir: Path to the outputs directory
        defog_src_path: Path to the Defog source code
        num_steps_list: list of number of steps to run experiments with
        dataset: "planar" or "qm9"
    """

    if num_steps_list is None:
        num_steps_list = [50, 1000]

    distortions = ["polydec", "polyinc", "revcos", "cos", "identity"]
    omegas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    etas = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

    experiments = [f"{dataset}.ckpt"]
    overrides_list = []

    # distortion experiments
    for steps, distortion in product(num_steps_list, distortions):
        overrides_list.append({
            "sample.sample_steps": steps,
            "sample.eta": 0.0,
            "sample.omega": 0.0 if dataset == "planar" else 0.05,
            "sample.time_distortion": distortion,
            "hydra.run.dir": str(outputs_dir / f"f3_distortion_{distortion}_{steps}_steps")
        })

    if dataset == "planar":
        # omega experiments
        for steps, omega in product(num_steps_list, omegas):
            overrides_list.append({
                "sample.sample_steps": steps,
                "sample.eta": 0.0,
                "sample.omega": omega,
                "sample.time_distortion": "identity",
                "hydra.run.dir": str(outputs_dir / f"f3_omega_{omega}_{steps}_steps")
            })

        # eta experiments
        for steps, eta in product(num_steps_list, etas):
            overrides_list.append({
                "sample.sample_steps": steps,
                "sample.eta": eta,
                "sample.omega": 0.0,
                "sample.time_distortion": "identity",
                "hydra.run.dir": str(outputs_dir / f"f3_eta_{eta}_{steps}_steps")
            })

    # run experiments
    for i, overrides in enumerate(overrides_list):
        run_defog_experiments(
            experiments=experiments,
            num_steps_list=[overrides.pop("sample.sample_steps")],
            outputs_dir=outputs_dir,
            defog_src_path=defog_src_path,
            overrides_per_experiment={f"{dataset}.ckpt": overrides},
            name_prefix=f"figure3_{i}",
            verbose=True,
            extra_args=["visualization=False"],
        )
            
def run_distortions(
    outputs_dir: Path,
    defog_src_path: Path,
    steps_list=None,
    a_polydec_list=None,
    a_cos_list=None,
    a_log_list=None,
):
    """
    Run distortion experiments varying parameters for different distortion types.
    
    Args:
        outputs_dir: Path to the outputs directory
        defog_src_path: Path to the Defog source code
        steps_list: list of number of steps to run experiments with
        a_polydec_list: list of 'a' parameters for polydec distortion
        a_cos_list: list of 'a' parameters for cos distortion
        a_log_list: list of 'a' parameters for log distortion
    """
    if steps_list is None:
        steps_list = [50, 1000]
    if a_polydec_list is None:
        a_polydec_list = [1.5, 2.5, 3.5, 4.5]
    if a_cos_list is None:
        a_cos_list = [1.4, 2.2, 3, 3.8]
    if a_log_list is None:
        a_log_list = [1, 5, 10, 100]

    experiments = ["planar"]
    overrides_list = []

    # polydec distortions
    for a_polydec, steps in product(a_polydec_list, steps_list):
        distortion = f"new_polydec_{a_polydec}"
        overrides_list.append({
            "sample.sample_steps": steps,
            "sample.eta": 0.0,
            "sample.omega": 0.0,
            "sample.time_distortion": distortion,
            "general.num_sample_fold": 5,
            "hydra.run.dir": str(outputs_dir / f"distortion_experiment_{distortion}_{steps}_steps")
        })

    # cos distortions
    for a_cos, steps in product(a_cos_list, steps_list):
        distortion = f"new_cos_{a_cos}"
        overrides_list.append({
            "sample.sample_steps": steps,
            "sample.eta": 0.0,
            "sample.omega": 0.0,
            "sample.time_distortion": distortion,
            "general.num_sample_fold": 5,
            "hydra.run.dir": str(outputs_dir / f"distortion_experiment_{distortion}_{steps}_steps")
        })

    # log distortions
    for a_log, steps in product(a_log_list, steps_list):
        distortion = f"log_{a_log}"
        overrides_list.append({
            "sample.sample_steps": steps,
            "sample.eta": 0.0,
            "sample.omega": 0.0,
            "sample.time_distortion": distortion,
            "general.num_sample_fold": 5,
            "hydra.run.dir": str(outputs_dir / f"distortion_experiment_{distortion}_{steps}_steps")
        })

    # Ejecutar experimentos
    for i, overrides in enumerate(overrides_list):
        run_defog_experiments(
            experiments=experiments,
            num_steps_list=[overrides.pop("sample.sample_steps")],
            outputs_dir=outputs_dir,
            defog_src_path=defog_src_path,
            overrides_per_experiment={"planar": overrides},
            name_prefix=f"distortion_{i}",
            verbose=True
        )
