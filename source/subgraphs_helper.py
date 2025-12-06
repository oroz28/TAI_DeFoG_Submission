from pathlib import Path
from .run_defog_helper import run_defog_experiments

def run_subgraphs(
    dataset: str, 
    outputs_dir: Path = Path("/home/group-2/Submission_Code/outputs"), 
    defog_src_path: Path = Path("/home/group-2/DeFoG/src"), 
    num_steps: list[int] = [50],
    num_folds: int = 5,
    subgraph_files: list[Path] = None,
    samples_per_fold: int = None,
):
    """
    Runs Defog experiments on fixed subgraphs for the specified dataset.
    
    Args:
        dataset: "planar", "qm9", or "qm9_with_h"
        outputs_dir: Path to the outputs directory
        defog_src_path: Path to the Defog source code
        num_steps: list of number of steps to run experiments with
        num_folds: number of folds for sampling
        subgraph_files: list of subgraph files to use (if None, uses all in default dir)
        samples_per_fold: number of samples to generate per fold (if None, uses default)
    """

    # select subgraph directory and checkpoint based on dataset
    #####################################################
    
    ############ FALTA
    
    #####################################################
    if dataset == "qm9":
        subgraphs_dir = Path("/home/group-2/DeFoG/src/fixed_subgraphs_qm9_pt")
        checkpoint = "/home/group-2/Submission_Code/checkpoints/qm9_no_h.ckpt"
    elif dataset == "planar":
        subgraphs_dir = Path("/home/group-2/TAI_defog/tai_defog/validation_planar_pt/")
        checkpoint = "/home/group-2/Submission_Code/checkpoints/planar.ckpt"
    elif dataset == "qm9_with_h":
        subgraphs_dir = Path("/home/group-2/TAI_defog/tai_defog/fixed_subgraphs_selected_qm9_pt")
        checkpoint = "/home/group-2/Submission_Code/checkpoints/qm9_with_h.ckpt"
        experiment = "qm9_with_h"
    else:
        raise ValueError("Dataset unknown.")

    if subgraph_files is None:
        subgraph_files = sorted(subgraphs_dir.glob("**/*.pt"))
        cv = ""
    else:
        subgraph_files = [Path(f) for f in subgraph_files]
        cv = "cv_"
        
    experiment = checkpoint.split("/")[-1]

    print(f"Found {len(subgraph_files)} subgraphs.")

    # run experiments for each subgraph file
    for sg_file in subgraph_files:

        run_dir_name = f"{cv}{dataset}_samples_subgraphs_{sg_file.stem}"
        out_dir = (outputs_dir / run_dir_name).resolve()

        overrides = {
            "general.num_sample_fold": num_folds,
            "general.test_only": checkpoint,
            "+fixed_subgraph_file": sg_file,
        }
        if samples_per_fold is not None:
            overrides["general.final_model_samples_to_generate"] = samples_per_fold
        
        run_defog_experiments(
            experiments=[experiment],
            num_steps_list=num_steps,
            outputs_dir=out_dir,
            defog_src_path=defog_src_path,
            overrides_per_experiment={experiment: overrides},
            name_prefix=f"{dataset}_subgraphs",
            verbose=True
        )

