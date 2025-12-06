from source.hyperparameter_search_helper import run_hyperparameter_search
from pathlib import Path
path = Path(__file__).resolve().parents[1] / "external" / "DeFoG" / "src"

# run grid search for qm9 dataset
run_hyperparameter_search(
    mode="grid",
    dataset="qm9",
    exp_name="qm9_conditional.ckpt",
    distortions=["polydec"],
    num_steps_list=[5,10,20,30,40,50],
    eta_list=[0.0],
    omega_list=[0.0],
    conditional_list=[-400],
    samples_to_generate=2048,
    defog_src = path,
)
