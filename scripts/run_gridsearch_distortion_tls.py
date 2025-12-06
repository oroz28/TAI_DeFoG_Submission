from source.hyperparameter_search_helper import run_hyperparameter_search
import numpy as np
from pathlib import Path
import sys

path = Path(__file__).resolve().parents[1]

args = sys.argv

run_hyperparameter_search(
    mode="grid",
    dataset=args[1],
    exp_name=f"{args[1]}.ckpt",
    distortions=["polydec"],
    num_steps_list=[50, 150, 200, 250, 300, 350, 400, 450, 500],  # 50 to 500 with step 50
    eta_list=np.arange(0.0, 251, 50).tolist(),  # 0 to 250 with step 50
    omega_list=np.arange(0.0, 1.1, 0.2).tolist(),  # 0.0 to 1.0 with step 0.2
    # samples_to_generate=100,
    defog_src = path / "external" / "DeFoG" / "src",
    outputs_root=path / f"{args[1]}_gridsearch_output_steps",
)