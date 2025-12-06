from source.hyperparameter_search_helper import run_hyperparameter_search
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[1] / "external" / "DeFoG" / "src"

# Ejecutar gridsearch de hiperparámetros para el dataset qm9_with_h condicional
run_hyperparameter_search(
    mode="grid",
    dataset="qm9_with_h",
    exp_name="qm9_with_h_conditional.ckpt",
    distortions=["polydec"],
    # num_steps_list=[100],
    num_steps_list = np.arange(150, 501, 50).tolist(),
    # eta_list=[0.0],
    # omega_list=[1.0],
    eta_list=[0, 25, 50, 100, 150, 200, 250],  # 0 to 10 with step 1
    omega_list=np.arange(0.0, 1.1, 0.2).tolist(),  # 0.0 to 1.0 with step 0.2
    conditional_list=[-400],
    samples_to_generate=2048,
    defog_src = path,
)
