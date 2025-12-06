from source.hyperparameter_search_helper import run_hyperparameter_search
import numpy as np
from pathlib import Path
path = Path(__file__).resolve().parents[1] / "external" / "DeFoG" / "src"

# Ejecutar gridsearch para el conjunto de datos planar
run_hyperparameter_search(
    mode="grid",
    dataset="tls",
    exp_name="tls.ckpt",
    distortions=["polydec"],
    num_steps_list=[10,20,30,40,50],
    eta_list=np.arange(0, 251, 50),
    omega_list=np.round(np.arange(0, 1.1, 0.2), 4),
    samples_to_generate=100,
    defog_src = path,
)
