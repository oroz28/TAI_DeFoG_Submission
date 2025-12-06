from source.hyperparameter_search_helper import run_hyperparameter_search
from pathlib import Path
path = Path(__file__).resolve().parents[1] / "external" / "DeFoG" / "src"

# Ejecutar randomsearch de hiperparámetros para el dataset planar
run_hyperparameter_search(
    mode="random",
    dataset="planar",
    exp_name="planar.ckpt",
    distortions=["polydec"],
    num_steps_list=[5,10,20,30,40,50],
    eta_list=[],          # ignored
    omega_list=[],        # ignored
    num_random_samples=100,
    samples_to_generate=100,
    defog_src = path,
)
    