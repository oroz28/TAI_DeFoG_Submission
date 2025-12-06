from source.hyperparameter_search_helper import make_objective
from pathlib import Path
import optuna
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
defoq_src = repo_root / "external" / "DeFoG" / "src"
outputs_dir = repo_root / "tls"
outputs_dir.mkdir(exist_ok=True)

exp_name = "tls.ckpt"
distortions = ["polydec"]
eta_bounds = (0.0, 0.0)#200.0)
omega_bounds = (0.0, 0.05)#1.0)
num_folds = 1

num_steps_list =  list(range(20, 51, 10))
num_steps_list =  [50, 100, 300, 500]

n_trials = 1

all_results = []

for num_steps in num_steps_list:
    objective = make_objective(
        exp_name, defoq_src, outputs_dir,
        distortions, eta_bounds, omega_bounds,
        num_steps, num_folds
    )


    study = optuna.create_study(
        study_name=f"defog_steps{num_steps}",
        load_if_exists=True,
        direction="maximize"
    )

    def stop_when_target_reached(study: optuna.Study, trial: optuna.Trial):
        if study.best_value is not None and study.best_value >= 100.0:
            study.stop()

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, callbacks=[stop_when_target_reached])

    best_params = study.best_params
    best_score = study.best_value

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    df["num_steps"] = num_steps
    df.to_csv(outputs_dir / f"bo_results_steps{num_steps}.csv", index=False)

    all_results.append({
        "num_steps": num_steps,
        **best_params,
        "best_score": best_score
    })

summary_df = pd.DataFrame(all_results)
summary_df.to_csv(outputs_dir / "bo_summary_best_params.csv", index=False)
print(summary_df)

