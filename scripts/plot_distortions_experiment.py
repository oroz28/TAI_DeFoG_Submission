from pathlib import Path
from source.figure_plot_helper import collect_metrics, plot_curves

repo_root = Path(__file__).resolve().parents[1]
outputs_dir = repo_root / "outputs"
results_dir = repo_root / "results" / "plots"
results_dir.mkdir(exist_ok=True, parents=True)

steps_list = [50, 1000]

# define distortions
as_polydec = [1.5, 2.5, 3.5, 4.5]
as_cos = [1.4, 2.2, 3, 3.8]
as_log = [1, 5, 10, 100]

distortions_new = [f"new_polydec_{x}" for x in as_polydec] + \
                  [f"new_cos_{x}" for x in as_cos] + \
                  [f"log_{x}" for x in as_log]

distortions_old = ["polyinc", "revcos", "identity", "cos", "polydec"]

# generate plots
for step in steps_list:
    # old metrics
    ratio_old, vun_old = collect_metrics(outputs_dir, "distortion", distortions_old, step, previous_experiment=True)
    # new metrics
    ratio_new, vun_new = collect_metrics(outputs_dir, "distortion", distortions_new, step, previous_experiment=False)

    # combine
    ratio_means = ratio_old + ratio_new
    vun_means = vun_old + vun_new
    distortions_combined = distortions_old + distortions_new

    out_path = results_dir / f"distortions_experiment_{step}_steps.png"
    plot_curves(
        distortions_combined,
        ratio_means,
        vun_means,
        title=f"Distortion Effects (planar, {step} steps)",
        out_path=out_path,
        xlabel="Time Distortion Type",
        steps=step,
        exp_type="distortion"
    )
    print(f"Plot saved to {out_path}")
    