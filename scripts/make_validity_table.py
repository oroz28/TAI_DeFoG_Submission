from pathlib import Path
from source.table_helper import make_results_table
import sys


args = sys.argv

repo_root = Path(__file__).resolve().parents[1]
results_dir = repo_root / "results" / "tables"
results_dir.mkdir(parents=True, exist_ok=True)

experiments = [args[1]]  # e.g., "qm9" or "planar"
steps = 50

for exp in experiments:
    folder = repo_root / "outputs"
    print(folder)

    # for planar
    if exp == "planar":
        matches = list(folder.glob(f"{exp}_samples_subgraphs_subgraph_graph_00*/**/test_*.txt"))
    # for qm9
    elif exp == "qm9":
        matches = list(folder.glob(f"{exp}_samples_subgraphs_subgraph_graph_*/**/test_*.txt"))
    # for planar cv
    elif exp == "planar_cv":
        matches = list(folder.glob(f"cv_planar*/**/epoch0*.txt"))
        matches.extend(list(folder.glob(f"t1_planar_50_steps/**/epoch0*.txt")))
    # for qm9 with h
    elif exp == "qm9_with_h":
        matches = list(folder.glob(f"{exp}_samples_subgraphs_graph*/**/test_*.txt"))
    # for qm9 with h cv
    elif exp == "qm9_with_h_cv":
        matches = list(folder.glob(f"cv_qm9*/**/epoch0*.txt"))
        matches.extend(list(folder.glob(f"qm9_with_h_baseline/**/epoch0*.txt")))
        
    print(len(matches))

    if not matches:
        print(f"No file found for {exp} with {steps} steps")
        continue

    file_map = {}
    for f in matches:
        subgraph_folder_name = f.parent.parent.name
        
        if exp == "planar_cv" or exp == "qm9_with_h_cv":
            fold_name = f.stem  # epoch0_res_fold3_
            key = (f"{exp}_{subgraph_folder_name}_{fold_name}", steps)
        else:
            key = (f"{exp}_{subgraph_folder_name}", steps)

        file_map[key] = f
        
    table_file = results_dir / f"validity_table_def_{exp}.csv"
    df = make_results_table(file_map, table_file, dataset=exp, validity=True)
    print(df)
