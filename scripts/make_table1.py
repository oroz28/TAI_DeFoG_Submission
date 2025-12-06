from pathlib import Path
from source.table_helper import make_results_table
import sys

# parameters
version = sys.argv[1] if len(sys.argv) > 1 else "new"  # default "new"
repo_root = Path(__file__).resolve().parents[1]
table_file = repo_root / "results" / "tables" / f"table1_{version}.csv"

# experiments and steps based on version
if version == "old":
    experiments = ["planar-59999-14h.ckpt", "tree-19999-5h.ckpt", "sbm-31999-40h.ckpt"]
    prefix = "old_ckpts"
elif version == "new":
    experiments = ["planar.ckpt", "tree.ckpt", "sbm.ckpt"]
    prefix = "new_ckpts"
else:
    raise ValueError(f"Unknown version: {version}")

steps_list = [50, 1000]

# build file map
file_map = {}
for exp in experiments:
    for steps in steps_list:
        folder = repo_root / "outputs" / f"t1_{prefix}_{exp}_{steps}_steps"
        print(folder)
        matches = list(folder.glob("test_epoch*.txt"))
        print(matches)
        if matches:
            file_map[(exp, steps)] = matches[0]

# generate table
df = make_results_table(file_map, table_file, validity=False)
print(df)
