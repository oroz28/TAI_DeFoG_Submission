#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from source.hyperparameter_search_helper import run_hyperparameter_search


def parse_list(arg, cast_fn=float):
    """
    Parses a string argument into a list.
    The argument can be:
    - A comma-separated list of values (e.g., "0.0,0.2,0.4")
    - A range with start:end:step (e.g., "0:1:0.2")
    - An empty string, which returns None
    
    Args:
        arg (str): The input string argument.
        cast_fn (function): Function to cast the string values (default: float).
    Returns:
        list or None: The parsed list of values or None if input is empty.
    """
    if arg is None or arg == "":
        return None
    
    if ":" in arg:
        start, end, step = map(cast_fn, arg.split(":"))
        # construct range inclusive of end
        return np.arange(start, end + step, step).tolist()
    
    return [cast_fn(x) for x in arg.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Unified DeFoG hyperparameter search runner")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["grid", "random"], required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--distortions", type=str, default="polydec")

    parser.add_argument("--num_steps_list", type=str, required=False, default="")
    parser.add_argument("--eta_list", type=str, required=False, default="")
    parser.add_argument("--omega_list", type=str, required=False, default="")
    parser.add_argument("--conditional_list", type=str, required=False, default="")
    
    parser.add_argument("--samples_to_generate", type=int, required=False, default=100)
    parser.add_argument("--num_random_samples", type=int, required=False, default=100)

    parser.add_argument("--outputs_root", type=str, required=False)
    parser.add_argument("--defog_src", type=str, required=False)

    args = parser.parse_args()

    # default paths
    base_path = Path(__file__).resolve().parents[1]
    defog_src = Path(args.defog_src) if args.defog_src else base_path / "external" / "DeFoG" / "src"
    outputs_root = Path(args.outputs_root) if args.outputs_root else None

    run_hyperparameter_search(
        mode=args.mode,
        dataset=args.dataset,
        exp_name=args.exp_name,
        distortions=[args.distortions],
        num_steps_list=parse_list(args.num_steps_list, cast_fn=int),
        eta_list=parse_list(args.eta_list, cast_fn=float),
        omega_list=parse_list(args.omega_list, cast_fn=float),
        conditional_list=parse_list(args.conditional_list, cast_fn=float),
        samples_to_generate=args.samples_to_generate,
        num_random_samples=args.num_random_samples,
        defog_src=defog_src,
        outputs_root=outputs_root,
    )


if __name__ == "__main__":
    main()
