                      

from __future__ import annotations

import argparse
import os

from src.rbm.calib import load_param_ranges
from src.rbm.hybrid_sweep import run_hybrid_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid optimisation parameter sweep")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the simulation config",
    )
    parser.add_argument(
        "--observed",
        default="data/kaibab_deer.csv",
        help="Observed deer CSV path",
    )
    parser.add_argument(
        "--outdir",
        default="experiments/hybrid_sweep_full",
        help="Directory to store sweep artefacts",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output directory before running",
    )
    args = parser.parse_args()

    if args.overwrite and os.path.exists(args.outdir):
        import shutil

        shutil.rmtree(args.outdir)

    ranges = load_param_ranges(args.config)
    summary = run_hybrid_sweep(
        config_path=args.config,
        param_ranges=ranges,
        random_trials_options=[100, 500, 2000],
        bo_iteration_options=[10, 30, 60],
        warm_start_options=[1, 5, 10, 20],
        acq_funcs=["EI", "PI", "LCB"],
        seeds=[42, 77],
        out_dir=args.outdir,
        observed_csv_path=args.observed,
        progress_interval_sec=5.0,
    )
    print(f"Sweep complete. Results written to {args.outdir}/summary.json")
    print("Aggregate sample:")
    for key in list(summary["aggregates"])[:5]:
        print(key, summary["aggregates"][key])


if __name__ == "__main__":
    main()
