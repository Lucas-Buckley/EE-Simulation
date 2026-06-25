#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import site
import sys


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_path():
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    # Ensure user site-packages (where pip --user installs go) is importable
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        user_paths = [user_site]
    else:
        user_paths = list(user_site)
    for path in user_paths:
        if path not in sys.path:
            sys.path.append(path)


def _prepare_output(path: str, *, is_dir: bool) -> None:
    """Remove any previous output and make sure directories exist."""

    if os.path.exists(path):
        if is_dir:
            shutil.rmtree(path)
        else:
            os.remove(path)
    if is_dir:
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def run_one(cfg_path: str, out_csv: str):
    from src.rbm.run_one import run_once
    _prepare_output(out_csv, is_dir=False)
    row = run_once(cfg_path, out_csv)
    print("Single-year written:", out_csv)
    print(json.dumps(row, indent=2))


def run_years(cfg_path: str, out_csv: str):
    from src.rbm.run_years import run_years
    _prepare_output(out_csv, is_dir=False)
    rows = run_years(cfg_path, out_csv)
    print("Multi-year written:", out_csv, "years:", len(rows))


def run_calibration(
    cfg_path: str,
    base_out_dir: str,
    trials: int,
    seed: int,
    observed_csv: str,
    compare: bool,
    optimizer: str,
    acq_func: str,
    hybrid_random_trials: int,
    hybrid_bo_iterations: int,
    hybrid_warm_k: int,
):
    from src.rbm.calib import load_param_ranges

    ranges = load_param_ranges(cfg_path)
    label = optimizer
    out_dir = os.path.join(base_out_dir, label)
    _prepare_output(out_dir, is_dir=True)
    if compare:
        if optimizer != "random":
            raise ValueError("Calibration compare currently supports only the random search optimizer")
        from src.rbm.calib import calibrate_compare_interpolation

        res = calibrate_compare_interpolation(
            cfg_path,
            param_ranges=ranges,
            trials=trials,
            seed=seed,
            out_dir=os.path.join(out_dir, "cmp"),
            observed_csv_path=observed_csv,
        )
        print(f"Calibration compare (interp_on/off) results saved to {out_dir}")
        print(json.dumps(res, indent=2))
    else:
        import time

        start = time.time()
        if optimizer == "bayes":
            from src.rbm.bayes_optimize import calibrate_bayes_opt

            best_params, best_score = calibrate_bayes_opt(
                config_path=cfg_path,
                observed_years=None,
                observed_deer=None,
                param_ranges=ranges,
                iterations=trials,
                seed=seed,
                out_dir=out_dir,
                observed_csv_path=observed_csv,
                interpolate_observed=True,
                acq_func=acq_func,
            )
        elif optimizer == "hybrid":
            from src.rbm.hybrid import calibrate_hybrid

            best_params, best_score, stage_info = calibrate_hybrid(
                config_path=cfg_path,
                param_ranges=ranges,
                random_trials=hybrid_random_trials,
                bo_iterations=hybrid_bo_iterations,
                warm_start_k=hybrid_warm_k,
                seed=seed,
                out_dir=out_dir,
                observed_years=None,
                observed_deer=None,
                observed_csv_path=observed_csv,
                interpolate_observed=True,
                acq_func=acq_func,
            )
        else:
            from src.rbm.calib import calibrate_random_search

            best_params, best_score = calibrate_random_search(
                cfg_path,
                observed_years=None,
                observed_deer=None,
                param_ranges=ranges,
                trials=trials,
                seed=seed,
                out_dir=out_dir,
                observed_csv_path=observed_csv,
                interpolate_observed=True,
            )
        duration = time.time() - start
        print(f"Best score ({label}):", best_score)
        print("Results directory:", out_dir)
        print(f"Elapsed time ({label}): {duration:.2f} seconds")
        print(json.dumps(best_params, indent=2))
        if optimizer == "hybrid":
            print("Stage summaries:")
            print(json.dumps(stage_info, indent=2))


def run_stochastic(cfg_path: str, out_csv: str, repeats: int, seed: int):
    from src.rbm.run_stochastic import run_years_stochastic
    _prepare_output(out_csv, is_dir=False)
    rows = run_years_stochastic(cfg_path, out_csv, repeats=repeats, seed=seed)
    print("Stochastic bands written:", out_csv, "years:", len(rows))


def main():
    _ensure_path()
    parser = argparse.ArgumentParser(description="Kaibab rule-based simulation CLI")
    parser.add_argument("action", choices=["one", "years", "calib", "calib-compare", "stoch", "all"], help="What to run")
    parser.add_argument("--config", default=os.path.join(_project_root(), "configs", "base.yaml"), dest="config", help="Path to config file")
    parser.add_argument("--outdir", default=os.path.join(_project_root(), "experiments"), dest="outdir", help="Output directory")
    parser.add_argument("--observed", default=os.path.join(_project_root(), "data", "kaibab_deer.csv"), dest="observed", help="Observed deer CSV path")
    parser.add_argument("--trials", type=int, default=100, help="Calibration trials/iterations (random or Bayesian)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--repeats", type=int, default=200, help="Stochastic repeats")
    parser.add_argument(
        "--optimizer",
        choices=["random", "bayes", "hybrid"],
        default="random",
        help="Calibration optimizer to use",
    )
    parser.add_argument(
        "--acq",
        choices=["EI", "PI", "LCB"],
        default="EI",
        help="Acquisition function (when --optimizer bayes)",
    )
    parser.add_argument("--hybrid-random-trials", type=int, default=200, help="Random-search trials for hybrid mode")
    parser.add_argument("--hybrid-bo-iterations", type=int, default=60, help="Bayesian refinement iterations for hybrid mode")
    parser.add_argument("--hybrid-warm-k", type=int, default=10, help="Number of top random trials to warm-start the Bayesian phase")
    args = parser.parse_args()

    cfg = os.path.abspath(args.config)
    outdir = os.path.abspath(args.outdir)
    observed = os.path.abspath(args.observed)

    if args.action == "calib-compare" and args.optimizer != "random":
        parser.error("calib-compare currently requires --optimizer random")

    if args.action in ("one", "all"):
        run_one(cfg, os.path.join(outdir, "run_one.csv"))

    if args.action in ("years", "all"):
        run_years(cfg, os.path.join(outdir, "run_years.csv"))

    if args.action in ("calib", "all"):
        run_calibration(
            cfg,
            os.path.join(outdir, "calib"),
            args.trials,
            args.seed,
            observed,
            compare=False,
            optimizer=args.optimizer,
            acq_func=args.acq,
            hybrid_random_trials=args.hybrid_random_trials,
            hybrid_bo_iterations=args.hybrid_bo_iterations,
            hybrid_warm_k=args.hybrid_warm_k,
        )
        if args.action == "all":
            for alt in (opt for opt in ["random", "bayes", "hybrid"] if opt != args.optimizer):
                run_calibration(
                    cfg,
                    os.path.join(outdir, "calib"),
                    args.trials,
                    args.seed,
                    observed,
                    compare=False,
                    optimizer=alt,
                    acq_func=args.acq,
                    hybrid_random_trials=args.hybrid_random_trials,
                    hybrid_bo_iterations=args.hybrid_bo_iterations,
                    hybrid_warm_k=args.hybrid_warm_k,
                )

    if args.action == "calib-compare":
        run_calibration(
            cfg,
            os.path.join(outdir, "calib"),
            args.trials,
            args.seed,
            observed,
            compare=True,
            optimizer=args.optimizer,
            acq_func=args.acq,
            hybrid_random_trials=args.hybrid_random_trials,
            hybrid_bo_iterations=args.hybrid_bo_iterations,
            hybrid_warm_k=args.hybrid_warm_k,
        )

    if args.action in ("stoch", "all"):
        run_stochastic(cfg, os.path.join(outdir, "run_stochastic.csv"), args.repeats, args.seed)


if __name__ == "__main__":
    main()
