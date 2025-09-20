#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_path():
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def run_one(cfg_path: str, out_csv: str):
    from src.rbm.run_one import run_once
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    row = run_once(cfg_path, out_csv)
    print("Single-year written:", out_csv)
    print(json.dumps(row, indent=2))


def run_years(cfg_path: str, out_csv: str):
    from src.rbm.run_years import run_years
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows = run_years(cfg_path, out_csv)
    print("Multi-year written:", out_csv, "years:", len(rows))


def run_calibration(cfg_path: str, out_dir: str, trials: int, seed: int, observed_csv: str, compare: bool):
    from src.rbm.calib import calibrate_random_search, calibrate_compare_interpolation
    ranges = {
        "vegetation": {"vegRate": (0.05, 0.25), "capMax": (60000, 180000), "browse": (0.0, 0.2)},
        "deer": {"birth": (0.6, 1.2), "surv": (0.6, 0.95)},
        "predation": {"predAtk": (1e-6, 1e-3), "predCap": (0.05, 0.6), "predEff": (1e-4, 5e-3)},
        "predators": {"mort": (0.05, 0.3)},
    }
    os.makedirs(out_dir, exist_ok=True)
    if compare:
        res = calibrate_compare_interpolation(
            cfg_path,
            param_ranges=ranges,
            trials=trials,
            seed=seed,
            out_dir=os.path.join(out_dir, "cmp"),
            observed_csv_path=observed_csv,
        )
        print("Calibration compare (interp_on/off):")
        print(json.dumps(res, indent=2))
    else:
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
        print("Best score:", best_score)
        print(json.dumps(best_params, indent=2))


def run_stochastic(cfg_path: str, out_csv: str, repeats: int, seed: int):
    from src.rbm.run_stochastic import run_years_stochastic
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows = run_years_stochastic(cfg_path, out_csv, repeats=repeats, seed=seed)
    print("Stochastic bands written:", out_csv, "years:", len(rows))


def main():
    _ensure_path()
    parser = argparse.ArgumentParser(description="Kaibab rule-based simulation CLI")
    parser.add_argument("action", choices=["one", "years", "calib", "calib-compare", "stoch", "all"], help="What to run")
    parser.add_argument("--config", default=os.path.join(_project_root(), "configs", "base.yaml"), dest="config", help="Path to config file")
    parser.add_argument("--outdir", default=os.path.join(_project_root(), "experiments"), dest="outdir", help="Output directory")
    parser.add_argument("--observed", default=os.path.join(_project_root(), "data", "kaibab_deer.csv"), dest="observed", help="Observed deer CSV path")
    parser.add_argument("--trials", type=int, default=1000, help="Calibration trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--repeats", type=int, default=200, help="Stochastic repeats")
    args = parser.parse_args()

    cfg = os.path.abspath(args.config)
    outdir = os.path.abspath(args.outdir)
    observed = os.path.abspath(args.observed)

    if args.action in ("one", "all"):
        run_one(cfg, os.path.join(outdir, "run_one.csv"))

    if args.action in ("years", "all"):
        run_years(cfg, os.path.join(outdir, "run_years.csv"))

    if args.action in ("calib", "all"):
        run_calibration(cfg, os.path.join(outdir, "calib"), args.trials, args.seed, observed, compare=False)

    if args.action == "calib-compare":
        run_calibration(cfg, os.path.join(outdir, "calib"), args.trials, args.seed, observed, compare=True)

    if args.action in ("stoch", "all"):
        run_stochastic(cfg, os.path.join(outdir, "run_stochastic.csv"), args.repeats, args.seed)


if __name__ == "__main__":
    main()


