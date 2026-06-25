from __future__ import annotations

import argparse
import json
import csv
import os
import random
import time
from dataclasses import asdict
from typing import Dict, Any, Tuple, List, Union

from .config import load_config
from .run_years import run_years
from .metrics import (
    compute_scaled_mse,
    compute_peak_timing_error,
    compute_peak_height_error,
    compute_crash_ratio_error,
)
from .observed import load_deer_observed_csv


ParamRanges = Dict[str, Dict[str, Tuple[float, float]]]

# Default parameter ranges used when the config file does not define calibRanges.
DEFAULT_PARAM_RANGES: ParamRanges = {
    "vegetation": {"vegRate": (0.01, 0.6), "capMax": (80_000.0, 400_000.0), "browse": (0.0, 0.6)},
    "deer": {"birth": (0.5, 1.4), "surv": (0.5, 0.98)},
    "predation": {"predAtk": (5e-7, 5e-3), "predCap": (0.05, 0.9), "predEff": (5e-5, 1e-2)},
    "predators": {"mort": (0.03, 0.5)},
}


def _copy_ranges(ranges: ParamRanges) -> ParamRanges:
    """Return a fresh copy of parameter bounds so callers can mutate safely."""

    return {
        group: {name: (float(bounds[0]), float(bounds[1])) for name, bounds in subgroup.items()}
        for group, subgroup in ranges.items()
    }


def load_param_ranges(config_path: str, fallback: ParamRanges | None = None) -> ParamRanges:
    """Load calibration ranges from config.calibRanges or fall back to defaults.

    Inputs:
      - config_path: path to the JSON config file.
      - fallback: optional dictionary to use when calibRanges is missing.

    Output:
      - Nested dictionary mapping group and parameter name to (low, high) tuples.
    """

    base = _copy_ranges(fallback or DEFAULT_PARAM_RANGES)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return base

    raw_ranges = raw.get("calibRanges")
    if not isinstance(raw_ranges, dict):
        return base

    ranges: ParamRanges = {}
    for group, subgroup in raw_ranges.items():
        if not isinstance(subgroup, dict):
            continue
        ranges[group] = {}
        for name, bounds in subgroup.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                continue
            low = float(bounds[0])
            high = float(bounds[1])
            if low > high:
                low, high = high, low
            ranges[group][name] = (low, high)

    if not ranges:
        return base
    return ranges


def _deepcopy_params(params: Any) -> Any:
    """Return a deep-copied plain dict version of params (dataclass or dict).

    Ensures we can mutate candidate parameter sets without affecting the original.
    """
    return json.loads(json.dumps(asdict(params))) if not isinstance(params, dict) else json.loads(json.dumps(params))


def _apply_candidate(params_dict: Dict[str, Any], candidate: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Return a copy of params with selected values replaced from a candidate.

    Inputs:
      - params_dict: a nested "parameters" dictionary shaped like this:
            {
              "vegetation": {"vegRate": 0.15, "capMax": 100000, "browse": 0.1},
              "deer": {"birth": 1.0, "surv": 0.85},
              "predation": {"predAtk": 0.0001, "predCap": 0.3, "predEff": 0.0015},
              "predators": {"mort": 0.12}
            }
        You can think of the first layer keys ("vegetation", "deer", ...) as parameter groups, and the
        inner keys (e.g., "vegRate", "birth") as individual parameter names within those groups.

      - candidate: a nested dictionary describing which parameters to change and to what values. For example:
            {
              "vegetation": {"vegRate": 0.18, "capMax": 120000},
              "deer": {"surv": 0.9}
            }
        This means: set vegetation.vegRate to 0.18, vegetation.capMax to 120000, and deer.surv to 0.9.

    Output:
      - a new parameters dictionary with those overrides applied, leaving all others unchanged.
    """
    newp = json.loads(json.dumps(params_dict))
    for group, kv in candidate.items():
        if group not in newp:
            continue
        for name, val in kv.items():
            newp[group][name] = float(val)
    return newp


def _sample_candidate(ranges: ParamRanges, rng: random.Random) -> Dict[str, Dict[str, float]]:
    """Create one random parameter set inside provided ranges.

    Input:
      - ranges: a nested dictionary describing uniform sampling ranges for parameters, e.g.:
            {
              "vegetation": {"vegRate": (0.05, 0.2), "capMax": (80000, 150000)},
              "deer": {"birth": (0.6, 1.2), "surv": (0.6, 0.95)}
            }
      - rng: a random.Random instance used for reproducibility.

    Output:
      - a nested dictionary with sampled numeric values in the same shape as ranges.
    """
    cand: Dict[str, Dict[str, float]] = {}
    for group, kv in ranges.items():
        cand[group] = {}
        for name, (lo, hi) in kv.items():
            cand[group][name] = rng.uniform(lo, hi)
    return cand


def _extract_deer_series(rows: List[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
    """Extract (years, deer) from run_years output rows using deerNxt where available."""
    years = [int(r["year"]) for r in rows]
    deer = [float(r.get("deerNxt", r.get("deer", 0.0))) for r in rows]
    return years, deer


def score_fit(observed_years: List[int], observed_deer: List[float], sim_years: List[int], sim_deer: List[float]) -> float:
    """Compute a single numerical score (lower is better) comparing simulated vs. observed deer.

    We first align by common years to make sure both lists refer to the same timestamps, then compute
    the scaled mean squared error so scores are comparable even if magnitudes differ.
    """
    # Align by intersection of years to be safe
    year_to_obs = {y: v for y, v in zip(observed_years, observed_deer)}
    year_to_sim = {y: v for y, v in zip(sim_years, sim_deer)}
    years = sorted(set(year_to_obs.keys()) & set(year_to_sim.keys()))
    obs = [year_to_obs[y] for y in years]
    sim = [year_to_sim[y] for y in years]
    return compute_scaled_mse(obs, sim)


def calibrate_random_search(
    config_path: str,
    observed_years: List[int] | None,
    observed_deer: List[float] | None,
    param_ranges: ParamRanges,
    trials: int = 100,
    seed: int = 42,
    out_dir: str | None = None,
    observed_csv_path: str | None = None,
    interpolate_observed: bool = True,
    return_trials: bool = False,
) -> Union[
    Tuple[Dict[str, Any], float],
    Tuple[Dict[str, Any], float, List[Dict[str, Any]]],
]:
    """Tune model parameters by trying random values within user-provided ranges.

    Inputs:
      - config_path: path to a config file to run the simulation (we reuse its inputs and initial state)
      - observed_years/observed_deer: real-world series to fit against
      - param_ranges: which parameters to vary and the min/max for each (see _sample_candidate docstring)
      - trials: how many random samples to evaluate
      - seed: random seed for reproducibility
      - out_dir: optional directory to save a JSON file of all trial results and the best parameters
      - observed_csv_path / interpolate_observed: control loading of observed data when arrays omitted
      - return_trials: when True, also return the list of per-trial dictionaries for downstream use

    Output:
      - `(best_params, best_score)` by default, or `(best_params, best_score, trial_rows)` when
        `return_trials` is True.
    """
    cfg = load_config(config_path)
    base_params_dict = _deepcopy_params(cfg.params)
    base_params_dict["init"] = asdict(cfg.init)
    rng = random.Random(seed)

    best_score = float("inf")
    best_params: Dict[str, Any] = base_params_dict
    trial_rows: List[Dict[str, Any]] = []
    best_progress: List[Dict[str, Any]] = []
    # Track best-so-far for other metrics (lower is better)
    best_peak_timing = float("inf")
    best_peak_height = float("inf")
    best_crash_ratio = float("inf")

    # Load observed data from CSV if not provided as arrays. Default to project data/kaibab_deer.csv
    if observed_years is None or observed_deer is None:
        if not observed_csv_path:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            observed_csv_path = os.path.join(project_root, "data", "kaibab_deer.csv")
        oy, ov, _ = load_deer_observed_csv(observed_csv_path, interpolate_missing=interpolate_observed)
        observed_years, observed_deer = oy, ov

    if observed_years is None or observed_deer is None:
        raise ValueError("Observed data not provided: pass arrays or observed_csv_path")

    for t in range(trials):
        cand = _sample_candidate(param_ranges, rng)
        cand_params = _apply_candidate(base_params_dict, cand)

        cand_full = json.loads(json.dumps(cand_params))
        cand_params_only = json.loads(json.dumps(cand_params))
        cand_init = asdict(cfg.init)
        if "init" in cand_params_only:
            init_overrides = cand_params_only.pop("init")
            for name, value in init_overrides.items():
                cand_init[name] = value

        import tempfile
        data = {
            "time": {"start": cfg.time.start, "end": cfg.time.end},
            "inputsCsv": getattr(cfg, "inputsCsv", None),
            "inputs": {"hunt": cfg.inputs.hunt, "ctrl": cfg.inputs.ctrl},
            "params": cand_params_only,
            "init": cand_init,
            "seeds": asdict(cfg.seeds),
        }
        # Remove None keys
        if data["inputsCsv"] is None:
            data.pop("inputsCsv")
        fd, tmp_cfg = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(data, f)
        try:
            rows = run_years(tmp_cfg, out_csv_path=os.devnull)
        finally:
            try:
                os.remove(tmp_cfg)
            except OSError:
                pass

        sim_years, sim_deer = _extract_deer_series(rows)
        # Align series by intersecting years for fair metric comparison
        obs_map = {y: v for y, v in zip(observed_years, observed_deer)}
        sim_map = {y: v for y, v in zip(sim_years, sim_deer)}
        years_common = sorted(set(obs_map.keys()) & set(sim_map.keys()))
        obs_aligned = [obs_map[y] for y in years_common]
        sim_aligned = [sim_map[y] for y in years_common]

        s = compute_scaled_mse(obs_aligned, sim_aligned)
        peak_time_err = compute_peak_timing_error(years_common, obs_aligned, years_common, sim_aligned)
        peak_ht_err = compute_peak_height_error(obs_aligned, sim_aligned)
        crash_ratio_err = compute_crash_ratio_error(obs_aligned, sim_aligned)

        trial_row = {
            "trial": t,
            "score": s,
            "metric.peak_timing": peak_time_err,
            "metric.peak_height": peak_ht_err,
            "metric.crash_ratio": crash_ratio_err,
        }
        for g, kv in cand.items():
            for n, v in kv.items():
                trial_row[f"{g}.{n}"] = v
        trial_rows.append(trial_row)
        if s < best_score:
            best_score = s
            best_params = cand_full
            # When a new best score is found, capture that trial's metric values
            best_peak_timing = peak_time_err
            best_peak_height = peak_ht_err
            best_crash_ratio = crash_ratio_err
        best_progress.append({
            "trial": t,
            "score": s,
            "best_so_far": best_score,
            "peak_timing": peak_time_err,
            "best_peak_timing": best_peak_timing,
            "peak_height": peak_ht_err,
            "best_peak_height": best_peak_height,
            "crash_ratio": crash_ratio_err,
            "best_crash_ratio": best_crash_ratio,
        })

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        with open(os.path.join(out_dir, f"trials_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(trial_rows, f, indent=2)
        with open(os.path.join(out_dir, f"best_params_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        # Also write trials as CSV for easy plotting
        # Build stable header: union of all keys encountered (sorted)
        header_keys = set()
        for row in trial_rows:
            header_keys.update(row.keys())
        header = ["trial", "score"] + sorted(k for k in header_keys if k not in ("trial", "score"))
        with open(os.path.join(out_dir, f"trials_{ts}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for row in trial_rows:
                w.writerow(row)
        # Write best-so-far progress CSV with additional metrics
        with open(os.path.join(out_dir, f"best_progress_{ts}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "trial",
                    "score",
                    "best_so_far",
                    "peak_timing",
                    "best_peak_timing",
                    "peak_height",
                    "best_peak_height",
                    "crash_ratio",
                    "best_crash_ratio",
                ],
            )
            w.writeheader()
            for row in best_progress:
                w.writerow(row)

    if return_trials:
        return best_params, best_score, trial_rows
    return best_params, best_score


def calibrate_compare_interpolation(
    config_path: str,
    param_ranges: ParamRanges,
    trials: int = 100,
    seed: int = 42,
    out_dir: str | None = None,
    observed_csv_path: str | None = None,
) -> Dict[str, Any]:
    """Run calibration twice: once with interpolation ON and once OFF for observed deer.

    This helps assess sensitivity to filling gaps in observed data. Results and trial logs are
    saved under out_dir/interp_on and out_dir/interp_off when out_dir is provided.

    Output: dict with keys {"interp_on": {"best_params", "best_score"}, "interp_off": {...}}
    """
    results: Dict[str, Any] = {}

    subdir_on = os.path.join(out_dir, "interp_on") if out_dir else None
    subdir_off = os.path.join(out_dir, "interp_off") if out_dir else None

    best_on, score_on = calibrate_random_search(
        config_path,
        observed_years=None,
        observed_deer=None,
        param_ranges=param_ranges,
        trials=trials,
        seed=seed,
        out_dir=subdir_on,
        observed_csv_path=observed_csv_path,
        interpolate_observed=True,
    )
    results["interp_on"] = {"best_params": best_on, "best_score": score_on}

    best_off, score_off = calibrate_random_search(
        config_path,
        observed_years=None,
        observed_deer=None,
        param_ranges=param_ranges,
        trials=trials,
        seed=seed,
        out_dir=subdir_off,
        observed_csv_path=observed_csv_path,
        interpolate_observed=False,
    )
    results["interp_off"] = {"best_params": best_off, "best_score": score_off}

    return results


def main_bayes_opt() -> None:
    """Command-line entry point for Bayesian optimization calibration."""

    from .bayes_optimize import calibrate_bayes_opt

    parser = argparse.ArgumentParser(description="Bayesian optimization calibration helper")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser.add_argument(
        "--config",
        default=os.path.join(repo_root, "configs", "base.yaml"),
        help="Path to the config file",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(repo_root, "experiments", "calib_bayes"),
        help="Where to write calibration artifacts",
    )
    parser.add_argument(
        "--observed",
        default=os.path.join(repo_root, "data", "kaibab_deer.csv"),
        help="Observed deer CSV path",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=40,
        help="Number of Bayesian optimization evaluations",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--acq",
        default="EI",
        choices=["EI", "PI", "LCB"],
        help="Acquisition function passed to skopt",
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Disable interpolation when loading observed deer",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    out_dir = os.path.abspath(args.outdir)
    observed = os.path.abspath(args.observed)

    ranges = load_param_ranges(config_path)
    best_params, best_score = calibrate_bayes_opt(
        config_path=config_path,
        observed_years=None,
        observed_deer=None,
        param_ranges=ranges,
        iterations=args.iterations,
        seed=args.seed,
        out_dir=out_dir,
        observed_csv_path=observed,
        interpolate_observed=not args.no_interpolate,
        acq_func=args.acq,
    )

    print("Best score:", best_score)
    print(json.dumps(best_params, indent=2))


if __name__ == "__main__":
    main_bayes_opt()
