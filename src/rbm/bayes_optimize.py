"""Bayesian optimization helper for tuning the rule-based Kaibab simulator.

This module provides a reusable calibration function that mirrors the
`calibrate_random_search` helper but swaps in Bayesian optimization (BO)
for smarter sampling. The BO backend is `skopt.gp_minimize`, which uses a
Gaussian-process model to balance exploration and exploitation.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Tuple

from skopt import gp_minimize
from skopt.space import Real

from .config import load_config
from .metrics import (
    compute_crash_ratio_error,
    compute_peak_height_error,
    compute_peak_timing_error,
    compute_scaled_mse,
)
from .observed import load_deer_observed_csv
from .run_years import run_years


ParamRanges = Dict[str, Dict[str, Tuple[float, float]]]


class _SearchSpace:
    """Represent the BO search space and convert between dicts and flat lists.

    Inputs:
      - ranges: nested dictionary of parameter bounds.

    Outputs:
      - Allows conversion from nested dict -> flat list and vice versa, while
        keeping a stable ordering for the optimizer.
    """

    def __init__(self, ranges: ParamRanges):
        self._order: List[Tuple[str, str]] = []
        self._bounds: List[Tuple[float, float]] = []
        for group in sorted(ranges.keys()):
            for name in sorted(ranges[group].keys()):
                low, high = ranges[group][name]
                if low > high:
                    raise ValueError(
                        f"Lower bound {low} is greater than upper bound {high} for {group}.{name}"
                    )
                self._order.append((group, name))
                self._bounds.append((low, high))

    @property
    def dimensions(self) -> List[Real]:
        """Return skopt dimension objects for the optimizer."""

        return [Real(low, high, name=f"{group}.{name}") for (group, name), (low, high) in zip(self._order, self._bounds)]

    def dict_to_list(self, params: Dict[str, Dict[str, float]]) -> List[float]:
        """Flatten a nested parameter dictionary into a list matching the search order."""

        flat: List[float] = []
        for group, name in self._order:
            flat.append(float(params[group][name]))
        return flat

    def list_to_dict(self, values: Iterable[float], base: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Inflate a flat list back into a nested parameter dictionary based on a base template."""

        nested = json.loads(json.dumps(base))
        for (group, name), value in zip(self._order, values):
            nested[group][name] = float(value)
        return nested


def _extract_deer_series(rows: List[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
    """Return (years, deer values) from run_years output rows."""

    years = [int(r["year"]) for r in rows]
    deer = [float(r.get("deerNxt", r.get("deer", 0.0))) for r in rows]
    return years, deer


def _align_series(
    observed_years: List[int],
    observed_deer: List[float],
    sim_years: List[int],
    sim_deer: List[float],
) -> Tuple[List[int], List[float], List[float]]:
    """Align observed and simulated data on common years."""

    obs_map = {y: v for y, v in zip(observed_years, observed_deer)}
    sim_map = {y: v for y, v in zip(sim_years, sim_deer)}
    years = sorted(set(obs_map.keys()) & set(sim_map.keys()))
    obs = [obs_map[y] for y in years]
    sim = [sim_map[y] for y in years]
    return years, obs, sim


def _objective_factory(
    config_path: str,
    base_params: Dict[str, Dict[str, float]],
    space: "_SearchSpace",
    observed_years: List[int],
    observed_deer: List[float],
    out_dir: str | None,
    seed: int,
) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create the objective function for BO and containers for logging trials."""

    trials: List[Dict[str, Any]] = []
    progress: List[Dict[str, Any]] = []
    best_score = float("inf")
    best_metrics = {
        "peak_timing": float("inf"),
        "peak_height": float("inf"),
        "crash_ratio": float("inf"),
    }

    def objective(flat_values: List[float]) -> float:
        nonlocal best_score
        cand = space.list_to_dict(flat_values, base_params)
        # Build temporary config JSON similar to calibrate_random_search
        import tempfile

        cfg = load_config(config_path)
        temp = {
            "time": {"start": cfg.time.start, "end": cfg.time.end},
            "inputs": {"hunt": cfg.inputs.hunt, "ctrl": cfg.inputs.ctrl},
            "params": cand,
            "init": asdict(cfg.init),
            "seeds": asdict(cfg.seeds),
        }
        fd, tmp_cfg = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(temp, f)
        try:
            rows = run_years(tmp_cfg, out_csv_path=os.devnull)
        finally:
            try:
                os.remove(tmp_cfg)
            except OSError:
                pass

        sim_years, sim_deer = _extract_deer_series(rows)
        years, obs, sim = _align_series(observed_years, observed_deer, sim_years, sim_deer)

        score = compute_scaled_mse(obs, sim)
        peak_timing = compute_peak_timing_error(years, obs, years, sim)
        peak_height = compute_peak_height_error(obs, sim)
        crash_ratio = compute_crash_ratio_error(obs, sim)

        trial_idx = len(trials)
        trial_row = {
            "trial": trial_idx,
            "score": score,
            "metric.peak_timing": peak_timing,
            "metric.peak_height": peak_height,
            "metric.crash_ratio": crash_ratio,
        }
        for (group, name), value in zip(space._order, flat_values):
            trial_row[f"{group}.{name}"] = value
        trials.append(trial_row)

        if score < best_score:
            best_score = score
            best_metrics.update(
                {
                    "peak_timing": peak_timing,
                    "peak_height": peak_height,
                    "crash_ratio": crash_ratio,
                }
            )

        progress.append(
            {
                "trial": trial_idx,
                "score": score,
                "best_so_far": best_score,
                "peak_timing": peak_timing,
                "best_peak_timing": best_metrics["peak_timing"],
                "peak_height": peak_height,
                "best_peak_height": best_metrics["peak_height"],
                "crash_ratio": crash_ratio,
                "best_crash_ratio": best_metrics["crash_ratio"],
            }
        )

        return score

    return objective, trials, progress


def calibrate_bayes_opt(
    config_path: str,
    observed_years: List[int] | None,
    observed_deer: List[float] | None,
    param_ranges: ParamRanges,
    iterations: int = 40,
    seed: int = 42,
    out_dir: str | None = None,
    observed_csv_path: str | None = None,
    interpolate_observed: bool = True,
    acq_func: str = "EI",
) -> Tuple[Dict[str, Any], float]:
    """Tune model parameters with Bayesian optimization.

    Inputs mirror `calibrate_random_search` and return the same tuple:
    (best_parameters, best_score).
    """

    cfg = load_config(config_path)
    if observed_years is None or observed_deer is None:
        if not observed_csv_path:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            observed_csv_path = os.path.join(project_root, "data", "kaibab_deer.csv")
        oy, ov, _ = load_deer_observed_csv(observed_csv_path, interpolate_missing=interpolate_observed)
        observed_years, observed_deer = oy, ov

    if observed_years is None or observed_deer is None:
        raise ValueError("Observed data not provided: pass arrays or observed_csv_path")

    if iterations < 10:
        raise ValueError("iterations must be at least 10 for gp_minimize to work reliably")

    space = _SearchSpace(param_ranges)
    base_params = asdict(cfg.params)
    objective, trials, progress = _objective_factory(
        config_path,
        base_params,
        space,
        list(observed_years),
        list(observed_deer),
        out_dir,
        seed,
    )

    result = gp_minimize(
        func=objective,
        dimensions=space.dimensions,
        n_calls=iterations,
        random_state=seed,
        acq_func=acq_func,
    )

    best_params = space.list_to_dict(result.x, base_params)
    best_score = float(result.fun)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        with open(os.path.join(out_dir, f"trials_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(trials, f, indent=2)
        header = ["trial", "score"] + sorted(k for k in trials[0].keys() if k not in {"trial", "score"}) if trials else [
            "trial",
            "score",
        ]
        with open(os.path.join(out_dir, f"trials_{ts}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in trials:
                writer.writerow(row)
        with open(os.path.join(out_dir, f"best_params_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        with open(os.path.join(out_dir, f"best_progress_{ts}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
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
            writer.writeheader()
            for row in progress:
                writer.writerow(row)

    return best_params, best_score
