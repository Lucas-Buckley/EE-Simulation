from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import asdict
from typing import Dict, Any, List, Tuple

from .config import load_config
from .state import State, Inputs
from .step import step


def _percentile(vals: List[float], p: float) -> float:
    """Compute the p-th percentile of a list using linear interpolation on sorted values.

    Inputs:
      - vals: list of numbers to summarize
      - p: percentile to compute in the range [0, 100]
    Output:
      - the value at percentile p (e.g., p=10 gives the 10th percentile). If vals is empty, returns NaN.

    Method: sort the values, find fractional index k = (n-1) * (p/100), and linearly interpolate
    between the floor and ceil neighbors (common definition used by NumPy/Excel variants).
    """
    if not vals:
        return math.nan
    xs = sorted(vals)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def _apply_noise(value: float, rng: random.Random, rel_sd: float, low: float | None = None) -> float:
    """Jitter a parameter by a small relative amount to simulate uncertainty.

    Inputs:
      - value: the original (positive) parameter value
      - rng: a random.Random instance for reproducibility
      - rel_sd: relative standard deviation (e.g., 0.05 ≈ 5% typical variation)
      - low: optional lower bound; if provided, result is clamped to be >= low
    Output:
      - a perturbed value close to the original. If rel_sd <= 0, returns value unchanged.

    Implementation: draws a normal deviate with mean 0 and std rel_sd, then scales value by (1 + noise).
    This approximates multiplicative/log-normal noise while remaining simple.
    """
    if rel_sd <= 0:
        return value
    # Draw normal(0, rel_sd) and exponentiate approximates log-normal for positivity
    noise = rng.normalvariate(0.0, rel_sd)
    v = value * (1.0 + noise)
    if low is not None:
        v = max(low, v)
    return v


def run_years_stochastic(
    config_path: str,
    out_csv_path: str,
    repeats: int = 100,
    seed: int = 42,
    rel_sd: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    """Run the simulation many times with small random jitters to produce mean and bands.

    Inputs:
      - config_path: path to the config file
      - out_csv_path: CSV to write per-year aggregates
      - repeats: number of Monte Carlo repeats
      - seed: base RNG seed (deterministic results)
      - rel_sd: relative standard deviations per parameter group name within params dict
        Example: {"vegetation.vegRate": 0.05, "deer.birth": 0.05, "deer.surv": 0.02,
                  "predation.predAtk": 0.05, "predators.mort": 0.02}

    Output: list of rows with per-year aggregates (mean, p10, p90) for deer and pred.
    """
    cfg = load_config(config_path)
    base_params = asdict(cfg.params)
    years = list(range(cfg.time.start, cfg.time.end + 1))

    # Defaults if rel_sd not given
    if rel_sd is None:
        rel_sd = {
            "vegetation.vegRate": 0.05,
            "deer.birth": 0.05,
            "deer.surv": 0.02,
            "predation.predAtk": 0.05,
            "predators.mort": 0.02,
        }

    # Storage per year across repeats
    deer_by_year: Dict[int, List[float]] = {y: [] for y in years}
    pred_by_year: Dict[int, List[float]] = {y: [] for y in years}

    rng = random.Random(seed)

    for r in range(repeats):
        # Jitter a copy of parameters for this repeat
        params = asdict(cfg.params)
        # Apply noise per specified paths
        for path, sd in rel_sd.items():
            group, name = path.split(".")
            if group in params and name in params[group]:
                params[group][name] = _apply_noise(float(params[group][name]), rng, sd, low=0.0)

        # Run deterministic years with these jittered params
        state = State(deer=cfg.init.deer, pred=cfg.init.pred, carry=cfg.init.carry)
        for i, y in enumerate(years):
            inputs = Inputs(hunt=cfg.inputs.hunt[i], ctrl=cfg.inputs.ctrl[i])
            next_state, _ = step(state, inputs, params)
            deer_by_year[y].append(next_state.deer)
            pred_by_year[y].append(next_state.pred)
            state = next_state

    # Aggregate
    rows: List[Dict[str, Any]] = []
    for y in years:
        dvals = deer_by_year[y]
        pvals = pred_by_year[y]
        row = {
            "year": y,
            "deer_mean": sum(dvals) / len(dvals) if dvals else math.nan,
            "deer_p10": _percentile(dvals, 10.0),
            "deer_p90": _percentile(dvals, 90.0),
            "pred_mean": sum(pvals) / len(pvals) if pvals else math.nan,
            "pred_p10": _percentile(pvals, 10.0),
            "pred_p90": _percentile(pvals, 90.0),
        }
        rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "deer_mean",
                "deer_p10",
                "deer_p90",
                "pred_mean",
                "pred_p10",
                "pred_p90",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


