from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

from .bayes_optimize import _SearchSpace, calibrate_bayes_opt
from .calib import calibrate_random_search
from .observed import load_deer_observed_csv


def calibrate_hybrid(
    config_path: str,
    param_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    random_trials: int,
    bo_iterations: int,
    warm_start_k: int = 10,
    seed: int = 42,
    out_dir: str | None = None,
    observed_years: List[int] | None = None,
    observed_deer: List[float] | None = None,
    observed_csv_path: str | None = None,
    interpolate_observed: bool = True,
    acq_func: str = "EI",
) -> Tuple[Dict[str, Dict[str, float]], float, Dict[str, Dict[str, float]]]:

    base_out = out_dir or os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        "experiments",
        "hybrid_run",
    )
    random_dir = os.path.join(base_out, "random")
    bayes_dir = os.path.join(base_out, "bayes")

    if observed_years is None or observed_deer is None:
        if not observed_csv_path:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            observed_csv_path = os.path.join(project_root, "data", "kaibab_deer.csv")
        oy, ov, _ = load_deer_observed_csv(observed_csv_path, interpolate_missing=interpolate_observed)
        observed_years = oy
        observed_deer = ov

    random_start = time.time()
    random_result = calibrate_random_search(
        config_path=config_path,
        observed_years=observed_years,
        observed_deer=observed_deer,
        param_ranges=param_ranges,
        trials=random_trials,
        seed=seed,
        out_dir=random_dir,
        observed_csv_path=observed_csv_path,
        interpolate_observed=interpolate_observed,
        return_trials=True,
    )
    best_random_params, best_random_score, trial_rows = random_result
    random_duration = time.time() - random_start

    space = _SearchSpace(param_ranges)
    sorted_trials = sorted(trial_rows, key=lambda row: row["score"])
    top_rows = sorted_trials[: max(1, min(warm_start_k, len(sorted_trials)))]
    x0 = []
    for row in top_rows:
        nested: Dict[str, Dict[str, float]] = {}
        for group, name in space._order:
            nested.setdefault(group, {})[name] = row[f"{group}.{name}"]
        x0.append(space.dict_to_list(nested))
    y0 = [row["score"] for row in top_rows]

    bo_iterations = max(bo_iterations, len(x0))
    bayes_start = time.time()
    best_bayes_params, best_bayes_score = calibrate_bayes_opt(
        config_path=config_path,
        observed_years=observed_years,
        observed_deer=observed_deer,
        param_ranges=param_ranges,
        iterations=bo_iterations,
        seed=seed,
        out_dir=bayes_dir,
        observed_csv_path=observed_csv_path,
        interpolate_observed=interpolate_observed,
        acq_func=acq_func,
        warm_start=(x0, y0),
    )
    bayes_duration = time.time() - bayes_start

    if best_bayes_score <= best_random_score:
        final_params = best_bayes_params
        final_score = best_bayes_score
    else:
        final_params = best_random_params
        final_score = best_random_score

    stage_logs = {
        "random": {
            "best_score": best_random_score,
            "duration_sec": random_duration,
            "out_dir": random_dir,
            "trials": random_trials,
        },
        "bayes": {
            "best_score": best_bayes_score,
            "duration_sec": bayes_duration,
            "out_dir": bayes_dir,
            "iterations": bo_iterations,
            "warm_start_k": len(x0),
        },
    }

    return final_params, final_score, stage_logs
