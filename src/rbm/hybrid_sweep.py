from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

from .hybrid import calibrate_hybrid
from .observed import load_deer_observed_csv


def run_hybrid_sweep(
    config_path: str,
    param_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    random_trials_options: Sequence[int],
    bo_iteration_options: Sequence[int],
    warm_start_options: Sequence[int],
    acq_funcs: Sequence[str],
    seeds: Sequence[int],
    out_dir: str,
    observed_csv_path: str | None = None,
    interpolate_observed: bool = True,
    max_runs: int | None = None,
    progress_interval_sec: float = 5.0,
) -> Dict[str, any]:

    os.makedirs(out_dir, exist_ok=True)

    if observed_csv_path:
        oy, ov, _ = load_deer_observed_csv(observed_csv_path, interpolate_missing=interpolate_observed)
    else:
        oy = ov = None

    results: List[Dict[str, any]] = []
    aggregates: Dict[Tuple[int, int, int, str], Dict[str, List[float]]] = defaultdict(lambda: {"scores": [], "durations": []})

    combinations = [
        (rt, bo, k, acq, seed)
        for rt in random_trials_options
        for bo in bo_iteration_options
        for k in warm_start_options
        for acq in acq_funcs
        for seed in seeds
    ]

    if max_runs is not None:
        combinations = combinations[:max_runs]

    last_print = time.time()
    total_runs = len(combinations)

    for idx, (rt, bo, k, acq, seed) in enumerate(combinations, start=1):
        combo_dir = os.path.join(out_dir, f"rt_{rt}_bo_{bo}_k_{k}_{acq}", f"seed_{seed}")
        best_params, best_score, stage_logs = calibrate_hybrid(
            config_path=config_path,
            param_ranges=param_ranges,
            random_trials=rt,
            bo_iterations=bo,
            warm_start_k=k,
            seed=seed,
            out_dir=combo_dir,
            observed_years=oy,
            observed_deer=ov,
            observed_csv_path=observed_csv_path,
            interpolate_observed=interpolate_observed,
            acq_func=acq,
        )

        duration = stage_logs["random"]["duration_sec"] + stage_logs["bayes"]["duration_sec"]
        record = {
            "random_trials": rt,
            "bo_iterations": bo,
            "warm_start_k": k,
            "acq_func": acq,
            "seed": seed,
            "best_score": best_score,
            "duration_sec": duration,
            "config": best_params,
            "stage_logs": stage_logs,
            "out_dir": combo_dir,
            "index": idx,
        }
        results.append(record)

        key = (rt, bo, k, acq)
        aggregates[key]["scores"].append(best_score)
        aggregates[key]["durations"].append(duration)

        now = time.time()
        if now - last_print >= progress_interval_sec or idx == total_runs:
            print(
                f"[hybrid_sweep] Completed {idx}/{total_runs} runs "
                f"(rt={rt}, bo={bo}, k={k}, acq={acq}, seed={seed})"
            )
            last_print = now

    aggregate_summary: Dict[str, Dict[str, float]] = {}
    for (rt, bo, k, acq), values in aggregates.items():
        scores = values["scores"]
        durations = values["durations"]
        key = f"rt_{rt}_bo_{bo}_k_{k}_{acq}"
        aggregate_summary[key] = {
            "best_score": min(scores),
            "worst_score": max(scores),
            "mean_score": mean(scores),
            "std_score": pstdev(scores) if len(scores) > 1 else 0.0,
            "mean_duration_sec": mean(durations),
            "std_duration_sec": pstdev(durations) if len(durations) > 1 else 0.0,
            "runs": len(scores),
        }

    summary = {
        "config": {
            "random_trials_options": list(random_trials_options),
            "bo_iteration_options": list(bo_iteration_options),
            "warm_start_options": list(warm_start_options),
            "acq_funcs": list(acq_funcs),
            "seeds": list(seeds),
            "max_runs": max_runs,
        },
        "results": results,
        "aggregates": aggregate_summary,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
