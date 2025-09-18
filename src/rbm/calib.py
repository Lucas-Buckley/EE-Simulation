from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from typing import Dict, Any, Tuple, List

from .config import load_config
from .run_years import run_years
from .metrics import compute_scaled_mse


ParamRanges = Dict[str, Dict[str, Tuple[float, float]]]


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
    observed_years: List[int],
    observed_deer: List[float],
    param_ranges: ParamRanges,
    trials: int = 100,
    seed: int = 42,
    out_dir: str | None = None,
) -> Tuple[Dict[str, Any], float]:
    """Tune model parameters by trying random values within user-provided ranges.

    Inputs:
      - config_path: path to a config file to run the simulation (we reuse its inputs and initial state)
      - observed_years/observed_deer: real-world series to fit against
      - param_ranges: which parameters to vary and the min/max for each (see _sample_candidate docstring)
      - trials: how many random samples to evaluate
      - seed: random seed for reproducibility
      - out_dir: optional directory to save a JSON file of all trial results and the best parameters

    Output:
      - (best_parameters_as_dict, best_score_value)
    """
    cfg = load_config(config_path)
    base_params_dict = _deepcopy_params(cfg.params)
    rng = random.Random(seed)

    best_score = float("inf")
    best_params: Dict[str, Any] = base_params_dict
    trial_rows: List[Dict[str, Any]] = []

    for t in range(trials):
        cand = _sample_candidate(param_ranges, rng)
        cand_params = _apply_candidate(base_params_dict, cand)
        # Build a temp config dict to pass to runner by writing to a temp file is heavy; instead override after load
        # Use run_years with a monkey-patched params via asdict structure consumed inside run_years → it calls load_config
        # To avoid re-writing config, we re-run run_years on a temporary copy: write a sidecar JSON? Keep simple: write temp file.
        import tempfile
        data = {
            "time": {"start": cfg.time.start, "end": cfg.time.end},
            "inputsCsv": getattr(cfg, "inputsCsv", None),
            "inputs": {"hunt": cfg.inputs.hunt, "ctrl": cfg.inputs.ctrl},
            "params": cand_params,
            "init": asdict(cfg.init),
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
        s = score_fit(observed_years, observed_deer, sim_years, sim_deer)
        trial_row = {"trial": t, "score": s}
        for g, kv in cand.items():
            for n, v in kv.items():
                trial_row[f"{g}.{n}"] = v
        trial_rows.append(trial_row)
        if s < best_score:
            best_score = s
            best_params = cand_params

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        with open(os.path.join(out_dir, f"trials_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(trial_rows, f, indent=2)
        with open(os.path.join(out_dir, f"best_params_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

    return best_params, best_score


