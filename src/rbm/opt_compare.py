from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence

from skopt import gp_minimize

from .bayes_optimize import _SearchSpace, _objective_factory
from .calib import calibrate_random_search, load_param_ranges
from .config import load_config
from .observed import load_deer_observed_csv
from .hybrid import calibrate_hybrid


class MultiStrategyEvaluator:

    def __init__(
        self,
        config_path: str,
        observed_csv_path: str | None = None,
        seeds: Sequence[int] = (42, 77),
        random_trials: int = 150,
        gp_iterations: int = 150,
        hybrid_random_trials: int = 120,
        hybrid_bo_iterations: int = 60,
        warm_start_top_k: int = 10,
        out_dir: str | None = None,
    ) -> None:
        self.config_path = os.path.abspath(config_path)
        self.cfg = load_config(self.config_path)
        self.seeds = list(seeds)
        self.random_trials = random_trials
        self.gp_iterations = gp_iterations
        self.hybrid_random_trials = hybrid_random_trials
        self.hybrid_bo_iterations = hybrid_bo_iterations
        self.warm_start_top_k = warm_start_top_k
        self.base_out_dir = Path(out_dir or (Path("experiments") / "strategy_compare"))
        self.base_out_dir.mkdir(parents=True, exist_ok=True)

        self.param_ranges = load_param_ranges(self.config_path)
        if observed_csv_path:
            oy, ov, _ = load_deer_observed_csv(observed_csv_path, interpolate_missing=True)
        else:
            project_root = Path(__file__).resolve().parents[2]
            default_csv = project_root / "data" / "kaibab_deer.csv"
            oy, ov, _ = load_deer_observed_csv(str(default_csv), interpolate_missing=True)
        self.observed_years = list(oy)
        self.observed_deer = list(ov)

    def run(self) -> Dict[str, Any]:
        summaries: Dict[str, Any] = {}
        summaries["random"] = self._run_random()
        summaries["gp"] = self._run_gp()
        summaries["hybrid"] = self._run_hybrid()
        return self._compute_stats(summaries)

                                                                            

    def _run_random(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for seed in self.seeds:
            out_dir = self._prep_out_dir("random", seed)
            start = time.time()
            best_params, best_score = calibrate_random_search(
                config_path=self.config_path,
                observed_years=self.observed_years,
                observed_deer=self.observed_deer,
                param_ranges=self.param_ranges,
                trials=self.random_trials,
                seed=seed,
                out_dir=str(out_dir),
                observed_csv_path=None,
                interpolate_observed=True,
            )
            duration = time.time() - start
            results.append(
                {
                    "seed": seed,
                    "best_score": best_score,
                    "duration_sec": duration,
                    "out_dir": str(out_dir),
                    "config": best_params,
                }
            )
        return results

    def _run_gp(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for seed in self.seeds:
            results.extend(
                self._run_skopt_strategy(
                    strategy="bayes",
                    minimize_func=gp_minimize,
                    iterations=self.gp_iterations,
                    seed=seed,
                    acq_func="EI",
                )
            )
        return results

    def _run_hybrid(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for seed in self.seeds:
            out_dir = self._prep_out_dir("hybrid", seed)
            best_params, best_score, stage_logs = calibrate_hybrid(
                config_path=self.config_path,
                param_ranges=self.param_ranges,
                random_trials=self.hybrid_random_trials,
                bo_iterations=self.hybrid_bo_iterations,
                warm_start_k=self.warm_start_top_k,
                seed=seed,
                out_dir=str(out_dir),
                observed_years=self.observed_years,
                observed_deer=self.observed_deer,
                observed_csv_path=None,
                interpolate_observed=True,
                acq_func="EI",
            )
            results.append(
                {
                    "seed": seed,
                    "best_score": best_score,
                    "duration_sec": stage_logs["random"]["duration_sec"] + stage_logs["bayes"]["duration_sec"],
                    "out_dir": str(out_dir),
                    "config": best_params,
                    "stages": stage_logs,
                }
            )
        return results

                                                                            

    def _run_skopt_strategy(
        self,
        strategy: str,
        minimize_func,
        iterations: int,
        seed: Optional[int] = None,
        acq_func: Optional[str] = None,
        out_dir_override: Optional[Path] = None,
        warm_start: Optional[tuple[List[List[float]], List[float]]] = None,
        return_logs: bool = False,
    ):
        seed_val = self.seeds[0] if seed is None else seed
        out_dir = out_dir_override or self._prep_out_dir(strategy, seed_val)

        base_params = asdict(self.cfg.params)
        space = _SearchSpace(self.param_ranges)
        objective, trials, progress = _objective_factory(
            self.config_path,
            base_params,
            space,
            self.observed_years,
            self.observed_deer,
            None,
            seed_val,
        )

        minimize_kwargs: Dict[str, Any] = {
            "func": objective,
            "dimensions": space.dimensions,
            "n_calls": iterations,
            "random_state": seed_val,
        }
        if minimize_func is gp_minimize:
            minimize_kwargs["acq_func"] = acq_func or "EI"
            if warm_start is not None:
                minimize_kwargs["x0"], minimize_kwargs["y0"] = warm_start

        start = time.time()
        result = minimize_func(**minimize_kwargs)
        duration = time.time() - start

        best_params = space.list_to_dict(result.x, base_params)
        best_score = float(result.fun)
        self._write_logs(out_dir, trials, progress, best_params)

        record = {
            "seed": seed_val,
            "best_score": best_score,
            "duration_sec": duration,
            "out_dir": str(out_dir),
            "config": best_params,
            "iterations": iterations,
        }

        if return_logs:
            return [record], trials
        return [record]

    def _prep_out_dir(self, strategy: str, seed: int) -> Path:
        path = self.base_out_dir / strategy / f"seed_{seed}"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_logs(out_dir: Path, trials: List[Dict[str, Any]], progress: List[Dict[str, Any]], best_params: Dict[str, Any]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        with open(out_dir / f"trials_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(trials, f, indent=2)
        header = ["trial", "score"] + [key for key in trials[0].keys() if key not in {"trial", "score"}] if trials else [
            "trial",
            "score",
        ]
        with open(out_dir / f"trials_{timestamp}.csv", "w", encoding="utf-8") as f:
            if trials:
                import csv

                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(trials)
        with open(out_dir / f"best_params_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        with open(out_dir / f"best_progress_{timestamp}.csv", "w", encoding="utf-8") as f:
            if progress:
                import csv

                writer = csv.DictWriter(
                    f,
                    fieldnames=list(progress[0].keys()),
                )
                writer.writeheader()
                writer.writerows(progress)

    @staticmethod
    def _compute_stats(raw: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for strategy, data in raw.items():
            if not data:
                continue
            valid_scores = [entry["best_score"] for entry in data if entry["best_score"] is not None]
            durations = [entry.get("duration_sec", 0.0) for entry in data if entry.get("duration_sec") is not None]
            stats: Dict[str, Any] = {"results": data}
            if valid_scores:
                stats.update(
                    {
                        "best_score": min(valid_scores),
                        "mean_score": mean(valid_scores),
                        "std_score": pstdev(valid_scores) if len(valid_scores) > 1 else 0.0,
                    }
                )
            if durations:
                stats.update(
                    {
                        "mean_duration_sec": mean(durations),
                        "std_duration_sec": pstdev(durations) if len(durations) > 1 else 0.0,
                    }
                )
            summary[strategy] = stats
        return summary
