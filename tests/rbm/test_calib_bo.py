import json
import importlib
import os
import tempfile
import unittest
from pathlib import Path

from src.rbm.calib import calibrate_random_search
from src.rbm import bayes_optimize


def _build_temp_config() -> str:
    cfg = {
        "time": {"start": 0, "end": 0},
        "inputs": {"hunt": [0.0], "ctrl": [0.0]},
        "params": {
            "vegetation": {"vegRate": 0.0, "capMax": 1.0, "browse": 0.0},
            "deer": {"birth": 0.5, "surv": 0.5},
            "predation": {"predAtk": 0.0, "predCap": 0.0, "predEff": 0.0},
            "predators": {"mort": 0.0},
        },
        "init": {"deer": 0.5, "pred": 0.0, "carry": 0.5},
        "seeds": {"main": 42},
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return path


def _make_stub_run_years():
    def stub_run_years(config_path: str, out_csv_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        birth = float(data["params"]["deer"]["birth"])
        value = birth
        year = int(data["time"]["start"])
        return [
            {
                "year": year,
                "deer": value,
                "deerNxt": value,
            }
        ]

    return stub_run_years


class TestCalibrationComparison(unittest.TestCase):
    def test_bayes_matches_random_on_synthetic_goal(self) -> None:
        config_path = _build_temp_config()
        try:
            observed_years = [0]
            observed_deer = [0.35]
            param_ranges = {"deer": {"birth": (0.0, 1.0)}}

            stub = _make_stub_run_years()
            calib_module = importlib.import_module("src.rbm.calib")
            original_run_years_bayes = bayes_optimize.run_years
            original_run_years_calib = calib_module.run_years
            try:
                bayes_optimize.run_years = stub  # type: ignore[attr-defined]
                calib_module.run_years = stub  # type: ignore

                seeds = [0, 1]
                tolerance = 5e-3
                for seed in seeds:
                    with tempfile.TemporaryDirectory() as d:
                        bo_dir = Path(d) / "bo"
                        rs_dir = Path(d) / "rs"
                        bo_params, bo_score = bayes_optimize.calibrate_bayes_opt(
                            config_path=config_path,
                            observed_years=observed_years,
                            observed_deer=observed_deer,
                            param_ranges=param_ranges,
                            iterations=15,
                            seed=seed,
                            out_dir=str(bo_dir),
                            observed_csv_path=None,
                            interpolate_observed=True,
                            acq_func="EI",
                        )
                        rs_params, rs_score = calibrate_random_search(
                            config_path=config_path,
                            observed_years=observed_years,
                            observed_deer=observed_deer,
                            param_ranges=param_ranges,
                            trials=15,
                            seed=seed,
                            out_dir=str(rs_dir),
                            observed_csv_path=None,
                            interpolate_observed=True,
                        )
                        self.assertLessEqual(bo_score, rs_score + tolerance)
                        self.assertLessEqual(abs(bo_params["deer"]["birth"] - observed_deer[0]), 0.05)
            finally:
                bayes_optimize.run_years = original_run_years_bayes  # type: ignore[attr-defined]
                calib_module.run_years = original_run_years_calib  # type: ignore
        finally:
            os.remove(config_path)


if __name__ == "__main__":
    unittest.main()
