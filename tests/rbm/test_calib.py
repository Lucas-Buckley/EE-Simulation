import os
import tempfile
import json
import unittest

from src.rbm.calib import calibrate_random_search


class TestCalib(unittest.TestCase):
    def test_calibration_runs_and_returns_best(self):
        # Minimal config using arrays (short horizon)
        cfg = {
            "time": {"start": 1900, "end": 1902},
            "inputs": {"hunt": [0, 0, 0], "ctrl": [0, 0, 0]},
            "params": {
                "vegetation": {"vegRate": 0.1, "capMax": 100.0, "browse": 0.0},
                "deer": {"birth": 0.8, "surv": 0.9},
                "predation": {"predAtk": 0.0, "predCap": 0.0, "predEff": 0.0},
                "predators": {"mort": 0.0},
            },
            "init": {"deer": 10.0, "pred": 0.0, "carry": 50.0},
            "seeds": {"main": 1},
        }
        fd_cfg, cfg_path = tempfile.mkstemp(suffix=".json")
        os.close(fd_cfg)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        # Observed series (simple increasing series)
        obs_years = [1900, 1901, 1902]
        obs_deer = [10.0, 11.0, 12.0]

        # Parameter ranges to search
        ranges = {
            "vegetation": {"vegRate": (0.05, 0.2), "capMax": (80.0, 150.0)},
            "deer": {"birth": (0.5, 1.2), "surv": (0.6, 0.95)},
        }

        try:
            best_params, best_score = calibrate_random_search(
                cfg_path,
                obs_years,
                obs_deer,
                ranges,
                trials=5,
                seed=123,
                out_dir=None,
            )
            self.assertIsInstance(best_params, dict)
            self.assertIsInstance(best_score, float)
        finally:
            os.remove(cfg_path)


if __name__ == "__main__":
    unittest.main()


