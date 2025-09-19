import os
import tempfile
import json
import unittest

from src.rbm.run_stochastic import run_years_stochastic


class TestRunStochastic(unittest.TestCase):
    def test_stochastic_runs_and_outputs_bands(self):
        cfg = {
            "time": {"start": 1900, "end": 1901},
            "inputs": {"hunt": [0, 0], "ctrl": [0, 0]},
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
        fd_csv, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd_csv)
        os.remove(csv_path)
        try:
            rows = run_years_stochastic(cfg_path, csv_path, repeats=10, seed=1)
            self.assertEqual(len(rows), 2)
            self.assertIn("deer_mean", rows[0])
            self.assertIn("deer_p10", rows[0])
            self.assertIn("deer_p90", rows[0])
        finally:
            os.remove(cfg_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()


