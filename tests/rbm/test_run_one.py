import os
import tempfile
import json
import unittest

from src.rbm.run_one import run_once


class TestRunOne(unittest.TestCase):
    def test_run_once_writes_csv(self):
        # Minimal config for one year
        data = {
            "time": {"start": 1905, "end": 1905},
            "inputs": {"hunt": [0], "ctrl": [0]},
            "params": {
                "vegetation": {"vegRate": 0.15, "capMax": 100000, "browse": 0.0},
                "deer": {"birth": 1.0, "surv": 0.85},
                "predation": {"predAtk": 0.0, "predCap": 0.0, "predEff": 0.0},
                "predators": {"mort": 0.0},
            },
            "init": {"deer": 100.0, "pred": 10.0, "carry": 200.0},
            "seeds": {"main": 1},
        }

        fd_cfg, cfg_path = tempfile.mkstemp(suffix=".json")
        os.close(fd_cfg)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        fd_csv, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd_csv)
        os.remove(csv_path)  # we want run_once to create it

        try:
            row = run_once(cfg_path, csv_path)
            self.assertTrue(os.path.exists(csv_path))
            self.assertEqual(row["year"], 1905)
            self.assertIn("deerNxt", row)
        finally:
            os.remove(cfg_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()


