import os
import tempfile
import json
import unittest

from src.rbm.run_years import run_years


class TestRunYears(unittest.TestCase):
    def test_run_years_length_and_csv(self):
        data = {
            "time": {"start": 1905, "end": 1907},
            "inputs": {"hunt": [0, 0, 0], "ctrl": [0, 0, 0]},
            "params": {
                "vegetation": {"vegRate": 0.10, "capMax": 1000.0, "browse": 0.0},
                "deer": {"birth": 1.0, "mort": 0.1},
                "predation": {"predAtk": 0.0, "predCap": 0.0, "predEff": 0.0},
                "predators": {"mort": 0.0},
            },
            "init": {"deer": 50.0, "pred": 0.0, "carry": 100.0},
            "seeds": {"main": 1},
        }

        fd_cfg, cfg_path = tempfile.mkstemp(suffix=".json")
        os.close(fd_cfg)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        fd_csv, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd_csv)
        os.remove(csv_path)

        try:
            rows = run_years(cfg_path, csv_path)
            self.assertEqual(len(rows), 3)
            self.assertTrue(os.path.exists(csv_path))
                                                                      
            self.assertEqual([r["year"] for r in rows], [1905, 1906, 1907])
        finally:
            os.remove(cfg_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()


