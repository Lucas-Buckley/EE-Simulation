import os
import tempfile
import json
import unittest

from src.rbm.config import load_config


def write_tmp_config(data: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def base_data():
    return {
        "time": {"start": 1905, "end": 1907},
        "inputs": {
            "hunt": [0, 0, 0],
            "ctrl": [0.3, 0.3, 0.1],
            "winter": [0, 1, 0],
        },
        "params": {
            "vegetation": {"vegRate": 0.15, "capMax": 100000, "browse": 0.10, "winPen": 0.05},
            "deer": {"birth": 1.0, "surv": 0.85, "wDeer": 0.2},
            "predation": {"predAtk": 0.0001, "predCap": 0.3, "predEff": 0.0015},
            "predators": {"mort": 0.12},
        },
        "init": {"deer": 30000, "pred": 200, "carry": 30000},
        "seeds": {"main": 42},
    }


class TestConfig(unittest.TestCase):
    def test_load_config_success(self):
        path = write_tmp_config(base_data())
        try:
            cfg = load_config(path)
            self.assertEqual(cfg.time.start, 1905)
            self.assertEqual(cfg.time.end, 1907)
            self.assertEqual(len(cfg.inputs.hunt), 3)
            self.assertEqual(cfg.params.vegetation.capMax, 100000)
            self.assertEqual(cfg.init.deer, 30000)
            self.assertEqual(cfg.seeds.main, 42)
        finally:
            os.remove(path)

    def test_inputs_length_validation(self):
        data = base_data()
        data["inputs"]["hunt"] = [0, 0]  # wrong length
        path = write_tmp_config(data)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_config(path)
            self.assertIn("length", str(ctx.exception))
        finally:
            os.remove(path)

    def test_time_range_validation(self):
        data = base_data()
        data["time"]["end"] = 1900
        path = write_tmp_config(data)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_config(path)
            self.assertIn("time.end", str(ctx.exception))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()


