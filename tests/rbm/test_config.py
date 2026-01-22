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
        "inputs": {"hunt": [0, 0, 0], "ctrl": [0.3, 0.3, 0.1]},
        "params": {
            "vegetation": {"vegRate": 0.15, "capMax": 100000, "browse": 0.10},
            "deer": {"birth": 1.0, "surv": 0.85},
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
        data["inputs"]["hunt"] = [0, 0]                
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

    def test_load_from_csv_overrides_time_and_inputs(self):
        import csv as _csv
        fd_csv, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd_csv)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["Year", "% Deer Hunted", "% Predators Killed"])
            w.writerow([1901, 0.0, 0.1])
            w.writerow([1902, 0.2, 0.0])

        data = base_data()
        data.pop("inputs", None)
        data["inputsCsv"] = csv_path
        data["time"] = {"start": 1800, "end": 1801}
        path = write_tmp_config(data)
        try:
            cfg = load_config(path)
            self.assertEqual(cfg.time.start, 1901)
            self.assertEqual(cfg.time.end, 1902)
            self.assertEqual(cfg.inputs.hunt, [0.0, 0.2])
            self.assertEqual(cfg.inputs.ctrl, [0.1, 0.0])
        finally:
            os.remove(path)
            os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()


