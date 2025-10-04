import json
import os
import unittest
from unittest import mock

from skopt.space import Real

from src.rbm import bayes_optimize


class TestSearchSpace(unittest.TestCase):
    def test_space_conversion_round_trip(self) -> None:
        ranges = {
            "deer": {"birth": (0.6, 1.2), "surv": (0.5, 0.95)},
            "vegetation": {"vegRate": (0.05, 0.2)},
        }
        space = bayes_optimize._SearchSpace(ranges)

        dims = space.dimensions
        self.assertEqual(len(dims), 3)
        self.assertIsInstance(dims[0], Real)
        names = [dim.name for dim in dims]
        self.assertIn("deer.birth", names)

        base = {
            "deer": {"birth": 0.9, "surv": 0.8},
            "vegetation": {"vegRate": 0.1},
        }
        flat = space.dict_to_list(base)
        for value in flat:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        rebuilt = space.list_to_dict(flat, base)
        self.assertEqual(rebuilt, base)


class TestObjective(unittest.TestCase):
    @mock.patch("src.rbm.bayes_optimize.run_years")
    def test_objective_reproducible(self, mock_run_years: mock.Mock) -> None:
        def fake_run_years(cfg_path: str, out_csv_path: str):
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            start = int(data["time"]["start"])
            end = int(data["time"]["end"])
            deer0 = float(data["init"]["deer"])
            growth = float(data["params"]["deer"]["surv"])
            rows = []
            deer = deer0
            for i, year in enumerate(range(start, end + 1)):
                deer = deer0 + growth * (i + 1)
                rows.append({"year": year, "deer": deer, "deerNxt": deer})
            return rows

        mock_run_years.side_effect = fake_run_years

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "base.yaml"))
        cfg = bayes_optimize.load_config(base_path)
        base_params = json.loads(json.dumps(bayes_optimize.asdict(cfg.params)))
        ranges = {
            "deer": {"surv": (0.5, 0.95)},
        }
        space = bayes_optimize._SearchSpace(ranges)

        objective, trials, progress = bayes_optimize._objective_factory(
            base_path,
            base_params,
            space,
            [cfg.time.start, cfg.time.start + 1],
            [cfg.init.deer, cfg.init.deer * 1.1],
            None,
            seed=0,
        )

        value_one = objective([0.7])
        value_two = objective([0.7])

        self.assertAlmostEqual(value_one, value_two)
        self.assertEqual(len(trials), 2)
        self.assertEqual(trials[0]["score"], value_one)


if __name__ == "__main__":
    unittest.main()
