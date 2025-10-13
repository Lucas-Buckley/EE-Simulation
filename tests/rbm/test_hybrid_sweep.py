import json
import os
import tempfile
import unittest
from unittest import mock


class TestHybridSweep(unittest.TestCase):
    def test_sweep_collects_stats(self) -> None:
        from src.rbm import hybrid_sweep

        def fake_calibrate_hybrid(config_path, param_ranges, random_trials, bo_iterations, warm_start_k, seed, out_dir, **kwargs):
            score = (random_trials / 400) + (bo_iterations / 80) + (warm_start_k / 50) + seed * 0.001
            stage_logs = {
                "random": {"best_score": score + 0.1, "duration_sec": 0.01, "out_dir": os.path.join(out_dir, "random"), "trials": random_trials},
                "bayes": {"best_score": score, "duration_sec": 0.02, "out_dir": os.path.join(out_dir, "bayes"), "iterations": bo_iterations, "warm_start_k": warm_start_k},
            }
            params = {"group": {"value": score}}
            return params, score, stage_logs

        with mock.patch("src.rbm.hybrid_sweep.calibrate_hybrid", side_effect=fake_calibrate_hybrid):
            with mock.patch("src.rbm.hybrid_sweep.load_deer_observed_csv", return_value=([0], [1.0], [])):
                with tempfile.TemporaryDirectory() as tmpdir:
                    summary = hybrid_sweep.run_hybrid_sweep(
                        config_path="cfg.json",
                        param_ranges={"group": {"value": (0.0, 1.0)}},
                        random_trials_options=[100, 200],
                        bo_iteration_options=[20, 40],
                        warm_start_options=[1, 10],
                        acq_funcs=["EI", "PI"],
                        seeds=[42, 77],
                        out_dir=tmpdir,
                        observed_csv_path="data.csv",
                        interpolate_observed=True,
                        max_runs=6,
                    )

                    self.assertIn("results", summary)
                    self.assertGreater(len(summary["results"]), 0)
                    aggregates = summary["aggregates"]
                    self.assertTrue(any(key.startswith("rt_100") for key in aggregates))
                    summary_path = os.path.join(tmpdir, "summary.json")
                    self.assertTrue(os.path.exists(summary_path))
                    with open(summary_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    self.assertEqual(data["aggregates"].keys(), summary["aggregates"].keys())


if __name__ == "__main__":
    unittest.main()
