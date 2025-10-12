import unittest
from unittest import mock

from src.rbm.hybrid import calibrate_hybrid


class TestCalibrateHybrid(unittest.TestCase):
    @mock.patch("src.rbm.hybrid.calibrate_bayes_opt")
    @mock.patch("src.rbm.hybrid.calibrate_random_search")
    def test_warm_start_passed(self, mock_random, mock_bayes):
        param_ranges = {"group": {"alpha": (0.0, 1.0)}}
        trial_rows = [
            {"score": 0.2, "group.alpha": 0.4},
            {"score": 0.5, "group.alpha": 0.9},
        ]
        mock_random.return_value = ({"group": {"alpha": 0.4}}, 0.2, trial_rows)
        mock_bayes.return_value = ({"group": {"alpha": 0.35}}, 0.15)

        final_params, final_score, stage_logs = calibrate_hybrid(
            config_path="cfg.json",
            param_ranges=param_ranges,
            random_trials=5,
            bo_iterations=3,
            warm_start_k=1,
            seed=1,
            out_dir="/tmp/hybrid",
            observed_years=[0],
            observed_deer=[1.0],
            observed_csv_path=None,
            interpolate_observed=True,
            acq_func="EI",
        )

        self.assertLessEqual(final_score, 0.2)
        mock_bayes.assert_called_once()
        kwargs = mock_bayes.call_args.kwargs
        self.assertIn("warm_start", kwargs)
        x0, y0 = kwargs["warm_start"]
        self.assertEqual(len(x0), 1)
        self.assertAlmostEqual(x0[0][0], 0.4)
        self.assertEqual(y0, [0.2])
        self.assertEqual(stage_logs["random"]["trials"], 5)
        self.assertEqual(stage_logs["bayes"]["warm_start_k"], 1)


if __name__ == "__main__":
    unittest.main()
