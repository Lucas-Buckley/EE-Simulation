import unittest

from src.rbm.metrics import (
    compute_scaled_mse,
    find_peak,
    compute_peak_timing_error,
    compute_peak_height_error,
    compute_crash_ratio,
    compute_crash_ratio_error,
)


class TestMetrics(unittest.TestCase):
    def test_scaled_mse(self):
        obs = [10, 20, 30]
        sim = [10, 25, 25]
        val = compute_scaled_mse(obs, sim)
        self.assertGreaterEqual(val, 0.0)

    def test_find_peak(self):
        v, i = find_peak([1, 5, 3])
        self.assertEqual(v, 5)
        self.assertEqual(i, 1)

    def test_peak_timing_error(self):
        y_obs = [1900, 1901, 1902]
        obs = [1, 3, 2]
        y_sim = [1900, 1901, 1902]
        sim = [2, 1, 5]
        err = compute_peak_timing_error(y_obs, obs, y_sim, sim)
        self.assertEqual(err, 1)

    def test_peak_height_error(self):
        obs = [1, 5, 2]
        sim = [1, 3, 2]
        err = compute_peak_height_error(obs, sim)
        self.assertAlmostEqual(err, (5 - 3) / 5)

    def test_crash_ratio(self):
        series = [1, 5, 1, 2]
        ratio = compute_crash_ratio(series)
        self.assertAlmostEqual(ratio, 1 / 5)

    def test_crash_ratio_error(self):
        obs = [1, 5, 1]
        sim = [1, 6, 3]
        err = compute_crash_ratio_error(obs, sim)
        self.assertAlmostEqual(err, abs((1/5) - (3/6)))


if __name__ == "__main__":
    unittest.main()


