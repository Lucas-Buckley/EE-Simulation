import os
import tempfile
import csv
import unittest

from src.rbm.observed import load_deer_observed_csv


class TestObservedLoader(unittest.TestCase):
    def test_loader_with_interpolation(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Year", "Deer Population"])
            w.writerow([1900, "1,000"])        
            w.writerow([1902, "3,000"])        
        try:
            years, vals, imputed = load_deer_observed_csv(path, interpolate_missing=True)
            self.assertEqual(years, [1900, 1901, 1902])
                                                                                     
            self.assertEqual(len(vals), 3)
            self.assertAlmostEqual(vals[1], 2000.0)
            self.assertEqual(imputed, [False, True, False])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()


