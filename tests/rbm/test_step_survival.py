import unittest

from src.rbm.state import State, Inputs
from src.rbm.step import step


class TestStepSurvival(unittest.TestCase):
    def test_winter_reduces_survival_fraction(self):
        s = State(deer=100.0, pred=0.0, carry=100.0)
        p = {
            "vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 0.0, "surv": 0.8, "wDeer": 0.25},
        }
        no_w = Inputs(hunt=0.0, ctrl=0.0, winter=0.0)
        hard_w = Inputs(hunt=0.0, ctrl=0.0, winter=2.0)
        _, d0 = step(s, no_w, p)
        _, d1 = step(s, hard_w, p)
        self.assertGreater(d0["survNat"], d1["survNat"])

    def test_survival_fraction_bounds(self):
        s = State(deer=100.0, pred=0.0, carry=0.0)  # food -> 0
        p = {
            "vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 0.0, "surv": 0.8, "wDeer": 10.0},  # strong winter sensitivity
        }
        w = Inputs(hunt=0.0, ctrl=0.0, winter=10.0)
        _, d = step(s, w, p)
        self.assertGreaterEqual(d["survNat"], 0.0)
        self.assertLessEqual(d["survNat"], 1.0)


if __name__ == "__main__":
    unittest.main()


