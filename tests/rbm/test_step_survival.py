import unittest

from src.rbm.state import State, Inputs
from src.rbm.step import step


class TestStepSurvival(unittest.TestCase):
    def test_survival_increases_with_food(self):
        s_low = State(deer=100.0, pred=0.0, carry=10.0)             
        s_high = State(deer=100.0, pred=0.0, carry=200.0)            
        p = {
            "vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0},
            "deer": {"birth": 0.0, "mort": 0.2},
        }
        inp = Inputs(hunt=0.0, ctrl=0.0)
        _, d_low = step(s_low, inp, p)
        _, d_high = step(s_high, inp, p)
        assert d_low["survNat"] <= d_high["survNat"]

    def test_survival_fraction_bounds(self):
        s = State(deer=100.0, pred=0.0, carry=0.0)             
        p = {
            "vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0},
            "deer": {"birth": 0.0, "mort": 0.2},
        }
        inp = Inputs(hunt=0.0, ctrl=0.0)
        _, d = step(s, inp, p)
        self.assertGreaterEqual(d["survNat"], 0.0)
        self.assertLessEqual(d["survNat"], 1.0)


if __name__ == "__main__":
    unittest.main()


