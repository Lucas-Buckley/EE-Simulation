import unittest

from src.rbm.state import State, Inputs
from src.rbm.step import step


class TestStepHunt(unittest.TestCase):
    def test_zero_hunt_no_removals(self):
        s = State(deer=500.0, pred=0.0, carry=500.0)
        inp = Inputs(hunt=0.0, ctrl=0.0, winter=0.0)
        params = {
            "vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 0.0, "surv": 1.0, "wDeer": 0.0},
            "predation": {"predAtk": 0.0, "predCap": 0.0, "predEff": 0.0},
        }
        _, d = step(s, inp, params)
        self.assertEqual(d["huntRem"], 0.0)

    def test_hunt_rate_is_clamped(self):
        s = State(deer=100.0, pred=0.0, carry=100.0)
        inp = Inputs(hunt=10.0, ctrl=0.0, winter=0.0)  # invalid high, should clamp to 1.0
        params = {
            "vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 0.0, "surv": 1.0, "wDeer": 0.0},
            "predation": {"predAtk": 0.0, "predCap": 0.0, "predEff": 0.0},
        }
        _, d = step(s, inp, params)
        self.assertEqual(d["huntRate"], 1.0)
        self.assertEqual(d["huntRem"], 100.0)


if __name__ == "__main__":
    unittest.main()


