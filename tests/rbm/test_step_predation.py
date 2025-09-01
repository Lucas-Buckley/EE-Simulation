import unittest

from src.rbm.state import State, Inputs
from src.rbm.step import step


class TestStepPredation(unittest.TestCase):
    def test_kill_increases_with_pred(self):
        s = State(deer=1000.0, pred=10.0, carry=1000.0)
        inp = Inputs(hunt=0.0, ctrl=0.0, winter=0.0)
        params = {
            "vegetation": {"vegRate": 0.0, "capMax": 100000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 0.0, "surv": 1.0, "wDeer": 0.0},
            "predation": {"predAtk": 0.001, "predCap": 0.9, "predEff": 0.001},
        }
        _, d1 = step(s, inp, params)
        s2 = State(deer=1000.0, pred=20.0, carry=1000.0)
        _, d2 = step(s2, inp, params)
        self.assertGreater(d2["kill"], d1["kill"])  # more predators -> more kills

    def test_kill_is_capped(self):
        s = State(deer=1000.0, pred=1e9, carry=1000.0)
        inp = Inputs(hunt=0.0, ctrl=0.0, winter=0.0)
        params = {
            "vegetation": {"vegRate": 0.0, "capMax": 100000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 0.0, "surv": 1.0, "wDeer": 0.0},
            "predation": {"predAtk": 1.0, "predCap": 0.3, "predEff": 0.001},
        }
        _, d = step(s, inp, params)
        self.assertAlmostEqual(d["kill"], 0.3 * 1000.0)


if __name__ == "__main__":
    unittest.main()


