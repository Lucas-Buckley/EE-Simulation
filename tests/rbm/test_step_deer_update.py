import unittest

from src.rbm.state import State, Inputs
from src.rbm.step import step


class TestStepDeerUpdate(unittest.TestCase):
    def test_deer_non_negative(self):
        s = State(deer=10.0, pred=0.0, carry=0.0)
        inp = Inputs(hunt=0.0, ctrl=0.0)
        p = {
            "vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 0.0, "surv": 0.0, "wDeer": 0.0},
            "predation": {"predAtk": 10.0, "predCap": 10.0, "predEff": 0.0},
        }
        ns, d = step(s, inp, p)
        self.assertGreaterEqual(ns.deer, 0.0)
        self.assertEqual(ns.deer, d["deer_next"])

    def test_growth_under_favorable(self):
        s = State(deer=50.0, pred=0.0, carry=1000.0)
        inp = Inputs(hunt=0.0, ctrl=0.0)
        p = {
            "vegetation": {"vegRate": 0.1, "capMax": 1000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 1.0, "surv": 0.9, "wDeer": 0.0},
            "predation": {"predAtk": 0.0, "predCap": 0.0, "predEff": 0.0},
        }
        ns, _ = step(s, inp, p)
        self.assertGreater(ns.deer, 50.0)


if __name__ == "__main__":
    unittest.main()


