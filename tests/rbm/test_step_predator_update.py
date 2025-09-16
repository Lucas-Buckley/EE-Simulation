import unittest

from src.rbm.state import State, Inputs
from src.rbm.step import step


class TestStepPredatorUpdate(unittest.TestCase):
    def test_ctrl_reduces_pred(self):
        s = State(deer=1000.0, pred=100.0, carry=1000.0)
        params = {
            "vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 0.0, "surv": 1.0, "wDeer": 0.0},
            "predation": {"predAtk": 0.0, "predCap": 0.0, "predEff": 0.001},
            "predators": {"mort": 0.1},
        }
        no_ctrl = Inputs(hunt=0.0, ctrl=0.0)
        hi_ctrl = Inputs(hunt=0.0, ctrl=0.5)
        ns0, _ = step(s, no_ctrl, params)
        ns1, _ = step(s, hi_ctrl, params)
        self.assertLess(ns1.pred, ns0.pred)

    def test_pred_non_negative(self):
        s = State(deer=0.0, pred=1.0, carry=0.0)
        inp = Inputs(hunt=0.0, ctrl=1.0)
        params = {
            "vegetation": {"vegRate": 0.0, "capMax": 1.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 0.0, "surv": 0.0, "wDeer": 0.0},
            "predation": {"predAtk": 0.0, "predCap": 0.0, "predEff": 0.0},
            "predators": {"mort": 0.9},
        }
        ns, _ = step(s, inp, params)
        self.assertGreaterEqual(ns.pred, 0.0)


if __name__ == "__main__":
    unittest.main()


