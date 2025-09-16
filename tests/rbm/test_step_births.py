import unittest

from src.rbm.state import State, Inputs
from src.rbm.step import step


class TestStepBirths(unittest.TestCase):
    def test_births_scale_with_food(self):
        s = State(deer=100.0, pred=0.0, carry=50.0)  # food=0.5 if carry unchanged
        inp = Inputs(hunt=0.0, ctrl=0.0)
        params = {
            "vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 1.0, "surv": 0.85, "wDeer": 0.2},
        }
        _, diag = step(s, inp, params)
        self.assertAlmostEqual(diag["food"], 0.5)
        self.assertAlmostEqual(diag["births"], 100.0 * 1.0 * 0.5)

    def test_food_capped_at_one(self):
        s = State(deer=80.0, pred=0.0, carry=200.0)
        inp = Inputs(hunt=0.0, ctrl=0.0)
        params = {
            "vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 0.8, "surv": 0.85, "wDeer": 0.2},
        }
        _, diag = step(s, inp, params)
        self.assertAlmostEqual(diag["food"], 1.0)
        self.assertAlmostEqual(diag["births"], 80.0 * 0.8 * 1.0)

    def test_zero_deer_zero_births(self):
        s = State(deer=0.0, pred=0.0, carry=100.0)
        inp = Inputs(hunt=0.0, ctrl=0.0)
        params = {
            "vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0, "winPen": 0.0},
            "deer": {"birth": 1.0, "surv": 0.85, "wDeer": 0.2},
        }
        _, diag = step(s, inp, params)
        self.assertEqual(diag["births"], 0.0)


if __name__ == "__main__":
    unittest.main()


