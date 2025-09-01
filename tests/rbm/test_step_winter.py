import unittest

from src.rbm.state import State, Inputs
from src.rbm.step import step


class TestStepWinter(unittest.TestCase):
    def test_winter_reduces_carry(self):
        s = State(deer=100.0, pred=0.0, carry=500.0)
        params = {"vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0, "winPen": 0.05}}
        no_winter = Inputs(hunt=0.0, ctrl=0.0, winter=0.0)
        hard_winter = Inputs(hunt=0.0, ctrl=0.0, winter=2.0)
        ns_no, _ = step(s, no_winter, params)
        ns_hard, diag = step(s, hard_winter, params)
        self.assertLess(ns_hard.carry, ns_no.carry)
        self.assertGreater(diag.get("winter_loss", 0.0), 0.0)

    def test_zero_winter_no_penalty(self):
        s = State(deer=0.0, pred=0.0, carry=500.0)
        inp = Inputs(hunt=0.0, ctrl=0.0, winter=0.0)
        params = {"vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.0, "winPen": 0.2}}
        ns, diag = step(s, inp, params)
        self.assertEqual(diag.get("winter_loss", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()


