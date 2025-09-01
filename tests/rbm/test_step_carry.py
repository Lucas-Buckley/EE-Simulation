import unittest

from src.rbm.state import State, Inputs
from src.rbm.step import step


class TestStepCarry(unittest.TestCase):
    def test_carry_grows_towards_cap(self):
        s = State(deer=0.0, pred=0.0, carry=100.0)
        inp = Inputs(hunt=0.0, ctrl=0.0, winter=0.0)
        params = {"vegetation": {"vegRate": 0.2, "capMax": 1000.0}}
        ns, diag = step(s, inp, params)
        self.assertGreater(ns.carry, 100.0)
        self.assertLessEqual(ns.carry, 1000.0)
        self.assertIn("carry_growth", diag)

    def test_carry_never_exceeds_cap(self):
        s = State(deer=0.0, pred=0.0, carry=1200.0)
        inp = Inputs(hunt=0.0, ctrl=0.0, winter=0.0)
        params = {"vegetation": {"vegRate": 0.5, "capMax": 1000.0}}
        ns, _ = step(s, inp, params)
        self.assertLessEqual(ns.carry, 1000.0)


if __name__ == "__main__":
    unittest.main()


