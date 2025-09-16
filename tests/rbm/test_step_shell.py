import unittest

from src.rbm.state import State, Inputs
from src.rbm.step import step


class TestStepShell(unittest.TestCase):
    def test_step_pass_through(self):
        s = State(deer=10.0, pred=2.0, carry=5.0)
        inp = Inputs(hunt=0.0, ctrl=0.0)
        ns, diag = step(s, inp, params={})
        self.assertIsInstance(ns, State)
        self.assertIsInstance(diag, dict)


if __name__ == "__main__":
    unittest.main()


