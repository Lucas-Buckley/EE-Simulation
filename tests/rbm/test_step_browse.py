import unittest

from src.rbm.state import State, Inputs
from src.rbm.step import step


class TestStepBrowse(unittest.TestCase):
    def test_browse_reduces_carry_when_deer_exceed_carry(self):
        s = State(deer=500.0, pred=0.0, carry=100.0)
        inp = Inputs(hunt=0.0, ctrl=0.0)
        params = {"vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.1}}
        ns, diag = step(s, inp, params)
        self.assertLess(ns.carry, 100.0)
        self.assertGreater(diag.get("browse_loss", 0.0), 0.0)

    def test_stronger_browse_lowers_carry_more(self):
        s = State(deer=500.0, pred=0.0, carry=100.0)
        inp = Inputs(hunt=0.0, ctrl=0.0)
        p1 = {"vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.05}}
        p2 = {"vegetation": {"vegRate": 0.0, "capMax": 1000.0, "browse": 0.2}}
        ns1, _ = step(s, inp, p1)
        ns2, _ = step(s, inp, p2)
        self.assertLess(ns2.carry, ns1.carry)


if __name__ == "__main__":
    unittest.main()


