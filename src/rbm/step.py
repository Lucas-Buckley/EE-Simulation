from __future__ import annotations

from typing import Tuple

from .state import State, Inputs

EPS = 1e-9

def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value

def step(state: State, inputs: Inputs, params: dict) -> Tuple[State, dict]:
    # Step 3: Carry update (logistic growth only; no browse, no winter yet)
    veg = params.get("vegetation", {})
    veg_rate = float(veg.get("vegRate", 0.0))
    cap_max = float(veg.get("capMax", max(state.carry, 1.0)))

    carry_growth = veg_rate * state.carry * (1.0 - state.carry / cap_max)
    carry_next = state.carry + carry_growth
    carry_next = _clamp(carry_next, EPS, cap_max)

    next_state = State(deer=state.deer, pred=state.pred, carry=carry_next)
    diag = {"carry_growth": carry_growth, "carry_next": carry_next}
    return next_state, diag


