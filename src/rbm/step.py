from __future__ import annotations

from typing import Tuple

from .state import State, Inputs


def step(state: State, inputs: Inputs, params: dict) -> Tuple[State, dict]:
    # Pass-through shell for Step 2; no logic yet.
    diag = {}
    return state, diag


