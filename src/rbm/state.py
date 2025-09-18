from __future__ import annotations

from dataclasses import dataclass


@dataclass
class State:
    """Container for ecosystem state at a single time step.

    Fields:
      - deer: number of deer (population units)
      - pred: number of predators (population units)
      - carry: carrying capacity proxy (max sustainable deer)
    """
    deer: float
    pred: float
    carry: float


@dataclass
class Inputs:
    """Human inputs applied during this time step.

    Fields:
      - hunt: deer hunting rate (fraction 0–1 of deer removed)
      - ctrl: predator control rate (fraction 0–1 of predators removed)
    """
    hunt: float
    ctrl: float


