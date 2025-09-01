from __future__ import annotations

from dataclasses import dataclass


@dataclass
class State:
    deer: float
    pred: float
    carry: float


@dataclass
class Inputs:
    hunt: float
    ctrl: float
    winter: float


