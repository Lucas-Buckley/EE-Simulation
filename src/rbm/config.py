from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TimeCfg:
    start: int
    end: int


@dataclass
class InputsCfg:
    hunt: List[float]
    ctrl: List[float]
    winter: List[float]


@dataclass
class VegCfg:
    vegRate: float
    capMax: float
    browse: float
    winPen: float


@dataclass
class DeerCfg:
    birth: float
    surv: float
    wDeer: float


@dataclass
class PredCfg:
    predAtk: float
    predCap: float
    predEff: float


@dataclass
class PredsMortCfg:
    mort: float


@dataclass
class ParamsCfg:
    vegetation: VegCfg
    deer: DeerCfg
    predation: PredCfg
    predators: PredsMortCfg


@dataclass
class InitCfg:
    deer: float
    pred: float
    carry: float


@dataclass
class SeedsCfg:
    main: int


@dataclass
class Config:
    time: TimeCfg
    inputs: InputsCfg
    params: ParamsCfg
    init: InitCfg
    seeds: SeedsCfg


def _require_keys(obj: Dict[str, Any], keys: List[str], ctx: str) -> None:
    for k in keys:
        if k not in obj:
            raise ValueError(f"Missing key '{k}' in {ctx}")


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        # Accept JSON-as-YAML for simplicity initially
        raw = json.load(f)

    _require_keys(raw, ["time", "inputs", "params", "init", "seeds"], "root")

    time = raw["time"]
    _require_keys(time, ["start", "end"], "time")
    time_cfg = TimeCfg(start=int(time["start"]), end=int(time["end"]))
    if time_cfg.end < time_cfg.start:
        raise ValueError("time.end must be >= time.start")

    inputs = raw["inputs"]
    _require_keys(inputs, ["hunt", "ctrl", "winter"], "inputs")
    for k in ("hunt", "ctrl", "winter"):
        if not isinstance(inputs[k], list):
            raise ValueError(f"inputs.{k} must be a list")
    steps = time_cfg.end - time_cfg.start + 1
    for name in ("hunt", "ctrl", "winter"):
        if len(inputs[name]) != steps:
            raise ValueError(f"inputs.{name} length {len(inputs[name])} must equal years {steps}")
    inputs_cfg = InputsCfg(
        hunt=[float(x) for x in inputs["hunt"]],
        ctrl=[float(x) for x in inputs["ctrl"]],
        winter=[float(x) for x in inputs["winter"]],
    )

    params = raw["params"]
    _require_keys(params, ["vegetation", "deer", "predation", "predators"], "params")
    veg = params["vegetation"]
    _require_keys(veg, ["vegRate", "capMax", "browse", "winPen"], "params.vegetation")
    veg_cfg = VegCfg(
        vegRate=float(veg["vegRate"]),
        capMax=float(veg["capMax"]),
        browse=float(veg["browse"]),
        winPen=float(veg["winPen"]),
    )
    deer = params["deer"]
    _require_keys(deer, ["birth", "surv", "wDeer"], "params.deer")
    deer_cfg = DeerCfg(birth=float(deer["birth"]), surv=float(deer["surv"]), wDeer=float(deer["wDeer"]))
    predn = params["predation"]
    _require_keys(predn, ["predAtk", "predCap", "predEff"], "params.predation")
    pred_cfg = PredCfg(
        predAtk=float(predn["predAtk"]),
        predCap=float(predn["predCap"]),
        predEff=float(predn["predEff"]),
    )
    preds = params["predators"]
    _require_keys(preds, ["mort"], "params.predators")
    preds_cfg = PredsMortCfg(mort=float(preds["mort"]))
    params_cfg = ParamsCfg(vegetation=veg_cfg, deer=deer_cfg, predation=pred_cfg, predators=preds_cfg)

    init = raw["init"]
    _require_keys(init, ["deer", "pred", "carry"], "init")
    init_cfg = InitCfg(deer=float(init["deer"]), pred=float(init["pred"]), carry=float(init["carry"]))

    seeds = raw["seeds"]
    _require_keys(seeds, ["main"], "seeds")
    seeds_cfg = SeedsCfg(main=int(seeds["main"]))

    return Config(time=time_cfg, inputs=inputs_cfg, params=params_cfg, init=init_cfg, seeds=seeds_cfg)


