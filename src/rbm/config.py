from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import csv


@dataclass
class TimeCfg:
    start: int
    end: int


@dataclass
class InputsCfg:
    hunt: List[float]
    ctrl: List[float]


@dataclass
class VegCfg:
    vegRate: float
    capMax: float
    browse: float


@dataclass
class DeerCfg:
    birth: float
    surv: float


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


def _load_inputs_csv(path: str) -> Dict[str, Any]:
    years: List[int] = []
    hunt: List[float] = []
    ctrl: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expected headers: Year, % Deer Hunted, % Predators Killed
        for row in reader:
            year = int(row.get("Year"))
            hunt_val = float(row.get("% Deer Hunted"))
            ctrl_val = float(row.get("% Predators Killed"))
            years.append(year)
            hunt.append(hunt_val)
            ctrl.append(ctrl_val)
    # Sort by year in case file isn't ordered
    sorted_triplets = sorted(zip(years, hunt, ctrl), key=lambda t: t[0])
    years = [t[0] for t in sorted_triplets]
    hunt = [t[1] for t in sorted_triplets]
    ctrl = [t[2] for t in sorted_triplets]
    return {"years": years, "hunt": hunt, "ctrl": ctrl}


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        # Accept JSON-as-YAML for simplicity initially
        raw = json.load(f)

    _require_keys(raw, ["time", "params", "init", "seeds"], "root")
    # inputs may be provided or overridden by inputsCsv

    time = raw["time"]
    _require_keys(time, ["start", "end"], "time")
    time_cfg = TimeCfg(start=int(time["start"]), end=int(time["end"]))
    if time_cfg.end < time_cfg.start:
        raise ValueError("time.end must be >= time.start")

    # Determine inputs from arrays or CSV
    inputs_cfg: InputsCfg
    if "inputsCsv" in raw or ("inputs" in raw and isinstance(raw["inputs"], dict) and "csv" in raw["inputs"]):
        csv_path = raw.get("inputsCsv") or raw["inputs"]["csv"]
        # Allow relative paths from project root
        if not os.path.isabs(csv_path):
            base_dir = os.path.dirname(os.path.abspath(path))
            csv_path = os.path.abspath(os.path.join(base_dir, os.pardir, csv_path) if csv_path.startswith("data/") else os.path.join(base_dir, csv_path))
        loaded = _load_inputs_csv(csv_path)
        years = loaded["years"]
        inputs_cfg = InputsCfg(hunt=[float(x) for x in loaded["hunt"]], ctrl=[float(x) for x in loaded["ctrl"]])
        # Override time range from CSV
        time_cfg = TimeCfg(start=min(years), end=max(years))
    else:
        inputs = raw["inputs"]
        _require_keys(inputs, ["hunt", "ctrl"], "inputs")
        for k in ("hunt", "ctrl"):
            if not isinstance(inputs[k], list):
                raise ValueError(f"inputs.{k} must be a list")
        steps = time_cfg.end - time_cfg.start + 1
        for name in ("hunt", "ctrl"):
            if len(inputs[name]) != steps:
                raise ValueError(f"inputs.{name} length {len(inputs[name])} must equal years {steps}")
        inputs_cfg = InputsCfg(
            hunt=[float(x) for x in inputs["hunt"]],
            ctrl=[float(x) for x in inputs["ctrl"]],
        )

    params = raw["params"]
    _require_keys(params, ["vegetation", "deer", "predation", "predators"], "params")
    veg = params["vegetation"]
    _require_keys(veg, ["vegRate", "capMax", "browse"], "params.vegetation")
    veg_cfg = VegCfg(
        vegRate=float(veg["vegRate"]),
        capMax=float(veg["capMax"]),
        browse=float(veg["browse"]),
    )
    deer = params["deer"]
    _require_keys(deer, ["birth", "surv"], "params.deer")
    deer_cfg = DeerCfg(birth=float(deer["birth"]), surv=float(deer["surv"]))
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


