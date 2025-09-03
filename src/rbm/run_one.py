from __future__ import annotations

import csv
import os
from dataclasses import asdict
from typing import Dict, Any

from .config import load_config
from .state import State, Inputs
from .step import step


OUTPUT_FIELDS = [
    "year",
    "deer",
    "pred",
    "carry",
    "births",
    "survNum",
    "kill",
    "huntRem",
    "predRec",
    "pMort",
    "ctrlRem",
    "deerNxt",
    "predNxt",
    "carryNxt",
]


def run_once(config_path: str, out_csv_path: str) -> Dict[str, Any]:
    cfg = load_config(config_path)

    # Initialize state and inputs for the first year
    year = cfg.time.start
    idx = 0
    state = State(deer=cfg.init.deer, pred=cfg.init.pred, carry=cfg.init.carry)
    inputs = Inputs(
        hunt=cfg.inputs.hunt[idx],
        ctrl=cfg.inputs.ctrl[idx],
        winter=cfg.inputs.winter[idx],
    )

    params_dict = asdict(cfg.params)

    next_state, diag = step(state, inputs, params_dict)

    row = {
        "year": year,
        "deer": state.deer,
        "pred": state.pred,
        "carry": state.carry,
        "births": diag.get("births", 0.0),
        "survNum": diag.get("survNum", 0.0),
        "kill": diag.get("kill", 0.0),
        "huntRem": diag.get("huntRem", 0.0),
        "predRec": diag.get("predRec", 0.0),
        "pMort": diag.get("pMort", 0.0),
        "ctrlRem": diag.get("ctrlRem", 0.0),
        "deerNxt": next_state.deer,
        "predNxt": next_state.pred,
        "carryNxt": diag.get("carry_next", next_state.carry),
    }

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    return row


if __name__ == "__main__":
    # Default: read configs/base.yaml and write experiments/run_one.csv
    base_cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "configs", "base.yaml")
    out_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "experiments", "run_one.csv")
    base_cfg = os.path.abspath(base_cfg)
    out_csv = os.path.abspath(out_csv)
    run_once(base_cfg, out_csv)


