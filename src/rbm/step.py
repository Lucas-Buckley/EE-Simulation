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
    veg = params.get("vegetation", {})
    veg_rate = float(veg.get("vegRate", 0.0))
    cap_max = float(veg.get("capMax", max(state.carry, 1.0)))
    browse = float(veg.get("browse", 0.0))
    deer_p = params.get("deer", {})
    birth_rate = float(deer_p.get("birth", 0.0))
    surv_base = float(deer_p.get("surv", 0.0))
    pred_p = params.get("predation", {})
    pred_kill = float(pred_p.get("predAtk", 0.0))
    pred_cap = float(pred_p.get("predCap", 0.0))
    pred_eff = float(pred_p.get("predEff", 0.0))
    preds_p = params.get("predators", {})
    pred_mort = float(preds_p.get("mort", 0.0))

                                                   
    carry_growth = veg_rate * state.carry * (1.0 - state.carry / cap_max)
    diag = {
        "carry_growth": carry_growth
    }

                                                                     
    browse_loss = browse * max(0.0, state.deer - state.carry)
    diag.update({
        "browse_loss": browse_loss
    })

                               
    carry_next = state.carry + carry_growth - browse_loss
    carry_next = _clamp(carry_next, EPS, cap_max)
    diag.update({
        "carry_next": carry_next
    })

                             
    food = min(1.0, state.carry / max(state.deer, 1.0))
    births = state.deer * birth_rate * food
    diag.update({
        "food": food,
        "births": births
    })

                                                     
    surv_nat_raw = surv_base * (0.5 + 0.5 * food)
    surv_nat = _clamp(surv_nat_raw, 0.0, 1.0)
    surv_num = state.deer * surv_nat
    diag.update({
        "survNat": surv_nat,
        "survNum": surv_num
    })

                                              
    pred_raw = pred_kill * state.pred * state.deer
    pred_cap = pred_cap * state.deer
    predation = min(pred_raw, pred_cap)
    diag.update({
        "predRaw": pred_raw,
        "predCap": pred_cap,
        "predation": predation
    })

                                                         
    hunt_rate = float(inputs.hunt)
    hunt_rem = hunt_rate * state.deer
    diag.update({
        "huntRate": hunt_rate,
        "huntRem": hunt_rem
    })
    
                                                                             
    deer_next_raw = surv_num + births - predation - hunt_rem
    deer_next = max(0.0, deer_next_raw)
    diag.update({
        "deer_next": deer_next
    })

                              
    pred_rec = pred_eff * predation
    p_mort = pred_mort * state.pred
    ctrl_rem = _clamp(float(inputs.ctrl), 0.0, 1.0) * state.pred
    pred_next_raw = state.pred + pred_rec - p_mort - ctrl_rem
    pred_next = max(0.0, pred_next_raw)
    diag.update({
        "predRec": pred_rec,
        "pMort": p_mort,
        "ctrlRem": ctrl_rem,
        "pred_next": pred_next
    })

    next_state = State(deer=deer_next, pred=pred_next, carry=carry_next)
    return next_state, diag


