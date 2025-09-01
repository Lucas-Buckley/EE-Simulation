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
    browse = float(veg.get("browse", 0.0))
    win_pen = float(veg.get("winPen", 0.0))
    deer_p = params.get("deer", {})
    birth_rate = float(deer_p.get("birth", 0.0))
    surv_base = float(deer_p.get("surv", 0.0))
    w_deer = float(deer_p.get("wDeer", 0.0))
    pred_p = params.get("predation", {})
    pred_atk = float(pred_p.get("predAtk", 0.0))
    pred_cap = float(pred_p.get("predCap", 0.0))

    carry_growth = veg_rate * state.carry * (1.0 - state.carry / cap_max)
    browse_loss = browse * max(0.0, state.deer - state.carry)
    winter_loss = win_pen * float(inputs.winter) * cap_max
    carry_next = state.carry + carry_growth - browse_loss - winter_loss
    carry_next = _clamp(carry_next, EPS, cap_max)

    # Step 6: Food ratio and births (do not update deer yet)
    food = min(1.0, carry_next / max(state.deer, 1.0))
    births = state.deer * birth_rate * food

    # Step 7: Natural survival fraction and survivors
    surv_nat_raw = surv_base * (0.5 + 0.5 * food) * (1.0 - w_deer * float(inputs.winter))
    # Clamp survival fraction to [0, 1]
    if surv_nat_raw < 0.0:
        surv_nat = 0.0
    elif surv_nat_raw > 1.0:
        surv_nat = 1.0
    else:
        surv_nat = surv_nat_raw
    surv_num = state.deer * surv_nat

    next_state = State(deer=state.deer, pred=state.pred, carry=carry_next)
    diag = {
        "carry_growth": carry_growth,
        "browse_loss": browse_loss,
        "winter_loss": winter_loss,
        "food": food,
        "births": births,
        "survNat": surv_nat,
        "survNum": surv_num,
        "carry_next": carry_next,
    }

    # Step 8: Predation kills (raw and capped)
    kill_raw = pred_atk * state.pred * state.deer
    kill_cap = pred_cap * state.deer
    kill = min(kill_raw, kill_cap)
    diag.update({
        "killRaw": kill_raw,
        "killCap": kill_cap,
        "kill": kill,
    })
    return next_state, diag


