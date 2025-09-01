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
    win_pen = float(veg.get("winPen", 0.0))
    deer_p = params.get("deer", {})
    birth_rate = float(deer_p.get("birth", 0.0))
    surv_base = float(deer_p.get("surv", 0.0))
    w_deer = float(deer_p.get("wDeer", 0.0))
    pred_p = params.get("predation", {})
    pred_atk = float(pred_p.get("predAtk", 0.0))
    pred_cap = float(pred_p.get("predCap", 0.0))

    # Step 3: Logistic carry growth (toward capMax)
    carry_growth = veg_rate * state.carry * (1.0 - state.carry / cap_max)
    diag = {
        "carry_growth": carry_growth
    }

    # Step 4: Browse impact (deer above carry reduce next-year carry)
    browse_loss = browse * max(0.0, state.deer - state.carry)
    diag.update({
        "browse_loss": browse_loss
    })

    # Step 5: Winter penalty on carry (fraction of capMax per winter unit)
    winter_loss = win_pen * float(inputs.winter) * cap_max
    carry_next = state.carry + carry_growth - browse_loss - winter_loss
    carry_next = _clamp(carry_next, EPS, cap_max)
    diag.update({
        "winter_loss": winter_loss,
        "carry_next": carry_next
    })

    # Step 6: Food and births
    food = min(1.0, carry_next / max(state.deer, 1.0))
    births = state.deer * birth_rate * food
    diag.update({
        "food": food,
        "births": births
    })

    # Step 7: Natural survival fraction and survivors
    surv_nat_raw = surv_base * (0.5 + 0.5 * food) * (1.0 - w_deer * float(inputs.winter))
    surv_nat = _clamp(surv_nat_raw, 0.0, 1.0)
    surv_num = state.deer * surv_nat
    diag.update({
        "survNat": surv_nat,
        "survNum": surv_num
    })

    # Step 8: Predation kills (raw and capped)
    kill_raw = pred_atk * state.pred * state.deer
    kill_cap = pred_cap * state.deer
    kill = min(kill_raw, kill_cap)
    diag.update({
        "killRaw": kill_raw,
        "killCap": kill_cap,
        "kill": kill
    })

    # Step 9: Hunting removals (clamp hunt rate to [0,1])
    hunt_rate = _clamp(float(inputs.hunt), 0.0, 1.0)
    hunt_rem = hunt_rate * state.deer
    diag.update({
        "huntRate": hunt_rate,
        "huntRem": hunt_rem
    })
    next_state = State(deer=state.deer, pred=state.pred, carry=carry_next)
    return next_state, diag


