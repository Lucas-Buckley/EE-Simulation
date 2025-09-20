from __future__ import annotations

from typing import Tuple

from .state import State, Inputs

EPS = 1e-9

def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a number between low and high.

    Inputs:
      - value: the number to limit
      - low: the minimum allowed value
      - high: the maximum allowed value
    Output:
      - value clipped so that low <= value <= high
    """
    if value < low:
        return low
    if value > high:
        return high
    return value

def step(state: State, inputs: Inputs, params: dict) -> Tuple[State, dict]:
    """Advance the ecosystem by one year using simple rules and return the next state and diagnostics.

    Inputs:
      - state: current populations and carrying capacity (deer, pred, carry)
      - inputs: human actions (hunt rate for deer, control rate for predators) for this year
      - params: model parameters grouped under 'vegetation', 'deer', 'predation', 'predators'
    Output:
      - (next_state, diag): next_state is the updated State; diag is a dictionary of intermediate values
    """
    veg = params.get("vegetation", {})
    veg_rate = float(veg.get("vegRate", 0.0))
    cap_max = float(veg.get("capMax", max(state.carry, 1.0)))
    browse = float(veg.get("browse", 0.0))
    deer_p = params.get("deer", {})
    birth_rate = float(deer_p.get("birth", 0.0))
    surv_base = float(deer_p.get("surv", 0.0))
    pred_p = params.get("predation", {})
    pred_atk = float(pred_p.get("predAtk", 0.0))
    pred_cap = float(pred_p.get("predCap", 0.0))
    pred_eff = float(pred_p.get("predEff", 0.0))
    preds_p = params.get("predators", {})
    pred_mort = float(preds_p.get("mort", 0.0))

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

    # Step 5: Winter penalty no longer implemented, update next carry
    winter_loss = 0.0
    carry_next = state.carry + carry_growth - browse_loss - winter_loss
    carry_next = _clamp(carry_next, EPS, cap_max)
    diag.update({
        "winter_loss": winter_loss,
        "carry_next": carry_next
    })

    # Step 6: Food and births
    food = min(1.0, state.carry / max(state.deer, 1.0))
    births = state.deer * birth_rate * food
    diag.update({
        "food": food,
        "births": births
    })

    # Step 7: Natural survival fraction and survivors (no winter term)
    surv_nat_raw = surv_base * (0.5 + 0.5 * food)
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
    
    # Step 10: Deer update and non-negativity (survivors + births - removals)
    deer_next_raw = surv_num + births - kill - hunt_rem
    deer_next = max(0.0, deer_next_raw)
    diag.update({
        "deer_next": deer_next
    })

    # Step 11: Predator update
    pred_rec = pred_eff * kill
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


