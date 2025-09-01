## Incremental Plan for Rule-based Kaibab Simulation

This plan breaks the project into very small increments. Each step lists the goal, concrete deliverables, and simple validation before moving on. Keep steps short; do not start the next step until validations pass.

> Validation rule: Any time you ask to implement a step, that step must also be validated immediately after implementation. Run the listed validations and report pass/fail before proceeding to the next step.

### Step 0: Repo scaffolding
- Goal: Set up folders and a minimal README for tracking work.
- Deliverables:
  - `data/`, `configs/`, `src/rbm/`, `tests/rbm/`, `experiments/`, `docs/`
  - Update `README.md` with how to run tests
- Validate:
  - Folders exist; README instructions render

### Step 1: Config schema stub
- Goal: Create a simple config loader with defaults matching `docs/rule_based_plan.md` names.
- Deliverables:
  - `configs/base.yaml` with keys: `time`, `inputs`, `params`, `init`, `seeds`
  - `src/rbm/config.py` (or `.ts`/`.js`) to read and validate
- Validate:
  - Load file and print parsed values
  - Unit test checks presence and types of required keys

### Step 2: State and one-step function shell
- Goal: Implement `step(state, inputs, params)` with no logic (pass-through) and dataclasses/types for state.
- Deliverables:
  - `src/rbm/state.py` with types for `deer`, `pred`, `carry`
  - `src/rbm/step.py` with placeholder `step`
- Validate:
  - Unit test ensures function signature and returns a state

### Step 3: Carry update (no browse, no winter)
- Goal: Implement logistic carry growth: `carryNxt = carry + vegRate*carry*(1 - carry/capMax)`
- Deliverables:
  - `src/rbm/step.py` carry update and clamp to `[eps, capMax]`
- Validate:
  - Unit: if `deer=0` and `winter=0`, carry grows toward `capMax`
  - Property: carry never exceeds `capMax`

### Step 4: Add browse impact
- Goal: Subtract `browse * max(0, deer - carry)` from carry update.
- Deliverables:
  - Update `step` to include browse term
- Validate:
  - Unit: when `deer > carry`, next carry is smaller vs without browse
  - Property: stronger `browse` lowers carry more (monotonic)

### Step 5: Add winter impact on carry
- Goal: Subtract `winPen * winter * capMax` from carry update.
- Deliverables:
  - Update `step` to include winter penalty
- Validate:
  - Unit: with higher `winter`, carry decreases more
  - Edge: if `winter=0`, no winter loss applied

### Step 6: Food ratio and births
- Goal: Implement `food = min(1, carry / max(deer, 1))` and `births = deer * birth * food`.
- Deliverables:
  - Update `step` with `food` and `births`
- Validate:
  - Unit: when `carry` doubles, births increase; when `deer=0`, births=0
  - Edge: when `carry >= deer`, `food=1`

### Step 7: Natural survival
- Goal: Implement `survNat = surv * (0.5 + 0.5*food) * (1 - wDeer*winter)` and `survNum = deer * survNat`.
- Deliverables:
  - Update `step` with survival terms
- Validate:
  - Unit: higher `winter` reduces `survNat`
  - Property: `0 <= survNat <= 1` when params in bounds

### Step 8: Predation kills (raw and cap)
- Goal: Implement `killRaw = predAtk * pred * deer`, `killCap = predCap * deer`, `kill = min(killRaw, killCap)`.
- Deliverables:
  - Update `step` with predation
- Validate:
  - Unit: increasing `pred` increases `kill`
  - Edge: `kill <= killCap`

### Step 9: Hunting removals
- Goal: Implement `huntRem = hunt * deer`.
- Deliverables:
  - Update `step` with hunting
- Validate:
  - Unit: when `hunt=0`, no hunting removals
  - Edge: cap `hunt` in [0,1] in validation

### Step 10: Deer update and non-negativity
- Goal: Compute `deerNxt = max(0, survNum + births - kill - huntRem)`.
- Deliverables:
  - Update `step` return values
- Validate:
  - Unit: `deerNxt` never negative
  - Scenario: with no preds/hunt and high carry, deer grows

### Step 11: Predator update
- Goal: Implement `predRec = predEff * kill`, `pMort = mort * pred`, `ctrlRem = ctrl * pred`, `predNxt = max(0, pred + predRec - pMort - ctrlRem)`.
- Deliverables:
  - Update `step` with predator dynamics
- Validate:
  - Unit: higher `ctrl` reduces `predNxt`
  - Edge: `predNxt` never negative

### Step 12: One-year runner and CSV output
- Goal: Add a runner that loads config, runs one step, and writes CSV of outputs.
- Deliverables:
  - `src/rbm/run_one.py` that prints and saves one-year outputs
- Validate:
  - CSV has fields listed in docs (year, deer, pred, carry, ...)

### Step 13: Multi-year runner
- Goal: Loop steps across years, applying `inputs` per year.
- Deliverables:
  - `src/rbm/run_years.py`
- Validate:
  - Deterministic run reproducible with same config
  - Outputs length equals number of years

### Step 14: Minimal Kaibab inputs
- Goal: Create a simple inputs timeline (hunt, ctrl, winter) matching the historical story.
- Deliverables:
  - `configs/kaibab_min.yaml`
- Validate:
  - Sanity check plots for deer boom then crash behavior

### Step 15: Basic metrics (MSE, peaks)
- Goal: Implement MSE (scaled), peak timing, peak height, crash ratio.
- Deliverables:
  - `src/rbm/metrics.py`
  - `tests/rbm/test_metrics.py`
- Validate:
  - Unit tests with small synthetic series

### Step 16: Calibration loop (grid/random search)
- Goal: Try parameter sets and keep the best few by metric score.
- Deliverables:
  - `src/rbm/calib.py`
  - `experiments/<timestamp>/` saved configs and scores
- Validate:
  - Re-running with same seed recreates top results

### Step 17: Stochastic runs and bands
- Goal: Add noise toggle and compute mean and 10–90% bands.
- Deliverables:
  - Update runner to support `seeds: [..]`
- Validate:
  - Coverage check: observed series mostly inside band

### Step 18: Documentation and examples
- Goal: Add notebook or README examples that show how to run and interpret metrics.
- Deliverables:
  - `notebooks/kaibab_walkthrough.ipynb` (or markdown report)
- Validate:
  - Steps run cleanly end to end