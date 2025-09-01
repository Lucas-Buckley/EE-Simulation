## Rule-based Predator–Prey Simulation Plan (Kaibab Project)

### 1) Purpose and scope
- **Objective**: Build a rule-based predator–prey ecosystem simulator whose outputs best fit historical observations from the Kaibab Plateau (deer boom/crash following predator control and protection policies). This plan covers modeling assumptions, rules, parameters, data needs, calibration, validation, and tests.
- **Outputs**: Time series for deer, predator populations, and forage/vegetation indices; event logs (e.g., predator control periods, severe winters); diagnostics and fit metrics.
- **Initial time resolution**: Annual time steps, with option to refine to seasonal later.

### 2) References and data
- **Historical narrative**: Kaibab deer episode (early 1900s–1930s) featuring hunting restrictions, predator control, subsequent deer irruption and crash.
- **Data needs**:
  - Deer abundance estimates by year (with uncertainty ranges if available)
  - Predator abundance indices (wolves, mountain lions, coyotes) or control effort proxies
  - Vegetation/forage availability proxy (qualitative or reconstructed indices)
  - Climate/severity indicators (e.g., severe winter years) as binary or intensity inputs
  - Policy timeline (hunting moratorium start/stop, predator control intensity by year)
- **Sources to compile**: Historical reports, NPS documents, state agency archives, academic reviews of the Kaibab case. Capture source citations and data provenance in `data/metadata.json`.

### 3) Model components
- **Entities**
  - Herbivore: deer (single aggregated population initially; optional age structure later)
  - Predators: aggregated predator population (initially single pool), with option to split by species later
  - Forage/vegetation: single-patch biomass or carrying-capacity proxy K(t)
  - Humans (exogenous): hunting pressure, predator control effort profiles over time

- **State variables (per year)**
  - Deer population: deer (non-negative)
  - Predator population: pred (non-negative)
  - Carrying capacity proxy: carry (positive)
  - Exogenous inputs: hunt (deer harvest rate), ctrl (predator control rate), winter (winter severity index)

- **Time step**: Annual (can extend to seasonal later: winter/summer sub-steps).

### 4) Rules (annual update — Calc I friendly)
Let the state at year t be (deer, pred, carry) and exogenous inputs (hunt, ctrl, winter).

All formulas use only multiplication, division, addition/subtraction, powers with small integers, and min/max.

1) Carrying capacity dynamics (simple logistic with browse and winter penalty)
   - carryNxt = carry + vegRate * carry * (1 - carry / capMax) - browse * max(0, deer - carry) - winPen * winter * capMax
   - vegRate: vegetation recovery rate (0 to 1 per year)
   - capMax: maximum carrying capacity (units consistent with deer)
   - browse: how much overbrowsing (deer > carry) reduces next year carry
   - winPen: fraction of capMax lost per unit winter severity (winter is 0 for normal, higher for severe)

2) Deer births and natural survival (simple caps based on food and winter)
   - Food ratio: food = min(1, carry / max(deer, 1))  // per-deer forage, capped at 1
   - Births: births = deer * birth * food
   - Natural survival fraction: survNat = surv * (0.5 + 0.5 * food) * (1 - wDeer * winter)
   - Natural survivors: survNum = deer * survNat
   - Parameters:
     - birth: max births per deer per year (e.g., 0.9)
     - surv: baseline survival without stress (e.g., 0.9)
     - wDeer: winter mortality sensitivity for deer (e.g., 0.1–0.3)

3) Predation on deer (linear in predators, capped)
   - Raw kills: killRaw = predAtk * pred * deer
   - Capacity cap: killCap = predCap * deer  // predators remove at most a fraction of deer
   - Kills: kill = min(killRaw, killCap)
   - Parameters:
     - predAtk: how strongly predators remove deer (per predator per deer)
     - predCap: max predator-caused fraction of deer in one year (0–1)

4) Deer removals by hunting (if applicable)
   - huntRem = hunt * deer  // hunt is fraction 0–1 or 0 if no hunting

5) Deer update (non-negative)
   - deerNxt = max(0, deer + births + survNum - kill - huntRem)

6) Predator dynamics (prey-coupled births, simple mortalities)
   - Predator recruits from prey intake: predRec = predEff * kill
   - Predator natural mortality: pMort = mort * pred
   - Predator control: ctrlRem = ctrl * pred  // ctrl is fraction 0–1
   - predNxt = max(0, pred + predRec - pMort - ctrlRem)

7) Stochasticity (optional toggles)
   - Add small random noise terms ε sampled once per year if desired (keep mean 0)
   - Severe winters: set winter > 0 for those years; otherwise winter = 0

Constraints and guards
- Populations are non-negative; fractional values allowed in deterministic mode; integers in stochastic/agent mode via draws.
- Bounds: carry in (0, capMax]; optionally cap deer to prevent numeric blow-ups during exploratory calibration.

### 5) Initialization
- Choose start year (e.g., 1905) with plausible initial values deer0, pred0, carry0 from historical context.
- Define annual input series across the horizon: hunt[year], ctrl[year], winter[year].

### 6) Parameters (quick ref; Calc I friendly)
- Vegetation:
  - vegRate: how fast carry regrows toward capMax when it is below the max. Example: vegRate = 0.15 means carry increases by about 15% of its current value this year (before other effects). Range: [0.05, 0.4].
  - capMax: the hard upper limit of how many deer the land can support (units: deer). Pick about 2–5× the first-year carry. Example: if carry0 = 30,000, set capMax between 60,000 and 150,000. Higher capMax allows larger peaks.
  - browse: how strongly extra deer above carry reduce next year’s carry (units: fraction per deer). The drop to carry is browse × (deer − carry). Example: if deer − carry = 10,000 and browse = 0.10, next year’s carry is reduced by 1,000.
  - winPen: fraction of capMax lost per unit of winter severity. The winter term subtracts winPen × winter × capMax. Example: if winter = 2 and winPen = 0.05 with capMax = 100,000, carry loses 0.05 × 2 × 100,000 = 10,000.
- Deer:
  - birth: the maximum number of fawns per deer per year in great food years (units: fawns per deer). Actual births scale by food. Range: [0.6, 1.2].
  - surv: baseline fraction of deer that survive natural causes when there is no stress (0–1). Range: [0.6, 0.95].
  - wDeer: how much winter hurts deer survival per unit of winter. If wDeer = 0.2 and winter = 1, survival is multiplied by (1 − 0.2) = 0.8. Range: [0, 0.3].
- Predation:
  - predAtk: how strongly predators remove deer (per predator per deer). Higher means more kills for the same pred and deer. Range: [1e-7, 1e-3].
  - predCap: the maximum fraction of the deer herd that predators can remove in one year (0–1). This caps kills so they cannot exceed predCap × deer. Range: [0.05, 0.6].
  - predEff: how efficiently deer kills turn into new predators (conversion from prey eaten to predator recruits). Range: [1e-4, 5e-3].
- Predators:
  - mort: baseline fraction of predators that die each year from natural causes (0–1). Range: [0.05, 0.3].
- Inputs (per year; not fixed params):
  - hunt: fraction of deer removed by hunting this year (0–1). 0 means no hunting.
  - ctrl: fraction of predators removed by control this year (0–1). 0 means no control.
  - winter: winter severity index (0 = normal; 1 = harsh; 2 = very harsh). Used in both carry and deer survival.

### 7) Outputs (quick ref) and logging
- Annual outputs (saved per year):
  - year: the calendar year of the record.
  - deer: deer population at the start of the year (units: deer).
  - pred: predator population at the start of the year (units: predators).
  - carry: carrying capacity at the start of the year (units: deer the land can support).
  - births: number of new deer born during this year (units: deer).
  - survNum: number of deer that survive natural causes this year (units: deer). This excludes predation and hunting.
  - kill: number of deer removed by predation this year (units: deer).
  - huntRem: number of deer removed by hunting this year (units: deer).
  - predRec: number of predators added due to deer kills (units: predators).
  - pMort: number of predators that die from natural causes this year (units: predators).
  - ctrlRem: number of predators removed by control this year (units: predators).
  - deerNxt: deer population at the end of the year (units: deer).
  - predNxt: predator population at the end of the year (units: predators).
  - carryNxt: carrying capacity at the end of the year (units: deer).
- Derived:
  - food: per-deer food ratio = carry / max(deer, 1), capped at 1 in the code (dimensionless).
  - dDeer: change in deer this year = deerNxt − deer (units: deer). dPred: change in predators = predNxt − pred (units: predators).
- Event logs: record policy changes (e.g., hunting ban on/off), predator control, and severe winters.
- Seeds and configs: record run metadata for full reproducibility.

### 8) Fit targets and evaluation metrics (Calc I friendly)
- **Primary fit**: deer time series shape (irruption and crash)
  - Mean squared error (MSE), scaled by observed mean: average of (sim − obs)^2, then divide by mean(obs)^2
  - Peak timing error: number of years between simulated and observed peak (aim ≤ 2)
  - Peak height error: percentage difference between simulated and observed peak (aim ≤ 20%)
  - Crash ratio error: compare D_trough/D_peak between sim and obs; aim within ±0.15
- **Secondary**: visual trend checks for predators vs control; K trend shows drop then rebound
- **Stochastic runs**: compute mean and 10th–90th percent bands across many runs; check if observed series mostly falls inside

### 9) Calibration strategy (Calc I friendly)
- Grid search or random search over parameter ranges; keep top N by primary fit score
- Hand-tune near best sets by increasing/decreasing one parameter at a time and re-running
- Validate by running multiple seeds (if noise enabled) and ensuring targets hold in most runs
- Record settings, seeds, and results in `experiments/`

### 10) Test plan (engineering tests — Calc I friendly)
- Unit tests (deterministic):
  - Forage logistic step respects bounds: if D=0 and W=0, K increases toward K_max
  - No-negative-population invariant after updates
  - Predation removals increase when P increases, holding others fixed
  - Hunting/control application respects [0,1] bounds and quotas
- Property-based tests:
  - Increasing predator control ctrl (with others fixed) does not increase predNxt
  - With high forage and no predators/harvest, deer growth is positive until near carry
  - Under extreme overbrowsing (deer ≫ carry), carry declines year-over-year
- Scenario tests:
  - Predator removal period produces deer peak followed by crash when K is degraded
  - Reintroducing predators reduces deer growth and stabilizes oscillations
  - Severe winter spike (W_t high) induces expected additional mortality
- Regression and reproducibility:
  - Fixed seed reproduces identical trajectories and metrics
  - Baseline config hashed; metrics compared against stored golden values with tolerances
- Statistical adequacy:
  - Coverage: observed values fall within simulated envelope (e.g., 80% band) at ≥ target rate

### 11) Acceptance thresholds (initial; refine as data clarifies)
- RMSE (scaled by observed mean) ≤ 0.25
- Peak timing error ≤ 2 years
- Peak magnitude error ≤ 20%
- Cross-correlation (best lag within ±3) ≥ 0.6
- Coverage of observed series by 80% simulation band ≥ 60%

### 12) Experiment protocol and artifacts
- Directory structure (proposed):
  - `data/` raw and processed data; `metadata.json` with sources
  - `configs/` YAML/JSON configs for model params and input schedules
  - `src/rbm/` rule-based model implementation
  - `tests/rbm/` unit, property, and scenario tests
  - `experiments/` runs with timestamped folders: config, seed, outputs, metrics
  - `notebooks/` EDA and fit diagnostics
- Reproducibility:
  - Global RNG seeding; record library versions and Git SHA
  - Deterministic mode toggle for unit tests
  - Metric computation scripts versioned and tested

### 13) Config schema (sketch — Calc I friendly)
```yaml
time:
  start: 1905
  end: 1940
inputs:
  hunt: [ ... per year ... ]
  ctrl: [ ... per year ... ]
  winter: [ ... per year ... ]
params:
  vegetation: { vegRate: 0.15, capMax: 100000, browse: 0.1, winPen: 0.05 }
  deer: { birth: 1.0, surv: 0.85, wDeer: 0.2 }
  predation: { predAtk: 0.0001, predCap: 0.3, predEff: 0.0015 }
  predators: { mort: 0.12 }
init:
  deer: 30000
  pred: 200
  carry: 30000
seeds:
  main: 42
```

### 14) Pseudocode for one simulation step (Calc I friendly)
```text
function step(state, inputs, params):
  (deer, pred, carry) = state
  (hunt, ctrl, winter) = inputs

  # Forage update
  carryNxt = carry + vegRate*carry*(1 - carry/capMax) - browse*max(0, deer - carry) - winPen*winter*capMax
  carryNxt = clamp(carryNxt, eps, capMax)

  # Deer components
  food = min(1, carry / max(deer, 1))
  births = deer * birth * food
  survNat = surv * (0.5 + 0.5*food) * (1 - wDeer*winter)
  survNum = deer * survNat
  huntRem = hunt * deer

  # Predation
  killRaw = predAtk * pred * deer
  killCap = predCap * deer
  kill = min(killRaw, killCap)

  # Deer update
  deerNxt = max(0, deer + births + survNum - kill - huntRem)

  # Predator components and update
  predRec = predEff * kill
  pMort = mort * pred
  ctrlRem = ctrl * pred
  predNxt = max(0, pred + predRec - pMort - ctrlRem)

  return (deerNxt, predNxt, carryNxt), diag
```

### 15) Milestones
- M1: Data compilation and input schedules finalized; baseline config created
- M2: Deterministic rule-based core implemented with unit tests green
- M3: Calibration phase 1 (global), candidate sets identified
- M4: Calibration phase 2 (local), acceptance thresholds met on deterministic run
- M5: Stochastic validation and sensitivity analyses, acceptance sustained
- M6: Documentation and experiment bundles published

### 16) Risks and mitigations
- Sparse or unreliable predator data: use control effort as proxy; widen priors
- Overfitting to deer series: incorporate vegetation and predator plausibility checks
- Structural mis-specification: run ablations (e.g., no K feedback) to confirm necessity of rules
- Equifinality: use multiple metrics and priors; report parameter uncertainty

### 17) Next actions
- Compile Kaibab data with provenance; build `inputs/` time series
- Implement `src/rbm` scaffolding and config loader
- Write tests in `tests/rbm` per Section 10
- Stand up calibration runner and metric reporters


