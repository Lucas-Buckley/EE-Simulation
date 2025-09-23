## Bayesian Optimization Calibration Plan

This plan layers a Bayesian optimization (BO) tuner on top of the existing rule-based Kaibab simulator. The random search algorithm stays as a baseline; the BO path should deliver a more sample-efficient fit to the historical deer series. Each step includes concrete deliverables and validation so we can stop after any milestone with a working artifact.

> Validation rule: Any time you implement a step, run the listed checks right away and report whether they pass before moving on.

> Code style rule: Every function must include a short docstring that explains, in plain language, what it does, what each input is, and what it returns.

> Plain-English rule: Write comments and docstrings so a non-expert can follow them. Define any unavoidable technical term briefly the first time it appears.

> Naming rule: Use clear, descriptive function and variable names. Prefer full words over abbreviations unless the term is standard and defined.

> I/O doc rule: Document inputs and outputs for every function, including types or shapes when helpful.

> Language rule: Write everything in plain English unless it is functional code.

### 0) Baseline sanity check
- **Goal**: Confirm the current configuration and metrics run cleanly so we have a reference point.
- **Deliverables**:
  - Re-run `calibrate_random_search` for a small trial count (e.g., 10) and stash the output CSV in `experiments/baseline_random_search/`.
  - Record the best scaled MSE, peak timing error, and crash ratio error in `experiments/baseline_random_search/summary.json`.
- **Validation**:
  - `python -m unittest discover -s tests` passes.
  - `python -c 'from src.rbm.calib import calibrate_random_search; ...'` reproduces the baseline summary without exceptions.

### 1) Plan the Bayesian optimization function
- **Goal**: Spell out the function interface and settings BO needs so it sits next to `calibrate_random_search` without surprises.
- **Deliverables**:
  - Short design notes in this file that cover:
    - The function name and arguments, for example `calibrate_bayes_opt(...)`.
    - Which parameter ranges and BO settings (number of trials, exploration weight) we will support.
    - Which library we will call for BO (for example `scikit-optimize` and its `gp_minimize` helper).
- **Validation**:
  - Checklist in this design section confirming the new function lines up with the patterns already used in `calib.py`.

### 2) Build the optimizer helper
- **Goal**: Add a module that exposes a reusable optimization helper that works without the CLI.
- **Deliverables**:
  - New file `src/rbm/bayes_optimize.py` with:
    - A function that turns `ParamRanges` into the bounds expected by the BO library.
    - An objective wrapper that runs `run_years` and returns the score number.
    - `calibrate_bayes_opt` with the same arguments as the random search helper (`config_path`, `param_ranges`, `iterations`, `seed`, and friends).
    - An optional logging hook that writes each trial into `experiments/<timestamp>/trials.csv`.
- **Validation**:
  - Unit test `tests/rbm/test_bayes_optimize.py::test_space_conversion` confirms bounds map correctly and respect min/max order.
  - Unit test `tests/rbm/test_bayes_optimize.py::test_objective_reproducible` checks two calls with the same arguments and seed give the same score.

### 3) Plug into the calibration scripts
- **Goal**: Let a user choose BO from the command line or helper scripts just like random search.
- **Deliverables**:
  - Update `src/rbm/calib.py` to expose a `main_bayes_opt` entry point that mirrors the random search flow with temporary configs and logging.
  - Add an `--optimizer {random, bayes}` flag to `scripts/run_kaibab.py` (if that script is in use).
  - Add a “Bayesian optimization calibration” section to `docs/notebooks/kaibab_walkthrough.md`.
- **Validation**:
  - Manual run: `python -c 'from src.rbm.bayes_optimize import calibrate_bayes_opt; ...'` completes at least five iterations without error and writes trial logs.
  - CLI smoke test: `python scripts/run_kaibab.py --optimizer bayes --config configs/base.yaml --trials 5` exits with code 0 and produces an experiments folder.

### 4) Add comparison tests
- **Goal**: Show that the BO path improves or at least matches the random search baseline on controlled runs.
- **Deliverables**:
  - Synthetic test in `tests/rbm/test_bayes_optimize.py::test_bo_beats_random`:
    - Use a simple quadratic objective (no simulator call) to show BO reaches a lower error within the same number of iterations.
  - Integration test `tests/rbm/test_calib_bo.py` that runs both optimizers for a very small trial count (about four runs) on the actual simulator and checks that the BO score is no worse than a small allowance above the random score.
  - Fixture helpers that stub `run_years` so the tests stay fast.
- **Validation**:
  - `python -m unittest tests/rbm/test_bayes_optimize.py` and `tests/rbm/test_calib_bo.py` pass locally and in the automated test run (CI).
  - If randomness causes flakes, add fixed seeds and describe them in the test docstrings.

### 5) Document how to run it and what we learned
- **Goal**: Capture how to run, compare, and explain the new optimizer for the essay.
- **Deliverables**:
  - Update `docs/notebooks/kaibab_walkthrough.md` and/or add `docs/rbm/bayes_opt_results.md` with:
    - Example commands.
    - A simple comparison table of the best-fit metrics versus random search.
    - Plain-language notes on runtime and iteration counts.
  - Add a short note in Step 9 of `docs/rbm/rule_based_plan.md` that points to BO as the preferred advanced calibration method.
- **Validation**:
  - Re-run the notebook or markdown steps end-to-end to confirm the commands still work.
  - Store the latest experiment metrics in `experiments/bayes_opt/summary.json` so we can cite them in the essay.

### 6) Optional stretch: explore uncertainty near the best answer
- **Goal**: If time remains, sample around the BO best point to describe parameter uncertainty.
- **Deliverables**:
  - Script `src/rbm/bayes_opt_post.py` that perturbs the best-found parameters and re-evaluates the metrics.
  - Plot of score versus parameter samples saved under `experiments/bayes_opt/posterior.png`.
- **Validation**:
  - Lightweight test that runs the perturbation script and produces a CSV; skip deeper statistical checks to keep runtime low.

### Test summary
- Unit: bound conversion, deterministic objective scoring, BO helper utilities.
- Property: BO never returns NaN scores; repeated runs with the same seed reproduce identical best metrics.
- Integration: The BO CLI produces an experiment directory with trial logs and a `best_params.json`; the comparison test shows BO is as good as or better than random search within a small tolerance for tiny budgets.
- Regression: Extend `experiments/README.md` (if created) with instructions to re-run both optimizers and compare recorded metrics before releases.
