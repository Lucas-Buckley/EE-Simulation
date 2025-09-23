## Kaibab Rule-based Simulation Walkthrough

This short guide shows how to:
- Run a single year and multi-year simulation
- Load observed deer, with and without interpolation
- Compute fit metrics
- Run a simple calibration search
- Produce stochastic bands

All commands assume the repo root as the working directory.

### 1) Single-year run
```bash
python -c 'from src.rbm.run_one import run_once; import os; cfg=os.path.abspath("configs/base.yaml"); out=os.path.abspath("experiments/run_one.csv"); print(run_once(cfg,out))'
```

Outputs one row to `experiments/run_one.csv` with start-year diagnostics.

### 2) Multi-year run
```bash
python -c 'from src.rbm.run_years import run_years; import os; cfg=os.path.abspath("configs/base.yaml"); out=os.path.abspath("experiments/run_years.csv"); rows=run_years(cfg,out); print(len(rows), "years written")'
```

Writes per-year outputs to `experiments/run_years.csv`.

### 3) Observed deer: load with and without interpolation
```python
from src.rbm.observed import load_deer_observed_csv
import os

obs_path = os.path.abspath("data/kaibab_deer.csv")

# With interpolation (fills internal missing years linearly)
years_i, deer_i, imputed_i = load_deer_observed_csv(obs_path, interpolate_missing=True)

# Without interpolation (only the rows present in the CSV)
years_raw, deer_raw, imputed_raw = load_deer_observed_csv(obs_path, interpolate_missing=False)

print(years_i[:5], deer_i[:5], imputed_i[:5])
print(years_raw)
```

### 4) Compute fit metrics (example)
```python
from src.rbm.metrics import compute_scaled_mse, find_peak, compute_peak_timing_error, compute_peak_height_error, compute_crash_ratio, compute_crash_ratio_error

# Suppose `sim_years`/`sim_deer` come from run_years and `years_i`/`deer_i` from step 3
def align(y_obs, v_obs, y_sim, v_sim):
    i = {y:v for y,v in zip(y_obs, v_obs)}
    j = {y:v for y,v in zip(y_sim, v_sim)}
    ys = sorted(set(i)&set(j))
    return ys, [i[y] for y in ys], [j[y] for y in ys]

ys, obs, sim = align(years_i, deer_i, sim_years, sim_deer)
print("Scaled MSE:", compute_scaled_mse(obs, sim))
print("Obs crash ratio:", compute_crash_ratio(obs))
print("Sim crash ratio:", compute_crash_ratio(sim))
```

### 5) Random search algorithm calibration
```python
from src.rbm.calib import calibrate_random_search, calibrate_compare_interpolation
import os

cfg = os.path.abspath("configs/base.yaml")
ranges = {
  "vegetation": {"vegRate": (0.05, 0.25), "capMax": (60000, 180000), "browse": (0.0, 0.2)},
  "deer": {"birth": (0.6, 1.2), "surv": (0.6, 0.95)},
  "predation": {"predAtk": (1e-6, 1e-3), "predCap": (0.05, 0.6), "predEff": (1e-4, 5e-3)},
  "predators": {"mort": (0.05, 0.3)}
}

out = os.path.abspath("experiments/calib")
best_params, best_score = calibrate_random_search(
    cfg,
    observed_years=None,
    observed_deer=None,
    param_ranges=ranges,
    trials=50,
    seed=42,
    out_dir=out,
    observed_csv_path=os.path.abspath("data/kaibab_deer.csv"),
    interpolate_observed=True,
)
print("Best score:", best_score)

# Compare with and without interpolation
cmp = calibrate_compare_interpolation(cfg, ranges, trials=30, seed=42, out_dir=os.path.join(out, "cmp"), observed_csv_path=os.path.abspath("data/kaibab_deer.csv"))
print(cmp)
```

### 6) Stochastic bands (mean, p10, p90)
```python
from src.rbm.run_stochastic import run_years_stochastic
import os

cfg = os.path.abspath("configs/base.yaml")
out_csv = os.path.abspath("experiments/run_stochastic.csv")
rows = run_years_stochastic(cfg, out_csv, repeats=200, seed=123)
print("Wrote:", out_csv)
```

### Notes
- Interpolation: linear fills for observed deer create synthetic points in gaps to stabilize metrics. Compare results with and without interpolation (Section 5).
- Reproducibility: Provide seeds for calibration and stochastic runs.
- Outputs: CSVs are written under `experiments/`. Inspect or plot as needed.

