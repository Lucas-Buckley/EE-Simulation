from __future__ import annotations

from typing import Iterable, List, Tuple


def _to_list(seq: Iterable[float]) -> List[float]:
    """Convert any iterable to a standard Python list of floats."""
    return list(seq)


def compute_scaled_mse(observed: Iterable[float], simulated: Iterable[float]) -> float:
    """Compute mean squared error scaled by the square of the observed mean.

    Inputs:
      - observed: target values to fit
      - simulated: predicted/simulated values
    Output:
      - a dimensionless error; lower is better. If mean(observed)==0, returns unscaled MSE.
    """
    obs = _to_list(observed)
    sim = _to_list(simulated)
    if len(obs) == 0 or len(sim) == 0:
        raise ValueError("observed and simulated must be non-empty")
    if len(obs) != len(sim):
        raise ValueError("observed and simulated must have the same length")
    n = len(obs)
    mse = sum((sim[i] - obs[i]) ** 2 for i in range(n)) / n
    mean_obs = sum(obs) / n
    if mean_obs == 0:
        return mse
    return mse / (mean_obs ** 2)


def find_peak(series: Iterable[float]) -> Tuple[float, int]:
    """Find the maximum value and its index in a series.

    Input:
      - series: iterable of numeric values
    Output:
      - (peak_value, peak_index)
    """
    seq = _to_list(series)
    if not seq:
        raise ValueError("series must be non-empty")
    peak_idx = max(range(len(seq)), key=lambda i: seq[i])
    return seq[peak_idx], peak_idx


def compute_peak_timing_error(
    years_obs: Iterable[int], observed: Iterable[float], years_sim: Iterable[int], simulated: Iterable[float]
) -> int:
    """Compute absolute difference (in years) between observed and simulated peak timings.

    Inputs:
      - years_obs/years_sim: year labels for each series
      - observed/simulated: data series aligned with years above
    Output:
      - non-negative integer number of years between the two peaks
    """
    y_obs = _to_list(years_obs)
    y_sim = _to_list(years_sim)
    obs = _to_list(observed)
    sim = _to_list(simulated)
    if not obs or not sim:
        raise ValueError("observed and simulated must be non-empty")
    if len(y_obs) != len(obs) or len(y_sim) != len(sim):
        raise ValueError("years and series length must match for observed and simulated")
    # find peaks
    _, p_obs = find_peak(obs)
    _, p_sim = find_peak(sim)
    return abs(y_sim[p_sim] - y_obs[p_obs])


def compute_peak_height_error(observed: Iterable[float], simulated: Iterable[float]) -> float:
    """Compute relative peak height error: |sim_peak - obs_peak| / max(obs_peak, 1e-12)."""
    obs = _to_list(observed)
    sim = _to_list(simulated)
    if not obs or not sim:
        raise ValueError("observed and simulated must be non-empty")
    obs_peak, _ = find_peak(obs)
    sim_peak, _ = find_peak(sim)
    denom = max(obs_peak, 1e-12)
    return abs(sim_peak - obs_peak) / denom


def compute_crash_ratio(series: Iterable[float]) -> float:
    """Compute crash ratio = trough_after_peak / peak.

    Trough is defined as the minimum value at or after the peak index. If the series has a single point,
    crash ratio is 1.0. If peak is zero, returns 0.0 to avoid division by zero.
    """
    seq = _to_list(series)
    if not seq:
        raise ValueError("series must be non-empty")
    peak_val, peak_idx = find_peak(seq)
    trough_val = min(seq[peak_idx:]) if peak_idx < len(seq) else seq[-1]
    if peak_val == 0:
        return 0.0
    return trough_val / peak_val


def compute_crash_ratio_error(observed: Iterable[float], simulated: Iterable[float]) -> float:
    """Compute absolute difference between simulated and observed crash ratios."""
    obs_ratio = compute_crash_ratio(observed)
    sim_ratio = compute_crash_ratio(simulated)
    return abs(sim_ratio - obs_ratio)


