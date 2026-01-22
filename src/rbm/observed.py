from __future__ import annotations

import csv
from typing import List, Tuple


def _parse_int(s: str) -> int:
    return int(s.strip())


def _parse_float_with_commas(s: str) -> float:
    return float(s.replace(",", "").strip())


def load_deer_observed_csv(
    path: str,
    year_field: str = "Year",
    value_field: str = "Deer Population",
    interpolate_missing: bool = True,
) -> Tuple[List[int], List[float], List[bool]]:
    raw_years: List[int] = []
    raw_vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = _parse_int(row[year_field])
            v = _parse_float_with_commas(row[value_field])
            raw_years.append(y)
            raw_vals.append(v)

                  
    pairs = sorted(zip(raw_years, raw_vals), key=lambda t: t[0])
    raw_years = [p[0] for p in pairs]
    raw_vals = [p[1] for p in pairs]

    if not interpolate_missing:
        return raw_years, raw_vals, [False] * len(raw_years)

                                                                        
    years: List[int] = []
    values: List[float] = []
    imputed: List[bool] = []

    for idx, (y, v) in enumerate(zip(raw_years, raw_vals)):
        years.append(y)
        values.append(v)
        imputed.append(False)
        if idx == len(raw_years) - 1:
            break
        y2 = raw_years[idx + 1]
        v2 = raw_vals[idx + 1]
        gap = y2 - y
        if gap > 1:
            step = (v2 - v) / gap
            for g in range(1, gap):
                yi = y + g
                vi = v + step * g
                years.append(yi)
                values.append(vi)
                imputed.append(True)

    return years, values, imputed


