"""Metrics aggregation helpers for export scripts."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def summarize_durations(values: list[float]) -> dict:
    """Return summary stats for a list of durations (seconds)."""

    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def bucketize(values: Iterable[float], *, bins: list[float]) -> dict:
    """Simple fixed-bin histogram.

    Returns dict with keys like '<=b0', '(b0,b1]', ..., '>last'.
    """

    out: Counter[str] = Counter()
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        placed = False
        prev = None
        for b in bins:
            if x <= b:
                if prev is None:
                    out[f"<= {b}"] += 1
                else:
                    out[f"({prev}, {b}]"] += 1
                placed = True
                break
            prev = b
        if not placed and bins:
            out[f"> {bins[-1]}"] += 1
    return dict(out)
