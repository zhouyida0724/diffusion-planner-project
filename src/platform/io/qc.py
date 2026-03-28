"""QC helpers for export scripts."""

from __future__ import annotations

import numpy as np


def is_finite(arr: np.ndarray) -> bool:
    """True if arr has no NaN/Inf when floating; always True for non-floats."""

    if np.issubdtype(arr.dtype, np.floating):
        return bool(np.isfinite(arr).all())
    return True


def route_min_dist_m(route_lanes: np.ndarray, route_avails: np.ndarray) -> float | None:
    """Compute min distance to route lane points for available route points.

    route_lanes: (25, 20, 12) and route_avails: (25, 20)
    """

    mask = route_avails > 0
    if not np.any(mask):
        return None
    xs = route_lanes[:, :, 0][mask]
    ys = route_lanes[:, :, 1][mask]
    d = np.sqrt(xs * xs + ys * ys)
    if d.size == 0:
        return None
    return float(d.min())
