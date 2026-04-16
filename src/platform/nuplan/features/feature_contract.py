"""Feature contract checks for diffusion planner inputs.

Goal
- Catch train↔infer drift early (schema, shapes, padding rules, NaNs).
- Contract is defined by *our exporter/training cache semantics*, not vendor defaults.

Enable
  DP_FEATURE_CONTRACT_CHECK=1

This is intentionally lightweight (no external deps).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def _enabled() -> bool:
    v = os.environ.get("DP_FEATURE_CONTRACT_CHECK", "0")
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def maybe_check_feature_contract(inputs: dict[str, Any], *, batched: bool) -> None:
    """Validate a minimal, high-signal subset of the feature contract.

    Args:
      inputs: either numpy arrays (unbatched) or torch tensors (batched).
      batched: True if tensors include leading batch dim.
    """

    if not _enabled():
        return

    def _np(x: Any) -> np.ndarray:
        # torch.Tensor support without importing torch
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    required = [
        "ego_current_state",
        "neighbor_agents_past",
        "static_objects",
        "lanes",
        "lanes_speed_limit",
        "lanes_has_speed_limit",
    ]
    for k in required:
        if k not in inputs:
            raise RuntimeError(f"[feature_contract] missing key: {k}")

    ego = _np(inputs["ego_current_state"])
    nb = _np(inputs["neighbor_agents_past"])
    st = _np(inputs["static_objects"])
    lanes = _np(inputs["lanes"])
    spd = _np(inputs["lanes_speed_limit"])
    has = _np(inputs["lanes_has_speed_limit"])

    if batched:
        if ego.shape[-1] != 10:
            raise RuntimeError(f"[feature_contract] ego_current_state last dim must be 10, got {ego.shape}")
        if nb.shape[-1] != 11:
            raise RuntimeError(f"[feature_contract] neighbor_agents_past last dim must be 11, got {nb.shape}")
        if lanes.shape[-1] != 12:
            raise RuntimeError(f"[feature_contract] lanes last dim must be 12, got {lanes.shape}")
    else:
        if ego.shape != (10,):
            raise RuntimeError(f"[feature_contract] ego_current_state shape must be (10,), got {ego.shape}")
        if nb.ndim != 3 or nb.shape[-1] != 11:
            raise RuntimeError(f"[feature_contract] neighbor_agents_past shape must be (P,V,11), got {nb.shape}")
        if lanes.ndim != 3 or lanes.shape[-1] != 12:
            raise RuntimeError(f"[feature_contract] lanes shape must be (L,Len,12), got {lanes.shape}")

    # Finite checks
    for name, arr in [("ego_current_state", ego), ("neighbor_agents_past", nb), ("static_objects", st), ("lanes", lanes), ("lanes_speed_limit", spd), ("lanes_has_speed_limit", has)]:
        if not np.isfinite(arr).all():
            raise RuntimeError(f"[feature_contract] {name} contains NaN/Inf")

    # Ego heading representation is cos/sin
    c = ego[..., 2]
    s = ego[..., 3]
    if (np.abs(c) > 1.05).any() or (np.abs(s) > 1.05).any():
        raise RuntimeError("[feature_contract] ego cos/sin out of range")

    # Neighbor tail semantics (ours): dims 8/9 are usually width/length (meters), dim10 is valid.
    # This catches accidental swap to vendor type-onehot(3) (mostly {0,1}).
    tail = nb[..., 8:]
    # only consider non-padding rows
    nonpad = np.any(nb[..., :8] != 0, axis=-1)
    if nonpad.any():
        tvals = tail[nonpad]
        # if *all* tail values are in {0,1}, it's very likely vendor onehot got fed in.
        uniq = np.unique(np.round(tvals.reshape(-1), 3))
        if len(uniq) > 0 and np.all(np.isin(uniq, [0.0, 1.0])):
            raise RuntimeError(
                "[feature_contract] neighbor_agents_past tail appears binary-only {0,1}. "
                "Expected width/length/valid-style tail (e.g., 1.8/4.5). Possible train↔infer schema mismatch."
            )

    # Speed limit tensors: allow either [B,L] or [L].
    if spd.ndim not in (1, 2):
        raise RuntimeError(f"[feature_contract] lanes_speed_limit ndim must be 1 or 2, got {spd.shape}")
    if has.shape != spd.shape:
        raise RuntimeError(f"[feature_contract] lanes_has_speed_limit shape must match lanes_speed_limit, got {has.shape} vs {spd.shape}")
