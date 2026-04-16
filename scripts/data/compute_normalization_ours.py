#!/usr/bin/env python3
"""Compute normalization stats from mmap-friendly training_arrays cache.

Reads arrays from:
  <cache_root>/**/arrays/*.npy

and writes a JSON compatible with both:
  - vendor-style diffusion_planner/normalization.json (keys: ego, neighbor, + conditioning keys)
  - our PaperModelConfig loading (state_mean/state_std + observation_norm)

Masking rules (train/infer consistent):
  - For dense feature tensors (ego_current_state, neighbor_agents_past, lanes, route_lanes, static_objects):
      a row/timestep is considered padding/invalid if ALL features are exactly 0.
  - For *_speed_limit tensors: if a corresponding *_has_speed_limit.npy exists, we only count entries where has_speed_limit==1.
      Otherwise we fall back to nonzero masking.
  - For state stats (ego/neighbor): we compute over current + future steps in 4D state space [x,y,cos,sin].
      Neighbor padding is masked per (agent,timestep) by all-zero check.

Example:
  python scripts/data/compute_normalization_ours.py \
    --cache-root outputs/cache/training_arrays/train_data_boston150w \
    --out outputs/cache/normalization_ours.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def _iter_arrays_dirs(cache_roots: list[Path]) -> Iterable[Path]:
    """Yield all arrays/ directories under one or more cache roots."""

    for cache_root in cache_roots:
        # arrays dirs typically look like: .../p*/shard_*/arrays
        for p in cache_root.rglob("arrays"):
            if p.is_dir():
                yield p


def _masked_reduce_sum_sumsq_count(x: np.ndarray, *, mask_rows_all_zero: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """Reduce any shape (...,D) into (sum[D], sumsq[D], count_rows)."""

    if x.size == 0:
        d = int(x.shape[-1]) if x.ndim >= 1 else 1
        return np.zeros((d,), dtype=np.float64), np.zeros((d,), dtype=np.float64), 0

    x2 = np.asarray(x)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    else:
        x2 = x2.reshape(-1, x2.shape[-1])

    if mask_rows_all_zero:
        valid = np.any(x2 != 0, axis=-1)
        x2 = x2[valid]
    if x2.shape[0] == 0:
        return np.zeros((x2.shape[1],), dtype=np.float64), np.zeros((x2.shape[1],), dtype=np.float64), 0

    x64 = x2.astype(np.float64, copy=False)
    s = x64.sum(axis=0)
    ss = (x64 * x64).sum(axis=0)
    return s, ss, int(x64.shape[0])


def _finalize_mean_std(sum_: np.ndarray, sumsq: np.ndarray, count: int, *, eps: float = 1e-6) -> tuple[list[float], list[float]]:
    if count <= 0:
        raise RuntimeError("No valid entries counted, cannot compute mean/std")
    mean = sum_ / float(count)
    var = sumsq / float(count) - mean * mean
    var = np.maximum(var, eps)
    std = np.sqrt(var)
    return mean.astype(np.float64).tolist(), std.astype(np.float64).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-roots",
        type=str,
        nargs="+",
        default=None,
        help="One or more roots under which to search **/arrays/*.npy (preferred).",
    )
    ap.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="(Deprecated) Single root under which to search **/arrays/*.npy. Use --cache-roots instead.",
    )
    ap.add_argument("--out", type=str, default="normalization_ours.json", help="Output JSON path")
    ap.add_argument("--max-arrays-dirs", type=int, default=0, help="If >0, only process first N arrays/ dirs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    roots_raw = list(args.cache_roots or [])
    if args.cache_root:
        roots_raw.append(str(args.cache_root))
    if not roots_raw:
        raise SystemExit("No cache roots provided. Use --cache-roots (or deprecated --cache-root).")

    cache_roots = [Path(r) for r in roots_raw]
    for r in cache_roots:
        if not r.exists():
            raise SystemExit(f"cache_root not found: {r}")

    # Conditioning features (ObservationNormalizer expects these keys).
    obs_keys = [
        "ego_current_state",
        "neighbor_agents_past",
        "static_objects",
        "lanes",
        "lanes_speed_limit",
        "route_lanes",
        "route_lanes_speed_limit",
    ]

    # State stats keys (StateNormalizer expects ego/neighbor).
    need_state = ["ego_current_state", "ego_agent_future", "neighbor_agents_past", "neighbor_agents_future"]

    # Running stats.
    obs_acc: dict[str, dict[str, object]] = {}
    for k in obs_keys:
        obs_acc[k] = {"sum": None, "sumsq": None, "count": 0}

    ego_sum = None
    ego_sumsq = None
    ego_count = 0
    nb_sum = None
    nb_sumsq = None
    nb_count = 0

    arrays_dirs = list(_iter_arrays_dirs(cache_roots))
    arrays_dirs.sort()
    if int(args.max_arrays_dirs or 0) > 0:
        arrays_dirs = arrays_dirs[: int(args.max_arrays_dirs)]

    if args.verbose:
        roots_s = ", ".join([str(r) for r in cache_roots])
        print(f"Found {len(arrays_dirs)} arrays/ dirs under: {roots_s}")

    for i, d in enumerate(arrays_dirs):
        # quick presence check
        present = {p.stem for p in d.glob("*.npy")}
        if not (set(obs_keys) & present) and not (set(need_state) & present):
            continue

        if args.verbose:
            print(f"[{i+1}/{len(arrays_dirs)}] {d}")

        # observation stats
        for k in obs_keys:
            fp = d / f"{k}.npy"
            if not fp.exists():
                continue
            x = np.load(fp, mmap_mode="r")

            if k.endswith("_speed_limit"):
                # If has_speed_limit exists, count only where it is true.
                has_fp = d / f"{k}_has_speed_limit.npy"
                if not has_fp.exists():
                    # route_lanes_speed_limit -> route_lanes_has_speed_limit
                    has_fp = d / (k.replace("_speed_limit", "_has_speed_limit") + ".npy")
                if has_fp.exists():
                    has = np.load(has_fp, mmap_mode="r")
                    m = np.asarray(has) != 0
                    x2 = np.asarray(x)
                    # speed_limit is typically [N,L] or [N]
                    x2 = x2[m]
                    s, ss, c = _masked_reduce_sum_sumsq_count(x2.reshape(-1, 1), mask_rows_all_zero=False)
                else:
                    s, ss, c = _masked_reduce_sum_sumsq_count(np.asarray(x).reshape(-1, 1), mask_rows_all_zero=True)
            else:
                s, ss, c = _masked_reduce_sum_sumsq_count(x, mask_rows_all_zero=True)

            if c <= 0:
                continue
            if obs_acc[k]["sum"] is None:
                obs_acc[k]["sum"] = s
                obs_acc[k]["sumsq"] = ss
                obs_acc[k]["count"] = int(c)
            else:
                obs_acc[k]["sum"] = obs_acc[k]["sum"] + s  # type: ignore[operator]
                obs_acc[k]["sumsq"] = obs_acc[k]["sumsq"] + ss  # type: ignore[operator]
                obs_acc[k]["count"] = int(obs_acc[k]["count"]) + int(c)

        # state stats
        fp_ecs = d / "ego_current_state.npy"
        fp_ef = d / "ego_agent_future.npy"
        fp_np = d / "neighbor_agents_past.npy"
        fp_nf = d / "neighbor_agents_future.npy"
        if fp_ecs.exists() and fp_ef.exists():
            ego_cur = np.load(fp_ecs, mmap_mode="r")[:, :4]  # [N,4]
            ego_f = np.load(fp_ef, mmap_mode="r")  # [N,T,3]
            ego_xy = ego_f[..., :2]
            h = ego_f[..., 2]
            ego_cs = np.stack([np.cos(h), np.sin(h)], axis=-1)
            ego_f4 = np.concatenate([ego_xy, ego_cs], axis=-1)  # [N,T,4]
            ego_all = np.concatenate([ego_cur[:, None, :], ego_f4], axis=1)  # [N,1+T,4]
            s, ss, c = _masked_reduce_sum_sumsq_count(ego_all, mask_rows_all_zero=True)
            if c > 0:
                ego_sum = s if ego_sum is None else ego_sum + s
                ego_sumsq = ss if ego_sumsq is None else ego_sumsq + ss
                ego_count += int(c)

        if fp_np.exists() and fp_nf.exists():
            nb_cur = np.load(fp_np, mmap_mode="r")[:, :, -1, :4]  # [N,P,T,4] -> current [N,P,4]
            nb_f = np.load(fp_nf, mmap_mode="r")  # [N,P,81,3] (often includes current)
            if nb_f.shape[-2] == 81:
                nb_f = nb_f[..., 1:, :]
            nb_xy = nb_f[..., :2]
            nb_h = nb_f[..., 2]
            nb_cs = np.stack([np.cos(nb_h), np.sin(nb_h)], axis=-1)
            nb_f4 = np.concatenate([nb_xy, nb_cs], axis=-1)  # [N,P,T,4]
            nb_all = np.concatenate([nb_cur[:, :, None, :], nb_f4], axis=2)  # [N,P,1+T,4]

            # Mask per-row (agent,timestep) if all zeros.
            s, ss, c = _masked_reduce_sum_sumsq_count(nb_all, mask_rows_all_zero=True)
            if c > 0:
                nb_sum = s if nb_sum is None else nb_sum + s
                nb_sumsq = ss if nb_sumsq is None else nb_sumsq + ss
                nb_count += int(c)

    # finalize
    out: dict[str, dict[str, list[float]]] = {}

    if ego_sum is None or ego_sumsq is None or ego_count <= 0:
        raise SystemExit("No ego state entries found. Is cache_root correct?")
    if nb_sum is None or nb_sumsq is None or nb_count <= 0:
        raise SystemExit("No neighbor state entries found. Is cache_root correct?")

    ego_mean, ego_std = _finalize_mean_std(ego_sum, ego_sumsq, ego_count)
    nb_mean, nb_std = _finalize_mean_std(nb_sum, nb_sumsq, nb_count)

    out["ego"] = {"mean": ego_mean, "std": ego_std}
    out["neighbor"] = {"mean": nb_mean, "std": nb_std}

    for k in obs_keys:
        acc = obs_acc[k]
        if acc["sum"] is None or int(acc["count"]) <= 0:
            continue
        mean, std = _finalize_mean_std(acc["sum"], acc["sumsq"], int(acc["count"]))  # type: ignore[arg-type]
        out[k] = {"mean": mean, "std": std}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    if args.verbose:
        print(f"Wrote: {out_path} (keys={list(out.keys())})")


if __name__ == "__main__":
    main()
