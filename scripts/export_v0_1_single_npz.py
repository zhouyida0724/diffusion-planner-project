#!/usr/bin/env python3
"""v0.1 batch export: read a plan index.jsonl and export into ONE NPZ + manifest + metrics.

Run INSIDE nuplan-simulation container.

Hard-skip rules:
- NaN/Inf in any required array
- shape mismatch
- route_lanes_avails_sum == 0 (routing missing)

Soft flag:
- route_min_dist_m > 30

Notes:
- Reuses map_api (one per run, since v0.1 uses single location).
- Disables per-frame debug logging inside extract_features to avoid massive I/O.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import sqlite3

# Import the refactored extractor by path (container may not treat /workspace/scripts as a package).
import importlib.util

_esf_path = "/workspace/scripts/extract_single_frame/extract_single_frame.py"
_spec = importlib.util.spec_from_file_location("extract_single_frame", _esf_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load extractor from {_esf_path}")
_esf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_esf)
esf = _esf


REQUIRED_KEYS = [
    "ego_current_state",
    "ego_past",
    "ego_agent_future",
    "neighbor_agents_past",
    "neighbor_agents_future",
    "static_objects",
    "lanes",
    "lanes_avails",
    "route_lanes",
    "route_lanes_avails",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
]


def ro_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def is_finite(arr: np.ndarray) -> bool:
    if np.issubdtype(arr.dtype, np.floating):
        return bool(np.isfinite(arr).all())
    return True


def route_min_dist_m(route_lanes: np.ndarray, route_avails: np.ndarray) -> float | None:
    # route_lanes: (25,20,12), avails: (25,20)
    mask = route_avails > 0
    if not np.any(mask):
        return None
    xs = route_lanes[:, :, 0][mask]
    ys = route_lanes[:, :, 1][mask]
    d = np.sqrt(xs * xs + ys * ys)
    if d.size == 0:
        return None
    return float(d.min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=str, required=True, help="Path to plan directory containing index.jsonl")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--map-root", type=str, default=esf.MAP_ROOT)
    ap.add_argument("--map-version", type=str, default=esf.MAP_VERSION)
    ap.add_argument("--limit", type=int, default=0, help="Optional: only process first N lines")
    args = ap.parse_args()

    # Avoid extremely noisy map RuntimeWarnings that slow down batch export via stderr I/O.
    warnings.filterwarnings("ignore", message=".*invalid value encountered in cast.*", category=RuntimeWarning)

    plan_dir = Path(args.plan)
    index_path = plan_dir / "index.jsonl"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    metrics_path = out_dir / "metrics.json"
    npz_path = out_dir / "data.npz"

    # Load plan records
    records = []
    with open(index_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            records.append(json.loads(line))

    if not records:
        raise RuntimeError("Empty plan")

    # Determine location/map from first record's DB metadata (truth)
    first_db = records[0]["db_path"]
    con0 = ro_connect(first_db)
    try:
        location = esf.get_location_from_log(con0)
        map_name = esf.map_name_from_location(location)
    finally:
        con0.close()

    map_api = esf.get_maps_api(args.map_root, args.map_version, map_name)

    # Prepare output buffers (append then stack at end)
    buffers: dict[str, list[np.ndarray]] = {k: [] for k in REQUIRED_KEYS}
    kept_meta = []

    hard_skip = 0
    soft_flag_far = 0
    soft_flag_counts = Counter()

    t0 = time.time()

    # Group by db_path to reuse sqlite connection
    by_db: dict[str, list[dict]] = {}
    for r in records:
        by_db.setdefault(r["db_path"], []).append(r)

    # Keep stable order: iterate records list but reuse connection cache
    con_cache: dict[str, sqlite3.Connection] = {}

    def get_con(db_path: str) -> sqlite3.Connection:
        if db_path not in con_cache:
            con_cache[db_path] = ro_connect(db_path)
        return con_cache[db_path]

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for idx, r in enumerate(records):
            db_path = r["db_path"]
            scene = r["scene_token_hex"]
            frame = int(r["frame_index"])
            sample_id = r.get("sample_id", f"{Path(db_path).name}:{scene}:{frame}")

            qc_flags: list[str] = []

            try:
                con = get_con(db_path)

                # Global timeout guard per sample (some map/route queries can hang).
                import signal

                class _Timeout(Exception):
                    pass

                def _alarm_handler(signum, frame_):
                    raise _Timeout("timeout")

                old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(5)
                try:
                    # Silence per-frame prints/warnings from the legacy extractor during batch runs.
                    import contextlib
                    with open(os.devnull, 'w') as _dn, contextlib.redirect_stdout(_dn):
                        feats = esf.extract_features(con, map_api, scene, frame, debug_log=False)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                # required keys
                for k in REQUIRED_KEYS:
                    if k not in feats:
                        raise RuntimeError(f"missing key {k}")
                    if not is_finite(feats[k]):
                        raise RuntimeError(f"non-finite {k}")

                # hard: routing missing
                route_av = feats["route_lanes_avails"]
                route_av_sum = int(np.sum(route_av))
                if route_av_sum == 0:
                    raise RuntimeError("route_lanes_avails_sum==0")

                # hard: lanes_avails all zero
                lanes_av_sum = int(np.sum(feats["lanes_avails"]))
                if lanes_av_sum == 0:
                    raise RuntimeError("lanes_avails_sum==0")

                # soft: route far
                rmin = route_min_dist_m(feats["route_lanes"], route_av)
                if rmin is not None and rmin > 30.0:
                    qc_flags.append("route_min_dist_gt_30m")
                    soft_flag_far += 1
                    soft_flag_counts["route_min_dist_gt_30m"] += 1

                # Keep
                for k in REQUIRED_KEYS:
                    buffers[k].append(feats[k])

                kept_meta.append(r)

                mf.write(
                    json.dumps(
                        {
                            **r,
                            "sample_id": sample_id,
                            "qc_hard_skip": False,
                            "qc_flags": qc_flags,
                            "route_lanes_avails_sum": route_av_sum,
                            "lanes_avails_sum": lanes_av_sum,
                            "route_min_dist_m": rmin,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            except _Timeout:
                hard_skip += 1
                mf.write(
                    json.dumps(
                        {
                            **r,
                            "sample_id": sample_id,
                            "qc_hard_skip": True,
                            "qc_error": "timeout",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            except Exception as e:
                hard_skip += 1
                mf.write(
                    json.dumps(
                        {
                            **r,
                            "sample_id": sample_id,
                            "qc_hard_skip": True,
                            "qc_error": str(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if (idx + 1) % 500 == 0:
                dt_s = time.time() - t0
                fps = (idx + 1) / max(dt_s, 1e-9)
                print(f"[{idx+1}/{len(records)}] processed; kept={len(kept_meta)} skip={hard_skip} fps={fps:.2f}", file=sys.stderr, flush=True)

    # Close DB connections
    for con in con_cache.values():
        try:
            con.close()
        except Exception:
            pass

    # Stack and write NPZ
    kept_n = len(kept_meta)
    if kept_n == 0:
        raise RuntimeError("No samples kept")

    stacked = {k: np.stack(buffers[k], axis=0) for k in REQUIRED_KEYS}
    np.savez_compressed(npz_path, **stacked)

    elapsed = time.time() - t0
    metrics = {
        "plan_dir": str(plan_dir),
        "map_name": map_name,
        "location": location,
        "planned": len(records),
        "kept": kept_n,
        "hard_skipped": hard_skip,
        "soft_flag_counts": dict(soft_flag_counts),
        "elapsed_s": elapsed,
        "fps_kept": kept_n / max(elapsed, 1e-9),
        "npz_path": str(npz_path),
        "npz_size_bytes": int(npz_path.stat().st_size),
        "manifest_path": str(manifest_path),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("DONE", file=sys.stderr, flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
