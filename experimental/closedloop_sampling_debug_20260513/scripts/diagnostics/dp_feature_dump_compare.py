#!/usr/bin/env python3
"""Compare runtime feature dumps (from DP_RUNTIME_FEATURE_DUMP_DIR) against offline extractor output.

This is meant to *prove* train↔infer alignment from results.

Runtime side:
  Set DP_RUNTIME_FEATURE_DUMP_DIR=/path and run closed-loop sim; planner dumps tick*.npz + tick*.json.

Offline side:
  This script re-runs the *same* core extractor used for exports on the dumped DB+timestamp,
  then diffs arrays key-by-key.

Notes
- We intentionally compare using timestamp_us (not frame_index) to avoid ambiguity.
- For DB queries we pick the nearest lidar_pc at/before timestamp within the same scene.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def _nearest_lidar_token(conn, *, timestamp_us: int, scenario_token_hex: str) -> bytes:
    """Resolve the lidar token at/before timestamp within the *same scene*.

    STRICT CONTRACT:
      - scenario_token_hex is a lidar_pc.token hex string.
      - We resolve its scene_token via lidar_pc(token=...), then pick the latest lidar_pc <= timestamp
        within that scene.

    This matches the runtime extractor contract used by DP dumps.
    """

    cur = conn.cursor()
    lidar_tok = bytes.fromhex(str(scenario_token_hex))
    cur.execute("SELECT scene_token FROM lidar_pc WHERE token = ? LIMIT 1", (lidar_tok,))
    rr = cur.fetchone()
    if rr is None:
        raise RuntimeError(f"invalid scenario_token (expected lidar_pc.token hex): {scenario_token_hex}")
    scene_tok = rr[0]

    cur.execute(
        """
        SELECT token
        FROM lidar_pc
        WHERE scene_token = ? AND timestamp <= ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (scene_tok, int(timestamp_us)),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("no lidar_pc <= timestamp in resolved scene")
    return row[0]


def main() -> None:
    # Ensure repo root is importable so `src.*` works when invoked from anywhere.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Also add vendor nuplan tree (nuplan-visualization) for nuplan.* imports used by the extractor.
    nuplan_vendor = repo_root / "nuplan-visualization"
    if nuplan_vendor.is_dir() and str(nuplan_vendor) not in sys.path:
        sys.path.insert(0, str(nuplan_vendor))

    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="path to one tickXXXX.json (runtime dump metadata)")
    ap.add_argument("--npz", default=None, help="optional path to tickXXXX.npz (defaults to sibling)")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=0.0)
    args = ap.parse_args()

    meta_p = Path(args.dump)
    meta = _load_json(meta_p)

    npz_p = Path(args.npz) if args.npz else meta_p.with_suffix(".npz")
    dumped = dict(np.load(npz_p, allow_pickle=False))

    db_path = meta.get("db_path")
    map_name = meta.get("map_name")
    scenario_token = meta.get("scenario_token")
    ts = int(meta.get("timestamp_us"))

    if not (db_path and map_name and scenario_token and ts > 0):
        raise RuntimeError(f"metadata incomplete: {meta}")

    # Lazy imports (nuplan deps)
    import sqlite3

    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    from src.platform.nuplan.features.extract_single_frame import (
        MAP_ROOT,
        MAP_VERSION,
        extract_features_at_timestamp,
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    map_api = get_maps_api(MAP_ROOT, MAP_VERSION, map_name)
    recomputed = extract_features_at_timestamp(
        conn,
        map_api,
        scenario_token_hex=str(scenario_token) if scenario_token else None,
        timestamp_us=int(ts),
        routing_mode="auto",
        debug_log=False,
    )

    # Fit recomputed arrays to dumped shapes (runtime may crop/pad to model cfg).
    fitted: dict[str, Any] = {}
    for k, a in dumped.items():
        if k not in recomputed:
            continue
        b = np.asarray(recomputed[k])
        if b.shape == a.shape and b.dtype == a.dtype:
            fitted[k] = b
            continue

        # Create a zero/false buffer and copy overlap.
        out = np.zeros(a.shape, dtype=a.dtype)
        slices = tuple(slice(0, min(int(b.shape[i]), int(a.shape[i]))) for i in range(len(a.shape)))
        out[slices] = b[slices]
        fitted[k] = out

    recomputed = fitted

    # Diff
    keys = sorted(set(dumped.keys()) & set(recomputed.keys()))
    bad = []
    for k in keys:
        a = np.asarray(dumped[k])
        b = np.asarray(recomputed[k])
        if a.shape != b.shape or a.dtype != b.dtype:
            bad.append((k, f"shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"))
            continue
        if a.dtype == np.bool_:
            if not np.array_equal(a, b):
                bad.append((k, "bool_mismatch"))
            continue
        if not np.allclose(a, b, rtol=float(args.rtol), atol=float(args.atol)):
            mx = float(np.max(np.abs(a - b)))
            bad.append((k, f"max_abs_diff={mx}"))

    if bad:
        print("MISMATCH:")
        for k, msg in bad:
            print(f"  {k}: {msg}")
        raise SystemExit(2)

    print("OK: dumped features match offline extractor for compared keys")


if __name__ == "__main__":
    main()
