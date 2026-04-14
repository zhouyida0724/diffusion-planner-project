#!/usr/bin/env python3
"""Compare cached training features (data.npz) vs re-extracted features (extract_single_frame).

This is a concrete proof harness for train↔infer alignment at the *feature semantics* level.

Inputs
- --shard-dir must contain:
    - data.npz
    - manifest.jsonl

The script will:
- pick one sample (by --index or --sample-id)
- load its cached tensors from data.npz
- re-extract features for the same (db_path, scene_token_hex, frame_index)
  using: src/platform/nuplan/features/extract_single_frame.extract_features
- compare key set, shapes, dtypes, and values (within tolerance)

Exit code
- 0: PASS
- 2: mismatch
- 3: missing external resources (db/map)

Note
This validates that the *extractor path* used to build training caches matches runtime extraction.
To validate actual closed-loop inputs, enable DP_FEATURE_CONTRACT_CHECK=1 in the planner.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
_NUPLAN_VIS = _REPO_ROOT / "nuplan-visualization"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if _NUPLAN_VIS.is_dir() and str(_NUPLAN_VIS) not in sys.path:
    sys.path.insert(0, str(_NUPLAN_VIS))


def _load_manifest(shard_dir: Path) -> list[dict[str, Any]]:
    mp = shard_dir / "manifest.jsonl"
    if not mp.is_file():
        raise FileNotFoundError(f"manifest.jsonl not found: {mp}")
    out: list[dict[str, Any]] = []
    with mp.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _pick_sample(manifest: list[dict[str, Any]], *, index: int | None, sample_id: str | None) -> tuple[int, dict[str, Any]]:
    if sample_id is not None:
        for i, obj in enumerate(manifest):
            if str(obj.get("sample_id")) == sample_id:
                return i, obj
        raise KeyError(f"sample_id not found in manifest: {sample_id}")

    if index is None:
        index = 0
    if index < 0 or index >= len(manifest):
        raise IndexError(f"index out of range: {index} (manifest size {len(manifest)})")
    return int(index), manifest[int(index)]


def _build_map_api(location: str):
    # location in manifest is usually a map name like "us-ma-boston".
    from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db

    from src.platform.nuplan.features.extract_single_frame import MAP_ROOT, MAP_VERSION

    maps_db = get_maps_db(str(MAP_ROOT), str(MAP_VERSION))
    factory = NuPlanMapFactory(maps_db)
    return factory.build_map_from_name(str(location))


def _compare_arrays(k: str, a: np.ndarray, b: np.ndarray, tol: float) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if a.dtype != b.dtype:
        # allow bool vs uint8 style drift to be caught explicitly
        return False, f"dtype {a.dtype} vs {b.dtype}"

    if a.dtype == np.bool_:
        eq = bool(np.array_equal(a, b))
        if not eq:
            diff = int(np.count_nonzero(a != b))
            return False, f"bool mismatch count={diff}"
        return True, "ok"

    # float/other numeric
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    diff = np.abs(da - db)
    mx = float(diff.max()) if diff.size else 0.0
    mean = float(diff.mean()) if diff.size else 0.0
    if mx > tol:
        return False, f"max_abs_diff={mx:.6g} mean_abs_diff={mean:.6g} tol={tol}"
    return True, f"max_abs_diff={mx:.6g} mean_abs_diff={mean:.6g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=str, required=True)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--sample-id", type=str, default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--routing-mode", type=str, default="auto", help="Passed to extract_features (auto|raw)")
    ap.add_argument(
        "--db-path-rewrite",
        action="append",
        default=[],
        help=(
            "Optional path prefix rewrite for manifest db_path, repeatable. "
            "Format: OLD=NEW. Example: --db-path-rewrite /media/datasets=/workspace/data/nuplan/data/cache"
        ),
    )
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    data_path = shard_dir / "data.npz"
    if not data_path.is_file():
        raise FileNotFoundError(f"data.npz not found: {data_path}")

    manifest = _load_manifest(shard_dir)
    row_idx, meta = _pick_sample(manifest, index=args.index, sample_id=args.sample_id)

    db_path_s = str(meta.get("db_path"))
    # Apply optional prefix rewrites (useful when running inside a container with different mounts).
    for rule in list(args.db_path_rewrite or []):
        if "=" not in rule:
            raise SystemExit(f"Invalid --db-path-rewrite (expected OLD=NEW): {rule}")
        old, new = rule.split("=", 1)
        if db_path_s.startswith(old):
            db_path_s = new + db_path_s[len(old) :]
    db_path = Path(db_path_s)
    scene_token_hex = str(meta.get("scene_token_hex"))
    frame_index = int(meta.get("frame_index"))
    location = str(meta.get("location"))

    print(f"[compare] sample_id={meta.get('sample_id')} row_idx={row_idx}")
    print(f"[compare] db_path={db_path}")
    print(f"[compare] scene_token_hex={scene_token_hex} frame_index={frame_index} location={location}")

    if not db_path.is_file():
        print(f"[compare][error] db_path does not exist on this machine: {db_path}")
        return 3

    try:
        map_api = _build_map_api(location)
    except Exception as e:
        print(f"[compare][error] failed to build map_api for location={location}: {e}")
        return 3

    cached = np.load(str(data_path))
    cached_keys = [k for k in cached.files]

    # load cached sample
    cached_sample: dict[str, np.ndarray] = {k: cached[k][row_idx] for k in cached_keys}

    # re-extract
    from src.platform.nuplan.features.extract_single_frame import extract_features

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        extracted = extract_features(
            conn,
            map_api,
            scene_token_hex,
            frame_index,
            debug_log=False,
            routing_mode=str(args.routing_mode),
        )
    finally:
        conn.close()

    extracted = {k: v for k, v in extracted.items() if not str(k).startswith("_")}

    # Compare key sets (ignore keys present only in cached, if extractor doesn't produce them)
    ck = set(cached_sample.keys())
    ek = set(extracted.keys())

    missing = sorted(list(ck - ek))
    extra = sorted(list(ek - ck))
    if missing or extra:
        print(f"[compare][mismatch] keyset missing_from_extractor={missing} extra_from_extractor={extra}")
        return 2

    ok_all = True
    for k in sorted(cached_sample.keys()):
        a = np.asarray(cached_sample[k])
        b = np.asarray(extracted[k])
        ok, msg = _compare_arrays(k, a, b, float(args.tol))
        if not ok:
            ok_all = False
            print(f"[compare][FAIL] {k}: {msg}")
        else:
            print(f"[compare][ok]   {k}: {msg}")

    if not ok_all:
        return 2

    print("[compare] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
