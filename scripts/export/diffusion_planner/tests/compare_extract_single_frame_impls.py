#!/usr/bin/env python3
"""Compare and benchmark single-frame extractor implementations (py vs cy).

Policy:
- bool/int arrays: exact equality
- float arrays: allclose (rtol=1e-5, atol=1e-5)

This script is intentionally lightweight (no pytest dependency).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import types
from pathlib import Path

import numpy as np

# Some nuPlan visualization modules import OpenCV; extraction doesn't need it.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))


_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "validation_output" / "bfsfix_manifest_5.json"
DEFAULT_MAP_ROOT = REPO_ROOT / "data" / "nuplan" / "maps"
DEFAULT_MAP_VERSION = "9.12.1817"

# Keep import behavior consistent with the legacy extractor entrypoint.
# Add nuplan-visualization to path (container behavior) + repo-local fallback.
sys.path.insert(0, "/workspace/nuplan-visualization")
_local_nuplan_vis = (REPO_ROOT / "nuplan-visualization").resolve()
if _local_nuplan_vis.is_dir():
    sys.path.insert(0, str(_local_nuplan_vis))

# Ensure repo root is importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api  # noqa: E402

from src.platform.nuplan.features import extract_single_frame as esf_py  # noqa: E402

try:
    from src.platform.nuplan.features import extract_single_frame_cy as esf_cy  # type: ignore  # noqa: E402
except Exception as e:
    esf_cy = None
    _cy_import_err = e
else:
    _cy_import_err = None


def _normalize_db_path(db_path: str) -> str:
    p = Path(db_path)
    if p.is_file():
        return str(p)

    # Common mount-name mismatch on this machine: 新加卷 vs 新加卷1.
    s = str(db_path)
    if "/media/zhouyida/新加卷/" in s:
        alt = s.replace("/media/zhouyida/新加卷/", "/media/zhouyida/新加卷1/")
        if Path(alt).is_file():
            return alt

    return str(p)


def _open_db(db_path: str) -> sqlite3.Connection:
    db_path = _normalize_db_path(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _assert_equal_array(name: str, a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} != {b.shape}")
    if a.dtype != b.dtype:
        raise AssertionError(f"{name}: dtype mismatch {a.dtype} != {b.dtype}")
    if not np.array_equal(a, b):
        raise AssertionError(f"{name}: array not equal (exact)")


def _assert_close_array(name: str, a: np.ndarray, b: np.ndarray, *, rtol: float, atol: float) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} != {b.shape}")
    if a.dtype != b.dtype:
        raise AssertionError(f"{name}: dtype mismatch {a.dtype} != {b.dtype}")
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        diff = np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))
        raise AssertionError(f"{name}: not allclose, max_abs_diff={diff}")


def _compare_features(
    feats_a: dict[str, np.ndarray],
    feats_b: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> None:
    if set(feats_a.keys()) != set(feats_b.keys()):
        missing = sorted(set(feats_b.keys()) - set(feats_a.keys()))
        extra = sorted(set(feats_a.keys()) - set(feats_b.keys()))
        raise AssertionError(f"keys mismatch. missing={missing} extra={extra}")

    for k in sorted(feats_a.keys()):
        a = feats_a[k]
        b = feats_b[k]

        # Avails should be exact.
        if k.endswith("avails") or a.dtype == np.bool_:
            _assert_equal_array(k, a, b)
        elif np.issubdtype(a.dtype, np.floating):
            _assert_close_array(k, a, b, rtol=rtol, atol=atol)
        else:
            _assert_equal_array(k, a, b)


def _time_fn(fn, *, warmup: int, runs: int) -> float:
    for _ in range(max(0, warmup)):
        fn()

    dts: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        dts.append(time.perf_counter() - t0)

    return float(np.median(np.array(dts, dtype=np.float64)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    ap.add_argument("--map-root", type=str, default=str(DEFAULT_MAP_ROOT))
    ap.add_argument("--map-version", type=str, default=str(DEFAULT_MAP_VERSION))
    ap.add_argument("--limit", type=int, default=5, help="Compare first N cases (0=all)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    if esf_cy is None:
        print("[compare] cy implementation unavailable; cannot compare.")
        print(f"[compare] import error: {_cy_import_err}")
        return 2

    manifest_path = Path(args.manifest)
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.limit and int(args.limit) > 0:
        cases = list(cases)[: int(args.limit)]

    failures: list[str] = []
    py_times: list[float] = []
    cy_times: list[float] = []

    for i, c in enumerate(cases):
        db_path = c["db_path"]
        scene_token_hex = c["scene_token_hex"]
        frame_index = int(c["frame_index"])

        try:
            con = _open_db(db_path)
            try:
                # map selection matches the legacy extractor logic
                location = esf_py.get_location_from_log(con)
                map_name = esf_py.map_name_from_location(location)
                map_api = get_maps_api(str(args.map_root), str(args.map_version), map_name)

                # correctness
                feats_py = esf_py.extract_features(con, map_api, scene_token_hex, frame_index)
                feats_cy = esf_cy.extract_features(con, map_api, scene_token_hex, frame_index)
                _compare_features(feats_py, feats_cy, rtol=float(args.rtol), atol=float(args.atol))

                # perf (reuse same map_api/conn)
                t_py = _time_fn(
                    lambda: esf_py.extract_features(con, map_api, scene_token_hex, frame_index),
                    warmup=int(args.warmup),
                    runs=int(args.runs),
                )
                t_cy = _time_fn(
                    lambda: esf_cy.extract_features(con, map_api, scene_token_hex, frame_index),
                    warmup=int(args.warmup),
                    runs=int(args.runs),
                )
                py_times.append(t_py)
                cy_times.append(t_cy)

            finally:
                con.close()

            sp = (t_py / t_cy) if t_cy > 0 else float("inf")
            print(
                f"[{i+1}/{len(cases)}] PASS scene={scene_token_hex} frame={frame_index} "
                f"py={t_py*1e3:.1f}ms cy={t_cy*1e3:.1f}ms speedup={sp:.2f}x"
            )

        except Exception as e:
            msg = f"[{i+1}/{len(cases)}] FAIL scene={scene_token_hex} frame={frame_index}: {e}"
            print(msg)
            failures.append(msg)

    if not py_times or not cy_times:
        print("\n[compare] no timings collected.")
    else:
        med_py = float(np.median(np.array(py_times, dtype=np.float64)))
        med_cy = float(np.median(np.array(cy_times, dtype=np.float64)))
        speedup = (med_py / med_cy) if med_cy > 0 else float("inf")
        print("\nSummary (median over cases):")
        print(f"  py: {med_py*1e3:.1f} ms")
        print(f"  cy: {med_cy*1e3:.1f} ms")
        print(f"  speedup: {speedup:.2f}x")

    if failures:
        print("\nFailures:")
        for m in failures:
            print("  " + m)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
