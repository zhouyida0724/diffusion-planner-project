#!/usr/bin/env python3
"""Regression test for scripts/extract_single_frame/extract_single_frame.py::extract_features.

Goal:
  - Keep NPZ outputs unchanged after refactor introducing extract_features(...).
  - Compare current extracted features against known-good saved NPZs.

This test is intentionally lightweight and does not require pytest.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np

# Some nuPlan visualization modules import OpenCV for image utilities.
# Feature extraction in this repo doesn't require cv2, so we stub it to
# keep this regression test runnable in minimal environments.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("pytest", types.ModuleType("pytest"))

# Import the refactored extractor as a plain module.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "extract_single_frame"))
import extract_single_frame as esf  # noqa: E402

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api  # noqa: E402


REPO_ROOT = _THIS_DIR.parent
DEFAULT_MANIFEST = REPO_ROOT / "validation_output" / "bfsfix_manifest_5.json"
DEFAULT_MAP_ROOT = REPO_ROOT / "data" / "nuplan" / "maps"
DEFAULT_MAP_VERSION = "9.12.1817"


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


def _assert_close_array(name: str, a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} != {b.shape}")
    if a.dtype != b.dtype:
        raise AssertionError(f"{name}: dtype mismatch {a.dtype} != {b.dtype}")
    if not np.allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True):
        # Show a small diff summary.
        diff = np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))
        raise AssertionError(f"{name}: not allclose, max_abs_diff={diff}")


def main() -> int:
    manifest_path = Path(os.environ.get("MANIFEST", str(DEFAULT_MANIFEST)))
    map_root = Path(os.environ.get("MAP_ROOT", str(DEFAULT_MAP_ROOT)))
    map_version = os.environ.get("MAP_VERSION", DEFAULT_MAP_VERSION)

    cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = list(cases)[:5]

    failures: list[str] = []

    for i, c in enumerate(cases):
        db_path = c["db_path"]
        scene_token_hex = c["scene_token_hex"]
        frame_index = int(c["frame_index"])

        npz_path = c.get("npz_path")
        if npz_path and npz_path.startswith("/workspace/"):
            # Normalize OpenClaw-in-repo paths.
            npz_path = str(REPO_ROOT / npz_path.removeprefix("/workspace/"))
        npz_path = str(npz_path) if npz_path else None

        try:
            assert npz_path and Path(npz_path).is_file(), f"missing golden npz: {npz_path}"

            con = _open_db(db_path)
            try:
                location = esf.get_location_from_log(con)
                map_name = esf.map_name_from_location(location)
                map_api = get_maps_api(str(map_root), map_version, map_name)

                feats = esf.extract_features(con, map_api, scene_token_hex, frame_index)
            finally:
                con.close()

            gold = dict(np.load(npz_path, allow_pickle=False))

            # Keys must match exactly.
            if set(feats.keys()) != set(gold.keys()):
                missing = sorted(set(gold.keys()) - set(feats.keys()))
                extra = sorted(set(feats.keys()) - set(gold.keys()))
                raise AssertionError(f"keys mismatch. missing={missing} extra={extra}")

            for k in sorted(gold.keys()):
                a = feats[k]
                b = gold[k]

                # Avails should be exact.
                if k.endswith("avails") or a.dtype == np.bool_:
                    _assert_equal_array(k, a, b)
                elif np.issubdtype(a.dtype, np.floating):
                    _assert_close_array(k, a, b)
                else:
                    _assert_equal_array(k, a, b)

            print(f"[{i+1}/{len(cases)}] PASS scene={scene_token_hex} frame={frame_index} gold={Path(npz_path).name}")
        except Exception as e:
            msg = f"[{i+1}/{len(cases)}] FAIL scene={scene_token_hex} frame={frame_index}: {e}"
            print(msg)
            failures.append(msg)

    if failures:
        print("\nFailures:")
        for m in failures:
            print("  " + m)
        return 1

    print(f"\nAll {len(cases)} cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
