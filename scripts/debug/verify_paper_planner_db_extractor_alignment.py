#!/usr/bin/env python3
"""Verify that PaperDiTDpmPlanner's new DB-backed extractor path is aligned.

We changed PaperDiTDpmPlanner to build model inputs using:
  src.platform.nuplan.features.extract_single_frame.extract_features

This script sanity-checks, on a handful of mini DB scenes, that:
- planner can resolve (db_path, scene_token_hex)
- planner timestamp->frame_index mapping is consistent
- planner-extracted tensors match extract_features output (value-wise) for the same frame_index

This validates the *inference path now matches training/export extractor semantics*.

Exit codes:
  0: PASS
  2: mismatch
  3: missing resources
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_sys_path() -> None:
    rr = _repo_root()
    for p in [rr, rr / "nuplan-visualization"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _build_map_api(map_root: Path, map_version: str, map_name: str):
    from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db

    maps_db = get_maps_db(str(map_root), str(map_version))
    return NuPlanMapFactory(maps_db).build_map_from_name(str(map_name))


def _to_numpy(x: Any) -> np.ndarray:
    import torch

    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _compare(a: np.ndarray, b: np.ndarray, *, tol: float) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if a.dtype != b.dtype:
        # allow float32/float64 mismatch by casting below
        pass
    da = a.astype(np.float64)
    db = b.astype(np.float64)
    d = np.abs(da - db)
    mx = float(d.max() if d.size else 0.0)
    mean = float(d.mean() if d.size else 0.0)
    if mx > tol:
        return False, f"max={mx:.6g} mean={mean:.6g} tol={tol}"
    return True, f"max={mx:.6g} mean={mean:.6g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, required=True)
    ap.add_argument("--map-root", type=str, default=None)
    ap.add_argument("--map-version", type=str, default="9.12.1817")
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--frame-indices", type=str, default="0,10,20")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    _add_sys_path()
    rr = _repo_root()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        print(f"[error] db not found: {db_path}")
        return 3

    map_root = Path(args.map_root).expanduser().resolve() if args.map_root else (rr / "data" / "nuplan" / "maps")

    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from nuplan.diffusion_planner.paper_dit_dpm_planner import PaperDiTDpmPlanner
    from src.platform.nuplan.features.extract_single_frame import build_nuplan_scenario_from_db, extract_features

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    map_name = str(cur.execute("select map_version from log limit 1").fetchone()[0])
    map_api = _build_map_api(map_root, args.map_version, map_name)

    rows = cur.execute("select token from scene order by rowid limit ?", (int(args.scenes),)).fetchall()
    scene_tokens = [r[0].hex() for r in rows]

    # small set of frame indices per scene
    frame_indices = [int(x) for x in str(args.frame_indices).split(",") if str(x).strip()]

    ok_all = True

    for si, scene_token_hex in enumerate(scene_tokens):
        print("\n" + "=" * 80)
        print(f"[scene {si}] scene_token_hex={scene_token_hex} map={map_name}")

        scenario = build_nuplan_scenario_from_db(conn, str(db_path), scene_token_hex, map_name)

        planner = PaperDiTDpmPlanner(
            ckpt_path="/dev/null",
            diffusion_steps=10,
            past_trajectory_sampling=TrajectorySampling(num_poses=20, time_horizon=2.0),
            future_trajectory_sampling=TrajectorySampling(num_poses=80, time_horizon=8.0),
            device="cpu",
            scenario=scenario,
        )
        # minimal init for extraction
        planner._map_api = map_api  # type: ignore[attr-defined]

        try:
            planner._init_db_feature_extractor()  # type: ignore[attr-defined]
        except Exception as e:
            ok_all = False
            print(f"[FAIL] init_db_feature_extractor: {type(e).__name__}: {e}")
            continue

        # access timestamps list
        ts_list = getattr(planner, "_ego_pose_timestamps")  # type: ignore[attr-defined]
        if not ts_list:
            ok_all = False
            print("[FAIL] empty ego_pose timestamps")
            continue

        for fi in frame_indices:
            if fi < 0 or fi >= len(ts_list):
                print(f"[skip] frame_index {fi} out of range (len={len(ts_list)})")
                continue

            ts = int(ts_list[fi])
            fi2 = int(planner._timestamp_to_frame_index(ts))  # type: ignore[attr-defined]
            if fi2 != fi:
                ok_all = False
                print(f"[FAIL] timestamp->frame_index mismatch: expected {fi} got {fi2}")

            # Extract via planner
            inputs = planner._extract_inputs_ours(timestamp_us=ts)  # type: ignore[attr-defined]
            inputs_np = {k: _to_numpy(v) for k, v in inputs.items() if k != "diffusion_steps"}

            # Extract directly
            feats = extract_features(conn, map_api, scene_token_hex, int(fi), debug_log=False, routing_mode="auto")

            # Compare keys we care about
            keys = sorted(set(inputs_np.keys()) & set(feats.keys()))
            for k in keys:
                a = np.asarray(feats[k])
                b = np.asarray(inputs_np[k])[0]  # drop batch dim
                ok, msg = _compare(a, b, tol=float(args.tol))
                if not ok:
                    ok_all = False
                    print(f"[FAIL] fi={fi} key={k}: {msg}")
                else:
                    print(f"[ ok ] fi={fi} key={k}: {msg}")

    if ok_all:
        print("\nPASS: planner DB extractor path matches extract_features")
        return 0

    print("\nFAIL: mismatches detected")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
