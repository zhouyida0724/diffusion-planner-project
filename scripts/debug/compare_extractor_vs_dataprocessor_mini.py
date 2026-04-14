#!/usr/bin/env python3
"""Compare feature semantics between:

A) src.platform.nuplan.features.extract_single_frame.extract_features  (training/export-style)
B) nuplan-visualization nuplan.diffusion_planner.data_process.DataProcessor.observation_adapter (paper planner inference)

This is a lightweight harness to validate train↔infer *feature semantics* on the mini DBs.

It intentionally tests only a handful of scenes (single-digit) and prints diffs.

Usage:
  PYTHONPATH=.:nuplan-visualization \
  python3 scripts/debug/compare_extractor_vs_dataprocessor_mini.py \
    --db data/nuplan/data/cache/mini/2021.05.25.14.16.10_veh-35_01690_02183.db \
    --scenes 3

Exit codes:
  0: no diffs found (unexpected)
  2: diffs found
  3: missing resources
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScenePick:
    scene_token_hex: str
    map_name: str
    frame_index_for_iter0: int
    initial_lidar_timestamp: int


def _build_map_api(map_root: Path, map_version: str, map_name: str):
    from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db

    maps_db = get_maps_db(str(map_root), str(map_version))
    factory = NuPlanMapFactory(maps_db)
    return factory.build_map_from_name(str(map_name))


def _pick_scenes(conn: sqlite3.Connection, *, k: int) -> list[tuple[bytes, bytes]]:
    cur = conn.cursor()
    rows = cur.execute("select token, log_token from scene order by rowid limit ?", (int(k),)).fetchall()
    return [(r[0], r[1]) for r in rows]


def _get_map_name(conn: sqlite3.Connection) -> str:
    # In our mini DBs, log.map_version stores the nuPlan map name (e.g. us-nv-las-vegas-strip).
    cur = conn.cursor()
    row = cur.execute("select map_version from log limit 1").fetchone()
    if not row or not row[0]:
        raise RuntimeError("failed to read map_version from log")
    return str(row[0])


def _get_initial_lidar_timestamp(conn: sqlite3.Connection, scene_token: bytes) -> int:
    cur = conn.cursor()
    row = cur.execute(
        "select timestamp from lidar_pc where scene_token=? order by timestamp asc limit 1", (scene_token,)
    ).fetchone()
    if not row:
        raise RuntimeError("no lidar_pc rows for scene")
    return int(row[0])


def _get_frame_index_for_timestamp(conn: sqlite3.Connection, log_token: bytes, timestamp: int) -> int:
    cur = conn.cursor()
    rows = cur.execute(
        "select timestamp from ego_pose where log_token=? order by timestamp", (log_token,)
    ).fetchall()
    ts = [int(r[0]) for r in rows]
    # Exact match preferred.
    try:
        return int(ts.index(int(timestamp)))
    except ValueError:
        # Nearest fallback (shouldn't happen often)
        arr = np.asarray(ts, dtype=np.int64)
        j = int(np.argmin(np.abs(arr - int(timestamp))))
        return j


def _to_numpy(x: Any) -> np.ndarray:
    import torch

    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (float, int, bool)):
        return np.asarray(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape:
        return {"shape_mismatch": 1.0}
    da = a.astype(np.float64)
    db = b.astype(np.float64)
    d = np.abs(da - db)
    return {"max": float(d.max() if d.size else 0.0), "mean": float(d.mean() if d.size else 0.0)}


def _squeeze_to_match(x: np.ndarray, target_ndim: int) -> np.ndarray:
    """Best-effort squeeze of leading singleton dims to match target ndim."""
    y = x
    while y.ndim > target_ndim and y.shape[0] == 1:
        y = np.squeeze(y, axis=0)
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, required=True)
    ap.add_argument("--map-root", type=str, default=None)
    ap.add_argument("--map-version", type=str, default="9.12.1817")
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        print(f"[error] db not found: {db_path}")
        return 3

    repo_root = Path(__file__).resolve().parents[2]
    # Ensure repo-local imports work when running from anywhere.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    nuplan_vis = repo_root / "nuplan-visualization"
    if nuplan_vis.is_dir() and str(nuplan_vis) not in sys.path:
        sys.path.insert(0, str(nuplan_vis))
    map_root = Path(args.map_root).expanduser().resolve() if args.map_root else (repo_root / "data" / "nuplan" / "maps")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        map_name = _get_map_name(conn)
        map_api = _build_map_api(map_root, args.map_version, map_name)

        picks: list[ScenePick] = []
        for scene_token, log_token in _pick_scenes(conn, k=int(args.scenes)):
            ts0 = _get_initial_lidar_timestamp(conn, scene_token)
            fi0 = _get_frame_index_for_timestamp(conn, log_token, ts0)
            picks.append(
                ScenePick(
                    scene_token_hex=scene_token.hex(),
                    map_name=map_name,
                    frame_index_for_iter0=int(fi0),
                    initial_lidar_timestamp=int(ts0),
                )
            )

        # Build inference DataProcessor with canonical dims.
        from nuplan.diffusion_planner.data_process.data_processor import DataProcessor

        cfg = SimpleNamespace(
            agent_num=32,
            static_objects_num=5,
            lane_num=70,
            lane_len=20,
            route_num=25,
            route_len=20,
        )
        dp = DataProcessor(cfg)

        from src.platform.nuplan.features.extract_single_frame import build_nuplan_scenario_from_db, extract_features

        any_diff = False

        for i, pick in enumerate(picks):
            print("\n" + "=" * 80)
            print(f"[scene {i}] token={pick.scene_token_hex} map={pick.map_name} frame_index_for_iter0={pick.frame_index_for_iter0}")

            try:
                # Scenario at iteration 0
                scenario = build_nuplan_scenario_from_db(conn, str(db_path), pick.scene_token_hex, pick.map_name)
                ego_state = scenario.get_ego_state_at_iteration(0)

                present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
                past_tracked_objects = [
                    to.tracked_objects
                    for to in scenario.get_past_tracked_objects(iteration=0, time_horizon=2.0, num_samples=20)
                ]
                observation_buffer = past_tracked_objects + [present_tracked_objects]
                traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
                route_roadblock_ids = list(scenario.get_route_roadblock_ids())

                history = SimpleNamespace(current_state=[ego_state], observation_buffer=observation_buffer)

                infer = dp.observation_adapter(history, traffic_light_data, map_api, route_roadblock_ids, device="cpu")
                infer_np = {k: _to_numpy(v) for k, v in dict(infer).items()}

                train = extract_features(
                    conn,
                    map_api,
                    pick.scene_token_hex,
                    int(pick.frame_index_for_iter0),
                    debug_log=False,
                )
                train_np = {k: _to_numpy(v) for k, v in dict(train).items()}
            except Exception as e:
                any_diff = True
                print(f"[scene {i}] SKIP due to error: {type(e).__name__}: {e}")
                continue

            inter = sorted(set(infer_np.keys()) & set(train_np.keys()))
            print(f"[keys] infer={len(infer_np)} train={len(train_np)} intersection={len(inter)}")

            # Compare a focused set first
            focus = [
                "ego_current_state",
                "neighbor_agents_past",
                "static_objects",
                "lanes",
                "lanes_avails",
                "route_lanes",
                "route_lanes_avails",
            ]
            for k in focus:
                if k not in infer_np or k not in train_np:
                    continue
                a = train_np[k]
                b0 = infer_np[k]
                b = _squeeze_to_match(b0, a.ndim)
                if a.shape != b.shape:
                    any_diff = True
                    print(f"[DIFF] {k}: shape train={a.shape} infer_raw={b0.shape} infer_squeezed={b.shape}")
                    continue
                dd = _diff(a, b)
                if dd.get("shape_mismatch", 0.0) or dd["max"] > float(args.tol):
                    any_diff = True
                    print(f"[DIFF] {k}: max={dd['max']:.6g} mean={dd['mean']:.6g}")
                else:
                    print(f"[ ok ] {k}: max={dd['max']:.6g} mean={dd['mean']:.6g}")

            # Special: inspect neighbor_agents_past last dims histogram (to catch type/size fields)
            if "neighbor_agents_past" in inter:
                na = train_np["neighbor_agents_past"]
                nb0 = infer_np["neighbor_agents_past"]
                nb = _squeeze_to_match(nb0, na.ndim)
                if na.shape == nb.shape and na.ndim == 3 and na.shape[-1] >= 10:
                    ta = na[..., 8:].reshape(-1)
                    tb = nb[..., 8:].reshape(-1)
                    # print a small signature of unique values (rounded)
                    ua = np.unique(np.round(ta.astype(np.float64), 3))[:10]
                    ub = np.unique(np.round(tb.astype(np.float64), 3))[:10]
                    print(f"[neighbor_agents_past[...,8:]] train uniq(head)={ua} infer uniq(head)={ub}")
                else:
                    print(
                        f"[neighbor_agents_past] shape train={na.shape} infer_raw={nb0.shape} infer_squeezed={nb.shape}"
                    )

        if any_diff:
            print("\n[result] DIFF FOUND (expected if train↔infer semantics mismatch)")
            return 2

        print("\n[result] NO DIFF FOUND")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
