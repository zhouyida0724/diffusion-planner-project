#!/usr/bin/env python3
"""Compare closed-loop planner conditioning features against offline extractor output.

Usage examples:

  # Compare an existing closed-loop dump against offline extraction.
  PYTHONPATH=/workspace/nuplan-visualization:/workspace \
    python3 scripts/sim/compare_closedloop_features.py \
      --simlog /workspace/outputs/.../d32c40cedb4d5025.msgpack.xz \
      --tick 0 \
      --dump-dir /workspace/outputs/debug_d32c40c_dump_fixed \
      --output-dir /workspace/outputs/feature_compare_d32c40c_tick0

  # Or, if no planner dump is available yet, compare the planner runtime adapter
  # (same online path used by DiffusionPlannerCkpt after the fix) directly.
  PYTHONPATH=/workspace/nuplan-visualization:/workspace \
    python3 scripts/sim/compare_closedloop_features.py \
      --simlog /workspace/outputs/.../d32c40cedb4d5025.msgpack.xz \
      --tick 0 \
      --output-dir /workspace/outputs/feature_compare_d32c40c_tick0
"""

from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sqlite3
from pathlib import Path

import msgpack
import numpy as np

from src.platform.nuplan.features.extract_single_frame import extract_features


CORE_COMPARE_KEYS = [
    "ego_current_state",
    "ego_past",
    "ego_agent_future",
    "neighbor_agents_past",
    "neighbor_agents_avails",
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


def _load_simlog(path: Path):
    raw = lzma.open(path, "rb").read()
    blob = msgpack.unpackb(raw, raw=False)
    return pickle.loads(blob)


def _neighbor_avails(arr: np.ndarray) -> np.ndarray:
    # Match the planner dump logic: any non-trivial feature on a timestep means available.
    return np.any(np.abs(arr) > 1e-8, axis=-1).astype(np.uint8)


def _build_runtime_features_from_simlog(simlog, tick: int) -> dict[str, np.ndarray]:
    scenario = simlog.scenario
    db_path = Path(getattr(scenario, "_log_file"))
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        feats = extract_features(
            conn,
            scenario.map_api,
            str(scenario.token),
            int(tick),
            debug_log=False,
            routing_mode="auto",
            route_roadblock_ids_override=list(scenario.get_route_roadblock_ids()),
        )
    finally:
        conn.close()

    feats = dict(feats)
    feats["neighbor_agents_avails"] = _neighbor_avails(feats["neighbor_agents_past"])
    return feats


def _load_online_dump(dump_dir: Path, tick: int) -> dict[str, np.ndarray]:
    path = dump_dir / f"tick_{tick:03d}_iter_{tick:03d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Online dump not found: {path}")
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    serializable = {}
    for k, v in arrays.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v
    np.savez_compressed(path, **serializable)


def _compare_arrays(name: str, online: np.ndarray, offline: np.ndarray, *, atol: float, rtol: float) -> dict:
    same_shape = tuple(online.shape) == tuple(offline.shape)
    result = {
        "key": name,
        "online_shape": tuple(int(x) for x in online.shape),
        "offline_shape": tuple(int(x) for x in offline.shape),
        "same_shape": bool(same_shape),
    }
    if not same_shape:
        result["ok"] = False
        result["reason"] = "shape_mismatch"
        return result

    if np.issubdtype(online.dtype, np.integer) or np.issubdtype(offline.dtype, np.integer):
        equal = np.array_equal(online, offline)
        result.update(
            {
                "ok": bool(equal),
                "equal": bool(equal),
                "online_nonzero": int(np.count_nonzero(online)),
                "offline_nonzero": int(np.count_nonzero(offline)),
            }
        )
        return result

    diff = np.abs(online.astype(np.float64) - offline.astype(np.float64))
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    ok = bool(np.allclose(online, offline, atol=atol, rtol=rtol))
    result.update(
        {
            "ok": ok,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "online_nonzero": int(np.count_nonzero(online)),
            "offline_nonzero": int(np.count_nonzero(offline)),
        }
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simlog", type=str, required=True)
    ap.add_argument("--tick", type=int, required=True)
    ap.add_argument("--dump-dir", type=str, default=None, help="Existing planner dump dir (optional).")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-5)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simlog = _load_simlog(Path(args.simlog))
    offline = _build_runtime_features_from_simlog(simlog, tick=int(args.tick))

    if args.dump_dir:
        online = _load_online_dump(Path(args.dump_dir), int(args.tick))
        online_source = "planner_dump"
    else:
        online = _build_runtime_features_from_simlog(simlog, tick=int(args.tick))
        online_source = "runtime_adapter"

    if "neighbor_agents_avails" not in online and "neighbor_agents_past" in online:
        online["neighbor_agents_avails"] = _neighbor_avails(online["neighbor_agents_past"])

    _save_npz(output_dir / "offline_features.npz", offline)
    _save_npz(output_dir / "online_features.npz", online)

    per_key = []
    failures = []
    for key in CORE_COMPARE_KEYS:
        if key not in online:
            failures.append({"key": key, "ok": False, "reason": "missing_in_online"})
            continue
        if key not in offline:
            failures.append({"key": key, "ok": False, "reason": "missing_in_offline"})
            continue
        res = _compare_arrays(key, online[key], offline[key], atol=float(args.atol), rtol=float(args.rtol))
        per_key.append(res)
        if not res.get("ok", False):
            failures.append(res)

    summary = {
        "simlog": str(args.simlog),
        "scenario_token": str(simlog.scenario.token),
        "tick": int(args.tick),
        "online_source": online_source,
        "failures": failures,
        "results": per_key,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if failures:
        print(json.dumps(summary, indent=2))
        raise SystemExit(1)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
