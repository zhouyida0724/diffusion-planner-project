#!/usr/bin/env python3
"""Select a diverse test_maneuver20 set from nuPlan test DBs.

The original selector filled each maneuver group from the first matching dense tag
sequence, which can produce 20 near-duplicate frames from one log/scene. This
script collects all GT-classified candidates first, then greedily selects with:
- unique lidar_pc token
- unique scene_token
- prefer at most one sample per log_name per maneuver group
- if needed, relax to multiple per log only with a large frame gap

Outputs are suffixed with `_diverse` and do not overwrite the original list.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# Reuse the verified Row schema / tag set from the original script, but use a fast local scanner.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_test_maneuver_scenarios import DEFAULT_CANDIDATE_TAGS, Row, _hexify  # noqa: E402


def _yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _iter_rows_fast(db_path: Path, *, types: set[str]) -> Iterable[Row]:
    """Fast version of the original selector's GT maneuver heuristic.

    The original `_iter_rows` issues several SQL queries per scenario_tag row.
    That is fine for quick one-off selection but too slow when collecting all
    candidates for diversity. Here we preload ego_pose once per DB and classify
    all candidate lidar_pc rows in memory.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT token, timestamp, x, y, qw, qx, qy, qz, log_token FROM ego_pose ORDER BY timestamp")
    poses = cur.fetchall()
    if len(poses) < 161:
        conn.close()
        return

    ep_token_to_idx = {_hexify(r["token"]): i for i, r in enumerate(poses)}
    xs = [float(r["x"]) for r in poses]
    ys = [float(r["y"]) for r in poses]
    yaws = [_yaw(float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])) for r in poses]

    # Match nuPlan's get_scenarios_from_db(): only scenes with at least 2 scenes
    # before and 2 after are buildable. Without this, a token can be in the DB/tag
    # table but silently disappear during ScenarioBuilder construction.
    cur.execute("SELECT token FROM scene ORDER BY name ASC")
    scenes = [_hexify(r["token"]) for r in cur.fetchall()]
    valid_scenes = set(scenes[2:-2]) if len(scenes) >= 5 else set()
    if not valid_scenes:
        conn.close()
        return

    q = f"""
    SELECT DISTINCT
      lp.token AS lidar_pc_token,
      lp.scene_token AS scene_token,
      lp.ego_pose_token AS ego_pose_token,
      lp.timestamp AS lidar_pc_timestamp,
      l.logfile AS log_name,
      l.location AS location,
      l.map_version AS map_version
    FROM lidar_pc lp
    INNER JOIN scenario_tag st ON lp.token = st.lidar_pc_token
    INNER JOIN lidar ld ON ld.token = lp.lidar_token
    INNER JOIN log l ON l.token = ld.log_token
    WHERE st.type IN ({','.join(['?'] * len(types))})
    """

    for r in cur.execute(q, tuple(sorted(types))):
        lidar_pc_token = _hexify(r["lidar_pc_token"])
        scene_token = _hexify(r["scene_token"])
        if scene_token not in valid_scenes:
            continue
        ego_pose_token = _hexify(r["ego_pose_token"])
        idx = ep_token_to_idx.get(ego_pose_token)
        if idx is None or idx + 160 >= len(poses):
            continue

        sample_idxs = list(range(idx, idx + 161, 2))
        travel = 0.0
        for a, b in zip(sample_idxs[:-1], sample_idxs[1:]):
            travel += math.hypot(xs[b] - xs[a], ys[b] - ys[a])
        if travel < 5.0:
            continue

        d = yaws[sample_idxs[-1]] - yaws[sample_idxs[0]]
        d = (d + math.pi) % (2 * math.pi) - math.pi
        if abs(d) >= (15.0 * math.pi / 180.0):
            group = "left" if d > 0 else "right"
        elif abs(d) <= (5.0 * math.pi / 180.0):
            group = "straight"
        else:
            continue

        yield Row(
            group=group,
            scenario_type=f"gt_{group}",
            lidar_pc_token=lidar_pc_token,
            scene_token=scene_token,
            ego_pose_token=ego_pose_token,
            frame_index=idx,
            timestamp_us=int(r["lidar_pc_timestamp"]),
            db_path=str(db_path),
            log_name=str(r["log_name"]),
            location=str(r["location"]),
            map_version=str(r["map_version"]),
            yaw_change_rad=float(d),
            travel_m=float(travel),
        )

    conn.close()


def _write_rows(rows: list[Row], out_dir: Path, suffix: str) -> tuple[Path, Path]:
    rich_csv = out_dir / f"test_maneuver20_scenarios_{suffix}.csv"
    with rich_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "group",
                "scenario_type",
                "lidar_pc_token",
                "scene_token",
                "ego_pose_token",
                "frame_index",
                "timestamp_us",
                "db_path",
                "log_name",
                "location",
                "map_version",
                "yaw_change_rad",
                "travel_m",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.group,
                    r.scenario_type,
                    r.lidar_pc_token,
                    r.scene_token,
                    r.ego_pose_token,
                    r.frame_index,
                    r.timestamp_us,
                    r.db_path,
                    r.log_name,
                    r.location,
                    r.map_version,
                    f"{r.yaw_change_rad:.6f}",
                    f"{r.travel_m:.3f}",
                ]
            )

    tokens_csv = out_dir / f"test_maneuver20_tokens_{suffix}.csv"
    with tokens_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token"])
        for r in rows:
            w.writerow([r.lidar_pc_token])

    for g in ("straight", "left", "right"):
        p = out_dir / f"test_maneuver20_{g}_tokens_{suffix}.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["token"])
            for r in rows:
                if r.group == g:
                    w.writerow([r.lidar_pc_token])

    return rich_csv, tokens_csv


def _can_pick_with_gap(candidate: Row, picked: list[Row], *, min_frame_gap: int) -> bool:
    for r in picked:
        if r.log_name == candidate.log_name and abs(r.frame_index - candidate.frame_index) < min_frame_gap:
            return False
    return True


def _round_robin_by_location(candidates: list[Row], rng: random.Random) -> list[Row]:
    by_loc: dict[str, list[Row]] = defaultdict(list)
    for r in candidates:
        by_loc[r.location].append(r)
    for rows in by_loc.values():
        rng.shuffle(rows)
    locs = list(by_loc)
    rng.shuffle(locs)
    ordered: list[Row] = []
    while any(by_loc.values()):
        for loc in list(locs):
            if by_loc[loc]:
                ordered.append(by_loc[loc].pop())
    return ordered


def _select_group(candidates: list[Row], *, target: int, rng: random.Random, min_frame_gap: int) -> list[Row]:
    # Dedupe exact lidar_pc tokens first; a token can have multiple candidate tags.
    by_token: dict[str, Row] = {}
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    for r in shuffled:
        by_token.setdefault(r.lidar_pc_token, r)

    ordered = _round_robin_by_location(list(by_token.values()), rng)
    picked: list[Row] = []
    used_scene: set[str] = set()
    used_log: set[str] = set()

    # Tier 1: strongest diversity, one scene and one log per maneuver.
    for r in ordered:
        if len(picked) >= target:
            break
        if r.scene_token in used_scene or r.log_name in used_log:
            continue
        picked.append(r)
        used_scene.add(r.scene_token)
        used_log.add(r.log_name)

    # Tier 2: allow repeated log, but still unique scene and large frame gap.
    for r in ordered:
        if len(picked) >= target:
            break
        if r.lidar_pc_token in {p.lidar_pc_token for p in picked} or r.scene_token in used_scene:
            continue
        if not _can_pick_with_gap(r, picked, min_frame_gap=min_frame_gap):
            continue
        picked.append(r)
        used_scene.add(r.scene_token)

    # Tier 3: unique scene only, if the test split is too small for the target.
    for r in ordered:
        if len(picked) >= target:
            break
        if r.lidar_pc_token in {p.lidar_pc_token for p in picked} or r.scene_token in used_scene:
            continue
        picked.append(r)
        used_scene.add(r.scene_token)

    # Tier 4: final fallback, unique token only. Should normally not trigger.
    for r in ordered:
        if len(picked) >= target:
            break
        if r.lidar_pc_token in {p.lidar_pc_token for p in picked}:
            continue
        picked.append(r)

    return picked[:target]


def _write_scenario_filter_yaml(tokens_csv: Path, yaml_path: Path) -> None:
    import pandas as pd

    tokens = pd.read_csv(tokens_csv)["token"].tolist()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w") as f:
        f.write("_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter\n")
        f.write("scenario_types: null\n")
        f.write("scenario_tokens:\n")
        for t in tokens:
            f.write(f"  - '{t}'\n")
        f.write("log_names: null\n")
        f.write("map_names: null\n")
        f.write("num_scenarios_per_type: null\n")
        f.write("limit_total_scenarios: null\n")
        f.write("timestamp_threshold_s: null\n")
        f.write("ego_displacement_minimum_m: null\n")
        f.write("expand_scenarios: false\n")
        f.write("remove_invalid_goals: false\n")
        f.write("shuffle: false\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-db-dir", default="/media/zhouyida/新加卷1/nuplan_datasets/data/cache/test")
    ap.add_argument("--out-dir", default="outputs/eval/test_maneuver20")
    ap.add_argument("--n-per-group", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-frame-gap-same-log", type=int, default=2000)
    ap.add_argument("--suffix", default="diverse")
    ap.add_argument(
        "--scenario-filter-yaml",
        default="nuplan-visualization/nuplan/planning/script/config/common/scenario_filter/one_hand_diverse60.yaml",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    test_dir = Path(args.test_db_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dbs = sorted(test_dir.glob("*.db"))
    if not dbs:
        raise FileNotFoundError(f"No .db under {test_dir}")

    types = set(DEFAULT_CANDIDATE_TAGS)
    all_rows: dict[str, list[Row]] = {"left": [], "right": [], "straight": []}
    seen_tokens: set[str] = set()
    for i, db in enumerate(dbs, 1):
        try:
            for r in _iter_rows_fast(db, types=types):
                if r.lidar_pc_token in seen_tokens:
                    continue
                seen_tokens.add(r.lidar_pc_token)
                all_rows[r.group].append(r)
        except Exception as e:
            print(f"[warn] skip db={db}: {e}", flush=True)
        if i % 20 == 0:
            print(f"[scan] {i}/{len(dbs)} dbs | counts=" + ", ".join(f"{g}:{len(v)}" for g, v in all_rows.items()), flush=True)

    picked_by_group = {
        g: _select_group(rows, target=args.n_per_group, rng=rng, min_frame_gap=args.min_frame_gap_same_log)
        for g, rows in all_rows.items()
    }

    for g, rows in picked_by_group.items():
        print(f"\n[group={g}] candidates={len(all_rows[g])} picked={len(rows)}")
        print("  unique logs", len({r.log_name for r in rows}), "unique scenes", len({r.scene_token for r in rows}), "locations", dict(Counter(r.location for r in rows)))
        print("  top logs", Counter(r.log_name for r in rows).most_common(5))
        if len(rows) < args.n_per_group:
            print(f"  [warn] requested {args.n_per_group}, only picked {len(rows)}")

    picked: list[Row] = []
    for g in ("straight", "left", "right"):
        picked.extend(picked_by_group[g])

    rich_csv, tokens_csv = _write_rows(picked, out_dir, args.suffix)
    _write_scenario_filter_yaml(tokens_csv, Path(args.scenario_filter_yaml))
    print(f"\n[ok] wrote {rich_csv}")
    print(f"[ok] wrote {tokens_csv}")
    print(f"[ok] wrote {args.scenario_filter_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
