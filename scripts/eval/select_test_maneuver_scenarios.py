#!/usr/bin/env python3
"""Select maneuver scenarios from nuPlan test cache DBs.

We scan `/media/.../nuplan_datasets/data/cache/test/*.db` and select N scenarios
for each maneuver group (straight/left/right) based on `scenario_tag`.

Outputs a CSV with all selected scenarios and some metadata needed for:
- open-loop eval (scene_token + ego_pose frame index)
- closed-loop sim (lidar_pc token)

Notes:
- Scenario tag tokens are lidar_pc tokens.
- Our single-frame extractor takes `scene_token_hex` + `frame_index` (ego_pose index).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


"""We do NOT assume test DBs have left/right scenario tags.

In the provided test split, scenario_tag contains tags like on_intersection,
traversing_intersection, etc, but may not include starting_left_turn/right_turn.

So we classify maneuvers using a GT-trajectory heuristic from ego_pose future:
- compute signed heading change over ~8s horizon
- left/right by sign
- straight if heading change small
"""


DEFAULT_CANDIDATE_TAGS = {
    # Intersection-ish tags to bias candidates toward turning/straight traversals.
    "on_intersection",
    "traversing_intersection",
    "starting_straight_stop_sign_intersection_traversal",
    "starting_straight_traffic_light_intersection_traversal",
}


@dataclass(frozen=True)
class Row:
    group: str
    scenario_type: str
    lidar_pc_token: str
    scene_token: str
    ego_pose_token: str
    frame_index: int
    timestamp_us: int
    db_path: str
    log_name: str
    location: str
    map_version: str
    yaw_change_rad: float
    travel_m: float


def _hexify(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        return x.hex()
    return str(x)


def _iter_rows(db_path: Path, *, types: set[str]) -> Iterable[Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Join scenario_tag -> lidar_pc -> lidar -> log.
    q = f"""
    SELECT
      st.type AS scenario_type,
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
    WHERE st.type IN ({','.join(['?']*len(types))})
    """

    for r in cur.execute(q, tuple(sorted(types))):
        scenario_type = str(r["scenario_type"])
        lidar_pc_token = _hexify(r["lidar_pc_token"])
        scene_token = _hexify(r["scene_token"])
        ego_pose_token = _hexify(r["ego_pose_token"])
        ts = int(r["lidar_pc_timestamp"])
        log_name = str(r["log_name"])
        location = str(r["location"])
        map_version = str(r["map_version"])

        # Compute ego_pose frame index (0-based) within that log by timestamp rank.
        # This is compatible with our extractor which orders ego_pose by timestamp.
        cur2 = conn.cursor()
        cur2.execute("SELECT timestamp, log_token FROM ego_pose WHERE token = ? LIMIT 1", (bytes.fromhex(ego_pose_token),))
        ep = cur2.fetchone()
        if ep is None:
            continue
        ep_ts = int(ep[0])
        log_token = ep[1]
        cur2.execute(
            "SELECT COUNT(*) - 1 FROM ego_pose WHERE log_token = ? AND timestamp <= ?",
            (log_token, ep_ts),
        )
        frame_index = int(cur2.fetchone()[0])

        # Fetch an 8s future window from ego_pose (20Hz in DB). We sample every 2 rows -> 10Hz.
        cur3 = conn.cursor()
        cur3.execute(
            "SELECT x, y, qw, qx, qy, qz FROM ego_pose WHERE log_token = ? ORDER BY timestamp LIMIT ? OFFSET ?",
            (log_token, 161, frame_index),
        )
        poses = cur3.fetchall()
        if poses is None or len(poses) < 41:
            # too short
            continue

        def _yaw(qw, qx, qy, qz):
            import math

            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            return math.atan2(siny_cosp, cosy_cosp)

        # downsample to 10Hz
        xs, ys, yaws = [], [], []
        for j in range(0, min(len(poses), 161), 2):
            pr = poses[j]
            x = float(pr[0])
            y = float(pr[1])
            qw, qx, qy, qz = map(float, pr[2:6])
            xs.append(x)
            ys.append(y)
            yaws.append(_yaw(qw, qx, qy, qz))

        import math

        travel = 0.0
        for j in range(1, len(xs)):
            dx = xs[j] - xs[j - 1]
            dy = ys[j] - ys[j - 1]
            travel += math.hypot(dx, dy)

        # unwrap heading change
        yaw0 = yaws[0]
        yaw_last = yaws[-1]
        d = yaw_last - yaw0
        d = (d + math.pi) % (2 * math.pi) - math.pi

        # thresholds
        if travel < 5.0:
            continue
        if abs(d) >= (15.0 * math.pi / 180.0):
            group = "left" if d > 0 else "right"
        elif abs(d) <= (5.0 * math.pi / 180.0):
            group = "straight"
        else:
            # ambiguous, skip
            continue

        yield Row(
            group=group,
            scenario_type=f"gt_{group}",
            lidar_pc_token=lidar_pc_token,
            scene_token=scene_token,
            ego_pose_token=ego_pose_token,
            frame_index=frame_index,
            timestamp_us=ts,
            db_path=str(db_path),
            log_name=log_name,
            location=location,
            map_version=map_version,
            yaw_change_rad=float(d),
            travel_m=float(travel),
        )

    conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test-db-dir",
        type=str,
        default="/media/zhouyida/新加卷1/nuplan_datasets/data/cache/test",
        help="Directory containing nuPlan test split .db files.",
    )
    ap.add_argument("--n-per-group", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        type=str,
        default="outputs/eval/test_maneuver20",
        help="Output directory (relative to repo root).",
    )
    args = ap.parse_args()

    test_dir = Path(args.test_db_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dbs = sorted(test_dir.glob("*.db"))
    if not dbs:
        raise FileNotFoundError(f"No .db under {test_dir}")

    types = set(DEFAULT_CANDIDATE_TAGS)

    # Build a shuffled DB order so we don't bias toward early files.
    rng = random.Random(int(args.seed))
    dbs = list(dbs)
    rng.shuffle(dbs)

    target = int(args.n_per_group)
    picked_by_group: dict[str, list[Row]] = {"left": [], "right": [], "straight": []}

    for db in dbs:
        # Stop early if done.
        if all(len(picked_by_group[g]) >= target for g in picked_by_group):
            break

        try:
            for r in _iter_rows(db, types=types):
                if len(picked_by_group[r.group]) < target:
                    picked_by_group[r.group].append(r)
                if all(len(picked_by_group[g]) >= target for g in picked_by_group):
                    break
        except Exception:
            continue

    picked: list[Row] = []
    for g in ("straight", "left", "right"):
        if len(picked_by_group[g]) < target:
            print(f"[warn] group={g} requested={target} found={len(picked_by_group[g])}")
        picked.extend(picked_by_group[g])

    # Write a rich CSV.
    rich_csv = out_dir / "test_maneuver20_scenarios.csv"
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
        for r in picked:
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

    # Write tokens-only CSV for closed-loop sim (first column token).
    tokens_csv = out_dir / "test_maneuver20_tokens.csv"
    with tokens_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token"])  # scripts/sim runner skips header
        for r in picked:
            w.writerow([r.lidar_pc_token])

    # Also split per group for convenience.
    for g in ("straight", "left", "right"):
        p = out_dir / f"test_maneuver20_{g}_tokens.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["token"])
            for r in picked:
                if r.group == g:
                    w.writerow([r.lidar_pc_token])

    print(f"[ok] wrote {rich_csv}")
    print(f"[ok] wrote {tokens_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
