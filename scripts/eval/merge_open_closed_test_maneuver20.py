#!/usr/bin/env python3
"""Merge open-loop jsonl + closed-loop parquet metrics for test maneuver set.

Inputs:
- outputs/eval/test_maneuver20/open_loop_metrics_sampler10.jsonl
- closed-loop runs under data/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/<timestamp>/

Outputs:
- outputs/eval/test_maneuver20/open_closed_per_scenario.csv
- outputs/eval/test_maneuver20/open_closed_group_summary.csv

Notes:
- closed-loop aggregator parquet contains extra rows like per-scenario_type aggregations and final_score.
  We filter to rows whose `scenario` looks like a 16-hex token and is present in open-loop.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

TOKEN_RE = re.compile(r"^[0-9a-f]{16}$")


def load_open_loop(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "lidar_pc_token" not in df.columns:
        raise ValueError("open-loop jsonl missing lidar_pc_token")
    return df


def load_closed_loop_agg(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "scenario" not in df.columns:
        raise ValueError(f"closed-loop parquet missing scenario: {parquet_path}")
    df = df.copy()
    df["scenario"] = df["scenario"].astype(str)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--open-loop-jsonl",
        type=str,
        default="outputs/eval/test_maneuver20/open_loop_metrics_sampler10.jsonl",
    )
    ap.add_argument(
        "--closed-loop-agg",
        type=str,
        action="append",
        required=True,
        help="Path to closed-loop aggregator parquet (repeatable).",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="outputs/eval/test_maneuver20/open_closed_per_scenario.csv",
    )
    ap.add_argument(
        "--out-summary-csv",
        type=str,
        default="outputs/eval/test_maneuver20/open_closed_group_summary.csv",
    )
    args = ap.parse_args()

    open_path = Path(args.open_loop_jsonl)
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_open = load_open_loop(open_path)
    tokens = set(df_open["lidar_pc_token"].astype(str).tolist())

    dfs_closed = []
    for p in args.closed_loop_agg:
        dfs_closed.append(load_closed_loop_agg(Path(p)))
    df_closed = pd.concat(dfs_closed, axis=0, ignore_index=True)

    # keep only per-scenario rows
    df_closed = df_closed[df_closed["scenario"].apply(lambda s: bool(TOKEN_RE.match(s)))].copy()
    df_closed = df_closed[df_closed["scenario"].isin(tokens)].copy()

    # If duplicates exist across runs, keep the last occurrence.
    df_closed = df_closed.drop_duplicates(subset=["scenario"], keep="last")

    # rename and select closed-loop columns we care about
    rename = {
        "scenario": "lidar_pc_token",
        "score": "closed_score",
        "drivable_area_compliance": "closed_drivable_area_compliance",
        "driving_direction_compliance": "closed_driving_direction_compliance",
        "ego_is_comfortable": "closed_ego_is_comfortable",
        "ego_is_making_progress": "closed_ego_is_making_progress",
        "ego_progress_along_expert_route": "closed_ego_progress_along_expert_route",
        "no_ego_at_fault_collisions": "closed_no_ego_at_fault_collisions",
        "speed_limit_compliance": "closed_speed_limit_compliance",
        "time_to_collision_within_bound": "closed_time_to_collision_within_bound",
    }
    keep_cols = [c for c in rename.keys() if c in df_closed.columns]
    df_closed = df_closed[keep_cols].rename(columns=rename)

    df = df_open.merge(df_closed, on="lidar_pc_token", how="left")

    # Write per-scenario
    df.to_csv(out_csv, index=False)

    # Group summary
    metric_cols = [
        "ade_1s",
        "ade_3s",
        "ade_5s",
        "ade_8s",
        "fde_1s",
        "fde_3s",
        "fde_5s",
        "fde_8s",
        "closed_score",
        "closed_drivable_area_compliance",
        "closed_driving_direction_compliance",
        "closed_ego_is_comfortable",
        "closed_ego_is_making_progress",
        "closed_ego_progress_along_expert_route",
        "closed_no_ego_at_fault_collisions",
        "closed_speed_limit_compliance",
        "closed_time_to_collision_within_bound",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    g = df.groupby("group")[metric_cols].agg(["mean", "std", "count"]).reset_index()
    # flatten columns
    g.columns = [
        ("group" if a == "group" else f"{a}_{b}")
        for a, b in ([("group", "")] + [(a, b) for a, b in g.columns.tolist()[1:]])
    ]
    g.to_csv(out_summary, index=False)

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_summary}")
    # quick counts
    have_closed = df["closed_score"].notna().sum() if "closed_score" in df.columns else 0
    print(f"[info] open_rows={len(df_open)} merged_rows={len(df)} have_closed={have_closed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
