from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "closedloop_fly_dump_analyze.py"


def _write_tick(
    dump_dir: Path,
    stem: str,
    *,
    token: str,
    tick: int,
    selected_xyh: np.ndarray | None,
    candidates_xyh: np.ndarray | None = None,
    selected_corner_drivable: np.ndarray | None = None,
    candidate_corner_offroad_counts: np.ndarray | None = None,
) -> None:
    meta = {
        "token": token,
        "tick": tick,
        "iteration": tick + 10,
        "timestamp_us": 1_700_000 + tick,
    }
    (dump_dir / f"{stem}.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

    arrays: dict[str, np.ndarray] = {
        "route_lanes": np.zeros((3, 5, 2), dtype=np.float32),
        "route_lanes_avails": np.array([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32),
    }
    if selected_xyh is not None:
        arrays["closed_loop_y"] = selected_xyh.astype(np.float32)
    if candidates_xyh is not None:
        arrays["candidate_trajectories"] = candidates_xyh.astype(np.float32)
    if selected_corner_drivable is not None:
        arrays["selected_corner_drivable"] = selected_corner_drivable.astype(bool)
    if candidate_corner_offroad_counts is not None:
        arrays["candidate_corner_offroad_counts"] = candidate_corner_offroad_counts.astype(np.int64)
    np.savez(dump_dir / f"{stem}.npz", **arrays)


def _run_analyzer(dump_dir: Path, out_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--dump-dir",
        str(dump_dir),
        "--out-dir",
        str(out_dir),
        *extra_args,
    ]
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def test_analyzer_outputs_csv_json_and_conservative_classifications(tmp_path: Path) -> None:
    dump_dir = tmp_path / "dumps"
    out_dir = tmp_path / "out"
    dump_dir.mkdir()
    _write_tick(
        dump_dir,
        "tick_000_iter_010",
        token="left_token",
        tick=0,
        selected_xyh=np.column_stack([np.arange(4), [0.0, 0.1, 0.2, 0.3], np.zeros(4)]),
        candidates_xyh=np.stack(
            [
                np.column_stack([np.arange(4), [0.0, 0.2, 0.5, 1.0], np.zeros(4)]),
                np.column_stack([np.arange(4), [0.0, 0.0, 0.0, 0.0], np.zeros(4)]),
            ]
        ),
        candidate_corner_offroad_counts=np.array([3, 0]),
    )
    _write_tick(
        dump_dir,
        "tick_001_iter_011",
        token="control_token",
        tick=1,
        selected_xyh=None,
    )
    tokens_csv = tmp_path / "tokens.csv"
    tokens_csv.write_text("token,group\nleft_token,left\ncontrol_token,control\n", encoding="utf-8")

    result = _run_analyzer(dump_dir, out_dir, "--tokens-csv", str(tokens_csv))

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader((out_dir / "closedloop_fly_summary.csv").open(encoding="utf-8")))
    assert [row["token"] for row in rows] == ["left_token", "control_token"]
    assert rows[0]["classification"] == "candidate_has_safer_option"
    assert rows[0]["selected_path_length"] == "4"
    assert rows[0]["route_lane_count"] == "2"
    assert rows[0]["has_candidates"] == "True"
    assert rows[0]["candidate_count"] == "2"
    assert rows[0]["selected_candidate_offroad_count"] == "3"
    assert rows[0]["best_candidate_offroad_count"] == "0"
    assert rows[1]["classification"] == "feature_or_dump_missing"

    summary = json.loads((out_dir / "closedloop_fly_summary.json").read_text(encoding="utf-8"))
    assert summary["total_ticks"] == 2
    assert summary["classification_counts"]["candidate_has_safer_option"] == 1
    assert summary["group_counts"]["left"]["candidate_has_safer_option"] == 1
    assert "candidate_has_safer_option" in (out_dir / "closedloop_fly_report.md").read_text(encoding="utf-8")


def test_analyzer_uses_corners_metric_and_future_drivable_fields(tmp_path: Path) -> None:
    dump_dir = tmp_path / "dumps"
    metrics_dir = tmp_path / "metrics"
    out_dir = tmp_path / "out"
    dump_dir.mkdir()
    metrics_dir.mkdir()
    selected = np.column_stack([np.arange(5, dtype=np.float32), np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.float32)])
    _write_tick(
        dump_dir,
        "tick_002_iter_012",
        token="metric_token",
        tick=2,
        selected_xyh=selected,
        selected_corner_drivable=np.ones((5, 4), dtype=bool),
    )
    pd.DataFrame(
        {
            "scenario_name": ["metric_token"],
            "time_series_values": [[True, True, False, False, True]],
        }
    ).to_parquet(metrics_dir / "corners_in_drivable_area.parquet")

    result = _run_analyzer(dump_dir, out_dir, "--metrics-dir", str(metrics_dir))

    assert result.returncode == 0, result.stderr
    row = next(csv.DictReader((out_dir / "closedloop_fly_summary.csv").open(encoding="utf-8")))
    assert row["metric_first_offroad_frame"] == "2"
    assert row["metric_last_offroad_frame"] == "3"
    assert row["metric_offroad_frames"] == "2"
    assert row["selected_corner_offroad_count"] == "0"
    assert row["classification"] == "corner_margin_suspect"


def test_analyzer_accepts_planner_closedloop_debug_keys(tmp_path: Path) -> None:
    dump_dir = tmp_path / "dumps"
    out_dir = tmp_path / "out"
    dump_dir.mkdir()
    meta = {
        "scenario_token": "planner_token",
        "tick": 3,
        "iteration_index": 13,
        "timestamp_us": 1_700_003,
        "selected_candidate_index": 1,
    }
    (dump_dir / "planner_token_tick000003_iter13_ts1700003.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    selected = np.column_stack([np.arange(4), np.zeros(4), np.zeros(4)]).astype(np.float32)
    candidates = np.stack([selected + 1.0, selected]).astype(np.float32)
    np.savez(
        dump_dir / "planner_token_tick000003_iter13_ts1700003.npz",
        selected_local_xyh=selected,
        candidates_local_xyh=candidates,
        candidate_corner_offroad_counts=np.array([2, 0], dtype=np.int64),
    )

    result = _run_analyzer(dump_dir, out_dir)

    assert result.returncode == 0, result.stderr
    row = next(csv.DictReader((out_dir / "closedloop_fly_summary.csv").open(encoding="utf-8")))
    assert row["token"] == "planner_token"
    assert row["selected_path_length"] == "4"
    assert row["has_candidates"] == "True"
    assert row["candidate_count"] == "2"
    assert row["selected_candidate_offroad_count"] == "0"
