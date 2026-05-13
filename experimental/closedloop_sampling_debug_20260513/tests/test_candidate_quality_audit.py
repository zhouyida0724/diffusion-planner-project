from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "dp_candidate_quality_audit.py"


def _traj(final_y: float, *, heading: float = 0.0, curve: float = 0.0, n: int = 6) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = 10.0 * t
    y = final_y * t + curve * t * (1.0 - t)
    h = heading * t
    return np.column_stack([x, y, h]).astype(np.float32)


def _run_audit(dump_dir: Path, out_csv: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dump-dir",
            str(dump_dir),
            "--out-csv",
            str(out_csv),
            *extra_args,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _read_one_row(path: Path) -> dict[str, str]:
    return next(csv.DictReader(path.open(encoding="utf-8")))


def test_collapsed_candidates_report_one_effective_mode(tmp_path: Path) -> None:
    dump_dir = tmp_path / "dumps"
    dump_dir.mkdir()
    candidate = _traj(0.0)
    np.savez(
        dump_dir / "tick_000.npz",
        candidate_xyh=np.stack([candidate, candidate, candidate]),
        selected_xyh=candidate,
        selected_index=np.array(1),
        candidate_corner_offroad_counts=np.array([0, 0, 0]),
    )

    result = _run_audit(dump_dir, tmp_path / "audit.csv")

    assert result.returncode == 0, result.stderr
    row = _read_one_row(tmp_path / "audit.csv")
    assert row["candidate_count"] == "3"
    assert row["selected_index"] == "1"
    assert row["effective_modes"] == "1"
    assert float(row["mode_entropy"]) == 0.0


def test_multimodal_candidates_report_multiple_effective_modes(tmp_path: Path) -> None:
    dump_dir = tmp_path / "dumps"
    dump_dir.mkdir()
    candidates = np.stack(
        [
            _traj(0.0),
            _traj(0.2),
            _traj(5.0, heading=0.6),
            _traj(-5.0, heading=-0.6),
        ]
    )
    np.savez(
        dump_dir / "tick_001.npz",
        candidates_local_xyh=candidates,
        selected_xyh=candidates[0],
        selected_index=np.array(0),
        candidate_corner_offroad_counts=np.array([0, 0, 0, 0]),
        candidate_dynamic_min_clearance_m=np.array([2.0, 2.0, 2.0, 2.0]),
        candidate_static_min_clearance_m=np.array([3.0, 3.0, 3.0, 3.0]),
        candidate_route_projection_final_lateral_error_m=np.array([0.0, 0.2, 5.0, -5.0]),
    )

    result = _run_audit(dump_dir, tmp_path / "audit.csv")

    assert result.returncode == 0, result.stderr
    row = _read_one_row(tmp_path / "audit.csv")
    assert int(row["effective_modes"]) > 1
    assert float(row["endpoint_pairwise_max"]) > 9.0
    assert float(row["trajectory_pairwise_mean"]) > 0.0
    assert float(row["final_heading_spread"]) > 1.0


def test_selector_miss_safe_candidate_when_selected_is_unsafe(tmp_path: Path) -> None:
    dump_dir = tmp_path / "dumps"
    dump_dir.mkdir()
    candidates = np.stack([_traj(0.0), _traj(3.0), _traj(-3.0)])
    np.savez(
        dump_dir / "tick_002.npz",
        candidate_xyh=candidates,
        selected_xyh=candidates[1],
        selected_index=np.array(1),
        candidate_corner_offroad_counts=np.array([0, 5, 0]),
    )

    result = _run_audit(dump_dir, tmp_path / "audit.csv")

    assert result.returncode == 0, result.stderr
    row = _read_one_row(tmp_path / "audit.csv")
    assert row["safe_candidate_count"] == "2"
    assert row["selected_is_safe"] == "False"
    assert row["best_available_offroad_count"] == "0"
    assert row["selector_miss_safe_candidate"] == "True"
