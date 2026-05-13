from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "dp_sampling_grid_plan.py"


def _write_scenarios(path: Path) -> None:
    rows = [
        {
            "token": "tok_left",
            "group": "maneuver",
            "scenario_type": "left_turn",
            "location": "sg-one-north",
            "map_version": "sg-v1",
            "log_name": "log_left",
            "yaw_change_rad": "1.1",
            "travel_m": "35",
        },
        {
            "token": "tok_left_dup",
            "group": "maneuver",
            "scenario_type": "left_turn",
            "location": "sg-one-north",
            "map_version": "sg-v1",
            "log_name": "log_left",
            "yaw_change_rad": "1.0",
            "travel_m": "30",
        },
        {
            "token": "tok_right",
            "group": "maneuver",
            "scenario_type": "right_turn",
            "location": "us-ma-boston",
            "map_version": "us-v1",
            "log_name": "log_right",
            "yaw_change_rad": "-1.2",
            "travel_m": "25",
        },
        {
            "token": "tok_cruise",
            "group": "straight",
            "scenario_type": "lane_following",
            "location": "us-pa-pittsburgh",
            "map_version": "us-v1",
            "log_name": "log_cruise",
            "yaw_change_rad": "0.02",
            "travel_m": "80",
        },
        {
            "token": "tok_low",
            "group": "straight",
            "scenario_type": "lane_following",
            "location": "us-nv-las-vegas",
            "map_version": "us-v1",
            "log_name": "log_low",
            "yaw_change_rad": "0.01",
            "travel_m": "7",
        },
        {
            "token": "tok_intersection",
            "group": "intersection",
            "scenario_type": "unknown",
            "location": "sg-queenstown",
            "map_version": "sg-v1",
            "log_name": "log_intersection",
            "yaw_change_rad": "0.3",
            "travel_m": "20",
        },
        {
            "token": "tok_high_yaw",
            "group": "maneuver",
            "scenario_type": "u_turn",
            "location": "us-ca-sf",
            "map_version": "us-v1",
            "log_name": "log_high_yaw",
            "yaw_change_rad": "2.4",
            "travel_m": "18",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_tool(tmp_path: Path, matrix: str = "main_grid") -> Path:
    scenarios_csv = tmp_path / "scenarios.csv"
    output_dir = tmp_path / f"plan-{matrix}"
    _write_scenarios(scenarios_csv)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-csv",
            str(scenarios_csv),
            "--output-dir",
            str(output_dir),
            "--matrix",
            matrix,
            "--checkpoint",
            "outputs/training/dp.ckpt",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir


def test_selects_diverse_scenarios_and_deduplicates_by_log(tmp_path: Path) -> None:
    output_dir = _run_tool(tmp_path)
    rows = list(csv.DictReader((output_dir / "selected_scenarios.csv").open(encoding="utf-8")))

    assert len(rows) == 6
    assert "tok_left_dup" not in {row["token"] for row in rows}
    assert {row["behavior_bucket"] for row in rows} >= {
        "left",
        "right",
        "straight_cruise",
        "straight_low_speed_stop_proxy",
        "intersection_proxy",
        "turn_high_yaw",
    }
    assert all(row["geo_cell"] == "unknown" for row in rows)


def test_main_grid_has_20_valid_configs_with_fixed_selector_samples(tmp_path: Path) -> None:
    output_dir = _run_tool(tmp_path, "main_grid")
    rows = list(csv.DictReader((output_dir / "run_matrix_main_grid.csv").open(encoding="utf-8")))

    assert len(rows) == 20
    assert {row["sampling_steps"] for row in rows} == {"2", "3", "5", "10"}
    assert {row["noise_scale"] for row in rows} == {"0.25", "0.5", "0.75", "1.0", "1.5"}
    assert all(row["selector_fixed_samples"] == "10" for row in rows)
    assert all("DP_TRAJ_SELECTOR_SAMPLES=10" in row["command"] for row in rows)
    assert all("DP_TRAJ_SELECTOR_TRACE_TICK0_ONLY=0" in row["command"] for row in rows)
    assert all("DP_CLOSEDLOOP_DEBUG_CANDIDATES=0" in row["command"] for row in rows)
    assert all("DP_CLOSEDLOOP_DEBUG_DUMP_DIR" not in row["command"] for row in rows)
    assert all("'/media/zhouyida/新加卷1/nuplan_datasets/data/cache/test'" in row["command"] for row in rows)


def test_dpm_probe_has_default_plus_local_variants(tmp_path: Path) -> None:
    output_dir = _run_tool(tmp_path, "dpm_probe")
    rows = list(csv.DictReader((output_dir / "run_matrix_dpm_probe.csv").open(encoding="utf-8")))

    assert len(rows) == 6
    assert [row["dpm_variant"] for row in rows] == [
        "default",
        "order_1",
        "order_3",
        "skip_time_uniform",
        "method_singlestep",
        "denoise_to_zero_0",
    ]
    assert rows[0]["dpm_env_json"] == "{}"


def test_tokens_csv_and_runbook_mark_proxy_labels(tmp_path: Path) -> None:
    output_dir = _run_tool(tmp_path)
    token_rows = list(csv.DictReader((output_dir / "selected_tokens.csv").open(encoding="utf-8")))
    runbook = (output_dir / "sampling_grid_runbook.md").read_text(encoding="utf-8")

    assert set(row["token"] for row in token_rows) == {
        "tok_left",
        "tok_right",
        "tok_cruise",
        "tok_low",
        "tok_intersection",
        "tok_high_yaw",
    }
    assert "straight_low_speed_stop_proxy" in runbook
    assert "candidate_count fixed samples=10" in runbook
