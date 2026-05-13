from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "dp_selector_trace_diversity_audit.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _run_audit(trace_jsonl: Path, out_csv: Path, out_md: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-jsonl",
            str(trace_jsonl),
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
            *extra_args,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _row(rows: list[dict[str, str]], level: str, group_id: str) -> dict[str, str]:
    for item in rows:
        if item["summary_level"] == level and item["group_id"] == group_id:
            return item
    raise AssertionError(f"missing row {level=} {group_id=}: {rows}")


def _safe_candidate(rejected: bool = False, mode: str | None = None) -> dict:
    item = {
        "prefix_offroad_steps": 0,
        "late_offroad_steps": 0,
        "rejected": rejected,
        "final_score": 1.0,
    }
    if mode is not None:
        item["mode"] = mode
    return item


def _unsafe_candidate() -> dict:
    return {
        "prefix_offroad_steps": 1,
        "late_offroad_steps": 0,
        "rejected": False,
        "final_score": -1.0,
    }


def test_trace_only_summarizes_safe_miss_and_diversity(tmp_path: Path) -> None:
    trace = tmp_path / "selector_trace.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "token": "tok-a",
                "tick": 1,
                "best_index": 1,
                "used_fallback": False,
                "diagnostics": [_safe_candidate(mode="left"), _unsafe_candidate(), _safe_candidate(mode="right")],
                "candidate_diversity": {
                    "effective_modes": 2,
                    "mode_entropy": 0.7,
                    "endpoint_pairwise_mean": 3.0,
                    "trajectory_pairwise_mean": 4.0,
                },
            },
            {
                "token": "tok-a",
                "tick": 2,
                "best_index": 0,
                "used_fallback": True,
                "diagnostics": [_safe_candidate(mode="left"), _unsafe_candidate()],
                "candidate_diversity": {
                    "effective_modes": 1,
                    "mode_entropy": 0.0,
                    "endpoint_pairwise_mean": 1.0,
                    "trajectory_pairwise_mean": 2.0,
                },
            },
        ],
    )

    result = _run_audit(trace, tmp_path / "audit.csv", tmp_path / "audit.md", "--config-id", "cfg", "--run-id", "run")

    assert result.returncode == 0, result.stderr
    rows = _read_rows(tmp_path / "audit.csv")
    token = _row(rows, "token", "tok-a")
    assert token["config_id"] == "cfg"
    assert token["run_id"] == "run"
    assert token["ticks"] == "2"
    assert float(token["candidate_count_median"]) == 2.5
    assert float(token["effective_modes_p50"]) == 1.5
    assert float(token["mode_entropy_mean"]) == 0.35
    assert float(token["endpoint_spread_mean"]) == 2.0
    assert float(token["trajectory_spread_mean"]) == 3.0
    assert float(token["safe_candidate_rate"]) == 0.6
    assert float(token["selector_miss_rate"]) == 0.5
    assert float(token["fallback_rate"]) == 0.5
    assert float(token["no_safe_candidate_tick_rate"]) == 0.0
    assert float(token["safe_mode_count"]) == 2.0


def test_missing_candidate_diversity_outputs_nan_and_markdown_missing_fields(tmp_path: Path) -> None:
    trace = tmp_path / "selector_trace.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "token": "tok-b",
                "tick": 1,
                "best_index": 0,
                "used_fallback": False,
                "diagnostics": [_safe_candidate(), _unsafe_candidate()],
            }
        ],
    )

    result = _run_audit(trace, tmp_path / "audit.csv", tmp_path / "audit.md")

    assert result.returncode == 0, result.stderr
    token = _row(_read_rows(tmp_path / "audit.csv"), "token", "tok-b")
    assert math.isnan(float(token["effective_modes_p50"]))
    assert math.isnan(float(token["mode_entropy_mean"]))
    assert math.isnan(float(token["endpoint_spread_mean"]))
    assert math.isnan(float(token["trajectory_spread_mean"]))
    assert token["safe_mode_count"] == "NaN"
    assert "candidate_diversity" in (tmp_path / "audit.md").read_text(encoding="utf-8")


def test_no_safe_candidate_tick_rate_and_overall_rows(tmp_path: Path) -> None:
    trace = tmp_path / "selector_trace.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "token": "tok-c",
                "tick": 1,
                "best_index": 0,
                "used_fallback": False,
                "diagnostics": [_unsafe_candidate(), _unsafe_candidate()],
                "candidate_diversity": {"effective_modes": 1},
            }
        ],
    )

    result = _run_audit(trace, tmp_path / "audit.csv", tmp_path / "audit.md")

    assert result.returncode == 0, result.stderr
    rows = _read_rows(tmp_path / "audit.csv")
    token = _row(rows, "token", "tok-c")
    overall = _row(rows, "overall", "overall")
    assert float(token["safe_candidate_rate"]) == 0.0
    assert float(token["no_safe_candidate_tick_rate"]) == 1.0
    assert float(token["selector_miss_rate"]) == 0.0
    assert overall["ticks"] == "1"


def test_accepts_pairwise_diversity_keys_and_top_level_mode_labels(tmp_path: Path) -> None:
    trace = tmp_path / "selector_trace.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "token": "tok-d",
                "tick": 1,
                "best_index": 0,
                "used_fallback": False,
                "diagnostics": [_safe_candidate(), _safe_candidate(), _unsafe_candidate()],
                "candidate_mode_labels": [0, 1, 1],
                "candidate_diversity": {
                    "effective_modes": 2,
                    "mode_entropy": 0.69,
                    "pairwise_final_l2_mean_m": 3.5,
                    "pairwise_rms_mean_m": 1.5,
                },
            }
        ],
    )

    result = _run_audit(trace, tmp_path / "audit.csv", tmp_path / "audit.md")

    assert result.returncode == 0, result.stderr
    token = _row(_read_rows(tmp_path / "audit.csv"), "token", "tok-d")
    assert float(token["effective_modes_p50"]) == 2.0
    assert float(token["mode_entropy_mean"]) == 0.69
    assert float(token["endpoint_spread_mean"]) == 3.5
    assert float(token["trajectory_spread_mean"]) == 1.5
    assert float(token["safe_mode_count"]) == 2.0
