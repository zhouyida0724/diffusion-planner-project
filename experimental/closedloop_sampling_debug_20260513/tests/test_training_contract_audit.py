from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "dp_training_contract_audit.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run_audit(tmp_path: Path, *manifest_globs: str) -> subprocess.CompletedProcess[str]:
    out_md = tmp_path / "audit.md"
    out_csv = tmp_path / "audit.csv"
    cmd = [sys.executable, str(SCRIPT)]
    for manifest_glob in manifest_globs:
        cmd.extend(["--manifest-glob", manifest_glob])
    cmd.extend(["--out-md", str(out_md), "--out-csv", str(out_csv)])
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def test_manifest_audit_summarizes_thresholds_tags_locations_and_examples(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "straight_ok",
                "location": "us-ma-boston",
                "tags": ["straight", "high_speed"],
                "route_min_dist_m": 2.5,
                "route_lanes_avails_sum": 0,
            },
            {
                "sample_id": "left_intersection_bad",
                "location": "us-nv-las-vegas-strip",
                "qc_flags": ["route_far"],
                "tags": ["turn_left", "intersection"],
                "route_min_dist_m": 6.1,
                "route_lanes_avails_sum": 3,
            },
            {
                "sample_id": "right_far_skip",
                "location": "us-nv-las-vegas-strip",
                "qc_flags": ["route_far", "label_missing"],
                "qc_hard_skip": True,
                "tags": ["RIGHT_TURN"],
                "route_min_dist_m": 31.0,
                "route_lanes_avails_sum": 15,
            },
            {
                "sample_id": "stationary",
                "location": "us-ma-boston",
                "tags": ["stationary"],
                "route_min_dist_m": 3.0,
                "route_lanes_avails_sum": 80,
            },
        ],
    )

    result = _run_audit(tmp_path, str(manifest))

    assert result.returncode == 0, result.stderr
    report = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "Total rows: 4" in report
    assert "Hard skips: 1 (25.00%)" in report
    assert "| >5m | 2 | 50.00% |" in report
    assert "| >10m | 1 | 25.00% |" in report
    assert "| left | 1 | 25.00% |" in report
    assert "| intersection | 1 | 25.00% |" in report
    assert "| us-nv-las-vegas-strip | 2 | 1 | 2 | 100.00% | 1 | 50.00% |" in report
    assert "route_min_dist_gt_30: right_far_skip" in report
    assert "qc_flag:route_far: left_intersection_bad, right_far_skip" in report

    rows = list(csv.DictReader((tmp_path / "audit.csv").open(encoding="utf-8")))
    total = next(row for row in rows if row["scope"] == "all" and row["key"] == "ALL")
    assert total["total_rows"] == "4"
    assert total["hard_skip_rows"] == "1"
    assert total["route_min_dist_gt_3"] == "2"
    assert total["route_min_dist_gt_30"] == "1"
    assert total["tag_left"] == "1"
    assert total["tag_high_speed"] == "1"


def test_manifest_audit_handles_repeated_globs_missing_fields_and_other_tags(tmp_path: Path) -> None:
    first = tmp_path / "part-a.jsonl"
    second = tmp_path / "part-b.jsonl"
    _write_jsonl(
        first,
        [
            {
                "sample_id": "missing_optional",
                "db_name": "mini",
                "scene_token_hex": "abc",
                "tags": [],
            }
        ],
    )
    _write_jsonl(
        second,
        [
            {
                "sample_id": "freeform",
                "location": "",
                "tags": ["lane_change"],
                "qc_flags": "flag_from_string",
                "route_min_dist_m": "4.2",
                "route_lanes_avails_sum": None,
            }
        ],
    )

    result = _run_audit(tmp_path, str(first), str(second))

    assert result.returncode == 0, result.stderr
    report = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "Total rows: 2" in report
    assert "| missing | 2 | 100.00% |" in report
    assert "| other | 2 | 100.00% |" in report
    assert "missing:route_min_dist_m: missing_optional" in report
    assert "qc_flag:flag_from_string: freeform" in report
    rows = list(csv.DictReader((tmp_path / "audit.csv").open(encoding="utf-8")))
    total = next(row for row in rows if row["scope"] == "all")
    assert total["route_min_dist_missing"] == "1"
    assert total["route_lanes_avails_sum_missing"] == "2"
    assert total["tag_other"] == "2"
