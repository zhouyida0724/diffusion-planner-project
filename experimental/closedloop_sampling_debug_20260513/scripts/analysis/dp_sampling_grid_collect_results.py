#!/usr/bin/env python3
"""Collect DP sampling-grid status and selector-diversity metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence


FIELDS = [
    "matrix",
    "run_id",
    "sampling_steps",
    "noise_scale",
    "dpm_variant",
    "status",
    "duration_s",
    "trace_exists",
    "trace_bytes",
    "ticks",
    "candidate_count_median",
    "effective_modes_p50",
    "mode_entropy_mean",
    "endpoint_spread_mean",
    "trajectory_spread_mean",
    "safe_candidate_rate",
    "selector_miss_rate",
    "fallback_rate",
    "no_safe_candidate_tick_rate",
    "missing_fields",
    "output_root",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def _overall_audit(output_root: Path) -> dict[str, str]:
    rows = _read_csv(output_root / "diversity_audit.csv")
    for row in rows:
        if row.get("summary_level") == "overall":
            return row
    return {}


def collect(matrix_csv: Path, record_dir: Path) -> list[dict[str, Any]]:
    matrix_rows = _read_csv(matrix_csv)
    status_by_run = {row.get("run_id", ""): row for row in _read_csv(record_dir / "status.csv")}
    rows: list[dict[str, Any]] = []
    for matrix_row in matrix_rows:
        run_id = matrix_row.get("run_id", "")
        status = status_by_run.get(run_id, {})
        audit = _overall_audit(Path(matrix_row.get("output_root", "")))
        rows.append(
            {
                **{key: matrix_row.get(key, "") for key in ("matrix", "run_id", "sampling_steps", "noise_scale", "dpm_variant", "output_root")},
                **{key: status.get(key, "") for key in ("status", "duration_s", "trace_exists", "trace_bytes")},
                **{
                    key: audit.get(key, "")
                    for key in (
                        "ticks",
                        "candidate_count_median",
                        "effective_modes_p50",
                        "mode_entropy_mean",
                        "endpoint_spread_mean",
                        "trajectory_spread_mean",
                        "safe_candidate_rate",
                        "selector_miss_rate",
                        "fallback_rate",
                        "no_safe_candidate_tick_rate",
                        "missing_fields",
                    )
                },
            }
        )
    return rows


def _metric_table(rows: list[dict[str, Any]], metric: str, reverse: bool, limit: int = 8) -> list[str]:
    complete = [row for row in rows if row.get("status") == "ok" and row.get(metric, "") != ""]
    complete.sort(key=lambda row: _to_float(str(row.get(metric, ""))), reverse=reverse)
    lines = [f"### {metric}", "", "| rank | run_id | steps | noise | value | no_safe | effective_modes | endpoint_spread |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(complete[:limit], start=1):
        lines.append(
            "| {rank} | {run_id} | {sampling_steps} | {noise_scale} | {value} | {no_safe} | {modes} | {endpoint} |".format(
                rank=rank,
                run_id=row.get("run_id", ""),
                sampling_steps=row.get("sampling_steps", ""),
                noise_scale=row.get("noise_scale", ""),
                value=row.get(metric, ""),
                no_safe=row.get("no_safe_candidate_tick_rate", ""),
                modes=row.get("effective_modes_p50", ""),
                endpoint=row.get("endpoint_spread_mean", ""),
            )
        )
    if not complete:
        lines.append("|  | no completed rows |  |  |  |  |  |  |")
    return lines


def write_md(path: Path, rows: list[dict[str, Any]], matrix_csv: Path, record_dir: Path) -> None:
    done = sum(1 for row in rows if row.get("status") == "ok")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    pending = len(rows) - done - failed
    lines = [
        "# DP Sampling Grid Results",
        "",
        f"- matrix_csv: `{matrix_csv}`",
        f"- record_dir: `{record_dir}`",
        f"- rows: {len(rows)}; ok={done}; failed={failed}; pending={pending}",
        "",
        "Metrics are shown separately; no weighted score is used.",
        "",
    ]
    lines.extend(_metric_table(rows, "no_safe_candidate_tick_rate", reverse=False))
    lines.extend([""])
    lines.extend(_metric_table(rows, "effective_modes_p50", reverse=True))
    lines.extend([""])
    lines.extend(_metric_table(rows, "endpoint_spread_mean", reverse=True))
    lines.extend([""])
    lines.extend(_metric_table(rows, "selector_miss_rate", reverse=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", required=True, type=Path)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()

    rows = collect(args.matrix_csv, args.record_dir)
    _write_csv(args.out_csv, rows)
    write_md(args.out_md, rows, args.matrix_csv, args.record_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
