#!/usr/bin/env python3
"""Summarize selector trace candidate diversity and safety proxies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


CSV_FIELDS = [
    "summary_level",
    "group_id",
    "config_id",
    "run_id",
    "token",
    "ticks",
    "candidate_count_median",
    "candidate_count_mean",
    "effective_modes_p50",
    "mode_entropy_mean",
    "endpoint_spread_mean",
    "trajectory_spread_mean",
    "safe_candidate_count_mean",
    "safe_candidate_rate",
    "safe_mode_count",
    "selector_miss_safe_candidate_count",
    "selector_miss_rate",
    "fallback_count",
    "fallback_rate",
    "no_safe_candidate_tick_count",
    "no_safe_candidate_tick_rate",
    "missing_fields",
]

DIVERSITY_KEYS = {
    "effective_modes": ("effective_modes", "effective_mode_count", "effective_modes_count"),
    "mode_entropy": ("mode_entropy", "candidate_mode_entropy"),
    "endpoint_spread": (
        "endpoint_spread",
        "pairwise_final_l2_mean_m",
        "endpoint_pairwise_mean",
        "endpoint_pairwise",
        "endpoint_pairwise_distance_mean",
    ),
    "trajectory_spread": (
        "trajectory_spread",
        "pairwise_rms_mean_m",
        "trajectory_pairwise_mean",
        "trajectory_pairwise",
        "trajectory_pairwise_distance_mean",
    ),
}
MISSING_FIELD_ORDER = [
    "diagnostics",
    "best_index",
    "used_fallback",
    "candidate_diversity",
    "candidate_diversity.effective_modes",
    "candidate_diversity.mode_entropy",
    "candidate_diversity.endpoint_spread",
    "candidate_diversity.trajectory_spread",
    "candidate_mode_labels",
]


def _nan() -> float:
    return float("nan")


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _finite_values(values: list[float]) -> list[float]:
    return [value for value in values if not _is_nan(value)]


def _mean(values: list[float]) -> float:
    finite = _finite_values(values)
    return sum(finite) / len(finite) if finite else _nan()


def _median(values: list[float]) -> float:
    finite = _finite_values(values)
    return float(median(finite)) if finite else _nan()


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else _nan()


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return _nan()
    try:
        return float(value)
    except (TypeError, ValueError):
        return _nan()


def _format(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.12g}"
    if value is None:
        return ""
    return str(value)


def _load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(item)
    return rows


def _candidate_items(item: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = item.get("diagnostics")
    if isinstance(diagnostics, dict):
        for key in ("candidates", "candidate_diagnostics", "items"):
            values = diagnostics.get(key)
            if isinstance(values, list):
                return [value for value in values if isinstance(value, dict)]
        return []
    if isinstance(diagnostics, list):
        return [value for value in diagnostics if isinstance(value, dict)]
    return []


def _is_rejected(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("rejected", False))


def _is_safe_candidate(candidate: dict[str, Any]) -> bool:
    return (
        _as_float(candidate.get("prefix_offroad_steps")) == 0.0
        and _as_float(candidate.get("late_offroad_steps")) == 0.0
        and not _is_rejected(candidate)
    )


def _mode_label(candidate: dict[str, Any]) -> str | None:
    for key in ("mode", "mode_label", "cluster_label", "candidate_mode"):
        value = candidate.get(key)
        if value is not None and value != "":
            return str(value)
    return None


def _first_metric(mapping: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in mapping:
            return _as_float(mapping[key])
    return _nan()


def _tick_summary(item: dict[str, Any], config_id: str, run_id: str) -> dict[str, Any]:
    candidates = _candidate_items(item)
    candidate_count = len(candidates) if candidates else _nan()
    safe_flags = [_is_safe_candidate(candidate) for candidate in candidates]
    safe_candidate_count = sum(1 for is_safe in safe_flags if is_safe)
    safe_candidate_rate = (
        float(safe_candidate_count) / float(len(candidates)) if candidates else _nan()
    )
    best_index = item.get("best_index", item.get("selected_index", item.get("selected_candidate_index")))
    try:
        selected_index = int(best_index) if best_index is not None else None
    except (TypeError, ValueError):
        selected_index = None
    selected_is_safe = (
        bool(safe_flags[selected_index])
        if selected_index is not None and 0 <= selected_index < len(safe_flags)
        else False
    )
    safe_exists = safe_candidate_count > 0
    trace_mode_labels = item.get("candidate_mode_labels")
    trace_mode_labels = trace_mode_labels if isinstance(trace_mode_labels, list) else []
    mode_labels = []
    for index, candidate in enumerate(candidates):
        if not _is_safe_candidate(candidate):
            continue
        label = _mode_label(candidate)
        if label is None and index < len(trace_mode_labels):
            label = str(trace_mode_labels[index])
        mode_labels.append(label)
    safe_mode_labels = {label for label in mode_labels if label is not None}
    safe_mode_count = (
        float(len(safe_mode_labels)) if mode_labels and all(label is not None for label in mode_labels) else _nan()
    )
    diversity = item.get("candidate_diversity")
    diversity = diversity if isinstance(diversity, dict) else {}
    missing: set[str] = set()
    if "diagnostics" not in item:
        missing.add("diagnostics")
    if best_index is None:
        missing.add("best_index")
    if "used_fallback" not in item:
        missing.add("used_fallback")
    if not diversity:
        missing.add("candidate_diversity")
    if _is_nan(safe_mode_count):
        missing.add("candidate_mode_labels")
    metrics = {
        "effective_modes": _first_metric(diversity, DIVERSITY_KEYS["effective_modes"]),
        "mode_entropy": _first_metric(diversity, DIVERSITY_KEYS["mode_entropy"]),
        "endpoint_spread": _first_metric(diversity, DIVERSITY_KEYS["endpoint_spread"]),
        "trajectory_spread": _first_metric(diversity, DIVERSITY_KEYS["trajectory_spread"]),
    }
    for name, value in metrics.items():
        if _is_nan(value):
            missing.add(f"candidate_diversity.{name}")
    return {
        "summary_level": "tick",
        "group_id": f"{item.get('token', 'unknown')}:{item.get('tick', '')}",
        "config_id": config_id,
        "run_id": run_id,
        "token": str(item.get("token", "unknown")),
        "tick": item.get("tick"),
        "candidate_count": candidate_count,
        "effective_modes": metrics["effective_modes"],
        "mode_entropy": metrics["mode_entropy"],
        "endpoint_spread": metrics["endpoint_spread"],
        "trajectory_spread": metrics["trajectory_spread"],
        "safe_candidate_count": float(safe_candidate_count) if candidates else _nan(),
        "safe_candidate_rate": safe_candidate_rate,
        "safe_mode_labels": safe_mode_labels,
        "safe_mode_count": safe_mode_count,
        "selector_miss_safe_candidate": bool(safe_exists and not selected_is_safe),
        "used_fallback": bool(item.get("used_fallback", False)),
        "no_safe_candidate_tick": bool(candidates and not safe_exists),
        "missing_fields": missing,
    }


def _summarize_group(
    level: str,
    group_id: str,
    ticks: list[dict[str, Any]],
    config_id: str,
    run_id: str,
    token: str = "",
) -> dict[str, Any]:
    tick_count = len(ticks)
    safe_candidates = sum(
        int(tick["safe_candidate_count"])
        for tick in ticks
        if not _is_nan(tick["safe_candidate_count"])
    )
    total_candidates = sum(
        int(tick["candidate_count"])
        for tick in ticks
        if not _is_nan(tick["candidate_count"])
    )
    missing = set().union(*(tick["missing_fields"] for tick in ticks)) if ticks else set()
    return {
        "summary_level": level,
        "group_id": group_id,
        "config_id": config_id,
        "run_id": run_id,
        "token": token,
        "ticks": tick_count,
        "candidate_count_median": _median([tick["candidate_count"] for tick in ticks]),
        "candidate_count_mean": _mean([tick["candidate_count"] for tick in ticks]),
        "effective_modes_p50": _median([tick["effective_modes"] for tick in ticks]),
        "mode_entropy_mean": _mean([tick["mode_entropy"] for tick in ticks]),
        "endpoint_spread_mean": _mean([tick["endpoint_spread"] for tick in ticks]),
        "trajectory_spread_mean": _mean([tick["trajectory_spread"] for tick in ticks]),
        "safe_candidate_count_mean": _mean([tick["safe_candidate_count"] for tick in ticks]),
        "safe_candidate_rate": _rate(safe_candidates, total_candidates),
        "safe_mode_count": (
            float(len(set().union(*(tick["safe_mode_labels"] for tick in ticks))))
            if "candidate_mode_labels" not in missing
            else _nan()
        ),
        "selector_miss_safe_candidate_count": sum(
            1 for tick in ticks if tick["selector_miss_safe_candidate"]
        ),
        "selector_miss_rate": _rate(
            sum(1 for tick in ticks if tick["selector_miss_safe_candidate"]), tick_count
        ),
        "fallback_count": sum(1 for tick in ticks if tick["used_fallback"]),
        "fallback_rate": _rate(sum(1 for tick in ticks if tick["used_fallback"]), tick_count),
        "no_safe_candidate_tick_count": sum(1 for tick in ticks if tick["no_safe_candidate_tick"]),
        "no_safe_candidate_tick_rate": _rate(
            sum(1 for tick in ticks if tick["no_safe_candidate_tick"]), tick_count
        ),
        "missing_fields": ";".join(field for field in MISSING_FIELD_ORDER if field in missing),
    }


def analyze(trace_jsonl: Path, config_id: str = "", run_id: str = "") -> list[dict[str, Any]]:
    items = _load_trace(trace_jsonl)
    tick_rows = [_tick_summary(item, config_id, run_id) for item in items]
    rows: list[dict[str, Any]] = []
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tick_rows:
        by_run[row["run_id"] or "run"].append(row)
        by_token[row["token"]].append(row)
    for current_run_id, ticks in sorted(by_run.items()):
        rows.append(_summarize_group("run", current_run_id, ticks, config_id, run_id))
    for token, ticks in sorted(by_token.items()):
        rows.append(_summarize_group("token", token, ticks, config_id, run_id, token=token))
    rows.append(_summarize_group("overall", "overall", tick_rows, config_id, run_id))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in CSV_FIELDS})


def write_md(rows: list[dict[str, Any]], path: Path, trace_jsonl: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = sorted({field for row in rows for field in str(row.get("missing_fields", "")).split(";") if field})
    lines = [
        "# Selector Trace Diversity Audit",
        "",
        f"- trace_jsonl: `{trace_jsonl}`",
        f"- summary_rows: {len(rows)}",
        "",
        "## Sort Columns",
        "",
        "- effective_modes_p50",
        "- mode_entropy_mean",
        "- endpoint_spread_mean",
        "- trajectory_spread_mean",
        "- safe_candidate_rate",
        "- no_safe_candidate_tick_rate",
        "- selector_miss_rate",
        "- fallback_rate",
        "",
        "## Missing Fields",
        "",
    ]
    if missing:
        lines.extend(f"- {field}" for field in missing)
    else:
        lines.append("- none")
    lines.extend(["", "## Summary", ""])
    lines.append("| level | group | ticks | safe_candidate_rate | selector_miss_rate | fallback_rate | no_safe_candidate_tick_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {summary_level} | {group_id} | {ticks} | {safe_candidate_rate} | {selector_miss_rate} | {fallback_rate} | {no_safe_candidate_tick_rate} |".format(
                **{key: _format(row.get(key, "")) for key in CSV_FIELDS}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--config-id", default="")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    rows = analyze(args.trace_jsonl, config_id=args.config_id, run_id=args.run_id)
    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md, args.trace_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
