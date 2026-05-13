#!/usr/bin/env python3
"""Audit diffusion-planner training manifests for supervision/cache contracts."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROUTE_THRESHOLDS = (3, 5, 10, 30)
ROUTE_LANE_BUCKETS = (
    ("missing", None, None),
    ("0", 0, 0),
    ("1-5", 1, 5),
    ("6-20", 6, 20),
    ("21-50", 21, 50),
    (">50", 51, None),
)
TAG_GROUPS = ("straight", "left", "right", "intersection", "stationary", "high_speed", "other")
CSV_FIELDS = [
    "scope",
    "key",
    "total_rows",
    "hard_skip_rows",
    "hard_skip_ratio",
    "route_min_dist_missing",
    *[f"route_min_dist_gt_{threshold}" for threshold in ROUTE_THRESHOLDS],
    *[f"route_min_dist_gt_{threshold}_ratio" for threshold in ROUTE_THRESHOLDS],
    *[f"route_lanes_avails_sum_{bucket[0]}" for bucket in ROUTE_LANE_BUCKETS],
    *[f"tag_{group}" for group in TAG_GROUPS],
]


@dataclass
class AuditStats:
    total_rows: int = 0
    hard_skip_rows: int = 0
    route_min_dist_missing: int = 0
    route_threshold_counts: Counter[int] = field(default_factory=Counter)
    route_lanes_buckets: Counter[str] = field(default_factory=Counter)
    tag_groups: Counter[str] = field(default_factory=Counter)
    qc_flags: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add_example(self, key: str, sample_id: str) -> None:
        if len(self.examples[key]) < 20:
            self.examples[key].append(sample_id)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _sample_id(row: dict[str, Any], row_index: int) -> str:
    for key in ("sample_id", "scene_token_hex", "db_name"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row_{row_index:08d}"


def _bucket_route_lanes(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "missing"
    for label, lower, upper in ROUTE_LANE_BUCKETS:
        if label == "missing":
            continue
        if lower is not None and numeric < lower:
            continue
        if upper is not None and numeric > upper:
            continue
        return label
    return ">50"


def _tag_groups(tags: Any) -> set[str]:
    groups: set[str] = set()
    normalized = [str(tag).strip().lower() for tag in _as_list(tags) if str(tag).strip()]
    for tag in normalized:
        compact = tag.replace("-", "_").replace(" ", "_")
        if "straight" in compact:
            groups.add("straight")
        if "left" in compact:
            groups.add("left")
        if "right" in compact:
            groups.add("right")
        if "intersect" in compact or compact in {"junction", "crossing"}:
            groups.add("intersection")
        if "stationary" in compact or "stopped" in compact or "parked" in compact:
            groups.add("stationary")
        if "high_speed" in compact or "fast" in compact:
            groups.add("high_speed")
    if not groups:
        groups.add("other")
    return groups


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{count / total * 100.0:.2f}%"


def _count_row(stats: AuditStats, row: dict[str, Any], row_index: int) -> None:
    stats.total_rows += 1
    sample_id = _sample_id(row, row_index)

    if _as_bool(row.get("qc_hard_skip")):
        stats.hard_skip_rows += 1
        stats.add_example("qc_hard_skip", sample_id)

    route_min_dist = _as_float(row.get("route_min_dist_m"))
    if route_min_dist is None:
        stats.route_min_dist_missing += 1
        stats.add_example("missing:route_min_dist_m", sample_id)
    else:
        for threshold in ROUTE_THRESHOLDS:
            if route_min_dist > threshold:
                stats.route_threshold_counts[threshold] += 1
                stats.add_example(f"route_min_dist_gt_{threshold}", sample_id)

    route_lanes_bucket = _bucket_route_lanes(row.get("route_lanes_avails_sum"))
    stats.route_lanes_buckets[route_lanes_bucket] += 1
    if route_lanes_bucket == "missing":
        stats.add_example("missing:route_lanes_avails_sum", sample_id)

    for group in _tag_groups(row.get("tags")):
        stats.tag_groups[group] += 1

    for flag in _as_list(row.get("qc_flags")):
        flag_name = str(flag).strip()
        if flag_name:
            stats.qc_flags[flag_name] += 1
            stats.add_example(f"qc_flag:{flag_name}", sample_id)


def _read_manifest(path: Path, all_stats: AuditStats, by_location: dict[str, AuditStats], start_index: int) -> int:
    row_index = start_index
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object row")
            row_index += 1
            _count_row(all_stats, row, row_index)
            location = str(row.get("location") or "missing")
            _count_row(by_location[location], row, row_index)
    return row_index


def collect_manifest_stats(patterns: list[str]) -> tuple[list[Path], AuditStats, dict[str, AuditStats]]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if not matches:
            matches = [Path(pattern)] if Path(pattern).exists() else []
        paths.extend(matches)
    unique_paths = sorted(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise ValueError("no manifest files matched --manifest-glob")

    all_stats = AuditStats()
    by_location: dict[str, AuditStats] = defaultdict(AuditStats)
    row_index = 0
    for path in unique_paths:
        row_index = _read_manifest(path, all_stats, by_location, row_index)
    return unique_paths, all_stats, dict(sorted(by_location.items()))


def _stats_to_csv_row(scope: str, key: str, stats: AuditStats) -> dict[str, str]:
    row = {
        "scope": scope,
        "key": key,
        "total_rows": str(stats.total_rows),
        "hard_skip_rows": str(stats.hard_skip_rows),
        "hard_skip_ratio": _pct(stats.hard_skip_rows, stats.total_rows),
        "route_min_dist_missing": str(stats.route_min_dist_missing),
    }
    for threshold in ROUTE_THRESHOLDS:
        count = stats.route_threshold_counts[threshold]
        row[f"route_min_dist_gt_{threshold}"] = str(count)
        row[f"route_min_dist_gt_{threshold}_ratio"] = _pct(count, stats.total_rows)
    for label, _, _ in ROUTE_LANE_BUCKETS:
        row[f"route_lanes_avails_sum_{label}"] = str(stats.route_lanes_buckets[label])
    for group in TAG_GROUPS:
        row[f"tag_{group}"] = str(stats.tag_groups[group])
    return row


def write_csv(path: Path, all_stats: AuditStats, by_location: dict[str, AuditStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(_stats_to_csv_row("all", "ALL", all_stats))
        for location, stats in by_location.items():
            writer.writerow(_stats_to_csv_row("location", location, stats))


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _summary_lines(stats: AuditStats) -> list[str]:
    lines = [
        f"Total rows: {stats.total_rows}",
        f"Hard skips: {stats.hard_skip_rows} ({_pct(stats.hard_skip_rows, stats.total_rows)})",
        "",
        "## Route Min Distance",
    ]
    lines.extend(
        _table(
            ["threshold", "count", "ratio"],
            [[f">{threshold}m", str(stats.route_threshold_counts[threshold]), _pct(stats.route_threshold_counts[threshold], stats.total_rows)] for threshold in ROUTE_THRESHOLDS],
        )
    )
    lines.extend(["", "## Route Lane Avails Sum"])
    lines.extend(
        _table(
            ["bucket", "count", "ratio"],
            [[label, str(stats.route_lanes_buckets[label]), _pct(stats.route_lanes_buckets[label], stats.total_rows)] for label, _, _ in ROUTE_LANE_BUCKETS],
        )
    )
    lines.extend(["", "## Tag Groups"])
    lines.extend(
        _table(
            ["tag_group", "count", "ratio"],
            [[group, str(stats.tag_groups[group]), _pct(stats.tag_groups[group], stats.total_rows)] for group in TAG_GROUPS],
        )
    )
    return lines


def write_markdown(path: Path, manifest_paths: list[Path], all_stats: AuditStats, by_location: dict[str, AuditStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training Contract Audit",
        "",
        "## Inputs",
        *[f"- `{manifest_path}`" for manifest_path in manifest_paths],
        "",
        "## Overall",
        *_summary_lines(all_stats),
        "",
        "## By Location",
    ]
    lines.extend(
        _table(
            ["location", "rows", "hard_skip", "route>3m", "route>3m_ratio", "route>30m", "route>30m_ratio"],
            [
                [
                    location,
                    str(stats.total_rows),
                    str(stats.hard_skip_rows),
                    str(stats.route_threshold_counts[3]),
                    _pct(stats.route_threshold_counts[3], stats.total_rows),
                    str(stats.route_threshold_counts[30]),
                    _pct(stats.route_threshold_counts[30], stats.total_rows),
                ]
                for location, stats in by_location.items()
            ],
        )
    )
    lines.extend(["", "## QC Flags"])
    if all_stats.qc_flags:
        lines.extend(
            _table(
                ["flag", "count", "ratio"],
                [[flag, str(count), _pct(count, all_stats.total_rows)] for flag, count in sorted(all_stats.qc_flags.items())],
            )
        )
    else:
        lines.append("No qc_flags present.")
    lines.extend(["", "## Bad Sample Examples"])
    if all_stats.examples:
        for key in sorted(all_stats.examples):
            lines.append(f"- {key}: {', '.join(all_stats.examples[key])}")
    else:
        lines.append("No bad sample examples found.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-glob", action="append", required=True, help="JSONL manifest path/glob; may be repeated")
    parser.add_argument("--out-md", required=True, type=Path, help="Markdown report path")
    parser.add_argument("--out-csv", required=True, type=Path, help="CSV summary path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_paths, all_stats, by_location = collect_manifest_stats(args.manifest_glob)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    write_markdown(args.out_md, manifest_paths, all_stats, by_location)
    write_csv(args.out_csv, all_stats, by_location)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
