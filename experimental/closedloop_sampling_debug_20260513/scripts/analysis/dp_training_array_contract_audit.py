#!/usr/bin/env python3
"""Audit diffusion-planner array-backed training cache contracts."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = (
    "ego_agent_future",
    "route_lanes",
    "route_lanes_avails",
    "lanes",
    "lanes_avails",
    "ego_current_state",
)
OPTIONAL_FIELDS = (
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "static_objects",
)
DRIVABLE_FIELD_HINTS = ("drivable", "map_mask", "road_mask")


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _future_time_indices(num_steps: int, *, hz: float = 10.0) -> dict[str, int]:
    if num_steps <= 0:
        return {}
    indices: dict[str, int] = {}
    for label, seconds in (("1s", 1.0), ("3s", 3.0), ("5s", 5.0)):
        indices[label] = min(num_steps - 1, max(0, int(round(seconds * hz)) - 1))
    indices["final"] = num_steps - 1
    return indices


def _min_distances_to_route(
    future_xy: np.ndarray,
    route_lanes: np.ndarray,
    route_avails: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, time_steps, _ = future_xy.shape
    distances = np.full((batch_size, time_steps), np.nan, dtype=np.float64)
    nearest_indices = np.full((batch_size, time_steps), -1, dtype=np.int64)
    flat_route_xy = np.asarray(route_lanes[..., :2], dtype=np.float64).reshape(batch_size, -1, 2)
    flat_route_avails = np.asarray(route_avails, dtype=bool).reshape(batch_size, -1)
    for batch_index in range(batch_size):
        valid_flat_indices = np.flatnonzero(flat_route_avails[batch_index])
        if valid_flat_indices.size == 0:
            continue
        points = flat_route_xy[batch_index, valid_flat_indices]
        query = np.asarray(future_xy[batch_index], dtype=np.float64)
        diff = query[:, None, :] - points[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        local_nearest = np.argmin(dist, axis=1)
        distances[batch_index] = dist[np.arange(time_steps), local_nearest]
        nearest_indices[batch_index] = valid_flat_indices[local_nearest]
    return distances, nearest_indices


def _nanmax_rows(values: np.ndarray) -> np.ndarray:
    result = np.full((values.shape[0],), np.nan, dtype=np.float64)
    finite_rows = np.any(np.isfinite(values), axis=1)
    if np.any(finite_rows):
        result[finite_rows] = np.nanmax(values[finite_rows], axis=1)
    return result


def _trajectory_dynamics(future: np.ndarray, *, dt: float = 0.1) -> dict[str, Any]:
    future = np.asarray(future, dtype=np.float64)
    batch_size = future.shape[0]
    if future.ndim != 3 or future.shape[1] < 2 or future.shape[2] < 3:
        empty = np.full((batch_size,), np.nan, dtype=np.float64)
        return {
            "valid_samples": 0,
            "max_abs_heading_change_rad": empty,
            "max_abs_step_heading_delta_rad": empty,
            "max_abs_yaw_rate_radps": empty,
            "max_abs_curvature_1pm": empty,
            "max_abs_lateral_accel_proxy_mps2": empty,
            "mean_speed_mps": empty,
            "max_speed_mps": empty,
        }
    xy = future[:, :, :2]
    heading = np.unwrap(future[:, :, 2], axis=1)
    dxy = np.diff(xy, axis=1)
    speed = np.linalg.norm(dxy, axis=2) / dt
    heading_delta = np.diff(heading, axis=1)
    yaw_rate = heading_delta / dt
    curvature = np.full_like(yaw_rate, np.nan, dtype=np.float64)
    valid_speed = speed > 0.5
    curvature[valid_speed] = yaw_rate[valid_speed] / speed[valid_speed]
    lateral_accel = speed * yaw_rate
    finite_motion = np.any(np.isfinite(speed), axis=1)
    return {
        "valid_samples": int(np.count_nonzero(finite_motion)),
        "max_abs_heading_change_rad": np.abs(heading[:, -1] - heading[:, 0]),
        "max_abs_step_heading_delta_rad": _nanmax_rows(np.abs(heading_delta)),
        "max_abs_yaw_rate_radps": _nanmax_rows(np.abs(yaw_rate)),
        "max_abs_curvature_1pm": _nanmax_rows(np.abs(curvature)),
        "max_abs_lateral_accel_proxy_mps2": _nanmax_rows(np.abs(lateral_accel)),
        "mean_speed_mps": np.nanmean(speed, axis=1),
        "max_speed_mps": _nanmax_rows(speed),
    }


def _quantiles(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    qs = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(qs[0]),
        "p01": float(qs[1]),
        "p05": float(qs[2]),
        "p25": float(qs[3]),
        "p50": float(qs[4]),
        "p75": float(qs[5]),
        "p95": float(qs[6]),
        "p99": float(qs[7]),
        "max": float(qs[8]),
    }


def _pct(count: int, total: int) -> str:
    return "0.00%" if total == 0 else f"{count / total * 100.0:.2f}%"


def _discover_array_dirs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        for match in matches:
            if match.is_dir() and (match / "ego_agent_future.npy").exists():
                paths.append(match)
            elif match.is_dir():
                paths.extend(path for path in match.glob("p*/shard_*/arrays") if (path / "ego_agent_future.npy").exists())
                paths.extend(path for path in match.glob("*/shard_*/arrays") if (path / "ego_agent_future.npy").exists())
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def _load_memmaps(array_dir: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for path in sorted(array_dir.glob("*.npy")):
        arrays[path.stem] = np.load(path, mmap_mode="r")
    return arrays


def _read_manifest_ids(array_dir: Path, count: int) -> list[str]:
    manifest = array_dir.parent / "manifest_kept.jsonl"
    ids: list[str] = []
    if not manifest.exists():
        return ids
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if len(ids) >= count:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                ids.append(f"{manifest.name}:bad_json:{len(ids)}")
                continue
            sample_id = row.get("sample_id") or row.get("scene_token_hex") or row.get("db_name") or f"row_{len(ids)}"
            ids.append(str(sample_id))
    return ids


def _schema_for_array_dir(array_dir: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "array_dir": str(array_dir),
        "sample_count": int(arrays["ego_agent_future"].shape[0]) if "ego_agent_future" in arrays else 0,
        "fields": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in sorted(arrays.items())
        },
        "missing_required": [name for name in REQUIRED_FIELDS if name not in arrays],
    }


def _append_metric(metrics: dict[str, list[float]], name: str, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    metrics[name].extend(float(value) for value in arr if np.isfinite(value))


def _route_heading_errors(
    future: np.ndarray,
    route_lanes: np.ndarray,
    nearest_indices: np.ndarray,
) -> np.ndarray:
    flat_route_dir = np.asarray(route_lanes[..., 2:4], dtype=np.float64).reshape(route_lanes.shape[0], -1, 2)
    errors = np.full(nearest_indices.shape, np.nan, dtype=np.float64)
    for batch_index in range(nearest_indices.shape[0]):
        for time_index in range(nearest_indices.shape[1]):
            flat_index = int(nearest_indices[batch_index, time_index])
            if flat_index < 0:
                continue
            direction = flat_route_dir[batch_index, flat_index]
            if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < 1e-6:
                continue
            route_heading = math.atan2(float(direction[1]), float(direction[0]))
            errors[batch_index, time_index] = abs(float(_wrap_pi(np.asarray(future[batch_index, time_index, 2] - route_heading))))
    return errors


def _valid_lane_counts(avails: np.ndarray) -> np.ndarray:
    reshaped = np.asarray(avails, dtype=bool).reshape(avails.shape[0], avails.shape[1], -1)
    return np.count_nonzero(np.any(reshaped, axis=2), axis=1)


def _audit_chunk(arrays: dict[str, np.ndarray], slc: slice, *, hz: float) -> dict[str, np.ndarray]:
    future = np.asarray(arrays["ego_agent_future"][slc], dtype=np.float64)
    route_lanes = np.asarray(arrays["route_lanes"][slc])
    route_avails = np.asarray(arrays["route_lanes_avails"][slc], dtype=bool)
    lanes = np.asarray(arrays["lanes"][slc])
    lanes_avails = np.asarray(arrays["lanes_avails"][slc], dtype=bool)
    ego_state = np.asarray(arrays["ego_current_state"][slc], dtype=np.float64)

    time_indices = _future_time_indices(future.shape[1], hz=hz)
    labels = list(time_indices)
    selected_future = future[:, [time_indices[label] for label in labels], :]
    route_distances, nearest_indices = _min_distances_to_route(selected_future[:, :, :2], route_lanes, route_avails)
    heading_errors = _route_heading_errors(selected_future, route_lanes, nearest_indices)

    route_xy = np.asarray(route_lanes[..., :2], dtype=np.float64)
    lane_xy = np.asarray(lanes[..., :2], dtype=np.float64)
    route_valid = route_avails[..., None]
    lane_valid = lanes_avails[..., None]
    route_valid_xy = np.where(route_valid, route_xy, np.nan)
    lane_valid_xy = np.where(lane_valid, lane_xy, np.nan)
    dynamics = _trajectory_dynamics(future, dt=1.0 / hz)

    output: dict[str, np.ndarray] = {
        "route_avail_points": np.count_nonzero(route_avails, axis=(1, 2)),
        "route_avail_ratio": np.mean(route_avails, axis=(1, 2)),
        "route_lane_count": _valid_lane_counts(route_avails),
        "lanes_avail_points": np.count_nonzero(lanes_avails, axis=(1, 2)),
        "lanes_avail_ratio": np.mean(lanes_avails, axis=(1, 2)),
        "lanes_lane_count": _valid_lane_counts(lanes_avails),
        "ego_future_start_norm_m": np.linalg.norm(future[:, 0, :2], axis=1),
        "ego_future_final_norm_m": np.linalg.norm(future[:, -1, :2], axis=1),
        "ego_current_xy_norm_m": np.linalg.norm(ego_state[:, :2], axis=1),
        "ego_current_cos_sin_norm": np.linalg.norm(ego_state[:, 2:4], axis=1) if ego_state.shape[1] >= 4 else np.full((future.shape[0],), np.nan),
        "route_abs_xy_max_m": np.nanmax(np.abs(route_valid_xy), axis=(1, 2, 3)),
        "lanes_abs_xy_max_m": np.nanmax(np.abs(lane_valid_xy), axis=(1, 2, 3)),
    }
    for label_index, label in enumerate(labels):
        output[f"gt_to_route_dist_{label}_m"] = route_distances[:, label_index]
        output[f"route_heading_abs_error_{label}_rad"] = heading_errors[:, label_index]
    for name, values in dynamics.items():
        if name != "valid_samples":
            output[f"gt_{name}"] = values
    return output


def _top_examples(sample_ids: list[str], values: list[float], *, limit: int = 10) -> list[dict[str, Any]]:
    rows = [(value, index) for index, value in enumerate(values) if np.isfinite(value)]
    rows.sort(reverse=True)
    return [{"sample_id": sample_ids[index] if index < len(sample_ids) else f"sample_{index}", "value": float(value)} for value, index in rows[:limit]]


def audit_array_dirs(array_dirs: list[Path], *, max_samples: int, per_shard: int, chunk_size: int, hz: float) -> dict[str, Any]:
    metrics: dict[str, list[float]] = defaultdict(list)
    schemas: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    sample_counts_by_shard: dict[str, int] = {}
    missing_required_by_shard: dict[str, list[str]] = {}
    drivable_fields: set[str] = set()
    total_sampled = 0

    for array_dir in array_dirs:
        if total_sampled >= max_samples:
            break
        arrays = _load_memmaps(array_dir)
        schema = _schema_for_array_dir(array_dir, arrays)
        schemas.append(schema)
        missing_required = schema["missing_required"]
        if missing_required:
            missing_required_by_shard[str(array_dir)] = missing_required
            continue
        for field_name in arrays:
            if any(hint in field_name.lower() for hint in DRIVABLE_FIELD_HINTS):
                drivable_fields.add(field_name)
        shard_count = int(arrays["ego_agent_future"].shape[0])
        take = min(per_shard, shard_count, max_samples - total_sampled)
        if take <= 0:
            continue
        shard_ids = _read_manifest_ids(array_dir, take)
        if len(shard_ids) < take:
            shard_ids.extend(f"{array_dir.parent.name}:row_{index}" for index in range(len(shard_ids), take))
        sample_ids.extend(shard_ids)
        sample_counts_by_shard[str(array_dir)] = take
        for start in range(0, take, chunk_size):
            stop = min(take, start + chunk_size)
            chunk_metrics = _audit_chunk(arrays, slice(start, stop), hz=hz)
            for name, values in chunk_metrics.items():
                _append_metric(metrics, name, values)
        total_sampled += take

    summaries = {name: _quantiles(values) for name, values in sorted(metrics.items())}
    route_thresholds: dict[str, dict[str, int]] = {}
    for name, values in metrics.items():
        if name.startswith("gt_to_route_dist_"):
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            route_thresholds[name] = {f"gt_{threshold}m": int(np.count_nonzero(arr > threshold)) for threshold in (1, 3, 5, 10, 30)}
    confirmations = {
        "gt_future_present": "ego_agent_future" in (schemas[0]["fields"] if schemas else {}),
        "route_present": all(name in (schemas[0]["fields"] if schemas else {}) for name in ("route_lanes", "route_lanes_avails")),
        "lanes_present": all(name in (schemas[0]["fields"] if schemas else {}) for name in ("lanes", "lanes_avails")),
        "ego_current_state_present": "ego_current_state" in (schemas[0]["fields"] if schemas else {}),
        "drivable_map_present": bool(drivable_fields),
    }
    concern_examples = {
        "worst_final_route_dist": _top_examples(sample_ids, metrics.get("gt_to_route_dist_final_m", [])),
        "worst_final_route_heading_error": _top_examples(sample_ids, metrics.get("route_heading_abs_error_final_rad", [])),
        "largest_gt_curvature": _top_examples(sample_ids, metrics.get("gt_max_abs_curvature_1pm", [])),
        "largest_gt_lateral_accel_proxy": _top_examples(sample_ids, metrics.get("gt_max_abs_lateral_accel_proxy_mps2", [])),
    }
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "array_dirs": [str(path) for path in array_dirs],
        "schemas": schemas,
        "sampled_total": total_sampled,
        "sample_counts_by_shard": sample_counts_by_shard,
        "missing_required_by_shard": missing_required_by_shard,
        "drivable_fields": sorted(drivable_fields),
        "confirmations": confirmations,
        "summaries": summaries,
        "route_threshold_counts": route_thresholds,
        "concern_examples": concern_examples,
    }


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "count", "mean", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric, summary in report["summaries"].items():
            row = {"metric": metric, **{field: summary.get(field, "") for field in fields if field != "metric"}}
            writer.writerow(row)


def _fmt_summary(summary: dict[str, Any]) -> str:
    if not summary or summary.get("count", 0) == 0:
        return "count=0"
    return (
        f"n={summary['count']} mean={summary['mean']:.4g} "
        f"p50={summary['p50']:.4g} p95={summary['p95']:.4g} p99={summary['p99']:.4g} max={summary['max']:.4g}"
    )


def write_markdown(path: Path, report: dict[str, Any], *, json_path: Path, csv_path: Path) -> None:
    lines = [
        "# Array Contract Audit",
        "",
        "## Outputs",
        f"- JSON: `{json_path}`",
        f"- CSV: `{csv_path}`",
        "",
        "## Inputs",
        f"- Shards discovered: {len(report['array_dirs'])}",
        f"- Samples audited: {report['sampled_total']}",
        "",
        "## Fields",
    ]
    if report["schemas"]:
        first = report["schemas"][0]
        for name, meta in first["fields"].items():
            lines.append(f"- `{name}`: shape={meta['shape']} dtype={meta['dtype']}")
    if report["missing_required_by_shard"]:
        lines.extend(["", "Missing required fields by shard:"])
        for shard, missing in report["missing_required_by_shard"].items():
            lines.append(f"- `{shard}`: {', '.join(missing)}")
    lines.extend(["", "## Key Metrics"])
    key_metrics = [
        "gt_to_route_dist_1s_m",
        "gt_to_route_dist_3s_m",
        "gt_to_route_dist_5s_m",
        "gt_to_route_dist_final_m",
        "route_avail_points",
        "route_avail_ratio",
        "route_heading_abs_error_final_rad",
        "gt_max_abs_heading_change_rad",
        "gt_max_abs_curvature_1pm",
        "gt_max_abs_lateral_accel_proxy_mps2",
        "ego_future_start_norm_m",
        "ego_current_xy_norm_m",
        "route_abs_xy_max_m",
    ]
    for metric in key_metrics:
        lines.append(f"- `{metric}`: {_fmt_summary(report['summaries'].get(metric, {}))}")
    lines.extend(["", "## Route Distance Thresholds"])
    for metric, counts in sorted(report["route_threshold_counts"].items()):
        total = report["summaries"].get(metric, {}).get("count", 0)
        pieces = [f"{threshold}={count} ({_pct(count, total)})" for threshold, count in counts.items()]
        lines.append(f"- `{metric}`: " + ", ".join(pieces))
    lines.extend(["", "## Confirmation Status"])
    for name, ok in report["confirmations"].items():
        lines.append(f"- `{name}`: {ok}")
    if not report["confirmations"]["drivable_map_present"]:
        lines.append("- GT drivable cannot be confirmed: no drivable/map-mask array field was present in the sampled cache arrays.")
    lines.extend(["", "## Worst Examples"])
    for name, examples in report["concern_examples"].items():
        if not examples:
            lines.append(f"- `{name}`: none")
            continue
        formatted = "; ".join(f"{row['sample_id']}={row['value']:.4g}" for row in examples[:5])
        lines.append(f"- `{name}`: {formatted}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arrays-glob",
        action="append",
        default=[],
        help="Glob/path to arrays dirs or a cache root containing p*/shard_*/arrays; may be repeated.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--per-shard", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--hz", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patterns = args.arrays_glob or ["outputs/cache/training_arrays/vegas_200k/p*/shard_*/arrays"]
    array_dirs = _discover_array_dirs(patterns)
    if not array_dirs:
        print("ERROR: no array-backed shard directories found")
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"array_contract_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    json_path = args.out_dir / f"{stem}.json"
    csv_path = args.out_dir / f"{stem}.csv"
    md_path = args.out_dir / f"{stem}.md"
    report = audit_array_dirs(
        array_dirs,
        max_samples=args.max_samples,
        per_shard=args.per_shard,
        chunk_size=args.chunk_size,
        hz=args.hz,
    )
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, report)
    write_markdown(md_path, report, json_path=json_path, csv_path=csv_path)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Audited {report['sampled_total']} samples from {len(array_dirs)} array dirs")
    if not report["confirmations"]["drivable_map_present"]:
        print("NOTE: GT drivable cannot be confirmed; no drivable/map-mask arrays were present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
