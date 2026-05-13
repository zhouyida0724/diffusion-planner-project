#!/usr/bin/env python3
"""Offline analyzer for closed-loop fly debug dumps.

The analyzer intentionally keeps classifications conservative.  It summarizes
per-tick JSON/NPZ dumps and optional nuPlan `corners_in_drivable_area` metrics
without claiming a root cause.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - exercised only in stripped envs.
    pd = None  # type: ignore[assignment]


CSV_FIELDS = [
    "token",
    "group",
    "tick",
    "iteration",
    "timestamp",
    "selected_path_length",
    "final_x",
    "final_y",
    "max_abs_y",
    "max_step_m",
    "route_lane_count",
    "has_candidates",
    "candidate_count",
    "selected_vs_candidates_min_final_l2",
    "selected_vs_candidates_mean_final_l2",
    "selected_candidate_index",
    "selected_candidate_offroad_count",
    "best_candidate_offroad_count",
    "candidate_offroad_count_min",
    "candidate_offroad_count_mean",
    "selected_corner_offroad_count",
    "metric_first_offroad_frame",
    "metric_last_offroad_frame",
    "metric_offroad_frames",
    "classification",
    "missing_fields",
    "dump_json",
    "dump_npz",
]

SELECTED_KEYS = (
    "selected_local_xyh",
    "closed_loop_y",
    "selected_xyh",
    "selected_trajectory",
    "selected_path",
    "selected_xy",
    "diffusion_plan",
)
CANDIDATE_KEYS = (
    "candidates_local_xyh",
    "candidate_xyh",
    "candidate_trajectories",
    "candidates",
    "candidate_paths",
    "candidate_xy",
    "sample_trajectories",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _safe_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_json_error": str(exc)}


def _load_npz(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    except Exception:
        return {}


def _first_present(arrays: dict[str, np.ndarray], keys: tuple[str, ...]) -> tuple[str | None, np.ndarray | None]:
    for key in keys:
        if key in arrays:
            return key, np.asarray(arrays[key])
    return None, None


def _as_xy(path: np.ndarray | None) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.asarray(path)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return None
    return arr[:, :2].astype(float)


def _as_candidates(candidates: np.ndarray | None) -> np.ndarray | None:
    if candidates is None:
        return None
    arr = np.asarray(candidates)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] < 2:
        return None
    return arr[:, :, :2].astype(float)


def _count_route_lanes(arrays: dict[str, np.ndarray]) -> int | str:
    avails = arrays.get("route_lanes_avails")
    if avails is not None:
        arr = np.asarray(avails)
        if arr.ndim >= 2:
            flat = arr.reshape(arr.shape[0], -1)
            return int(np.sum(np.any(flat > 0, axis=1)))
    lanes = arrays.get("route_lanes")
    if lanes is not None:
        arr = np.asarray(lanes)
        if arr.ndim >= 2:
            return int(arr.shape[0])
    return ""


def _path_stats(xy: np.ndarray | None) -> dict[str, Any]:
    if xy is None:
        return {
            "selected_path_length": 0,
            "final_x": "",
            "final_y": "",
            "max_abs_y": "",
            "max_step_m": "",
            "path_is_reasonable": False,
        }
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.asarray([], dtype=float)
    max_step = float(np.max(steps)) if steps.size else 0.0
    finite = bool(np.all(np.isfinite(xy)))
    return {
        "selected_path_length": int(len(xy)),
        "final_x": float(xy[-1, 0]),
        "final_y": float(xy[-1, 1]),
        "max_abs_y": float(np.max(np.abs(xy[:, 1]))),
        "max_step_m": max_step,
        "path_is_reasonable": bool(finite and max_step <= 20.0 and len(xy) >= 2),
    }


def _candidate_stats(
    selected_xy: np.ndarray | None,
    candidates_xy: np.ndarray | None,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "has_candidates": False,
        "candidate_count": 0,
        "selected_vs_candidates_min_final_l2": "",
        "selected_vs_candidates_mean_final_l2": "",
        "selected_candidate_index": "",
        "selected_candidate_offroad_count": "",
        "best_candidate_offroad_count": "",
        "candidate_offroad_count_min": "",
        "candidate_offroad_count_mean": "",
        "candidate_has_safer_option": False,
    }
    if candidates_xy is not None:
        out["has_candidates"] = True
        out["candidate_count"] = int(candidates_xy.shape[0])
        if selected_xy is not None:
            selected_final = selected_xy[-1, :2]
            candidate_finals = candidates_xy[:, -1, :2]
            final_l2 = np.linalg.norm(candidate_finals - selected_final[None, :], axis=1)
            out["selected_vs_candidates_min_final_l2"] = float(np.min(final_l2))
            out["selected_vs_candidates_mean_final_l2"] = float(np.mean(final_l2))

    counts = arrays.get("candidate_corner_offroad_counts")
    if counts is None:
        counts = arrays.get("candidate_offroad_counts")
    if counts is not None:
        flat_counts = np.asarray(counts).reshape(-1).astype(float)
        if flat_counts.size:
            selected_index = int(meta.get("selected_candidate_index", meta.get("best_index", 0)) or 0)
            if selected_index < 0 or selected_index >= flat_counts.size:
                selected_index = 0
            selected_count = float(flat_counts[selected_index])
            best_count = float(np.min(flat_counts))
            out["selected_candidate_index"] = selected_index
            out["selected_candidate_offroad_count"] = int(selected_count) if selected_count.is_integer() else selected_count
            out["best_candidate_offroad_count"] = int(best_count) if best_count.is_integer() else best_count
            out["candidate_offroad_count_min"] = out["best_candidate_offroad_count"]
            out["candidate_offroad_count_mean"] = float(np.mean(flat_counts))
            out["candidate_has_safer_option"] = bool(best_count < selected_count)
    return out


def _selected_corner_offroad_count(arrays: dict[str, np.ndarray]) -> int | str:
    for key in ("selected_corner_drivable", "selected_drivable", "selected_path_drivable"):
        if key in arrays:
            flags = np.asarray(arrays[key]).astype(bool)
            return int(np.size(flags) - np.count_nonzero(flags))
    return ""


def _parse_tick_from_name(stem: str) -> int | str:
    match = re.search(r"tick[_-]?(\d+)", stem)
    return int(match.group(1)) if match else ""


def _parse_iter_from_name(stem: str) -> int | str:
    match = re.search(r"iter[_-]?(\d+)", stem)
    return int(match.group(1)) if match else ""


def _find_dump_pairs(dump_dir: Path) -> list[tuple[Path, Path | None]]:
    json_by_key = {path.relative_to(dump_dir).with_suffix(""): path for path in dump_dir.rglob("*.json")}
    npz_by_key = {path.relative_to(dump_dir).with_suffix(""): path for path in dump_dir.rglob("*.npz")}
    keys = sorted(set(json_by_key) | set(npz_by_key))
    return [
        (json_by_key.get(key, dump_dir / key.with_suffix(".json")), npz_by_key.get(key))
        for key in keys
    ]


def _token_groups(tokens_csv: Path | None) -> dict[str, str]:
    if tokens_csv is None or not tokens_csv.exists():
        return {}
    with tokens_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "token" not in reader.fieldnames:
            raise SystemExit(f"--tokens-csv must contain a token column: {tokens_csv}")
        group_col = "group" if "group" in reader.fieldnames else None
        out: dict[str, str] = {}
        for row in reader:
            token = str(row.get("token", "")).strip()
            if token:
                out[token] = str(row.get(group_col, "")).strip() if group_col else ""
        return out


def _load_corner_metrics(metrics_dir: Path | None) -> dict[str, dict[str, int]]:
    if metrics_dir is None:
        return {}
    path = metrics_dir / "corners_in_drivable_area.parquet"
    if not path.exists() or pd is None:
        return {}
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return {}
    if "scenario_name" not in frame.columns or "time_series_values" not in frame.columns:
        return {}
    metrics: dict[str, dict[str, int]] = {}
    for _, row in frame.iterrows():
        token = str(row["scenario_name"])
        values = list(row["time_series_values"])
        off_frames = [idx for idx, value in enumerate(values) if not bool(value)]
        metrics[token] = {
            "metric_first_offroad_frame": int(off_frames[0]) if off_frames else -1,
            "metric_last_offroad_frame": int(off_frames[-1]) if off_frames else -1,
            "metric_offroad_frames": int(len(off_frames)),
        }
    return metrics


def _classify(row: dict[str, Any], missing_fields: list[str]) -> str:
    if missing_fields:
        return "feature_or_dump_missing"
    if bool(row.get("candidate_has_safer_option", False)):
        return "candidate_has_safer_option"
    metric_offroad = int(row.get("metric_offroad_frames") or 0)
    selected_corner_count = row.get("selected_corner_offroad_count", "")
    has_selected_drivable = selected_corner_count != ""
    if metric_offroad > 0 and bool(row.get("path_is_reasonable", False)):
        selected_dump_offroad = int(selected_corner_count or 0) if has_selected_drivable else 0
        if selected_dump_offroad == 0:
            return "corner_margin_suspect"
    if not has_selected_drivable and metric_offroad <= 0:
        return "selected_already_offroad_unknown"
    return "needs_manual_review"


def analyze(dump_dir: Path, metrics_dir: Path | None, tokens_csv: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = _token_groups(tokens_csv)
    corner_metrics = _load_corner_metrics(metrics_dir)
    rows: list[dict[str, Any]] = []

    for json_path, npz_path in _find_dump_pairs(dump_dir):
        meta = _read_json(json_path) if json_path.exists() else {}
        arrays = _load_npz(npz_path)
        _, selected_raw = _first_present(arrays, SELECTED_KEYS)
        _, candidates_raw = _first_present(arrays, CANDIDATE_KEYS)
        selected_xy = _as_xy(selected_raw)
        candidates_xy = _as_candidates(candidates_raw)
        token = str(meta.get("token", meta.get("scenario_token", meta.get("scenario_name", ""))))
        tick = meta.get("tick", meta.get("tick_index", _parse_tick_from_name(json_path.stem)))
        iteration = meta.get("iteration", meta.get("iteration_index", _parse_iter_from_name(json_path.stem)))
        timestamp = meta.get("timestamp", meta.get("timestamp_us", ""))
        missing_fields: list[str] = []
        if not token:
            missing_fields.append("token")
        if npz_path is None or not npz_path.exists():
            missing_fields.append("npz")
        if selected_xy is None:
            missing_fields.append("selected_path")

        row: dict[str, Any] = {
            "token": token,
            "group": groups.get(token, ""),
            "tick": tick,
            "iteration": iteration,
            "timestamp": timestamp,
            "route_lane_count": _count_route_lanes(arrays),
            "selected_corner_offroad_count": _selected_corner_offroad_count(arrays),
            "metric_first_offroad_frame": "",
            "metric_last_offroad_frame": "",
            "metric_offroad_frames": "",
            "missing_fields": ",".join(missing_fields),
            "dump_json": str(json_path),
            "dump_npz": str(npz_path) if npz_path is not None else "",
        }
        row.update(_path_stats(selected_xy))
        row.update(_candidate_stats(selected_xy, candidates_xy, arrays, meta))
        if token in corner_metrics:
            row.update(corner_metrics[token])
        row["classification"] = _classify(row, missing_fields)
        row.pop("candidate_has_safer_option", None)
        row.pop("path_is_reasonable", None)
        rows.append({key: _safe_scalar(row.get(key, "")) for key in CSV_FIELDS})

    rows.sort(key=lambda row: (int(row.get("tick") or -1), str(row.get("token", "")), str(row.get("dump_npz", ""))))
    summary = _build_summary(rows, dump_dir, metrics_dir, tokens_csv)
    return rows, summary


def _build_summary(rows: list[dict[str, Any]], dump_dir: Path, metrics_dir: Path | None, tokens_csv: Path | None) -> dict[str, Any]:
    classification_counts = Counter(str(row["classification"]) for row in rows)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group = str(row.get("group") or "ungrouped")
        group_counts[group][str(row["classification"])] += 1
    return {
        "dump_dir": str(dump_dir),
        "metrics_dir": str(metrics_dir) if metrics_dir is not None else "",
        "tokens_csv": str(tokens_csv) if tokens_csv is not None else "",
        "total_ticks": len(rows),
        "classification_counts": dict(sorted(classification_counts.items())),
        "group_counts": {group: dict(sorted(counts.items())) for group, counts in sorted(group_counts.items())},
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(rows: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Closed-loop fly dump analysis",
        "",
        "## Inputs",
        f"- dump_dir: `{summary['dump_dir']}`",
        f"- metrics_dir: `{summary['metrics_dir'] or '<not provided>'}`",
        f"- tokens_csv: `{summary['tokens_csv'] or '<not provided>'}`",
        "",
        "## Classification rules",
        "- feature_or_dump_missing: critical JSON/NPZ or selected trajectory fields are absent.",
        "- candidate_has_safer_option: candidate offroad counts exist and the best candidate has fewer offroad counts than selected.",
        "- corner_margin_suspect: nuPlan corners metric shows offroad while the selected trajectory is finite/continuous and dump drivable evidence does not show offroad.",
        "- selected_already_offroad_unknown: no dump drivable flags and no corners metric are available, so selected offroad status cannot be judged.",
        "- needs_manual_review: fallback bucket; no root cause is asserted.",
        "",
        "## Counts",
    ]
    for name, count in summary["classification_counts"].items():
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Group Differences"])
    group_counts = summary.get("group_counts", {})
    if group_counts:
        for group, counts in group_counts.items():
            count_text = ", ".join(f"{name}={count}" for name, count in counts.items())
            lines.append(f"- {group}: {count_text}")
    else:
        lines.append("- No token groups provided.")

    suspect_order = {
        "candidate_has_safer_option": 0,
        "corner_margin_suspect": 1,
        "feature_or_dump_missing": 2,
        "selected_already_offroad_unknown": 3,
        "needs_manual_review": 4,
    }
    ranked = sorted(
        rows,
        key=lambda row: (
            suspect_order.get(str(row["classification"]), 99),
            -(int(row.get("metric_offroad_frames") or 0) if str(row.get("metric_offroad_frames", "")).strip() else 0),
            str(row.get("token", "")),
            int(row.get("tick") or -1),
        ),
    )
    lines.extend(["", "## Top Suspect Ticks"])
    for row in ranked[:20]:
        lines.append(
            "- token={token} tick={tick} group={group} class={classification} "
            "metric_off={metric_offroad_frames} cand={candidate_count} max_abs_y={max_abs_y}".format(**row)
        )
    if not rows:
        lines.append("- No dump ticks found.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", required=True, type=Path, help="Closed-loop debug dump directory.")
    parser.add_argument("--metrics-dir", type=Path, default=None, help="Optional nuPlan metrics directory.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--tokens-csv", type=Path, default=None, help="Optional CSV with token column and optional group column.")
    args = parser.parse_args()

    if not args.dump_dir.exists():
        raise SystemExit(f"--dump-dir does not exist: {args.dump_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = analyze(args.dump_dir, args.metrics_dir, args.tokens_csv)
    _write_csv(rows, args.out_dir / "closedloop_fly_summary.csv")
    (args.out_dir / "closedloop_fly_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_report(rows, summary, args.out_dir / "closedloop_fly_report.md")
    print(f"Wrote {len(rows)} tick rows to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
