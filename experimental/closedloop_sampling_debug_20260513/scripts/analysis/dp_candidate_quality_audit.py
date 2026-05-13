#!/usr/bin/env python3
"""Audit closed-loop candidate pool quality and modal diversity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


CSV_FIELDS = [
    "token",
    "tick",
    "dump_npz",
    "candidate_count",
    "selected_index",
    "endpoint_pairwise_mean",
    "endpoint_pairwise_max",
    "trajectory_pairwise_mean",
    "final_heading_spread",
    "mean_curvature_spread",
    "effective_modes",
    "mode_entropy",
    "safe_candidate_count",
    "safe_mode_count",
    "selected_is_safe",
    "best_available_offroad_count",
    "selector_miss_safe_candidate",
]

CANDIDATE_KEYS = (
    "candidate_xyh",
    "candidates_local_xyh",
    "candidate_trajectories",
    "candidates",
    "sample_trajectories",
)
SELECTED_KEYS = (
    "selected_xyh",
    "selected_local_xyh",
    "closed_loop_y",
    "selected_trajectory",
    "diffusion_plan",
)
SELECTED_INDEX_KEYS = (
    "selected_index",
    "selected_candidate_index",
    "best_index",
)
OFFROAD_COUNT_KEYS = (
    "candidate_corner_offroad_counts",
    "candidate_offroad_counts",
    "candidate_center_offroad_counts",
    "candidate_rear_offroad_counts",
)
DRIVABLE_KEYS = (
    "candidate_corner_drivable",
    "candidate_center_drivable",
    "candidate_rear_drivable",
    "candidate_drivable",
)
LATERAL_KEYS = (
    "candidate_route_projection_final_lateral_error_m",
    "candidate_final_lateral_error_m",
    "candidate_lateral_offset_m",
)
CLEARANCE_KEYS = (
    "candidate_dynamic_min_clearance_m",
    "candidate_static_min_clearance_m",
)


def _nan() -> float:
    return float("nan")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    except Exception:
        return {}


def _first_present(arrays: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in arrays:
            return np.asarray(arrays[key])
    return None


def _as_candidates(raw: np.ndarray | None) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] < 2:
        return None
    if arr.shape[2] == 2:
        zeros = np.zeros((*arr.shape[:2], 1), dtype=arr.dtype)
        arr = np.concatenate([arr[:, :, :2], zeros], axis=2)
    return arr[:, :, :3].astype(float)


def _as_selected(raw: np.ndarray | None) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return None
    if arr.shape[1] == 2:
        arr = np.column_stack([arr[:, :2], np.zeros(arr.shape[0], dtype=arr.dtype)])
    return arr[:, :3].astype(float)


def _scalar_int(value: Any) -> int | None:
    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return None
        out = int(arr[0])
        return out
    except Exception:
        return None


def _parse_tick(path: Path) -> int | str:
    match = re.search(r"tick[_-]?(\d+)", path.stem)
    return int(match.group(1)) if match else ""


def _parse_token(path: Path) -> str:
    match = re.match(r"(?P<token>[0-9a-fA-F]{16})_tick", path.stem)
    if match:
        return match.group("token")
    return ""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_selector_trace(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    trace: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            for key in ("dump_stem", "stem", "npz_stem"):
                if key in item:
                    trace[str(item[key])] = item
            tick = item.get("tick", item.get("tick_index"))
            if tick is not None:
                trace[f"tick:{tick}"] = item
    return trace


def _trace_for(path: Path, tick: int | str, trace: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return trace.get(path.stem, trace.get(f"tick:{tick}", {}))


def _selected_index(
    arrays: dict[str, np.ndarray],
    candidates: np.ndarray | None,
    selected: np.ndarray | None,
    trace_item: dict[str, Any],
) -> int | None:
    for key in SELECTED_INDEX_KEYS:
        if key in arrays:
            idx = _scalar_int(arrays[key])
            if idx is not None:
                return idx
        if key in trace_item:
            idx = _scalar_int(trace_item[key])
            if idx is not None:
                return idx
    if candidates is None or selected is None:
        return None
    final_l2 = np.linalg.norm(candidates[:, -1, :2] - selected[-1, :2][None, :], axis=1)
    return int(np.argmin(final_l2)) if final_l2.size else None


def _pairwise(values: np.ndarray, reducer: str) -> float:
    n = values.shape[0]
    if n < 2:
        return 0.0 if n == 1 else _nan()
    distances: list[float] = []
    for left in range(n - 1):
        for right in range(left + 1, n):
            if reducer == "endpoint":
                distances.append(float(np.linalg.norm(values[left] - values[right])))
            else:
                distances.append(float(np.mean(np.linalg.norm(values[left] - values[right], axis=1))))
    return float(np.mean(distances)) if distances else _nan()


def _endpoint_pairwise(candidates: np.ndarray | None) -> tuple[float, float]:
    if candidates is None:
        return _nan(), _nan()
    finals = candidates[:, -1, :2]
    n = finals.shape[0]
    if n == 1:
        return 0.0, 0.0
    distances = [
        float(np.linalg.norm(finals[left] - finals[right]))
        for left in range(n - 1)
        for right in range(left + 1, n)
    ]
    return float(np.mean(distances)), float(np.max(distances))


def _trajectory_pairwise_mean(candidates: np.ndarray | None) -> float:
    if candidates is None:
        return _nan()
    return _pairwise(candidates[:, :, :2], "trajectory")


def _mean_abs_curvature(path_xyh: np.ndarray) -> float:
    if path_xyh.shape[0] < 3:
        return 0.0
    heading = np.unwrap(path_xyh[:, 2])
    ds = np.linalg.norm(np.diff(path_xyh[:, :2], axis=0), axis=1)
    dtheta = np.abs(np.diff(heading))
    denom = ds[: dtheta.size]
    valid = denom > 1e-6
    if not np.any(valid):
        return 0.0
    return float(np.mean(dtheta[valid] / denom[valid]))


def _candidate_curvatures(candidates: np.ndarray | None) -> np.ndarray | None:
    if candidates is None:
        return None
    return np.asarray([_mean_abs_curvature(candidate) for candidate in candidates], dtype=float)


def _spread(values: np.ndarray | None) -> float:
    if values is None or values.size == 0:
        return _nan()
    return float(np.max(values) - np.min(values))


def _lateral_feature(arrays: dict[str, np.ndarray], count: int) -> np.ndarray | None:
    raw = _first_present(arrays, LATERAL_KEYS)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size < count:
        return None
    return arr[:count]


def _cluster_modes(features: np.ndarray | None) -> tuple[int | None, float, np.ndarray | None]:
    if features is None or features.size == 0:
        return None, _nan(), None
    order = np.lexsort(tuple(features[:, idx] for idx in reversed(range(features.shape[1]))))
    centers: list[np.ndarray] = []
    counts: list[int] = []
    labels = np.full(features.shape[0], -1, dtype=int)
    for idx in order:
        feature = features[idx]
        chosen = None
        for cluster_idx, center in enumerate(centers):
            if float(np.linalg.norm(feature - center)) <= 1.0:
                chosen = cluster_idx
                break
        if chosen is None:
            centers.append(feature.copy())
            counts.append(1)
            chosen = len(centers) - 1
        else:
            counts[chosen] += 1
            centers[chosen] = centers[chosen] + (feature - centers[chosen]) / counts[chosen]
        labels[idx] = chosen
    probs = np.asarray(counts, dtype=float) / float(np.sum(counts))
    entropy = float(-np.sum(probs * np.log(probs))) if probs.size else _nan()
    return len(counts), entropy, labels


def _mode_features(candidates: np.ndarray | None, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if candidates is None:
        return None, None
    curvatures = _candidate_curvatures(candidates)
    final_heading = np.unwrap(candidates[:, -1, 2])
    parts = [
        candidates[:, -1, 0] / 2.0,
        candidates[:, -1, 1] / 2.0,
        final_heading / 0.35,
        curvatures / 0.10,
    ]
    lateral = _lateral_feature(arrays, candidates.shape[0])
    if lateral is not None:
        parts.append(lateral / 1.0)
    return np.column_stack(parts), curvatures


def _offroad_counts(arrays: dict[str, np.ndarray], count: int) -> np.ndarray | None:
    raw = _first_present(arrays, OFFROAD_COUNT_KEYS)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size < count:
        return None
    return arr[:count]


def _candidate_safe_mask(arrays: dict[str, np.ndarray], count: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    known = np.zeros(count, dtype=bool)
    safe = np.ones(count, dtype=bool)
    offroad = _offroad_counts(arrays, count)
    if offroad is not None:
        known |= np.isfinite(offroad)
        safe &= offroad == 0
    for key in DRIVABLE_KEYS:
        if key not in arrays:
            continue
        arr = np.asarray(arrays[key]).astype(bool)
        if arr.shape[0] != count:
            continue
        flat = arr.reshape(count, -1)
        known |= True
        safe &= np.all(flat, axis=1)
    for key in CLEARANCE_KEYS:
        if key not in arrays:
            continue
        arr = np.asarray(arrays[key], dtype=float).reshape(-1)
        if arr.size < count:
            continue
        values = arr[:count]
        finite = np.isfinite(values)
        known |= finite
        safe &= np.logical_or(~finite, values >= 0.0)
    if not np.any(known):
        return None, offroad
    safe &= known
    return safe, offroad


def _format(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "NaN"
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def _row_for_npz(path: Path, selector_trace: dict[str, dict[str, Any]]) -> dict[str, Any]:
    arrays = _load_npz(path)
    candidates = _as_candidates(_first_present(arrays, CANDIDATE_KEYS))
    selected = _as_selected(_first_present(arrays, SELECTED_KEYS))
    count = int(candidates.shape[0]) if candidates is not None else 0
    tick = _parse_tick(path)
    token = _parse_token(path)
    trace_item = _trace_for(path, tick, selector_trace)
    if not token and trace_item.get("token") is not None:
        token = str(trace_item.get("token"))
    selected_index = _selected_index(arrays, candidates, selected, trace_item)
    if selected_index is not None and (selected_index < 0 or selected_index >= count):
        selected_index = None

    endpoint_mean, endpoint_max = _endpoint_pairwise(candidates)
    trajectory_mean = _trajectory_pairwise_mean(candidates)
    curvatures: np.ndarray | None
    features, curvatures = _mode_features(candidates, arrays)
    effective_modes, entropy, labels = _cluster_modes(features)
    final_heading_spread = _spread(np.unwrap(candidates[:, -1, 2]) if candidates is not None else None)
    curvature_spread = _spread(curvatures)
    safe_mask, offroad_counts = _candidate_safe_mask(arrays, count)
    if safe_mask is None:
        safe_candidate_count: int | None = None
        selected_is_safe: bool | str = "unknown"
        selector_miss: bool | str = "unknown"
        safe_mode_count: int | None = None
    else:
        safe_candidate_count = int(np.count_nonzero(safe_mask))
        if selected_index is None:
            selected_is_safe = "unknown"
            selector_miss = "unknown"
        else:
            selected_is_safe = bool(safe_mask[selected_index])
            selector_miss = bool(safe_candidate_count > 0 and not selected_is_safe)
        if labels is None:
            safe_mode_count = None
        else:
            safe_mode_count = int(len(set(int(label) for label in labels[safe_mask])))

    best_offroad = float(np.min(offroad_counts)) if offroad_counts is not None and offroad_counts.size else _nan()
    return {
        "token": token,
        "tick": tick,
        "dump_npz": str(path),
        "candidate_count": count,
        "selected_index": selected_index,
        "endpoint_pairwise_mean": endpoint_mean,
        "endpoint_pairwise_max": endpoint_max,
        "trajectory_pairwise_mean": trajectory_mean,
        "final_heading_spread": final_heading_spread,
        "mean_curvature_spread": curvature_spread,
        "effective_modes": effective_modes,
        "mode_entropy": entropy,
        "safe_candidate_count": safe_candidate_count,
        "safe_mode_count": safe_mode_count,
        "selected_is_safe": selected_is_safe,
        "best_available_offroad_count": best_offroad,
        "selector_miss_safe_candidate": selector_miss,
    }


def analyze(dump_dir: Path, selector_trace: Path | None = None) -> list[dict[str, str]]:
    trace = _load_selector_trace(selector_trace)
    rows = [_row_for_npz(path, trace) for path in sorted(dump_dir.rglob("*.npz"))]
    rows.sort(key=lambda row: (str(row.get("token") or ""), row["tick"] if isinstance(row["tick"], int) else 10**12, str(row["dump_npz"])))
    return [{field: _format(row.get(field)) for field in CSV_FIELDS} for row in rows]


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, str]], path: Path, dump_dir: Path, selector_trace: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    miss_count = sum(row["selector_miss_safe_candidate"] == "True" for row in rows)
    by_token: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_token.setdefault(row.get("token") or "<unknown>", []).append(row)
    lines = [
        "# DP candidate quality audit",
        "",
        f"- dump_dir: `{dump_dir}`",
        f"- selector_trace: `{selector_trace or '<not provided>'}`",
        f"- ticks: {len(rows)}",
        f"- selector_miss_safe_candidate ticks: {miss_count}",
        "",
        "## By Token",
    ]
    for token, token_rows in sorted(by_token.items()):
        modes = [_to_float(row.get("effective_modes")) for row in token_rows]
        safe = [_to_float(row.get("safe_candidate_count")) for row in token_rows]
        endpoint = [_to_float(row.get("endpoint_pairwise_mean")) for row in token_rows]
        entropy = [_to_float(row.get("mode_entropy")) for row in token_rows]
        best_offroad = [_to_float(row.get("best_available_offroad_count")) for row in token_rows]
        token_miss = sum(row["selector_miss_safe_candidate"] == "True" for row in token_rows)
        selected_safe = sum(row["selected_is_safe"] == "True" for row in token_rows)
        lines.append(
            "- token={token} ticks={ticks} mode_mean={mode_mean:.3g} mode_p50={mode_p50:.3g} "
            "entropy_mean={entropy_mean:.3g} safe_mean={safe_mean:.3g} selected_safe_ticks={selected_safe} "
            "miss_ticks={miss} endpoint_mean={endpoint_mean:.3g} best_offroad_p50={best_offroad_p50:.3g}".format(
                token=token,
                ticks=len(token_rows),
                mode_mean=_mean(modes),
                mode_p50=_median(modes),
                entropy_mean=_mean(entropy),
                safe_mean=_mean(safe),
                selected_safe=selected_safe,
                miss=token_miss,
                endpoint_mean=_mean(endpoint),
                best_offroad_p50=_median(best_offroad),
            )
        )
    lines.extend(["", "## First Ticks"])
    for row in rows[:50]:
        lines.append(
            "- token={token} tick={tick} cand={candidate_count} modes={effective_modes} "
            "safe={safe_candidate_count} selected_safe={selected_is_safe} miss={selector_miss_safe_candidate}".format(**row)
        )
    if not rows:
        lines.append("- No `.npz` dumps found.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _mean(values: list[float]) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite else float("nan")


def _median(values: list[float]) -> float:
    finite = _finite(values)
    return float(np.median(finite)) if finite else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", required=True, type=Path)
    parser.add_argument("--selector-trace", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()
    if not args.dump_dir.exists():
        raise SystemExit(f"--dump-dir does not exist: {args.dump_dir}")
    rows = analyze(args.dump_dir, args.selector_trace)
    if args.out_csv is not None:
        write_csv(rows, args.out_csv)
    if args.out_md is not None:
        write_md(rows, args.out_md, args.dump_dir, args.selector_trace)
    if args.out_csv is None and args.out_md is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"audited {len(rows)} ticks", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
