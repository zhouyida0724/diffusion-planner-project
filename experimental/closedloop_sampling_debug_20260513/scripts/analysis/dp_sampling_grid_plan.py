#!/usr/bin/env python3
"""Plan closed-loop diffusion sampling grid runs without launching simulation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sqlite3
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_SCRIPT = Path("scripts/sim/run_diffusion_simulation.py")
DEFAULT_INPUT_CSV = Path("outputs/eval/closedloop_regression_20260506/closedloop_diverse60_scenarios.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/eval/dp_sampling_grid_plan")
DEFAULT_DATA_ROOT = "/media/zhouyida/新加卷1/nuplan_datasets/data/cache/test"
SAMPLING_STEPS = [2, 3, 5, 10]
NOISE_SCALES = [0.25, 0.5, 0.75, 1.0, 1.5]
SELECTOR_FIXED_SAMPLES = 10
DEFAULT_SCENARIO_COUNT = 16
GEO_CELL_M = 100.0
DPM_VARIANTS = [
    ("default", {}),
    ("order_1", {"DP_DPM_ORDER": "1"}),
    ("order_3", {"DP_DPM_ORDER": "3"}),
    ("skip_time_uniform", {"DP_DPM_SKIP": "time_uniform"}),
    ("method_singlestep", {"DP_DPM_METHOD": "singlestep"}),
    ("denoise_to_zero_0", {"DP_DPM_DENOISE_TO_ZERO": "0"}),
]
PRIMARY_BUCKET_TARGETS = {
    "left": 4,
    "right": 4,
    "straight_cruise": 4,
    "straight_low_speed_stop_proxy": 4,
}
STOP_TAG_WORDS = ("all_way_stop", "stopline", "stop_sign", "stationary", "stopping", "waiting")
INTERACTION_TAG_WORDS = ("near_", "crosswalk", "construction", "pedestrian", "long_vehicle")


def _float_value(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return None


def _first_present(row: dict[str, str], names: Sequence[str]) -> float | None:
    for name in names:
        value = _float_value(row, name)
        if value is not None:
            return value
    return None


def _split_tags(row: dict[str, str]) -> set[str]:
    return {tag.strip().lower() for tag in row.get("nuplan_tags", "").split(";") if tag.strip()}


def _has_any_tag(tags: set[str], words: Sequence[str]) -> bool:
    return any(any(word in tag for word in words) for tag in tags)


def behavior_bucket(row: dict[str, str]) -> str:
    text = " ".join(str(row.get(key, "")).lower() for key in ("group", "scenario_type"))
    tags = _split_tags(row)
    yaw = _float_value(row, "yaw_change_rad") or 0.0
    travel = _float_value(row, "travel_m") or 0.0
    abs_yaw = abs(yaw)

    if abs_yaw >= 1.8:
        return "turn_high_yaw"
    if "left" in text or yaw >= 0.55:
        return "left"
    if "right" in text or yaw <= -0.55:
        return "right"
    if _has_any_tag(tags, STOP_TAG_WORDS) or (abs_yaw < 0.18 and travel < 12.0):
        return "straight_low_speed_stop_proxy"
    if "high_magnitude_speed" in tags or "medium_magnitude_speed" in tags or (abs_yaw < 0.18 and travel >= 14.0):
        return "straight_cruise"
    if "intersection" in text or "cross" in text or (0.18 <= abs_yaw < 0.55 and travel >= 8.0):
        return "intersection_proxy"
    return "straight_cruise"


def geo_cell(row: dict[str, str], cell_m: float = GEO_CELL_M) -> str:
    x = _first_present(row, ("ego_x", "x", "ego_center_x", "ego_start_x", "start_x"))
    y = _first_present(row, ("ego_y", "y", "ego_center_y", "ego_start_y", "start_y"))
    if x is None or y is None:
        return "unknown"
    return f"{math.floor(x / cell_m)}:{math.floor(y / cell_m)}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scenario_token(row: dict[str, str]) -> str:
    for key in ("token", "scenario_token", "lidar_pc_token"):
        value = row.get(key)
        if value:
            return value
    raise ValueError(f"scenario row has no token-like field: {row}")


def enrich_row(row: dict[str, str], index: int) -> dict[str, str]:
    item = dict(row)
    item["token"] = scenario_token(row)
    item.setdefault("nuplan_tags", "")
    item.setdefault("metadata_status", "csv_only")

    db_path = item.get("db_path", "")
    lidar_pc_token = item.get("lidar_pc_token") or item["token"]
    ego_pose_token = item.get("ego_pose_token", "")
    if db_path and Path(db_path).exists() and lidar_pc_token:
        try:
            with sqlite3.connect(db_path) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT type FROM scenario_tag WHERE hex(lidar_pc_token)=upper(?) ORDER BY type",
                    (lidar_pc_token,),
                )
                tags = [str(value[0]) for value in cursor.fetchall() if value[0]]
                item["nuplan_tags"] = ";".join(tags)

                pose = None
                if ego_pose_token:
                    cursor.execute(
                        "SELECT x, y FROM ego_pose WHERE hex(token)=upper(?)",
                        (ego_pose_token,),
                    )
                    pose = cursor.fetchone()
                if pose is None:
                    cursor.execute(
                        """
                        SELECT ep.x, ep.y
                        FROM lidar_pc AS lp
                        JOIN ego_pose AS ep ON lp.ego_pose_token = ep.token
                        WHERE hex(lp.token)=upper(?)
                        """,
                        (lidar_pc_token,),
                    )
                    pose = cursor.fetchone()
                if pose is not None:
                    item["ego_x"] = f"{float(pose[0]):.3f}"
                    item["ego_y"] = f"{float(pose[1]):.3f}"
                item["metadata_status"] = "db_ok"
        except (OSError, sqlite3.Error, ValueError) as exc:
            item["metadata_status"] = f"db_error:{type(exc).__name__}"
    elif db_path:
        item["metadata_status"] = "db_missing"

    item["behavior_bucket"] = behavior_bucket(item)
    item["geo_cell"] = geo_cell(item)
    item["_input_order"] = str(index)
    return item


def _selection_score(item: dict[str, str], location_counts: dict[str, int]) -> tuple[int, int, int, int]:
    tags = _split_tags(item)
    interesting = int(_has_any_tag(tags, STOP_TAG_WORDS)) + int(_has_any_tag(tags, INTERACTION_TAG_WORDS))
    travel = _float_value(item, "travel_m") or 0.0
    bucket = item.get("behavior_bucket", "")
    if bucket == "straight_cruise":
        motion_score = -int(travel * 100)
    elif bucket == "straight_low_speed_stop_proxy":
        motion_score = int(travel * 100)
    else:
        motion_score = -int(abs(_float_value(item, "yaw_change_rad") or 0.0) * 1000)
    return (
        location_counts.get(item.get("location", ""), 0),
        -interesting,
        motion_score,
        int(item.get("_input_order", "0")),
    )


def select_scenarios(rows: Sequence[dict[str, str]], target_count: int = DEFAULT_SCENARIO_COUNT) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        enriched.append(enrich_row(row, index))

    selected: list[dict[str, str]] = []
    selected_tokens: set[str] = set()
    seen_logs: set[tuple[str, str]] = set()
    seen_geo: set[tuple[str, str, str]] = set()
    location_counts: dict[str, int] = {}

    def try_add(item: dict[str, str]) -> bool:
        if item["token"] in selected_tokens:
            return False
        log_key = (item.get("location", ""), item.get("log_name", ""))
        if log_key in seen_logs:
            return False
        cell = item.get("geo_cell", "unknown")
        if cell != "unknown":
            geo_key = (item.get("location", ""), item.get("map_version", ""), cell)
            if geo_key in seen_geo:
                return False
            seen_geo.add(geo_key)
        seen_logs.add(log_key)
        selected_tokens.add(item["token"])
        location = item.get("location", "")
        location_counts[location] = location_counts.get(location, 0) + 1
        selected.append(item)
        return True

    for bucket, bucket_target in PRIMARY_BUCKET_TARGETS.items():
        for _ in range(bucket_target):
            candidates = [
                item for item in enriched if item["behavior_bucket"] == bucket and item["token"] not in selected_tokens
            ]
            candidates.sort(key=lambda item: _selection_score(item, location_counts))
            if not any(try_add(item) for item in candidates):
                break

    while len(selected) < target_count:
        if len(selected) >= target_count:
            break
        candidates = [item for item in enriched if item["token"] not in selected_tokens]
        candidates.sort(key=lambda item: _selection_score(item, location_counts))
        if not any(try_add(candidate) for candidate in candidates):
            break

    for item in selected:
        item.pop("_input_order", None)
    return selected


def write_csv(path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def token_rows(selected: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "token": row["token"],
            "behavior_bucket": row["behavior_bucket"],
            "location": row.get("location", ""),
            "log_name": row.get("log_name", ""),
            "geo_cell": row.get("geo_cell", "unknown"),
            "nuplan_tags": row.get("nuplan_tags", ""),
        }
        for row in selected
    ]


def env_prefix(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_matrix_row(
    *,
    matrix_name: str,
    run_id: str,
    token_csv: Path,
    checkpoint: str,
    data_root: str,
    output_root: Path,
    sampling_steps: int,
    noise_scale: float,
    dpm_variant: str,
    dpm_env: dict[str, str],
) -> dict[str, str]:
    run_output_root = output_root / matrix_name / run_id
    trace_path = run_output_root / "selector_trace.jsonl"
    env = {
        "DP_TRAJ_SELECTOR_ENABLE": "1",
        "DP_TRAJ_SELECTOR_SAMPLES": str(SELECTOR_FIXED_SAMPLES),
        "DP_TRAJ_SELECTOR_TRACE_JSONL": str(trace_path),
        "DP_TRAJ_SELECTOR_TRACE_TICK0_ONLY": "0",
        "DP_CLOSEDLOOP_DEBUG_CANDIDATES": "0",
        "DP_SAMPLER_NOISE_SCALE": str(noise_scale),
        "NUPLAN_EXP_ROOT": str(run_output_root),
    }
    env.update(dpm_env)
    cmd = [
        "python3",
        str(SIM_SCRIPT),
        "--scenarios_file",
        str(token_csv),
        "--checkpoint",
        checkpoint,
        "--sampling-steps",
        str(sampling_steps),
        "--data-root",
        data_root,
    ]
    command = f"{env_prefix(env)} {shell_join(cmd)}"
    return {
        "matrix": matrix_name,
        "run_id": run_id,
        "scenario_set_path": str(token_csv),
        "sampling_steps": str(sampling_steps),
        "noise_scale": str(noise_scale),
        "dpm_variant": dpm_variant,
        "dpm_env_json": json.dumps(dpm_env, ensure_ascii=False, sort_keys=True),
        "dpm_vars": " ".join(f"{key}={value}" for key, value in sorted(dpm_env.items())),
        "selector_fixed_samples": str(SELECTOR_FIXED_SAMPLES),
        "trace_path": str(trace_path),
        "output_root": str(run_output_root),
        "env_json": json.dumps(env, ensure_ascii=False, sort_keys=True),
        "command": command,
    }


def main_grid_rows(token_csv: Path, checkpoint: str, data_root: str, output_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sampling_steps in SAMPLING_STEPS:
        for noise_scale in NOISE_SCALES:
            run_id = f"steps{sampling_steps}_noise{str(noise_scale).replace('.', 'p')}"
            rows.append(
                build_matrix_row(
                    matrix_name="main_grid",
                    run_id=run_id,
                    token_csv=token_csv,
                    checkpoint=checkpoint,
                    data_root=data_root,
                    output_root=output_root,
                    sampling_steps=sampling_steps,
                    noise_scale=noise_scale,
                    dpm_variant="default",
                    dpm_env={},
                )
            )
    return rows


def dpm_probe_rows(
    token_csv: Path,
    checkpoint: str,
    data_root: str,
    output_root: Path,
    sampling_steps: int,
    noise_scale: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for variant_name, dpm_env in DPM_VARIANTS:
        rows.append(
            build_matrix_row(
                matrix_name="dpm_probe",
                run_id=f"dpm_{variant_name}",
                token_csv=token_csv,
                checkpoint=checkpoint,
                data_root=data_root,
                output_root=output_root,
                sampling_steps=sampling_steps,
                noise_scale=noise_scale,
                dpm_variant=variant_name,
                dpm_env=dpm_env,
            )
        )
    return rows


def matrix_fieldnames() -> list[str]:
    return [
        "matrix",
        "run_id",
        "scenario_set_path",
        "sampling_steps",
        "noise_scale",
        "dpm_variant",
        "dpm_env_json",
        "dpm_vars",
        "selector_fixed_samples",
        "trace_path",
        "output_root",
        "env_json",
        "command",
    ]


def write_runbook(path: Path, selected: Sequence[dict[str, str]], matrix_paths: Sequence[Path], args: argparse.Namespace) -> None:
    bucket_counts: dict[str, int] = {}
    location_counts: dict[str, int] = {}
    metadata_counts: dict[str, int] = {}
    for row in selected:
        bucket_counts[row["behavior_bucket"]] = bucket_counts.get(row["behavior_bucket"], 0) + 1
        location = row.get("location", "")
        location_counts[location] = location_counts.get(location, 0) + 1
        status = row.get("metadata_status", "")
        metadata_counts[status] = metadata_counts.get(status, 0) + 1
    lines = [
        "# DP closed-loop sampling grid runbook",
        "",
        "## Scope",
        "",
        "- 本工具只生成 scenario 选择、tokens、run matrix 和 runbook；不要启动仿真。",
        f"- candidate_count fixed samples=10；不搜索候选数量。",
        "- DPM knobs 只作为 second-stage 局部 variants，不进入 main full-combination grid。",
        "",
        "## Scenario selection",
        "",
        f"- Input CSV: `{args.input_csv}`",
        f"- Selected CSV: `{path.parent / 'selected_scenarios.csv'}`",
        f"- Tokens CSV: `{path.parent / 'selected_tokens.csv'}`",
        f"- Selected rows: {len(selected)} / target {args.scenario_count}",
        f"- Buckets: {json.dumps(bucket_counts, ensure_ascii=False, sort_keys=True)}",
        f"- Locations: {json.dumps(location_counts, ensure_ascii=False, sort_keys=True)}",
        f"- Metadata status: {json.dumps(metadata_counts, ensure_ascii=False, sort_keys=True)}",
        f"- Geo cell: `{GEO_CELL_M:.0f}m` cell from DB ego pose when available.",
        "- 行为分类来自 `group/scenario_type/yaw_change_rad/travel_m`，并用 DB `scenario_tag` 区分 stop/interactions proxy。",
        "- `straight_low_speed_stop_proxy` 是低速/stop-tag proxy；不把它冒充成真实停车 case。",
        "- 去重规则：同一 `location + log_name` 最多 1 个；DB ego pose 可用时同一 `location + map_version + geo_cell` 最多 1 个。",
        "- 矩阵默认只写 selector JSONL trace；不启用 closed-loop npz/debug dumps，以降低 I/O。",
        "",
        "## Matrices",
        "",
    ]
    for matrix_path in matrix_paths:
        lines.append(f"- `{matrix_path}`")
    lines.extend(
        [
            "",
            "## Example",
            "",
            "```bash",
            f"python3 {SIM_SCRIPT} --scenarios_file {shlex.quote(str(path.parent / 'selected_tokens.csv'))} --checkpoint {shlex.quote(args.checkpoint)} --sampling-steps 10 --data-root {shlex.quote(args.data_root)}",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--matrix", choices=["main_grid", "dpm_probe", "all"], default="all")
    parser.add_argument("--checkpoint", default="outputs/training/diffusion_planner.ckpt")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dpm-probe-sampling-steps", type=int, default=10)
    parser.add_argument("--dpm-probe-noise-scale", type=float, default=1.0)
    parser.add_argument("--scenario-count", type=int, default=DEFAULT_SCENARIO_COUNT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_scenarios(read_rows(input_csv), target_count=args.scenario_count)
    selected_path = output_dir / "selected_scenarios.csv"
    tokens_path = output_dir / "selected_tokens.csv"
    write_csv(selected_path, selected)
    write_csv(tokens_path, token_rows(selected), ["token", "behavior_bucket", "location", "log_name", "geo_cell", "nuplan_tags"])

    matrix_paths: list[Path] = []
    if args.matrix in ("main_grid", "all"):
        path = output_dir / "run_matrix_main_grid.csv"
        write_csv(path, main_grid_rows(tokens_path, args.checkpoint, args.data_root, output_dir), matrix_fieldnames())
        matrix_paths.append(path)
    if args.matrix in ("dpm_probe", "all"):
        path = output_dir / "run_matrix_dpm_probe.csv"
        write_csv(
            path,
            dpm_probe_rows(
                tokens_path,
                args.checkpoint,
                args.data_root,
                output_dir,
                args.dpm_probe_sampling_steps,
                args.dpm_probe_noise_scale,
            ),
            matrix_fieldnames(),
        )
        matrix_paths.append(path)

    write_runbook(output_dir / "sampling_grid_runbook.md", selected, matrix_paths, args)
    print(f"[ok] wrote {selected_path}")
    print(f"[ok] wrote {tokens_path}")
    for path in matrix_paths:
        print(f"[ok] wrote {path}")
    print(f"[ok] wrote {output_dir / 'sampling_grid_runbook.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
