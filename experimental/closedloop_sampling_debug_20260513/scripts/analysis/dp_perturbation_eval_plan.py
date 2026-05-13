#!/usr/bin/env python3
"""Generate a closed-loop perturbation eval run matrix and runbook.

This script plans deterministic perturbation runs only. It does not launch
nuPlan simulations and intentionally does not modify planner/model code.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_SCRIPT = REPO_ROOT / "scripts" / "sim" / "run_diffusion_simulation.py"
SIM_WRAPPER = REPO_ROOT / "scripts" / "run_diffusion_simulation.py"
ROOT_LEGACY_SIM = REPO_ROOT / "run_diffusion_simulation.py"


@dataclass(frozen=True)
class Perturbation:
    run_id: str
    kind: str
    lateral_m: float
    heading_deg: float
    is_baseline: bool = False


def parse_float_list(value: str) -> list[float]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return [float(item) for item in items]


def signed_label(value: float, unit: str) -> str:
    sign = "p" if value >= 0 else "m"
    magnitude = str(abs(value)).replace(".", "p")
    return f"{sign}{magnitude}{unit}"


def build_perturbations(lateral_m: Sequence[float], heading_deg: Sequence[float]) -> list[Perturbation]:
    runs = [Perturbation(run_id="baseline", kind="baseline", lateral_m=0.0, heading_deg=0.0, is_baseline=True)]
    for magnitude in lateral_m:
        for sign in (-1.0, 1.0):
            value = sign * float(magnitude)
            runs.append(
                Perturbation(
                    run_id=f"lat_{signed_label(value, 'm')}",
                    kind="lateral",
                    lateral_m=value,
                    heading_deg=0.0,
                )
            )
    for magnitude in heading_deg:
        for sign in (-1.0, 1.0):
            value = sign * float(magnitude)
            runs.append(
                Perturbation(
                    run_id=f"head_{signed_label(value, 'deg')}",
                    kind="heading",
                    lateral_m=0.0,
                    heading_deg=value,
                )
            )
    return runs


def source_text(paths: Iterable[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def simulator_capabilities() -> dict[str, object]:
    text = source_text([SIM_SCRIPT, SIM_WRAPPER, ROOT_LEGACY_SIM, REPO_ROOT / "src" / "platform" / "nuplan" / "runners" / "simulation.py"])
    args = sorted(
        {
            flag
            for flag in [
                "--scenarios_file",
                "--scenario",
                "--num",
                "--planner",
                "--checkpoint",
                "--ckpt",
                "--sampling-steps",
                "--data-root",
            ]
            if flag in text
        }
    )
    env_vars = sorted(
        {
            name
            for name in [
                "NUPLAN_MAPS_ROOT",
                "NUPLAN_DATA_ROOT",
                "NUPLAN_EXP_ROOT",
                "PYTHONPATH",
                "DP_TRAJ_SELECTOR_ENABLE",
                "DP_TRAJ_SELECTOR_SAMPLES",
                "DP_TRAJ_SELECTOR_TRACE_JSONL",
                "DP_CLOSEDLOOP_DEBUG_DUMP_DIR",
            ]
            if name in source_text([SIM_SCRIPT, REPO_ROOT / "src" / "platform" / "nuplan" / "planners" / "diffusion_planner_ckpt_planner.py", REPO_ROOT / "src" / "platform" / "nuplan" / "runners" / "simulation.py"])
        }
    )
    perturb_needles = [
        "DP_INITIAL_PERTURB",
        "INITIAL_PERTURB",
        "initial_perturb",
        "ego_initial_perturb",
        "lateral_perturb",
        "heading_perturb",
    ]
    injection_supported = any(needle in text for needle in perturb_needles)
    return {"args": args, "env_vars": env_vars, "injection_supported": injection_supported}


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_command(
    *,
    token_csv: Path,
    checkpoint: str,
    data_root: str,
    sampling_steps: int,
    selector_enable: bool,
    perturbation: Perturbation,
    run_output_root: Path,
    injection_supported: bool,
) -> tuple[str, dict[str, str]]:
    env: dict[str, str] = {
        "NUPLAN_EXP_ROOT": str(run_output_root),
        "DP_CLOSEDLOOP_DEBUG_DUMP_DIR": str(run_output_root / "debug_dumps"),
    }
    if selector_enable:
        env["DP_TRAJ_SELECTOR_ENABLE"] = "1"
        env["DP_TRAJ_SELECTOR_TRACE_JSONL"] = str(run_output_root / "selector_trace.jsonl")
    if injection_supported:
        env["DP_INITIAL_PERTURB_LATERAL_M"] = f"{perturbation.lateral_m:g}"
        env["DP_INITIAL_PERTURB_HEADING_DEG"] = f"{perturbation.heading_deg:g}"

    cmd = [
        "python3",
        str(SIM_SCRIPT.relative_to(REPO_ROOT)),
        "--scenarios_file",
        str(token_csv),
        "--checkpoint",
        checkpoint,
        "--sampling-steps",
        str(int(sampling_steps)),
        "--data-root",
        str(data_root),
    ]
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    return f"{prefix} {shell_join(cmd)}", env


def matrix_rows(args: argparse.Namespace, perturbations: Sequence[Perturbation], capabilities: dict[str, object]) -> list[dict[str, str]]:
    injection_supported = bool(capabilities["injection_supported"])
    rows: list[dict[str, str]] = []
    for perturbation in perturbations:
        run_output_root = Path(args.base_output_root) / perturbation.run_id
        command, env = build_command(
            token_csv=Path(args.token_csv),
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            sampling_steps=args.sampling_steps,
            selector_enable=args.selector_enable,
            perturbation=perturbation,
            run_output_root=run_output_root,
            injection_supported=injection_supported,
        )
        rows.append(
            {
                "run_id": perturbation.run_id,
                "perturbation_kind": perturbation.kind,
                "lateral_m": f"{perturbation.lateral_m:g}",
                "heading_deg": f"{perturbation.heading_deg:g}",
                "is_baseline": str(perturbation.is_baseline).lower(),
                "token_csv": str(args.token_csv),
                "checkpoint": str(args.checkpoint),
                "data_root": str(args.data_root),
                "sampling_steps": str(int(args.sampling_steps)),
                "selector_enable": str(bool(args.selector_enable)).lower(),
                "output_root": str(run_output_root),
                "injection_supported": str(injection_supported).lower(),
                "env_json": json.dumps(env, ensure_ascii=False, sort_keys=True),
                "command": command,
            }
        )
    return rows


def write_matrix(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "perturbation_kind",
        "lateral_m",
        "heading_deg",
        "is_baseline",
        "token_csv",
        "checkpoint",
        "data_root",
        "sampling_steps",
        "selector_enable",
        "output_root",
        "injection_supported",
        "env_json",
        "command",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_runbook(path: Path, rows: Sequence[dict[str, str]], capabilities: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    args_list = ", ".join(str(item) for item in capabilities["args"]) or "(none found)"
    env_list = ", ".join(str(item) for item in capabilities["env_vars"]) or "(none found)"
    injection_supported = bool(capabilities["injection_supported"])
    status = (
        "发现疑似初始扰动注入接口，可按 matrix 中 env 执行。"
        if injection_supported
        else "当前未发现可用的初始扰动注入接口；matrix 先生成 baseline/perturbation 计划，扰动列不可真正生效。"
    )
    example_data_root = shlex.quote(str(rows[0]["data_root"])) if rows else "DATA_ROOT"
    lines = [
        "# Closed-loop perturbation eval runbook",
        "",
        "## 目标",
        "",
        "- 判断 open-loop OK 但 closed-loop 飞是否来自模型缺恢复能力；route 问题冻结，不在本计划中变更。",
        "- 只生成 run matrix 和命令，不启动长仿真。",
        "",
        "## Run matrix",
        "",
        f"- CSV: `{path.parent / 'perturbation_run_matrix.csv'}`",
        f"- deterministic perturbations: baseline, lateral +/-0.5m +/-1.0m, heading +/-3deg +/-6deg（或 CLI 指定 grid）。",
        f"- 当前状态: {status}",
        "",
        "## 当前 run_diffusion_simulation.py 能力",
        "",
        f"- 支持参数: {args_list}",
        f"- 相关环境变量: {env_list}",
        "- selector 开启方式: `DP_TRAJ_SELECTOR_ENABLE=1`；脚本也会写 `DP_TRAJ_SELECTOR_TRACE_JSONL`。",
        "",
        "## 初始扰动注入缺口",
        "",
        "- 当前未发现可用的初始扰动注入接口，不能仅靠 CLI/env 对 closed-loop 初始 ego state 加 lateral/heading perturbation。",
        "- 最小 hook 建议：在 simulation 初始化 ego state 前读取 `DP_INITIAL_PERTURB_LATERAL_M` 和 `DP_INITIAL_PERTURB_HEADING_DEG`，只平移/旋转初始 rear-axle pose，不改 planner/model。",
        "- hook 应记录实际生效值到仿真输出 metadata，baseline 必须保持两个 env 为空或 0。",
        "",
        "## Hydra 中文 data-root quoting 建议",
        "",
        f"- shell 中推荐把 Hydra override 写成 `scenario_builder.data_root={example_data_root}`，避免中文、空格或特殊字符被 shell/Hydra 拆分。",
        "- 本脚本生成的 `command` 已用 `shlex.quote` 对 `--data-root`、输出目录和 env 值做 shell quoting。",
        "",
        "## Commands",
        "",
    ]
    for row in rows:
        lines.extend([f"### {row['run_id']}", "", "```bash", row["command"], "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate closed-loop perturbation eval run matrix and runbook")
    parser.add_argument("--token-csv", required=True, help="CSV containing scenario tokens")
    parser.add_argument("--base-output-root", required=True, help="Base output root for matrix, runbook, and planned runs")
    parser.add_argument("--checkpoint", required=True, help="Diffusion planner checkpoint path")
    parser.add_argument("--data-root", required=True, help="nuPlan data root / cache root")
    parser.add_argument("--sampling-steps", type=int, required=True, help="Diffusion sampling steps")
    parser.add_argument("--selector-enable", action="store_true", help="Enable trajectory selector env in generated commands")
    parser.add_argument("--lateral-m", type=parse_float_list, default=parse_float_list("0.5,1.0"))
    parser.add_argument("--heading-deg", type=parse_float_list, default=parse_float_list("3,6"))
    parser.add_argument("--matrix-csv", default="", help="Optional explicit matrix CSV path")
    parser.add_argument("--runbook-md", default="", help="Optional explicit runbook markdown path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_output_root = Path(args.base_output_root)
    perturbations = build_perturbations(args.lateral_m, args.heading_deg)
    capabilities = simulator_capabilities()
    rows = matrix_rows(args, perturbations, capabilities)
    matrix_path = Path(args.matrix_csv) if args.matrix_csv else base_output_root / "perturbation_run_matrix.csv"
    runbook_path = Path(args.runbook_md) if args.runbook_md else base_output_root / "perturbation_runbook.md"
    write_matrix(matrix_path, rows)
    write_runbook(runbook_path, rows, capabilities)
    print(f"Wrote matrix: {matrix_path}")
    print(f"Wrote runbook: {runbook_path}")
    print(f"Perturbation injection supported: {str(bool(capabilities['injection_supported'])).lower()}")


if __name__ == "__main__":
    main()
