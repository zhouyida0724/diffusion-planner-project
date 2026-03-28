#!/usr/bin/env python3
"""run_diffusion_simulation.py - Run nuPlan closed-loop simulation with Diffusion Planner.

CLI is kept stable; implementation uses `src.platform.nuplan` helpers.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.platform.nuplan.planners.overrides import diffusion_planner_overrides, idm_planner_overrides
from src.platform.nuplan.runners.simulation import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SCENARIO_FILTER_CONFIG_FILE,
    build_run_simulation_cmd,
    make_invocation,
    run,
    write_scenario_filter_yaml_for_tokens,
)

CONFIG_FILE = DEFAULT_SCENARIO_FILTER_CONFIG_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Diffusion Planner simulation")
    parser.add_argument("--scenarios_file", type=str, default=None, help="CSV file with scenario tokens")
    parser.add_argument("--scenario", type=str, default=None, help="Specific scenario token")
    parser.add_argument("--num", type=int, default=None, help="Number of random scenarios")
    parser.add_argument(
        "--planner",
        type=str,
        default="diffusion_planner",
        choices=["diffusion_planner", "diffusion", "idm_planner", "idm"],
        help=(
            "Planner type. Accepts: diffusion_planner|diffusion (same) or idm_planner|idm (same). "
            "Default: diffusion_planner"
        ),
    )
    # Keep both spellings for backwards/forwards compatibility.
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Checkpoint path. Supports legacy .pth as well as training-skeleton .pt checkpoints under "
            "outputs/training/<exp>/checkpoint_*.pt."
        ),
    )
    args = parser.parse_args()

    # Ensure both nuPlan vendor code and our repo code are importable in the simulation subprocess.
    # Put repo root first so `src.*` is always resolvable.
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = f"{repo_root}:/workspace/nuplan-visualization"

    planner_key = args.planner
    if planner_key in {"idm", "idm_planner"}:
        planner_overrides = idm_planner_overrides().args
    else:
        ckpt = args.checkpoint
        if ckpt is None:
            # Keep old default for legacy users, but explicit is preferred.
            ckpt = "/workspace/checkpoints/model.pth"
        else:
            # Make ckpt path robust to Hydra changing CWD during runs.
            p = Path(ckpt)
            if not p.is_absolute():
                repo_root = Path(__file__).resolve().parents[2]
                p = (repo_root / p).resolve()
            ckpt = str(p)
        planner_overrides = diffusion_planner_overrides(ckpt).args

    if args.scenarios_file:
        tokens = []
        with open(args.scenarios_file, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if parts and parts[0]:
                    tokens.append(parts[0])

        print(f"Loaded {len(tokens)} scenarios from {args.scenarios_file}")
        write_scenario_filter_yaml_for_tokens(scenario_tokens=tokens, config_file=CONFIG_FILE)

        cmd = build_run_simulation_cmd(
            planner_overrides=planner_overrides,
            scenario_filter="one_hand_picked_scenario",
            data_root=DEFAULT_DATA_ROOT,
        )

    elif args.scenario:
        write_scenario_filter_yaml_for_tokens(scenario_tokens=[args.scenario], config_file=CONFIG_FILE)
        cmd = build_run_simulation_cmd(
            planner_overrides=planner_overrides,
            scenario_filter="one_hand_picked_scenario",
            data_root=DEFAULT_DATA_ROOT,
        )

    elif args.num:
        cmd = build_run_simulation_cmd(
            planner_overrides=planner_overrides,
            scenario_filter="simulation_test_split",
            data_root=DEFAULT_DATA_ROOT,
            extra_overrides=[f"scenario_filter.limit_total_scenarios={args.num}"],
        )

    else:
        cmd = build_run_simulation_cmd(
            planner_overrides=planner_overrides,
            scenario_filter="simulation_test_split",
            data_root=DEFAULT_DATA_ROOT,
            extra_overrides=["scenario_filter.limit_total_scenarios=1"],
        )

    print(f"Running: {' '.join(cmd)}")
    invocation = make_invocation(cmd=cmd, pythonpath=pythonpath)
    raise SystemExit(run(invocation))


if __name__ == "__main__":
    main()
