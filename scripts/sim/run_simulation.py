#!/usr/bin/env python3
"""run_simulation.py - Run nuPlan closed-loop simulation with IDM Planner.

NOTE: This script's CLI is kept stable. The implementation delegates the
nuPlan/Hydra glue to `src.platform.nuplan` helpers.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.platform.nuplan.planners.overrides import idm_planner_overrides
from src.platform.nuplan.runners.simulation import (
    DEFAULT_SCENARIO_FILTER_CONFIG_FILE,
    build_run_simulation_cmd,
    copy_scenario_filter_yaml,
    make_invocation,
    run,
    write_scenario_filter_yaml_for_log_names,
    write_scenario_filter_yaml_for_tokens,
)

CONFIG_FILE = DEFAULT_SCENARIO_FILTER_CONFIG_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDM simulation")
    parser.add_argument("--scenarios_file", type=str, default=None, help="CSV file with log names")
    parser.add_argument("--yaml_file", type=str, default=None, help="YAML config file with scenario filter")
    parser.add_argument("--scenario", type=str, default=None, help="Specific scenario token")
    parser.add_argument("--num", type=int, default=None, help="Number of random scenarios")
    args = parser.parse_args()

    # Require explicit specification (keep original behavior/message)
    if not args.scenarios_file and not args.scenario and not args.num:
        parser.error("必须指定 --scenarios_file, --scenario 或 --num")

    # Match previous env behavior
    pythonpath = "/workspace/nuplan-visualization"

    planner_cfg = idm_planner_overrides().args

    if args.yaml_file:
        copy_scenario_filter_yaml(src_yaml_file=args.yaml_file, dst_config_file=CONFIG_FILE)
        cmd = build_run_simulation_cmd(
            planner_overrides=planner_cfg,
            scenario_filter="one_hand_picked_scenario",
            extra_overrides=None,
        )

    elif args.scenarios_file:
        # Read log names from CSV (first column is log_name)
        log_names = []
        with open(args.scenarios_file, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if parts and parts[0]:
                    log_names.append(parts[0])

        print(f"Loaded {len(log_names)} log names from {args.scenarios_file}")
        write_scenario_filter_yaml_for_log_names(log_names=log_names, config_file=CONFIG_FILE)

        cmd = build_run_simulation_cmd(
            planner_overrides=planner_cfg,
            scenario_filter="one_hand_picked_scenario",
            extra_overrides=None,
        )

    elif args.scenario:
        # Run a specific scenario token via the hand-picked scenario filter
        write_scenario_filter_yaml_for_tokens(scenario_tokens=[args.scenario], config_file=CONFIG_FILE)
        cmd = build_run_simulation_cmd(
            planner_overrides=planner_cfg,
            scenario_filter="one_hand_picked_scenario",
            extra_overrides=None,
        )

    elif args.num:
        # Limit scenario count via ScenarioFilter (works across nuPlan versions)
        cmd = build_run_simulation_cmd(
            planner_overrides=planner_cfg,
            scenario_filter="simulation_test_split",
            extra_overrides=[f"scenario_filter.limit_total_scenarios={args.num}"],
        )

    else:
        cmd = build_run_simulation_cmd(
            planner_overrides=planner_cfg,
            scenario_filter="simulation_test_split",
            extra_overrides=["scenario_filter.limit_total_scenarios=1"],
        )

    print(f"Running: {' '.join(cmd)}")
    invocation = make_invocation(cmd=cmd, pythonpath=pythonpath)
    raise SystemExit(run(invocation))


if __name__ == "__main__":
    main()
