"""nuPlan simulation runner glue.

This module centralizes the small bits of scripting glue used to:
- write a temporary/hand-picked ScenarioFilter YAML into nuPlan's config tree
- assemble Hydra override arguments for `nuplan.planning.script.run_simulation`
- launch the simulation entrypoint with an explicit environment

It intentionally does *not* try to wrap/replace nuPlan's Hydra CLI; it only
reduces duplication in repo scripts.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_SCENARIO_FILTER_CONFIG_FILE = (
    "/workspace/nuplan-visualization/nuplan/planning/script/config/common/"
    "scenario_filter/one_hand_picked_scenario.yaml"
)

DEFAULT_DATA_ROOT = "/workspace/data/nuplan/data/cache/mini"


@dataclass(frozen=True)
class SimulationInvocation:
    cmd: List[str]
    env: Dict[str, str]


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def set_env_pythonpath(env: Dict[str, str], pythonpath: str) -> Dict[str, str]:
    """Return a copy of env with PYTHONPATH set."""
    merged = dict(env)
    merged["PYTHONPATH"] = pythonpath
    return merged


def set_env_nuplan_exp_root(env: Dict[str, str], exp_root: str | Path) -> Dict[str, str]:
    """Return a copy of env with NUPLAN_EXP_ROOT set."""
    merged = dict(env)
    merged["NUPLAN_EXP_ROOT"] = str(exp_root)
    return merged


def write_scenario_filter_yaml_for_tokens(
    *,
    scenario_tokens: Sequence[str],
    config_file: str = DEFAULT_SCENARIO_FILTER_CONFIG_FILE,
) -> None:
    ensure_parent_dir(config_file)
    tokens_yaml = "\n".join([f"  - '{t}'" for t in scenario_tokens])
    Path(config_file).write_text(
        """_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter
scenario_types: null
scenario_tokens:
"""
        + (tokens_yaml + "\n" if tokens_yaml else "")
        + """log_names: null
map_names: null
num_scenarios_per_type: 1
limit_total_scenarios: null
timestamp_threshold_s: null
ego_displacement_minimum_m: null
expand_scenarios: false
remove_invalid_goals: true
shuffle: false
""",
        encoding="utf-8",
    )


def write_scenario_filter_yaml_for_log_names(
    *,
    log_names: Sequence[str],
    config_file: str = DEFAULT_SCENARIO_FILTER_CONFIG_FILE,
) -> None:
    ensure_parent_dir(config_file)
    logs_yaml = "\n".join([f"  - '{name}'" for name in log_names])
    Path(config_file).write_text(
        """_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter

scenario_tokens: null
log_names:
"""
        + (logs_yaml + "\n" if logs_yaml else "")
        + """scenario_type: null
map_names: null
num_scenarios_per_type: 1
limit_total_scenarios: null
timestamp_threshold_s: null
ego_displacement_minimum_m: null
expand_scenarios: false
remove_invalid_goals: true
shuffle: false
""",
        encoding="utf-8",
    )


def copy_scenario_filter_yaml(*, src_yaml_file: str | Path, dst_config_file: str) -> None:
    ensure_parent_dir(dst_config_file)
    Path(dst_config_file).write_text(Path(src_yaml_file).read_text(encoding="utf-8"), encoding="utf-8")


def build_run_simulation_cmd(
    *,
    simulation_preset: str = "closed_loop_nonreactive_agents",
    planner_overrides: Sequence[str],
    scenario_filter: str,
    scenario_builder: str = "nuplan",
    data_root: str = DEFAULT_DATA_ROOT,
    worker: str = "sequential",
    verbose: bool = True,
    extra_overrides: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd: List[str] = [
        "python3",
        "-m",
        "nuplan.planning.script.run_simulation",
        f"+simulation={simulation_preset}",
        *list(planner_overrides),
        f"scenario_builder={scenario_builder}",
        f"scenario_filter={scenario_filter}",
        f"scenario_builder.data_root={data_root}",
        f"worker={worker}",
    ]
    if verbose:
        cmd.append("verbose=true")
    if extra_overrides:
        cmd.extend(list(extra_overrides))
    return cmd


def run(invocation: SimulationInvocation) -> int:
    completed = subprocess.run(invocation.cmd, env=invocation.env)
    return int(completed.returncode)


def make_invocation(
    *,
    cmd: List[str],
    base_env: Optional[Dict[str, str]] = None,
    pythonpath: Optional[str] = None,
    nuplan_exp_root: Optional[str | Path] = None,
) -> SimulationInvocation:
    env = dict(base_env or os.environ)
    if pythonpath is not None:
        env = set_env_pythonpath(env, pythonpath)
    if nuplan_exp_root is not None:
        env = set_env_nuplan_exp_root(env, nuplan_exp_root)
    return SimulationInvocation(cmd=cmd, env=env)
