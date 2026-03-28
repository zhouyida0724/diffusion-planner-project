"""Planner override helpers for nuPlan's Hydra configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PlannerOverrides:
    """Hydra override args for selecting/configuring a planner."""

    # E.g. ["planner=idm_planner"] or ["planner=diffusion_planner", "+diffusion_planner.ckpt_path=/path"]
    args: List[str]


def idm_planner_overrides() -> PlannerOverrides:
    return PlannerOverrides(args=["planner=idm_planner"])


def diffusion_planner_overrides(ckpt_path: Optional[str] = None) -> PlannerOverrides:
    args: List[str] = ["planner=diffusion_planner"]
    if ckpt_path:
        args.append(f"+diffusion_planner.ckpt_path={ckpt_path}")
    return PlannerOverrides(args=args)
