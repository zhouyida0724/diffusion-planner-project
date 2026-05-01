from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np


DrivableChecker = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EgoPose2D:
    x: float
    y: float
    heading: float


@dataclass
class CandidateTrajectory:
    """Candidate trajectory in the current ego-local frame.

    local_xyh: [T,3] columns are x, y, heading in ego-local coordinates.
    """

    local_xyh: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SelectionContext:
    ego_pose: EgoPose2D
    route_centerlines_local: list[np.ndarray]
    drivable_checker: DrivableChecker
    vehicle_half_length: float
    vehicle_half_width: float
    rear_axle_to_center_dist: float = 0.0
    dt: float = 0.1
    previous_selected_local: np.ndarray | None = None


@dataclass
class CandidateDiagnostics:
    prefix_offroad_steps: int
    late_offroad_steps: int
    prefix_progress_m: float
    prefix_progress_shortfall_m: float
    total_progress_m: float
    prefix_end_lateral_error_m: float
    total_end_lateral_error_m: float
    prefix_end_heading_error_rad: float
    total_end_heading_error_rad: float
    consistency_l2: float
    final_score: float
    rejected: bool
    reject_reason: str | None = None


@dataclass
class SelectionResult:
    best_index: int
    diagnostics: list[CandidateDiagnostics]
    survivor_indices: list[int]
    used_fallback: bool
