from __future__ import annotations

import numpy as np

from src.platform.nuplan.trajectory_selector import CandidateTrajectory, EgoPose2D, PrefixSelectorConfig, PrefixTrajectorySelector, SelectionContext


def _straight_route(length: int = 80) -> list[np.ndarray]:
    x = np.linspace(0.0, 40.0, length, dtype=np.float32)
    y = np.zeros_like(x)
    return [np.stack([x, y], axis=1)]


def _drivable_checker(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    return np.abs(pts[:, 1]) <= 2.0


def _context(prev: np.ndarray | None = None) -> SelectionContext:
    return SelectionContext(
        ego_pose=EgoPose2D(x=0.0, y=0.0, heading=0.0),
        route_centerlines_local=_straight_route(),
        drivable_checker=_drivable_checker,
        vehicle_half_length=0.5,
        vehicle_half_width=0.5,
        dt=0.1,
        previous_selected_local=prev,
    )


def _traj(y_end: float, *, n: int = 80, x_end: float = 20.0) -> np.ndarray:
    x = np.linspace(0.5, x_end, n, dtype=np.float32)
    y = np.linspace(0.0, y_end, n, dtype=np.float32)
    h = np.zeros_like(x)
    return np.stack([x, y, h], axis=1)


def test_selector_rejects_prefix_offroad_and_prefers_consistent_candidate() -> None:
    selector = PrefixTrajectorySelector(PrefixSelectorConfig())
    prev = _traj(0.2)
    good = CandidateTrajectory(local_xyh=_traj(0.1))
    inconsistent = CandidateTrajectory(local_xyh=_traj(1.5))
    offroad_prefix = CandidateTrajectory(local_xyh=_traj(8.0))

    result = selector.select([inconsistent, offroad_prefix, good], _context(prev=prev))

    assert result.best_index == 2
    assert result.diagnostics[1].rejected is True
    assert result.diagnostics[1].reject_reason == "prefix_offroad"
    assert result.diagnostics[2].consistency_l2 < result.diagnostics[0].consistency_l2


def test_selector_allows_late_offroad_without_prefix_reject() -> None:
    selector = PrefixTrajectorySelector(PrefixSelectorConfig())
    late_offroad = CandidateTrajectory(local_xyh=_traj(4.0))

    result = selector.select([late_offroad], _context())

    assert result.best_index == 0
    assert result.diagnostics[0].rejected is False
    assert result.diagnostics[0].prefix_offroad_steps == 0
    assert result.diagnostics[0].late_offroad_steps > 0


def test_selector_prefers_zero_offroad_candidate_even_with_zero_prefix_progress() -> None:
    selector = PrefixTrajectorySelector(PrefixSelectorConfig())
    unsafe_but_progressing = CandidateTrajectory(local_xyh=_traj(4.0, x_end=20.0))
    safe_but_stopped = CandidateTrajectory(local_xyh=_traj(0.0, x_end=0.5))

    result = selector.select([unsafe_but_progressing, safe_but_stopped], _context())

    assert result.best_index == 1
    assert result.diagnostics[1].rejected is False
    assert result.diagnostics[1].prefix_progress_m == 0.0
    assert result.diagnostics[1].prefix_progress_shortfall_m > 0.0
    assert result.diagnostics[0].late_offroad_steps > 0


def test_selector_prefers_least_total_offroad_when_all_survivors_are_unsafe() -> None:
    selector = PrefixTrajectorySelector(PrefixSelectorConfig(max_prefix_end_lateral_error_m=10.0))
    more_progress_but_more_offroad = CandidateTrajectory(local_xyh=_traj(4.0, x_end=20.0))
    slow_but_less_offroad = CandidateTrajectory(local_xyh=_traj(2.5, x_end=1.0))

    result = selector.select([more_progress_but_more_offroad, slow_but_less_offroad], _context())
    total_offroad = [d.prefix_offroad_steps + d.late_offroad_steps for d in result.diagnostics]

    assert total_offroad[0] > total_offroad[1] > 0
    assert result.best_index == 1


def test_selector_fallback_picks_least_bad_candidate_when_all_rejected() -> None:
    selector = PrefixTrajectorySelector(PrefixSelectorConfig(max_prefix_end_lateral_error_m=0.05))
    a = CandidateTrajectory(local_xyh=_traj(0.4, x_end=20.0))
    b = CandidateTrajectory(local_xyh=_traj(1.2, x_end=15.0))

    result = selector.select([a, b], _context())

    assert result.used_fallback is True
    assert result.best_index == 0
    assert result.diagnostics[0].rejected is False
    assert result.diagnostics[1].rejected is True
