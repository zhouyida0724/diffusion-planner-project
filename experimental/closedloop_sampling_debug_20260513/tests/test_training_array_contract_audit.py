import math

import numpy as np

from scripts.analysis.dp_training_array_contract_audit import (
    _future_time_indices,
    _min_distances_to_route,
    _trajectory_dynamics,
)


def test_future_time_indices_clamp_to_available_horizon() -> None:
    assert _future_time_indices(80, hz=10.0) == {"1s": 9, "3s": 29, "5s": 49, "final": 79}
    assert _future_time_indices(12, hz=10.0) == {"1s": 9, "3s": 11, "5s": 11, "final": 11}


def test_min_distances_to_route_use_availability_mask() -> None:
    future_xy = np.array([[[0.0, 0.0], [3.0, 4.0]]], dtype=np.float32)
    route_lanes = np.zeros((1, 2, 2, 12), dtype=np.float32)
    route_lanes[0, 0, :, :2] = np.array([[0.0, 1.0], [100.0, 100.0]], dtype=np.float32)
    route_lanes[0, 1, :, :2] = np.array([[3.0, 0.0], [3.0, 5.0]], dtype=np.float32)
    route_avails = np.array([[[True, False], [True, True]]])

    distances, nearest_indices = _min_distances_to_route(future_xy, route_lanes, route_avails)

    np.testing.assert_allclose(distances, np.array([[1.0, 1.0]]), atol=1e-6)
    assert nearest_indices.tolist() == [[0, 3]]


def test_trajectory_dynamics_reports_curvature_and_lateral_accel_proxy() -> None:
    theta = np.linspace(0.0, math.pi / 2.0, 11, dtype=np.float32)
    radius = 10.0
    xy = np.stack([radius * np.sin(theta), radius * (1.0 - np.cos(theta))], axis=1)
    heading = theta[:, None]
    future = np.concatenate([xy, heading], axis=1)[None, :, :]

    metrics = _trajectory_dynamics(future, dt=0.1)

    assert metrics["valid_samples"] == 1
    assert metrics["max_abs_heading_change_rad"][0] > 1.5
    assert 0.05 < metrics["max_abs_curvature_1pm"][0] < 0.2
    assert metrics["max_abs_lateral_accel_proxy_mps2"][0] > 1.0
