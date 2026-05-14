from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace

import numpy as np


def _install_roadblock_utils_stub() -> None:
    module_name = "nuplan.diffusion_planner.data_process.roadblock_utils"
    if module_name in sys.modules:
        return

    for package_name in (
        "nuplan.diffusion_planner",
        "nuplan.diffusion_planner.data_process",
    ):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package

    stub = types.ModuleType(module_name)
    stub.BreadthFirstSearchRoadBlock = object
    stub.route_roadblock_correction = lambda *args, **kwargs: []
    sys.modules[module_name] = stub


def _planner_module():
    _install_roadblock_utils_stub()
    import src.platform.nuplan.planners.diffusion_planner_ckpt_planner as planner_module

    return planner_module


def _ego_state(x: float, y: float, heading: float) -> SimpleNamespace:
    return SimpleNamespace(
        rear_axle=SimpleNamespace(x=x, y=y, heading=heading),
        center=SimpleNamespace(x=x, y=y, heading=heading),
        dynamic_car_state=SimpleNamespace(
            center_velocity_2d=SimpleNamespace(x=0.0, y=0.0),
            center_acceleration_2d=SimpleNamespace(x=0.0, y=0.0),
        ),
        car_footprint=SimpleNamespace(vehicle_parameters=SimpleNamespace(width=2.1, length=4.8)),
    )


def test_runtime_ego_neighbor_slot_heading_is_relative_to_current_ego() -> None:
    planner_module = _planner_module()
    past_heading = math.radians(30.0)
    current_heading = math.radians(60.0)

    neighbor_slot = planner_module._build_ego_neighbor_slot(
        [_ego_state(0.0, 0.0, past_heading), _ego_state(1.0, 0.0, current_heading)],
        time_len=3,
    )

    np.testing.assert_allclose(neighbor_slot[-1, 2:4], np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        neighbor_slot[-2, 2:4],
        np.array([math.cos(past_heading - current_heading), math.sin(past_heading - current_heading)], dtype=np.float32),
        atol=1e-6,
    )
