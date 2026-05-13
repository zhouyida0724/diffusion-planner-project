from __future__ import annotations

import sqlite3
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


def _base_feature_arrays(extract_module):
    neighbor_past = np.zeros(
        (extract_module.MAX_NEIGHBORS, extract_module.NEIGHBOR_HISTORY_LEN, 11),
        dtype=np.float32,
    )
    neighbor_future = np.zeros(
        (extract_module.MAX_NEIGHBORS, extract_module.NEIGHBOR_FUTURE_LEN, 3),
        dtype=np.float32,
    )
    return {
        "ego_current_state": np.arange(10, dtype=np.float32),
        "ego_future": np.ones((extract_module.EGO_FUTURE_LEN, 3), dtype=np.float32),
        "neighbor_past": neighbor_past,
        "neighbor_future": neighbor_future,
        "static_objects": np.zeros((extract_module.MAX_STATIC_OBJECTS, 10), dtype=np.float32),
        "lanes": np.zeros((extract_module.MAX_LANES, extract_module.POLYLINE_LEN, 12), dtype=np.float32),
        "lanes_avails": np.zeros((extract_module.MAX_LANES, extract_module.POLYLINE_LEN), dtype=np.bool_),
        "route_lanes": np.zeros((extract_module.MAX_ROUTE_LANES, extract_module.POLYLINE_LEN, 12), dtype=np.float32),
        "route_lanes_avails": np.zeros((extract_module.MAX_ROUTE_LANES, extract_module.POLYLINE_LEN), dtype=np.bool_),
        "speed": np.zeros((extract_module.MAX_LANES,), dtype=np.float32),
        "has_speed": np.zeros((extract_module.MAX_LANES,), dtype=np.float32),
        "route_speed": np.zeros((extract_module.MAX_ROUTE_LANES,), dtype=np.float32),
        "route_has_speed": np.zeros((extract_module.MAX_ROUTE_LANES,), dtype=np.float32),
    }


def _patch_extractor_dependencies(monkeypatch, extract_module, calls: dict[str, list[tuple[float, float, float]]]):
    arrays = _base_feature_arrays(extract_module)

    monkeypatch.setattr(
        extract_module,
        "get_lidar_token_at_or_before_timestamp",
        lambda *args, **kwargs: (b"\x01" * 8, 123),
    )
    monkeypatch.setattr(
        extract_module,
        "extract_ego_data",
        lambda *args, **kwargs: (
            arrays["ego_current_state"].copy(),
            arrays["ego_future"].copy(),
            arrays["neighbor_past"].copy(),
            1.0,
            2.0,
            0.3,
            0,
            [],
        ),
    )
    monkeypatch.setattr(extract_module, "get_traffic_lights_at_timestamp", lambda *args, **kwargs: {})
    monkeypatch.setattr(extract_module, "_get_db_path_from_conn", lambda conn: "/tmp/fake.db")
    monkeypatch.setattr(
        extract_module,
        "build_nuplan_scenario_from_db",
        lambda *args, **kwargs: SimpleNamespace(get_route_roadblock_ids=lambda: ["rb"]),
    )
    monkeypatch.setattr(extract_module, "get_pruned_route_roadblock_ids", lambda *args, **kwargs: ["rb"])

    def extract_neighbor_agents(conn, center_timestamp, ego_x, ego_y, ego_heading):
        calls.setdefault("neighbors", []).append((float(ego_x), float(ego_y), float(ego_heading)))
        return arrays["neighbor_past"].copy(), arrays["neighbor_future"].copy()

    def extract_static_objects(conn, center_timestamp, ego_x, ego_y, ego_heading):
        calls.setdefault("static", []).append((float(ego_x), float(ego_y), float(ego_heading)))
        return arrays["static_objects"].copy()

    def extract_lanes(point, map_api, *, ego_heading, **kwargs):
        calls.setdefault("lanes", []).append((float(point.x), float(point.y), float(ego_heading)))
        return (
            arrays["lanes"].copy(),
            arrays["lanes_avails"].copy(),
            arrays["speed"].copy(),
            arrays["has_speed"].copy(),
        )

    def extract_route_lanes(point, map_api, *, ego_heading, **kwargs):
        calls.setdefault("route_lanes", []).append((float(point.x), float(point.y), float(ego_heading)))
        return (
            arrays["route_lanes"].copy(),
            arrays["route_lanes_avails"].copy(),
            arrays["route_speed"].copy(),
            arrays["route_has_speed"].copy(),
        )

    monkeypatch.setattr(extract_module, "extract_neighbor_agents", extract_neighbor_agents)
    monkeypatch.setattr(extract_module, "extract_static_objects", extract_static_objects)
    monkeypatch.setattr(extract_module, "extract_lanes", extract_lanes)
    monkeypatch.setattr(extract_module, "extract_route_lanes", extract_route_lanes)
    return arrays


def test_extract_features_uses_db_pose_without_local_frame_override(monkeypatch) -> None:
    _install_roadblock_utils_stub()
    from src.platform.nuplan.features import extract_single_frame as extract_module

    calls: dict[str, list[tuple[float, float, float]]] = {}
    _patch_extractor_dependencies(monkeypatch, extract_module, calls)

    features = extract_module.extract_features_at_timestamp(
        sqlite3.connect(":memory:"),
        SimpleNamespace(map_name="fake-map"),
        scenario_token_hex="01" * 8,
        timestamp_us=456,
    )

    assert calls["neighbors"] == [(1.0, 2.0, 0.3)]
    assert calls["static"] == [(1.0, 2.0, 0.3)]
    assert calls["lanes"] == [(1.0, 2.0, 0.3)]
    assert calls["route_lanes"] == [(1.0, 2.0, 0.3)]
    np.testing.assert_array_equal(features["ego_current_state"], np.arange(10, dtype=np.float32))


def test_extract_features_uses_runtime_local_frame_and_runtime_ego_overrides(monkeypatch) -> None:
    _install_roadblock_utils_stub()
    from src.platform.nuplan.features import extract_single_frame as extract_module

    calls: dict[str, list[tuple[float, float, float]]] = {}
    arrays = _patch_extractor_dependencies(monkeypatch, extract_module, calls)
    runtime_current = np.full((10,), 7.0, dtype=np.float32)
    runtime_past = np.full((extract_module.NEIGHBOR_HISTORY_LEN, 3), 8.0, dtype=np.float32)
    runtime_neighbor_slot = np.full((extract_module.NEIGHBOR_HISTORY_LEN, 11), 9.0, dtype=np.float32)

    features = extract_module.extract_features_at_timestamp(
        sqlite3.connect(":memory:"),
        SimpleNamespace(map_name="fake-map"),
        scenario_token_hex="01" * 8,
        timestamp_us=456,
        local_frame=extract_module.FeatureLocalFrame(x=10.0, y=20.0, heading=1.2, source="runtime_sim"),
        ego_current_state_override=runtime_current,
        ego_past_override=runtime_past,
        ego_neighbor_past_override=runtime_neighbor_slot,
    )

    assert calls["neighbors"] == [(10.0, 20.0, 1.2)]
    assert calls["static"] == [(10.0, 20.0, 1.2)]
    assert calls["lanes"] == [(10.0, 20.0, 1.2)]
    assert calls["route_lanes"] == [(10.0, 20.0, 1.2)]
    np.testing.assert_array_equal(features["ego_current_state"], runtime_current)
    np.testing.assert_array_equal(features["ego_past"], runtime_past)
    np.testing.assert_array_equal(features["neighbor_agents_past"][0], runtime_neighbor_slot)
    np.testing.assert_array_equal(arrays["neighbor_past"][0], np.zeros_like(arrays["neighbor_past"][0]))


def test_runtime_local_frame_reprojects_ego_future_label(monkeypatch) -> None:
    _install_roadblock_utils_stub()
    from src.platform.nuplan.features import extract_single_frame as extract_module

    def quat_from_heading(heading: float) -> dict[str, float]:
        return {
            "qw": float(np.cos(heading / 2.0)),
            "qx": 0.0,
            "qy": 0.0,
            "qz": float(np.sin(heading / 2.0)),
        }

    calls: dict[str, list[tuple[float, float, float]]] = {}
    arrays = _patch_extractor_dependencies(monkeypatch, extract_module, calls)
    all_poses = []
    for idx in range(11):
        heading = 0.0 if idx < 10 else 0.5
        all_poses.append({"x": 0.0, "y": 0.0, **quat_from_heading(heading)})
    all_poses[0]["x"] = 1.0
    all_poses[0]["y"] = 2.0
    all_poses[10]["x"] = 12.0
    all_poses[10]["y"] = 20.0

    monkeypatch.setattr(
        extract_module,
        "extract_ego_data",
        lambda *args, **kwargs: (
            arrays["ego_current_state"].copy(),
            np.full((extract_module.EGO_FUTURE_LEN, 3), -99.0, dtype=np.float32),
            arrays["neighbor_past"].copy(),
            1.0,
            2.0,
            0.3,
            0,
            all_poses,
        ),
    )

    features = extract_module.extract_features_at_timestamp(
        sqlite3.connect(":memory:"),
        SimpleNamespace(map_name="fake-map"),
        scenario_token_hex="01" * 8,
        timestamp_us=456,
        local_frame=extract_module.FeatureLocalFrame(x=10.0, y=20.0, heading=0.0, source="runtime_sim"),
    )

    np.testing.assert_allclose(features["ego_agent_future"][0], np.array([2.0, 0.0, 0.5], dtype=np.float32))


def test_build_ego_runtime_features_uses_local_current_heading() -> None:
    from tests.test_closedloop_debug_dump import _fake_ego, _planner_module

    planner_module = _planner_module()
    ego_current_state, _, _ = planner_module._build_ego_runtime_features([_fake_ego()], history_len=21)

    np.testing.assert_allclose(ego_current_state[:4], np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))


def test_paper_runtime_passes_simulated_local_frame_and_ego_overrides(monkeypatch, tmp_path) -> None:
    from tests.test_closedloop_debug_dump import _fake_current_input, _fake_feats, _planner_module

    planner_module = _planner_module()
    FeatureLocalFrame = planner_module.FeatureLocalFrame
    captured: dict[str, object] = {}

    def fake_extract_features_at_timestamp(*args, **kwargs):
        captured.update(kwargs)
        feats = _fake_feats()
        feats.update(
            {
                "ego_current_state": np.zeros((10,), dtype=np.float32),
                "ego_agent_future": np.zeros((80, 3), dtype=np.float32),
                "neighbor_agents_past": np.zeros((32, 21, 11), dtype=np.float32),
                "neighbor_agents_future": np.zeros((32, 80, 3), dtype=np.float32),
                "static_objects": np.zeros((5, 10), dtype=np.float32),
                "lanes_speed_limit": np.zeros((2,), dtype=np.float32),
                "lanes_has_speed_limit": np.zeros((2,), dtype=np.float32),
                "route_lanes_speed_limit": np.zeros((2,), dtype=np.float32),
                "route_lanes_has_speed_limit": np.zeros((2,), dtype=np.float32),
            }
        )
        return feats

    monkeypatch.setattr(planner_module, "extract_features_at_timestamp", fake_extract_features_at_timestamp)
    planner = planner_module.DiffusionPlannerCkpt.__new__(planner_module.DiffusionPlannerCkpt)
    planner._map_api = SimpleNamespace(map_name="fake-map")
    planner._scenario = SimpleNamespace(token=bytes.fromhex("01" * 8), _log_file="/tmp/fake.db")
    planner._runtime_debug = False
    planner._feature_dump_dir = str(tmp_path)
    planner._feature_dump_k = 1
    planner._feature_dump_ticks = 0
    planner._feature_sanity_debug = False
    planner._feature_sanity_ticks = 0
    planner._feature_sanity_k = 0
    planner._get_feature_conn = lambda: sqlite3.connect(":memory:")

    cfg = SimpleNamespace(
        time_len=21,
        future_len=80,
        agent_num=32,
        static_objects_num=5,
        static_objects_state_dim=10,
        lane_num=2,
        lane_len=3,
        route_num=2,
    )
    features = planner._build_paper_runtime_features(_fake_current_input(), cfg)

    local_frame = captured["local_frame"]
    assert captured["scenario_token_hex"] == "01" * 8
    assert isinstance(local_frame, FeatureLocalFrame)
    assert local_frame.x == 10.0
    assert local_frame.y == 20.0
    assert local_frame.heading == 0.25
    assert local_frame.source == "runtime_sim"
    assert local_frame.convention == "rear_axle"
    np.testing.assert_allclose(captured["ego_current_state_override"][:4], [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(features["ego_current_state"], captured["ego_current_state_override"])
    np.testing.assert_array_equal(features["ego_past"], captured["ego_past_override"])
    np.testing.assert_array_equal(features["neighbor_agents_past"][0], captured["ego_neighbor_past_override"])

    meta = next(tmp_path.glob("*.json")).read_text()
    assert '"scenario_token": "0101010101010101"' in meta
    assert '"feature_local_frame_source": "runtime_sim"' in meta
    assert '"feature_local_frame_convention": "rear_axle"' in meta
