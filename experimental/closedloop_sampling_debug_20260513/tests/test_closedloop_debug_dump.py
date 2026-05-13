from __future__ import annotations

import json
import math
import sqlite3
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _install_extract_single_frame_stub() -> None:
    module_name = "src.platform.nuplan.features.extract_single_frame"
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)

    @dataclass(frozen=True)
    class FeatureLocalFrame:
        x: float
        y: float
        heading: float
        source: str = "db_log"
        convention: str = "rear_axle"

    stub.FeatureLocalFrame = FeatureLocalFrame
    for name in (
        "extract_lanes",
        "extract_neighbor_agents",
        "extract_route_lanes",
        "extract_static_objects",
        "extract_features_at_timestamp",
        "get_traffic_lights_at_timestamp",
    ):
        setattr(stub, name, lambda *args, **kwargs: None)

    def get_lidar_token_at_or_before_timestamp(conn, *, timestamp_us: int, scenario_token_hex: str | None = None, debug_log: bool = False):
        cur = conn.cursor()
        lidar_token = bytes.fromhex(str(scenario_token_hex))
        cur.execute("SELECT scene_token FROM lidar_pc WHERE token = ? LIMIT 1", (lidar_token,))
        scene_row = cur.fetchone()
        if scene_row is None:
            raise AssertionError("missing scenario lidar token")
        cur.execute(
            """
            SELECT token, timestamp
            FROM lidar_pc
            WHERE scene_token = ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (scene_row[0], int(timestamp_us)),
        )
        row = cur.fetchone()
        if row is None:
            raise AssertionError("missing lidar token at or before timestamp")
        return row[0], int(row[1])

    def quaternion_to_heading(qw, qx, qy, qz):
        return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    stub.get_lidar_token_at_or_before_timestamp = get_lidar_token_at_or_before_timestamp
    stub.quaternion_to_heading = quaternion_to_heading
    sys.modules[module_name] = stub


def _planner_module():
    _install_extract_single_frame_stub()
    import src.platform.nuplan.planners.diffusion_planner_ckpt_planner as planner_module

    return planner_module


def _fake_ego() -> SimpleNamespace:
    return SimpleNamespace(
        rear_axle=SimpleNamespace(x=10.0, y=20.0, heading=0.25),
        center=SimpleNamespace(x=11.0, y=21.0, heading=0.25),
        time_us=1234567,
        dynamic_car_state=SimpleNamespace(),
        car_footprint=SimpleNamespace(
            vehicle_parameters=SimpleNamespace(width=2.1, length=4.8, half_width=1.05, half_length=2.4),
            rear_axle_to_center_dist=1.4,
        ),
    )


def _fake_current_input() -> SimpleNamespace:
    return SimpleNamespace(
        iteration=SimpleNamespace(index=7, time_us=1234567),
        history=SimpleNamespace(ego_states=[_fake_ego()]),
    )


def _fake_feats() -> dict[str, np.ndarray]:
    route_lanes_avails = np.zeros((2, 3), dtype=np.float32)
    route_lanes_avails[0, :2] = 1.0
    lanes_avails = np.ones((2, 3), dtype=np.float32)
    return {
        "route_lanes": np.ones((2, 3, 12), dtype=np.float32),
        "route_lanes_avails": route_lanes_avails,
        "lanes": np.full((2, 3, 12), 2.0, dtype=np.float32),
        "lanes_avails": lanes_avails,
    }


def _configured_planner(tmp_path: Path, monkeypatch, enabled: bool):
    planner_module = _planner_module()
    planner = planner_module.DiffusionPlannerCkpt.__new__(planner_module.DiffusionPlannerCkpt)
    planner._scenario = SimpleNamespace(token="scenario/token:01", log_name="log/name:01", _log_file="/tmp/log.db")
    if enabled:
        monkeypatch.setenv("DP_CLOSEDLOOP_DEBUG_DUMP_DIR", str(tmp_path))
        monkeypatch.setenv("DP_CLOSEDLOOP_DEBUG_START_TICK", "0")
        monkeypatch.setenv("DP_CLOSEDLOOP_DEBUG_END_TICK", "-1")
        monkeypatch.setenv("DP_CLOSEDLOOP_DEBUG_CANDIDATES", "1")
    else:
        monkeypatch.delenv("DP_CLOSEDLOOP_DEBUG_DUMP_DIR", raising=False)
        monkeypatch.delenv("DP_CLOSEDLOOP_DEBUG_START_TICK", raising=False)
        monkeypatch.delenv("DP_CLOSEDLOOP_DEBUG_END_TICK", raising=False)
        monkeypatch.delenv("DP_CLOSEDLOOP_DEBUG_CANDIDATES", raising=False)
    planner._configure_closedloop_debug_dump_from_env()
    return planner


def test_closedloop_debug_dump_disabled_writes_nothing_and_preserves_arrays(tmp_path: Path, monkeypatch) -> None:
    planner = _configured_planner(tmp_path, monkeypatch, enabled=False)
    selected_local = np.array([[1.0, 2.0, 0.1], [3.0, 4.0, 0.2]], dtype=np.float32)
    selected_world = selected_local + np.array([10.0, 20.0, 0.25], dtype=np.float32)
    before = selected_local.copy()

    planner._maybe_dump_closedloop_debug(
        current_input=_fake_current_input(),
        tick=0,
        selected_local_xyh=selected_local,
        selected_world_xyh=selected_world,
        feats_np=_fake_feats(),
        candidates_local_xyh=[selected_local + 1.0],
    )

    assert list(tmp_path.iterdir()) == []
    np.testing.assert_array_equal(selected_local, before)


def test_closedloop_debug_dump_enabled_writes_json_and_npz_payload(tmp_path: Path, monkeypatch) -> None:
    planner = _configured_planner(tmp_path, monkeypatch, enabled=True)
    planner._last_feature_local_frame_meta = {
        "feature_local_frame_source": "runtime_sim",
        "feature_local_frame": {"x": 10.0, "y": 20.0, "heading": 0.25},
        "feature_local_frame_convention": "rear_axle",
    }
    selected_local = np.array([[1.0, 2.0, 0.1], [3.0, 4.0, 0.2]], dtype=np.float32)
    selected_world = selected_local + np.array([10.0, 20.0, 0.25], dtype=np.float32)

    planner._maybe_dump_closedloop_debug(
        current_input=_fake_current_input(),
        tick=0,
        selected_local_xyh=selected_local,
        selected_world_xyh=selected_world,
        feats_np=_fake_feats(),
        candidates_local_xyh=[selected_local + 1.0, selected_local + 2.0],
    )

    json_paths = sorted(tmp_path.glob("*.json"))
    npz_paths = sorted(tmp_path.glob("*.npz"))
    assert len(json_paths) == 1
    assert len(npz_paths) == 1
    assert "scenario_token_01" in json_paths[0].stem
    assert "tick000000" in json_paths[0].stem
    assert "iter7" in json_paths[0].stem
    assert "ts1234567" in json_paths[0].stem

    meta = json.loads(json_paths[0].read_text())
    assert meta["scenario_token"] == "scenario/token:01"
    assert meta["scenario_log"] == "log/name:01"
    assert meta["tick"] == 0
    assert meta["iteration_index"] == 7
    assert meta["timestamp_us"] == 1234567
    assert meta["route_lane_count"] == 1
    assert meta["runtime_rear_axle"] == {"x": 10.0, "y": 20.0, "heading": 0.25, "time_us": 1234567}
    assert meta["feature_local_frame_source"] == "runtime_sim"
    assert meta["feature_local_frame"] == {"x": 10.0, "y": 20.0, "heading": 0.25}
    assert meta["feature_local_frame_convention"] == "rear_axle"
    assert meta["db_rear_axle"] is None
    assert meta["db_minus_runtime_world"] is None
    assert meta["runtime_in_db_local"] is None
    assert meta["env_config"]["DP_CLOSEDLOOP_DEBUG_CANDIDATES"] == 1
    assert len(meta["feature_sha1"]) == 40
    assert meta["selected_local_trajectory_stats"]["num_poses"] == 2
    assert meta["route_feature_sanity"]["route_lane_count"] == 1
    assert meta["route_feature_sanity"]["route_avail_points"] == 2
    assert meta["candidate_route_projection"]["num_candidates"] == 1
    assert meta["candidate_diversity"]["num_candidates"] == 1
    assert meta["candidate_obstacle_proxies"]["num_candidates"] == 1

    payload = np.load(npz_paths[0])
    assert set(
        [
            "selected_local_xyh",
            "selected_world_xyh",
            "closed_loop_y",
            "selected_xyh",
            "selected_world_rear_xyh",
            "selected_center_world_xyh",
            "selected_corners_world_xy",
            "selected_corner_drivable",
            "selected_center_drivable",
            "selected_rear_drivable",
            "route_lanes",
            "route_lanes_avails",
            "lanes",
            "lanes_avails",
            "candidates_local_xyh",
            "candidate_xyh",
            "candidate_route_projection_final_lateral_error_m",
            "candidate_route_projection_remaining_to_route_end_m",
            "candidate_dynamic_min_clearance_m",
            "candidate_static_min_clearance_m",
        ]
    ).issubset(payload.files)
    np.testing.assert_array_equal(payload["selected_local_xyh"], selected_local)
    np.testing.assert_array_equal(payload["closed_loop_y"], selected_local)
    np.testing.assert_array_equal(payload["selected_xyh"], selected_local)
    np.testing.assert_array_equal(payload["selected_world_rear_xyh"], selected_world)
    assert payload["selected_center_world_xyh"].shape == (2, 3)
    assert payload["selected_corners_world_xy"].shape == (2, 4, 2)
    assert payload["selected_corner_drivable"].shape == (2, 4)
    assert payload["selected_center_drivable"].shape == (2,)
    assert payload["selected_rear_drivable"].shape == (2,)
    assert payload["candidates_local_xyh"].shape == (1, 2, 3)
    np.testing.assert_array_equal(payload["candidate_xyh"], payload["candidates_local_xyh"])
    assert payload["candidate_route_projection_final_lateral_error_m"].shape == (1,)
    assert payload["candidate_route_projection_remaining_to_route_end_m"].shape == (1,)
    assert payload["candidate_dynamic_min_clearance_m"].shape == (1,)
    assert payload["candidate_static_min_clearance_m"].shape == (1,)
    assert meta["vehicle_dimensions"]["rear_axle_to_center"] == 1.4
    assert "selected_corner_offroad_count" in meta


def test_db_ego_pose_at_timestamp_extracts_pose_and_lidar_metadata() -> None:
    planner_module = _planner_module()
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE lidar_pc (token BLOB, timestamp INTEGER, scene_token BLOB)")
    conn.execute("CREATE TABLE scene (token BLOB, log_token BLOB)")
    conn.execute("CREATE TABLE ego_pose (log_token BLOB, timestamp INTEGER, x REAL, y REAL, qw REAL, qx REAL, qy REAL, qz REAL)")
    scenario_token = bytes.fromhex("abababababababab")
    scene_token = b"scene"
    log_token = b"log"
    heading = 0.4
    conn.execute("INSERT INTO lidar_pc VALUES (?, ?, ?)", (scenario_token, 1_000, scene_token))
    conn.execute("INSERT INTO scene VALUES (?, ?)", (scene_token, log_token))
    conn.execute(
        "INSERT INTO ego_pose VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (log_token, 995, 12.0, 23.0, math.cos(heading / 2.0), 0.0, 0.0, math.sin(heading / 2.0)),
    )

    pose = planner_module._db_ego_pose_at_timestamp(conn, scenario_token.hex(), 1_005)

    assert pose is not None
    assert pose["x"] == 12.0
    assert pose["y"] == 23.0
    assert pose["timestamp_us"] == 995
    assert pose["lidar_timestamp_us"] == 1_000
    assert pose["lidar_token"] == scenario_token.hex()
    assert pose["heading"] == pytest.approx(heading)


def test_closedloop_debug_dump_includes_db_runtime_pose_delta(tmp_path: Path, monkeypatch) -> None:
    planner = _configured_planner(tmp_path, monkeypatch, enabled=True)
    scenario_token = bytes.fromhex("cdcdcdcdcdcdcdcd")
    planner._scenario = SimpleNamespace(token=scenario_token.hex(), log_name="log/name:01", _log_file="/tmp/log.db")

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE lidar_pc (token BLOB, timestamp INTEGER, scene_token BLOB)")
    conn.execute("CREATE TABLE scene (token BLOB, log_token BLOB)")
    conn.execute("CREATE TABLE ego_pose (log_token BLOB, timestamp INTEGER, x REAL, y REAL, qw REAL, qx REAL, qy REAL, qz REAL)")
    scene_token = b"scene"
    log_token = b"log"
    db_heading = 0.15
    conn.execute("INSERT INTO lidar_pc VALUES (?, ?, ?)", (scenario_token, 1_234_500, scene_token))
    conn.execute("INSERT INTO scene VALUES (?, ?)", (scene_token, log_token))
    conn.execute(
        "INSERT INTO ego_pose VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (log_token, 1_234_400, 8.0, 18.0, math.cos(db_heading / 2.0), 0.0, 0.0, math.sin(db_heading / 2.0)),
    )
    monkeypatch.setattr(planner, "_get_feature_conn", lambda: conn)
    selected_local = np.array([[1.0, 2.0, 0.1], [3.0, 4.0, 0.2]], dtype=np.float32)
    selected_world = selected_local + np.array([10.0, 20.0, 0.25], dtype=np.float32)

    planner._maybe_dump_closedloop_debug(
        current_input=_fake_current_input(),
        tick=0,
        selected_local_xyh=selected_local,
        selected_world_xyh=selected_world,
        feats_np=_fake_feats(),
    )

    meta = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert meta["runtime_rear_axle"] == {"x": 10.0, "y": 20.0, "heading": 0.25, "time_us": 1234567}
    assert meta["db_rear_axle"]["x"] == 8.0
    assert meta["db_rear_axle"]["y"] == 18.0
    assert meta["db_rear_axle"]["heading"] == pytest.approx(db_heading)
    assert meta["db_rear_axle"]["timestamp_us"] == 1_234_400
    assert meta["db_rear_axle"]["lidar_timestamp_us"] == 1_234_500
    assert meta["db_rear_axle"]["lidar_token"] == scenario_token.hex()
    assert meta["db_minus_runtime_world"] == pytest.approx({"dx": -2.0, "dy": -2.0, "dheading": -0.1})

    c = math.cos(db_heading)
    s = math.sin(db_heading)
    expected_local_x = c * 2.0 + s * 2.0
    expected_local_y = -s * 2.0 + c * 2.0
    assert meta["runtime_in_db_local"] == pytest.approx(
        {"x": expected_local_x, "y": expected_local_y, "heading": 0.1}
    )
