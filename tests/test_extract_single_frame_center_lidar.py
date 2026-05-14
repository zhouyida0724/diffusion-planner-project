from __future__ import annotations

import sqlite3
import sys
import types

import pytest


def _install_extract_single_frame_import_stubs() -> None:
    def module(name: str) -> types.ModuleType:
        stub = types.ModuleType(name)
        sys.modules.setdefault(name, stub)
        return sys.modules[name]

    state_representation = module("nuplan.common.actor_state.state_representation")
    state_representation.Point2D = object
    vehicle_parameters = module("nuplan.common.actor_state.vehicle_parameters")
    vehicle_parameters.get_pacifica_parameters = lambda: None
    maps_datatypes = module("nuplan.common.maps.maps_datatypes")
    maps_datatypes.SemanticMapLayer = object
    nuplan_scenario = module("nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario")
    nuplan_scenario.NuPlanScenario = object
    roadblock_utils = module("nuplan.diffusion_planner.data_process.roadblock_utils")
    roadblock_utils.BreadthFirstSearchRoadBlock = object
    roadblock_utils.route_roadblock_correction = lambda *args, **kwargs: None


_install_extract_single_frame_import_stubs()

from src.platform.nuplan.features.extract_single_frame import get_target_frame


def _create_center_lidar_db() -> tuple[sqlite3.Connection, bytes, bytes]:
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE scene (token BLOB PRIMARY KEY, log_token BLOB)")
    cursor.execute(
        "CREATE TABLE ego_pose (token BLOB PRIMARY KEY, timestamp INTEGER, log_token BLOB)"
    )
    cursor.execute(
        "CREATE TABLE lidar_pc (token BLOB PRIMARY KEY, timestamp INTEGER, scene_token BLOB, ego_pose_token BLOB)"
    )

    scene_token = bytes.fromhex("01" * 16)
    log_token = bytes.fromhex("02" * 16)
    cursor.execute("INSERT INTO scene VALUES (?, ?)", (scene_token, log_token))
    return conn, scene_token, log_token


def test_scene_lidar_returns_target_lidar_ego_pose(monkeypatch) -> None:
    monkeypatch.setenv("EXPORT_FRAME_INDEX_MODE", "scene_lidar")
    conn, scene_token, log_token = _create_center_lidar_db()
    cursor = conn.cursor()
    earlier_ego_token = bytes.fromhex("03" * 16)
    target_ego_token = bytes.fromhex("04" * 16)
    target_lidar_token = bytes.fromhex("05" * 16)

    cursor.execute(
        "INSERT INTO ego_pose VALUES (?, ?, ?)",
        (earlier_ego_token, 90, log_token),
    )
    cursor.execute(
        "INSERT INTO ego_pose VALUES (?, ?, ?)",
        (target_ego_token, 95, log_token),
    )
    cursor.execute(
        "INSERT INTO lidar_pc VALUES (?, ?, ?, ?)",
        (target_lidar_token, 100, scene_token, target_ego_token),
    )

    center_token, center_timestamp, _ = get_target_frame(conn, scene_token.hex(), 0)

    assert center_token == target_ego_token
    assert center_timestamp == 100
    cursor.execute(
        """
        SELECT token
        FROM lidar_pc
        WHERE timestamp <= ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (center_timestamp,),
    )
    assert cursor.fetchone()[0] == target_lidar_token


def test_scene_lidar_out_of_range_clamps_to_last_lidar(monkeypatch) -> None:
    monkeypatch.setenv("EXPORT_FRAME_INDEX_MODE", "scene_lidar")
    conn, scene_token, log_token = _create_center_lidar_db()
    cursor = conn.cursor()
    first_ego_token = bytes.fromhex("06" * 16)
    last_ego_token = bytes.fromhex("07" * 16)
    first_lidar_token = bytes.fromhex("08" * 16)
    last_lidar_token = bytes.fromhex("09" * 16)

    cursor.execute("INSERT INTO ego_pose VALUES (?, ?, ?)", (first_ego_token, 95, log_token))
    cursor.execute("INSERT INTO ego_pose VALUES (?, ?, ?)", (last_ego_token, 145, log_token))
    cursor.execute(
        "INSERT INTO lidar_pc VALUES (?, ?, ?, ?)",
        (first_lidar_token, 100, scene_token, first_ego_token),
    )
    cursor.execute(
        "INSERT INTO lidar_pc VALUES (?, ?, ?, ?)",
        (last_lidar_token, 150, scene_token, last_ego_token),
    )

    center_token, center_timestamp, _ = get_target_frame(conn, scene_token.hex(), 999)

    assert center_token == last_ego_token
    assert center_timestamp == 150


def test_scene_lidar_missing_target_ego_pose_raises(monkeypatch) -> None:
    monkeypatch.setenv("EXPORT_FRAME_INDEX_MODE", "scene_lidar")
    conn, scene_token, log_token = _create_center_lidar_db()
    cursor = conn.cursor()
    missing_ego_token = bytes.fromhex("0a" * 16)
    target_lidar_token = bytes.fromhex("0b" * 16)
    legacy_ego_token = bytes.fromhex("0c" * 16)

    cursor.execute(
        "INSERT INTO ego_pose VALUES (?, ?, ?)",
        (legacy_ego_token, 90, log_token),
    )
    cursor.execute(
        "INSERT INTO lidar_pc VALUES (?, ?, ?, ?)",
        (target_lidar_token, 100, scene_token, missing_ego_token),
    )

    with pytest.raises(RuntimeError, match="ego_pose"):
        get_target_frame(conn, scene_token.hex(), 0)
