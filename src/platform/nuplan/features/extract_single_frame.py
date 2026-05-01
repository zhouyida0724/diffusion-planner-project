#!/usr/bin/env python3
"""Core single-frame feature extraction for nuPlan scenarios.

This module is the reusable implementation behind:
  - scripts/extract_single_frame/extract_single_frame.py

Design goals:
  - Keep the legacy script entrypoint/CLI behavior unchanged.
  - Provide a stable import path for other tooling (batch exporters, tests).

NOTE: This file intentionally mirrors the legacy extractor's logic closely to
avoid changing outputs.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import time
from collections import Counter

import numpy as np
from shapely import LineString

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

from nuplan.diffusion_planner.data_process.roadblock_utils import (
    BreadthFirstSearchRoadBlock,
    route_roadblock_correction,
)

# -------------------------------------------------------------------------------------------------
# Defaults / constants (kept identical to legacy script)
# -------------------------------------------------------------------------------------------------

# Prefer container-style /workspace paths; fall back to repo-local paths when running on host.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_NUPLAN_DATA_ROOT_DEFAULT = "/workspace/data/nuplan"
_NUPLAN_DATA_ROOT_FALLBACK = os.path.join(_REPO_ROOT, "data", "nuplan")
_NUPLAN_DATA_ROOT = os.environ.get("NUPLAN_DATA_ROOT", _NUPLAN_DATA_ROOT_DEFAULT)
if not os.path.isdir(_NUPLAN_DATA_ROOT):
    _NUPLAN_DATA_ROOT = _NUPLAN_DATA_ROOT_FALLBACK

DB_PATH = os.environ.get(
    "NUPLAN_DB_PATH",
    os.path.join(
        _NUPLAN_DATA_ROOT,
        "data",
        "cache",
        "mini",
        "2021.06.14.17.26.26_veh-38_04544_04920.db",
    ),
)
# Support both legacy NUPLAN_MAP_ROOT and nuPlan's more common NUPLAN_MAPS_ROOT.
MAP_ROOT = os.environ.get(
    "NUPLAN_MAP_ROOT",
    os.environ.get("NUPLAN_MAPS_ROOT", os.path.join(_NUPLAN_DATA_ROOT, "maps")),
)
MAP_VERSION = os.environ.get("NUPLAN_MAP_VERSION", "9.12.1817")
MAP_NAME = os.environ.get("NUPLAN_MAP_NAME", "us-nv-las-vegas-strip")

# Feature dimensions
EGO_FUTURE_LEN = 80
EGO_HISTORY_LEN = 21  # 21 frames history, 10Hz sampling
NEIGHBOR_HISTORY_LEN = 21
NEIGHBOR_FUTURE_LEN = 81  # 8 sec = 81 points @ 0.1s
MAX_NEIGHBORS = 32
MAX_STATIC_OBJECTS = 5
MAX_LANES = 70
MAX_ROUTE_LANES = 25
POLYLINE_LEN = 20
LANE_DIM = 12

# Debug logging paths
_DEBUG_LOG_DIR_DEFAULT = "/workspace/data_process/debug_log"
_DEBUG_LOG_DIR_FALLBACK = os.path.join(_REPO_ROOT, "data_process", "debug_log")

# Log style control:
# - legacy: verbose per-sample debug logging
# - quiet (default): suppress per-sample INFO logs; aggregate/limit WARNING prints
EXTRACT_LOG_STYLE = os.environ.get("EXTRACT_LOG_STYLE", "quiet").strip().lower()

# Aggregated warning stats for batch exporters to read.
LOG_WARNING_COUNTS: Counter[str] = Counter()
LOG_WARNING_TOTAL: int = 0


class _QuietLogger:
    def __init__(self, *, print_first_n_per_key: int = 3):
        self._print_first_n_per_key = int(print_first_n_per_key)

    def info(self, msg: str, *args, **kwargs):
        return

    def warning(self, msg: str, *args, **kwargs):
        global LOG_WARNING_TOTAL
        LOG_WARNING_TOTAL += 1
        key = str(msg).split("\n", 1)[0][:200]
        LOG_WARNING_COUNTS[key] += 1

        if LOG_WARNING_COUNTS[key] <= self._print_first_n_per_key:
            try:
                print(f"[WARN] {msg}")
            except Exception:
                pass

    def error(self, msg: str, *args, **kwargs):
        try:
            print(f"[ERROR] {msg}")
        except Exception:
            pass


def _init_debug_logger(*, output_path: str | None = None):
    """Initialize a per-run debug logger writing to debug_log dir."""
    logger = logging.getLogger("extract_single_frame_debug")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    debug_dir = _DEBUG_LOG_DIR_DEFAULT
    try:
        os.makedirs(debug_dir, exist_ok=True)
        test_path = os.path.join(debug_dir, ".write_test")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
    except Exception:
        debug_dir = os.path.abspath(_DEBUG_LOG_DIR_FALLBACK)
        os.makedirs(debug_dir, exist_ok=True)

    base = "extract_single_frame"
    if output_path:
        base = os.path.splitext(os.path.basename(output_path))[0]
    log_path = os.path.join(debug_dir, f"{base}.log")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    logger.info("=== extract_single_frame debug start ===")
    logger.info(f"log_path={log_path}")
    return logger


# Module logger used by helper functions.
if EXTRACT_LOG_STYLE == "legacy":
    DEBUG_LOGGER = _init_debug_logger(output_path=None)
else:
    DEBUG_LOGGER = _QuietLogger(print_first_n_per_key=int(os.environ.get("EXTRACT_WARN_PRINT_N", "3")))


def reset_log_stats() -> None:
    global LOG_WARNING_TOTAL
    LOG_WARNING_COUNTS.clear()
    LOG_WARNING_TOTAL = 0


def get_log_stats() -> dict:
    return {
        "warning_total": int(LOG_WARNING_TOTAL),
        "warning_by_key": dict(LOG_WARNING_COUNTS),
    }


# -------------------------------------------------------------------------------------------------
# Optional timing profiling (gated by env var)
# -------------------------------------------------------------------------------------------------


def _extract_profile_enabled() -> bool:
    v = os.environ.get("EXTRACT_PROFILE", "0")
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


@contextlib.contextmanager
def _timing_ctx(timing: dict[str, float] | None, key: str):
    if timing is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        timing[key] = float(timing.get(key, 0.0) + dt)


# -------------------------------------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------------------------------------


def quaternion_to_heading(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def get_rotation_matrix_2d(heading):
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    return np.array([[cos_h, -sin_h], [sin_h, cos_h]])


def transform_to_ego_frame(x, y, ego_x, ego_y, ego_heading):
    dx = x - ego_x
    dy = y - ego_y
    R = get_rotation_matrix_2d(-ego_heading)
    local_x, local_y = R @ np.array([dx, dy])
    return local_x, local_y


# -------------------------------------------------------------------------------------------------
# DB / scenario / map helpers
# -------------------------------------------------------------------------------------------------


def load_db(db_path: str | None = None):
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_location_from_log(conn: sqlite3.Connection) -> str | None:
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT location FROM log LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return None
        loc = row[0]
        if loc is None:
            return None
        return str(loc)
    except Exception:
        return None


def map_name_from_location(location: str | None) -> str:
    if not location:
        return "us-ma-boston"

    known_map_names = {
        "us-nv-las-vegas-strip",
        "us-ma-boston",
        "us-pa-pittsburgh-hazelwood",
        "sg-one-north",
    }
    if location in known_map_names:
        return location

    location_to_map = {
        "las_vegas": "us-nv-las-vegas-strip",
        "boston": "us-ma-boston",
        "pittsburgh": "us-pa-pittsburgh-hazelwood",
        "singapore": "sg-one-north",
    }
    return location_to_map.get(location, "us-ma-boston")


def build_nuplan_scenario_from_db(
    conn: sqlite3.Connection,
    db_path: str,
    scene_token_hex: str,
    map_name: str,
) -> NuPlanScenario:
    cursor = conn.cursor()
    scene_token_bytes = bytes.fromhex(scene_token_hex)

    cursor.execute("SELECT name FROM scene WHERE token = ? LIMIT 1", (scene_token_bytes,))
    row = cursor.fetchone()
    scenario_type = str(row[0]) if row and row[0] is not None else "unknown"

    cursor.execute(
        "SELECT token, timestamp FROM lidar_pc WHERE scene_token = ? ORDER BY timestamp ASC LIMIT 1",
        (scene_token_bytes,),
    )
    lidar_row = cursor.fetchone()
    assert lidar_row is not None, f"Unable to find lidar_pc row for scene_token={scene_token_hex}"

    initial_lidar_token_hex = (
        lidar_row[0].hex() if isinstance(lidar_row[0], (bytes, bytearray)) else str(lidar_row[0])
    )
    initial_lidar_timestamp = int(lidar_row[1])

    data_root = os.path.dirname(db_path)

    return NuPlanScenario(
        data_root=data_root,
        log_file_load_path=db_path,
        initial_lidar_token=initial_lidar_token_hex,
        initial_lidar_timestamp=initial_lidar_timestamp,
        scenario_type=scenario_type,
        map_root=MAP_ROOT,
        map_version=MAP_VERSION,
        map_name=map_name,
        scenario_extraction_info=None,
        ego_vehicle_parameters=get_pacifica_parameters(),
        sensor_root=None,
    )


def get_pruned_route_roadblock_ids(
    conn: sqlite3.Connection,
    db_path: str,
    scene_token_hex: str,
    map_api,
    map_name: str,
) -> list[str]:
    scenario = build_nuplan_scenario_from_db(conn, db_path, scene_token_hex, map_name)
    route_roadblock_ids = list(scenario.get_route_roadblock_ids())

    try:
        ego_state_0 = scenario.get_ego_state_at_iteration(0)
        route_roadblock_ids = list(route_roadblock_correction(ego_state_0, map_api, route_roadblock_ids))
    except Exception as e:
        try:
            DEBUG_LOGGER.warning(f"route_roadblock_correction failed, using raw route ids: {e}")
        except Exception:
            pass

    deduped: list[str] = []
    seen = set()
    for rb_id in route_roadblock_ids:
        if rb_id in seen:
            continue
        seen.add(rb_id)
        deduped.append(rb_id)
    return deduped


def _dedup_keep_order(seq: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in seq:
        if x is None:
            continue
        sx = str(x)
        if sx in seen:
            continue
        seen.add(sx)
        out.append(sx)
    return out


def bfs_bridge_route_if_needed(
    map_api,
    ego_point: Point2D,
    pruned_route_ids: list[str],
    *,
    intersection_pruned: int,
    radius: float = 150.0,
    k_targets: int = 10,
    max_depth: int = 80,
):
    pruned_route_ids = [str(x) for x in (pruned_route_ids or [])]
    if not pruned_route_ids:
        return pruned_route_ids, 0, False, None, "empty_pruned_route"

    try:
        layers = map_api.get_proximal_map_objects(
            ego_point,
            radius,
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        )
    except Exception:
        layers = {}

    nearest = None
    nearest_dist = float("inf")
    for layer_type in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
        for lane_obj in (layers.get(layer_type, []) or []):
            try:
                poly = lane_obj.polygon
                if poly is None:
                    continue
                c = poly.centroid
                d = float(((c.x - ego_point.x) ** 2 + (c.y - ego_point.y) ** 2) ** 0.5)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest = lane_obj
            except Exception:
                continue

    ego_rb = None
    try:
        if nearest is not None and hasattr(nearest, "get_roadblock_id"):
            ego_rb = str(nearest.get_roadblock_id())
    except Exception:
        ego_rb = None

    if not ego_rb:
        return pruned_route_ids, 0, False, None, "no_ego_rb"

    if intersection_pruned != 0:
        return pruned_route_ids, 0, False, ego_rb, f"skip_intersection_pruned={intersection_pruned}"

    if ego_rb in set(pruned_route_ids):
        return pruned_route_ids, 0, True, ego_rb, "ego_rb_in_pruned_route"

    bfs_max_time_s = 0.5
    try:
        bfs_max_time_s = float(os.environ.get("BFS_MAX_TIME_S", "0.5"))
    except Exception:
        bfs_max_time_s = 0.5

    try:
        bfs = BreadthFirstSearchRoadBlock(ego_rb, map_api)
        (bridge_path, bridge_ids), found = bfs.search(
            target_roadblock_id=pruned_route_ids[:k_targets],
            max_depth=max_depth,
            max_time_s=bfs_max_time_s,
        )
        bridge_ids = [str(x) for x in (bridge_ids or [])]
    except TimeoutError:
        return pruned_route_ids, 0, False, ego_rb, "bfs_exception: timeout"
    except Exception as e:
        return pruned_route_ids, 0, False, ego_rb, f"bfs_exception: {e}"

    if not found or not bridge_ids:
        return pruned_route_ids, 0, False, ego_rb, "bfs_not_found"

    new_route = _dedup_keep_order(bridge_ids + pruned_route_ids)
    return new_route, int(len(bridge_ids)), True, ego_rb, "bfs_bridge_found"


# -------------------------------------------------------------------------------------------------
# Raw DB queries / feature builders (kept close to legacy)
# -------------------------------------------------------------------------------------------------


def get_target_frame(conn, scenario_token, frame_index):
    cursor = conn.cursor()

    try:
        scenario_token_bytes = bytes.fromhex(scenario_token)
        cursor.execute("SELECT token FROM scene WHERE token = ?", (scenario_token_bytes,))
        scenario = cursor.fetchone()

        if scenario:
            # First get log_token from scene (matching extract_ego_data approach)
            cursor.execute("SELECT log_token FROM scene WHERE token = ?", (scenario_token_bytes,))
            log_token = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT ep.token, ep.timestamp
                FROM ego_pose ep
                WHERE ep.log_token = ?
                ORDER BY ep.timestamp
            """,
                (log_token,),
            )
            frames = cursor.fetchall()

            if frame_index < len(frames):
                target_frame = frames[frame_index]
                return target_frame[0], target_frame[1], target_frame[0]
    except Exception:
        pass

    cursor.execute("SELECT token, timestamp FROM ego_pose ORDER BY timestamp")
    all_poses = cursor.fetchall()

    if frame_index < len(all_poses):
        pose = all_poses[frame_index]
        return pose["token"], pose["timestamp"], pose["token"]

    pose = all_poses[0]
    return pose["token"], pose["timestamp"], pose["token"]


def get_traffic_lights_at_timestamp(conn, timestamp, map_name):
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT tls.status, tls.lane_connector_id, lp.timestamp
            FROM traffic_light_status tls
            JOIN lidar_pc lp ON tls.lidar_pc_token = lp.token
            WHERE lp.timestamp <= ?
            ORDER BY lp.timestamp DESC
            LIMIT 50
        """,
            (timestamp,),
        )
        results = cursor.fetchall()

        traffic_lights = {}
        for row in results:
            lane_id = row[1]
            state = row[0]
            traffic_lights[lane_id] = state

        return traffic_lights
    except Exception:
        return {}


def extract_ego_data(conn, center_token, center_timestamp, scenario_token):
    cursor = conn.cursor()

    scenario_token_bytes = bytes.fromhex(scenario_token)
    cursor.execute("SELECT log_token FROM scene WHERE token = ?", (scenario_token_bytes,))
    result = cursor.fetchone()
    if result is None:
        log_token = None
    else:
        log_token = result[0]

    if log_token:
        cursor.execute(
            """
            SELECT token, timestamp, x, y, z, qw, qx, qy, qz,
                   vx, vy, vz, acceleration_x, acceleration_y
            FROM ego_pose
            WHERE log_token = ?
            ORDER BY timestamp
        """,
            (log_token,),
        )
    else:
        cursor.execute(
            """
            SELECT token, timestamp, x, y, z, qw, qx, qy, qz,
                   vx, vy, vz, acceleration_x, acceleration_y
            FROM ego_pose
            ORDER BY timestamp
        """
        )
    all_poses = cursor.fetchall()

    center_idx = None
    for i, row in enumerate(all_poses):
        if row["token"] == center_token:
            center_idx = i
            break

    if center_idx is None:
        for i, row in enumerate(all_poses):
            if abs(row["timestamp"] - center_timestamp) < 1000000:
                center_idx = i
                break

    if center_idx is None:
        center_idx = len(all_poses) // 2

    ego_row = all_poses[center_idx]
    ego_x = ego_row["x"]
    ego_y = ego_row["y"]
    ego_heading = quaternion_to_heading(ego_row["qw"], ego_row["qx"], ego_row["qy"], ego_row["qz"])
    ego_vx = ego_row["vx"]
    ego_vy = ego_row["vy"]
    ego_ax = ego_row["acceleration_x"]
    ego_ay = ego_row["acceleration_y"]

    R = get_rotation_matrix_2d(-ego_heading)
    v_local = R @ np.array([ego_vx, ego_vy])

    ego_current_state = np.array(
        [
            0.0,
            0.0,
            np.cos(ego_heading),
            np.sin(ego_heading),
            v_local[0],
            v_local[1],
            ego_ax,
            ego_ay,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    ego_future = np.zeros((EGO_FUTURE_LEN, 3), dtype=np.float32)
    for i in range(EGO_FUTURE_LEN):
        idx = center_idx + (i + 1) * 10
        if idx < len(all_poses):
            future_row = all_poses[idx]
            dx, dy = transform_to_ego_frame(future_row["x"], future_row["y"], ego_x, ego_y, ego_heading)
            future_heading = quaternion_to_heading(
                future_row["qw"],
                future_row["qx"],
                future_row["qy"],
                future_row["qz"],
            )
            dheading = future_heading - ego_heading
            while dheading > np.pi:
                dheading -= 2 * np.pi
            while dheading < -np.pi:
                dheading += 2 * np.pi
            ego_future[i] = [dx, dy, dheading]

    ego_past = np.zeros((EGO_HISTORY_LEN, 3), dtype=np.float32)
    for i in range(EGO_HISTORY_LEN):
        idx = center_idx - (EGO_HISTORY_LEN - 1 - i) * 10
        if idx >= 0:
            past_row = all_poses[idx]
            dx, dy = transform_to_ego_frame(past_row["x"], past_row["y"], ego_x, ego_y, ego_heading)
            heading = quaternion_to_heading(past_row["qw"], past_row["qx"], past_row["qy"], past_row["qz"])
            dheading = heading - ego_heading
            while dheading > np.pi:
                dheading -= 2 * np.pi
            while dheading < -np.pi:
                dheading += 2 * np.pi
            ego_past[i] = [dx, dy, dheading]

    neighbor_past = np.zeros((MAX_NEIGHBORS, NEIGHBOR_HISTORY_LEN, 11), dtype=np.float32)

    for i in range(NEIGHBOR_HISTORY_LEN):
        # ego_pose is 100Hz; neighbor_agents_past is defined at 10Hz (2s @ 10Hz),
        # so we downsample by 10 frames per step (see mds/FEATURE_EXTRACTION_GUIDE.md).
        idx = center_idx - (NEIGHBOR_HISTORY_LEN - 1 - i) * 10
        if idx >= 0:
            past_row = all_poses[idx]
            dx, dy = transform_to_ego_frame(past_row["x"], past_row["y"], ego_x, ego_y, ego_heading)
            past_heading = quaternion_to_heading(
                past_row["qw"],
                past_row["qx"],
                past_row["qy"],
                past_row["qz"],
            )
            v_local = R @ np.array([past_row["vx"], past_row["vy"]])
            neighbor_past[0, i] = [
                dx,
                dy,
                np.cos(past_heading),
                np.sin(past_heading),
                v_local[0],
                v_local[1],
                past_row["acceleration_x"],
                past_row["acceleration_y"],
                1.8,
                4.5,
                1.0,
            ]
        else:
            neighbor_past[0, i, -1] = 0.0

    return ego_current_state, ego_past, ego_future, neighbor_past, ego_x, ego_y, ego_heading, center_idx, all_poses


def extract_neighbor_agents(conn, center_timestamp, ego_x, ego_y, ego_heading):
    cursor = conn.cursor()

    cursor.execute(
        "SELECT token, timestamp FROM lidar_pc WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
        (center_timestamp,),
    )
    center_lidar = cursor.fetchone()

    if center_lidar is None:
        return (
            np.zeros((MAX_NEIGHBORS, NEIGHBOR_HISTORY_LEN, 11), dtype=np.float32),
            np.zeros((MAX_NEIGHBORS, NEIGHBOR_FUTURE_LEN, 3), dtype=np.float32),
        )

    center_lidar_token = center_lidar["token"]

    cursor.execute("SELECT DISTINCT track_token FROM lidar_box WHERE lidar_pc_token = ?", (center_lidar_token,))
    tracks = cursor.fetchall()

    neighbor_past = np.zeros((MAX_NEIGHBORS, NEIGHBOR_HISTORY_LEN, 11), dtype=np.float32)
    neighbor_future = np.zeros((MAX_NEIGHBORS, NEIGHBOR_FUTURE_LEN, 3), dtype=np.float32)

    R = get_rotation_matrix_2d(-ego_heading)

    for agent_idx, track in enumerate(tracks[: MAX_NEIGHBORS - 1]):
        track_token = track["track_token"]

        cursor.execute(
            """
            SELECT lp.timestamp, lb.x, lb.y, lb.z, lb.yaw, lb.vx, lb.vy, t.width, t.length
            FROM lidar_box lb
            JOIN track t ON lb.track_token = t.token
            JOIN lidar_pc lp ON lb.lidar_pc_token = lp.token
            WHERE lb.track_token = ?
            ORDER BY lp.timestamp
        """,
            (track_token,),
        )

        boxes = cursor.fetchall()
        if len(boxes) == 0:
            continue

        center_box_idx = 0
        min_diff = float("inf")
        for i, box in enumerate(boxes):
            diff = abs(box["timestamp"] - center_timestamp)
            if diff < min_diff:
                min_diff = diff
                center_box_idx = i

        if min_diff > 100000000:
            continue

        for i in range(NEIGHBOR_HISTORY_LEN):
            # lidar_box is 20Hz; neighbor past contract is 10Hz (21 points over 2s)
            # => sample every 2 frames from lidar_box.
            idx = center_box_idx - 2 * (NEIGHBOR_HISTORY_LEN - 1 - i)
            if idx >= 0:
                box = boxes[idx]
                dx, dy = transform_to_ego_frame(box["x"], box["y"], ego_x, ego_y, ego_heading)
                heading = box["yaw"]
                v_local = R @ np.array([box["vx"], box["vy"]])
                neighbor_past[agent_idx + 1, i] = [
                    dx,
                    dy,
                    np.cos(heading),
                    np.sin(heading),
                    v_local[0],
                    v_local[1],
                    0.0,
                    0.0,
                    box["width"],
                    box["length"],
                    1.0,
                ]

        last_valid_idx = None
        last_valid_dx = None
        last_valid_dy = None
        last_valid_heading = None

        for i in range(NEIGHBOR_FUTURE_LEN):
            idx = center_box_idx + (i + 1) * 2
            if idx < len(boxes):
                box = boxes[idx]
                dx, dy = transform_to_ego_frame(box["x"], box["y"], ego_x, ego_y, ego_heading)
                heading = box["yaw"]
                dheading = heading - ego_heading
                while dheading > np.pi:
                    dheading -= 2 * np.pi
                while dheading < -np.pi:
                    dheading += 2 * np.pi
                neighbor_future[agent_idx + 1, i] = [dx, dy, dheading]
                last_valid_idx = i
                last_valid_dx = dx
                last_valid_dy = dy
                last_valid_heading = dheading

        did_padding = False
        padding_mode = "none"

        def _normalize_angle(a: float) -> float:
            while a > np.pi:
                a -= 2 * np.pi
            while a < -np.pi:
                a += 2 * np.pi
            return a

        if last_valid_idx is None:
            did_padding = True
            padding_mode = "fill_current"
            center_box = boxes[center_box_idx]
            fill_dx, fill_dy = transform_to_ego_frame(center_box["x"], center_box["y"], ego_x, ego_y, ego_heading)
            fill_heading = _normalize_angle(center_box["yaw"] - ego_heading)
            for i in range(NEIGHBOR_FUTURE_LEN):
                neighbor_future[agent_idx + 1, i] = [fill_dx, fill_dy, fill_heading]

        elif last_valid_idx == 0 and last_valid_idx < NEIGHBOR_FUTURE_LEN - 1:
            did_padding = True
            padding_mode = "fill_constant_last"
            for i in range(1, NEIGHBOR_FUTURE_LEN):
                neighbor_future[agent_idx + 1, i] = [last_valid_dx, last_valid_dy, last_valid_heading]

        elif last_valid_idx < NEIGHBOR_FUTURE_LEN - 1:
            did_padding = True
            padding_mode = "fill_const_vel"
            prev_dx = neighbor_future[agent_idx + 1, last_valid_idx - 1, 0]
            prev_dy = neighbor_future[agent_idx + 1, last_valid_idx - 1, 1]
            velocity_x = last_valid_dx - prev_dx
            velocity_y = last_valid_dy - prev_dy
            for i in range(last_valid_idx + 1, NEIGHBOR_FUTURE_LEN):
                neighbor_future[agent_idx + 1, i, 0] = last_valid_dx + velocity_x * (i - last_valid_idx)
                neighbor_future[agent_idx + 1, i, 1] = last_valid_dy + velocity_y * (i - last_valid_idx)
                neighbor_future[agent_idx + 1, i, 2] = last_valid_heading

        if EXTRACT_LOG_STYLE == "legacy":
            try:
                valid_count = (last_valid_idx + 1) if last_valid_idx is not None else 0
                padded_count = (NEIGHBOR_FUTURE_LEN - valid_count) if did_padding else 0
                fut_str = np.array2string(
                    neighbor_future[agent_idx + 1],
                    precision=3,
                    suppress_small=True,
                    separator=", ",
                )
                DEBUG_LOGGER.info(
                    f"neighbor_future agent={agent_idx+1} track_token={track_token.hex() if isinstance(track_token, (bytes, bytearray)) else track_token} "
                    f"boxes_total={len(boxes)} center_box_idx={center_box_idx} min_diff={min_diff} "
                    f"valid_count={valid_count}/{NEIGHBOR_FUTURE_LEN} padded={did_padding} padded_count={padded_count} mode={padding_mode} "
                    f"after_padding={fut_str}"
                )
            except Exception as e:
                DEBUG_LOGGER.warning(f"neighbor_future debug log failed for agent={agent_idx+1}: {e}")

    return neighbor_past, neighbor_future


def extract_static_objects(conn, center_timestamp, ego_x, ego_y, ego_heading):
    cursor = conn.cursor()

    cursor.execute("SELECT token FROM lidar_pc WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1", (center_timestamp,))
    result = cursor.fetchone()

    if result is None:
        return np.zeros((MAX_STATIC_OBJECTS, 10), dtype=np.float32)

    cursor.execute(
        "SELECT x, y, z, width, length, height, yaw FROM lidar_box WHERE lidar_pc_token = ?",
        (result["token"],),
    )
    boxes = cursor.fetchall()

    static_objects = np.zeros((MAX_STATIC_OBJECTS, 10), dtype=np.float32)

    for i, box in enumerate(boxes[:MAX_STATIC_OBJECTS]):
        dx, dy = transform_to_ego_frame(box["x"], box["y"], ego_x, ego_y, ego_heading)
        static_objects[i] = [
            dx,
            dy,
            box["z"],
            box["width"],
            box["length"],
            box["height"],
            box["yaw"],
            0.0,
            0.0,
            1.0,
        ]

    return static_objects


def _interpolate_points(line, num_points):
    line = LineString(line)
    if line.length == 0:
        return np.zeros((num_points, 2), dtype=np.float64)
    new_line = np.concatenate([line.interpolate(d).coords._coords for d in np.linspace(0, line.length, num_points)])
    return new_line


def _lane_polyline_process_with_avails(
    lane_obj,
    centerline_coords,
    left_coords,
    right_coords,
    traffic_light_state,
    ego_point,
    ego_heading,
    *,
    sample_local_around_ego: bool = False,
):
    lane_feature = np.zeros((POLYLINE_LEN, LANE_DIM), dtype=np.float32)
    avails = np.zeros(POLYLINE_LEN, dtype=np.bool_)

    local_window = False
    if sample_local_around_ego:
        local_window = os.environ.get("ROUTE_LANE_SAMPLE_LOCAL_AROUND_EGO", "0") == "1"

    if len(centerline_coords) >= 2:
        coords_to_sample = centerline_coords
        if local_window and ego_point is not None:
            try:
                d2 = [((x - ego_point.x) ** 2 + (y - ego_point.y) ** 2) for x, y in centerline_coords]
                j = int(np.argmin(d2))
                half = int(os.environ.get("LANE_SAMPLE_LOCAL_HALF_NODES", "20"))
                lo = max(0, j - half)
                hi = min(len(centerline_coords), j + half + 1)
                if hi - lo >= 2:
                    coords_to_sample = centerline_coords[lo:hi]
            except Exception:
                coords_to_sample = centerline_coords

        sampled = _interpolate_points(coords_to_sample, POLYLINE_LEN)
    else:
        sampled = np.zeros((POLYLINE_LEN, 2), dtype=np.float64)

    left_sampled = np.zeros((POLYLINE_LEN, 2), dtype=np.float64)
    right_sampled = np.zeros((POLYLINE_LEN, 2), dtype=np.float64)

    if len(left_coords) >= 2:
        left_sampled = _interpolate_points(left_coords, POLYLINE_LEN)
    if len(right_coords) >= 2:
        right_sampled = _interpolate_points(right_coords, POLYLINE_LEN)

    if np.all(sampled == 0):
        return lane_feature, avails

    avails[:] = True

    for i in range(POLYLINE_LEN):
        cx, cy = sampled[i]
        dx, dy = transform_to_ego_frame(cx, cy, ego_point.x, ego_point.y, ego_heading)

        lane_feature[i, 0] = dx
        lane_feature[i, 1] = dy

        if i < POLYLINE_LEN - 1:
            next_cx, next_cy = sampled[i + 1]
            vec_dx = next_cx - cx
            vec_dy = next_cy - cy
            vec_len = np.sqrt(vec_dx**2 + vec_dy**2)
            if vec_len > 0:
                vec_dx /= vec_len
                vec_dy /= vec_len
            lane_feature[i, 2] = vec_dx
            lane_feature[i, 3] = vec_dy

        lx, ly = left_sampled[i]
        ldx, ldy = transform_to_ego_frame(lx, ly, ego_point.x, ego_point.y, ego_heading)
        lane_feature[i, 4] = ldx - dx
        lane_feature[i, 5] = ldy - dy

        rx, ry = right_sampled[i]
        rdx, rdy = transform_to_ego_frame(rx, ry, ego_point.x, ego_point.y, ego_heading)
        lane_feature[i, 6] = rdx - dx
        lane_feature[i, 7] = rdy - dy

        lane_feature[i, 8] = traffic_light_state[0]
        lane_feature[i, 9] = traffic_light_state[1]
        lane_feature[i, 10] = traffic_light_state[2]
        lane_feature[i, 11] = traffic_light_state[3]

    return lane_feature, avails


def extract_lanes(point, map_api, radius=100, max_lanes=70, ego_heading=0, traffic_light_data=None):
    layers = map_api.get_proximal_map_objects(
        point,
        radius,
        [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
    )

    lanes = np.zeros((max_lanes, POLYLINE_LEN, LANE_DIM), dtype=np.float32)
    lanes_avails = np.zeros((max_lanes, POLYLINE_LEN), dtype=np.bool_)
    speed_limits = np.zeros(max_lanes, dtype=np.float32)
    has_speed_limits = np.zeros(max_lanes, dtype=np.float32)

    traffic_light_lookup = {}
    if traffic_light_data:
        for lane_id, tl_state in traffic_light_data.items():
            lane_id_str = str(lane_id)
            if tl_state == "green":
                traffic_light_lookup[lane_id_str] = [1, 0, 0, 0]
            elif tl_state == "yellow":
                traffic_light_lookup[lane_id_str] = [0, 1, 0, 0]
            elif tl_state == "red":
                traffic_light_lookup[lane_id_str] = [0, 0, 1, 0]
            else:
                traffic_light_lookup[lane_id_str] = [0, 0, 0, 1]

    lanes_with_dist = []

    for layer_type in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
        if layer_type not in layers:
            continue

        for lane_obj in layers[layer_type]:
            try:
                baseline_path = lane_obj.baseline_path
                centerline_coords = [(node.x, node.y) for node in baseline_path.discrete_path]

                if len(centerline_coords) < 2:
                    continue

                left_boundary_coords = []
                if hasattr(lane_obj, "left_boundary") and lane_obj.left_boundary:
                    left_boundary_coords = [(node.x, node.y) for node in lane_obj.left_boundary.discrete_path]

                right_boundary_coords = []
                if hasattr(lane_obj, "right_boundary") and lane_obj.right_boundary:
                    right_boundary_coords = [(node.x, node.y) for node in lane_obj.right_boundary.discrete_path]

                dist = np.mean([np.sqrt((x - point.x) ** 2 + (y - point.y) ** 2) for x, y in centerline_coords])

                lanes_with_dist.append(
                    {
                        "obj": lane_obj,
                        "centerline": centerline_coords,
                        "left": left_boundary_coords,
                        "right": right_boundary_coords,
                        "dist": dist,
                    }
                )
            except Exception:
                continue

    lanes_with_dist.sort(key=lambda x: x["dist"])

    lane_idx = 0
    for lane_data in lanes_with_dist:
        if lane_idx >= max_lanes:
            break

        lane_obj = lane_data["obj"]
        centerline_coords = lane_data["centerline"]
        left_boundary_coords = lane_data["left"]
        right_boundary_coords = lane_data["right"]

        lane_id = lane_obj.id
        traffic_light_state = traffic_light_lookup.get(lane_id, [0, 0, 0, 1])

        lane_feature, avails = _lane_polyline_process_with_avails(
            lane_obj,
            centerline_coords,
            left_boundary_coords,
            right_boundary_coords,
            traffic_light_state,
            point,
            ego_heading,
        )

        lanes[lane_idx] = lane_feature
        lanes_avails[lane_idx] = avails

        try:
            sl = lane_obj.speed_limit_mps
            if sl is not None:
                speed_limits[lane_idx] = float(sl)
                has_speed_limits[lane_idx] = 1.0
        except Exception:
            pass

        lane_idx += 1

    if EXTRACT_LOG_STYLE == "legacy":
        try:
            DEBUG_LOGGER.info(
                f"lanes nonzero={int(np.count_nonzero(lanes))} avails_true={int(np.count_nonzero(lanes_avails))} filled_lanes={lane_idx}/{max_lanes}"
            )
        except Exception as e:
            DEBUG_LOGGER.warning(f"lanes debug log failed: {e}")

    return lanes, lanes_avails, speed_limits, has_speed_limits


def extract_route_lanes(
    point,
    map_api,
    radius=150,
    max_route_lanes=25,
    ego_heading=0,
    traffic_light_data=None,
    route_roadblock_ids: list[str] | None = None,
):
    layers = map_api.get_proximal_map_objects(
        point,
        radius,
        [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
    )

    route_lanes = np.zeros((max_route_lanes, POLYLINE_LEN, LANE_DIM), dtype=np.float32)
    route_lanes_avails = np.zeros((max_route_lanes, POLYLINE_LEN), dtype=np.bool_)
    route_speed_limits = np.zeros(max_route_lanes, dtype=np.float32)
    route_has_speed_limits = np.zeros(max_route_lanes, dtype=np.float32)

    route_roadblock_id_set = set(route_roadblock_ids) if route_roadblock_ids else None

    traffic_light_lookup = {}
    if traffic_light_data:
        for lane_id, tl_state in traffic_light_data.items():
            lane_id_str = str(lane_id)
            if tl_state == "green":
                traffic_light_lookup[lane_id_str] = [1, 0, 0, 0]
            elif tl_state == "yellow":
                traffic_light_lookup[lane_id_str] = [0, 1, 0, 0]
            elif tl_state == "red":
                traffic_light_lookup[lane_id_str] = [0, 0, 1, 0]
            else:
                traffic_light_lookup[lane_id_str] = [0, 0, 0, 1]

    all_lanes = []

    for layer_type in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
        if layer_type not in layers:
            continue

        for lane_obj in layers[layer_type]:
            try:
                if route_roadblock_id_set is not None:
                    rb_id = None
                    try:
                        if hasattr(lane_obj, "get_roadblock_id"):
                            rb_id = lane_obj.get_roadblock_id()
                        elif hasattr(lane_obj, "roadblock_id"):
                            rb_id = lane_obj.roadblock_id
                    except Exception:
                        rb_id = None

                    if rb_id is None or str(rb_id) not in route_roadblock_id_set:
                        continue

                polygon = lane_obj.polygon
                if polygon is None:
                    continue

                centroid = polygon.centroid
                dist = np.sqrt((centroid.x - point.x) ** 2 + (centroid.y - point.y) ** 2)

                baseline_path = lane_obj.baseline_path
                centerline_coords = [(node.x, node.y) for node in baseline_path.discrete_path]

                left_boundary_coords = []
                if hasattr(lane_obj, "left_boundary") and lane_obj.left_boundary:
                    left_boundary_coords = [(node.x, node.y) for node in lane_obj.left_boundary.discrete_path]

                right_boundary_coords = []
                if hasattr(lane_obj, "right_boundary") and lane_obj.right_boundary:
                    right_boundary_coords = [(node.x, node.y) for node in lane_obj.right_boundary.discrete_path]

                all_lanes.append(
                    {
                        "obj": lane_obj,
                        "centerline": centerline_coords,
                        "left": left_boundary_coords,
                        "right": right_boundary_coords,
                        "dist": dist,
                    }
                )
            except Exception:
                continue

    all_lanes.sort(key=lambda x: x["dist"])

    route_idx = 0
    for lane_data in all_lanes[:max_route_lanes]:
        if route_idx >= max_route_lanes:
            break

        lane_obj = lane_data["obj"]
        centerline_coords = lane_data["centerline"]
        left_boundary_coords = lane_data["left"]
        right_boundary_coords = lane_data["right"]

        lane_id = lane_obj.id
        traffic_light_state = traffic_light_lookup.get(lane_id, [0, 0, 0, 1])

        lane_feature, avails = _lane_polyline_process_with_avails(
            lane_obj,
            centerline_coords,
            left_boundary_coords,
            right_boundary_coords,
            traffic_light_state,
            point,
            ego_heading,
            sample_local_around_ego=True,
        )

        route_lanes[route_idx] = lane_feature
        route_lanes_avails[route_idx] = avails

        try:
            sl = lane_obj.speed_limit_mps
            if sl is not None:
                route_speed_limits[route_idx] = float(sl)
                route_has_speed_limits[route_idx] = 1.0
        except Exception:
            pass

        route_idx += 1

    if EXTRACT_LOG_STYLE == "legacy":
        try:
            DEBUG_LOGGER.info(
                f"route_lanes nonzero={int(np.count_nonzero(route_lanes))} avails_true={int(np.count_nonzero(route_lanes_avails))} filled_route_lanes={route_idx}/{max_route_lanes}"
            )
        except Exception as e:
            DEBUG_LOGGER.warning(f"route_lanes debug log failed: {e}")

    return route_lanes, route_lanes_avails, route_speed_limits, route_has_speed_limits


def _get_db_path_from_conn(conn: sqlite3.Connection) -> str:
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA database_list")
        rows = cur.fetchall()
        for r in rows:
            if len(r) >= 3 and r[1] == "main":
                return str(r[2])
        if rows and len(rows[0]) >= 3:
            return str(rows[0][2])
    except Exception:
        pass
    return ""


def extract_features(
    conn,
    map_api,
    scenario_token_hex: str,
    frame_index: int,
    *,
    debug_log: bool = True,
    routing_mode: str = "auto",
) -> dict[str, np.ndarray]:
    """Pure feature extraction.

    Returns a dict of arrays matching the saved NPZ keys.
    """

    db_path = _get_db_path_from_conn(conn)
    map_name = getattr(map_api, "map_name", None) or getattr(map_api, "_map_name", None) or MAP_NAME

    timing: dict[str, float] | None = {} if _extract_profile_enabled() else None

    with _timing_ctx(timing, "get_target_frame"):
        center_token, center_timestamp, _ = get_target_frame(conn, scenario_token_hex, int(frame_index))

    with _timing_ctx(timing, "extract_ego_data"):
        ego_current_state, ego_past, ego_future, neighbor_past, ego_x, ego_y, ego_heading, _, _ = extract_ego_data(
            conn, center_token, center_timestamp, scenario_token_hex
        )

    with _timing_ctx(timing, "get_traffic_lights_at_timestamp"):
        traffic_light_data = get_traffic_lights_at_timestamp(conn, center_timestamp, map_name)

    with _timing_ctx(timing, "extract_neighbor_agents"):
        neighbor_past_agents, neighbor_future = extract_neighbor_agents(conn, center_timestamp, ego_x, ego_y, ego_heading)

    with _timing_ctx(timing, "fill_neighbor_past"):
        for i in range(1, MAX_NEIGHBORS):
            if np.any(neighbor_past_agents[i, :, -1] != 0):
                neighbor_past[i] = neighbor_past_agents[i]

    with _timing_ctx(timing, "extract_static_objects"):
        static_objects = extract_static_objects(conn, center_timestamp, ego_x, ego_y, ego_heading)

    point = Point2D(ego_x, ego_y)

    routing_mode = (routing_mode or "auto").strip().lower()

    with _timing_ctx(timing, "get_raw_route_roadblock_ids"):
        try:
            scenario = build_nuplan_scenario_from_db(conn, db_path, scenario_token_hex, map_name)
            route_roadblock_ids_raw = list(scenario.get_route_roadblock_ids())
        except Exception:
            route_roadblock_ids_raw = []

    if routing_mode == "auto":
        with _timing_ctx(timing, "get_pruned_route_roadblock_ids"):
            try:
                route_roadblock_ids = get_pruned_route_roadblock_ids(conn, db_path, scenario_token_hex, map_api, map_name)
            except Exception:
                route_roadblock_ids = None
    else:
        route_roadblock_ids = list(route_roadblock_ids_raw)

    with _timing_ctx(timing, "extract_lanes"):
        lanes, lanes_avails, lanes_speed_limit, lanes_has_speed_limit = extract_lanes(
            point,
            map_api,
            radius=100,
            max_lanes=MAX_LANES,
            ego_heading=ego_heading,
            traffic_light_data=traffic_light_data,
        )

    with _timing_ctx(timing, "extract_route_lanes_old"):
        route_lanes_old, route_lanes_avails_old, _, _ = extract_route_lanes(
            point,
            map_api,
            radius=150,
            max_route_lanes=MAX_ROUTE_LANES,
            ego_heading=ego_heading,
            traffic_light_data=traffic_light_data,
            route_roadblock_ids=route_roadblock_ids,
        )
    avails_sum_old = int(np.count_nonzero(route_lanes_avails_old))

    proximal_rb_ids: set[str] = set()
    with _timing_ctx(timing, "map_api.get_proximal_map_objects"):
        try:
            prox_layers = map_api.get_proximal_map_objects(
                point,
                150,
                [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
            )
            for lt in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
                for lane_obj in (prox_layers.get(lt, []) or []):
                    try:
                        rb = lane_obj.get_roadblock_id() if hasattr(lane_obj, "get_roadblock_id") else None
                        if rb is not None:
                            proximal_rb_ids.add(str(rb))
                    except Exception:
                        continue
        except Exception:
            prox_layers = {}

    pruned_route_set = set([str(x) for x in (route_roadblock_ids or [])])
    intersection_pruned = int(len(proximal_rb_ids.intersection(pruned_route_set)))

    def _min_dist_m(lanes_xy12: np.ndarray, avails_25x20: np.ndarray) -> float | None:
        try:
            m = avails_25x20 > 0
            if not np.any(m):
                return None
            xs = lanes_xy12[:, :, 0][m]
            ys = lanes_xy12[:, :, 1][m]
            if xs.size == 0:
                return None
            return float(np.sqrt(xs * xs + ys * ys).min())
        except Exception:
            return None

    rmin_old_m = _min_dist_m(route_lanes_old, route_lanes_avails_old)

    new_route_ids = route_roadblock_ids or []
    bridge_found = False
    bridge_len = 0

    ego_rb = None
    try:
        nearest_obj = None
        nearest_dist = float("inf")
        for lt in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
            for lane_obj in (prox_layers.get(lt, []) or []):
                try:
                    poly = lane_obj.polygon
                    if poly is None:
                        continue
                    c = poly.centroid
                    d = float(((c.x - point.x) ** 2 + (c.y - point.y) ** 2) ** 0.5)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_obj = lane_obj
                except Exception:
                    continue
        if nearest_obj is not None and hasattr(nearest_obj, "get_roadblock_id"):
            ego_rb = str(nearest_obj.get_roadblock_id())
    except Exception:
        ego_rb = None

    bridge_reason = "skip"

    bfs_called = False
    realign_from_overlap = False
    overlap_idx: int | None = None
    bfs_triggered_by = "none"

    route_list = [str(x) for x in (route_roadblock_ids or [])]
    route_set = set(route_list)
    ego_rb_in_route = bool(ego_rb) and (ego_rb in route_set)

    off_route = not ego_rb_in_route
    if not ego_rb:
        off_route = True

    bad_route_geom = (avails_sum_old == 0) or (rmin_old_m is None) or (float(rmin_old_m) > 30.0)

    if routing_mode == "nofix":
        need_bridge = False
        bfs_triggered_by = "disabled"
        bridge_reason = "nofix"
        new_route_ids = list(route_list)

    elif routing_mode == "realign":
        need_bridge = False
        bfs_triggered_by = "disabled"
        try:
            inter = proximal_rb_ids.intersection(route_set)
            if inter:
                pos = {v: i for i, v in enumerate(route_list)}
                idx_min = min(pos[x] for x in inter if x in pos)
                if idx_min is not None and idx_min > 0:
                    overlap_idx = int(idx_min)
                    realign_from_overlap = True
                    new_route_ids = route_list[idx_min:]
                    bridge_found = True
                    bridge_len = 0
                    bridge_reason = f"realign_from_overlap_idx={idx_min}"
                else:
                    bridge_reason = "overlap_idx=0 (no_realign)"
                    new_route_ids = list(route_list)
            else:
                bridge_reason = "no_overlap_for_realign"
                new_route_ids = list(route_list)
        except Exception:
            bridge_reason = "overlap_realign_exception"
            new_route_ids = list(route_list)

    elif routing_mode == "bfs":
        need_bridge = bool(route_list) and off_route
        if need_bridge and (intersection_pruned == 0):
            bfs_called = True
            bfs_triggered_by = "off_route"
            with _timing_ctx(timing, "bfs_bridge_route_if_needed"):
                new_route_ids, bridge_len, bridge_found, ego_rb_bfs, bridge_reason = bfs_bridge_route_if_needed(
                    map_api,
                    point,
                    list(route_list),
                    intersection_pruned=intersection_pruned,
                    radius=150,
                    k_targets=10,
                    max_depth=80,
                )
            if not ego_rb:
                ego_rb = ego_rb_bfs
            ego_rb_in_route = bool(ego_rb) and (ego_rb in set([str(x) for x in (route_roadblock_ids or [])]))
        elif need_bridge and (intersection_pruned != 0):
            bfs_triggered_by = f"skip_intersection_pruned={intersection_pruned}"
            bridge_reason = "bfs_skipped_intersection_pruned"
            new_route_ids = list(route_list)
        else:
            bfs_triggered_by = "gate_not_met"
            bridge_reason = "bfs_gate_not_met"
            new_route_ids = list(route_list)

    else:
        if bad_route_geom and route_list:
            try:
                inter = proximal_rb_ids.intersection(route_set)
                if inter:
                    pos = {v: i for i, v in enumerate(route_list)}
                    idx_min = min(pos[x] for x in inter if x in pos)
                    if idx_min is not None and idx_min > 0:
                        overlap_idx = int(idx_min)
                        realign_from_overlap = True
                        new_route_ids = route_list[idx_min:]
                        bridge_found = True
                        bridge_len = 0
                        bridge_reason = f"realign_from_overlap_idx={idx_min} (bad_route_geom)"
                    else:
                        bridge_reason = "overlap_idx=0 (no_realign)"
                else:
                    bridge_reason = "no_overlap_for_realign"
            except Exception:
                bridge_reason = "overlap_realign_exception"

        need_bridge = bool(route_list) and off_route and bad_route_geom and (not realign_from_overlap)
        if need_bridge and (intersection_pruned == 0):
            bfs_called = True
            bfs_triggered_by = "off_route_and_bad_route_geom"
            with _timing_ctx(timing, "bfs_bridge_route_if_needed"):
                new_route_ids, bridge_len, bridge_found, ego_rb_bfs, bridge_reason = bfs_bridge_route_if_needed(
                    map_api,
                    point,
                    list(route_list),
                    intersection_pruned=intersection_pruned,
                    radius=150,
                    k_targets=10,
                    max_depth=80,
                )
            if not ego_rb:
                ego_rb = ego_rb_bfs
            ego_rb_in_route = bool(ego_rb) and (ego_rb in set([str(x) for x in (route_roadblock_ids or [])]))
        elif need_bridge and (intersection_pruned != 0):
            bfs_triggered_by = f"skip_intersection_pruned={intersection_pruned}"
        elif need_bridge and realign_from_overlap:
            bfs_triggered_by = "realign_succeeded"
        else:
            if not need_bridge:
                bfs_triggered_by = "gate_not_met"

    with _timing_ctx(timing, "extract_route_lanes_new"):
        route_lanes, route_lanes_avails, route_lanes_speed_limit, route_lanes_has_speed_limit = extract_route_lanes(
            point,
            map_api,
            radius=150,
            max_route_lanes=MAX_ROUTE_LANES,
            ego_heading=ego_heading,
            traffic_light_data=traffic_light_data,
            route_roadblock_ids=new_route_ids,
        )
    avails_sum_new = int(np.count_nonzero(route_lanes_avails))

    if debug_log:
        try:
            out_dir = "/workspace/validation_output"
            try:
                os.makedirs(out_dir, exist_ok=True)
                test_path = os.path.join(out_dir, ".write_test")
                with open(test_path, "w") as f:
                    f.write("ok")
                os.remove(test_path)
            except Exception:
                out_dir = os.path.abspath(os.path.join(_REPO_ROOT, "validation_output"))
                os.makedirs(out_dir, exist_ok=True)

            log_path = os.path.join(out_dir, "bfs_single_case.log")
            with open(log_path, "a") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"db={os.path.basename(db_path)} scene_token={scenario_token_hex} frame={int(frame_index)} map={map_name}\n")
                f.write(
                    f"intersection_pruned={intersection_pruned} ego_rb={ego_rb} bridge_found={bridge_found} bridge_len={bridge_len} reason={bridge_reason}\n"
                )
                f.write(
                    f"route_len_old={(len(route_roadblock_ids) if route_roadblock_ids else 0)} route_len_new={(len(new_route_ids) if new_route_ids else 0)}\n"
                )
                f.write(f"avails_sum_old={avails_sum_old} avails_sum_new={avails_sum_new}\n")

            import json

            json_path = os.path.join(out_dir, "bfs_single_case_result.json")
            with open(json_path, "w") as f:
                json.dump(
                    {
                        "db_basename": os.path.basename(db_path),
                        "scene_token_hex": scenario_token_hex,
                        "frame_index": int(frame_index),
                        "map_name": map_name,
                        "intersection_pruned": int(intersection_pruned),
                        "ego_rb": ego_rb,
                        "bridge_found": bool(bridge_found),
                        "bridge_len": int(bridge_len),
                        "bridge_reason": bridge_reason,
                        "route_len_old": int(len(route_roadblock_ids) if route_roadblock_ids else 0),
                        "route_len_new": int(len(new_route_ids) if new_route_ids else 0),
                        "avails_sum_old": int(avails_sum_old),
                        "avails_sum_new": int(avails_sum_new),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    out = {
        "ego_current_state": ego_current_state,
        "ego_past": ego_past,
        "ego_agent_future": ego_future,
        "neighbor_agents_past": neighbor_past,
        "neighbor_agents_future": neighbor_future,
        "static_objects": static_objects,
        "lanes": lanes,
        "lanes_avails": lanes_avails,
        "route_lanes": route_lanes,
        "route_lanes_avails": route_lanes_avails,
        "lanes_speed_limit": lanes_speed_limit,
        "lanes_has_speed_limit": lanes_has_speed_limit,
        "route_lanes_speed_limit": route_lanes_speed_limit,
        "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
    }

    if timing is not None:
        out["_timing"] = dict(timing)
        route_prefix10 = [str(x) for x in (route_roadblock_ids or [])[:10]]
        proximal_rb_count = int(len(proximal_rb_ids)) if proximal_rb_ids is not None else 0
        proximal_overlap_count = 0
        try:
            proximal_overlap_count = int(len(proximal_rb_ids.intersection(route_set))) if proximal_rb_ids is not None else 0
        except Exception:
            proximal_overlap_count = 0

        out["_profile_flags"] = {
            "need_bridge": bool(need_bridge),
            "bfs_called": bool(bfs_called),
            "bfs_triggered_by": str(bfs_triggered_by),
            "bridge_found": bool(bridge_found),
            "bridge_len": int(bridge_len),
            "bridge_reason": str(bridge_reason),
            "intersection_pruned": int(intersection_pruned),
            "ego_rb": (None if ego_rb is None else str(ego_rb)),
            "ego_rb_in_route": bool(ego_rb_in_route),
            "off_route": bool(off_route),
            "bad_route_geom": bool(bad_route_geom),
            "rmin_old_m": (None if rmin_old_m is None else float(rmin_old_m)),
            "realign_from_overlap": bool(realign_from_overlap),
            "overlap_idx": (None if overlap_idx is None else int(overlap_idx)),
            "route_len_old": int(len(route_roadblock_ids) if route_roadblock_ids else 0),
            "route_len_new": int(len(new_route_ids) if new_route_ids else 0),
            "avails_sum_old": int(avails_sum_old),
            "avails_sum_new": int(avails_sum_new),
            "route_prefix10": route_prefix10,
            "proximal_rb_count": proximal_rb_count,
            "proximal_overlap_count": proximal_overlap_count,
        }

    return out
