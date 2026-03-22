#!/usr/bin/env python3
"""
Complete feature extraction for nuplan scenario using map_api with valid marking
Based on GitHub source code (map_process.py):
- dim 0-1: polyline (x, y) - center line coordinates
- dim 2-3: polyline_vector (dx, dy) - adjacent point difference  
- dim 4-5: polyline_to_left - left boundary relative position (left_boundary - center)
- dim 6-7: polyline_to_right - right boundary relative position (right_boundary - center)
- dim 8-11: traffic_light - 4-dim one-hot (green/yellow/red/unknown)
- avails_array: boolean array marking valid data points
"""

import sqlite3
import numpy as np
import os
import logging
from collections import Counter
from datetime import datetime
from shapely import LineString

# Add nuplan-visualization to path
import sys
sys.path.insert(0, '/workspace/nuplan-visualization')
# Fallback for environments where /workspace is not mounted/writable
_local_nuplan_vis = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'nuplan-visualization'))
if os.path.isdir(_local_nuplan_vis):
    sys.path.insert(0, _local_nuplan_vis)

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.diffusion_planner.data_process.roadblock_utils import route_roadblock_correction, BreadthFirstSearchRoadBlock

# Constants
DB_PATH = '/workspace/data/nuplan/data/cache/mini/2021.06.14.17.26.26_veh-38_04544_04920.db'
MAP_ROOT = '/workspace/data/nuplan/maps'
MAP_VERSION = '9.12.1817'
MAP_NAME = 'us-nv-las-vegas-strip'

# Feature dimensions
EGO_FUTURE_LEN = 80
EGO_HISTORY_LEN = 21  # 21帧历史，10Hz采样
NEIGHBOR_HISTORY_LEN = 21
NEIGHBOR_FUTURE_LEN = 81  # 8秒 = 81点 @ 0.1s间隔
MAX_NEIGHBORS = 32
MAX_STATIC_OBJECTS = 5
MAX_LANES = 70
MAX_ROUTE_LANES = 25
POLYLINE_LEN = 20
LANE_DIM = 12

# Target scenario
SCENARIO_TOKEN = '037db12ac9125b9a'
CENTER_FRAME_INDEX = 17486  # 第200帧

OUTPUT_PATH = '/workspace/data_process/npz_scenes/test_ego_past.npz'
CSV_OUTPUT_PATH = '/workspace/diffusion-planner-project/data_process/npz_scenes/las_vegas_hs_17486.csv'


# ------------------------
# Debug logging (no logic changes)
# ------------------------
_DEBUG_LOG_DIR_DEFAULT = '/workspace/data_process/debug_log'
_DEBUG_LOG_DIR_FALLBACK = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data_process', 'debug_log')

def _init_debug_logger():
    """Initialize a per-run debug logger writing to debug_log dir."""
    logger = logging.getLogger('extract_single_frame_debug')
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Prefer default path; if not writable, fall back to repo-local data_process/debug_log
    debug_dir = _DEBUG_LOG_DIR_DEFAULT
    try:
        os.makedirs(debug_dir, exist_ok=True)
        test_path = os.path.join(debug_dir, '.write_test')
        with open(test_path, 'w') as f:
            f.write('ok')
        os.remove(test_path)
    except Exception:
        debug_dir = os.path.abspath(_DEBUG_LOG_DIR_FALLBACK)
        os.makedirs(debug_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(OUTPUT_PATH))[0] if 'OUTPUT_PATH' in globals() else 'extract_single_frame'
    log_path = os.path.join(debug_dir, f'{base}.log')

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    logger.info('=== extract_single_frame debug start ===')
    logger.info(f'OUTPUT_PATH={OUTPUT_PATH}')
    logger.info(f'DB_PATH={DB_PATH}')
    logger.info(f'MAP_NAME={MAP_NAME}')
    logger.info(f'SCENARIO_TOKEN={SCENARIO_TOKEN} CENTER_FRAME_INDEX={CENTER_FRAME_INDEX}')
    logger.info(f'log_path={log_path}')
    return logger


# Log style control:
# - legacy: keep very verbose per-sample debug logging (file I/O + many warnings to stdout)
# - quiet (default): suppress per-sample INFO logs; aggregate/limit WARNING prints
EXTRACT_LOG_STYLE = os.environ.get('EXTRACT_LOG_STYLE', 'quiet').strip().lower()

# Aggregated warning stats for batch exporters to read.
LOG_WARNING_COUNTS: Counter[str] = Counter()
LOG_WARNING_TOTAL: int = 0

class _QuietLogger:
    def __init__(self, *, print_first_n_per_key: int = 3):
        self._print_first_n_per_key = int(print_first_n_per_key)

    def info(self, msg: str, *args, **kwargs):
        # keep silent in quiet mode
        return

    def warning(self, msg: str, *args, **kwargs):
        global LOG_WARNING_TOTAL
        LOG_WARNING_TOTAL += 1
        key = str(msg).split('\n', 1)[0][:200]
        LOG_WARNING_COUNTS[key] += 1

        # Only print first N occurrences per key to avoid stderr/stdout spam.
        if LOG_WARNING_COUNTS[key] <= self._print_first_n_per_key:
            try:
                print(f"[WARN] {msg}")
            except Exception:
                pass

    def error(self, msg: str, *args, **kwargs):
        # errors should still surface
        try:
            print(f"[ERROR] {msg}")
        except Exception:
            pass


def reset_log_stats() -> None:
    global LOG_WARNING_TOTAL
    LOG_WARNING_COUNTS.clear()
    LOG_WARNING_TOTAL = 0


def get_log_stats() -> dict:
    return {
        'warning_total': int(LOG_WARNING_TOTAL),
        'warning_by_key': dict(LOG_WARNING_COUNTS),
    }


# Initialize logger object
if EXTRACT_LOG_STYLE == 'legacy':
    DEBUG_LOGGER = _init_debug_logger()
else:
    DEBUG_LOGGER = _QuietLogger(print_first_n_per_key=int(os.environ.get('EXTRACT_WARN_PRINT_N', '3')))


def quaternion_to_heading(qw, qx, qy, qz):
    """Convert quaternion to heading angle (yaw)"""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def get_rotation_matrix_2d(heading):
    """Get 2D rotation matrix"""
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    return np.array([[cos_h, -sin_h], [sin_h, cos_h]])


def transform_to_ego_frame(x, y, ego_x, ego_y, ego_heading):
    """Transform coordinates to ego vehicle frame"""
    dx = x - ego_x
    dy = y - ego_y
    R = get_rotation_matrix_2d(-ego_heading)
    local_x, local_y = R @ np.array([dx, dy])
    return local_x, local_y


def load_db(db_path: str | None = None):
    """Load database connection."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_location_from_log(conn: sqlite3.Connection) -> str | None:
    """Read the actual scenario location from the DB log table (authoritative source)."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT location FROM log LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return None
        # sqlite3.Row supports both index and key access
        loc = row[0]
        if loc is None:
            return None
        return str(loc)
    except Exception:
        return None


def map_name_from_location(location: str | None) -> str:
    """Map DB `log.location` value to a nuPlan map name.

    Notes:
      - Some DBs store a *city* string (e.g. 'boston', 'las_vegas').
      - Some DBs store the *map name itself* (e.g. 'us-pa-pittsburgh-hazelwood').
    """

    if not location:
        return 'us-ma-boston'

    # If DB already stores the map name, accept it directly.
    known_map_names = {
        'us-nv-las-vegas-strip',
        'us-ma-boston',
        'us-pa-pittsburgh-hazelwood',
        'sg-one-north',
    }
    if location in known_map_names:
        return location

    location_to_map = {
        'las_vegas': 'us-nv-las-vegas-strip',
        'boston': 'us-ma-boston',
        'pittsburgh': 'us-pa-pittsburgh-hazelwood',
        'singapore': 'sg-one-north',
    }
    return location_to_map.get(location, 'us-ma-boston')


def build_nuplan_scenario_from_db(conn: sqlite3.Connection, db_path: str, scene_token_hex: str, map_name: str) -> NuPlanScenario:
    """Build a minimal NuPlanScenario so we can call scenario.get_route_roadblock_ids()."""
    cursor = conn.cursor()
    scene_token_bytes = bytes.fromhex(scene_token_hex)

    cursor.execute('SELECT name FROM scene WHERE token = ? LIMIT 1', (scene_token_bytes,))
    row = cursor.fetchone()
    scenario_type = str(row[0]) if row and row[0] is not None else 'unknown'

    # Find the initial lidar_pc token for this *scene* (earliest timestamp within the scene).
    # NOTE: `lidar_pc.prev_token` links the whole log, not the scene segment, so do NOT use prev_token IS NULL here.
    cursor.execute(
        'SELECT token, timestamp FROM lidar_pc WHERE scene_token = ? ORDER BY timestamp ASC LIMIT 1',
        (scene_token_bytes,),
    )
    lidar_row = cursor.fetchone()
    assert lidar_row is not None, f'Unable to find lidar_pc row for scene_token={scene_token_hex}'

    initial_lidar_token_hex = lidar_row[0].hex() if isinstance(lidar_row[0], (bytes, bytearray)) else str(lidar_row[0])
    initial_lidar_timestamp = int(lidar_row[1])

    # Note: data_root is only used for remote download; with a local absolute db_path it's fine.
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
    """Get scenario route roadblock ids and (optionally) correct/prune them."""
    scenario = build_nuplan_scenario_from_db(conn, db_path, scene_token_hex, map_name)
    route_roadblock_ids = list(scenario.get_route_roadblock_ids())

    # Apply correction to handle off-route start / disconnected route segments.
    try:
        ego_state_0 = scenario.get_ego_state_at_iteration(0)
        route_roadblock_ids = list(route_roadblock_correction(ego_state_0, map_api, route_roadblock_ids))
    except Exception as e:
        try:
            DEBUG_LOGGER.warning(f'route_roadblock_correction failed, using raw route ids: {e}')
        except Exception:
            pass

    # De-dup while preserving order
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
    """Minimal BFS bridge experiment for a single case.

    Logic:
      - Find nearest Lane/LaneConnector to ego, use its roadblock_id as ego_rb.
      - If intersection_pruned == 0 and ego_rb not in pruned_route_ids:
            BFS from ego_rb to one of pruned_route_ids[:k_targets]
        then prepend bridge_ids to pruned_route_ids.

    Returns:
      new_route_ids: list[str]
      bridge_len: int
      found: bool
      ego_rb: str | None
      reason: str
    """

    pruned_route_ids = [str(x) for x in (pruned_route_ids or [])]
    if not pruned_route_ids:
        return pruned_route_ids, 0, False, None, 'empty_pruned_route'

    # Find nearest lane / lane_connector
    try:
        layers = map_api.get_proximal_map_objects(
            ego_point,
            radius,
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        )
    except Exception:
        layers = {}

    nearest = None
    nearest_dist = float('inf')
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
        if nearest is not None and hasattr(nearest, 'get_roadblock_id'):
            ego_rb = str(nearest.get_roadblock_id())
    except Exception:
        ego_rb = None

    if not ego_rb:
        return pruned_route_ids, 0, False, None, 'no_ego_rb'

    if intersection_pruned != 0:
        return pruned_route_ids, 0, False, ego_rb, f'skip_intersection_pruned={intersection_pruned}'

    if ego_rb in set(pruned_route_ids):
        return pruned_route_ids, 0, True, ego_rb, 'ego_rb_in_pruned_route'

    try:
        bfs = BreadthFirstSearchRoadBlock(ego_rb, map_api)
        (bridge_path, bridge_ids), found = bfs.search(target_roadblock_id=pruned_route_ids[:k_targets], max_depth=max_depth)
        bridge_ids = [str(x) for x in (bridge_ids or [])]
    except Exception as e:
        return pruned_route_ids, 0, False, ego_rb, f'bfs_exception: {e}'

    if not found or not bridge_ids:
        return pruned_route_ids, 0, False, ego_rb, 'bfs_not_found'

    new_route = _dedup_keep_order(bridge_ids + pruned_route_ids)
    return new_route, int(len(bridge_ids)), True, ego_rb, 'bfs_bridge_found'



def get_target_frame(conn, scenario_token, frame_index):
    """Get the target frame based on scenario token and frame index"""
    cursor = conn.cursor()
    
    try:
        scenario_token_bytes = bytes.fromhex(scenario_token)
        cursor.execute('SELECT token FROM scene WHERE token = ?', (scenario_token_bytes,))
        scenario = cursor.fetchone()
        
        if scenario:
            print(f"Found scenario: {scenario_token}")
            # First get log_token from scene (matching extract_ego_data approach)
            cursor.execute('SELECT log_token FROM scene WHERE token = ?', (scenario_token_bytes,))
            log_token = cursor.fetchone()[0]
            
            # Then query ego_pose filtered by log_token
            cursor.execute('''
                SELECT ep.token, ep.timestamp 
                FROM ego_pose ep
                WHERE ep.log_token = ?
                ORDER BY ep.timestamp
            ''', (log_token,))
            frames = cursor.fetchall()
            
            if frame_index < len(frames):
                target_frame = frames[frame_index]
                return target_frame[0], target_frame[1], target_frame[0]
    except Exception as e:
        print(f"Scenario lookup error: {e}")
    
    cursor.execute('SELECT token, timestamp FROM ego_pose ORDER BY timestamp')
    all_poses = cursor.fetchall()
    
    if frame_index < len(all_poses):
        pose = all_poses[frame_index]
        return pose['token'], pose['timestamp'], pose['token']
    
    pose = all_poses[0]
    return pose['token'], pose['timestamp'], pose['token']


def get_traffic_lights_at_timestamp(conn, timestamp, map_name):
    """Get traffic light status at a specific timestamp"""
    cursor = conn.cursor()
    
    try:
        # Fixed query: use correct table structure
        # traffic_light_status has: token, lidar_pc_token, lane_connector_id, status
        cursor.execute('''
            SELECT tls.status, tls.lane_connector_id, lp.timestamp
            FROM traffic_light_status tls
            JOIN lidar_pc lp ON tls.lidar_pc_token = lp.token
            WHERE lp.timestamp <= ?
            ORDER BY lp.timestamp DESC
            LIMIT 50
        ''', (timestamp,))
        results = cursor.fetchall()
        
        traffic_lights = {}
        for row in results:
            # Use lane_connector_id as the lane identifier
            lane_id = row[1]
            state = row[0]
            traffic_lights[lane_id] = state
        
        print(f"  Found {len(traffic_lights)} traffic lights")
        return traffic_lights
    except Exception as e:
        print(f"Traffic light query error: {e}")
        return {}


def extract_ego_data(conn, center_token, center_timestamp, scenario_token):
    """Extract ego pose data around center frame"""
    cursor = conn.cursor()
    
    # Get log_token from scene to filter ego_pose
    scenario_token_bytes = bytes.fromhex(scenario_token)
    cursor.execute('SELECT log_token FROM scene WHERE token = ?', (scenario_token_bytes,))
    result = cursor.fetchone()
    if result is None:
        print("Warning: scene not found, querying all ego_pose")
        log_token = None
    else:
        log_token = result[0]
    
    # Query ego_pose filtered by log_token
    if log_token:
        cursor.execute('''
            SELECT token, timestamp, x, y, z, qw, qx, qy, qz, 
                   vx, vy, vz, acceleration_x, acceleration_y
            FROM ego_pose 
            WHERE log_token = ?
            ORDER BY timestamp
        ''', (log_token,))
    else:
        cursor.execute('''
            SELECT token, timestamp, x, y, z, qw, qx, qy, qz, 
                   vx, vy, vz, acceleration_x, acceleration_y
            FROM ego_pose 
            ORDER BY timestamp
        ''')
    all_poses = cursor.fetchall()
    
    center_idx = None
    for i, row in enumerate(all_poses):
        if row['token'] == center_token:
            center_idx = i
            break
    
    if center_idx is None:
        for i, row in enumerate(all_poses):
            if abs(row['timestamp'] - center_timestamp) < 1000000:
                center_idx = i
                break
    
    if center_idx is None:
        center_idx = len(all_poses) // 2
    
    print(f"Center index: {center_idx}, total poses: {len(all_poses)}")
    
    ego_row = all_poses[center_idx]
    ego_x = ego_row['x']
    ego_y = ego_row['y']
    ego_heading = quaternion_to_heading(ego_row['qw'], ego_row['qx'], ego_row['qy'], ego_row['qz'])
    ego_vx = ego_row['vx']
    ego_vy = ego_row['vy']
    ego_ax = ego_row['acceleration_x']
    ego_ay = ego_row['acceleration_y']
    
    R = get_rotation_matrix_2d(-ego_heading)
    v_local = R @ np.array([ego_vx, ego_vy])
    
    ego_current_state = np.array([
        0.0, 0.0,
        np.cos(ego_heading), np.sin(ego_heading),
        v_local[0], v_local[1],
        ego_ax, ego_ay,
        1.0, 1.0
    ], dtype=np.float32)
    
    ego_future = np.zeros((EGO_FUTURE_LEN, 3), dtype=np.float32)
    # ego_pose is 100Hz, sample every 10 frames for 0.1s interval (10Hz)
    for i in range(EGO_FUTURE_LEN):
        idx = center_idx + (i + 1) * 10  # Every 10 frames = 0.1s at 100Hz
        if idx < len(all_poses):
            future_row = all_poses[idx]
            dx, dy = transform_to_ego_frame(future_row['x'], future_row['y'], ego_x, ego_y, ego_heading)
            future_heading = quaternion_to_heading(future_row['qw'], future_row['qx'], future_row['qy'], future_row['qz'])
            dheading = future_heading - ego_heading
            while dheading > np.pi: dheading -= 2 * np.pi
            while dheading < -np.pi: dheading += 2 * np.pi
            ego_future[i] = [dx, dy, dheading]
    
    # Extract ego_past: 21帧历史，10Hz采样
    ego_past = np.zeros((EGO_HISTORY_LEN, 3), dtype=np.float32)  # x, y, heading
    for i in range(EGO_HISTORY_LEN):
        idx = center_idx - (EGO_HISTORY_LEN - 1 - i) * 10
        if idx >= 0:
            past_row = all_poses[idx]
            dx, dy = transform_to_ego_frame(past_row['x'], past_row['y'], ego_x, ego_y, ego_heading)
            heading = quaternion_to_heading(past_row['qw'], past_row['qx'], past_row['qy'], past_row['qz'])
            dheading = heading - ego_heading
            while dheading > np.pi: dheading -= 2 * np.pi
            while dheading < -np.pi: dheading += 2 * np.pi
            ego_past[i] = [dx, dy, dheading]
    
    neighbor_past = np.zeros((MAX_NEIGHBORS, NEIGHBOR_HISTORY_LEN, 11), dtype=np.float32)
    
    for i in range(NEIGHBOR_HISTORY_LEN):
        idx = center_idx - (NEIGHBOR_HISTORY_LEN - 1 - i)
        if idx >= 0:
            past_row = all_poses[idx]
            dx, dy = transform_to_ego_frame(past_row['x'], past_row['y'], ego_x, ego_y, ego_heading)
            past_heading = quaternion_to_heading(past_row['qw'], past_row['qx'], past_row['qy'], past_row['qz'])
            v_local = R @ np.array([past_row['vx'], past_row['vy']])
            neighbor_past[0, i] = [
                dx, dy, np.cos(past_heading), np.sin(past_heading),
                v_local[0], v_local[1], past_row['acceleration_x'], past_row['acceleration_y'],
                1.8, 4.5, 1.0
            ]
        else:
            neighbor_past[0, i, -1] = 0.0
    
    return ego_current_state, ego_past, ego_future, neighbor_past, ego_x, ego_y, ego_heading, center_idx, all_poses


def extract_neighbor_agents(conn, center_timestamp, ego_x, ego_y, ego_heading):
    """Extract neighbor agent data"""
    cursor = conn.cursor()
    
    cursor.execute('SELECT token, timestamp FROM lidar_pc WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1', (center_timestamp,))
    center_lidar = cursor.fetchone()
    
    if center_lidar is None:
        return np.zeros((MAX_NEIGHBORS, NEIGHBOR_HISTORY_LEN, 11), dtype=np.float32), \
               np.zeros((MAX_NEIGHBORS, NEIGHBOR_FUTURE_LEN, 3), dtype=np.float32)
    
    center_lidar_token = center_lidar['token']
    
    cursor.execute('SELECT DISTINCT track_token FROM lidar_box WHERE lidar_pc_token = ?', (center_lidar_token,))
    tracks = cursor.fetchall()
    print(f"Found {len(tracks)} tracks at center frame")
    
    neighbor_past = np.zeros((MAX_NEIGHBORS, NEIGHBOR_HISTORY_LEN, 11), dtype=np.float32)
    neighbor_future = np.zeros((MAX_NEIGHBORS, NEIGHBOR_FUTURE_LEN, 3), dtype=np.float32)
    
    R = get_rotation_matrix_2d(-ego_heading)
    
    for agent_idx, track in enumerate(tracks[:MAX_NEIGHBORS - 1]):
        track_token = track['track_token']
        
        cursor.execute('''
            SELECT lp.timestamp, lb.x, lb.y, lb.z, lb.yaw, lb.vx, lb.vy, t.width, t.length
            FROM lidar_box lb
            JOIN track t ON lb.track_token = t.token
            JOIN lidar_pc lp ON lb.lidar_pc_token = lp.token
            WHERE lb.track_token = ?
            ORDER BY lp.timestamp
        ''', (track_token,))
        
        boxes = cursor.fetchall()
        if len(boxes) == 0:
            continue
        
        center_box_idx = 0
        min_diff = float('inf')
        for i, box in enumerate(boxes):
            diff = abs(box['timestamp'] - center_timestamp)
            if diff < min_diff:
                min_diff = diff
                center_box_idx = i
        
        if min_diff > 100000000:
            continue
        
        for i in range(NEIGHBOR_HISTORY_LEN):
            idx = center_box_idx - (NEIGHBOR_HISTORY_LEN - 1 - i)
            if idx >= 0:
                box = boxes[idx]
                dx, dy = transform_to_ego_frame(box['x'], box['y'], ego_x, ego_y, ego_heading)
                heading = box['yaw']
                v_local = R @ np.array([box['vx'], box['vy']])
                neighbor_past[agent_idx + 1, i] = [
                    dx, dy, np.cos(heading), np.sin(heading),
                    v_local[0], v_local[1], 0.0, 0.0,
                    box['width'], box['length'], 1.0
                ]
        
        # 提取未来轨迹：20Hz -> 10Hz采样，每2帧取1
        last_valid_idx = None
        last_valid_dx = None
        last_valid_dy = None
        last_valid_heading = None
        
        for i in range(NEIGHBOR_FUTURE_LEN):
            idx = center_box_idx + (i + 1) * 2  # 每2帧取1 = 10Hz
            if idx < len(boxes):
                box = boxes[idx]
                dx, dy = transform_to_ego_frame(box['x'], box['y'], ego_x, ego_y, ego_heading)
                heading = box['yaw']
                dheading = heading - ego_heading
                while dheading > np.pi: dheading -= 2 * np.pi
                while dheading < -np.pi: dheading += 2 * np.pi
                neighbor_future[agent_idx + 1, i] = [dx, dy, dheading]
                last_valid_idx = i
                last_valid_dx = dx
                last_valid_dy = dy
                last_valid_heading = dheading
        
        # 末尾填充：
        # - 若完全没有有效future点：用当前帧位置(从past/center_box获取)填充
        # - 若只有1个有效点：直接用该点常值填充
        # - 若>=2个有效点：用最后2个有效点估计速度，按匀速模型填充
        did_padding = False
        padding_mode = "none"

        def _normalize_angle(a: float) -> float:
            while a > np.pi:
                a -= 2 * np.pi
            while a < -np.pi:
                a += 2 * np.pi
            return a

        if last_valid_idx is None:
            # case 1: no valid future points -> fill with current(frame) position
            did_padding = True
            padding_mode = "fill_current"
            center_box = boxes[center_box_idx]
            fill_dx, fill_dy = transform_to_ego_frame(center_box['x'], center_box['y'], ego_x, ego_y, ego_heading)
            fill_heading = _normalize_angle(center_box['yaw'] - ego_heading)
            for i in range(NEIGHBOR_FUTURE_LEN):
                neighbor_future[agent_idx + 1, i] = [fill_dx, fill_dy, fill_heading]

        elif last_valid_idx == 0 and last_valid_idx < NEIGHBOR_FUTURE_LEN - 1:
            # case 2: only one valid point -> constant fill (no velocity estimation)
            did_padding = True
            padding_mode = "fill_constant_last"
            for i in range(1, NEIGHBOR_FUTURE_LEN):
                neighbor_future[agent_idx + 1, i] = [last_valid_dx, last_valid_dy, last_valid_heading]

        elif last_valid_idx < NEIGHBOR_FUTURE_LEN - 1:
            # case 3: >=2 valid points -> constant velocity fill
            did_padding = True
            padding_mode = "fill_const_vel"
            prev_dx = neighbor_future[agent_idx + 1, last_valid_idx - 1, 0]
            prev_dy = neighbor_future[agent_idx + 1, last_valid_idx - 1, 1]
            velocity_x = last_valid_dx - prev_dx  # 每0.1s的位移
            velocity_y = last_valid_dy - prev_dy
            for i in range(last_valid_idx + 1, NEIGHBOR_FUTURE_LEN):
                neighbor_future[agent_idx + 1, i, 0] = last_valid_dx + velocity_x * (i - last_valid_idx)
                neighbor_future[agent_idx + 1, i, 1] = last_valid_dy + velocity_y * (i - last_valid_idx)
                neighbor_future[agent_idx + 1, i, 2] = last_valid_heading  # 方向保持不变

        # Debug: record valid points, padding behavior, and filled data for each agent
        if EXTRACT_LOG_STYLE == 'legacy':
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
    """Extract static objects"""
    cursor = conn.cursor()
    
    cursor.execute('SELECT token FROM lidar_pc WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1', (center_timestamp,))
    result = cursor.fetchone()
    
    if result is None:
        return np.zeros((MAX_STATIC_OBJECTS, 10), dtype=np.float32)
    
    cursor.execute('SELECT x, y, z, width, length, height, yaw FROM lidar_box WHERE lidar_pc_token = ?', (result['token'],))
    boxes = cursor.fetchall()
    
    static_objects = np.zeros((MAX_STATIC_OBJECTS, 10), dtype=np.float32)
    R = get_rotation_matrix_2d(-ego_heading)
    
    for i, box in enumerate(boxes[:MAX_STATIC_OBJECTS]):
        dx, dy = transform_to_ego_frame(box['x'], box['y'], ego_x, ego_y, ego_heading)
        static_objects[i] = [dx, dy, box['z'], box['width'], box['length'], box['height'], box['yaw'], 0.0, 0.0, 1.0]
    
    return static_objects


def _interpolate_points(line, num_points):
    """Interpolate points to fixed number using shapely"""
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
    """
    Process lane to create polyline features with valid marking:
    - dim 0-1: polyline (x, y)
    - dim 2-3: polyline_vector (dx, dy)
    - dim 4-5: polyline_to_left
    - dim 6-7: polyline_to_right
    - dim 8-11: traffic_light (green/yellow/red/unknown)
    - Returns: (lane_feature, avails)
    """
    lane_feature = np.zeros((POLYLINE_LEN, LANE_DIM), dtype=np.float32)
    avails = np.zeros(POLYLINE_LEN, dtype=np.bool_)
    
    # Sampling policy:
    # - legacy: interpolate full baseline polyline to POLYLINE_LEN points
    # - optional (route lanes only): sample a local window around the point closest to ego to avoid
    #   "lane exists but all points are far" when ego is near one end of a long polyline.
    local_window = False
    if sample_local_around_ego:
        local_window = os.environ.get('ROUTE_LANE_SAMPLE_LOCAL_AROUND_EGO', '0') == '1'

    if len(centerline_coords) >= 2:
        coords_to_sample = centerline_coords
        if local_window and ego_point is not None:
            try:
                # Find closest node index to ego
                d2 = [((x - ego_point.x) ** 2 + (y - ego_point.y) ** 2) for x, y in centerline_coords]
                j = int(np.argmin(d2))
                # Take a local window of nodes around j
                half = int(os.environ.get('LANE_SAMPLE_LOCAL_HALF_NODES', '20'))
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
    """Extract lane information using map_api with proper boundary extraction"""
    layers = map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
    
    lanes = np.zeros((max_lanes, POLYLINE_LEN, LANE_DIM), dtype=np.float32)
    lanes_avails = np.zeros((max_lanes, POLYLINE_LEN), dtype=np.bool_)
    speed_limits = np.zeros(max_lanes, dtype=np.float32)
    has_speed_limits = np.zeros(max_lanes, dtype=np.float32)
    
    # Convert traffic_light_data keys to strings to match lane_obj.id
    traffic_light_lookup = {}
    if traffic_light_data:
        for lane_id, tl_state in traffic_light_data.items():
            lane_id_str = str(lane_id)  # Convert int to string
            if tl_state == 'green':
                traffic_light_lookup[lane_id_str] = [1, 0, 0, 0]
            elif tl_state == 'yellow':
                traffic_light_lookup[lane_id_str] = [0, 1, 0, 0]
            elif tl_state == 'red':
                traffic_light_lookup[lane_id_str] = [0, 0, 1, 0]
            else:
                traffic_light_lookup[lane_id_str] = [0, 0, 0, 1]
    
    print(f"  Traffic light lookup keys: {list(traffic_light_lookup.keys())[:5]}...")
    
    lanes_with_dist = []
    
    for layer_type in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
        if layer_type not in layers:
            continue
        
        lane_list = layers[layer_type]
        
        for lane_obj in lane_list:
            try:
                baseline_path = lane_obj.baseline_path
                centerline_coords = [(node.x, node.y) for node in baseline_path.discrete_path]
                
                if len(centerline_coords) < 2:
                    continue
                
                left_boundary_coords = []
                if hasattr(lane_obj, 'left_boundary') and lane_obj.left_boundary:
                    left_boundary_coords = [(node.x, node.y) for node in lane_obj.left_boundary.discrete_path]
                
                right_boundary_coords = []
                if hasattr(lane_obj, 'right_boundary') and lane_obj.right_boundary:
                    right_boundary_coords = [(node.x, node.y) for node in lane_obj.right_boundary.discrete_path]
                
                dist = np.mean([np.sqrt((x - point.x)**2 + (y - point.y)**2) for x, y in centerline_coords])
                
                lanes_with_dist.append({
                    'obj': lane_obj,
                    'centerline': centerline_coords,
                    'left': left_boundary_coords,
                    'right': right_boundary_coords,
                    'dist': dist
                })
            except Exception as e:
                continue
    
    lanes_with_dist.sort(key=lambda x: x['dist'])
    
    lane_idx = 0
    for lane_data in lanes_with_dist:
        if lane_idx >= max_lanes:
            break
        
        lane_obj = lane_data['obj']
        centerline_coords = lane_data['centerline']
        left_boundary_coords = lane_data['left']
        right_boundary_coords = lane_data['right']
        
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
        except:
            pass
        
        lane_idx += 1
    
    # Debug: record nonzero counts
    if EXTRACT_LOG_STYLE == 'legacy':
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
    """Extract route lanes (filtered by route_roadblock_ids if provided) based on ego position."""
    layers = map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
    
    route_lanes = np.zeros((max_route_lanes, POLYLINE_LEN, LANE_DIM), dtype=np.float32)
    route_lanes_avails = np.zeros((max_route_lanes, POLYLINE_LEN), dtype=np.bool_)
    route_speed_limits = np.zeros(max_route_lanes, dtype=np.float32)
    route_has_speed_limits = np.zeros(max_route_lanes, dtype=np.float32)

    route_roadblock_id_set = set(route_roadblock_ids) if route_roadblock_ids else None
    
    # Convert traffic_light_data keys to strings to match lane_obj.id
    traffic_light_lookup = {}
    if traffic_light_data:
        for lane_id, tl_state in traffic_light_data.items():
            lane_id_str = str(lane_id)  # Convert int to string
            if tl_state == 'green':
                traffic_light_lookup[lane_id_str] = [1, 0, 0, 0]
            elif tl_state == 'yellow':
                traffic_light_lookup[lane_id_str] = [0, 1, 0, 0]
            elif tl_state == 'red':
                traffic_light_lookup[lane_id_str] = [0, 0, 1, 0]
            else:
                traffic_light_lookup[lane_id_str] = [0, 0, 0, 1]
    
    all_lanes = []
    
    for layer_type in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
        if layer_type not in layers:
            continue
        
        lane_list = layers[layer_type]
        
        for lane_obj in lane_list:
            try:
                # Filter by route roadblock ids (route roadblock id chain).
                if route_roadblock_id_set is not None:
                    rb_id = None
                    try:
                        if hasattr(lane_obj, 'get_roadblock_id'):
                            rb_id = lane_obj.get_roadblock_id()
                        elif hasattr(lane_obj, 'roadblock_id'):
                            rb_id = lane_obj.roadblock_id
                    except Exception:
                        rb_id = None

                    # Skip lanes/lane_connectors not belonging to (pruned) route roadblocks.
                    if rb_id is None or str(rb_id) not in route_roadblock_id_set:
                        continue

                polygon = lane_obj.polygon
                if polygon is None:
                    continue
                
                centroid = polygon.centroid
                dist = np.sqrt((centroid.x - point.x)**2 + (centroid.y - point.y)**2)
                
                baseline_path = lane_obj.baseline_path
                centerline_coords = [(node.x, node.y) for node in baseline_path.discrete_path]
                
                left_boundary_coords = []
                if hasattr(lane_obj, 'left_boundary') and lane_obj.left_boundary:
                    left_boundary_coords = [(node.x, node.y) for node in lane_obj.left_boundary.discrete_path]
                
                right_boundary_coords = []
                if hasattr(lane_obj, 'right_boundary') and lane_obj.right_boundary:
                    right_boundary_coords = [(node.x, node.y) for node in lane_obj.right_boundary.discrete_path]
                
                all_lanes.append({
                    'obj': lane_obj,
                    'centerline': centerline_coords,
                    'left': left_boundary_coords,
                    'right': right_boundary_coords,
                    'dist': dist
                })
            except Exception as e:
                continue
    
    all_lanes.sort(key=lambda x: x['dist'])
    
    route_idx = 0
    for lane_data in all_lanes[:max_route_lanes]:
        if route_idx >= max_route_lanes:
            break
        
        lane_obj = lane_data['obj']
        centerline_coords = lane_data['centerline']
        left_boundary_coords = lane_data['left']
        right_boundary_coords = lane_data['right']
        
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
        except:
            pass
        
        route_idx += 1
    
    # Debug: record nonzero counts
    if EXTRACT_LOG_STYLE == 'legacy':
        try:
            DEBUG_LOGGER.info(
                f"route_lanes nonzero={int(np.count_nonzero(route_lanes))} avails_true={int(np.count_nonzero(route_lanes_avails))} filled_route_lanes={route_idx}/{max_route_lanes}"
            )
        except Exception as e:
            DEBUG_LOGGER.warning(f"route_lanes debug log failed: {e}")

    return route_lanes, route_lanes_avails, route_speed_limits, route_has_speed_limits


def generate_csv_summary(features, csv_path):
    """Generate CSV file with field summaries"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w') as f:
        f.write("field_name,shape,dtype,min,max,mean,nonzero_count,nonzero_percent\n")
        
        for name, arr in features.items():
            arr_min = float(arr.min())
            arr_max = float(arr.max())
            arr_mean = float(arr.mean())
            nonzero = np.count_nonzero(arr)
            total = arr.size
            nonzero_pct = nonzero / total * 100
            
            f.write(f"{name},{arr.shape},{arr.dtype},{arr_min:.4f},{arr_max:.4f},{arr_mean:.4f},{nonzero},{nonzero_pct:.2f}%\n")
    
    print(f"CSV summary saved to: {csv_path}")




def _get_db_path_from_conn(conn: sqlite3.Connection) -> str:
    """Best-effort absolute/relative path for the opened sqlite DB."""
    try:
        cur = conn.cursor()
        cur.execute('PRAGMA database_list')
        rows = cur.fetchall()
        # rows: [(seq,name,file), ...] -> pick main
        for r in rows:
            if len(r) >= 3 and r[1] == 'main':
                return str(r[2])
        if rows and len(rows[0]) >= 3:
            return str(rows[0][2])
    except Exception:
        pass
    return ''


def extract_features(conn, map_api, scenario_token_hex: str, frame_index: int, *, debug_log: bool = True) -> dict[str, np.ndarray]:
    """Pure feature extraction.

    Args:
        conn: sqlite3 connection (row_factory should be sqlite3.Row).
        map_api: nuPlan map api instance for the scenario location.
        scenario_token_hex: scene.token hex string.
        frame_index: index within ego_pose rows filtered by scene.log_token (ORDER BY timestamp).

    Returns:
        Dict of feature arrays, matching the saved NPZ keys.
    """
    db_path = _get_db_path_from_conn(conn)
    map_name = getattr(map_api, 'map_name', None) or getattr(map_api, '_map_name', None) or MAP_NAME

    center_token, center_timestamp, _ = get_target_frame(conn, scenario_token_hex, int(frame_index))

    ego_current_state, ego_past, ego_future, neighbor_past, ego_x, ego_y, ego_heading, _, _ =         extract_ego_data(conn, center_token, center_timestamp, scenario_token_hex)

    traffic_light_data = get_traffic_lights_at_timestamp(conn, center_timestamp, map_name)

    neighbor_past_agents, neighbor_future = extract_neighbor_agents(conn, center_timestamp, ego_x, ego_y, ego_heading)
    for i in range(1, MAX_NEIGHBORS):
        if np.any(neighbor_past_agents[i, :, -1] != 0):
            neighbor_past[i] = neighbor_past_agents[i]

    static_objects = extract_static_objects(conn, center_timestamp, ego_x, ego_y, ego_heading)

    point = Point2D(ego_x, ego_y)

    # Route roadblock ids from scenario.get_route_roadblock_ids (+ correction/pruning)
    try:
        route_roadblock_ids = get_pruned_route_roadblock_ids(conn, db_path, scenario_token_hex, map_api, map_name)
    except Exception:
        route_roadblock_ids = None

    lanes, lanes_avails, lanes_speed_limit, lanes_has_speed_limit = extract_lanes(
        point, map_api, radius=100, max_lanes=MAX_LANES, ego_heading=ego_heading,
        traffic_light_data=traffic_light_data
    )

    # ---- Single-case BFS bridge minimal experiment ----
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
    try:
        prox_layers = map_api.get_proximal_map_objects(
            point,
            150,
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        )
        for lt in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
            for lane_obj in (prox_layers.get(lt, []) or []):
                try:
                    rb = lane_obj.get_roadblock_id() if hasattr(lane_obj, 'get_roadblock_id') else None
                    if rb is not None:
                        proximal_rb_ids.add(str(rb))
                except Exception:
                    continue
    except Exception:
        prox_layers = {}

    pruned_route_set = set([str(x) for x in (route_roadblock_ids or [])])
    intersection_pruned = int(len(proximal_rb_ids.intersection(pruned_route_set)))

    # Decide whether to apply BFS-bridge / route realignment.
    # Legacy trigger: intersection_pruned==0.
    # New trigger (for soft-flag root cause): even if intersection_pruned>0, if the nearest route lane is still far,
    # it likely means the route_roadblock_ids are not aligned to ego's current segment (near overlap appears only late).
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

    bridge_trigger_dist_m = float(os.environ.get('BFS_BRIDGE_TRIGGER_DIST_M', '10'))
    rmin_old_m = _min_dist_m(route_lanes_old, route_lanes_avails_old)

    new_route_ids = route_roadblock_ids or []
    bridge_found = False
    bridge_len = 0
    ego_rb = None
    bridge_reason = 'skip'

    need_bridge = False
    if route_roadblock_ids:
        if intersection_pruned == 0:
            need_bridge = True
            bridge_reason = 'intersection_pruned==0'
        elif (rmin_old_m is not None) and (rmin_old_m > bridge_trigger_dist_m):
            # Soft-flag trigger: route exists but is far. First try a cheap realignment: if there is any overlap
            # between proximal roadblocks and route roadblocks, start the route from the earliest overlap.
            try:
                route_list = [str(x) for x in route_roadblock_ids]
                inter = proximal_rb_ids.intersection(set(route_list))
                if inter:
                    pos = {v: i for i, v in enumerate(route_list)}
                    idx_min = min(pos[x] for x in inter if x in pos)
                    if idx_min is not None and idx_min > 0:
                        new_route_ids = list(route_roadblock_ids)[idx_min:]
                        bridge_found = True
                        bridge_len = 0
                        ego_rb = None
                        bridge_reason = f'realign_from_overlap_idx={idx_min} (rmin_old_m>{bridge_trigger_dist_m})'
                    else:
                        need_bridge = True
                        bridge_reason = f'rmin_old_m>{bridge_trigger_dist_m}'
                else:
                    need_bridge = True
                    bridge_reason = f'rmin_old_m>{bridge_trigger_dist_m}'
            except Exception:
                need_bridge = True
                bridge_reason = f'rmin_old_m>{bridge_trigger_dist_m}'

    if need_bridge and route_roadblock_ids:
        new_route_ids, bridge_len, bridge_found, ego_rb, bridge_reason = bfs_bridge_route_if_needed(
            map_api,
            point,
            list(route_roadblock_ids),
            intersection_pruned=intersection_pruned,
            radius=150,
            k_targets=10,
            max_depth=80,
        )

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

    # Persist experiment logs to avoid stdout truncation (kept identical to legacy main).
    # Batch export should disable this to avoid massive I/O.
    if debug_log:
        try:
            out_dir = '/workspace/validation_output'
            try:
                os.makedirs(out_dir, exist_ok=True)
                test_path = os.path.join(out_dir, '.write_test')
                with open(test_path, 'w') as f:
                    f.write('ok')
                os.remove(test_path)
            except Exception:
                out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'validation_output'))
                os.makedirs(out_dir, exist_ok=True)

            log_path = os.path.join(out_dir, 'bfs_single_case.log')
            with open(log_path, 'a') as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(
                    f"db={os.path.basename(db_path)} scene_token={scenario_token_hex} frame={int(frame_index)} map={map_name}\n"
                )
                f.write(
                    f"intersection_pruned={intersection_pruned} ego_rb={ego_rb} bridge_found={bridge_found} bridge_len={bridge_len} reason={bridge_reason}\n"
                )
                f.write(
                    f"route_len_old={(len(route_roadblock_ids) if route_roadblock_ids else 0)} route_len_new={(len(new_route_ids) if new_route_ids else 0)}\n"
                )
                f.write(f"avails_sum_old={avails_sum_old} avails_sum_new={avails_sum_new}\n")

            import json
            json_path = os.path.join(out_dir, 'bfs_single_case_result.json')
            with open(json_path, 'w') as f:
                json.dump(
                    {
                        'db_basename': os.path.basename(db_path),
                        'scene_token_hex': scenario_token_hex,
                        'frame_index': int(frame_index),
                        'map_name': map_name,
                        'intersection_pruned': int(intersection_pruned),
                        'ego_rb': ego_rb,
                        'bridge_found': bool(bridge_found),
                        'bridge_len': int(bridge_len),
                        'bridge_reason': bridge_reason,
                        'route_len_old': int(len(route_roadblock_ids) if route_roadblock_ids else 0),
                        'route_len_new': int(len(new_route_ids) if new_route_ids else 0),
                        'avails_sum_old': int(avails_sum_old),
                        'avails_sum_new': int(avails_sum_new),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    return {
        'ego_current_state': ego_current_state,
        'ego_past': ego_past,
        'ego_agent_future': ego_future,
        'neighbor_agents_past': neighbor_past,
        'neighbor_agents_future': neighbor_future,
        'static_objects': static_objects,
        'lanes': lanes,
        'lanes_avails': lanes_avails,
        'route_lanes': route_lanes,
        'route_lanes_avails': route_lanes_avails,
        'lanes_speed_limit': lanes_speed_limit,
        'lanes_has_speed_limit': lanes_has_speed_limit,
        'route_lanes_speed_limit': route_lanes_speed_limit,
        'route_lanes_has_speed_limit': route_lanes_has_speed_limit,
    }

def run_extraction(db_path: str,
                   scenario_token: str,
                   center_frame_index: int,
                   output_path: str,
                   csv_output_path: str | None = None):
    """Programmatic entrypoint for batch runs.

    This keeps the core logic in this file, while allowing wrappers to set
    per-scene parameters safely.
    """
    global DB_PATH, SCENARIO_TOKEN, CENTER_FRAME_INDEX, OUTPUT_PATH, CSV_OUTPUT_PATH, DEBUG_LOGGER

    DB_PATH = db_path
    SCENARIO_TOKEN = scenario_token
    CENTER_FRAME_INDEX = int(center_frame_index)
    OUTPUT_PATH = output_path
    if csv_output_path is not None:
        CSV_OUTPUT_PATH = csv_output_path

    # Re-init debug logger so each output gets its own log file name.
    try:
        DEBUG_LOGGER = _init_debug_logger()
    except Exception:
        pass

    return main()


def main():
    print("=" * 60)
    print("Starting complete feature extraction WITH VALID MARKING...")
    print(f"Scenario: {SCENARIO_TOKEN}, Frame: {CENTER_FRAME_INDEX}")
    print("=" * 60)
    
    conn = load_db()
    cursor = conn.cursor()

    # Determine the correct map from DB metadata (do NOT infer from filename).
    global MAP_NAME
    location = get_location_from_log(conn)
    MAP_NAME = map_name_from_location(location)
    print(f"DB location: {location} -> MAP_NAME: {MAP_NAME}")
    try:
        DEBUG_LOGGER.info(f"DB location={location} -> MAP_NAME={MAP_NAME}")
    except Exception:
        pass

    print("\n[5/7] Loading map API...")
    map_api = get_maps_api(MAP_ROOT, MAP_VERSION, MAP_NAME)
    print(f"  Map: {map_api.map_name}")

    # Feature extraction (pure function)
    features = extract_features(conn, map_api, SCENARIO_TOKEN, CENTER_FRAME_INDEX)

    conn.close()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH, **features)

    print(f"\n{'=' * 60}")
    print(f"Saved NPZ to: {OUTPUT_PATH}")
    print(f"{'=' * 60}")

    generate_csv_summary(features, CSV_OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("FEATURE SUMMARY REPORT")
    print("=" * 60)

    for name, arr in features.items():
        arr_min = arr.min()
        arr_max = arr.max()
        nonzero = np.count_nonzero(arr)
        total = arr.size
        nonzero_pct = nonzero / total * 100
        print(f"\n{name}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Value range: [{arr_min:.4f}, {arr_max:.4f}]")
        print(f"  Non-zero: {nonzero}/{total} ({nonzero_pct:.1f}%)")

    print("\n" + "=" * 60)
    print("LANE FEATURE DETAIL (dim 0-11)")
    print("=" * 60)
    lane_example = features['lanes'][0]
    for dim in range(LANE_DIM):
        dim_data = lane_example[:, dim]
        nonzero = np.count_nonzero(dim_data)
        print(f"  dim {dim}: nonzero={nonzero}/{POLYLINE_LEN}, min={dim_data.min():.4f}, max={dim_data.max():.4f}")

    print("\n" + "=" * 60)
    print("LANE VALID MARKING DETAIL")
    print("=" * 60)
    lanes_avails = features['lanes_avails']
    valid_lanes = np.sum(np.any(lanes_avails, axis=1))
    total_valid_points = np.sum(lanes_avails)
    total_points = lanes_avails.size
    print(f"  Valid lanes: {valid_lanes}/{MAX_LANES}")
    print(f"  Valid points: {total_valid_points}/{total_points} ({total_valid_points/total_points*100:.1f}%)")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':

    main()
