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
from shapely import LineString

# Add nuplan-visualization to path
import sys
sys.path.insert(0, '/home/zhouyida/.openclaw/workspace/diffusion-planner-project/nuplan-visualization')

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData
from nuplan.common.actor_state.state_representation import Point2D

# Constants
DB_PATH = '/home/zhouyida/.openclaw/workspace/diffusion-planner-project/data/nuplan/data/cache/mini/2021.06.28.16.29.11_veh-38_01415_01821.db'
MAP_ROOT = '/home/zhouyida/.openclaw/workspace/diffusion-planner-project/data/nuplan/maps'
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

# Target scenario 4: Singapore 18701 (2021.10.06.07.26.10_veh-52_00006_00398.db)
DB_PATH = '/home/zhouyida/.openclaw/workspace/diffusion-planner-project/data/nuplan/data/cache/mini/2021.10.06.07.26.10_veh-52_00006_00398.db'
MAP_NAME = 'sg-one-north'

SCENARIO_TOKEN = '6a066b79aedd5ad3'
CENTER_FRAME_INDEX = 18701

OUTPUT_PATH = '/workspace/data_process/npz_scenes/singapore_18701_with_past.npz'
CSV_OUTPUT_PATH = '/workspace/data_process/npz_scenes/singapore_18701.csv'


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


def load_db():
    """Load database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
        
        # 末尾填充：用最后2个有效点计算速度，按匀速模型填充剩余帧
        if last_valid_idx is not None and last_valid_idx < NEIGHBOR_FUTURE_LEN - 1:
            if last_valid_idx >= 1:
                # 获取最后两个有效点的位置
                prev_dx = neighbor_future[agent_idx + 1, last_valid_idx - 1, 0]
                prev_dy = neighbor_future[agent_idx + 1, last_valid_idx - 1, 1]
                velocity_x = last_valid_dx - prev_dx  # 每0.1s的位移
                velocity_y = last_valid_dy - prev_dy
            else:
                # 只有一个有效点，速度为0
                velocity_x = 0
                velocity_y = 0
            
            # 匀速填充
            for i in range(last_valid_idx + 1, NEIGHBOR_FUTURE_LEN):
                neighbor_future[agent_idx + 1, i, 0] = last_valid_dx + velocity_x * (i - last_valid_idx)
                neighbor_future[agent_idx + 1, i, 1] = last_valid_dy + velocity_y * (i - last_valid_idx)
                neighbor_future[agent_idx + 1, i, 2] = last_valid_heading  # 方向保持不变
    
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


def _lane_polyline_process_with_avails(lane_obj, centerline_coords, left_coords, right_coords, traffic_light_state, ego_point, ego_heading):
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
    
    if len(centerline_coords) >= 2:
        sampled = _interpolate_points(centerline_coords, POLYLINE_LEN)
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
            lane_obj, centerline_coords, left_boundary_coords, right_boundary_coords,
            traffic_light_state, point, ego_heading
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
    
    return lanes, lanes_avails, speed_limits, has_speed_limits


def extract_route_lanes(point, map_api, radius=150, max_route_lanes=25, ego_heading=0, traffic_light_data=None):
    """Extract route lanes based on ego position"""
    layers = map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
    
    route_lanes = np.zeros((max_route_lanes, POLYLINE_LEN, LANE_DIM), dtype=np.float32)
    route_lanes_avails = np.zeros((max_route_lanes, POLYLINE_LEN), dtype=np.bool_)
    route_speed_limits = np.zeros(max_route_lanes, dtype=np.float32)
    route_has_speed_limits = np.zeros(max_route_lanes, dtype=np.float32)
    
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
            lane_obj, centerline_coords, left_boundary_coords, right_boundary_coords,
            traffic_light_state, point, ego_heading
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


def main():
    print("=" * 60)
    print("Starting complete feature extraction WITH VALID MARKING...")
    print(f"Scenario: {SCENARIO_TOKEN}, Frame: {CENTER_FRAME_INDEX}")
    print("=" * 60)
    
    conn = load_db()
    cursor = conn.cursor()
    
    center_token, center_timestamp, ego_pose_token = get_target_frame(conn, SCENARIO_TOKEN, CENTER_FRAME_INDEX)
    print(f"Center token: {center_token}")
    print(f"Center timestamp: {center_timestamp}")
    
    print("\n[1/7] Extracting ego data...")
    ego_current_state, ego_past, ego_future, neighbor_past, ego_x, ego_y, ego_heading, center_idx, all_poses = \
        extract_ego_data(conn, center_token, center_timestamp, SCENARIO_TOKEN)
    print(f"  ego_current_state: {ego_current_state.shape}")
    print(f"  ego_past: {ego_past.shape}")
    print(f"  ego_agent_future: {ego_future.shape}")
    
    print("\n[2/7] Extracting traffic light data...")
    traffic_light_data = get_traffic_lights_at_timestamp(conn, center_timestamp, MAP_NAME)
    print(f"  Found {len(traffic_light_data)} traffic lights")
    
    print("\n[3/7] Extracting neighbor agents...")
    neighbor_past_agents, neighbor_future = extract_neighbor_agents(conn, center_timestamp, ego_x, ego_y, ego_heading)
    
    for i in range(1, MAX_NEIGHBORS):
        if np.any(neighbor_past_agents[i, :, -1] != 0):
            neighbor_past[i] = neighbor_past_agents[i]
    
    print(f"  neighbor_agents_past: {neighbor_past.shape}")
    print(f"  neighbor_agents_future: {neighbor_future.shape}")
    
    print("\n[4/7] Extracting static objects...")
    static_objects = extract_static_objects(conn, center_timestamp, ego_x, ego_y, ego_heading)
    print(f"  static_objects: {static_objects.shape}")
    
    print("\n[5/7] Loading map API...")
    map_api = get_maps_api(MAP_ROOT, MAP_VERSION, MAP_NAME)
    print(f"  Map: {map_api.map_name}")
    
    point = Point2D(ego_x, ego_y)
    
    print("\n[6/7] Extracting lanes with boundaries and valid marking...")
    lanes, lanes_avails, lanes_speed_limit, lanes_has_speed_limit = extract_lanes(
        point, map_api, radius=100, max_lanes=MAX_LANES, ego_heading=ego_heading,
        traffic_light_data=traffic_light_data
    )
    print(f"  lanes: {lanes.shape}")
    print(f"  lanes_avails: {lanes_avails.shape}")
    print(f"  lanes_speed_limit: {lanes_speed_limit.shape}")
    print(f"  lanes_has_speed_limit: {lanes_has_speed_limit.shape}")
    
    print("\n[7/7] Extracting route lanes with boundaries and valid marking...")
    route_lanes, route_lanes_avails, route_lanes_speed_limit, route_lanes_has_speed_limit = extract_route_lanes(
        point, map_api, radius=150, max_route_lanes=MAX_ROUTE_LANES, ego_heading=ego_heading,
        traffic_light_data=traffic_light_data
    )
    print(f"  route_lanes: {route_lanes.shape}")
    print(f"  route_lanes_avails: {route_lanes_avails.shape}")
    print(f"  route_lanes_speed_limit: {route_lanes_speed_limit.shape}")
    print(f"  route_lanes_has_speed_limit: {route_lanes_has_speed_limit.shape}")
    
    conn.close()
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH,
        ego_current_state=ego_current_state,
        ego_past=ego_past,
        ego_agent_future=ego_future,
        neighbor_agents_past=neighbor_past,
        neighbor_agents_future=neighbor_future,
        static_objects=static_objects,
        lanes=lanes,
        lanes_avails=lanes_avails,
        route_lanes=route_lanes,
        route_lanes_avails=route_lanes_avails,
        lanes_speed_limit=lanes_speed_limit,
        lanes_has_speed_limit=lanes_has_speed_limit,
        route_lanes_speed_limit=route_lanes_speed_limit,
        route_lanes_has_speed_limit=route_lanes_has_speed_limit)
    
    print(f"\n{'=' * 60}")
    print(f"Saved NPZ to: {OUTPUT_PATH}")
    print(f"{'=' * 60}")
    
    features = {
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
        'route_lanes_has_speed_limit': route_lanes_has_speed_limit
    }
    
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
    lane_example = lanes[0]
    for dim in range(LANE_DIM):
        dim_data = lane_example[:, dim]
        nonzero = np.count_nonzero(dim_data)
        print(f"  dim {dim}: nonzero={nonzero}/{POLYLINE_LEN}, min={dim_data.min():.4f}, max={dim_data.max():.4f}")
    
    print("\n" + "=" * 60)
    print("LANE VALID MARKING DETAIL")
    print("=" * 60)
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
