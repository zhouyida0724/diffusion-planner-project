#!/usr/bin/env python3
"""
Data Preprocessing Script for Diffusion Planner

This script processes nuPlan scenarios into NPZ format for training.
Reference: https://github.com/ZhengYinan-AIR/Diffusion-Planner

Usage:
    python preprocess.py --data_path /path/to/nuplan/data --map_path /path/to/maps --save_path ./cache --num_scenarios 100
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add nuplan-visualization to path
NUPLAN_VIS_PATH = '/workspace/nuplan-visualization'
DIFFUSION_PATH = '/workspace'

sys.path.insert(0, NUPLAN_VIS_PATH)
sys.path.insert(0, DIFFUSION_PATH)

import numpy as np
from tqdm import tqdm


def build_scenario_filter(
    num_scenarios: int = None,
    scenario_tokens: list = None,
    log_names: list = None,
    shuffle: bool = True
):
    """
    Build scenario filter for selecting scenarios from nuPlan.
    """
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    
    return ScenarioFilter(
        scenario_types=None,
        scenario_tokens=scenario_tokens,
        log_names=log_names,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=num_scenarios,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=True,
        remove_invalid_goals=False,
        shuffle=shuffle,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None
    )


def load_scenarios(
    data_path: str,
    map_path: str,
    num_scenarios: int = None,
    log_names: list = None
):
    """
    Load scenarios from nuPlan dataset.
    """
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
    
    map_version = "nuplan-maps-v1.0"
    sensor_root = None
    db_files = None
    
    # Build scenario filter
    scenario_filter = build_scenario_filter(
        num_scenarios=num_scenarios,
        log_names=log_names
    )
    
    # Create scenario builder
    builder = NuPlanScenarioBuilder(data_path, map_path, sensor_root, db_files, map_version)
    
    # Create worker for parallel processing
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    
    # Get scenarios
    scenarios = builder.get_scenarios(scenario_filter, worker)
    
    return scenarios


def process_scenario(scenario, config: dict):
    """
    Process a single scenario into NPZ format.
    
    Args:
        scenario: nuPlan scenario object
        config: Configuration dict
        
    Returns:
        Path to saved NPZ file
    """
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.diffusion_planner.data_process.agent_process import (
        agent_past_process,
        sampled_tracked_objects_to_array_list,
        sampled_static_objects_to_array_list,
        agent_future_process
    )
    from nuplan.diffusion_planner.data_process.map_process import (
        get_neighbor_vector_set_map,
        map_process
    )
    from nuplan.diffusion_planner.data_process.ego_process import (
        get_ego_past_array_from_scenario,
        get_ego_future_array_from_scenario,
        calculate_additional_ego_states
    )
    from nuplan.diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
    
    # Temporal parameters
    past_time_horizon = 2.0
    num_past_poses = int(10 * past_time_horizon)
    future_time_horizon = 8.0
    num_future_poses = int(10 * future_time_horizon)
    
    # Agent parameters
    num_agents = config.get('agent_num', 32)
    num_static = config.get('static_objects_num', 5)
    max_ped_bike = 10
    
    # Map parameters
    radius = 100
    map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES']
    lane_num = config.get('lane_num', 70)
    lane_len = config.get('lane_len', 20)
    route_num = config.get('route_num', 25)
    route_len = config.get('route_len', 20)
    
    # Get scenario info
    map_name = scenario._map_name
    token = scenario.token
    map_api = scenario.map_api
    
    # Ego and agent past
    ego_state = scenario.initial_ego_state
    ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
    anchor_ego_state = np.array([
        ego_state.rear_axle.x, 
        ego_state.rear_axle.y, 
        ego_state.rear_axle.heading
    ], dtype=np.float64)
    
    ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(
        scenario, num_past_poses, past_time_horizon
    )
    
    # Get past tracked objects
    present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
    past_tracked_objects = [
        tracked_objects.tracked_objects
        for tracked_objects in scenario.get_past_tracked_objects(
            iteration=0, time_horizon=past_time_horizon, num_samples=num_past_poses
        )
    ]
    sampled_past_observations = past_tracked_objects + [present_tracked_objects]
    neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(
        sampled_past_observations
    )
    
    static_objects, static_objects_types = sampled_static_objects_to_array_list(
        present_tracked_objects
    )
    
    ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
        agent_past_process(
            ego_agent_past, neighbor_agents_past, neighbor_agents_types,
            num_agents, static_objects, static_objects_types,
            num_static, max_ped_bike, anchor_ego_state
        )
    
    # Map processing
    route_roadblock_ids = scenario.get_route_roadblock_ids()
    traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
    
    if route_roadblock_ids != ['']:
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
    
    coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
        map_api, map_features, ego_coords, radius, traffic_light_data
    )
    
    max_elements = {
        'LANE': lane_num,
        'LEFT_BOUNDARY': lane_num,
        'RIGHT_BOUNDARY': lane_num,
        'ROUTE_LANES': route_num
    }
    max_points = {
        'LANE': lane_len,
        'LEFT_BOUNDARY': lane_len,
        'RIGHT_BOUNDARY': lane_len,
        'ROUTE_LANES': route_len
    }
    
    vector_map = map_process(
        route_roadblock_ids, anchor_ego_state, coords, traffic_light_data,
        speed_limit, lane_route, map_features, max_elements, max_points
    )
    
    # Ego and agent future
    ego_agent_future = get_ego_future_array_from_scenario(
        scenario, ego_state, num_future_poses, future_time_horizon
    )
    
    future_tracked_objects = [
        tracked_objects.tracked_objects
        for tracked_objects in scenario.get_future_tracked_objects(
            iteration=0, time_horizon=future_time_horizon, num_samples=num_future_poses
        )
    ]
    
    sampled_future_observations = [present_tracked_objects] + future_tracked_objects
    future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(
        sampled_future_observations
    )
    neighbor_agents_future = agent_future_process(
        anchor_ego_state, future_tracked_objects_array_list,
        num_agents, neighbor_indices
    )
    
    # Ego current state
    ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)
    
    # Gather data
    data = {
        "map_name": map_name,
        "token": token,
        "ego_current_state": ego_current_state,
        "ego_agent_future": ego_agent_future,
        "neighbor_agents_past": neighbor_agents_past,
        "neighbor_agents_future": neighbor_agents_future,
        "static_objects": static_objects
    }
    data.update(vector_map)
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess nuPlan data for Diffusion Planner training'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='/workspace/data/nuplan/data/cache/mini',
        help='Path to nuPlan data'
    )
    parser.add_argument(
        '--map_path',
        type=str,
        default='/workspace/data/nuplan/maps',
        help='Path to nuPlan maps'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='/workspace/data/processed',
        help='Path to save processed data'
    )
    parser.add_argument(
        '--num_scenarios',
        type=int,
        default=10,
        help='Number of scenarios to process'
    )
    parser.add_argument(
        '--agent_num',
        type=int,
        default=32,
        help='Number of agents to process'
    )
    parser.add_argument(
        '--static_objects_num',
        type=int,
        default=5,
        help='Number of static objects'
    )
    parser.add_argument(
        '--lane_num',
        type=int,
        default=70,
        help='Number of lanes'
    )
    parser.add_argument(
        '--lane_len',
        type=int,
        default=20,
        help='Number of points per lane'
    )
    parser.add_argument(
        '--route_num',
        type=int,
        default=25,
        help='Number of route lanes'
    )
    parser.add_argument(
        '--route_len',
        type=int,
        default=20,
        help='Number of points per route lane'
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'agent_num': args.agent_num,
        'static_objects_num': args.static_objects_num,
        'lane_num': args.lane_num,
        'lane_len': args.lane_len,
        'route_num': args.route_num,
        'route_len': args.route_len
    }
    
    # Create output directory
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading scenarios from {args.data_path}...")
    scenarios = load_scenarios(
        data_path=args.data_path,
        map_path=args.map_path,
        num_scenarios=args.num_scenarios
    )
    print(f"Loaded {len(scenarios)} scenarios")
    
    print(f"Processing {len(scenarios)} scenarios...")
    success_count = 0
    
    for scenario in tqdm(scenarios, desc="Processing"):
        try:
            data = process_scenario(scenario, config)
            
            # Save to disk
            output_file = save_path / f"{data['map_name']}_{data['token']}.npz"
            np.savez(output_file, **data)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing scenario {scenario.token}: {e}")
            continue
    
    print(f"Successfully processed {success_count}/{len(scenarios)} scenarios")
    print(f"Saved to: {args.save_path}")
    
    # Save file list
    npz_files = list(save_path.glob('*.npz'))
    file_list = [f.name for f in npz_files]
    
    list_path = save_path / 'training_data_list.json'
    with open(list_path, 'w') as f:
        json.dump(file_list, f, indent=2)
    
    print(f"Saved file list to: {list_path}")


if __name__ == '__main__':
    main()
