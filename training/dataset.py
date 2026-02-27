"""
MVP Dataset for Diffusion Planner
Simple PyTorch dataset for loading NPZ files
"""
import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset


class DiffusionPlannerDataset(Dataset):
    """Simple dataset for loading NPZ data"""
    
    def __init__(self, data_dir, data_list, config):
        self.data_dir = data_dir
        self.config = config
        
        # Load data list
        list_path = os.path.join(data_dir, data_list) if not data_list.startswith('/') else data_list
        with open(list_path, 'r') as f:
            self.data_list = json.load(f)
        
        print(f"Loaded {len(self.data_list)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # Load NPZ file
        npz_path = os.path.join(self.data_dir, self.data_list[idx])
        data = np.load(npz_path)
        
        # Extract fields - handle different array shapes
        ego_current_state = torch.from_numpy(data['ego_current_state']).float()
        ego_agent_future = torch.from_numpy(data['ego_agent_future']).float()
        
        neighbor_agents_past = torch.from_numpy(data['neighbor_agents_past']).float()
        neighbor_agents_future = torch.from_numpy(data['neighbor_agents_future']).float()
        
        static_objects = torch.from_numpy(data['static_objects']).float()
        lanes = torch.from_numpy(data['lanes']).float()
        lanes_speed_limit = torch.from_numpy(data['lanes_speed_limit']).float()
        lanes_has_speed_limit = torch.from_numpy(data['lanes_has_speed_limit']).float()
        
        route_lanes = torch.from_numpy(data['route_lanes']).float()
        route_lanes_speed_limit = torch.from_numpy(data['route_lanes_speed_limit']).float()
        route_lanes_has_speed_limit = torch.from_numpy(data['route_lanes_has_speed_limit']).float()
        
        # Truncate/pad to expected sizes based on config
        # neighbor_agents_past: (32, 21, 11) -> keep first past_neighbor_num
        neighbor_agents_past = neighbor_agents_past[:self.config.past_neighbor_num]
        
        # neighbor_agents_future: (32, 80, 3) -> keep first predicted_neighbor_num
        neighbor_agents_future = neighbor_agents_future[:self.config.predicted_neighbor_num]
        
        # lanes: (70, 20, 12) -> keep first lane_num
        lanes = lanes[:self.config.lane_num]
        lanes_speed_limit = lanes_speed_limit[:self.config.lane_num]
        lanes_has_speed_limit = lanes_has_speed_limit[:self.config.lane_num]
        
        # route_lanes: (25, 20, 12) -> keep first route_num
        route_lanes = route_lanes[:self.config.route_num]
        route_lanes_speed_limit = route_lanes_speed_limit[:self.config.route_num]
        route_lanes_has_speed_limit = route_lanes_has_speed_limit[:self.config.route_num]
        
        # static_objects: (5, 10) -> keep first static_objects_num
        static_objects = static_objects[:self.config.static_objects_num]
        
        return {
            'ego_current_state': ego_current_state,
            'ego_agent_future': ego_agent_future,
            'neighbor_agents_past': neighbor_agents_past,
            'neighbor_agents_future': neighbor_agents_future,
            'static_objects': static_objects,
            'lanes': lanes,
            'lanes_speed_limit': lanes_speed_limit,
            'lanes_has_speed_limit': lanes_has_speed_limit,
            'route_lanes': route_lanes,
            'route_lanes_speed_limit': route_lanes_speed_limit,
            'route_lanes_has_speed_limit': route_lanes_has_speed_limit,
        }
