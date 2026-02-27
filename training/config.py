"""
MVP Training Config for Diffusion Planner
Simplified configuration for minimal training setup
"""
import torch
import json
import os

class TrainConfig:
    """Simple config class for training"""
    
    # Model architecture parameters
    hidden_dim = 192
    num_heads = 6
    encoder_depth = 3
    decoder_depth = 3
    encoder_drop_path_rate = 0.3
    decoder_drop_path_rate = 0.3
    
    # Data parameters
    agent_num = 33  # 1 ego + 32 neighbors
    static_objects_num = 5
    lane_num = 70
    route_num = 25
    
    # Time parameters
    time_len = 21  # past trajectory length
    future_len = 80  # future trajectory length
    lane_len = 20
    
    # Prediction parameters
    past_neighbor_num = 32
    predicted_neighbor_num = 32
    
    # State dimensions
    static_objects_state_dim = 10
    
    # Diffusion model type
    diffusion_model_type = "x_start"  # or "score"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training parameters
    batch_size = 2
    learning_rate = 1e-4
    max_epochs = 10
    
    # Data paths
    data_dir = "/workspace/data/test_preprocess_output"
    data_list = "/workspace/data/test_preprocess_output/training_data_list.json"
    
    # Output
    output_dir = "/workspace/training_outputs"
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # Load normalization from JSON
        norm_path = "/workspace/diffusion_planner/normalization.json"
        if os.path.exists(norm_path):
            with open(norm_path, 'r') as f:
                self.normalization = json.load(f)
        else:
            # Default normalization values
            self.normalization = {
                "ego": {"mean": [10, 0, 0, 0], "std": [20, 20, 1, 1]},
                "neighbor": {"mean": [10, 0, 0, 0], "std": [20, 20, 1, 1]},
                "ego_current_state": {"mean": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0], "std": [20, 20, 1, 1, 20, 20, 20, 20, 1, 1]},
                "neighbor_agents_past": {"mean": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "std": [20, 20, 1, 1, 20, 20, 20, 20, 1, 1, 1]},
                "static_objects": {"mean": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0], "std": [20, 20, 1, 1, 20, 20, 1, 1, 1, 1]},
                "lanes": {"mean": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "std": [20, 20, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1]},
                "lanes_speed_limit": {"mean": [0], "std": [20]},
                "route_lanes": {"mean": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "std": [20, 20, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1]},
                "route_lanes_speed_limit": {"mean": [0], "std": [20]},
            }
        
        # Set guidance function to None for training
        self.guidance_fn = None
        
        # Import path
        self.model_import_path = "nuplan.diffusion_planner.model.diffusion_planner"
        
        # Create normalizers - required by the model
        self._create_normalizers()
    
    def _create_normalizers(self):
        """Create state and observation normalizers"""
        from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
        
        # Extract mean/std from normalization dict
        ego_mean = self.normalization.get("ego", {}).get("mean", [10, 0, 0, 0])
        ego_std = self.normalization.get("ego", {}).get("std", [20, 20, 1, 1])
        neighbor_mean = self.normalization.get("neighbor", {}).get("mean", [10, 0, 0, 0])
        neighbor_std = self.normalization.get("neighbor", {}).get("std", [20, 20, 1, 1])
        
        # Create state normalizer - ego + neighbors
        mean = [[ego_mean]] + [[neighbor_mean]] * (self.predicted_neighbor_num - 1)
        std = [[ego_std]] + [[neighbor_std]] * (self.predicted_neighbor_num - 1)
        
        self.state_normalizer = StateNormalizer(mean, std)
        
        # Create observation normalizer (simplified - just use state normalizer for now)
        self.observation_normalizer = self.state_normalizer
