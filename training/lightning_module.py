"""
MVP Training Module for Diffusion Planner
PyTorch Lightning module for training
"""
import os
import sys

# Add workspace to path
sys.path.insert(0, '/workspace/nuplan-visualization')
sys.path.insert(0, '/workspace')

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from training.config import TrainConfig
from training.dataset import DiffusionPlannerDataset


class DiffusionPlannerLightningModule(pl.LightningModule):
    """Lightning module for training Diffusion Planner"""
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Build the model
        self._build_model()
        
    def _build_model(self):
        """Build the Diffusion Planner model"""
        from diffusion_planner.model.diffusion_planner import Diffusion_Planner
        
        # Create model with config
        self.model = Diffusion_Planner(self.config)
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
    def forward(self, batch):
        """Forward pass"""
        # Prepare inputs for model
        inputs = self._prepare_inputs(batch)
        
        # Forward through encoder and decoder
        encoder_outputs, decoder_outputs = self.model(inputs)
        
        return decoder_outputs
    
    def _prepare_inputs(self, batch):
        """Prepare batch for model input"""
        device = self.device
        
        # Extract batch data
        ego_current_state = batch['ego_current_state'].to(device)
        ego_agent_future = batch['ego_agent_future'].to(device)
        neighbor_agents_past = batch['neighbor_agents_past'].to(device)
        neighbor_agents_future = batch['neighbor_agents_future'].to(device)
        static_objects = batch['static_objects'].to(device)
        lanes = batch['lanes'].to(device)
        lanes_speed_limit = batch['lanes_speed_limit'].to(device)
        lanes_has_speed_limit = batch['lanes_has_speed_limit'].to(device)
        route_lanes = batch['route_lanes'].to(device)
        route_lanes_speed_limit = batch['route_lanes_speed_limit'].to(device)
        route_lanes_has_speed_limit = batch['route_lanes_has_speed_limit'].to(device)
        
        B = ego_current_state.shape[0]
        
        # Prepare current states (for both ego and neighbors)
        # ego_current_state: (B, 10) -> take first 4: x, y, cos, sin
        ego_current = ego_current_state[:, :4]
        
        # neighbor_agents_past: (B, P, V, D) where P=32, V=21, D=11
        # Take current time step (last): (B, P, D)
        neighbor_current = neighbor_agents_past[:, :, -1, :4]  # (B, P, 4)
        
        # Build current states: (B, 1+P, 4) = (B, 33, 4)
        current_states = torch.cat([ego_current.unsqueeze(1), neighbor_current], dim=1)
        
        # Sample trajectories for diffusion training
        # Ground truth future: (B, 1+P, 1+future_len, 4)
        # ego_agent_future: (B, 80, 3) -> (B, 1, 80, 3)
        # neighbor_agents_future: (B, P, 80, 3)
        ego_future = ego_agent_future.unsqueeze(1)  # (B, 1, 80, 3)
        
        # Add heading (cos, sin) - use zeros as placeholder
        ego_future_with_heading = torch.cat([
            ego_future[..., :2],  # x, y
            torch.zeros_like(ego_future[..., :1]),  # cos placeholder
            torch.zeros_like(ego_future[..., :1]),  # sin placeholder
        ], dim=-1)  # (B, 1, 80, 4)
        
        neighbor_future = neighbor_agents_future  # (B, P, 80, 3)
        neighbor_future_with_heading = torch.cat([
            neighbor_future[..., :2],
            torch.zeros_like(neighbor_future[..., :1]),
            torch.zeros_like(neighbor_future[..., :1]),
        ], dim=-1)  # (B, P, 80, 4)
        
        # Combine ego and neighbor futures: (B, 1+P, 80, 4)
        future_states = torch.cat([ego_future_with_heading, neighbor_future_with_heading], dim=1)
        
        # Add current state as first frame: (B, 1+P, 1+80, 4)
        sampled_trajectories = torch.cat([
            current_states.unsqueeze(2),
            future_states
        ], dim=2)
        
        # Diffusion timestep: uniform random in [0, 1]
        diffusion_time = torch.rand(B, device=device)
        
        # Build inputs dict
        inputs = {
            'ego_current_state': ego_current_state,
            'neighbor_agents_past': neighbor_agents_past,
            'static_objects': static_objects,
            'lanes': lanes,
            'lanes_speed_limit': lanes_speed_limit,
            'lanes_has_speed_limit': lanes_has_speed_limit,
            'route_lanes': route_lanes,
            'route_lanes_speed_limit': route_lanes_speed_limit,
            'route_lanes_has_speed_limit': route_lanes_has_speed_limit,
            # Training-specific inputs
            'sampled_trajectories': sampled_trajectories,
            'diffusion_time': diffusion_time,
        }
        
        return inputs
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Forward pass
        outputs = self.forward(batch)
        
        # Get predictions - the model predicts score (velocity field)
        # For simplicity, use MSE loss on trajectory
        predicted = outputs['score']  # (B, P, 1+future_len, 4)
        
        # Get ground truth trajectories
        device = self.device
        ego_current_state = batch['ego_current_state'].to(device)
        ego_agent_future = batch['ego_agent_future'].to(device)
        neighbor_agents_future = batch['neighbor_agents_future'].to(device)
        neighbor_agents_past = batch['neighbor_agents_past'].to(device)
        
        B = ego_current_state.shape[0]
        P = self.config.predicted_neighbor_num + 1  # 1 ego + neighbors
        future_len = self.config.future_len
        
        # Build ground truth: (B, P, 1+future_len, 4)
        # Ego: (B, 1, 1, 4) current + (B, 1, future_len, 4) future
        ego_current = ego_current_state[:, :4].unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 4)
        ego_future = torch.cat([
            ego_agent_future[..., :2],
            torch.zeros_like(ego_agent_future[..., :1]),
            torch.zeros_like(ego_agent_future[..., :1]),
        ], dim=-1).unsqueeze(1)  # (B, 1, 80, 4)
        
        # Neighbor: current + future
        neighbor_current = neighbor_agents_past[:, :, -1, :4].unsqueeze(2)  # (B, P-1, 1, 4)
        neighbor_future = torch.cat([
            neighbor_agents_future[..., :2],
            torch.zeros_like(neighbor_agents_future[..., :1]),
            torch.zeros_like(neighbor_agents_future[..., :1]),
        ], dim=-1)  # (B, P-1, 80, 4)
        
        # Combine
        gt_current = torch.cat([ego_current, neighbor_current], dim=1)  # (B, P, 1, 4)
        gt_future = torch.cat([ego_future, neighbor_future], dim=1)  # (B, P, 80, 4)
        gt_trajectories = torch.cat([gt_current, gt_future], dim=2)  # (B, P, 81, 4)
        
        # Compute loss (only on valid positions, ignore padding)
        # For MVP, just compute MSE loss
        loss = self.mse_loss(predicted, gt_trajectories)
        
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
    def train_dataloader(self):
        """Create training dataloader"""
        dataset = DiffusionPlannerDataset(
            self.config.data_dir,
            self.config.data_list,
            self.config
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        return dataloader
