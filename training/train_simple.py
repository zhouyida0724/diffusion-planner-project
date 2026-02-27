#!/usr/bin/env python3
"""
MVP Training Script for Diffusion Planner
Simple PyTorch training with TensorBoard
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Add workspace to path - nuplan-visualization contains the source code
sys.path.insert(0, '/workspace/nuplan-visualization')
sys.path.insert(0, '/workspace')

from training.config import TrainConfig
from training.dataset import DiffusionPlannerDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Diffusion Planner')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                        default='/workspace/data/test_preprocess_output',
                        help='Path to data directory')
    parser.add_argument('--data_list', type=str,
                        default='training_data_list.json',
                        help='Path to data list JSON file')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=2,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=192,
                        help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of attention heads')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str,
                        default='/workspace/training_outputs',
                        help='Output directory')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


class SimpleDiffusionPlannerTrainer:
    """Simple trainer for Diffusion Planner"""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Build model
        self._build_model()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs
        )
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'tb_logs'))
        
    def _build_model(self):
        """Build the Diffusion Planner model"""
        from diffusion_planner.model.diffusion_planner import Diffusion_Planner
        
        # Create model with config
        self.model = Diffusion_Planner(self.config)
        self.model.to(self.device)
        self.model.train()
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
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
        ego_current = ego_current_state[:, :4]
        
        # neighbor_agents_past: (B, P, V, D) where P=32, V=21, D=11
        neighbor_current = neighbor_agents_past[:, :, -1, :4]  # (B, P, 4)
        
        # Build current states: (B, 1+P, 4)
        current_states = torch.cat([ego_current.unsqueeze(1), neighbor_current], dim=1)
        
        # Sample trajectories for diffusion training
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
    
    def train_step(self, batch):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        inputs = self._prepare_inputs(batch)
        
        try:
            encoder_outputs, decoder_outputs = self.model(inputs)
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Return a dummy loss for testing
            return torch.tensor(0.0, requires_grad=True)
        
        # Get predictions
        predicted = decoder_outputs['score']  # (B, P, 1+future_len, 4)
        
        # Get ground truth trajectories
        ego_current_state = batch['ego_current_state'].to(self.device)
        ego_agent_future = batch['ego_agent_future'].to(self.device)
        neighbor_agents_future = batch['neighbor_agents_future'].to(self.device)
        neighbor_agents_past = batch['neighbor_agents_past'].to(self.device)
        
        B = ego_current_state.shape[0]
        P = self.config.predicted_neighbor_num + 1  # 1 ego + neighbors
        
        # Build ground truth: (B, P, 1+future_len, 4)
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
        
        # Compute loss
        loss = self.mse_loss(predicted, gt_trajectories)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss.item()
            num_batches += 1
            
            print(f"  Batch {num_batches}: loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, dataloader, num_epochs):
        """Full training loop"""
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            self.model.train()
            avg_loss = self.train_epoch(dataloader, epoch)
            
            # Log to TensorBoard
            self.writer.add_scalar('train_loss', avg_loss, epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
            
            # Step scheduler
            self.scheduler.step()
            
            # Save checkpoint with "module." prefix for compatibility with inference code
            # The inference code expects keys with "module." prefix
            state_dict = self.model.state_dict()
            state_dict_with_module = {f"module.{k}": v for k, v in state_dict.items()}
            
            checkpoint_path = os.path.join(
                self.config.output_dir, 
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': state_dict_with_module,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.model


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Diffusion Planner - MVP Training")
    print("=" * 60)
    
    # Create config
    config = TrainConfig(
        data_dir=args.data_dir,
        data_list=args.data_list,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        output_dir=args.output_dir,
        device=args.device
    )
    
    print(f"Config:")
    print(f"  - Data directory: {config.data_dir}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Max epochs: {config.max_epochs}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Device: {config.device}")
    print(f"  - Hidden dim: {config.hidden_dim}")
    print(f"  - Num heads: {config.num_heads}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create dataset
    dataset = DiffusionPlannerDataset(
        config.data_dir,
        config.data_list,
        config
    )
    
    print(f"\nDataset: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    # Create trainer and train
    trainer = SimpleDiffusionPlannerTrainer(config)
    model = trainer.train(dataloader, config.max_epochs)
    
    print(f"\nTensorBoard logs: {os.path.join(config.output_dir, 'tb_logs')}")
    print("To view TensorBoard, run: tensorboard --logdir={}".format(
        os.path.join(config.output_dir, 'tb_logs')
    ))


if __name__ == '__main__':
    main()
