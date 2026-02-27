#!/usr/bin/env python3
"""
MVP Training Script for Diffusion Planner
Minimal training setup with PyTorch Lightning and TensorBoard
"""
import os
import sys
import argparse
import torch

# Add workspace to path
sys.path.insert(0, '/workspace/nuplan-visualization')
sys.path.insert(0, '/workspace')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TensorBoardLogger

from training.config import TrainConfig
from training.lightning_module import DiffusionPlannerLightningModule


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
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


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
        device=args.device if torch.cuda.is_available() else 'cpu'
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
    os.makedirs(os.path.join(config.output_dir, 'tb_logs'), exist_ok=True)
    
    # Create Lightning module
    model = DiffusionPlannerLightningModule(config)
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename='diffusion_planner-{epoch:02d}-{train_loss:.4f}',
        save_top_k=1,
        monitor='train_loss',
        mode='min'
    )
    
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(config.output_dir, 'tb_logs'),
        name='diffusion_planner'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu' if config.device == 'cuda' else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, tb_logger],
        logger=tb_logger,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        log_every_n_steps=1
    )
    
    print("\nStarting training...")
    print("=" * 60)
    
    # Train
    trainer.fit(model)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {config.output_dir}")
    print(f"TensorBoard logs: {os.path.join(config.output_dir, 'tb_logs')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
