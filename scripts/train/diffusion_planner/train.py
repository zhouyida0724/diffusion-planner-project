#!/usr/bin/env python3
"""Train Diffusion Planner (minimal skeleton).

Example:
  python scripts/train/diffusion_planner/train.py \
    --data-root exports_local/boston50w_prod/slice02_N12_20260326_105143/shards/shard_000 \
    --exp-name smoke_mlp \
    --steps 200 --batch-size 32

Notes:
  - Uses sharded NPZ + manifest.jsonl produced by our export pipeline.
  - Avoids the legacy `training/` directory (intentionally).
  - Writes outputs under outputs/training/<exp_name>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH when launched as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from src.methods.diffusion_planner.data.npz_dataset import ShardedNpzDataset
from src.methods.diffusion_planner.models.simple_mlp import SimpleFutureMLP
from src.methods.diffusion_planner.train.trainer import TrainConfig, train_loop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True,
                   help="Slice dir, shards dir, or shard dir containing data.npz + manifest.jsonl")
    p.add_argument("--exp-name", type=str, required=True)

    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    p.add_argument("--max-samples", type=int, default=None,
                   help="Optionally cap number of kept samples to index (for smoke tests).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    ds = ShardedNpzDataset(args.data_root, max_samples=args.max_samples)
    print(f"Dataset size: {len(ds)} kept samples | x_dim={ds.x_dim} | y_T={ds.y_T}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
        drop_last=True,
    )

    model = SimpleFutureMLP(x_dim=ds.x_dim, T=ds.y_T)

    cfg = TrainConfig(
        exp_name=args.exp_name,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        seed=args.seed,
        device=args.device,
    )

    exp_dir = train_loop(cfg=cfg, model=model, train_loader=dl)
    print(f"Done. Outputs written to: {exp_dir}")


if __name__ == "__main__":
    main()
