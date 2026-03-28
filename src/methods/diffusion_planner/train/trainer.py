"""Minimal PyTorch trainer used for diffusion planner smoke tests.

Design goals:
  - deterministic-ish (seed)
  - explicit shape + NaN/Inf assertions
  - lightweight checkpointing
  - writes to outputs/training/<exp_name>/

This is intentionally not PyTorch Lightning to keep the control surface small.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from src.methods.diffusion_planner.diffusion.losses import masked_mse
from src.methods.diffusion_planner.diffusion.scheduler import NoiseSchedule, q_sample

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    exp_name: str
    out_root: str = "outputs/training"
    steps: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 2
    log_every: int = 10
    ckpt_every: int = 200
    seed: int = 0
    device: str = "cuda"  # 'cuda' or 'cpu'


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _assert_finite(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        bad = t[~torch.isfinite(t)]
        raise FloatingPointError(f"{name} has NaN/Inf; example bad values: {bad[:5]}")


def train_loop(
    *,
    cfg: TrainConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    max_grad_norm: Optional[float] = 1.0,
) -> Path:
    """Run training and return the experiment directory."""

    seed_everything(cfg.seed)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    exp_dir = Path(cfg.out_root) / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    (exp_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    t0 = time.time()
    step = 0

    loader_it = iter(train_loader)

    while step < cfg.steps:
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(train_loader)
            batch = next(loader_it)

        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)

        assert x.ndim == 2, f"x ndim {x.ndim}"
        assert y.ndim == 3 and y.shape[-1] == 3, f"y shape {tuple(y.shape)}"

        _assert_finite(x, "x")
        _assert_finite(y, "y")

        y_hat = model(x)
        assert y_hat.shape == y.shape, f"y_hat {tuple(y_hat.shape)} vs y {tuple(y.shape)}"
        _assert_finite(y_hat, "y_hat")

        loss = torch.mean((y_hat - y) ** 2)
        _assert_finite(loss, "loss")

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        opt.step()

        if (step % cfg.log_every) == 0 or step == cfg.steps - 1:
            dt = time.time() - t0
            msg = f"step {step:05d} | loss {loss.item():.6f} | {dt:.1f}s"
            print(msg, flush=True)
            with (exp_dir / "train.log").open("a") as f:
                f.write(msg + "\n")

        if (step % cfg.ckpt_every) == 0 or step == cfg.steps - 1:
            ckpt_path = exp_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "cfg": asdict(cfg),
                },
                ckpt_path,
            )

        step += 1

    # Convenience symlink/copy: latest
    latest = exp_dir / "checkpoint_latest.pt"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(os.path.basename(ckpt_path))
    except Exception:
        # fallback: copy
        import shutil

        shutil.copy2(ckpt_path, latest)

    return exp_dir


def train_loop_diffusion_eps(
    *,
    cfg: TrainConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    schedule_cfg: NoiseSchedule = NoiseSchedule(),
    max_grad_norm: Optional[float] = 1.0,
) -> Path:
    """Train an eps-pred diffusion model.

    Expected model signature:
        eps_hat = model(x, y_t, t)

    Where:
        x: [B, x_dim]
        y_t: [B, T, 3]
        t: [B]

    Loss:
        MSE(eps_hat, eps) (optionally masked)

    This is a minimal training loop meant for smoke tests.
    """

    seed_everything(cfg.seed)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    exp_dir = Path(cfg.out_root) / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    schedule = schedule_cfg.make(device=device)

    model.train()
    t0 = time.time()
    step = 0
    loader_it = iter(train_loader)

    while step < cfg.steps:
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(train_loader)
            batch = next(loader_it)

        x = batch["x"].to(device=device, dtype=torch.float32)
        y0 = batch["y"].to(device=device, dtype=torch.float32)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device=device)

        assert x.ndim == 2
        assert y0.ndim == 3 and y0.shape[-1] == 3

        _assert_finite(x, "x")
        _assert_finite(y0, "y0")

        B = x.shape[0]
        t = torch.randint(low=0, high=schedule.num_steps, size=(B,), device=device, dtype=torch.int64)
        eps = torch.randn_like(y0)
        y_t = q_sample(schedule=schedule, x0=y0, t=t, noise=eps)

        eps_hat = model(x, y_t, t)
        assert eps_hat.shape == eps.shape
        _assert_finite(eps_hat, "eps_hat")

        loss = masked_mse(eps_hat, eps, mask=mask)
        _assert_finite(loss, "loss")

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        opt.step()

        if (step % cfg.log_every) == 0 or step == cfg.steps - 1:
            dt = time.time() - t0
            msg = f"step {step:05d} | loss {loss.item():.6f} | {dt:.1f}s"
            print(msg, flush=True)
            with (exp_dir / "train.log").open("a") as f:
                f.write(msg + "\n")

        if (step % cfg.ckpt_every) == 0 or step == cfg.steps - 1:
            ckpt_path = exp_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "cfg": asdict(cfg),
                },
                ckpt_path,
            )

        step += 1

    latest = exp_dir / "checkpoint_latest.pt"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(os.path.basename(ckpt_path))
    except Exception:
        import shutil

        shutil.copy2(ckpt_path, latest)

    return exp_dir
