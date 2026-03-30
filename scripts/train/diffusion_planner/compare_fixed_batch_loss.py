#!/usr/bin/env python3
"""Compare fixed-batch training loss between two checkpoints.

This matches `train_loop_paper_dit_xstart` loss:
  loss = mean((pred - x0n)**2)
where pred is the decoder score for noised x_t and x0n is normalized x0.

We fix:
- DataLoader shuffle order via torch.Generator(manual_seed)
- t and noise sampling (same tensors reused for both checkpoints)

Example:
  python3 scripts/train/diffusion_planner/compare_fixed_batch_loss.py \
    --train-roots exports_local/boston50w_prod \
    --cache-root outputs/cache/training_arrays \
    --ckpt-a outputs/training/EXP/checkpoint_step_000000.pt \
    --ckpt-b outputs/training/EXP/checkpoint_step_009999.pt \
    --device cuda
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

# Ensure repo root is on PYTHONPATH when launched as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torch.utils.data import DataLoader

from scripts.train.diffusion_planner.train import _expand_roots
from src.methods.diffusion_planner.data.feature_npz_dataset import ShardedNpzFeatureDataset
from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner, PaperModelConfig
from src.methods.diffusion_planner.paper.train.paper_trainer import _build_joint_trajectories_x0


def _paper_cfg_from_ckpt(d: dict) -> PaperModelConfig:
    raw = dict(d.get("paper_config") or {})
    allowed = {f.name for f in fields(PaperModelConfig)}
    clean = {k: raw[k] for k in list(raw.keys()) if k in allowed}
    return PaperModelConfig(**clean)


@torch.no_grad()
def _loss_on_fixed_batch(
    *,
    ckpt_path: str,
    batch: dict,
    paper_cfg: PaperModelConfig,
    x0n: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
) -> float:
    payload = torch.load(ckpt_path, map_location="cpu")
    model = PaperDiffusionPlanner(paper_cfg)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing={missing} unexpected={unexpected}")

    model.to(device)
    model.train(True)

    mean, std = model.sde.marginal_prob(x0n, t)
    xt = mean + std * noise
    xt[:, :, 0, :] = x0n[:, :, 0, :]

    inputs = {
        "ego_current_state": batch["ego_current_state"],
        "neighbor_agents_past": batch["neighbor_agents_past"],
        "static_objects": batch["static_objects"],
        "lanes": batch["lanes"],
        "lanes_speed_limit": batch["lanes_speed_limit"],
        "lanes_has_speed_limit": batch["lanes_has_speed_limit"],
        "route_lanes": batch["route_lanes"],
        "route_lanes_speed_limit": batch.get("route_lanes_speed_limit"),
        "route_lanes_has_speed_limit": batch.get("route_lanes_has_speed_limit"),
        "sampled_trajectories": xt,
        "diffusion_time": t,
    }

    _, dec_out = model(inputs)
    pred = dec_out["score"]
    loss = torch.mean((pred - x0n) ** 2)
    return float(loss.item())


def _state_l2_delta(ckpt_a: str, ckpt_b: str) -> float:
    a = torch.load(ckpt_a, map_location="cpu")["model_state"]
    b = torch.load(ckpt_b, map_location="cpu")["model_state"]
    keys = sorted(set(a.keys()) & set(b.keys()))
    ss = 0.0
    for k in keys:
        da = a[k].float()
        db = b[k].float()
        ss += float(torch.sum((db - da) ** 2).item())
    return ss ** 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-roots", type=str, nargs="+", required=True)
    p.add_argument("--train-slices", type=str, nargs="+", default=None)
    p.add_argument("--cache-root", type=str, default="outputs/cache/training_arrays")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ckpt-a", type=str, required=True)
    p.add_argument("--ckpt-b", type=str, required=True)
    args = p.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    train_roots = _expand_roots(args.train_roots, args.train_slices)
    ds = ShardedNpzFeatureDataset(train_roots, cache_root=args.cache_root)

    # Fix shuffle order deterministically.
    g = torch.Generator()
    g.manual_seed(int(args.seed))

    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=g,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda" and torch.cuda.is_available()),
        drop_last=True,
    )

    it = iter(dl)
    batch = next(it)
    batch = {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # Use ckpt-a to recover the exact paper config, then build x0n/t/noise once.
    payload_a = torch.load(args.ckpt_a, map_location="cpu")
    paper_cfg = _paper_cfg_from_ckpt(payload_a)

    # Build x0n once (independent of weights).
    x0_4, _ = _build_joint_trajectories_x0(
        batch,
        predicted_neighbor_num=int(paper_cfg.predicted_neighbor_num),
        future_len=int(paper_cfg.future_len),
    )
    # state_normalizer is deterministic given config.
    x0n = paper_cfg.build_state_normalizer()(x0_4)

    # Fix diffusion noise schedule randomness.
    torch.manual_seed(int(args.seed))
    B = x0n.shape[0]
    t = torch.rand((B,), device=device, dtype=torch.float32)
    noise = torch.randn_like(x0n)

    loss_a = _loss_on_fixed_batch(
        ckpt_path=args.ckpt_a,
        batch=batch,
        paper_cfg=paper_cfg,
        x0n=x0n,
        t=t,
        noise=noise,
        device=device,
    )

    loss_b = _loss_on_fixed_batch(
        ckpt_path=args.ckpt_b,
        batch=batch,
        paper_cfg=paper_cfg,
        x0n=x0n,
        t=t,
        noise=noise,
        device=device,
    )

    rel = (loss_a - loss_b) / max(loss_a, 1e-12)
    print(f"loss_a({Path(args.ckpt_a).name}) = {loss_a:.8f}")
    print(f"loss_b({Path(args.ckpt_b).name}) = {loss_b:.8f}")
    print(f"relative_improvement = {rel*100:.3f}% (positive means loss decreased)")

    if not (loss_b < loss_a):
        dn = _state_l2_delta(args.ckpt_a, args.ckpt_b)
        print(f"param_delta_l2 = {dn:.6f}")


if __name__ == "__main__":
    main()
