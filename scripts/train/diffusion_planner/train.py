#!/usr/bin/env python3
"""Train Diffusion Planner (minimal skeleton).

Example:
  python scripts/train/diffusion_planner/train.py \
    --train-roots exports_local/boston50w_prod/slice02_N12_20260326_105143/shards/shard_000 \
    --exp-name smoke_mlp \
    --steps 200 --batch-size 32

Notes:
  - Uses sharded NPZ + manifest.jsonl produced by our export pipeline.
  - Avoids the legacy `training/` directory (intentionally).
  - Writes outputs under outputs/training/<exp_name>/

Perf:
  - Writes perf metrics to outputs/training/<exp_name>/perf.json
    (step-time EMA + p50/p90, throughput, GPU stats).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure repo root is on PYTHONPATH when launched as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from src.methods.diffusion_planner.data.npz_dataset import ShardedNpzDataset
from src.methods.diffusion_planner.models.eps_mlp import EpsMLP
from src.methods.diffusion_planner.models.simple_mlp import SimpleFutureMLP
from src.methods.diffusion_planner.train.trainer import TrainConfig, train_loop, train_loop_diffusion_eps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Back-compat: old single-root flag.
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="(Deprecated) Single root (slice/shards/shard dir). Use --train-roots instead.",
    )

    p.add_argument(
        "--train-roots",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more roots. Each can be a slice dir, a shards/ dir, a shard_*/ dir, "
            "or a boston50w_prod root when used with --train-slices (or when slices are auto-discovered)."
        ),
    )
    p.add_argument(
        "--val-roots",
        type=str,
        nargs="+",
        default=None,
        help="Optional validation roots (same conventions as --train-roots).",
    )

    p.add_argument(
        "--train-slices",
        type=str,
        nargs="+",
        default=None,
        help="When a train-root points to boston50w_prod, pick these slice directory names.",
    )
    p.add_argument(
        "--val-slices",
        type=str,
        nargs="+",
        default=None,
        help="When a val-root points to boston50w_prod, pick these slice directory names.",
    )

    # exp-name is optional for convenience.
    p.add_argument("--exp-name", type=str, default=None)

    p.add_argument(
        "--mode",
        type=str,
        default="diffusion_eps",
        choices=["mlp", "diffusion_eps"],
        help="Training mode. 'mlp' keeps the original regression baseline.",
    )

    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # perf knobs (optional)
    p.add_argument("--perf-window", type=int, default=100, help="Window size for p50/p90 step time.")
    p.add_argument("--perf-smi-every", type=int, default=100, help="Query nvidia-smi every N steps.")
    p.add_argument(
        "--profiler-steps",
        type=int,
        default=0,
        help="If >0, run torch.profiler for the first N steps to estimate FLOPs (best-effort).",
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally cap number of kept samples to index (for smoke tests).",
    )

    return p.parse_args()


def _discover_slice_dirs(dataset_root: Path) -> list[Path]:
    """Heuristic: treat a root as a dataset root if it contains slice subdirs with shards/."""

    if not dataset_root.is_dir():
        return []

    slice_dirs: list[Path] = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and (p / "shards").is_dir():
            slice_dirs.append(p)
    return slice_dirs


def _expand_roots(roots: list[str] | None, slices: list[str] | None) -> list[str]:
    """Expand boston50w_prod roots + slice list into explicit slice dirs.

    If slices is None and a root looks like a boston50w_prod directory (contains
    subdirectories with shards/), we auto-expand to all slices.
    """

    if not roots:
        return []

    out: list[str] = []
    for r in roots:
        rp = Path(r).expanduser().resolve()

        if slices is None:
            auto = _discover_slice_dirs(rp)
            if auto:
                out.extend([str(p) for p in auto])
            else:
                out.append(str(rp))
            continue

        # If user provided slices, we interpret rp as a dataset root (e.g. boston50w_prod).
        for s in slices:
            out.append(str((rp / s).resolve()))

    return out


def main() -> None:
    args = parse_args()

    if args.exp_name is None:
        args.exp_name = f"{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"

    if args.train_roots is None and args.data_root is not None:
        train_roots = [args.data_root]
    else:
        train_roots = args.train_roots or []

    val_roots = args.val_roots or []

    train_roots = _expand_roots(train_roots, args.train_slices)
    val_roots = _expand_roots(val_roots, args.val_slices)

    if not train_roots:
        raise SystemExit("No training roots provided. Use --train-roots (or legacy --data-root).")

    ds = ShardedNpzDataset(train_roots, max_samples=args.max_samples)
    print(f"Dataset size: {len(ds)} kept samples | x_dim={ds.x_dim} | y_T={ds.y_T}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
        drop_last=True,
    )

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
        perf_window=args.perf_window,
        perf_smi_every=args.perf_smi_every,
        profiler_steps=args.profiler_steps,
    )

    # Pre-create exp dir so we can write data stats even if training crashes.
    exp_dir = Path(cfg.out_root) / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "data_stats.json").write_text(json.dumps(ds.get_data_stats(), indent=2))

    if args.mode == "mlp":
        model = SimpleFutureMLP(x_dim=ds.x_dim, T=ds.y_T)
        exp_dir = train_loop(cfg=cfg, model=model, train_loader=dl)
    else:
        model = EpsMLP(x_dim=ds.x_dim, T=ds.y_T)
        exp_dir = train_loop_diffusion_eps(cfg=cfg, model=model, train_loader=dl)

    print(f"Done. Outputs written to: {exp_dir}")


if __name__ == "__main__":
    main()
