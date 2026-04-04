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
from src.methods.diffusion_planner.data.feature_npz_dataset import ShardedNpzFeatureDataset
from src.methods.diffusion_planner.models.eps_mlp import EpsMLP
from src.methods.diffusion_planner.models.simple_mlp import SimpleFutureMLP
from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner, PaperModelConfig
from src.methods.diffusion_planner.paper.train import train_loop_paper_dit_xstart
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
        choices=["mlp", "diffusion_eps", "paper_dit_dpm"],
        help="Training mode. 'mlp' keeps the original regression baseline.",
    )

    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)

    # paper_dit_dpm loss weights
    p.add_argument(
        "--alpha-planning-loss",
        type=float,
        default=1.0,
        help="Weight on neighbor prediction loss for paper_dit_dpm (total = ego + alpha * neighbor).",
    )
    p.add_argument(
        "--amp",
        type=str,
        default="off",
        choices=["off", "bf16", "fp16"],
        help="Mixed precision mode. bf16 preferred; fp16 uses GradScaler. Default: off.",
    )
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
        "--cache-root",
        type=str,
        default="outputs/cache/training_arrays",
        help="Training cache root (mmap-friendly arrays). Used by paper_dit_dpm dataset.",
    )

    # step-level breakdown profiling (wall-clock timers)
    p.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="If >0, collect a per-step time breakdown for the first N steps and write it into perf.json.",
    )
    p.add_argument(
        "--profile-every",
        type=int,
        default=0,
        help="If >0, print the breakdown every N steps (only during profiled steps).",
    )

    # fast validation (paper_dit_dpm only)
    p.add_argument(
        "--fast-val-roots",
        type=str,
        nargs="+",
        default=None,
        help="Optional roots for fast validation (same conventions as --train-roots).",
    )
    p.add_argument(
        "--fast-val-slices",
        type=str,
        nargs="+",
        default=["slice05"],
        help="When a fast-val-root points to boston50w_prod, pick these slice directory names. Default: slice05",
    )
    p.add_argument("--fast-val-every", type=int, default=500, help="Run fast validation every N steps (0 disables).")
    p.add_argument(
        "--fast-val-max-samples",
        type=int,
        default=2048,
        help="Max number of samples used for fast validation (deterministic prefix).",
    )
    p.add_argument("--fast-val-batch-size", type=int, default=64)
    p.add_argument("--fast-val-num-workers", type=int, default=2)

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally cap number of kept samples to index (for smoke tests).",
    )

    p.add_argument("--tb-enable", action="store_true", help="Enable TensorBoard logging (default: off).")
    p.add_argument("--tb-every", type=int, default=0, help="Log TensorBoard images/scalars every N steps (0 disables image logging).")
    p.add_argument("--tb-num-samples", type=int, default=1, help="Number of random samples to visualize per TB logging step.")
    p.add_argument("--tb-denoise-k", type=int, default=10, help="Number of denoise panels to render per sample (for xt/x0_pred).")
    p.add_argument("--tb-image-size", type=int, default=800, help="Best-effort render size for TB images.")

    return p.parse_args()


def _discover_slice_dirs(dataset_root: Path) -> list[Path]:
    """Heuristic: treat a root as a dataset root if it contains slice subdirs with shards/.

    We only include slices that appear to contain at least one valid shard.
    """

    if not dataset_root.is_dir():
        return []

    slice_dirs: list[Path] = []
    for p in sorted(dataset_root.iterdir()):
        if not (p.is_dir() and (p / "shards").is_dir()):
            continue

        shards_root = p / "shards"
        has_any = False
        for sd in shards_root.glob("shard_*"):
            if (sd / "data.npz").is_file() and (sd / "manifest.jsonl").is_file():
                has_any = True
                break

        if has_any:
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

    fast_val_loader = None

    if args.mode == "paper_dit_dpm":
        ds = ShardedNpzFeatureDataset(
            train_roots,
            max_samples=args.max_samples,
            cache_root=args.cache_root,
        )
        print(f"Dataset size: {len(ds)} kept samples | mode=paper_dit_dpm")
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
            drop_last=True,
        )

        # Optional fast validation loader (deterministic prefix subset).
        if int(getattr(args, "fast_val_every", 0) or 0) > 0 and (args.fast_val_roots is not None):
            fv_roots = _expand_roots(args.fast_val_roots or [], args.fast_val_slices)
            if fv_roots:
                fv_ds = ShardedNpzFeatureDataset(
                    fv_roots,
                    max_samples=int(args.fast_val_max_samples),
                    cache_root=args.cache_root,
                )
                fast_val_loader = DataLoader(
                    fv_ds,
                    batch_size=int(args.fast_val_batch_size),
                    shuffle=False,
                    num_workers=int(args.fast_val_num_workers),
                    pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
                    drop_last=False,
                )
                print(f"Fast-val dataset size: {len(fv_ds)} kept samples | every={int(args.fast_val_every)}")
            else:
                print("[fast-val] No fast-val roots expanded; fast validation disabled.")
        elif int(getattr(args, "fast_val_every", 0) or 0) > 0:
            print("[fast-val] --fast-val-every set but --fast-val-roots not provided; fast validation disabled.")
    else:
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
        profile_steps=args.profile_steps,
        profile_every=args.profile_every,
        amp=args.amp,
        alpha_planning_loss=args.alpha_planning_loss,
        tb_enable=bool(args.tb_enable),
        tb_every=int(args.tb_every),
        tb_num_samples=int(args.tb_num_samples),
        tb_denoise_k=int(args.tb_denoise_k),
        tb_image_size=int(args.tb_image_size),
    )

    # Pre-create exp dir so we can write data stats even if training crashes.
    exp_dir = Path(cfg.out_root) / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "data_stats.json").write_text(json.dumps(ds.get_data_stats(), indent=2))

    if args.mode == "mlp":
        model = SimpleFutureMLP(x_dim=ds.x_dim, T=ds.y_T)
        exp_dir = train_loop(cfg=cfg, model=model, train_loader=dl)
    elif args.mode == "diffusion_eps":
        model = EpsMLP(x_dim=ds.x_dim, T=ds.y_T)
        exp_dir = train_loop_diffusion_eps(cfg=cfg, model=model, train_loader=dl)
    elif args.mode == "paper_dit_dpm":
        # Infer feature shapes from dataset to avoid hard-coded mismatches.
        s0 = ds[0]
        nb = s0["neighbor_agents_past"]
        st = s0["static_objects"]
        ln = s0["lanes"]
        rt = s0["route_lanes"]
        ego_f = s0["ego_agent_future"]

        future_len = int(ego_f.shape[0])
        if future_len == 81:
            future_len = 80

        paper_cfg = PaperModelConfig(
            device=args.device,
            agent_num=int(nb.shape[0]),
            predicted_neighbor_num=int(nb.shape[0] - 1) if int(nb.shape[0]) > 32 else int(nb.shape[0]),
            time_len=int(nb.shape[1]),
            static_objects_num=int(st.shape[0]),
            lane_num=int(ln.shape[0]),
            route_num=int(rt.shape[0]),
            lane_len=int(ln.shape[1]),
            future_len=future_len,
        )
        model = PaperDiffusionPlanner(paper_cfg)
        exp_dir = train_loop_paper_dit_xstart(
            cfg=cfg,
            model=model,
            train_loader=dl,
            fast_val_loader=fast_val_loader,
            fast_val_every=int(getattr(args, "fast_val_every", 0) or 0),
        )
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    print(f"Done. Outputs written to: {exp_dir}")


if __name__ == "__main__":
    main()
