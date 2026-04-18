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
from torch.utils.data import ConcatDataset, Subset


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
    p.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"])
    p.add_argument("--lr-min-ratio", type=float, default=1.0, help="For cosine: lr_min = lr * lr_min_ratio")
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument(
        "--resume-ckpt",
        type=str,
        default=None,
        help="Resume from a checkpoint_step_XXXXXX.pt produced by this trainer.",
    )
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

    p.add_argument(
        "--normalization-file",
        type=str,
        default=None,
        help=(
            "Optional normalization JSON (see scripts/data/compute_normalization_ours.py). "
            "If provided, we will load state_mean/state_std and observation_norm into the paper model, "
            "and those stats will be stored in checkpoints for eval/infer."
        ),
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

    # tensorboard (optional)
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write TensorBoard event files under outputs/training/<exp>/tb (paper_dit_dpm only).",
    )
    p.add_argument(
        "--tb-dir",
        type=str,
        default=None,
        help="TensorBoard log dir (default: outputs/training/<exp>/tb).",
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

    # Multi-city fast evaluation (equal status). Each city uses the same N and the same schedule.
    p.add_argument("--fast-eval-every", type=int, default=0, help="Run multi-city fast eval every N steps (0 disables).")
    p.add_argument("--fast-eval-n", type=int, default=1024, help="Number of fast-eval samples per city.")
    p.add_argument("--fast-eval-seed", type=int, default=0, help="Seed for deterministic fast-eval subset selection.")

    p.add_argument(
        "--fast-eval-mode",
        type=str,
        default="proxy",
        choices=["proxy", "sampler"],
        help="Fast-eval mode. proxy=t=0 one-step (cheap). sampler=run DPM sampler (matches inference; expensive).",
    )
    p.add_argument(
        "--fast-eval-diffusion-steps",
        type=int,
        default=10,
        help="When --fast-eval-mode=sampler, number of DPM sampler steps.",
    )

    p.add_argument(
        "--fast-eval-turn-source",
        type=str,
        default="tags_or_gt",
        choices=["tags_or_gt", "tags", "gt"],
        help="How to classify turning for fast-eval breakdown.",
    )

    # Fast-eval turn/straight breakdown (GT-based)
    p.add_argument(
        "--fast-eval-turn-angle-deg",
        type=float,
        default=15.0,
        help="Turning threshold in degrees based on GT heading change (default: 15deg).",
    )
    p.add_argument(
        "--fast-eval-turn-min-travel-m",
        type=float,
        default=5.0,
        help="Min GT travel distance (meters) before classifying turning (default: 5m).",
    )
    p.add_argument(
        "--fast-eval-batch-size",
        type=int,
        default=None,
        help="Batch size for fast-eval loaders (default: --fast-val-batch-size).",
    )
    p.add_argument(
        "--fast-eval-num-workers",
        type=int,
        default=None,
        help="Num workers for fast-eval loaders (default: --fast-val-num-workers).",
    )

    p.add_argument("--fast-eval-boston-roots", type=str, nargs="+", default=None)
    p.add_argument("--fast-eval-boston-slices", type=str, nargs="+", default=None)
    p.add_argument("--fast-eval-pittsburgh-roots", type=str, nargs="+", default=None)
    p.add_argument("--fast-eval-vegas-roots", type=str, nargs="+", default=None)

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally cap number of kept samples to index (for smoke tests).",
    )

    p.add_argument("--tb-enable", action="store_true", help="Enable TensorBoard logging (default: off).")
    p.add_argument("--tb-every", type=int, default=2000, help="Log TensorBoard images/scalars every N steps (0 disables image logging).")
    p.add_argument("--tb-num-samples", type=int, default=1, help="Number of random samples to visualize per TB logging step.")
    p.add_argument("--tb-denoise-k", type=int, default=10, help="Number of denoise panels to render per sample (for xt/x0_pred).")
    p.add_argument(
        "--tb-denoise-mode",
        type=str,
        default="t_sweep",
        choices=["t_sweep", "forward_noise", "sampler", "all"],
        help="TensorBoard denoise visualization mode for paper_dit_dpm.",
    )
    p.add_argument(
        "--tb-sampler-steps",
        type=int,
        default=10,
        help="Number of DPM-Solver steps for sampler TensorBoard intermediates.",
    )
    p.add_argument("--tb-image-size", type=int, default=800, help="Best-effort render size for TB images.")

    # ego-history masking augmentation (paper_dit_dpm)
    p.add_argument(
        "--mask-ego-history-prob",
        type=float,
        default=0.0,
        help="With probability p (per-sample), mask ego history in conditioning features (default: 0).",
    )
    p.add_argument(
        "--mask-ego-history-keep-last",
        type=int,
        default=1,
        choices=[0, 1],
        help="When masking, keep the last timestep (current) and only mask the past. 1=yes (default), 0=no.",
    )
    p.add_argument(
        "--tb-prefer-turn",
        type=int,
        default=1,
        choices=[0, 1],
        help="When pre-sampling TB examples, prefer turning-tagged samples if available. 1=yes (default), 0=no.",
    )

    # loss spike dump (diagnosis)
    p.add_argument("--spike-dump", action="store_true", help="Dump top-k samples when train loss spikes.")
    p.add_argument("--spike-start", type=int, default=2000, help="Start dumping spikes after this step.")
    p.add_argument("--spike-thresh", type=float, default=0.9, help="Dump when loss >= this threshold.")
    p.add_argument("--spike-topk", type=int, default=8, help="How many top samples to dump per spike batch.")

    return p.parse_args()


def _stable_int(s: str) -> int:
    import hashlib

    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _build_balanced_subset_dataset(
    *,
    roots: list[str],
    n_total: int,
    seed: int,
    cache_root: str,
) -> tuple[ConcatDataset, set[tuple[str, int]]]:
    """Create a deterministic subset dataset with (roughly) equal samples from each root.

    This is used for multi-city fast-eval so that each city is evaluated on the same number of samples,
    without one partition dominating (e.g. always taking from p0 first).
    """

    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    if not roots:
        raise ValueError("roots is empty")

    dss = []
    perms: list[list[int]] = []
    ptrs: list[int] = []
    chosen_keys: set[tuple[str, int]] = set()
    for r in roots:
        ds_i = ShardedNpzFeatureDataset([r], max_samples=None, cache_root=cache_root)
        dss.append(ds_i)
        L = len(ds_i)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + _stable_int(r))
        perm = torch.randperm(L, generator=g).tolist() if L > 0 else []
        perms.append(perm)
        ptrs.append(0)

    m = len(dss)
    base = n_total // m
    rem = n_total % m
    target = [base + (1 if i < rem else 0) for i in range(m)]

    chosen: list[list[int]] = [[] for _ in range(m)]
    # first pass: take target per root
    for i in range(m):
        take = min(target[i], len(perms[i]))
        chosen[i].extend(perms[i][:take])
        ptrs[i] = take

    # redistribute any remaining budget to roots with remaining capacity
    remaining = n_total - sum(len(x) for x in chosen)
    while remaining > 0:
        progressed = False
        for i in range(m):
            if remaining <= 0:
                break
            if ptrs[i] < len(perms[i]):
                chosen[i].append(perms[i][ptrs[i]])
                ptrs[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    # Record unique sample keys for blacklist: (shard_dir, row_idx)
    for ds_i, idxs in zip(dss, chosen):
        for idx in idxs:
            shard_idx, row_idx, _meta = ds_i._index[int(idx)]  # type: ignore[attr-defined]
            spec = ds_i.shards[int(shard_idx)]
            chosen_keys.add((str(spec.shard_dir.resolve()), int(row_idx)))

    subsets = [Subset(ds_i, idxs) for ds_i, idxs in zip(dss, chosen) if len(idxs) > 0]
    if not subsets:
        # Fall back to empty concat (should not happen in practice)
        return ConcatDataset([]), set()
    return ConcatDataset(subsets), chosen_keys


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
    fast_eval_loaders: dict[str, DataLoader] = {}
    fast_eval_exclude_keys: set[tuple[str, int]] = set()

    if args.mode == "paper_dit_dpm":
        def _feature_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
            """Custom collate for feature dataset.

            Default PyTorch collation fails on variable-length `meta.tags` (list-of-strings).
            We stack tensor fields normally, but keep meta fields as python lists.
            """

            if not batch:
                return {}

            out: dict[str, Any] = {}
            # tensor keys
            for k, v0 in batch[0].items():
                if k == "meta":
                    continue
                if torch.is_tensor(v0):
                    # Dataset returns per-sample tensors (no batch dim). Stack into [B, ...].
                    out[k] = torch.stack([b[k] for b in batch], dim=0)
                else:
                    out[k] = [b.get(k) for b in batch]

            # meta: keep as dict of lists (not recursively collated)
            metas = [b.get("meta") or {} for b in batch]
            meta_keys: set[str] = set()
            for m in metas:
                if isinstance(m, dict):
                    meta_keys |= set(m.keys())
            meta_out: dict[str, Any] = {}
            for mk in sorted(meta_keys):
                meta_out[mk] = [(m.get(mk) if isinstance(m, dict) else None) for m in metas]
            out["meta"] = meta_out
            return out

        # Multi-city fast evaluation (equal status) + strict blacklist isolation.
        # We build fast-eval subsets first, then blacklist them from the training dataset.
        #
        # NOTE: Even if fast-eval itself is disabled (fast_eval_every=0), we still
        # want stable, per-city TensorBoard visualization examples when TB image
        # logging is enabled. Without per-city loaders, TB falls back to sampling
        # from the mixed-city training batch, which can easily produce "all straight"
        # panels and mis-labeled city sections.
        need_fast_eval_loaders = (int(getattr(args, "fast_eval_every", 0) or 0) > 0) or bool(getattr(args, "tb_enable", False))

        if need_fast_eval_loaders:
            n_city = int(getattr(args, "fast_eval_n", 1024) or 1024)
            seed_city = int(getattr(args, "fast_eval_seed", 0) or 0)

            # If user did not explicitly provide per-city roots, infer them from
            # the expanded training roots by substring match.
            tr_lower = [str(r).lower() for r in train_roots]
            infer_boston = [r for r, rl in zip(train_roots, tr_lower) if "boston" in rl]
            infer_pgh = [r for r, rl in zip(train_roots, tr_lower) if "pittsburgh" in rl]
            infer_vegas = [r for r, rl in zip(train_roots, tr_lower) if "vegas" in rl]

            if getattr(args, "fast_eval_boston_roots", None) is None and infer_boston:
                args.fast_eval_boston_roots = list(infer_boston)
                args.fast_eval_boston_slices = None
            if getattr(args, "fast_eval_pittsburgh_roots", None) is None and infer_pgh:
                args.fast_eval_pittsburgh_roots = list(infer_pgh)
            if getattr(args, "fast_eval_vegas_roots", None) is None and infer_vegas:
                args.fast_eval_vegas_roots = list(infer_vegas)

            def _maybe_add_city(city: str, roots: list[str] | None, slices: list[str] | None = None) -> None:
                nonlocal fast_eval_exclude_keys
                if not roots:
                    return
                expanded = _expand_roots(list(roots), slices)
                if not expanded:
                    return
                ds_city, keys_city = _build_balanced_subset_dataset(
                    roots=expanded,
                    n_total=n_city,
                    seed=seed_city,
                    cache_root=args.cache_root,
                )
                if len(keys_city) <= 0:
                    return
                fast_eval_exclude_keys |= set(keys_city)
                fast_eval_loaders[city] = DataLoader(
                    ds_city,
                    batch_size=int(getattr(args, "fast_eval_batch_size", None) or args.fast_val_batch_size),
                    shuffle=False,
                    num_workers=int(getattr(args, "fast_eval_num_workers", None) or args.fast_val_num_workers),
                    pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
                    drop_last=False,
                    collate_fn=_feature_collate,
                )

            _maybe_add_city("boston", args.fast_eval_boston_roots, args.fast_eval_boston_slices)
            _maybe_add_city("pittsburgh", args.fast_eval_pittsburgh_roots, None)
            _maybe_add_city("vegas", args.fast_eval_vegas_roots, None)

            if fast_eval_loaders:
                cities = ",".join(sorted(fast_eval_loaders.keys()))
                print(
                    f"Fast-eval (multi-city) enabled: every={int(args.fast_eval_every)} per_city_n={n_city} seed={seed_city} "
                    f"cities=[{cities}] blacklist_train_keys={len(fast_eval_exclude_keys)}",
                    flush=True,
                )

        ds = ShardedNpzFeatureDataset(
            train_roots,
            max_samples=args.max_samples,
            exclude_keys=(fast_eval_exclude_keys or None),
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
            collate_fn=_feature_collate,
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
                    collate_fn=_feature_collate,
                )
                print(f"Fast-val dataset size: {len(fv_ds)} kept samples | every={int(args.fast_val_every)}")
            else:
                print("[fast-val] No fast-val roots expanded; fast validation disabled.")
        elif int(getattr(args, "fast_val_every", 0) or 0) > 0:
            print("[fast-val] --fast-val-every set but --fast-val-roots not provided; fast validation disabled.")

        # (fast-eval loaders already built above for blacklist isolation)
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
        lr_schedule=str(getattr(args, "lr_schedule", "constant")),
        lr_min_ratio=float(getattr(args, "lr_min_ratio", 1.0)),
        lr_warmup_steps=int(getattr(args, "lr_warmup_steps", 0) or 0),
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
        # tensorboard (scalar + image logging)
        # NOTE: tb_enable already initializes a SummaryWriter in paper_trainer (and logs scalars).
        # If we also set tensorboard=True here, paper_trainer would create a second SummaryWriter
        # pointing at the same logdir, producing an events.*.1 file. TensorBoard's directory watcher
        # often gets stuck on that .1 file and the UI shows a single point.
        tensorboard=bool(getattr(args, "tensorboard", False)),
        tb_dir=getattr(args, "tb_dir", None),
        # richer image logging (paper_dit_dpm)
        tb_enable=bool(args.tb_enable),
        tb_every=int(args.tb_every),
        tb_num_samples=int(args.tb_num_samples),
        tb_denoise_k=int(args.tb_denoise_k),
        tb_denoise_mode=str(getattr(args, "tb_denoise_mode", "t_sweep")),
        tb_sampler_steps=int(getattr(args, "tb_sampler_steps", args.tb_denoise_k)),
        tb_image_size=int(args.tb_image_size),

        tb_prefer_turn=bool(int(getattr(args, "tb_prefer_turn", 1))),

        # ego-history masking augmentation
        mask_ego_history_prob=float(getattr(args, "mask_ego_history_prob", 0.0) or 0.0),
        mask_ego_history_keep_last=bool(int(getattr(args, "mask_ego_history_keep_last", 1))),

        # spike dump
        spike_dump=bool(getattr(args, "spike_dump", False)),
        spike_start=int(getattr(args, "spike_start", 2000) or 2000),
        spike_thresh=float(getattr(args, "spike_thresh", 0.9) or 0.9),
        spike_topk=int(getattr(args, "spike_topk", 8) or 8),

        # fast-eval behavior
        fast_eval_mode=str(getattr(args, "fast_eval_mode", "proxy") or "proxy"),
        fast_eval_diffusion_steps=int(getattr(args, "fast_eval_diffusion_steps", 10) or 10),
        fast_eval_turn_source=str(getattr(args, "fast_eval_turn_source", "tags_or_gt") or "tags_or_gt"),
        fast_eval_turn_angle_deg=float(getattr(args, "fast_eval_turn_angle_deg", 15.0) or 15.0),
        fast_eval_turn_min_travel_m=float(getattr(args, "fast_eval_turn_min_travel_m", 5.0) or 5.0),

        # resume
        resume_ckpt=str(getattr(args, "resume_ckpt", None)) if getattr(args, "resume_ckpt", None) else None,
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
        # Normalization is mandatory for paper_dit_dpm.
        # Without observation normalization, training/TB sampling can drift badly and become misleading.
        norm_path = getattr(args, "normalization_file", None)
        if not norm_path:
            raise SystemExit(
                "paper_dit_dpm requires --normalization-file (precomputed stats). "
                "Example: --normalization-file outputs/cache/normalization_ours_bos150w_vegas_pgh.json"
            )
        if not Path(str(norm_path)).expanduser().is_file():
            raise SystemExit(f"--normalization-file not found: {norm_path}")

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

        # Load normalization stats computed from training cache.
        data = json.loads(Path(norm_path).read_text())
        if "ego" in data and "neighbor" in data:
            ego_mean = data["ego"]["mean"]
            ego_std = data["ego"]["std"]
            nb_mean = data["neighbor"]["mean"]
            nb_std = data["neighbor"]["std"]
            paper_cfg.state_mean = [[ego_mean]] + [[nb_mean]] * int(paper_cfg.predicted_neighbor_num)
            paper_cfg.state_std = [[ego_std]] + [[nb_std]] * int(paper_cfg.predicted_neighbor_num)
        # ObservationNormalizer consumes everything except ego/neighbor.
        obs_norm = {k: v for k, v in data.items() if k not in ("ego", "neighbor")}
        paper_cfg.observation_norm = obs_norm if obs_norm else None
        if paper_cfg.observation_norm is None:
            raise SystemExit(f"Invalid normalization JSON (empty observation_norm): {norm_path}")

        model = PaperDiffusionPlanner(paper_cfg)
        exp_dir = train_loop_paper_dit_xstart(
            cfg=cfg,
            model=model,
            train_loader=dl,
            fast_val_loader=fast_val_loader,
            fast_val_every=int(getattr(args, "fast_val_every", 0) or 0),
            fast_eval_loaders=(fast_eval_loaders or None),
            fast_eval_every=int(getattr(args, "fast_eval_every", 0) or 0),
        )
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    print(f"Done. Outputs written to: {exp_dir}")


if __name__ == "__main__":
    main()
