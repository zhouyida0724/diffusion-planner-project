#!/usr/bin/env python3
"""Post-hoc fast-eval for paper diffusion planner checkpoints, sliced by maneuver tags.

For an experiment directory containing checkpoint_step_*.pt, this script:
  1) picks N checkpoints evenly spaced from earliest to latest
  2) builds 3 deterministic sample subsets (left/right/straight) from manifest tags
  3) evaluates ADE/FDE at 1/3/5/8s for each subset, in either proxy or sampler mode
  4) appends one JSON line per checkpoint to an output .jsonl

This script also supports evaluating a training run that spans multiple resume
directories by passing multiple --exp-dirs. All checkpoints are merged and
selected evenly across the full step range.

Group assignment (priority): left > right > straight. If a sample matches multiple tags,
we assign it to the first matching group by that priority.

Straight tags:
  - starting_straight_traffic_light_intersection_traversal
  - starting_straight_stop_sign_intersection_traversal

NOTE:
  In practice, strict straight tags can be sparse. If you want 4096 samples for
  "straight", use --straight-mode=non_turn (default) which defines straight as
  samples that are not left/right.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import math

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore


def _repo_root() -> Path:
    # scripts/eval/fast_eval_maneuver_ckpts.py -> repo root
    return Path(__file__).resolve().parents[2]


# Ensure repo is on sys.path when invoked as a script.
_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Repo imports (after sys.path fix)
from src.methods.diffusion_planner.data.feature_npz_dataset import ShardedNpzFeatureDataset
from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner, PaperModelConfig
from src.methods.diffusion_planner.paper.train.paper_trainer import _build_joint_trajectories_x0, _compute_neighbor_masks


_LEFT_TAG = "starting_left_turn"
_RIGHT_TAG = "starting_right_turn"
_STRAIGHT_TAGS = {
    "starting_straight_traffic_light_intersection_traversal",
    "starting_straight_stop_sign_intersection_traversal",
}


def _normalize_city(raw: str) -> str:
    """Normalize manifest 'location' to canonical city labels.

    We use these canonical names in outputs/plots:
      - boston
      - pittsburgh
      - las_vegas
    """

    s = str(raw or "").lower()
    if "boston" in s:
        return "boston"
    if "pittsburgh" in s:
        return "pittsburgh"
    if "vegas" in s or "las_vegas" in s or "las-vegas" in s:
        return "las_vegas"
    return str(raw or "unknown")


def _resolve_ckpt_path(p: Path) -> Path:
    """Resolve symlinks and rewrite '/workspace/...' absolute targets to local repo."""

    p = Path(p)
    try:
        if p.is_symlink():
            target = Path(os.readlink(str(p)))
            if not target.is_absolute():
                target = (p.parent / target).resolve()
            else:
                target = target
        else:
            target = p

        # fully resolve if possible
        try:
            target = target.expanduser().resolve()
        except Exception:
            target = target.expanduser()

        parts = target.parts
        if len(parts) >= 2 and parts[0] == os.sep and parts[1] == "workspace":
            rel = Path(*parts[2:])
            return _REPO_ROOT / rel
        return target
    except Exception:
        return p


def _torch_load_compat(path: Path, *, map_location: torch.device) -> Any:
    """torch.load compat for torch>=2.6 weights_only default change."""

    path = Path(path)
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _select_evenly_spaced(items: list[Path], k: int) -> list[Path]:
    if not items:
        return []
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)

    if k == 1:
        return [items[0]]

    # pick indices evenly spaced including endpoints
    idxs = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    out = []
    seen = set()
    for ii in idxs:
        ii = int(max(0, min(len(items) - 1, ii)))
        if ii in seen:
            continue
        seen.add(ii)
        out.append(items[ii])
    # if rounding caused duplicates, fill with remaining in order
    if len(out) < k:
        for j, it in enumerate(items):
            if j not in seen:
                out.append(it)
                if len(out) >= k:
                    break
    return out[:k]


def _group_from_tags(tags: list[str] | Any) -> str | None:
    if not isinstance(tags, list):
        return None
    # priority: left > right > straight
    for t in tags:
        if not isinstance(t, str):
            continue
        if _LEFT_TAG in t:
            return "left"
    for t in tags:
        if not isinstance(t, str):
            continue
        if _RIGHT_TAG in t:
            return "right"
    for t in tags:
        if not isinstance(t, str):
            continue
        if any(st in t for st in _STRAIGHT_TAGS):
            return "straight"
    return None


def _step_from_ckpt_name(p: Path) -> int:
    s = p.stem
    # checkpoint_step_000123
    try:
        return int(s.split("checkpoint_step_")[-1])
    except Exception:
        return 0


def _collect_ckpts(exp_dirs: list[Path]) -> list[Path]:
    """Collect checkpoint_step_*.pt from one or more exp dirs.

    De-dupes by step (keeps the first seen for a given step).
    """

    by_step: dict[int, Path] = {}
    for d in exp_dirs:
        for p in sorted(d.glob("checkpoint_step_*.pt")):
            step = _step_from_ckpt_name(p)
            if step not in by_step:
                by_step[step] = p
    return [by_step[k] for k in sorted(by_step.keys())]


def _group_from_tags_ext(tags: list[str] | Any, *, straight_mode: str) -> str | None:
    """Return group in {left,right,straight}.

    straight_mode:
      - tag: require an explicit straight tag
      - non_turn: straight = not left/right
    """

    if not isinstance(tags, list):
        return None

    # left/right always explicit
    for t in tags:
        if isinstance(t, str) and _LEFT_TAG in t:
            return "left"
    for t in tags:
        if isinstance(t, str) and _RIGHT_TAG in t:
            return "right"

    if straight_mode == "non_turn":
        return "straight"

    # strict straight tag
    for t in tags:
        if isinstance(t, str) and any(st in t for st in _STRAIGHT_TAGS):
            return "straight"
    return None


def _alloc_per_city(total: int, cities: list[str]) -> dict[str, int]:
    if not cities:
        return {}
    base = total // len(cities)
    rem = total - base * len(cities)
    out = {c: base for c in cities}
    # deterministic remainder allocation by city name order
    for c in sorted(cities)[:rem]:
        out[c] += 1
    return out


def _build_group_city_subsets(
    ds: ShardedNpzFeatureDataset,
    *,
    samples_per_group: int,
    seed: int,
    straight_mode: str,
    cities: list[str] | None = None,
) -> dict[str, dict[str, list[int]]]:
    """Return indices per group per city.

    Uses manifest meta in ds._index for tags+location, so it is cheap.
    """

    # group -> city -> idxs
    groups: dict[str, dict[str, list[int]]] = {"left": {}, "right": {}, "straight": {}}

    # first pass: gather
    for ds_idx, (_shard_idx, _row_idx, meta) in enumerate(getattr(ds, "_index")):
        if not isinstance(meta, dict):
            continue
        tags = meta.get("tags")
        city = _normalize_city(str(meta.get("location") or "unknown"))
        if cities is not None and city not in cities:
            continue
        g = _group_from_tags_ext(tags, straight_mode=straight_mode)
        if g not in groups:
            continue
        groups[g].setdefault(city, []).append(int(ds_idx))

    # deterministic shuffle per group/city
    out: dict[str, dict[str, list[int]]] = {"left": {}, "right": {}, "straight": {}}
    for g, by_city in groups.items():
        rng = random.Random(int(seed) + hash(g) % 100000)
        for city, idxs in by_city.items():
            idxs = list(idxs)
            rng.shuffle(idxs)
            out[g][city] = idxs

    # allocate quotas and trim
    all_cities = sorted({c for g in out.values() for c in g.keys()})
    if cities is not None:
        all_cities = [c for c in cities if c in set(all_cities)]

    quotas = {g: _alloc_per_city(int(samples_per_group), all_cities) for g in ("left", "right", "straight")}

    picked: dict[str, dict[str, list[int]]] = {"left": {}, "right": {}, "straight": {}}
    for g in ("left", "right", "straight"):
        # initial per-city picks
        deficits = 0
        surplus_pool: list[tuple[str, list[int]]] = []
        for city in all_cities:
            want = int(quotas[g].get(city, 0))
            cand = out[g].get(city, [])
            take = min(want, len(cand))
            picked[g][city] = cand[:take]
            if take < want:
                deficits += (want - take)
            # keep remaining for redistribution
            if len(cand) > take:
                surplus_pool.append((city, cand[take:]))

        # redistribute deficits from other cities with remaining candidates
        if deficits > 0 and surplus_pool:
            # deterministic: iterate cities in order
            for city, rem in sorted(surplus_pool, key=lambda x: x[0]):
                if deficits <= 0:
                    break
                extra = min(deficits, len(rem))
                if extra > 0:
                    picked[g].setdefault(city, []).extend(rem[:extra])
                    deficits -= extra

        total_picked = sum(len(v) for v in picked[g].values())
        if total_picked < samples_per_group:
            print(
                f"[warn] group={g}: requested {samples_per_group} total samples but only found {total_picked} across cities={all_cities}",
                file=sys.stderr,
                flush=True,
            )

    return picked


def _collate_feature_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate feature dataset samples.

    Torch default_collate fails on meta['tags'] because it's a ragged list.
    We stack tensor fields and collate meta into lists.
    """

    if not batch:
        return {}

    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        if k == "meta":
            meta_list = [b.get("meta") or {} for b in batch]
            meta_out: dict[str, Any] = {}
            # keep common fields as lists
            for mk in {"sample_id", "shard_dir", "row_idx", "t", "location", "tags"}:
                meta_out[mk] = [m.get(mk) for m in meta_list]
            out["meta"] = meta_out
            continue

        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = default_collate(vals)
        else:
            out[k] = vals
    return out


def _eval_loader(
    *,
    model: PaperDiffusionPlanner,
    loader: DataLoader,
    mode: str,
    diffusion_steps: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    horizon_idxs = [9, 29, 49, 79]
    # Map idx->tag like trainer: idx 9 => 1s
    horizon_tags: list[tuple[int, str]] = []
    for hh in horizon_idxs:
        sec = (int(hh) + 1) / 10.0
        tag = f"{sec:g}s".replace(".", "p")
        horizon_tags.append((int(hh), tag))

    sum_ade: dict[str, float] = {tag: 0.0 for _hh, tag in horizon_tags}
    sum_fde: dict[str, float] = {tag: 0.0 for _hh, tag in horizon_tags}
    n_total = 0

    Pn = int(model.config.predicted_neighbor_num)
    Tf = int(model.config.future_len)

    was_training = model.training
    if mode == "sampler":
        model.eval()
    else:
        model.train(True)

    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v) for k, v in batch.items()}
            B = int(batch["ego_current_state"].shape[0])

            if mode == "proxy":
                x0_4, _ = _build_joint_trajectories_x0(batch, predicted_neighbor_num=Pn, future_len=Tf)
                x0n = model.config.state_normalizer(x0_4)
                neighbor_mask_full, _neighbors_future_valid = _compute_neighbor_masks(
                    batch,
                    predicted_neighbor_num=Pn,
                    future_len=Tf,
                )
                x0n_masked = x0n.clone()
                x0n_masked[:, 1:, :, :][neighbor_mask_full] = 0.0

                t = torch.zeros((B,), device=device, dtype=torch.float32)
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
                    "sampled_trajectories": x0n_masked,
                    "diffusion_time": t,
                }
                _enc, dec_out = model(inputs)
                pred_n = dec_out["score"]  # [B,P,1+T,4] normalized
                pred_x0 = model.config.state_normalizer.inverse(pred_n)
                pred_ego_xy = pred_x0[:, 0, 1:, :2]
            elif mode == "sampler":
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
                    "diffusion_steps": int(diffusion_steps),
                }
                _enc, dec_out = model(inputs)
                pred_raw = dec_out["prediction"]  # [B,P,T,4] inverse space
                pred_ego_xy = pred_raw[:, 0, :, :2]
            else:
                raise ValueError(f"unknown mode: {mode}")

            gt = batch["ego_agent_future"]
            if gt.shape[-2] == Tf + 1:
                gt = gt[:, 1:, :]
            gt_xy = gt[..., :2]

            d = torch.linalg.norm(pred_ego_xy - gt_xy, dim=-1)  # [B,T]

            for hh, tag in horizon_tags:
                if hh < 0 or hh >= int(d.shape[1]):
                    continue
                ade_b = d[:, : hh + 1].mean(dim=-1)  # [B]
                fde_b = d[:, hh]  # [B]
                sum_ade[tag] += float(ade_b.sum().detach().cpu().item())
                sum_fde[tag] += float(fde_b.sum().detach().cpu().item())

            n_total += int(B)

    if was_training:
        model.train(True)
    else:
        model.eval()

    metrics: dict[str, float] = {}
    denom = float(max(n_total, 1))
    for _hh, tag in horizon_tags:
        metrics[f"ade_{tag}"] = sum_ade[tag] / denom
        metrics[f"fde_{tag}"] = sum_fde[tag] / denom
    return metrics, int(n_total)


def _plot_results(rows: list[dict[str, Any]], *, out_dir: Path) -> None:
    if plt is None:
        print("[warn] matplotlib unavailable, skipping plots", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # rows: one per ckpt
    steps = sorted({int(r["step"]) for r in rows if "step" in r})
    if not steps:
        return

    cities = ["boston", "pittsburgh", "las_vegas"]
    maneuvers = ["straight", "left", "right"]
    horizons = ["1s", "3s", "5s", "8s"]

    # helper to pull a series
    def series(city: str, man: str, metric: str) -> list[float]:
        out = []
        for st in steps:
            r = next((x for x in rows if int(x.get("step", -1)) == st), None)
            if r is None:
                out.append(float("nan"))
                continue
            key = f"{metric}_{city}_{man}"
            out.append(float(r.get(key, float("nan"))))
        return out

    # 1) For each maneuver: compare cities
    for man in maneuvers:
        fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
        for j, hz in enumerate(horizons):
            ax_ade = axes[0, j]
            ax_fde = axes[1, j]
            for city in cities:
                ax_ade.plot(steps, series(city, man, f"ade_{hz}"), label=city)
                ax_fde.plot(steps, series(city, man, f"fde_{hz}"), label=city)
            ax_ade.set_title(f"{man} ADE@{hz}")
            ax_fde.set_title(f"{man} FDE@{hz}")
            ax_fde.set_xlabel("step")
            ax_ade.grid(True, alpha=0.3)
            ax_fde.grid(True, alpha=0.3)
        axes[0, 0].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"maneuver_{man}__cities.png", dpi=150)
        plt.close(fig)

    # 2) For each city: compare maneuvers
    for city in cities:
        fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
        for j, hz in enumerate(horizons):
            ax_ade = axes[0, j]
            ax_fde = axes[1, j]
            for man in maneuvers:
                ax_ade.plot(steps, series(city, man, f"ade_{hz}"), label=man)
                ax_fde.plot(steps, series(city, man, f"fde_{hz}"), label=man)
            ax_ade.set_title(f"{city} ADE@{hz}")
            ax_fde.set_title(f"{city} FDE@{hz}")
            ax_fde.set_xlabel("step")
            ax_ade.grid(True, alpha=0.3)
            ax_fde.grid(True, alpha=0.3)
        axes[0, 0].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"city_{city}__maneuvers.png", dpi=150)
        plt.close(fig)


def _load_model_from_ckpt(ckpt_path: Path, *, device: torch.device) -> PaperDiffusionPlanner:
    ckpt_path = _resolve_ckpt_path(ckpt_path)
    ckpt = _torch_load_compat(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"invalid checkpoint payload type: {type(ckpt)}")

    # Prefer explicit paper config stored by ckpt_payload.
    cfg_d = ckpt.get("paper_config")
    if isinstance(cfg_d, dict):
        try:
            cfg = PaperModelConfig(**cfg_d)
        except Exception:
            # best-effort: fall back to defaults then patch known keys
            cfg = PaperModelConfig()
            for k, v in cfg_d.items():
                if hasattr(cfg, k):
                    try:
                        setattr(cfg, k, v)
                    except Exception:
                        pass
    else:
        cfg = PaperModelConfig()

    cfg.device = str(device)

    model = PaperDiffusionPlanner(cfg)

    state = ckpt.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint missing 'model_state' dict")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[warn] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
            flush=True,
        )
        if missing:
            print(f"  missing (first 20): {missing[:20]}", file=sys.stderr, flush=True)
        if unexpected:
            print(f"  unexpected (first 20): {unexpected[:20]}", file=sys.stderr, flush=True)

    model = model.to(device)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp-dirs",
        required=True,
        nargs="+",
        type=str,
        help="One or more outputs/training/<exp> dirs. Use multiple for resumed runs.",
    )
    ap.add_argument("--data-roots", required=True, nargs="+", type=str, help="One or more cache roots across cities")
    ap.add_argument("--samples-per-group", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", type=str, default="sampler", choices=["proxy", "sampler"])
    ap.add_argument("--diffusion-steps", type=int, default=10, help="sampler mode")
    ap.add_argument("--num-ckpts", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--straight-mode",
        type=str,
        default="non_turn",
        choices=["non_turn", "tag"],
        help="How to define straight group. non_turn=not left/right (recommended). tag=use starting_straight_* tags only.",
    )
    ap.add_argument(
        "--cities",
        type=str,
        nargs="+",
        default=["boston", "pittsburgh", "las_vegas"],
        help="Which city labels to include (matches manifest 'location').",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="default: <first_exp_dir>/post_fast_eval_maneuver_n<samples>.jsonl",
    )
    ap.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="default: <output_parent>/post_fast_eval_maneuver_plots/",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output jsonl instead of appending")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    exp_dirs = [Path(p).expanduser().resolve() for p in args.exp_dirs]
    for d in exp_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"exp dir not found: {d}")

    out_path = Path(args.output) if args.output else (exp_dirs[0] / f"post_fast_eval_maneuver_n{int(args.samples_per_group)}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if bool(args.overwrite) and out_path.exists():
        out_path.unlink()

    plots_dir = Path(args.plots_dir) if args.plots_dir else (out_path.parent / "post_fast_eval_maneuver_plots")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Expand cache roots: if a root contains p0..p4 style partitions, include them.
    roots_expanded: list[str] = []
    for r in list(args.data_roots):
        p = Path(r).expanduser().resolve()
        if p.is_dir():
            parts = sorted([d for d in p.iterdir() if d.is_dir() and d.name.startswith("p") and d.name[1:].isdigit()])
            if parts:
                roots_expanded.extend([str(d) for d in parts])
                continue
        roots_expanded.append(str(p))

    # Load dataset (cache-only roots supported).
    ds = ShardedNpzFeatureDataset(
        roots_expanded,
        cache_root="outputs/cache/training_arrays",
    )

    # Build per-city subsets for each group (total samples_per_group per group, evenly split across cities).
    group_city_idxs = _build_group_city_subsets(
        ds,
        samples_per_group=int(args.samples_per_group),
        seed=int(args.seed),
        straight_mode=str(args.straight_mode),
        cities=list(args.cities) if args.cities else None,
    )

    loaders: dict[str, dict[str, DataLoader]] = {"left": {}, "right": {}, "straight": {}}
    for g in ("left", "right", "straight"):
        for city, idxs in group_city_idxs[g].items():
            subset = Subset(ds, idxs)
            loaders[g][city] = DataLoader(
                subset,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=(device.type == "cuda"),
                drop_last=False,
                collate_fn=_collate_feature_samples,
            )

    # Collect checkpoints across exp dirs.
    ckpts = _collect_ckpts(exp_dirs)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint_step_*.pt under exp_dirs={exp_dirs}")
    sel = _select_evenly_spaced(ckpts, int(args.num_ckpts))

    print(f"[info] exp_dirs={','.join(str(d) for d in exp_dirs)}")
    print(f"[info] ckpts_found={len(ckpts)} ckpts_selected={len(sel)}")
    for g in ("left", "right", "straight"):
        for city in sorted(loaders[g].keys()):
            print(f"[info] group={g} city={city} n={len(group_city_idxs[g][city])}")

    # Evaluate and write results.
    rows: list[dict[str, Any]] = []
    for ckpt_path in sel:
        step = _step_from_ckpt_name(ckpt_path)
        model = _load_model_from_ckpt(ckpt_path, device=device)

        row: dict[str, Any] = {
            "step": int(step),
            "ckpt_path": str(_resolve_ckpt_path(ckpt_path)),
            "mode": str(args.mode),
            "diffusion_steps": int(args.diffusion_steps) if args.mode == "sampler" else None,
            "seed": int(args.seed),
            "samples_per_group": int(args.samples_per_group),
            "straight_mode": str(args.straight_mode),
            "cities": list(args.cities),
        }

        # Evaluate each city separately per group.
        for g in ("left", "right", "straight"):
            for city, ldr in loaders[g].items():
                m, n = _eval_loader(
                    model=model,
                    loader=ldr,
                    mode=str(args.mode),
                    diffusion_steps=int(args.diffusion_steps),
                    device=device,
                )
                row[f"count_{city}_{g}"] = int(n)
                for k, v in m.items():
                    row[f"{k}_{city}_{g}"] = float(v)

        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        rows.append(row)

        print(f"[ok] wrote step={step} -> {out_path}", flush=True)

    # plots
    _plot_results(rows, out_dir=plots_dir)
    if rows:
        print(f"[ok] plots -> {plots_dir}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
