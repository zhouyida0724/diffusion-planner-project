from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import nullcontext

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

# Optional TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from src.methods.diffusion_planner.train.trainer import TrainConfig, _assert_finite, _maybe_sync, _PerfTracker, seed_everything
from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner
from src.methods.diffusion_planner.paper.model.diffusion_utils import dpm_solver_pytorch as dpm
from .tb_visualizer import render_npz_style_scene, render_xy_scatter, render_xy_scatter_with_context, stitch_montage


def _t_to_tag(t: float) -> str:
    # stable string for TB tags (avoid '.'; keep precision near 0)
    t = float(t)
    if t < 0.01:
        s = f"{t:.3f}"
    else:
        s = f"{t:.2f}"
    return s.replace(".", "p")


def _read_proc_self_io() -> dict[str, int]:
    """Best-effort process IO counters from /proc/self/io."""

    try:
        out: dict[str, int] = {}
        with open("/proc/self/io", "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                try:
                    out[k] = int(v)
                except Exception:
                    continue
        return out
    except Exception:
        return {}


def _read_rusage() -> dict[str, Any]:
    """Best-effort CPU usage snapshot."""

    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "ru_utime_s": float(getattr(ru, "ru_utime", 0.0)),
            "ru_stime_s": float(getattr(ru, "ru_stime", 0.0)),
            "ru_maxrss_kb": int(getattr(ru, "ru_maxrss", 0)),
            "ru_inblock": int(getattr(ru, "ru_inblock", 0)),
            "ru_oublock": int(getattr(ru, "ru_oublock", 0)),
            "ru_nvcsw": int(getattr(ru, "ru_nvcsw", 0)),
            "ru_nivcsw": int(getattr(ru, "ru_nivcsw", 0)),
        }
    except Exception:
        return {}


def _build_joint_trajectories_x0(batch: dict, *, predicted_neighbor_num: int, future_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build x0 trajectories as [B,P,1+T,4] and current_states [B,P,4].

    Uses ego_current_state, ego_agent_future, neighbor_agents_past, neighbor_agents_future.
    """

    ego_current_state = batch["ego_current_state"]  # [B,10]
    ego_agent_future = batch["ego_agent_future"]  # [B,T,3] (x,y,heading)

    neighbor_past = batch["neighbor_agents_past"]
    neighbor_future = batch["neighbor_agents_future"]

    B = ego_current_state.shape[0]

    ego_current = ego_current_state[:, :4]

    # Decide if neighbor_past includes ego row.
    if neighbor_past.shape[1] == predicted_neighbor_num + 1:
        neighbors_current = neighbor_past[:, 1 : 1 + predicted_neighbor_num, -1, :4]
        neighbors_future_raw = neighbor_future[:, 1 : 1 + predicted_neighbor_num] if neighbor_future.shape[1] == predicted_neighbor_num + 1 else neighbor_future[:, :predicted_neighbor_num]
    else:
        neighbors_current = neighbor_past[:, :predicted_neighbor_num, -1, :4]
        neighbors_future_raw = neighbor_future[:, :predicted_neighbor_num]

    current_states = torch.cat([ego_current[:, None, :], neighbors_current], dim=1)  # [B,P,4]

    # futures to 4D
    if ego_agent_future.shape[-2] == future_len + 1:
        ego_agent_future = ego_agent_future[:, 1:, :]
    elif ego_agent_future.shape[-2] != future_len:
        raise RuntimeError(f"ego_agent_future has unexpected T={ego_agent_future.shape[-2]} (expected {future_len} or {future_len+1})")

    ego_xy = ego_agent_future[..., :2]
    ego_h = ego_agent_future[..., 2]
    ego_cos = torch.cos(ego_h)[..., None]
    ego_sin = torch.sin(ego_h)[..., None]
    ego_future_4 = torch.cat([ego_xy, ego_cos, ego_sin], dim=-1)[:, None, :, :]  # [B,1,T,4]

    # Some exports include current state as the first timestep in *_future arrays.
    if neighbors_future_raw.shape[-2] == future_len + 1:
        neighbors_future_raw = neighbors_future_raw[..., 1:, :]
    elif neighbors_future_raw.shape[-2] != future_len:
        raise RuntimeError(f"neighbor_agents_future has unexpected T={neighbors_future_raw.shape[-2]} (expected {future_len} or {future_len+1})")

    nb_xy = neighbors_future_raw[..., :2]
    nb_h = neighbors_future_raw[..., 2]
    nb_cos = torch.cos(nb_h)[..., None]
    nb_sin = torch.sin(nb_h)[..., None]
    nb_future_4 = torch.cat([nb_xy, nb_cos, nb_sin], dim=-1)  # [B,P-1,T,4]

    future_states = torch.cat([ego_future_4, nb_future_4], dim=1)  # [B,P,T,4]

    # prepend current
    x0 = torch.cat([current_states[:, :, None, :], future_states], dim=2)  # [B,P,1+T,4]
    assert x0.shape[2] == future_len + 1
    return x0, current_states


def _compute_neighbor_masks(
    batch: dict,
    *,
    predicted_neighbor_num: int,
    future_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute neighbor masks matching the official Diffusion-Planner loss.

    Returns:
      neighbor_mask_full: [B,Pn,1+T] bool (True means invalid -> should be zeroed)
      neighbors_future_valid: [B,Pn,T] bool (True means valid future step)

    Mask rule: a neighbor timestep is invalid if all coords are exactly 0.
    """

    neighbor_past = batch["neighbor_agents_past"]
    neighbor_future = batch["neighbor_agents_future"]

    # Decide if past/future arrays include ego row.
    if neighbor_past.shape[1] == predicted_neighbor_num + 1:
        neighbors_current = neighbor_past[:, 1 : 1 + predicted_neighbor_num, -1, :4]
        neighbors_future_raw = (
            neighbor_future[:, 1 : 1 + predicted_neighbor_num]
            if neighbor_future.shape[1] == predicted_neighbor_num + 1
            else neighbor_future[:, :predicted_neighbor_num]
        )
    else:
        neighbors_current = neighbor_past[:, :predicted_neighbor_num, -1, :4]
        neighbors_future_raw = neighbor_future[:, :predicted_neighbor_num]

    # Some exports include current as first timestep in *_future.
    if neighbors_future_raw.shape[-2] == future_len + 1:
        neighbors_future_raw = neighbors_future_raw[..., 1:, :]
    elif neighbors_future_raw.shape[-2] != future_len:
        raise RuntimeError(
            f"neighbor_agents_future has unexpected T={neighbors_future_raw.shape[-2]} (expected {future_len} or {future_len+1})"
        )

    # invalid if all zeros
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0.0), dim=-1) == 0
    neighbor_future_mask = torch.sum(torch.ne(neighbors_future_raw, 0.0), dim=-1) == 0  # [B,Pn,T]

    neighbor_mask_full = torch.cat([neighbor_current_mask.unsqueeze(-1), neighbor_future_mask], dim=-1)  # [B,Pn,1+T]
    neighbors_future_valid = ~neighbor_future_mask
    return neighbor_mask_full, neighbors_future_valid


def _ade_fde_at_horizons(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    horizon_idxs: list[int],
) -> dict[str, float]:
    """Compute ADE/FDE at several horizons.

    Args:
      pred_xy: [B,T,2]
      gt_xy:   [B,T,2]
      horizon_idxs: 0-based indices into T (inclusive horizon).

    Returns flat dict with keys like ade_1s, fde_1s.
    """

    assert pred_xy.ndim == 3 and gt_xy.ndim == 3
    assert pred_xy.shape == gt_xy.shape

    B, T, _ = pred_xy.shape
    d = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # [B,T]

    out: dict[str, float] = {}
    for h in horizon_idxs:
        hh = int(h)
        if hh < 0 or hh >= T:
            continue
        ade = d[:, : hh + 1].mean()
        fde = d[:, hh].mean()
        # Map idx->seconds assuming 10Hz (idx 9 => 1s).
        sec = (hh + 1) / 10.0
        tag = f"{sec:g}s".replace(".", "p")
        out[f"ade_{tag}"] = float(ade.detach().cpu().item())
        out[f"fde_{tag}"] = float(fde.detach().cpu().item())

    return out


def train_loop_paper_dit_xstart(
    *,
    cfg: TrainConfig,
    model: PaperDiffusionPlanner,
    train_loader: DataLoader,
    max_grad_norm: Optional[float] = 1.0,
    fast_val_loader: Optional[DataLoader] = None,
    fast_val_every: int = 0,
    fast_eval_loaders: Optional[dict[str, DataLoader]] = None,
    fast_eval_every: int = 0,
) -> Path:
    """Train paper DiT with x_start objective under VP-SDE.

    Loss: MSE( model(x_t,t,cond), x0 )
    where x_t is noised from x0 using model.sde.marginal_prob.
    """

    seed_everything(cfg.seed)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    exp_dir = Path(cfg.out_root) / cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    spike_path = exp_dir / "spike_dump.jsonl"

    tb = None
    tb_enable = bool(getattr(cfg, "tb_enable", False))
    tb_every = int(getattr(cfg, "tb_every", 0) or 0)
    tb_num_samples = max(1, int(getattr(cfg, "tb_num_samples", 1) or 1))
    tb_denoise_k = max(1, int(getattr(cfg, "tb_denoise_k", 10) or 1))
    tb_denoise_mode = str(getattr(cfg, "tb_denoise_mode", "t_sweep") or "t_sweep")
    tb_sampler_steps = max(1, int(getattr(cfg, "tb_sampler_steps", tb_denoise_k) or tb_denoise_k))
    tb_image_size = int(getattr(cfg, "tb_image_size", 800) or 800)
    tb_warned_disabled = False

    # Pre-sample stable visualization examples from fast-eval loaders (if available).
    # This keeps TB tags stable across steps and aligned across cities.
    tb_vis_by_city: dict[str, list[dict[str, Any]]] = {}
    if fast_eval_loaders is not None:
        for _city, _ldr in fast_eval_loaders.items():
            try:
                _batch = next(iter(_ldr))
            except Exception:
                continue
            B0 = None
            for _v in _batch.values():
                if torch.is_tensor(_v):
                    B0 = int(_v.shape[0])
                    break
            if B0 is None or B0 <= 0:
                continue
            samples: list[dict[str, Any]] = []
            for i in range(min(2, B0)):
                s: dict[str, Any] = {}
                for k, v in _batch.items():
                    if torch.is_tensor(v):
                        s[k] = v[i : i + 1].to(device=device, dtype=torch.float32)
                    else:
                        s[k] = v
                samples.append(s)
            if samples:
                tb_vis_by_city[str(_city)] = samples

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Optional resume
    start_step = 0
    resume_ckpt = getattr(cfg, "resume_ckpt", None)
    if resume_ckpt:
        # Load directly onto the training device so optimizer state tensors match param device.
        ckpt = torch.load(str(resume_ckpt), map_location=device)
        if isinstance(ckpt, dict):
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"], strict=True)
            if "optimizer_state" in ckpt:
                try:
                    opt.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    print(f"[resume] optimizer state load failed: {e}; continuing with fresh optimizer", flush=True)
            start_step = int(ckpt.get("step", 0)) + 1
            print(f"[resume] loaded {resume_ckpt}; start_step={start_step}", flush=True)
        else:
            print(f"[resume] invalid checkpoint payload type: {type(ckpt)}; ignoring resume", flush=True)
    if tb_enable and SummaryWriter is not None:
        try:
            # TensorBoard sometimes gets confused when multiple event files share the same base name
            # and one gets an auto-suffix like ".1". Make filenames unique per process start.
            tb = SummaryWriter(log_dir=str(exp_dir / "tb"), filename_suffix=f".{uuid.uuid4().hex[:8]}")
            tb.add_text("meta/exp_name", str(cfg.exp_name), 0)
            tb.add_text("meta/mode", "paper_dit_dpm", 0)
            tb.add_text("meta/tb_image_logging", f"enabled every={tb_every} samples={tb_num_samples} denoise_k={tb_denoise_k}", 0)
        except Exception as e:
            print(f"[tb] disabled: failed to initialize SummaryWriter: {e}", flush=True)
            tb = None
    elif tb_enable and SummaryWriter is None:
        print("[tb] disabled: torch.utils.tensorboard is unavailable in this environment", flush=True)

    # (optimizer created above; may have been resumed)

    def _lr_for_step(step_i: int) -> float:
        """Compute LR for this step based on cfg.* schedule knobs."""

        base_lr = float(cfg.lr)
        sched = str(getattr(cfg, "lr_schedule", "constant") or "constant").lower()
        warmup = int(getattr(cfg, "lr_warmup_steps", 0) or 0)

        if warmup > 0 and step_i < warmup:
            return base_lr * float(step_i + 1) / float(max(warmup, 1))

        if sched == "cosine":
            import math

            min_ratio = float(getattr(cfg, "lr_min_ratio", 1.0) or 1.0)
            min_lr = base_lr * min_ratio
            denom = max(int(cfg.steps) - warmup, 1)
            prog = float(step_i - warmup) / float(denom)
            prog = max(0.0, min(1.0, prog))
            return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog)))

        return base_lr

    # AMP (stability-first):
    # - bf16 autocast (preferred): typically stable without GradScaler.
    # - fp16 autocast: use GradScaler + unscale before clipping.
    amp_mode = str(getattr(cfg, "amp", "off") or "off").lower()
    if amp_mode == "off":
        # Temporary back-compat escape hatch: allow opting into AMP without changing callers.
        # Prefer wiring a proper CLI flag into scripts/train/diffusion_planner/train.py.
        env_amp = os.environ.get("DIFFPLANNER_AMP", "").strip().lower()
        if env_amp:
            amp_mode = env_amp

    if amp_mode not in {"off", "bf16", "fp16"}:
        raise ValueError(f"Unknown amp mode: {amp_mode}")

    use_amp = (amp_mode != "off") and (device.type == "cuda") and torch.cuda.is_available()
    if amp_mode == "bf16":
        amp_dtype = torch.bfloat16
        scaler = None
    elif amp_mode == "fp16":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    else:
        amp_dtype = None
        scaler = None

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(use_amp))
        if device.type == "cuda"
        else nullcontext()
    )

    perf = _PerfTracker(exp_dir=exp_dir, cfg=cfg, device=device)

    # TensorBoard (optional)
    # We already initialize `tb` above when tb_enable=True. Avoid creating a second SummaryWriter
    # pointing at the same logdir, because it will produce events.*.1 and TensorBoard may get stuck.
    writer = None
    if bool(getattr(cfg, "tensorboard", False)) and tb is None:
        if SummaryWriter is None:
            print("[tb] SummaryWriter unavailable (torch.utils.tensorboard not installed); skipping TensorBoard logs.", flush=True)
        else:
            tb_dir = getattr(cfg, "tb_dir", None) or str(exp_dir / "tb")
            writer = SummaryWriter(log_dir=tb_dir, filename_suffix=f".{uuid.uuid4().hex[:8]}")
            print(f"[tb] writing TensorBoard logs to: {tb_dir}", flush=True)

    model.train()
    t0 = time.time()
    step = int(start_step)
    loader_it = iter(train_loader)

    # When cfg.steps == 0 (eval-only runs), the training loop body never executes.
    # Keep ckpt_path defined so we can still write checkpoint_latest or skip safely.
    ckpt_path: Optional[Path] = None

    Pn = int(model.config.predicted_neighbor_num)
    Tf = int(model.config.future_len)

    # -----------------
    # fast validation
    # -----------------
    fast_val_every = int(fast_val_every or 0)
    fast_val_path = exp_dir / "fast_val.jsonl"

    # New: multi-domain fast evaluation (city-equal) with macro-average.
    fast_eval_every = int(fast_eval_every or 0)
    fast_eval_path = exp_dir / "fast_eval.jsonl"

    def _run_fast_eval(step_i: int) -> None:
        if not fast_eval_loaders or fast_eval_every <= 0:
            return

        fast_eval_mode = str(getattr(cfg, "fast_eval_mode", "proxy") or "proxy")
        if fast_eval_mode not in ("proxy", "sampler"):
            fast_eval_mode = "proxy"

        # NOTE: proxy mode historically used train-mode per request.
        # Sampler mode must match inference, so we run in eval() temporarily.
        was_training = model.training
        if fast_eval_mode == "sampler":
            model.eval()
        else:
            model.train()  # keep train-mode for legacy proxy
        t_eval0 = time.time()

        horizon_idxs = [9, 29, 49, 79]
        alpha_planning_loss = float(getattr(cfg, "alpha_planning_loss", 1.0) or 1.0)

        per_city: dict[str, dict[str, float]] = {}

        def _eval_one(loader: DataLoader) -> tuple[dict[str, float], int]:
            n_total = 0
            sum_metrics: dict[str, float] = {}
            with torch.no_grad():
                for batch in loader:
                    batch = {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v) for k, v in batch.items()}

                    B = int(batch["ego_current_state"].shape[0])

                    if fast_eval_mode == "proxy":
                        # --- teacher-forced proxy: one-step x0 prediction at t=0 ---
                        x0_4, _ = _build_joint_trajectories_x0(batch, predicted_neighbor_num=Pn, future_len=Tf)
                        x0n = model.config.state_normalizer(x0_4)

                        neighbor_mask_full, neighbors_future_valid = _compute_neighbor_masks(
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

                        with autocast_ctx:
                            _, dec_out = model(inputs)
                        pred = dec_out["score"]  # [B,P,1+T,4] (normalized)

                        # val loss proxy (future-only, official masking)
                        pred_fut = pred[:, :, 1:, :]
                        gt_fut = x0n_masked[:, :, 1:, :]
                        dpm_loss = torch.sum((pred_fut - gt_fut) ** 2, dim=-1)  # [B,P,T]
                        ego_loss = dpm_loss[:, 0, :].mean()
                        nb_elems = dpm_loss[:, 1:, :][neighbors_future_valid]
                        nb_loss = nb_elems.mean() if nb_elems.numel() > 0 else torch.tensor(0.0, device=dpm_loss.device)
                        val_loss_proxy = ego_loss + alpha_planning_loss * nb_loss

                        # ADE/FDE on ego (x,y)
                        pred_x0 = model.config.state_normalizer.inverse(pred)
                        pred_ego_xy = pred_x0[:, 0, 1:, :2]
                    else:
                        # --- sampler mode: run inference sampler (matches closed-loop) ---
                        steps_s = int(getattr(cfg, "fast_eval_diffusion_steps", 10) or 10)
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
                            "diffusion_steps": steps_s,
                        }
                        with autocast_ctx:
                            _, dec_out = model(inputs)
                        # dec_out["prediction"]: [B,P,T,4] in inverse space
                        pred_raw = dec_out["prediction"]  # [B,P,T,4] raw
                        pred_ego_xy = pred_raw[:, 0, :, :2]

                        # In sampler mode we still want a per-city "loss-like" scalar that is
                        # consistent with inference. We compute an MSE in normalized space
                        # between sampled prediction and GT (future only), using the same
                        # neighbor validity masking as training.
                        x0_4, _ = _build_joint_trajectories_x0(batch, predicted_neighbor_num=Pn, future_len=Tf)
                        x0n = model.config.state_normalizer(x0_4)
                        neighbor_mask_full, neighbors_future_valid = _compute_neighbor_masks(
                            batch,
                            predicted_neighbor_num=Pn,
                            future_len=Tf,
                        )
                        x0n_masked = x0n.clone()
                        x0n_masked[:, 1:, :, :][neighbor_mask_full] = 0.0
                        gt_fut_n = x0n_masked[:, :, 1:, :]  # [B,P,T,4]
                        pred_fut_n = model.config.state_normalizer(pred_raw)  # [B,P,T,4]
                        dpm_loss = torch.sum((pred_fut_n - gt_fut_n) ** 2, dim=-1)  # [B,P,T]
                        ego_loss = dpm_loss[:, 0, :].mean()
                        nb_elems = dpm_loss[:, 1:, :][neighbors_future_valid]
                        nb_loss = nb_elems.mean() if nb_elems.numel() > 0 else torch.tensor(0.0, device=dpm_loss.device)
                        val_loss_proxy = ego_loss + alpha_planning_loss * nb_loss

                    gt_ego_future = batch["ego_agent_future"]
                    if gt_ego_future.shape[-2] == Tf + 1:
                        gt_ego_future = gt_ego_future[:, 1:, :]
                    gt_ego_xy = gt_ego_future[..., :2]

                    ade_fde = _ade_fde_at_horizons(pred_ego_xy, gt_ego_xy, horizon_idxs)

                    metrics_batch: dict[str, float] = {
                        "val_loss_proxy": float(val_loss_proxy.detach().float().cpu().item()) if torch.is_tensor(val_loss_proxy) else float("nan"),
                    }
                    metrics_batch.update(ade_fde)

                    n_total += B
                    for k, v in metrics_batch.items():
                        sum_metrics[k] = sum_metrics.get(k, 0.0) + float(v) * B

            if n_total <= 0:
                return {}, 0
            metrics = {k: (v / n_total) for k, v in sum_metrics.items()}
            return metrics, n_total

        # evaluate each city equally (no city is "main")
        for city, loader in fast_eval_loaders.items():
            metrics, n = _eval_one(loader)
            if not metrics:
                continue
            metrics.update(
                {
                    "city": str(city),
                    "step": int(step_i),
                    "n": int(n),
                    "wall_s": float(time.time() - t_eval0),
                    "ts": time.time(),
                }
            )
            per_city[str(city)] = dict(metrics)

            parts = [f"{k}={metrics[k]:.6f}" for k in sorted(metrics.keys()) if k.startswith(("val_loss_proxy", "ade_", "fde_"))]
            print(f"[fast-eval] city={city} step {step_i:05d} | n={n} | " + " | ".join(parts), flush=True)

            if tb is not None:
                try:
                    tb.add_scalar(f"fast_eval/{city}/val_loss_proxy", float(metrics.get("val_loss_proxy", float("nan"))), int(step_i))
                    for k, v in metrics.items():
                        if k.startswith("ade_") or k.startswith("fde_"):
                            tb.add_scalar(f"fast_eval/{city}/{k}", float(v), int(step_i))
                except Exception:
                    pass

            with fast_eval_path.open("a") as f:
                f.write(json.dumps(metrics) + "\n")

        # macro-average across cities (equal weight)
        if per_city:
            keys = [k for k in next(iter(per_city.values())).keys() if k.startswith(("val_loss_proxy", "ade_", "fde_"))]
            macro: dict[str, float] = {}
            for k in keys:
                vals = [float(m[k]) for m in per_city.values() if k in m]
                if vals:
                    macro[k] = float(sum(vals) / len(vals))
            macro_rec: dict[str, float | int | str] = {
                "city": "macro",
                "step": int(step_i),
                "n": int(min(int(m.get("n", 0)) for m in per_city.values()) if per_city else 0),
                "wall_s": float(time.time() - t_eval0),
                "ts": time.time(),
                **macro,
            }

            if tb is not None:
                try:
                    tb.add_scalar("fast_eval_macro/val_loss_proxy", float(macro.get("val_loss_proxy", float("nan"))), int(step_i))
                    for k, v in macro.items():
                        if k.startswith("ade_") or k.startswith("fde_"):
                            tb.add_scalar(f"fast_eval_macro/{k}", float(v), int(step_i))
                    tb.flush()
                except Exception:
                    pass

            with fast_eval_path.open("a") as f:
                f.write(json.dumps(macro_rec) + "\n")

        # restore model mode
        if fast_eval_mode == "sampler":
            model.train(was_training)

    def _run_fast_val(step_i: int) -> None:
        if fast_val_loader is None or fast_val_every <= 0:
            return

        model.train()  # keep train-mode per request
        t_eval0 = time.time()

        horizon_idxs = [9, 29, 49, 79]
        alpha_planning_loss = float(getattr(cfg, "alpha_planning_loss", 1.0) or 1.0)

        n_total = 0
        sum_metrics: dict[str, float] = {}

        with torch.no_grad():
            for batch in fast_val_loader:
                batch = {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v) for k, v in batch.items()}

                x0_4, _ = _build_joint_trajectories_x0(batch, predicted_neighbor_num=Pn, future_len=Tf)
                x0n = model.config.state_normalizer(x0_4)

                neighbor_mask_full, neighbors_future_valid = _compute_neighbor_masks(
                    batch,
                    predicted_neighbor_num=Pn,
                    future_len=Tf,
                )

                x0n_masked = x0n.clone()
                x0n_masked[:, 1:, :, :][neighbor_mask_full] = 0.0

                B = int(x0n_masked.shape[0])
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

                with autocast_ctx:
                    _, dec_out = model(inputs)
                pred = dec_out["score"]  # [B,P,1+T,4]

                # --- val loss proxy (future-only, official masking) ---
                pred_fut = pred[:, :, 1:, :]
                gt_fut = x0n_masked[:, :, 1:, :]
                dpm_loss = torch.sum((pred_fut - gt_fut) ** 2, dim=-1)  # [B,P,T]
                ego_loss = dpm_loss[:, 0, :].mean()
                nb_elems = dpm_loss[:, 1:, :][neighbors_future_valid]
                nb_loss = nb_elems.mean() if nb_elems.numel() > 0 else torch.tensor(0.0, device=dpm_loss.device)
                val_loss_proxy = ego_loss + alpha_planning_loss * nb_loss

                # --- ADE/FDE on ego (x,y) ---
                # denormalize and take ego future
                pred_x0 = model.config.state_normalizer.inverse(pred) if hasattr(model.config.state_normalizer, "inverse") else None
                if pred_x0 is None:
                    # Fallback: if inverse not available, compute on normalized space (still monotonic-ish).
                    pred_ego_xy = pred[:, 0, 1:, :2]
                else:
                    pred_ego_xy = pred_x0[:, 0, 1:, :2]

                gt_ego_future = batch["ego_agent_future"]
                if gt_ego_future.shape[-2] == Tf + 1:
                    gt_ego_future = gt_ego_future[:, 1:, :]
                gt_ego_xy = gt_ego_future[..., :2]

                ade_fde = _ade_fde_at_horizons(pred_ego_xy, gt_ego_xy, horizon_idxs)

                metrics_batch: dict[str, float] = {
                    "val_loss_proxy": float(val_loss_proxy.detach().float().cpu().item()),
                }
                metrics_batch.update(ade_fde)

                n_total += B
                for k, v in metrics_batch.items():
                    sum_metrics[k] = sum_metrics.get(k, 0.0) + float(v) * B

        if n_total <= 0:
            return

        metrics = {k: (v / n_total) for k, v in sum_metrics.items()}
        metrics["step"] = int(step_i)
        metrics["n"] = int(n_total)
        metrics["wall_s"] = float(time.time() - t_eval0)
        metrics["ts"] = time.time()

        # stdout
        parts = [f"{k}={metrics[k]:.6f}" for k in sorted(metrics.keys()) if k.startswith(("val_loss_proxy", "ade_", "fde_"))]
        print(f"[fast-val] step {step_i:05d} | n={n_total} | " + " | ".join(parts), flush=True)

        # tensorboard
        if tb is not None:
            try:
                tb.add_scalar("fast_val/val_loss_proxy", float(metrics.get("val_loss_proxy", float("nan"))), int(step_i))
                for k, v in metrics.items():
                    if k.startswith("ade_") or k.startswith("fde_"):
                        tb.add_scalar(f"fast_val/{k}", float(v), int(step_i))
                tb.flush()
            except Exception:
                pass

        # jsonl
        with fast_val_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")

        # optionally attach to perf record
        if perf.records and int(perf.records[-1].get("step", -1)) == int(step_i):
            perf.records[-1]["fast_val"] = dict(metrics)
            perf.write_perf_json()

    # initial fast-val at step 0
    if fast_val_loader is not None and fast_val_every > 0:
        _run_fast_val(0)

    # initial fast-eval at step 0
    if fast_eval_loaders is not None and fast_eval_every > 0:
        _run_fast_eval(0)

    profile_steps = int(getattr(cfg, "profile_steps", 0) or 0)
    profile_every = int(getattr(cfg, "profile_every", 0) or 0)

    def _do_profile(i: int) -> bool:
        return profile_steps > 0 and i < profile_steps

    def _do_profile_print(i: int) -> bool:
        if not _do_profile(i):
            return False
        if profile_every <= 0:
            return True
        return (i % profile_every) == 0

    while step < cfg.steps:
        if fast_val_loader is not None and fast_val_every > 0 and (step % fast_val_every) == 0 and step != 0:
            _run_fast_val(step)

        if fast_eval_loaders is not None and fast_eval_every > 0 and (step % fast_eval_every) == 0 and step != 0:
            _run_fast_eval(step)

        step_t0 = time.time()
        breakdown: dict[str, Any] = {}

        # Best-effort host stats (for IO stalls).
        io0 = _read_proc_self_io() if _do_profile(step) else {}
        ru0 = _read_rusage() if _do_profile(step) else {}

        # dataloader
        t_seg = time.perf_counter()
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(train_loader)
            batch = next(loader_it)
        breakdown["dataloader_next_s"] = float(time.perf_counter() - t_seg)

        # move tensors
        t_seg = time.perf_counter()
        batch = {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v) for k, v in batch.items()}
        breakdown["to_device_s"] = float(time.perf_counter() - t_seg)

        # build + diffusion setup (count as misc, but keep a couple sub-keys for debugging)
        t_seg = time.perf_counter()
        x0_4, _current_states = _build_joint_trajectories_x0(batch, predicted_neighbor_num=Pn, future_len=Tf)
        _assert_finite(x0_4, "x0")
        breakdown["build_x0_s"] = float(time.perf_counter() - t_seg)

        t_seg = time.perf_counter()
        x0n = model.config.state_normalizer(x0_4)  # [B,P,1+T,4]

        # Neighbor invalid masks (official behavior):
        # - mask current if neighbor_current is all zeros
        # - mask future steps if neighbor_future is all zeros
        neighbor_mask_full, neighbors_future_valid = _compute_neighbor_masks(
            batch,
            predicted_neighbor_num=Pn,
            future_len=Tf,
        )

        x0n_masked = x0n.clone()
        x0n_masked[:, 1:, :, :][neighbor_mask_full] = 0.0

        B = x0n_masked.shape[0]
        t = torch.rand((B,), device=device, dtype=torch.float32)

        # Noise only the future (t=1..T); keep current fixed.
        noise_fut = torch.randn_like(x0n_masked[:, :, 1:, :])
        mean, std = model.sde.marginal_prob(x0n_masked[:, :, 1:, :], t)
        xt_fut = mean + std * noise_fut
        xt = torch.cat([x0n_masked[:, :, :1, :], xt_fut], dim=2)
        breakdown["misc_setup_s"] = float(time.perf_counter() - t_seg)

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

        # forward
        if _do_profile(step):
            _maybe_sync(device)
        t_seg = time.perf_counter()
        with autocast_ctx:
            _, dec_out = model(inputs)
        if _do_profile(step):
            _maybe_sync(device)
        breakdown["forward_s"] = float(time.perf_counter() - t_seg)

        pred = dec_out["score"]  # [B,P,1+T,4]
        _assert_finite(pred, "pred")

        # loss
        if _do_profile(step):
            _maybe_sync(device)
        t_seg = time.perf_counter()
        alpha_planning_loss = float(getattr(cfg, "alpha_planning_loss", 1.0) or 1.0)
        with autocast_ctx:
            # Official Diffusion-Planner (x_start) behavior:
            # - supervise only future timesteps (exclude current)
            # - ignore invalid neighbor futures via mask
            pred_fut = pred[:, :, 1:, :]  # [B,P,T,4]
            gt_fut = x0n_masked[:, :, 1:, :]  # [B,P,T,4]

            dpm_loss = torch.sum((pred_fut - gt_fut) ** 2, dim=-1)  # [B,P,T]

            ego_planning_loss = dpm_loss[:, 0, :].mean()

            nb_elems = dpm_loss[:, 1:, :][neighbors_future_valid]
            if nb_elems.numel() > 0:
                neighbor_prediction_loss = nb_elems.mean()
            else:
                neighbor_prediction_loss = torch.tensor(0.0, device=dpm_loss.device)

            loss = ego_planning_loss + alpha_planning_loss * neighbor_prediction_loss
        if _do_profile(step):
            _maybe_sync(device)
        breakdown["loss_s"] = float(time.perf_counter() - t_seg)
        breakdown["ego_planning_loss"] = float(ego_planning_loss.detach().float().item())
        breakdown["neighbor_prediction_loss"] = float(neighbor_prediction_loss.detach().float().item())
        _assert_finite(loss, "loss")

        # LR schedule (per-step)
        lr_now = _lr_for_step(int(step))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        # Spike dump (data/step diagnosis)
        if bool(getattr(cfg, "spike_dump", False)) and int(step) >= int(getattr(cfg, "spike_start", 2000) or 2000):
            try:
                loss_val = float(loss.detach().float().cpu().item())
                thresh = float(getattr(cfg, "spike_thresh", 0.9) or 0.9)
                if loss_val >= thresh:
                    topk = int(getattr(cfg, "spike_topk", 8) or 8)

                    # per-sample ego/neighbor losses
                    ego_per = dpm_loss[:, 0, :].mean(dim=1)  # [B]
                    nb = dpm_loss[:, 1:, :]  # [B,P-1,T]
                    valid = neighbors_future_valid
                    valid_f = valid.to(dtype=nb.dtype)
                    nb_sum = (nb * valid_f).sum(dim=(1, 2))
                    nb_cnt = valid_f.sum(dim=(1, 2)).clamp(min=1.0)
                    nb_per = nb_sum / nb_cnt
                    total_per = ego_per + float(alpha_planning_loss) * nb_per

                    k = min(int(topk), int(total_per.shape[0]))
                    vals, idxs = torch.topk(total_per.detach().float().cpu(), k=k)

                    meta = batch.get("meta")

                    def _jsonable(v: Any) -> Any:
                        """Convert common tensor/numpy/path-ish values into JSON-serializable types."""

                        try:
                            import numpy as np  # type: ignore
                        except Exception:  # pragma: no cover
                            np = None  # type: ignore

                        if v is None:
                            return None
                        if torch.is_tensor(v):
                            vv = v.detach().cpu()
                            if vv.numel() == 1:
                                return vv.item()
                            # Avoid gigantic dumps; keep small vectors readable.
                            try:
                                return vv.tolist()
                            except Exception:
                                return str(vv)
                        if np is not None and isinstance(v, (getattr(np, "integer", ()), getattr(np, "floating", ()))):
                            try:
                                return v.item()
                            except Exception:
                                return float(v)
                        if isinstance(v, (bytes, bytearray)):
                            try:
                                return v.decode("utf-8", errors="replace")
                            except Exception:
                                return str(v)
                        # Path-like
                        try:
                            from pathlib import Path

                            if isinstance(v, Path):
                                return str(v)
                        except Exception:
                            pass
                        # Basic python types are already JSONable.
                        if isinstance(v, (str, int, float, bool)):
                            return v
                        # Lists/tuples: convert elements
                        if isinstance(v, (list, tuple)):
                            return [_jsonable(x) for x in v]
                        # Dict: convert values
                        if isinstance(v, dict):
                            return {str(k): _jsonable(val) for k, val in v.items()}
                        # Fallback
                        return str(v)

                    def _meta_at(i: int) -> dict[str, Any]:
                        if isinstance(meta, list) and i < len(meta):
                            m = meta[i]
                            if isinstance(m, dict):
                                return {str(k): _jsonable(v) for k, v in m.items()}
                            return {"meta": _jsonable(m)}
                        if isinstance(meta, dict):
                            out: dict[str, Any] = {}
                            for kk, vv in meta.items():
                                try:
                                    if isinstance(vv, (list, tuple)) and i < len(vv):
                                        out[kk] = _jsonable(vv[i])
                                    else:
                                        out[kk] = _jsonable(vv)
                                except Exception:
                                    continue
                            return out
                        return {}

                    def _infer_city(shard_dir: Any) -> str | None:
                        if shard_dir is None:
                            return None
                        s = str(shard_dir).lower()
                        if "boston" in s:
                            return "boston"
                        if "pittsburgh" in s:
                            return "pittsburgh"
                        if "vegas" in s:
                            return "vegas"
                        return None

                    rec = {
                        "step": int(step),
                        "loss": float(loss_val),
                        "lr": float(lr_now),
                        "alpha_planning_loss": float(alpha_planning_loss),
                        "thresh": float(thresh),
                        "topk": [],
                        "ts": time.time(),
                    }
                    for rank, (v, j) in enumerate(zip(vals.tolist(), idxs.tolist())):
                        m = _meta_at(int(j))
                        shard_dir = m.get("shard_dir")
                        rec["topk"].append(
                            {
                                "rank": int(rank),
                                "total": float(v),
                                "ego": float(ego_per[int(j)].detach().float().cpu().item()),
                                "nb": float(nb_per[int(j)].detach().float().cpu().item()),
                                "t": float(t[int(j)].detach().float().cpu().item()) if torch.is_tensor(t) else None,
                                "sample_id": m.get("sample_id"),
                                "shard_dir": shard_dir,
                                "row_idx": m.get("row_idx"),
                                "city": _infer_city(shard_dir),
                            }
                        )

                    with spike_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(
                        f"[spike] step {int(step):05d} loss={loss_val:.6f} lr={lr_now:.2e} "
                        f"top0_total={rec['topk'][0]['total']:.6f} city={rec['topk'][0].get('city')}",
                        flush=True,
                    )
            except Exception as e:
                print(f"[spike] dump failed: {e}", flush=True)

        # optim
        if _do_profile(step):
            _maybe_sync(device)
        t_seg = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        breakdown["zero_grad_s"] = float(time.perf_counter() - t_seg)

        if _do_profile(step):
            _maybe_sync(device)
        t_seg = time.perf_counter()
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            # unscale before clipping / finite checks
            scaler.unscale_(opt)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if _do_profile(step):
            _maybe_sync(device)
        breakdown["backward_s"] = float(time.perf_counter() - t_seg)

        if _do_profile(step):
            _maybe_sync(device)
        t_seg = time.perf_counter()
        if scaler is not None and scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if _do_profile(step):
            _maybe_sync(device)
        breakdown["optim_step_s"] = float(time.perf_counter() - t_seg)

        # overall step
        _maybe_sync(device)
        step_s = time.time() - step_t0
        perf.on_step_end(step=step, step_s=step_s, loss=float(loss.item()))

        # attach breakdown + host stats to perf record (perf.json)
        if _do_profile(step) and perf.records:
            rec = perf.records[-1]
            rec["breakdown"] = dict(breakdown)
            rec["rusage"] = dict(ru0) if ru0 else {}
            rec["proc_io0"] = dict(io0) if io0 else {}
            io1 = _read_proc_self_io()
            ru1 = _read_rusage()
            rec["proc_io1"] = dict(io1) if io1 else {}
            rec["rusage1"] = dict(ru1) if ru1 else {}
            try:
                rec["proc_io_delta"] = {k: int(io1.get(k, 0)) - int(io0.get(k, 0)) for k in set(io0) | set(io1)}
            except Exception:
                rec["proc_io_delta"] = {}

            # Print a compact breakdown for diagnosis.
            if _do_profile_print(step):
                keys = [
                    "dataloader_next_s",
                    "to_device_s",
                    "build_x0_s",
                    "misc_setup_s",
                    "forward_s",
                    "loss_s",
                    "zero_grad_s",
                    "backward_s",
                    "optim_step_s",
                ]
                parts = [f"{k}={breakdown.get(k, float('nan')):.4f}" for k in keys]
                print("[profile] step %05d | %s" % (step, " | ".join(parts)), flush=True)

        if (step % cfg.log_every) == 0 or step == cfg.steps - 1:
            dt = time.time() - t0
            last = perf.records[-1]
            msg = (
                f"step {step:05d} | loss {loss.item():.6f} | "
                f"step_s {last['step_s']:.4f} (ema {last['ema_step_s']:.4f}) | "
                f"samples/s {last['samples_s']:.1f} | {dt:.1f}s"
            )
            print(msg, flush=True)
            with (exp_dir / "train.log").open("a") as f:
                f.write(msg + "\n")
            perf.write_perf_json()

            if tb is not None:
                try:
                    tb.add_scalar("train/loss", float(loss.detach().float().cpu().item()), int(step))
                    tb.add_scalar("train/ego_planning_loss", float(ego_planning_loss.detach().float().cpu().item()), int(step))
                    tb.add_scalar("train/neighbor_prediction_loss", float(neighbor_prediction_loss.detach().float().cpu().item()), int(step))
                    tb.add_scalar("perf/step_s", float(last.get("step_s", 0.0)), int(step))
                    tb.add_scalar("perf/samples_s", float(last.get("samples_s", 0.0)), int(step))
                    tb.add_scalar("opt/lr", float(opt.param_groups[0].get("lr", cfg.lr)), int(step))
                except Exception:
                    pass

        should_log_tb_images = tb is not None and tb_every > 0 and ((step % tb_every) == 0 or step == cfg.steps - 1)
        if should_log_tb_images:
            model_was_training = model.training
            try:
                model.eval()

                # TB layout (stable tags):
                #   viz/<city>/sample_<k>
                #   denoise/<city>/sample_<k>
                #   forward/<city>/sample_<k>
                # No per-solver-step spam.

                cities = ["boston", "pittsburgh", "vegas"]
                vis_per_city = max(1, int(tb_num_samples) // max(1, len(cities)))
                vis_per_city = min(vis_per_city, 2)

                def _fallback_from_train_batch(n: int, *, salt: int) -> list[dict[str, Any]]:
                    B_vis = int(batch["ego_current_state"].shape[0])
                    if B_vis <= 0:
                        return []
                    g = torch.Generator(device="cpu")
                    g.manual_seed(int(cfg.seed) + int(step) * 10007 + int(salt))
                    idxs = torch.randperm(B_vis, generator=g)[: min(n, B_vis)].tolist()
                    outs: list[dict[str, Any]] = []
                    for sample_idx in idxs:
                        b1: dict[str, Any] = {}
                        for k, v in batch.items():
                            if torch.is_tensor(v) and v.shape[0] == B_vis:
                                b1[k] = v[sample_idx : sample_idx + 1]
                            else:
                                b1[k] = v
                        outs.append(b1)
                    return outs

                for ci, city in enumerate(cities):
                    samples = tb_vis_by_city.get(city)
                    if not samples:
                        samples = _fallback_from_train_batch(vis_per_city, salt=ci)
                    if not samples:
                        continue

                    for slot, batch1 in enumerate(samples[:vis_per_city]):
                        with torch.no_grad():
                            x0_4_1, _ = _build_joint_trajectories_x0(batch1, predicted_neighbor_num=Pn, future_len=Tf)
                            x0n_1 = model.config.state_normalizer(x0_4_1)
                            neighbor_mask_full_1, _ = _compute_neighbor_masks(batch1, predicted_neighbor_num=Pn, future_len=Tf)
                            x0n_masked_1 = x0n_1.clone()
                            x0n_masked_1[:, 1:, :, :][neighbor_mask_full_1] = 0.0

                            # IMPORTANT: match training/inference by applying observation normalization
                            # to conditioning features before feeding the encoder/DiT sampler.
                            enc_inputs_vis_raw = {
                                "ego_current_state": batch1["ego_current_state"],
                                "neighbor_agents_past": batch1["neighbor_agents_past"],
                                "static_objects": batch1["static_objects"],
                                "lanes": batch1["lanes"],
                                "lanes_speed_limit": batch1["lanes_speed_limit"],
                                "lanes_has_speed_limit": batch1["lanes_has_speed_limit"],
                                "route_lanes": batch1["route_lanes"],
                            }
                            if batch1.get("route_lanes_speed_limit") is not None:
                                enc_inputs_vis_raw["route_lanes_speed_limit"] = batch1["route_lanes_speed_limit"]
                            if batch1.get("route_lanes_has_speed_limit") is not None:
                                enc_inputs_vis_raw["route_lanes_has_speed_limit"] = batch1["route_lanes_has_speed_limit"]

                            enc_inputs_vis = (
                                model.config.observation_normalizer(enc_inputs_vis_raw)
                                if hasattr(model.config, "observation_normalizer")
                                else enc_inputs_vis_raw
                            )
                            enc_vis = model.encoder(enc_inputs_vis)

                            neighbors_past_1 = batch1["neighbor_agents_past"]
                            if neighbors_past_1.shape[1] == Pn + 1:
                                neighbors_current_1 = neighbors_past_1[:, 1 : 1 + Pn, -1, :4]
                            else:
                                neighbors_current_1 = neighbors_past_1[:, :Pn, -1, :4]
                            neighbor_current_mask_1 = torch.sum(torch.ne(neighbors_current_1[..., :4], 0), dim=-1) == 0

                            # Forward-noise montage (q(x_t|x0)).
                            base_noise = torch.randn_like(x0n_masked_1[:, :, 1:, :])
                            t_forward = [0.01, 0.1, 0.5, 1.0]
                            forward_panels: list[Any] = []
                            for t_val in t_forward:
                                t_vis = torch.full((1,), float(t_val), device=device, dtype=torch.float32)
                                mean, std = model.sde.marginal_prob(x0n_masked_1[:, :, 1:, :], t_vis)
                                xt_fut = mean + std * base_noise
                                xt_vis = torch.cat([x0n_masked_1[:, :, :1, :], xt_fut], dim=2)
                                xt_inv = model.config.state_normalizer.inverse(xt_vis)
                                xt_xy = xt_inv[0, 0, 1:, :2]
                                img = render_xy_scatter_with_context(
                                    batch1,
                                    sample_idx=0,
                                    xy=xt_xy,
                                    title=f"forward xt | t={t_val:.2f}",
                                    image_size=tb_image_size,
                                    marker="x",
                                    alpha=0.85,
                                    connect_line=False,
                                )
                                forward_panels.append(img)
                            forward_m = stitch_montage(forward_panels, rows=2, cols=2)
                            if forward_m is not None:
                                tb.add_image(
                                    f"forward/{city}/sample_{slot}",
                                    torch.from_numpy(forward_m).permute(2, 0, 1),
                                    int(step),
                                )

                            # Sampler-consistent denoise + viz.
                            if tb_denoise_mode in ["sampler", "all"]:
                                try:
                                    current_states_norm = x0n_masked_1[:, :, 0, :]
                                    P = int(current_states_norm.shape[1])
                                    future_noise = torch.randn((1, P, Tf, 4), device=device, dtype=torch.float32) * 0.5
                                    xT = torch.cat([current_states_norm[:, :, None, :], future_noise], dim=2).reshape(1, P, -1)

                                    def initial_state_constraint(xt: torch.Tensor, t: torch.Tensor, step_i: int):
                                        xt2 = xt.reshape(1, P, Tf + 1, 4)
                                        xt2[:, :, 0, :] = current_states_norm
                                        return xt2.reshape(1, P, -1)

                                    noise_schedule = dpm.NoiseScheduleVP(schedule="linear")
                                    model_fn = dpm.model_wrapper(
                                        model.decoder.decoder.dit,
                                        noise_schedule,
                                        model_type=model.decoder.decoder.dit.model_type,
                                        model_kwargs={
                                            "cross_c": enc_vis["encoding"],
                                            "route_lanes": enc_inputs_vis.get("route_lanes", batch1["route_lanes"]),
                                            "neighbor_current_mask": neighbor_current_mask_1,
                                        },
                                        guidance_type="uncond",
                                    )
                                    dpm_solver = dpm.DPM_Solver(
                                        model_fn,
                                        noise_schedule,
                                        algorithm_type="dpmsolver++",
                                        correcting_xt_fn=initial_state_constraint,
                                    )

                                    x0_flat, inter = dpm_solver.sample(
                                        xT,
                                        steps=tb_sampler_steps,
                                        order=2,
                                        skip_type="logSNR",
                                        method="multistep",
                                        denoise_to_zero=True,
                                        return_intermediate=True,
                                    )

                                    t0 = 1.0 / float(noise_schedule.total_N)
                                    timesteps = dpm_solver.get_time_steps(
                                        skip_type="logSNR",
                                        t_T=float(noise_schedule.T),
                                        t_0=float(t0),
                                        N=tb_sampler_steps,
                                        device=device,
                                    )
                                    t_labels = [float(timesteps[0].item())]
                                    for j in range(1, int(timesteps.shape[0])):
                                        t_labels.append(float(timesteps[j].item()))
                                    if len(inter) == len(t_labels) + 1:
                                        t_labels.append(float(t0))

                                    # Pick 4 snapshots close to t≈[1.0,0.5,0.1,t0].
                                    targets = [1.0, 0.5, 0.1, float(t0)]
                                    idxs = []
                                    for tt in targets:
                                        best_i, best_d = 0, 1e9
                                        for i, t_i in enumerate(t_labels[: len(inter)]):
                                            d = abs(float(t_i) - float(tt))
                                            if d < best_d:
                                                best_d = d
                                                best_i = i
                                        idxs.append(best_i)
                                    uniq = []
                                    for i in idxs:
                                        if i not in uniq:
                                            uniq.append(i)

                                    denoise_panels: list[Any] = []
                                    for i in uniq[:4]:
                                        xt_view = inter[i].reshape(1, P, Tf + 1, 4)
                                        xt_inv = model.config.state_normalizer.inverse(xt_view)
                                        xt_xy = xt_inv[0, 0, 1:, :2]
                                        t_i = t_labels[i] if i < len(t_labels) else float("nan")
                                        img = render_xy_scatter_with_context(
                                            batch1,
                                            sample_idx=0,
                                            xy=xt_xy,
                                            title=f"denoise xt | t={t_i:.3f}",
                                            image_size=tb_image_size,
                                            marker="x",
                                            alpha=0.9,
                                            connect_line=False,
                                        )
                                        denoise_panels.append(img)
                                    while len(denoise_panels) < 4:
                                        denoise_panels.append(None)
                                    denoise_m = stitch_montage(denoise_panels[:4], rows=2, cols=2)
                                    if denoise_m is not None:
                                        tb.add_image(
                                            f"denoise/{city}/sample_{slot}",
                                            torch.from_numpy(denoise_m).permute(2, 0, 1),
                                            int(step),
                                        )

                                    x0_view = x0_flat.reshape(1, P, Tf + 1, 4)
                                    x0_inv = model.config.state_normalizer.inverse(x0_view)
                                    pred_xy = x0_inv[0, 0, 1:, :2]
                                    scene_img = render_npz_style_scene(
                                        batch1,
                                        sample_idx=0,
                                        ego_future_xy=pred_xy,
                                        title=f"viz | step={step} city={city} slot={slot}",
                                        image_size=tb_image_size,
                                    )
                                    if scene_img is not None:
                                        tb.add_image(
                                            f"viz/{city}/sample_{slot}",
                                            torch.from_numpy(scene_img).permute(2, 0, 1),
                                            int(step),
                                        )
                                except Exception as e:
                                    print(f"[tb] sampler denoise logging failed: {e}", flush=True)

                tb.flush()
            except Exception as e:
                if not tb_warned_disabled:
                    print(f"[tb] image logging warning: {e}", flush=True)
                    tb_warned_disabled = True
            finally:
                if model_was_training:
                    model.train(True)

        if (step % cfg.ckpt_every) == 0 or step == cfg.steps - 1:
            ckpt_path = exp_dir / f"checkpoint_step_{step:06d}.pt"
            payload = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "cfg": asdict(cfg),
            }
            payload.update(model.ckpt_payload())
            torch.save(payload, ckpt_path)

        step += 1

    # final fast-val / fast-eval at step=cfg.steps
    # For cfg.steps == 0, we already ran the initial eval at step 0, so skip the final one.
    if int(cfg.steps) > 0:
        if fast_val_loader is not None and fast_val_every > 0:
            _run_fast_val(int(cfg.steps))

        if fast_eval_loaders is not None and fast_eval_every > 0:
            _run_fast_eval(int(cfg.steps))

    latest = exp_dir / "checkpoint_latest.pt"
    # If no checkpoint was written (e.g. cfg.steps==0), prefer linking to resume_ckpt.
    if ckpt_path is None:
        resume_ckpt = getattr(cfg, "resume_ckpt", None)
        if resume_ckpt:
            try:
                ckpt_path = Path(str(resume_ckpt)).expanduser().resolve()
            except Exception:
                ckpt_path = None

    if ckpt_path is not None and Path(ckpt_path).exists():
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(os.path.basename(str(ckpt_path)))
        except Exception:
            import shutil

            shutil.copy2(str(ckpt_path), latest)

    perf.write_perf_json()

    if tb is not None:
        try:
            tb.flush()
            tb.close()
        except Exception:
            pass

    return exp_dir
