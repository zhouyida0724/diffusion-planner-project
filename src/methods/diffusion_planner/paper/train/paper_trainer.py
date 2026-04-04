from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
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
from .tb_visualizer import render_npz_style_scene, render_xy_scatter, render_xy_scatter_with_context


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

    tb = None
    tb_enable = bool(getattr(cfg, "tb_enable", False))
    tb_every = int(getattr(cfg, "tb_every", 0) or 0)
    tb_num_samples = max(1, int(getattr(cfg, "tb_num_samples", 1) or 1))
    tb_denoise_k = max(1, int(getattr(cfg, "tb_denoise_k", 10) or 1))
    tb_denoise_mode = str(getattr(cfg, "tb_denoise_mode", "t_sweep") or "t_sweep")
    tb_sampler_steps = max(1, int(getattr(cfg, "tb_sampler_steps", tb_denoise_k) or tb_denoise_k))
    tb_image_size = int(getattr(cfg, "tb_image_size", 800) or 800)
    tb_warned_disabled = False
    if tb_enable and SummaryWriter is not None:
        try:
            tb = SummaryWriter(log_dir=str(exp_dir / "tb"))
            tb.add_text("meta/exp_name", str(cfg.exp_name), 0)
            tb.add_text("meta/mode", "paper_dit_dpm", 0)
            tb.add_text("meta/tb_image_logging", f"enabled every={tb_every} samples={tb_num_samples} denoise_k={tb_denoise_k}", 0)
        except Exception as e:
            print(f"[tb] disabled: failed to initialize SummaryWriter: {e}", flush=True)
            tb = None
    elif tb_enable and SummaryWriter is None:
        print("[tb] disabled: torch.utils.tensorboard is unavailable in this environment", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

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

    model.train()
    t0 = time.time()
    step = 0
    loader_it = iter(train_loader)

    Pn = int(model.config.predicted_neighbor_num)
    Tf = int(model.config.future_len)

    # -----------------
    # fast validation
    # -----------------
    fast_val_every = int(fast_val_every or 0)
    fast_val_path = exp_dir / "fast_val.jsonl"

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
                tb.add_scalar("fast_val/val_loss_proxy", float(metrics.get("val_loss_proxy", 0.0)), int(step_i))
                for k, v in metrics.items():
                    if k.startswith(("ade_", "fde_")):
                        tb.add_scalar(f"fast_val/{k}", float(v), int(step_i))
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

                B_vis = int(batch["ego_current_state"].shape[0])
                num_vis = min(tb_num_samples, B_vis)

                # Pick random samples from the current batch (deterministic-ish per step).
                g = torch.Generator(device="cpu")
                g.manual_seed(int(cfg.seed) + int(step) * 10007)
                vis_idxs = torch.randperm(B_vis, generator=g)[:num_vis].tolist()

                for sample_idx in vis_idxs:
                    # --- scene image (npz-style) ---
                    infer_inputs = {
                        "ego_current_state": batch["ego_current_state"][sample_idx : sample_idx + 1],
                        "neighbor_agents_past": batch["neighbor_agents_past"][sample_idx : sample_idx + 1],
                        "static_objects": batch["static_objects"][sample_idx : sample_idx + 1],
                        "lanes": batch["lanes"][sample_idx : sample_idx + 1],
                        "lanes_speed_limit": batch["lanes_speed_limit"][sample_idx : sample_idx + 1],
                        "lanes_has_speed_limit": batch["lanes_has_speed_limit"][sample_idx : sample_idx + 1],
                        "route_lanes": batch["route_lanes"][sample_idx : sample_idx + 1],
                        "route_lanes_speed_limit": batch.get("route_lanes_speed_limit")[sample_idx : sample_idx + 1] if torch.is_tensor(batch.get("route_lanes_speed_limit")) else batch.get("route_lanes_speed_limit"),
                        "route_lanes_has_speed_limit": batch.get("route_lanes_has_speed_limit")[sample_idx : sample_idx + 1] if torch.is_tensor(batch.get("route_lanes_has_speed_limit")) else batch.get("route_lanes_has_speed_limit"),
                    }

                    # Best-effort inference trajectory for overlay.
                    pred_xy = None
                    try:
                        pred_xy = model.sample_trajectory(infer_inputs, diffusion_steps=max(tb_denoise_k, 4))[:, :2]
                    except Exception:
                        pred_xy = None

                    scene_img = render_npz_style_scene(
                        batch,
                        sample_idx=sample_idx,
                        ego_future_xy=pred_xy,
                        title=f"step={step} sample={sample_idx}",
                        image_size=tb_image_size,
                    )
                    if scene_img is not None:
                        tb.add_image(
                            f"viz/scene/sample_{sample_idx}",
                            torch.from_numpy(scene_img).permute(2, 0, 1),
                            int(step),
                        )

                    # --- denoise scatter (xt + x0_pred), logged as multiple images ---
                    # Build a B=1 batch for faster visualization compute.
                    batch1: dict[str, Any] = {}
                    for k, v in batch.items():
                        if torch.is_tensor(v) and v.shape[0] == B_vis:
                            batch1[k] = v[sample_idx : sample_idx + 1]
                        else:
                            batch1[k] = v

                    with torch.no_grad():
                        x0_4_1, _ = _build_joint_trajectories_x0(batch1, predicted_neighbor_num=Pn, future_len=Tf)
                        x0n_1 = model.config.state_normalizer(x0_4_1)
                        neighbor_mask_full_1, _ = _compute_neighbor_masks(batch1, predicted_neighbor_num=Pn, future_len=Tf)
                        x0n_masked_1 = x0n_1.clone()
                        x0n_masked_1[:, 1:, :, :][neighbor_mask_full_1] = 0.0

                        # Fixed noise is nice-to-have for smoother convergence visuals, but torch 2.1
                        # doesn't support passing a per-call Generator to randn_like on all devices.
                        # Use a fresh random noise tensor here (still shows convergence clearly).
                        base_noise = torch.randn_like(x0n_masked_1[:, :, 1:, :])

                        # Include an explicit t=1.0 panel (max noise) and go close to 0 for the final panel.
                        t_values = torch.linspace(1.0, 0.01, steps=tb_denoise_k, device=device, dtype=torch.float32)
                        t_values_forward = torch.linspace(0.01, 1.0, steps=tb_denoise_k, device=device, dtype=torch.float32)

                        # Precompute encoder outputs once for this sample.
                        enc_inputs_vis = {
                            "ego_current_state": batch1["ego_current_state"],
                            "neighbor_agents_past": batch1["neighbor_agents_past"],
                            "static_objects": batch1["static_objects"],
                            "lanes": batch1["lanes"],
                            "lanes_speed_limit": batch1["lanes_speed_limit"],
                            "lanes_has_speed_limit": batch1["lanes_has_speed_limit"],
                            "route_lanes": batch1["route_lanes"],
                            "route_lanes_speed_limit": batch1.get("route_lanes_speed_limit"),
                            "route_lanes_has_speed_limit": batch1.get("route_lanes_has_speed_limit"),
                        }
                        enc_vis = model.encoder(enc_inputs_vis)

                        # Build neighbor current mask (same as Decoder.forward training branch)
                        neighbors_past_1 = batch1["neighbor_agents_past"]
                        if neighbors_past_1.shape[1] == Pn + 1:
                            neighbors_current_1 = neighbors_past_1[:, 1 : 1 + Pn, -1, :4]
                        else:
                            neighbors_current_1 = neighbors_past_1[:, :Pn, -1, :4]
                        neighbor_current_mask_1 = torch.sum(torch.ne(neighbors_current_1[..., :4], 0), dim=-1) == 0

                        # ------------------------------
                        # (A) t-sweep panels (training-style)
                        # ------------------------------
                        if tb_denoise_mode in ["t_sweep", "all"]:
                            for i, t_scalar in enumerate(t_values):
                                t_val = float(t_scalar.item())
                                t_vis = torch.full((1,), t_val, device=device, dtype=torch.float32)
                                mean, std = model.sde.marginal_prob(x0n_masked_1[:, :, 1:, :], t_vis)
                                xt_fut = mean + std * base_noise
                                xt_vis = torch.cat([x0n_masked_1[:, :, :1, :], xt_fut], dim=2)

                                with autocast_ctx:
                                    out_flat = model.decoder.decoder.dit(
                                        xt_vis.reshape(1, 1 + Pn, -1),
                                        t_vis,
                                        enc_vis["encoding"],
                                        batch1["route_lanes"],
                                        neighbor_current_mask_1,
                                    )
                                x0_pred_vis = out_flat.reshape(1, 1 + Pn, -1, 4)  # normalized x0

                                xt_inv = model.config.state_normalizer.inverse(xt_vis)
                                x0_pred_inv = model.config.state_normalizer.inverse(x0_pred_vis)

                                xt_xy = xt_inv[0, 0, 1:, :2]
                                x0_pred_xy = x0_pred_inv[0, 0, 1:, :2]

                                xt_img = render_xy_scatter_with_context(
                                    batch1,
                                    sample_idx=0,
                                    xy=xt_xy,
                                    title=f"xt | step={step} sample={sample_idx} panel={i:02d} t={t_val:.2f}",
                                    image_size=tb_image_size,
                                    marker="x",
                                    alpha=0.85,
                                )
                                if xt_img is not None:
                                    tb.add_image(
                                        f"denoise/xt/sample_{sample_idx}/panel_{i:02d}",
                                        torch.from_numpy(xt_img).permute(2, 0, 1),
                                        int(step),
                                    )

                                x0_img = render_xy_scatter_with_context(
                                    batch1,
                                    sample_idx=0,
                                    xy=x0_pred_xy,
                                    title=f"x0_pred | step={step} sample={sample_idx} panel={i:02d} t={t_val:.2f}",
                                    image_size=tb_image_size,
                                    marker=".",
                                    alpha=0.95,
                                )
                                if x0_img is not None:
                                    tb.add_image(
                                        f"denoise/x0_pred/sample_{sample_idx}/panel_{i:02d}",
                                        torch.from_numpy(x0_img).permute(2, 0, 1),
                                        int(step),
                                    )

                        # ------------------------------
                        # (B) forward-noise panels (q(x_t | x0), low->high noise)
                        # ------------------------------
                        if tb_denoise_mode in ["forward_noise", "all"]:
                            for i, t_scalar in enumerate(t_values_forward):
                                t_val = float(t_scalar.item())
                                t_tag = _t_to_tag(t_val)
                                t_vis = torch.full((1,), t_val, device=device, dtype=torch.float32)
                                mean, std = model.sde.marginal_prob(x0n_masked_1[:, :, 1:, :], t_vis)
                                xt_fut = mean + std * base_noise
                                xt_vis = torch.cat([x0n_masked_1[:, :, :1, :], xt_fut], dim=2)

                                xt_inv = model.config.state_normalizer.inverse(xt_vis)
                                xt_xy = xt_inv[0, 0, 1:, :2]

                                xt_img = render_xy_scatter_with_context(
                                    batch1,
                                    sample_idx=0,
                                    xy=xt_xy,
                                    title=f"forward-noise xt | step={step} sample={sample_idx} panel={i:02d}/{len(t_values_forward)-1} t={t_val:.2f}",
                                    image_size=tb_image_size,
                                    marker="x",
                                    alpha=0.85,
                                )
                                if xt_img is not None:
                                    tb.add_image(
                                        f"forward_noise/xt/sample_{sample_idx}/panel_{i:02d}_t_{t_tag}",
                                        torch.from_numpy(xt_img).permute(2, 0, 1),
                                        int(step),
                                    )

                        # ------------------------------
                        # (C) sampler intermediates (true iterative denoising)
                        # ------------------------------
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
                                        "route_lanes": batch1["route_lanes"],
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

                                for i, xt_flat in enumerate(inter):
                                    t_i = t_labels[i] if i < len(t_labels) else float("nan")
                                    t_tag = _t_to_tag(t_i) if t_i == t_i else "nan"
                                    xt_view = xt_flat.reshape(1, P, Tf + 1, 4)
                                    xt_inv = model.config.state_normalizer.inverse(xt_view)
                                    xt_xy = xt_inv[0, 0, 1:, :2]

                                    xt_img = render_xy_scatter_with_context(
                                        batch1,
                                        sample_idx=0,
                                        xy=xt_xy,
                                        title=f"sampler xt | step={step} sample={sample_idx} solver_step={i:02d}/{len(inter)-1} t={t_i:.4f}",
                                        image_size=tb_image_size,
                                        marker="x",
                                        alpha=0.9,
                                    )
                                    if xt_img is not None:
                                        tb.add_image(
                                            f"sampler/xt/sample_{sample_idx}/solver_{i:02d}_t_{t_tag}",
                                            torch.from_numpy(xt_img).permute(2, 0, 1),
                                            int(step),
                                        )
                            except Exception as e:
                                print(f"[tb] sampler-intermediate logging failed: {e}", flush=True)

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

    # final fast-val at step=cfg.steps
    if fast_val_loader is not None and fast_val_every > 0:
        _run_fast_val(int(cfg.steps))

    latest = exp_dir / "checkpoint_latest.pt"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(os.path.basename(ckpt_path))
    except Exception:
        import shutil

        shutil.copy2(ckpt_path, latest)

    perf.write_perf_json()

    if tb is not None:
        try:
            tb.flush()
            tb.close()
        except Exception:
            pass

    return exp_dir
