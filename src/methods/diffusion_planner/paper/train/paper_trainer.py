from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.methods.diffusion_planner.train.trainer import TrainConfig, _assert_finite, _maybe_sync, _PerfTracker, seed_everything
from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner


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


def train_loop_paper_dit_xstart(
    *,
    cfg: TrainConfig,
    model: PaperDiffusionPlanner,
    train_loader: DataLoader,
    max_grad_norm: Optional[float] = 1.0,
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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    perf = _PerfTracker(exp_dir=exp_dir, cfg=cfg, device=device)

    model.train()
    t0 = time.time()
    step = 0
    loader_it = iter(train_loader)

    Pn = int(model.config.predicted_neighbor_num)
    Tf = int(model.config.future_len)

    while step < cfg.steps:
        step_t0 = time.time()
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(train_loader)
            batch = next(loader_it)

        # move tensors
        batch = {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v) for k, v in batch.items()}

        x0_4, current_states = _build_joint_trajectories_x0(batch, predicted_neighbor_num=Pn, future_len=Tf)
        _assert_finite(x0_4, "x0")

        # normalize
        x0n = model.config.state_normalizer(x0_4)

        B = x0n.shape[0]
        t = torch.rand((B,), device=device, dtype=torch.float32)
        noise = torch.randn_like(x0n)

        mean, std = model.sde.marginal_prob(x0n, t)
        xt = mean + std * noise
        # hard constraint current state
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
        pred = dec_out["score"]  # [B,P,1+T,4]
        _assert_finite(pred, "pred")

        loss = torch.mean((pred - x0n) ** 2)
        _assert_finite(loss, "loss")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

        _maybe_sync(device)
        step_s = time.time() - step_t0
        perf.on_step_end(step=step, step_s=step_s, loss=float(loss.item()))

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

    latest = exp_dir / "checkpoint_latest.pt"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(os.path.basename(ckpt_path))
    except Exception:
        import shutil

        shutil.copy2(ckpt_path, latest)

    perf.write_perf_json()
    return exp_dir
