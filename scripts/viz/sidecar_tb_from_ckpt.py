#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.methods.diffusion_planner.paper.model.diffusion_utils import dpm_solver_pytorch as dpm

from src.methods.diffusion_planner.data.feature_npz_dataset import ShardedNpzFeatureDataset
from src.methods.diffusion_planner.paper.model.diffusion_planner import PaperDiffusionPlanner, PaperModelConfig
from src.methods.diffusion_planner.paper.train.paper_trainer import _build_joint_trajectories_x0, _compute_neighbor_masks
from src.methods.diffusion_planner.paper.train.tb_visualizer import render_npz_style_scene, render_xy_scatter_with_context


def _t_to_tag(t: float) -> str:
    """Stable tag for time value.

    Keep higher precision near 0 so t=0.001 doesn't collapse to 0.00.
    """

    t = float(t)
    if t < 0.01:
        s = f"{t:.3f}"
    else:
        s = f"{t:.2f}"
    return s.replace(".", "p")


def _paper_cfg_from_ckpt(payload: dict[str, Any]) -> PaperModelConfig:
    raw = dict(payload.get("paper_config") or {})
    allowed = {f.name for f in fields(PaperModelConfig)}
    clean = {k: raw[k] for k in list(raw.keys()) if k in allowed}
    return PaperModelConfig(**clean)


def _to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, dtype=torch.float32)
        else:
            out[k] = v
    return out


def _batch_size1(batch: dict[str, Any], idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] > idx:
            out[k] = v[idx : idx + 1]
        else:
            out[k] = v
    return out


def _batchify_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a single dataset sample (no batch dim) into a batch of size 1."""

    out: dict[str, Any] = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out


def _turning_score(sample: dict[str, Any]) -> float:
    """Heuristic score for "hard" scenes: ego turning + more neighbors.

    Uses only fields available in ShardedNpzFeatureDataset.
    """

    score = 0.0
    fut = sample.get("ego_agent_future")
    if torch.is_tensor(fut) and fut.ndim >= 2 and fut.shape[0] >= 5:
        xy = fut[:, :2]
        dxy = xy[1:] - xy[:-1]
        # avoid zero-length steps
        keep = (dxy.abs().sum(dim=-1) > 1e-4)
        dxy = dxy[keep]
        if dxy.shape[0] >= 3:
            headings = torch.atan2(dxy[:, 1], dxy[:, 0])
            # unwrap roughly
            dh = headings[1:] - headings[:-1]
            dh = (dh + torch.pi) % (2 * torch.pi) - torch.pi
            total_turn = float(dh.abs().sum().item())
            max_turn = float(dh.abs().max().item())
            score += 2.0 * total_turn + 1.0 * max_turn

        # prefer bigger motion (typically intersections are in motion)
        disp = float(torch.norm(xy[-1] - xy[0]).item())
        score += 0.02 * disp

    nb = sample.get("neighbor_agents_past")
    if torch.is_tensor(nb) and nb.ndim >= 3:
        cur = nb[:, -1, :4]
        present = (cur.abs().sum(dim=-1) > 1e-4)
        score += 0.1 * float(present.sum().item())

    return float(score)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Checkpoint sidecar TensorBoard visualizer")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--logdir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--slice", default=None, help="Optional slice name appended under --data-root")
    p.add_argument("--cache-root", default="outputs/cache/training_arrays")
    p.add_argument("--k", type=int, default=10, help="Number of t-sweep panels (training-style forward-noise visualization)")
    p.add_argument("--sampler-steps", type=int, default=10, help="Number of DPM-Solver steps for sampler-intermediate visualization")
    p.add_argument("--image-size", type=int, default=800)
    p.add_argument("--sample-idx", type=int, default=0, help="Start sample index")
    p.add_argument("--num-samples", type=int, default=1, help="How many samples to visualize")

    # sample selection
    p.add_argument(
        "--select",
        type=str,
        default="sequential",
        choices=["sequential", "turning"],
        help="How to choose samples: sequential (sample_idx..), or turning (search a window and pick turning scenes)",
    )
    p.add_argument(
        "--search-window",
        type=int,
        default=2000,
        help="When --select=turning, scan this many samples starting from --sample-idx and pick top-K by score.",
    )

    # visualizations
    p.add_argument("--no-t-sweep", action="store_true", help="Disable the t-sweep (x_t/x0_pred across t)")
    p.add_argument(
        "--log-forward-noise",
        action="store_true",
        help="Also log a forward noising sequence (t asc: low->high) to visualize the add-noise process.",
    )

    p.add_argument("--no-sampler", action="store_true", help="Disable sampler-intermediate panels")
    p.add_argument("--sampler-log-x0pred", action="store_true", help="Also log x0_pred at each sampler intermediate xt")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    logdir = Path(args.logdir)
    data_root = Path(args.data_root)
    if args.slice:
        data_root = data_root / args.slice

    device = torch.device("cpu")
    payload = torch.load(ckpt_path, map_location="cpu")
    step = int(payload.get("step", -1))

    paper_cfg = _paper_cfg_from_ckpt(payload)
    paper_cfg.device = "cpu"

    model = PaperDiffusionPlanner(paper_cfg)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()

    ds = ShardedNpzFeatureDataset(data_root, cache_root=args.cache_root)

    start_idx = int(args.sample_idx)
    num_samples = max(1, int(getattr(args, "num_samples", 1) or 1))
    if start_idx < 0 or start_idx >= len(ds):
        raise IndexError(f"sample-idx {start_idx} out of range for dataset len={len(ds)}")

    selected_indices: list[int] = []
    if str(getattr(args, "select", "sequential")) == "turning":
        win = max(1, int(getattr(args, "search_window", 2000) or 2000))
        end_scan = min(len(ds), start_idx + win)
        # scan and pick top-K scores
        scored: list[tuple[float, int]] = []
        for idx in range(start_idx, end_scan):
            try:
                s = ds[idx]
                scored.append((_turning_score(s), idx))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        selected_indices = [idx for _, idx in scored[:num_samples]]
    else:
        end_idx = min(len(ds), start_idx + num_samples)
        selected_indices = list(range(start_idx, end_idx))

    Pn = int(paper_cfg.predicted_neighbor_num)
    Tf = int(paper_cfg.future_len)

    logdir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(logdir))

    tb.add_text("meta/ckpt", str(ckpt_path), step)
    tb.add_text("meta/data_root", str(data_root), step)
    tb.add_text("meta/selection", str(getattr(args, "select", "sequential")), step)
    tb.add_text("meta/selected_indices", str(selected_indices), step)

    for sample_idx in selected_indices:
        sample = ds[sample_idx]
        batch = _batchify_sample(sample)
        batch = _to_device_batch(batch, device)
        batch1 = _batch_size1(batch, 0)

        infer_inputs = {
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

        pred_xy = None
        try:
            pred_xy = model.sample_trajectory(infer_inputs, diffusion_steps=max(int(args.k), 4))[:, :2]
        except Exception as e:
            print(f"[warn] sample_trajectory failed (sample_idx={sample_idx}): {e}", flush=True)

        x0_4, _ = _build_joint_trajectories_x0(batch1, predicted_neighbor_num=Pn, future_len=Tf)
        x0n = model.config.state_normalizer(x0_4)
        neighbor_mask_full, _ = _compute_neighbor_masks(batch1, predicted_neighbor_num=Pn, future_len=Tf)
        x0n_masked = x0n.clone()
        x0n_masked[:, 1:, :, :][neighbor_mask_full] = 0.0

        base_noise = torch.randn_like(x0n_masked[:, :, 1:, :])
        t_values = torch.linspace(1.0, 0.01, steps=max(1, int(args.k)), device=device, dtype=torch.float32)
        t_values_forward = torch.linspace(0.01, 1.0, steps=max(1, int(args.k)), device=device, dtype=torch.float32)

        enc_vis = model.encoder(infer_inputs)

        meta = batch1.get("meta")
        token = None
        if isinstance(meta, dict):
            token = meta.get("token") or meta.get("scene_token")

        tb.add_text(f"meta/sample_{sample_idx}/sample_meta", str(meta), step)

        prefix = f"ckpt_{step:06d}/sample_{sample_idx}"
        if token is not None:
            prefix = prefix + f"/token_{token}"

        scene_img = render_npz_style_scene(
            batch1,
            sample_idx=0,
            ego_future_xy=pred_xy,
            title=f"ckpt_step={step} sample_idx={sample_idx} token={token}",
            image_size=int(args.image_size),
        )
        if scene_img is not None:
            tb.add_image(f"viz/scene/{prefix}", torch.from_numpy(scene_img).permute(2, 0, 1), step)

        neighbors_past = batch1["neighbor_agents_past"]
        if neighbors_past.shape[1] == Pn + 1:
            neighbors_current = neighbors_past[:, 1 : 1 + Pn, -1, :4]
        else:
            neighbors_current = neighbors_past[:, :Pn, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0

        if not bool(args.no_t_sweep):
            for i, t_scalar in enumerate(t_values):
                t_val = float(t_scalar.item())
                t_tag = _t_to_tag(t_val)

                t_vis = torch.full((1,), t_val, device=device, dtype=torch.float32)
                mean, std = model.sde.marginal_prob(x0n_masked[:, :, 1:, :], t_vis)
                xt_fut = mean + std * base_noise
                xt_vis = torch.cat([x0n_masked[:, :, :1, :], xt_fut], dim=2)

                out_flat = model.decoder.decoder.dit(
                    xt_vis.reshape(1, 1 + Pn, -1),
                    t_vis,
                    enc_vis["encoding"],
                    batch1["route_lanes"],
                    neighbor_current_mask,
                )
                x0_pred_vis = out_flat.reshape(1, 1 + Pn, -1, 4)

                xt_inv = model.config.state_normalizer.inverse(xt_vis)
                x0_pred_inv = model.config.state_normalizer.inverse(x0_pred_vis)
                xt_xy = xt_inv[0, 0, 1:, :2]
                x0_pred_xy = x0_pred_inv[0, 0, 1:, :2]

                xt_img = render_xy_scatter_with_context(
                    batch1,
                    sample_idx=0,
                    xy=xt_xy,
                    title=f"t-sweep xt (hi->lo noise) | ckpt={step} sample={sample_idx} panel={i:02d}/{len(t_values)-1} t={t_val:.2f}",
                    image_size=int(args.image_size),
                    marker="x",
                    alpha=0.85,
                )
                if xt_img is not None:
                    tb.add_image(
                        f"t_sweep/xt/{prefix}/panel_{i:02d}_t_{t_tag}",
                        torch.from_numpy(xt_img).permute(2, 0, 1),
                        step,
                    )

                x0_img = render_xy_scatter_with_context(
                    batch1,
                    sample_idx=0,
                    xy=x0_pred_xy,
                    title=f"t-sweep x0_pred | ckpt={step} sample={sample_idx} panel={i:02d}/{len(t_values)-1} t={t_val:.2f}",
                    image_size=int(args.image_size),
                    marker=".",
                    alpha=0.95,
                )
                if x0_img is not None:
                    tb.add_image(
                        f"t_sweep/x0_pred/{prefix}/panel_{i:02d}_t_{t_tag}",
                        torch.from_numpy(x0_img).permute(2, 0, 1),
                        step,
                    )

        if bool(getattr(args, "log_forward_noise", False)):
            # This is NOT sampling; it is q(x_t | x0) forward noising for intuition.
            for i, t_scalar in enumerate(t_values_forward):
                t_val = float(t_scalar.item())
                t_tag = _t_to_tag(t_val)

                t_vis = torch.full((1,), t_val, device=device, dtype=torch.float32)
                mean, std = model.sde.marginal_prob(x0n_masked[:, :, 1:, :], t_vis)
                xt_fut = mean + std * base_noise
                xt_vis = torch.cat([x0n_masked[:, :, :1, :], xt_fut], dim=2)

                xt_inv = model.config.state_normalizer.inverse(xt_vis)
                xt_xy = xt_inv[0, 0, 1:, :2]

                xt_img = render_xy_scatter_with_context(
                    batch1,
                    sample_idx=0,
                    xy=xt_xy,
                    title=f"forward-noise xt (lo->hi noise) | ckpt={step} sample={sample_idx} panel={i:02d}/{len(t_values_forward)-1} t={t_val:.2f}",
                    image_size=int(args.image_size),
                    marker="x",
                    alpha=0.85,
                )
                if xt_img is not None:
                    tb.add_image(
                        f"forward_noise/xt/{prefix}/panel_{i:02d}_t_{t_tag}",
                        torch.from_numpy(xt_img).permute(2, 0, 1),
                        step,
                    )

        # ------------------------------
        # Sampler intermediates (true iterative denoising)
        # ------------------------------
        if not bool(args.no_sampler):
            try:
                # Work in normalized trajectory space (as in training): x0n_masked contains current+future in normalized units.
                current_states_norm = x0n_masked[:, :, 0, :]  # [1,P,4]
                P = int(current_states_norm.shape[1])
    
                # Start from pure noise for the future; keep current fixed via correcting_xt_fn.
                future_noise = torch.randn((1, P, Tf, 4), device=device, dtype=torch.float32) * 0.5
                xT = torch.cat([current_states_norm[:, :, None, :], future_noise], dim=2).reshape(1, P, -1)  # [1,P,(1+T)*4]
    
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
                        "neighbor_current_mask": neighbor_current_mask,
                    },
                    guidance_type="uncond",
                )
                dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++", correcting_xt_fn=initial_state_constraint)
    
                n_steps = int(args.sampler_steps)
                x0_flat, inter = dpm_solver.sample(
                    xT,
                    steps=n_steps,
                    order=2,
                    skip_type="logSNR",
                    method="multistep",
                    denoise_to_zero=True,
                    return_intermediate=True,
                )
    
                # Build t labels consistent with DPM-Solver timesteps.
                t0 = 1.0 / float(noise_schedule.total_N)
                timesteps = dpm_solver.get_time_steps(skip_type="logSNR", t_T=float(noise_schedule.T), t_0=float(t0), N=n_steps, device=device)
                # timesteps: [n_steps+1] from T -> t0
                t_labels = [float(timesteps[0].item())]
                for i in range(1, int(timesteps.shape[0])):
                    t_labels.append(float(timesteps[i].item()))
                if len(inter) == len(t_labels) + 1:
                    # extra denoise_to_zero panel
                    t_labels.append(float(t0))
    
                sampler_base = f"sampler/xt/{prefix}"
                for i, xt_flat in enumerate(inter):
                    t_i = t_labels[i] if i < len(t_labels) else float('nan')
                    t_tag = _t_to_tag(t_i) if t_i == t_i else "nan"
    
                    xt_view = xt_flat.reshape(1, P, Tf + 1, 4)
                    xt_inv = model.config.state_normalizer.inverse(xt_view)
                    xt_xy = xt_inv[0, 0, 1:, :2]
    
                    img = render_xy_scatter_with_context(
                        batch1,
                        sample_idx=0,
                        xy=xt_xy,
                        title=f"sampler xt | ckpt={step} sample={sample_idx} solver_step={i:02d}/{len(inter)-1} t={t_i:.4f}",
                        image_size=int(args.image_size),
                        marker="x",
                        alpha=0.9,
                    )
                    if img is not None:
                        tb.add_image(f"{sampler_base}/solver_{i:02d}_t_{t_tag}", torch.from_numpy(img).permute(2, 0, 1), step)
    
                    if bool(args.sampler_log_x0pred):
                        # also log x0_pred at this solver step
                        t_vis = torch.full((1,), float(t_i), device=device, dtype=torch.float32)
                        out_flat = model.decoder.decoder.dit(
                            xt_flat,
                            t_vis,
                            enc_vis["encoding"],
                            batch1["route_lanes"],
                            neighbor_current_mask,
                        )
                        x0_pred = out_flat.reshape(1, P, Tf + 1, 4)
                        x0_pred_inv = model.config.state_normalizer.inverse(x0_pred)
                        x0_xy = x0_pred_inv[0, 0, 1:, :2]
                        img2 = render_xy_scatter_with_context(
                            batch1,
                            sample_idx=0,
                            xy=x0_xy,
                            title=f"sampler x0_pred | ckpt={step} sample={sample_idx} solver_step={i:02d}/{len(inter)-1} t={t_i:.4f}",
                            image_size=int(args.image_size),
                            marker=".",
                            alpha=0.95,
                        )
                        if img2 is not None:
                            tb.add_image(
                                f"sampler/x0_pred/{prefix}/solver_{i:02d}_t_{t_tag}",
                                torch.from_numpy(img2).permute(2, 0, 1),
                                step,
                            )
    
                # final x0 from solver
                x0_view = x0_flat.reshape(1, P, Tf + 1, 4)
                x0_inv = model.config.state_normalizer.inverse(x0_view)
                x0_xy = x0_inv[0, 0, 1:, :2]
                x0_img = render_xy_scatter_with_context(
                    batch1,
                    sample_idx=0,
                    xy=x0_xy,
                    title=f"sampler x0_final | ckpt={step} sample={sample_idx} steps={n_steps}",
                    image_size=int(args.image_size),
                    marker=".",
                    alpha=0.95,
                )
                if x0_img is not None:
                    tb.add_image(f"sampler/x0_final/{prefix}", torch.from_numpy(x0_img).permute(2, 0, 1), step)
    
            except Exception as e:
                print(f"[warn] sampler-intermediate logging failed: {e}", flush=True)
    
    tb.flush()
    tb.close()

    print(f"Wrote TensorBoard sidecar images to {logdir}")
    print(f"checkpoint_step={step}")
    print(f"dataset_len={len(ds)} sample_idx={sample_idx}")
    print(f"data_root={data_root}")
    if isinstance(batch1.get('meta'), dict):
        print(f"sample_meta={batch1['meta']}")


if __name__ == "__main__":
    main()
