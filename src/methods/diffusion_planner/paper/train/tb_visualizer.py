from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x)


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _figure_to_rgb(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()


def render_npz_style_scene(
    batch: dict[str, Any],
    *,
    sample_idx: int,
    ego_future_xy: torch.Tensor | np.ndarray | None = None,
    title: str | None = None,
) -> np.ndarray | None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrowPatch

    ego_state = _to_numpy(batch["ego_current_state"])[sample_idx]
    lanes = _to_numpy(batch["lanes"])[sample_idx]
    route_lanes = _to_numpy(batch["route_lanes"])[sample_idx]
    route_lanes_avails = _to_numpy(batch.get("route_lanes_avails"))[sample_idx] if batch.get("route_lanes_avails") is not None else None
    ego_future = _to_numpy(batch["ego_agent_future"])[sample_idx]
    neighbor_past = _to_numpy(batch["neighbor_agents_past"])[sample_idx]
    neighbor_future = _to_numpy(batch.get("neighbor_agents_future"))[sample_idx] if batch.get("neighbor_agents_future") is not None else None
    static_objects = _to_numpy(batch.get("static_objects"))[sample_idx] if batch.get("static_objects") is not None else None

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    try:
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title or f"sample={sample_idx}")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

        for lane_idx in range(lanes.shape[0]):
            lane_x = lanes[lane_idx, :, 0]
            lane_y = lanes[lane_idx, :, 1]
            if np.allclose(lane_x, 0) and np.allclose(lane_y, 0):
                continue
            left_x = lanes[lane_idx, :, 0] + lanes[lane_idx, :, 4]
            left_y = lanes[lane_idx, :, 1] + lanes[lane_idx, :, 5]
            right_x = lanes[lane_idx, :, 0] + lanes[lane_idx, :, 6]
            right_y = lanes[lane_idx, :, 1] + lanes[lane_idx, :, 7]
            ax.plot(left_x[(left_x != 0) | (left_y != 0)], left_y[(left_x != 0) | (left_y != 0)], "b--", linewidth=1, alpha=0.35)
            ax.plot(right_x[(right_x != 0) | (right_y != 0)], right_y[(right_x != 0) | (right_y != 0)], "r--", linewidth=1, alpha=0.35)

        for rlane_idx in range(route_lanes.shape[0]):
            lane_x = route_lanes[rlane_idx, :, 0]
            lane_y = route_lanes[rlane_idx, :, 1]
            if np.allclose(lane_x, 0) and np.allclose(lane_y, 0):
                continue
            if route_lanes_avails is not None:
                valid = route_lanes_avails[rlane_idx] > 0
            else:
                valid = (lane_x != 0) | (lane_y != 0)
            if not np.any(valid):
                continue
            x_coords = lane_x[valid]
            y_coords = lane_y[valid]
            ax.plot(x_coords, y_coords, "-", color="#FFFF99", alpha=0.9, linewidth=2)

        arrow = FancyArrowPatch((0, 0), (4, 0), arrowstyle="-|>", mutation_scale=15, facecolor="red", edgecolor="darkred", linewidth=2)
        ax.add_patch(arrow)
        ego_circle = patches.Circle((0, 0), radius=1.5, facecolor="red", edgecolor="darkred", linewidth=2)
        ax.add_patch(ego_circle)

        ax.plot(ego_future[:, 0], ego_future[:, 1], "g-", linewidth=2.5, alpha=0.85, label="ego_gt")

        if ego_future_xy is not None:
            pred_xy = _to_numpy(ego_future_xy)
            ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="orange", linewidth=2.0, alpha=0.95, label="ego_pred")

        for agent_idx in range(neighbor_past.shape[0]):
            curr_x = neighbor_past[agent_idx, -1, 0]
            curr_y = neighbor_past[agent_idx, -1, 1]
            past = neighbor_past[agent_idx, :, :2]
            if abs(curr_x) < 0.1 and abs(curr_y) < 0.1:
                continue
            valid_mask = (past[:, 0] != 0) | (past[:, 1] != 0)
            if not np.any(valid_mask):
                continue
            ax.plot(past[valid_mask, 0], past[valid_mask, 1], "b-", linewidth=1.5, alpha=0.55)
            ax.plot(curr_x, curr_y, "o", markersize=5, color="blue", alpha=0.85)

        if neighbor_future is not None:
            for agent_idx in range(neighbor_future.shape[0]):
                future = neighbor_future[agent_idx, :, :2]
                valid_mask = (future[:, 0] != 0) | (future[:, 1] != 0)
                if np.any(valid_mask):
                    ax.plot(future[valid_mask, 0], future[valid_mask, 1], "g-", linewidth=1.2, alpha=0.35)

        if static_objects is not None and static_objects.ndim == 2 and static_objects.shape[-1] >= 2:
            valid = np.any(np.abs(static_objects[:, :2]) > 1e-6, axis=1)
            if np.any(valid):
                ax.scatter(static_objects[valid, 0], static_objects[valid, 1], s=20, c="black", alpha=0.5, marker="x", label="static")

        ax.legend(loc="upper right")
        fig.tight_layout()
        return _figure_to_rgb(fig)
    finally:
        plt.close(fig)


def render_denoise_scatter(
    model: Any,
    batch: dict[str, Any],
    *,
    sample_idx: int,
    num_panels: int,
    predicted_neighbor_num: int,
    future_len: int,
    device: torch.device,
    build_joint_trajectories_x0: Any,
    compute_neighbor_masks: Any,
) -> np.ndarray | None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    with torch.no_grad():
        x0_4, _ = build_joint_trajectories_x0(batch, predicted_neighbor_num=predicted_neighbor_num, future_len=future_len)
        x0n = model.config.state_normalizer(x0_4)
        neighbor_mask_full, _ = compute_neighbor_masks(batch, predicted_neighbor_num=predicted_neighbor_num, future_len=future_len)
        x0n_masked = x0n.clone()
        x0n_masked[:, 1:, :, :][neighbor_mask_full] = 0.0

        B = int(x0n_masked.shape[0])
        base_noise = torch.randn_like(x0n_masked[:, :, 1:, :])
        enc_inputs = {
            "ego_current_state": batch["ego_current_state"],
            "neighbor_agents_past": batch["neighbor_agents_past"],
            "static_objects": batch["static_objects"],
            "lanes": batch["lanes"],
            "lanes_speed_limit": batch["lanes_speed_limit"],
            "lanes_has_speed_limit": batch["lanes_has_speed_limit"],
            "route_lanes": batch["route_lanes"],
            "route_lanes_speed_limit": batch.get("route_lanes_speed_limit"),
            "route_lanes_has_speed_limit": batch.get("route_lanes_has_speed_limit"),
        }
        enc = model.encoder(enc_inputs)

        k = max(1, int(num_panels))
        t_values = torch.linspace(0.95, 0.05, steps=k, device=device, dtype=torch.float32)
        gt_xy = x0_4[sample_idx, 0, 1:, :2].detach().cpu().float().numpy()

        cols = min(5, k)
        rows = int(math.ceil(k / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=100)
        axes = np.array(axes).reshape(-1)
        try:
            for i, t_scalar in enumerate(t_values):
                t = torch.full((B,), float(t_scalar.item()), device=device, dtype=torch.float32)
                mean, std = model.sde.marginal_prob(x0n_masked[:, :, 1:, :], t)
                xt_fut = mean + std * base_noise
                xt = torch.cat([x0n_masked[:, :, :1, :], xt_fut], dim=2)

                neighbors_past = batch["neighbor_agents_past"]
                if neighbors_past.shape[1] == predicted_neighbor_num + 1:
                    neighbors_current = neighbors_past[:, 1 : 1 + predicted_neighbor_num, -1, :4]
                else:
                    neighbors_current = neighbors_past[:, :predicted_neighbor_num, -1, :4]
                neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
                pred = model.decoder.decoder.dit(
                    xt.reshape(B, 1 + predicted_neighbor_num, -1),
                    t,
                    enc["encoding"],
                    batch["route_lanes"],
                    neighbor_current_mask,
                ).reshape(B, 1 + predicted_neighbor_num, -1, 4)
                pred_inv = model.config.state_normalizer.inverse(pred)

                xt_inv = model.config.state_normalizer.inverse(xt)
                xt_xy = xt_inv[sample_idx, 0, 1:, :2].detach().cpu().float().numpy()
                x0_pred_xy = pred_inv[sample_idx, 0, 1:, :2].detach().cpu().float().numpy()

                ax = axes[i]
                time_color = np.linspace(0.1, 1.0, len(gt_xy))
                ax.scatter(xt_xy[:, 0], xt_xy[:, 1], c=time_color, cmap="Greys", s=18, alpha=0.55, marker="x", label="x_t" if i == 0 else None)
                ax.scatter(x0_pred_xy[:, 0], x0_pred_xy[:, 1], c=time_color, cmap="viridis", s=24, alpha=0.9, label="x0_pred" if i == 0 else None)
                ax.scatter(gt_xy[:, 0], gt_xy[:, 1], facecolors="none", edgecolors="limegreen", s=32, linewidths=1.0, label="gt" if i == 0 else None)
                ax.set_title(f"t={float(t_scalar.item()):.2f}")
                ax.set_xlim(-50, 50)
                ax.set_ylim(-50, 50)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.25)

            for j in range(k, len(axes)):
                axes[j].axis("off")
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right")
            fig.suptitle(f"ego future denoise scatter | sample={sample_idx}", fontsize=14)
            fig.tight_layout()
            return _figure_to_rgb(fig)
        finally:
            plt.close(fig)
