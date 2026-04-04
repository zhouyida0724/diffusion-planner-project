from __future__ import annotations

"""TensorBoard visualization helpers for paper_dit_dpm.

Goals:
- Keep matplotlib imports lazy + headless-safe (Agg backend).
- Provide small, composable renderers:
  * NPZ-style single-frame scene image from in-memory tensors.
  * Simple (x,y) scatter plot image for trajectories (no connecting lines).

All functions return uint8 HWC RGB arrays suitable for SummaryWriter.add_image.
"""

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
    image_size: int = 800,
) -> np.ndarray | None:
    """Render an NPZ-style scene visualization for one sample in a training batch."""

    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrowPatch

    # Pull per-sample arrays (CPU numpy)
    ego_state = _to_numpy(batch["ego_current_state"])[sample_idx]
    lanes = _to_numpy(batch["lanes"])[sample_idx]
    route_lanes = _to_numpy(batch["route_lanes"])[sample_idx]
    route_lanes_avails = (
        _to_numpy(batch.get("route_lanes_avails"))[sample_idx]
        if batch.get("route_lanes_avails") is not None
        else None
    )
    ego_future = _to_numpy(batch["ego_agent_future"])[sample_idx]
    neighbor_past = _to_numpy(batch["neighbor_agents_past"])[sample_idx]
    neighbor_future = (
        _to_numpy(batch.get("neighbor_agents_future"))[sample_idx]
        if batch.get("neighbor_agents_future") is not None
        else None
    )
    static_objects = (
        _to_numpy(batch.get("static_objects"))[sample_idx]
        if batch.get("static_objects") is not None
        else None
    )

    # Best-effort sizing: 10 inches at dpi ~ image_size/10.
    dpi = max(80, int(image_size / 10))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
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

        # Lanes (left/right boundaries)
        for lane_idx in range(lanes.shape[0]):
            lane_x = lanes[lane_idx, :, 0]
            lane_y = lanes[lane_idx, :, 1]
            if np.allclose(lane_x, 0) and np.allclose(lane_y, 0):
                continue
            left_x = lanes[lane_idx, :, 0] + lanes[lane_idx, :, 4]
            left_y = lanes[lane_idx, :, 1] + lanes[lane_idx, :, 5]
            right_x = lanes[lane_idx, :, 0] + lanes[lane_idx, :, 6]
            right_y = lanes[lane_idx, :, 1] + lanes[lane_idx, :, 7]
            m_left = (left_x != 0) | (left_y != 0)
            m_right = (right_x != 0) | (right_y != 0)
            ax.plot(left_x[m_left], left_y[m_left], "b--", linewidth=1, alpha=0.35)
            ax.plot(right_x[m_right], right_y[m_right], "r--", linewidth=1, alpha=0.35)

        # Route lanes
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
            ax.plot(lane_x[valid], lane_y[valid], "-", color="#FFFF99", alpha=0.9, linewidth=2)

        # Ego marker at origin
        arrow = FancyArrowPatch(
            (0, 0),
            (4, 0),
            arrowstyle="-|>",
            mutation_scale=15,
            facecolor="red",
            edgecolor="darkred",
            linewidth=2,
        )
        ax.add_patch(arrow)
        ego_circle = patches.Circle((0, 0), radius=1.5, facecolor="red", edgecolor="darkred", linewidth=2)
        ax.add_patch(ego_circle)

        # Ego GT + pred
        ax.plot(ego_future[:, 0], ego_future[:, 1], "g-", linewidth=2.5, alpha=0.85, label="ego_gt")
        if ego_future_xy is not None:
            pred_xy = _to_numpy(ego_future_xy)
            ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="orange", linewidth=2.0, alpha=0.95, label="ego_pred")

        # Neighbor past/future (light)
        for agent_idx in range(neighbor_past.shape[0]):
            curr_x = float(neighbor_past[agent_idx, -1, 0])
            curr_y = float(neighbor_past[agent_idx, -1, 1])
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

        # Static objects (points)
        if static_objects is not None and static_objects.ndim == 2 and static_objects.shape[-1] >= 2:
            valid = np.any(np.abs(static_objects[:, :2]) > 1e-6, axis=1)
            if np.any(valid):
                ax.scatter(static_objects[valid, 0], static_objects[valid, 1], s=20, c="black", alpha=0.5, marker="x", label="static")

        ax.legend(loc="upper right")
        fig.tight_layout()
        return _figure_to_rgb(fig)
    finally:
        plt.close(fig)


def render_xy_scatter(
    xy: torch.Tensor | np.ndarray,
    *,
    title: str,
    image_size: int = 800,
    xlim: float = 50.0,
    marker: str = ".",
    alpha: float = 0.9,
) -> np.ndarray | None:
    """Render a simple XY scatter image (no lines), for showing denoise convergence."""

    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    xy_np = _to_numpy(xy)
    if xy_np.ndim != 2 or xy_np.shape[1] != 2:
        raise ValueError(f"xy must be [T,2], got {xy_np.shape}")

    dpi = max(80, int(image_size / 10))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=dpi)
    try:
        ax.scatter(xy_np[:, 0], xy_np[:, 1], s=28, alpha=alpha, marker=marker, c=np.linspace(0.1, 1.0, xy_np.shape[0]), cmap="viridis")
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-xlim, xlim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        fig.tight_layout()
        return _figure_to_rgb(fig)
    finally:
        plt.close(fig)
