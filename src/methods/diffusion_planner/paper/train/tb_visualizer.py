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

import math
import numpy as np
import torch


def _ensure_pillow_resampling_compat() -> None:
    """Provide PIL.Image.Resampling fallback for older Pillow releases.

    Some TensorBoard image paths expect ``Image.Resampling`` to exist, but older
    Pillow versions only expose module-level constants like ``Image.LANCZOS`` or
    ``Image.ANTIALIAS``. Best-effort monkey-patch a tiny compatibility shim so
    TB image logging keeps working without warnings.
    """

    try:
        from PIL import Image
    except Exception:
        return

    if getattr(Image, "Resampling", None) is not None:
        return

    lanczos = getattr(Image, "LANCZOS", None)
    bicubic = getattr(Image, "BICUBIC", None)
    antialias = getattr(Image, "ANTIALIAS", None)
    bilinear = getattr(Image, "BILINEAR", None)
    nearest = getattr(Image, "NEAREST", None)

    class _CompatResampling:
        NEAREST = nearest if nearest is not None else 0
        BILINEAR = bilinear if bilinear is not None else (nearest if nearest is not None else 0)
        BICUBIC = bicubic if bicubic is not None else (bilinear if bilinear is not None else (nearest if nearest is not None else 0))
        LANCZOS = lanczos if lanczos is not None else (bicubic if bicubic is not None else (antialias if antialias is not None else (bilinear if bilinear is not None else (nearest if nearest is not None else 0))))

    Image.Resampling = _CompatResampling


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x)


def angle_difference_radians(angle1_rad: float, angle2_rad: float) -> float:
    """Return the smallest absolute difference between two angles (radians)."""
    diff = abs(float(angle1_rad) - float(angle2_rad))
    normalized_diff = diff % (2 * math.pi)
    return (2 * math.pi) - normalized_diff if normalized_diff > math.pi else normalized_diff


def _safe_import_matplotlib():
    _ensure_pillow_resampling_compat()
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
    ego_future_overlays: list[dict[str, Any]] | None = None,
    ego_past_xy: torch.Tensor | np.ndarray | None = None,
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
    # Kept for box heading visualization parity with src/platform/viz/npz_viz.py
    ego_heading = float(np.arctan2(ego_state[3], ego_state[2]))
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
            ax.plot(left_x[m_left], left_y[m_left], "b--", linewidth=1, alpha=0.5)
            ax.plot(right_x[m_right], right_y[m_right], "r--", linewidth=1, alpha=0.5)

        # Route lanes (with simple direction arrows)
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

            # ~5 arrows per lane
            if x_coords.shape[0] >= 2:
                arrow_interval = max(1, int(x_coords.shape[0] // 5))
                for i in range(0, int(x_coords.shape[0]) - 1, arrow_interval):
                    dx = float(x_coords[i + 1] - x_coords[i])
                    dy = float(y_coords[i + 1] - y_coords[i])
                    if (dx * dx + dy * dy) ** 0.5 > 1e-2:
                        ax.annotate(
                            "",
                            xy=(float(x_coords[i + 1]), float(y_coords[i + 1])),
                            xytext=(float(x_coords[i]), float(y_coords[i])),
                            arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.2, alpha=0.9),
                        )

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

        # Ego past / GT future / pred (match visualize_npz.py style: GT blue, pred orange)
        if ego_past_xy is not None:
            past_xy = _to_numpy(ego_past_xy)
            if past_xy.ndim == 2 and past_xy.shape[1] >= 2:
                ax.plot(past_xy[:, 0], past_xy[:, 1], color="purple", linewidth=2.5, alpha=0.85, label="ego_past")
                ax.scatter(past_xy[:, 0], past_xy[:, 1], s=18, c="purple", alpha=0.75)
        if np.any(np.abs(ego_future[:, :2]) > 1e-6):
            ax.plot(ego_future[:, 0], ego_future[:, 1], "b-", linewidth=3, alpha=0.8, label="ego_gt")
        if ego_future_xy is not None:
            pred_xy = _to_numpy(ego_future_xy)
            ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="orange", linewidth=3, alpha=0.9, label="ego_pred")

        # Additional ego future overlays (e.g., IDM plan vs diffusion plan)
        if ego_future_overlays:
            for ov in ego_future_overlays:
                try:
                    xy = _to_numpy(ov.get("xy"))
                    if xy is None or xy.ndim != 2 or xy.shape[1] < 2:
                        continue
                    color = str(ov.get("color", "cyan"))
                    label = str(ov.get("label", "overlay"))
                    lw = float(ov.get("lw", 2.5))
                    alpha = float(ov.get("alpha", 0.9))
                    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, alpha=alpha, label=label)
                except Exception:
                    continue

        # Neighbor past trajectories (more visible, with ids)
        agent_id = 0
        for agent_idx in range(neighbor_past.shape[0]):
            curr_x = float(neighbor_past[agent_idx, -1, 0])
            curr_y = float(neighbor_past[agent_idx, -1, 1])
            past = neighbor_past[agent_idx, :, :2]
            if abs(curr_x) < 0.1 and abs(curr_y) < 0.1:
                continue
            if abs(curr_x) > 50 or abs(curr_y) > 50:
                continue
            valid_mask = (past[:, 0] != 0) | (past[:, 1] != 0)
            if not np.any(valid_mask):
                continue
            ax.annotate(str(agent_id), (curr_x, curr_y), fontsize=9, fontweight="bold", color="red", xytext=(4, 4), textcoords="offset points")
            agent_id += 1
            ax.plot(past[valid_mask, 0], past[valid_mask, 1], "b-", linewidth=2.5, alpha=0.75)
            ax.plot(curr_x, curr_y, "o", markersize=6, color="blue", alpha=0.9)

            # Bounding box visualization (match npz_viz conventions)
            try:
                if neighbor_past.shape[-1] >= 11:
                    # types
                    type_v = float(neighbor_past[agent_idx, -1, 8])
                    type_p = float(neighbor_past[agent_idx, -1, 9])
                    type_b = float(neighbor_past[agent_idx, -1, 10])

                    if type_v > 0.5:
                        agent_type = "Vehicle"
                        color = "blue"
                    elif type_p > 0.5:
                        agent_type = "Pedestrian"
                        color = "orange"
                    elif type_b > 0.5:
                        agent_type = "Bicycle"
                        color = "purple"
                    else:
                        agent_type = None
                        color = "cyan"

                    width = float(neighbor_past[agent_idx, -1, 6])
                    length = float(neighbor_past[agent_idx, -1, 7])
                    if width == 0:
                        width = 2.0
                    if length == 0:
                        length = 4.0

                    cos_h = float(neighbor_past[agent_idx, -1, 2])
                    sin_h = float(neighbor_past[agent_idx, -1, 3])
                    neighbor_heading = angle_difference_radians(ego_heading, float(np.arctan2(sin_h, cos_h)))

                    if agent_type == "Vehicle":
                        # Matplotlib rotates rectangles around the lower-left corner by default,
                        # which makes the box appear offset from the (curr_x,curr_y) center point.
                        # Rotate around center when possible; fall back to a manual transform.
                        try:
                            rect = patches.Rectangle(
                                (curr_x - length / 2, curr_y - width / 2),
                                length,
                                width,
                                angle=math.degrees(neighbor_heading),
                                rotation_point="center",
                                facecolor=color,
                                edgecolor="black",
                                linewidth=1,
                                alpha=0.35,
                            )
                        except TypeError:
                            rect = patches.Rectangle(
                                (curr_x - length / 2, curr_y - width / 2),
                                length,
                                width,
                                angle=0.0,
                                facecolor=color,
                                edgecolor="black",
                                linewidth=1,
                                alpha=0.35,
                            )
                            try:
                                import matplotlib.transforms as transforms

                                rect.set_transform(transforms.Affine2D().rotate_around(curr_x, curr_y, neighbor_heading) + ax.transData)
                            except Exception:
                                pass
                        ax.add_patch(rect)
                    elif agent_type == "Pedestrian":
                        radius = max(width, length) / 2
                        circ = patches.Circle((curr_x, curr_y), radius, facecolor=color, edgecolor="black", linewidth=1, alpha=0.35)
                        ax.add_patch(circ)
                    elif agent_type == "Bicycle":
                        diamond = patches.RegularPolygon(
                            (curr_x, curr_y),
                            numVertices=4,
                            radius=max(width, length) / 2,
                            orientation=neighbor_heading,
                            facecolor=color,
                            edgecolor="black",
                            linewidth=1,
                            alpha=0.35,
                        )
                        ax.add_patch(diamond)
            except Exception:
                pass

        # Neighbor future trajectories
        if neighbor_future is not None:
            for agent_idx in range(neighbor_future.shape[0]):
                future = neighbor_future[agent_idx, :, :2]
                valid_mask = (future[:, 0] != 0) | (future[:, 1] != 0)
                if np.any(valid_mask):
                    ax.plot(future[valid_mask, 0], future[valid_mask, 1], "g-", linewidth=2.5, alpha=0.55)

        # Static objects (points) — many nuPlan exports currently have these as all-zeros.
        if static_objects is not None and static_objects.ndim == 2 and static_objects.shape[-1] >= 2:
            valid = np.any(np.abs(static_objects[:, :2]) > 1e-6, axis=1)
            if np.any(valid):
                ax.scatter(static_objects[valid, 0], static_objects[valid, 1], s=40, c="black", alpha=0.65, marker="x", label="static")

        ax.legend(loc="upper right")
        fig.tight_layout()
        return _figure_to_rgb(fig)
    finally:
        plt.close(fig)


def render_xy_scatter_with_context(
    batch: dict[str, Any],
    *,
    sample_idx: int,
    xy: torch.Tensor | np.ndarray,
    title: str,
    image_size: int = 800,
    marker: str = ".",
    alpha: float = 0.95,
    ego_past_xy: torch.Tensor | np.ndarray | None = None,
) -> np.ndarray | None:
    """Render XY scatter over the same NPZ-style scene context (unified coordinates).

    This is used for denoise panels so the viewer can see xt/x0_pred convergence *in context*.
    """

    plt = _safe_import_matplotlib()
    if plt is None:
        return None

    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrowPatch

    xy_np = _to_numpy(xy)
    if xy_np.ndim != 2 or xy_np.shape[1] != 2:
        raise ValueError(f"xy must be [T,2], got {xy_np.shape}")

    ego_state = _to_numpy(batch["ego_current_state"])[sample_idx]
    ego_heading = float(np.arctan2(ego_state[3], ego_state[2]))

    lanes = _to_numpy(batch["lanes"])[sample_idx]
    route_lanes = _to_numpy(batch["route_lanes"])[sample_idx]
    route_lanes_avails = (
        _to_numpy(batch.get("route_lanes_avails"))[sample_idx]
        if batch.get("route_lanes_avails") is not None
        else None
    )
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

    dpi = max(80, int(image_size / 10))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
    try:
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

        # Lanes
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
            ax.plot(left_x[m_left], left_y[m_left], "b--", linewidth=1, alpha=0.5)
            ax.plot(right_x[m_right], right_y[m_right], "r--", linewidth=1, alpha=0.5)

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
            x_coords = lane_x[valid]
            y_coords = lane_y[valid]
            ax.plot(x_coords, y_coords, "-", color="#FFFF99", alpha=0.9, linewidth=2)

        # Ego marker at origin
        arrow = FancyArrowPatch((0, 0), (4, 0), arrowstyle="-|>", mutation_scale=15, facecolor="red", edgecolor="darkred", linewidth=2)
        ax.add_patch(arrow)
        ego_circle = patches.Circle((0, 0), radius=1.5, facecolor="red", edgecolor="darkred", linewidth=2)
        ax.add_patch(ego_circle)

        # Neighbors
        for agent_idx in range(neighbor_past.shape[0]):
            curr_x = float(neighbor_past[agent_idx, -1, 0])
            curr_y = float(neighbor_past[agent_idx, -1, 1])
            past = neighbor_past[agent_idx, :, :2]
            if abs(curr_x) < 0.1 and abs(curr_y) < 0.1:
                continue
            if abs(curr_x) > 50 or abs(curr_y) > 50:
                continue
            valid_mask = (past[:, 0] != 0) | (past[:, 1] != 0)
            if not np.any(valid_mask):
                continue
            ax.plot(past[valid_mask, 0], past[valid_mask, 1], "b-", linewidth=2.0, alpha=0.65)
            ax.plot(curr_x, curr_y, "o", markersize=5, color="blue", alpha=0.85)

            # Optional agent shape overlay (vehicle box / pedestrian circle / bicycle diamond)
            try:
                if neighbor_past.shape[-1] >= 11:
                    type_v = float(neighbor_past[agent_idx, -1, 8])
                    type_p = float(neighbor_past[agent_idx, -1, 9])
                    type_b = float(neighbor_past[agent_idx, -1, 10])

                    if type_v > 0.5:
                        agent_type = "Vehicle"
                        color = "blue"
                    elif type_p > 0.5:
                        agent_type = "Pedestrian"
                        color = "orange"
                    elif type_b > 0.5:
                        agent_type = "Bicycle"
                        color = "purple"
                    else:
                        agent_type = None
                        color = "cyan"

                    width = float(neighbor_past[agent_idx, -1, 6])
                    length = float(neighbor_past[agent_idx, -1, 7])
                    if width == 0:
                        width = 2.0
                    if length == 0:
                        length = 4.0

                    cos_h = float(neighbor_past[agent_idx, -1, 2])
                    sin_h = float(neighbor_past[agent_idx, -1, 3])
                    neighbor_heading = angle_difference_radians(ego_heading, float(np.arctan2(sin_h, cos_h)))

                    if agent_type == "Vehicle":
                        # Rotate around center so the box stays centered on (curr_x,curr_y).
                        try:
                            rect = patches.Rectangle(
                                (curr_x - length / 2, curr_y - width / 2),
                                length,
                                width,
                                angle=math.degrees(neighbor_heading),
                                rotation_point="center",
                                facecolor=color,
                                edgecolor="black",
                                linewidth=1,
                                alpha=0.25,
                            )
                        except TypeError:
                            rect = patches.Rectangle(
                                (curr_x - length / 2, curr_y - width / 2),
                                length,
                                width,
                                angle=0.0,
                                facecolor=color,
                                edgecolor="black",
                                linewidth=1,
                                alpha=0.25,
                            )
                            try:
                                import matplotlib.transforms as transforms

                                rect.set_transform(transforms.Affine2D().rotate_around(curr_x, curr_y, neighbor_heading) + ax.transData)
                            except Exception:
                                pass
                        ax.add_patch(rect)
                    elif agent_type == "Pedestrian":
                        radius = max(width, length) / 2
                        circ = patches.Circle((curr_x, curr_y), radius, facecolor=color, edgecolor="black", linewidth=1, alpha=0.25)
                        ax.add_patch(circ)
                    elif agent_type == "Bicycle":
                        diamond = patches.RegularPolygon(
                            (curr_x, curr_y),
                            numVertices=4,
                            radius=max(width, length) / 2,
                            orientation=neighbor_heading,
                            facecolor=color,
                            edgecolor="black",
                            linewidth=1,
                            alpha=0.25,
                        )
                        ax.add_patch(diamond)
            except Exception:
                pass

        if neighbor_future is not None:
            for agent_idx in range(neighbor_future.shape[0]):
                future = neighbor_future[agent_idx, :, :2]
                valid_mask = (future[:, 0] != 0) | (future[:, 1] != 0)
                if np.any(valid_mask):
                    ax.plot(future[valid_mask, 0], future[valid_mask, 1], "g-", linewidth=2.0, alpha=0.35)

        if static_objects is not None and static_objects.ndim == 2 and static_objects.shape[-1] >= 2:
            valid = np.any(np.abs(static_objects[:, :2]) > 1e-6, axis=1)
            if np.any(valid):
                ax.scatter(static_objects[valid, 0], static_objects[valid, 1], s=30, c="black", alpha=0.5, marker="x")

        if ego_past_xy is not None:
            past_xy = _to_numpy(ego_past_xy)
            if past_xy.ndim == 2 and past_xy.shape[1] >= 2:
                ax.plot(past_xy[:, 0], past_xy[:, 1], color="purple", linewidth=2.0, alpha=0.8)
                ax.scatter(past_xy[:, 0], past_xy[:, 1], s=16, c="purple", alpha=0.7)

        # Overlay scatter points (denoise trajectory)
        ax.scatter(
            xy_np[:, 0],
            xy_np[:, 1],
            s=36,
            alpha=alpha,
            marker=marker,
            c=np.linspace(0.1, 1.0, xy_np.shape[0]),
            cmap="viridis",
        )
        if xy_np.shape[0] >= 2:
            ax.plot(xy_np[:, 0], xy_np[:, 1], color="orange", linewidth=2.5, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
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
