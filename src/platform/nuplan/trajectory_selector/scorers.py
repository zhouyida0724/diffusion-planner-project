from __future__ import annotations

import math

import numpy as np

from .types import EgoPose2D, SelectionContext


def wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def local_xy_to_world(local_xy: np.ndarray, ego_pose: EgoPose2D) -> np.ndarray:
    c = math.cos(ego_pose.heading)
    s = math.sin(ego_pose.heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    world = (rot @ local_xy[:, :2].T).T
    world[:, 0] += ego_pose.x
    world[:, 1] += ego_pose.y
    return world


def local_heading_to_world(local_heading: np.ndarray, ego_pose: EgoPose2D) -> np.ndarray:
    return local_heading + ego_pose.heading


def vehicle_corners_world(x: float, y: float, heading: float, half_l: float, half_w: float) -> np.ndarray:
    pts = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]], dtype=np.float64)
    c = math.cos(heading)
    s = math.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (rot @ pts.T).T + np.array([x, y], dtype=np.float64)


def offroad_steps(local_xyh: np.ndarray, context: SelectionContext, *, steps: int | None = None, start: int = 0) -> int:
    end = len(local_xyh) if steps is None else min(len(local_xyh), start + steps)
    headings = local_heading_to_world(local_xyh[:, 2], context.ego_pose)
    world_xy = local_xy_to_world(local_xyh[:, :2], context.ego_pose)
    bad = 0
    for i in range(start, end):
        heading = float(headings[i])
        center_x = float(world_xy[i, 0]) + float(context.rear_axle_to_center_dist) * math.cos(heading)
        center_y = float(world_xy[i, 1]) + float(context.rear_axle_to_center_dist) * math.sin(heading)
        corners = vehicle_corners_world(
            center_x,
            center_y,
            heading,
            context.vehicle_half_length,
            context.vehicle_half_width,
        )
        flags = np.asarray(context.drivable_checker(corners), dtype=bool)
        if flags.shape[0] != 4 or not np.all(flags):
            bad += 1
    return bad


def _flatten_route(route_centerlines_local: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pts: list[np.ndarray] = []
    for line in route_centerlines_local:
        arr = np.asarray(line, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
            continue
        pts.append(arr[:, :2])
    if not pts:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    cat = np.concatenate(pts, axis=0)
    if len(cat) == 1:
        return cat, np.zeros((1,), dtype=np.float64)
    seg = np.linalg.norm(cat[1:] - cat[:-1], axis=1)
    cum = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg)])
    return cat, cum


def project_to_route(local_xy: np.ndarray, route_centerlines_local: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    route_pts, route_s = _flatten_route(route_centerlines_local)
    if len(route_pts) == 0:
        n = len(local_xy)
        return np.full((n,), np.inf), np.zeros((n,), dtype=np.float64), np.zeros((n,), dtype=np.float64)

    diffs = local_xy[:, None, :] - route_pts[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    idx = np.argmin(dist, axis=1)
    nearest_dist = dist[np.arange(len(local_xy)), idx]
    progress = route_s[idx]

    heading = np.zeros((len(local_xy),), dtype=np.float64)
    for i, ridx in enumerate(idx):
        lo = max(0, int(ridx) - 1)
        hi = min(len(route_pts) - 1, int(ridx) + 1)
        delta = route_pts[hi] - route_pts[lo]
        heading[i] = math.atan2(float(delta[1]), float(delta[0])) if np.linalg.norm(delta) > 1e-6 else 0.0
    return nearest_dist, progress, heading


def progress_m(local_xy: np.ndarray, route_centerlines_local: list[np.ndarray], *, upto: int | None = None) -> float:
    n = len(local_xy) if upto is None else min(len(local_xy), upto)
    if n <= 0:
        return 0.0
    _, route_progress, _ = project_to_route(local_xy[:n, :2], route_centerlines_local)
    return float(route_progress[-1] - route_progress[0]) if len(route_progress) else 0.0


def end_lateral_error_m(local_xy: np.ndarray, route_centerlines_local: list[np.ndarray], *, upto: int | None = None) -> float:
    n = len(local_xy) if upto is None else min(len(local_xy), upto)
    if n <= 0:
        return float("inf")
    nearest_dist, _, _ = project_to_route(local_xy[:n, :2], route_centerlines_local)
    return float(nearest_dist[-1]) if len(nearest_dist) else float("inf")


def end_heading_error_rad(local_xyh: np.ndarray, route_centerlines_local: list[np.ndarray], *, upto: int | None = None) -> float:
    n = len(local_xyh) if upto is None else min(len(local_xyh), upto)
    if n <= 0:
        return math.pi
    _, _, route_heading = project_to_route(local_xyh[:n, :2], route_centerlines_local)
    return abs(wrap_pi(float(local_xyh[n - 1, 2]) - float(route_heading[-1]))) if len(route_heading) else math.pi


def consistency_l2(local_xyh: np.ndarray, previous_selected_local: np.ndarray | None, *, upto: int) -> float:
    if previous_selected_local is None:
        return 0.0
    n = min(len(local_xyh), len(previous_selected_local), max(1, upto))
    if n <= 0:
        return 0.0
    d = local_xyh[:n, :2] - previous_selected_local[:n, :2]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def smoothness_cost(local_xyh: np.ndarray, *, upto: int | None = None) -> float:
    n = len(local_xyh) if upto is None else min(len(local_xyh), upto)
    if n <= 2:
        return 0.0
    xy = local_xyh[:n, :2]
    dd = xy[2:] - 2.0 * xy[1:-1] + xy[:-2]
    return float(np.mean(np.linalg.norm(dd, axis=1)))
