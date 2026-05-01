from __future__ import annotations

from dataclasses import replace

from .config import PrefixSelectorConfig
from .scorers import consistency_l2, end_heading_error_rad, end_lateral_error_m, offroad_steps, progress_m, smoothness_cost
from .types import CandidateDiagnostics, CandidateTrajectory, SelectionContext, SelectionResult


class PrefixTrajectorySelector:
    """Simple rule-based selector over multi-sample diffusion trajectories.

    This module is intentionally planner-agnostic. The planner should only translate
    its runtime state into CandidateTrajectory + SelectionContext and call select().
    """

    def __init__(self, config: PrefixSelectorConfig | None = None):
        self._config = config or PrefixSelectorConfig()

    @property
    def config(self) -> PrefixSelectorConfig:
        return self._config

    def select(self, candidates: list[CandidateTrajectory], context: SelectionContext) -> SelectionResult:
        if not candidates:
            raise ValueError("candidates must be non-empty")

        prefix_steps = self._config.prefix_steps(context.dt)
        consistency_steps = self._config.consistency_steps(context.dt)

        diagnostics: list[CandidateDiagnostics] = []
        survivor_indices: list[int] = []

        for idx, cand in enumerate(candidates):
            local_xyh = cand.local_xyh
            prefix_off = offroad_steps(local_xyh, context, steps=prefix_steps)
            late_off = offroad_steps(local_xyh, context, start=prefix_steps)
            prefix_prog = progress_m(local_xyh, context.route_centerlines_local, upto=prefix_steps)
            progress_shortfall = max(0.0, float(self._config.min_prefix_progress_m) - float(prefix_prog))
            total_prog = progress_m(local_xyh, context.route_centerlines_local)
            prefix_lat = end_lateral_error_m(local_xyh, context.route_centerlines_local, upto=prefix_steps)
            total_lat = end_lateral_error_m(local_xyh, context.route_centerlines_local)
            prefix_head = end_heading_error_rad(local_xyh, context.route_centerlines_local, upto=prefix_steps)
            total_head = end_heading_error_rad(local_xyh, context.route_centerlines_local)
            consistency = consistency_l2(local_xyh, context.previous_selected_local, upto=consistency_steps)
            smoothness = smoothness_cost(local_xyh)

            rejected = False
            reject_reason = None
            if prefix_off > self._config.max_prefix_offroad_steps:
                rejected = True
                reject_reason = "prefix_offroad"
            elif prefix_lat > self._config.max_prefix_end_lateral_error_m:
                rejected = True
                reject_reason = "prefix_route_lateral"

            final_score = (
                self._config.w_prefix_offroad * float(prefix_off)
                + self._config.w_prefix_end_lateral * float(prefix_lat)
                + self._config.w_prefix_end_heading * float(prefix_head)
                - self._config.w_prefix_progress * float(prefix_prog)
                + self._config.w_prefix_progress_shortfall * float(progress_shortfall)
                + self._config.w_consistency * float(consistency)
                + self._config.w_late_offroad * float(late_off)
                + self._config.w_smoothness * float(smoothness)
            )

            diag = CandidateDiagnostics(
                prefix_offroad_steps=int(prefix_off),
                late_offroad_steps=int(late_off),
                prefix_progress_m=float(prefix_prog),
                prefix_progress_shortfall_m=float(progress_shortfall),
                total_progress_m=float(total_prog),
                prefix_end_lateral_error_m=float(prefix_lat),
                total_end_lateral_error_m=float(total_lat),
                prefix_end_heading_error_rad=float(prefix_head),
                total_end_heading_error_rad=float(total_head),
                consistency_l2=float(consistency),
                final_score=float(final_score),
                rejected=bool(rejected),
                reject_reason=reject_reason,
            )
            diagnostics.append(diag)
            if not rejected:
                survivor_indices.append(idx)

        def _safety_rank(i: int) -> tuple[int, int, float]:
            diag = diagnostics[i]
            total_offroad = diag.prefix_offroad_steps + diag.late_offroad_steps
            return (total_offroad, diag.prefix_offroad_steps, diag.final_score)

        used_fallback = False
        if survivor_indices:
            best_index = min(survivor_indices, key=_safety_rank)
        else:
            used_fallback = True
            best_index = min(
                range(len(candidates)),
                key=lambda i: (
                    diagnostics[i].prefix_offroad_steps + diagnostics[i].late_offroad_steps,
                    diagnostics[i].prefix_offroad_steps,
                    diagnostics[i].prefix_end_lateral_error_m,
                    -diagnostics[i].total_progress_m,
                ),
            )
            survivor_indices = [best_index]
            diagnostics[best_index] = replace(diagnostics[best_index], rejected=False, reject_reason="fallback_selected")

        return SelectionResult(
            best_index=int(best_index),
            diagnostics=diagnostics,
            survivor_indices=[int(i) for i in survivor_indices],
            used_fallback=used_fallback,
        )
