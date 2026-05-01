from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixSelectorConfig:
    prefix_seconds: float = 3.0
    consistency_seconds: float = 2.0

    max_prefix_offroad_steps: int = 0
    min_prefix_progress_m: float = 1.0
    max_prefix_end_lateral_error_m: float = 3.0

    w_prefix_offroad: float = 100.0
    w_prefix_end_lateral: float = 3.0
    w_prefix_end_heading: float = 1.5
    w_prefix_progress: float = 2.0
    w_prefix_progress_shortfall: float = 2.0
    w_consistency: float = 2.5
    w_late_offroad: float = 0.3
    w_smoothness: float = 0.2

    fallback_offroad_scale: float = 100.0

    def prefix_steps(self, dt: float) -> int:
        return max(1, int(round(self.prefix_seconds / max(dt, 1e-3))))

    def consistency_steps(self, dt: float) -> int:
        return max(1, int(round(self.consistency_seconds / max(dt, 1e-3))))
