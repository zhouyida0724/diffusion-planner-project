# Closed-Loop Sampling Debug Experiment 2026-05-13

This directory archives the experimental tooling used to debug Diffusion Planner
closed-loop failures where open-loop metrics looked acceptable but closed-loop
simulation produced unsafe trajectories.

Contents:

- `scripts/analysis/`: one-off audits, selector trace analysis, sampling-grid
  planning/execution/collection, and visualization helpers.
- `scripts/diagnostics/`: feature-dump and fast-eval alignment diagnostics.
- `scripts/monitor/`: experiment monitoring helpers.
- `tests/`: tests for the experimental scripts.
- `patches/closedloop_runtime_selector_debug.patch`: production-tree debug
  instrumentation and runtime/selector changes that were useful for the
  experiment but are intentionally not part of the training PRs.
- `docs/superpowers/`: design/plan notes written during the runtime-frame
  investigation.

This package is intentionally not imported by the production training path. To
replay the experiment, copy scripts back to their original paths or apply the
patch in a disposable worktree, then use the recorded runbooks under
`outputs/analysis/closedloop_sampling_grid_20260513/`.
