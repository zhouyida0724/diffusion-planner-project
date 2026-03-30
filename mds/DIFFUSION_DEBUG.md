# Diffusion Planner Closed-loop Debug Flags

This repo has **gated** runtime debug logs to diagnose issues like:
- Diffusion output collapsing to a nearly fixed trajectory across scenes
- Conditioning tensors being all-zeros / constant
- Normalizer scale mismatch (inverse() shrinking motion)

## 1) Conditioning / feature variability

Enable:

```bash
export DIFFPLANNER_DEBUG_COND=1
export DIFFPLANNER_DEBUG_K=10   # optional (default 5)
```

This prints (first K planner ticks only):
- lanes / route_lanes shapes + nonzero counts
- lanes_avails_sum / route_lanes_avails_sum
- route_roadblock_ids length
- dynamic context nonzero counts (neighbor past/future, static objects, ego_agent_future)

What to look for:
- If `route_lanes_nz` / `lanes_nz` are ~0 or nearly constant across different scenarios, the model is effectively unconditioned.
- If dynamic tensors are always zero, the encoder may see no agents/static context.

## 2) Sampler / normalization sanity

Enable:

```bash
export DIFFPLANNER_DEBUG_SAMPLER=1
export DIFFPLANNER_DEBUG_K=10   # optional (default 5)
```

This prints (first K calls only) sampler stats inside the paper-model decoder:
- `xT_future_mean/std` for the initial noisy trajectory
- a few initial noise values (`noise0=[...]`) to confirm noise changes
- `x0_future_mean/std` before inverse normalization
- `inv_future_mean/std` after inverse normalization

What to look for:
- If `noise0` is identical across scenes/ticks, seeding may be forcing determinism.
- If `x0_future_*` has energy but `inv_future_*` collapses to tiny values, the **state_normalizer** mean/std likely mismatches training.

## Back-compat

`DP_RUNTIME_DEBUG=1` still enables debug and will also turn on the above logs.
