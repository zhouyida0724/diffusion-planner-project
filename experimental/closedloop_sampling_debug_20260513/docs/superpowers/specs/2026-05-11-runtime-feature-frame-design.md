# Runtime Feature Frame Design

## Problem

Closed-loop paper runtime currently calls the offline DB extractor by timestamp, so map, agents, static objects, and route features are projected in the DB/log ego frame. The model output is then executed through `transform_predictions_to_states()` using the simulated closed-loop ego history. Once simulated ego deviates from DB ego, conditioning and execution use different local frames, which explains the open-loop-ok / closed-loop-fly failure mode.

## Selected Approach

Use an explicit local-frame override in the shared feature extractor. Keep the default path bitwise-compatible with export/training by using the DB ego pose when no override is provided. In closed-loop runtime, pass the simulated current ego pose as the local frame and pass simulated ego current/history arrays as overrides.

## Data Contract

- DB/log still owns scenario token, resolved lidar timestamp, traffic lights, raw actors, static boxes, route ids, and map context.
- Runtime ego owns the local projection frame for closed-loop features.
- Closed-loop `ego_current_state`, `ego_past`, and ego slot in `neighbor_agents_past` come from simulated history.
- `ego_agent_future` and `neighbor_agents_future` are retained for schema compatibility but are not consumed by inference; when produced under runtime frame they must be projected consistently or zero-padded by existing fit logic.
- Feature dumps should record that feature local frame source is `runtime_sim` so postmortems can distinguish DB-aligned and runtime-aligned dumps.

## Scope

Modify only the Python extractor/planner runtime path and targeted tests. Do not change training cache generation semantics, route-correction algorithms, selector logic, or vendor code.

## Verification

- Unit tests prove default extractor behavior still uses DB pose, runtime overrides change the projection origin, planner passes simulated ego as the local frame, and runtime ego state uses local `[0, 0, 1, 0]`.
- Existing closed-loop debug/analyzer tests continue to pass.
- A reproducible smoke simulation on token `bca0c55205865506` compares the same diagnostic outputs before/after this change.
