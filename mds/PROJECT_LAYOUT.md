# PROJECT_LAYOUT.md

This repo is a **shared environment for reproducing multiple planning papers**, not just Diffusion Planner.

Core goals:
1) Keep **paper-specific code isolated** (so adding another paper doesn’t pollute existing implementations).
2) Keep **nuPlan integration/platform code generic**.
3) Keep **entrypoint scripts stable** (avoid breaking existing workflows).
4) Enforce **permissions discipline** for Docker outputs (avoid root-owned artifacts on host).

---

## Directory layout (target)

### 1) Source code

```
src/
  platform/                  # paper-agnostic platform layer
    nuplan/                  # nuPlan integration (generic)
      planners/              # planner wrappers (call into a chosen method)
      runners/               # simulation runners (stable entrypoints)
      features/              # runtime feature adapters/extractors
      utils/
    io/                      # manifest/index/npz backends (generic)
    viz/                     # generic visualization helpers

  methods/                   # paper-specific implementations (isolated)
    diffusion_planner/       # Diffusion Planner method (model/diffusion/loss/train)
    # planTF/
    # pluto/
    # <other_paper>/
```

**Rule:** code in `src/methods/<paper>/` must not contain nuPlan runner glue. nuPlan glue lives in `src/platform/nuplan/`.

### 2) Scripts (stable user-facing entrypoints)

```
scripts/
  docker/     # docker setup scripts (training + nuplan-simulation)
  export/     # plan/export scripts
  viz/        # visualization scripts
  sim/        # run_simulation / nuboard scripts
  train/      # training entrypoints
```

**Compatibility rule:** do not break existing commands.
- Prefer keeping old script filenames as thin wrappers that forward to new modules.
- If a script becomes obsolete, delete it only after confirming it is unreferenced and the replacement is documented.

### 3) Third-party dependencies

```
third_party/
  nuplan-devkit/             # upstream nuPlan code (ideally as a submodule)
  # nuplan-visualization/    # if needed, also keep upstream here
```

**Rule:** avoid editing `third_party/` directly. If needed, keep patches documented.

### 4) Generated artifacts (ignored by git)

```
outputs/
  export/
  viz/
  training/
  sim/
```

All generated artifacts should go to `outputs/`.

---

## Docker permissions policy (must-follow)

Problem: if containers write to bind-mounted host paths as `root`, the host files become **root-owned** and later steps fail.

Policy:
- When a Docker container writes to host-mounted directories, run it as the host user:
  - `--user $(id -u):$(id -g)` (or explicit `1000:1000`)
- Default output locations for scripts must be under `outputs/`.

---

## Migration plan (safe, step-by-step)

We migrate without breaking existing workflows.

### Phase 0 — Freeze stable entrypoints
- Keep current `scripts/*.py` and `scripts/*.sh` entrypoints working.

### Phase 1 — Introduce new layout (no behavior change)
- Add `src/platform` + `src/methods` skeleton.
- Add `outputs/` and update docs.

### Phase 2 — Move logic behind wrappers (with smoke tests)
For each area (viz/export/sim/train):
- Move implementation into `src/` modules.
- Keep `scripts/*` as thin wrappers.
- Run smoke tests:
  1) simulation smoke (1 scenario)
  2) extract_single_frame smoke (fixed case)
  3) visualize_npz smoke (1 sample)
  4) export smoke (limit=100)

### Phase 3 — Delete obsolete scripts
Only after:
- wrappers exist
- docs updated
- grep confirms no references

---

## Obsolete script policy
- One-off / mini-only scripts that write to root-owned dirs are generally obsolete.
- Prefer deterministic, parameterized scripts that write to `outputs/`.
