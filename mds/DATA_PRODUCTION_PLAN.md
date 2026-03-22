# Data Production Plan (v0.1 → v1)

> Scope: turn the current **single-frame** feature export workflow into a **batch data production pipeline** with:
> - reproducible planning (what to export)
> - sharded export (how to run, resume, parallelize)
> - lightweight per-frame sanity checks (basic correctness signals)
> - optional visualization sampling for audits
>
> Constraint: **all extraction/export code must run inside Docker** (`nuplan-simulation`).

---

## 0. Terminology

- **DB**: a nuPlan sqlite database file (`*.db`).
- **Scene**: a row in `scene` table (token identifies a drive segment).
- **Frame**: an ego_pose index within a scene’s `log_token` time-ordered ego poses.
- **Sample**: one exported training item (typically one frame).
- **Plan / Index**: the concrete list of samples to export.
- **Shard**: a chunk of the plan, exported independently (for resume / parallelism).

---

## 1. Pipeline overview (steps + inputs/outputs)

### Step A — Inventory (optional but recommended)
**Goal**: understand dataset size/composition before choosing stride/quotas.

- **Input**: DB root dir (e.g. `/media/.../data/cache/train_*`)
- **Output**:
  - `inventory/db_inventory.jsonl` (per-db: location, scene count, ego_pose count stats)
  - `inventory/location_summary.json` (counts by location)
  - `inventory/scene_length_histograms/*.json` (optional)

### Step B — PLAN generation (**重点展开**, see §2)
**Goal**: generate a reproducible list of samples (db, scene, frame).

- **Input**:
  - DB root(s)
  - sampling policy (stride, per-location quotas, random seed, filters)
- **Output**:
  - `plans/<plan_name>/index.jsonl` (the sample list)
  - `plans/<plan_name>/plan_config.yaml` (exact parameters used)
  - `plans/<plan_name>/stats.json` (counts by location, estimated size)
  - `plans/<plan_name>/shards/shard_000.jsonl ...` (split index)

### Step C — Export (sharded)
**Goal**: run feature extraction on each shard with timeouts and lightweight checks.

- **Input**: `shard_*.jsonl`, extraction code
- **Output (per shard)**:
  - `exports/<plan_name>/shards/shard_000.npz` (multi-frame arrays)
  - `exports/<plan_name>/shards/shard_000.manifest.jsonl` (per-sample metadata + flags)
  - `exports/<plan_name>/shards/shard_000.metrics.json` (aggregate stats)
  - `exports/<plan_name>/shards/shard_000.log`

### Step D — QC aggregation
**Goal**: merge shard-level QC into a plan-level view and produce anomaly lists.

- **Input**: shard manifests/metrics
- **Output**:
  - `exports/<plan_name>/qc/summary.json`
  - `exports/<plan_name>/qc/anomalies.jsonl` (route_lanes_avails==0, NaN, etc.)
  - `exports/<plan_name>/qc/location_breakdown.csv`

### Step E — Visualization sampling (audit only)
**Goal**: keep human-in-the-loop sanity with limited cost.

- **Input**:
  - random sample list and anomaly list
  - viz script
- **Output**:
  - `exports/<plan_name>/viz/<location>/*.png`
  - `exports/<plan_name>/viz/manifest.jsonl`

---

## 2. Step B — PLAN generation (重点)

### 2.0 Training distribution control: location + tags (design goal)
We want training-time control over sample distribution by:
- **location** (city)
- **scenario tags** (from DB `scenario_tag.type`)

Principle:
- Do **not** bake a single distribution into exported NPZ.
- Export should carry **complete metadata** (location + tags) so training can reweight.
- PLAN generation should compute **tag distribution statistics** and (optionally) produce a tag-balanced plan variant.

For v0.1 (10k frames) we will:
1) include `location` and `tags` for each planned sample
2) compute and save tag distribution over the plan
3) keep sampling simple (P1 stride) first, then decide if we need tag-balancing based on stats.

### 2.1 Tag definition at frame level (how to attach tags to a frame)
`scenario_tag` is keyed by `lidar_pc_token`, so for a frame at `center_timestamp_us`:
- find the nearest (or last <=) `lidar_pc` at that timestamp within the same log
- collect all `scenario_tag.type` rows for that `lidar_pc_token`
- write them as `tags: [..]` in the plan record

(If needed later, we can widen to a time window and union tags.)

### 2.2 Plan record schema (with tags)

`index.jsonl` schema (proposed):
```json
{
  "sample_id": "<stable string>",
  "location": "sg-one-north",
  "db_path": "/media/.../train_singapore/xxx.db",
  "db_name": "xxx.db",
  "scene_token_hex": "8ede12113bf9547b",
  "log_token_hex": "...",
  "frame_index": 23102,
  "timestamp_us": 1632906374707603,
  "lidar_pc_token_hex": "...",
  "tags": ["high_magnitude_speed", "traversing_intersection"]
}
```

### 2.3 Tag distribution outputs
During plan generation, compute:
- frequency of each tag (`count(tag)`)
- co-occurrence summary (optional)
- frequency of “no tag” samples

Write to:
- `plans/<plan_name>/tag_stats.json`
- (optional) `plans/<plan_name>/tag_stats.csv`

### 2.4 Optional: tag-balanced plan variant (future)
If training requires near-uniform tag coverage, we can generate a second plan:
- `index_tag_balanced.jsonl`

Implementation idea (later): choose a **primary_tag** per sample using a priority list, then enforce per-primary-tag quotas.

---


### 2.1 Why we need a plan
A “plan” decouples **what to export** from **how/when we run export**.

Benefits:
- reproducibility: same seed + config → same sample list
- parallelism: shard files can run on multiple processes/machines
- resume: if shard_017 fails, rerun only that shard
- analytics: estimate size/time before spending days exporting

### 2.2 Plan file format
We use a simple JSONL (1 line per sample).

`index.jsonl` schema (proposed):
```json
{
  "sample_id": "<stable string>",
  "location": "sg-one-north",
  "db_path": "/media/.../train_singapore/xxx.db",
  "db_name": "xxx.db",
  "scene_token_hex": "8ede12113bf9547b",
  "log_token_hex": "...",
  "frame_index": 23102,
  "timestamp_us": 1632906374707603
}
```
Notes:
- `sample_id` should be deterministic (e.g. `{db_name}:{scene}:{frame}`) to avoid duplicates.
- Store both `location` and DB path; do **not** infer location from filename.

### 2.3 Plan config
Write the exact plan parameters alongside the index:
- seed
- db roots included/excluded
- per-location quotas
- stride policy
- filters (min ego poses, skip short scenes)

### 2.4 Sampling policies (initial options)
We’ll start with policies that are easy to reason about:

**Policy P0 (v0.1)** — “small-city smoke test”
- Choose the **smallest** location (fewest planned frames) from inventory.
- Export exactly **10,000 frames** from that location.

**Policy P1** — uniform stride within scenes
- For each scene, choose frame indices: `start + k*stride`.
- Pros: time-uniform; Cons: may bias toward longer scenes.

**Policy P2** — fixed quota per scene
- Each scene contributes up to `K` frames sampled uniformly.
- Pros: increases scene diversity.

**Policy P3** — location-balanced quotas
- Ensure each location contributes a target fraction/number.

For v0.1 we keep it simple: location chosen first, then P1 or P2 inside it.

### 2.5 Sharding strategy
Sharding is a pure function of the index.

Proposed defaults:
- `frames_per_shard = 2000` (→ 10k frames = 5 shards)
- shard file naming: `shard_{i:03d}.jsonl`

Shard outputs mirror shard inputs.

### 2.6 Plan-time estimates (cheap but useful)
During plan creation, also estimate:
- **estimated_npz_size_per_frame** (use measured ~0.152 MB/frame from the recent 50-frame run as an initial constant)
- total estimated disk = `num_frames * 0.152MB` (to be updated after v0.1 run)
- expected export time range (rough; refined after v0.1 benchmark)

---

## 3. v0.1 execution: ~10k frames (scene-window sampling) from the smallest location

### 3.1 Goals
- measure **meaningful** throughput (frames/s) → requires map_api reuse (minimal refactor)
- measure disk footprint (MB/frame; total)
- validate a production-like loop (plan → export → QC) without hangs

### 3.2 Sampling for v0.1 (finalized requirement)
Instead of global-uniform sampling, we do **scene-level selection** and then export **uniform stride within the whole scene** (no windowing):

1) Choose a target location (start with the smallest location by ego_pose frames).
2) Randomly select **N scenes = 200** (scene tokens from DB `scene` table).
3) For each selected scene, export frame indices within that scene's log:
   - `0, 8, 16, 24, ...` (stride = 8)
   - **cap_per_scene = 100** (at most 100 frames per scene)

Notes:
- This yields ~ `200 * 100 = 20,000` frames (order-of-magnitude target for v0.1).
- PLAN must record the exact scene list + seed + stride + cap for reproducibility.

### 3.3 Procedure
1) Inventory → choose smallest location
2) Generate plan with **scene selection (200 scenes) + stride=8 + cap_per_scene=100** + tag enrichment + tag stats
3) Export as a single NPZ (for v0.1), with QC manifests/metrics
4) (Optional later) visualization audit after NPZ is produced

### 3.4 Success criteria
- Export completes without hangs (timeouts prevent deadlocks)
- No NaN/Inf in outputs (hard fail → skip)
- route_lanes truly missing (route_lanes_avails_sum==0) is treated as **hard skip**
- QC summary produced (skip counts + soft flag counts)

---

## 4. Lightweight per-frame correctness checks (export-time)

Checks are meant to be **fast** and to catch catastrophic errors:

Hard failures (skip sample):
- any NaN/Inf in required fields
- unexpected tensor shape
- lanes_avails all-zero

Soft flags (record + keep sample, but track rate):
- route_lanes_avails all-zero
- route_lanes valid points all far from ego (e.g. min_dist > 50m; threshold configurable)
- neighbor_future implausible jump (optional threshold)

All flags go into `shard_XXX.manifest.jsonl`.

---

## 5. NPZ organization (export output format)

Avoid “one frame per file”. For production:

- One shard → one NPZ containing stacked arrays:
  - `ego_current_state`: `(T, 10)`
  - `ego_past`: `(T, 21, 3)`
  - ...
  - `route_lanes`: `(T, 25, 20, 12)`
  - avails similarly

- Keep `manifest.jsonl` alongside, mapping each `t` index to its `(db, scene, frame)`.

---

## 6. Visualization sampling (audit)

- For each shard:
  - random `K` samples (e.g. 10)
  - plus all samples with soft flags
- Aggregate by location for review

---

## 7. v0.1 TODO checklist (what we will implement next)

### TODO-P (Plan)
- [ ] Implement PLAN generator that:
  - [ ] inventories train_* locations and picks smallest location
  - [ ] enumerates **scene tokens** (from `scene` table) within that location
  - [ ] randomly selects **200 scenes** (seeded)
  - [ ] for each scene, generates frame indices `0,8,16,...` (stride=8) with **cap_per_scene=100**
  - [ ] enriches each sample with `location` + `tags` (scenario_tag aligned via lidar_pc)
  - [ ] outputs `index.jsonl`, `scene_list.json`, `plan_config.yaml`, `stats.json`, `tag_stats.json`

### TODO-R (Minimal refactor for meaningful timing)
- [ ] Minimal refactor of single-frame extraction to allow **map_api reuse**:
  - [ ] expose an API like `extract_features(conn, map_api, scene_token, frame_index) -> dict`
  - [ ] keep CLI/debug behavior intact

### TODO-E (Export v0.1)
- [ ] Export script that:
  - [ ] reads the v0.1 plan
  - [ ] loops samples with timeout protection
  - [ ] QC: hard-skip NaN/Inf, shape mismatch, `route_lanes_avails_sum==0`
  - [ ] QC: soft-flag only `route_min_dist > 30m`
  - [ ] writes **one** NPZ file + manifest + metrics

### TODO-V (Visualization)
- [ ] Deferred until after v0.1 NPZ is produced.

---

## 8. v0.1 current status (2026-03-22)

### Plan
- Plan dir: `plans/plan_v0.1_stride8_scene200_cap100_train_boston_20260322_1644/`
- Planned samples: 20,000 (200 scenes × cap_per_scene 100, stride=8)

### Export (small-scale debug run)
A full end-to-end small run **completed successfully** (plan → export → produced NPZ + manifest + metrics):
- Output dir (container):
  `exports/debug_limit3000_timeout5s_clean_20260322_105500/`
- planned: 3000
- kept: 2760
- hard_skipped: 240 (all `route_lanes_avails_sum==0`)
- timeouts: 0
- soft_flag `route_min_dist_gt_30m`: 669
- elapsed: 1743.6s (~29.1 min)
- throughput: 1.58 kept fps
- npz size: 184,078,508 bytes (~175.6 MiB)

**Important note:** the run was slower than expected due to extremely frequent warning logs
(`route_roadblock_correction failed ... NoneType has no attribute 'id'`). For production export, we should
suppress or aggregate these warnings (count them in metrics) to avoid stderr I/O bottlenecks.

## 9. Next after v0.1

After reboot / next session:
- Decide logging policy for production export (silence repetitive warnings).
- Run full 20k export using the same plan.
- Update MB/frame and fps estimates and refine sampling strategy.
