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
  - `plans/<plan_name>/shards/shard_000.jsonl ...` (optional split index files)

### Step C — Export (sharded)
**Goal**: run feature extraction on each shard with timeouts and lightweight checks.

- **Input**: `index.jsonl` (or `shard_*.jsonl`), extraction code
- **Output (per shard)**:
  - `exports/<plan_name>/shards/shard_000/data.npz` (multi-frame arrays)
  - `exports/<plan_name>/shards/shard_000/manifest.jsonl` (per-sample metadata + flags)
  - `exports/<plan_name>/shards/shard_000/metrics.json` (aggregate stats)
  - `exports/<plan_name>/shards/shard_000/run.stderr.log`

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

---

## 3. Multi-process sharded export (v0.1+)

### 3.1 Design goal
We need to reduce wall-clock time while keeping:
- **reproducibility** (same plan + same shard params → same samples)
- **no overlap / no missing** samples
- **easy resume** (rerun only failed shards)

### 3.2 Recommended sharding rule (no overlap / no missing)
Use **deterministic modulo sharding** on the plan row index.

Let plan rows be 0-based `plan_row_idx` in `index.jsonl`.
For `num_shards = N`:
- assign row `i` to shard `sid = i % N`

Each export worker runs with a fixed `--shard-id sid --num-shards N` and processes only its own rows.

This guarantees:
- no overlap: a row has exactly one `sid`
- no missing: union over `sid=0..N-1` covers all rows

### 3.3 Per-shard outputs
Each shard writes to its own directory:

`exports/<plan_name>/shards/shard_{sid:03d}/`
- `data.npz`
- `manifest.jsonl`
- `metrics.json`
- `RUN_INFO.json`
- `run.stderr.log`

Manifest should contain alignment fields:
- `plan_row_idx` (row number in index.jsonl)
- `shard_id` (sid)
- `t` (index within shard's NPZ, 0..kept-1)
- `sample_id`

This makes downstream joins robust and debuggable.

### 3.4 Driver (single machine)
In `nuplan-simulation` container, start N workers in parallel.
Example (N=4):

```bash
PLAN=plans/<plan_name>
OUT_ROOT=exports/<plan_name>/shards
N=4
for SID in 0 1 2 3; do
  python3 scripts/export_v0_1_single_npz.py \
    --plan $PLAN \
    --out $OUT_ROOT/shard_$(printf "%03d" $SID) \
    --num-shards $N \
    --shard-id $SID \
    --limit 2000 \
    2> $OUT_ROOT/shard_$(printf "%03d" $SID)/run.stderr.log &
done
wait
```

Notes:
- `--limit` applies to the first M rows of the plan (useful for smoke tests). Shards will split only within that prefix.
- Each worker must set quiet log mode to avoid I/O bottleneck:
  - `EXTRACT_LOG_STYLE=quiet EXTRACT_WARN_PRINT_N=0`

### 3.5 Resume strategy
If `metrics.json` exists and marks shard as success, skip rerun; otherwise rerun only that shard.

---

## 4. v0.1 execution: ~20k frames (scene-level selection)

### 4.1 Sampling for v0.1 (finalized requirement)
We do **scene-level selection** and then export **uniform stride within each scene**:

1) Choose a target location (start with the smallest location by ego_pose frames).
2) Randomly select **N scenes = 200** (scene tokens from DB `scene` table).
3) For each selected scene, export frame indices within that scene's log:
   - `0, 8, 16, 24, ...` (stride = 8)
   - **cap_per_scene = 100**

### 4.2 Success criteria
- Export completes without hangs (timeouts prevent deadlocks)
- No NaN/Inf in outputs (hard skip)
- route_lanes truly missing (`route_lanes_avails_sum==0`) treated as hard skip
- QC summary produced (skip counts + soft flag counts)
