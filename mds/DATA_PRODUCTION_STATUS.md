# Data Production Status (rolling)

## 2026-03-22

### What’s done
- v0.1 plan generation (scene-token level + tags) is working.
- Single-frame extractor was minimally refactored to expose `extract_features(...)` with output regression test (5 cases).
- v0.1 exporter can complete an end-to-end debug run and produce:
  - `data.npz`
  - `manifest.jsonl`
  - `metrics.json`

### Current plan (v0.1)
- Location: train_boston (smallest by ego_pose rows among train_*)
- Sampling: 200 scenes, stride=8, cap_per_scene=100 → 20k planned
- Plan dir: `plans/plan_v0.1_stride8_scene200_cap100_train_boston_20260322_1644/`

### Debug export run (completed)
- Output dir (container): `exports/debug_limit3000_timeout5s_clean_20260322_105500/`
- planned=3000, kept=2760, hard_skipped=240, timeout=0
- soft_flag(route_min_dist_gt_30m)=669
- elapsed=1743.6s, fps_kept=1.58
- npz_size=184,078,508 bytes

### Blocking issue for speed
- Repetitive warning log spam (route_roadblock_correction failed) dominates stderr I/O and reduces throughput.
- For production export, warnings should be suppressed or aggregated (counted in metrics) to recover expected fps.

### Updates (2026-03-23)
- **External disk mount propagation issue fixed**: container couldn't see `/media/zhouyida/新加卷1` until `docker restart nuplan-simulation`.
- **Boston export throughput verified**: on `train_boston` we recovered to **~5.3 fps** on a 500-sample run with timing breakdown:
  - `fps_kept≈5.30`, `extract_total_s≈93.3/94.4s` (extract dominates), `npz_save_s≈1.03s`.
- **Multi-process sharded export implemented** in `scripts/export_v0_1_single_npz.py`:
  - modulo sharding by plan row index: `plan_row_idx % num_shards == shard_id`
  - manifest now includes `plan_row_idx`, `shard_id`, and shard-local `t` (for alignment)
  - added a simple driver script: `scripts/run_export_sharded_mod.sh`
- **N=4 sharded smoke test** (limit=2000, plan prefix) completed with **no overlap / no missing**:
  - planned=2000, kept=1976, hard_skipped=24, timeout=0
  - per-shard planned=500
- **Soft-flag issue** (`route_min_dist_gt_30m`) persists and is treated as a routing-quality problem; current plan is to store it as meta for training-time reweighting rather than blocking export.

### Updates (2026-03-26)
- **Routing empty-route root cause confirmed**: the majority of BFS calls were due to `route_lanes_old` being empty (`avails_sum_old==0`, `rmin_old_m=None`), which correlates with `ego_rb_in_route=false` (route roadblock ids misaligned with ego). Overlap-realign is often a no-op in these cases.
- **Profiling infrastructure added** (gated by `EXTRACT_PROFILE=1`):
  - per-submethod timing (`_timing`) aggregated into shard `metrics.json`
  - BFS trigger flags / reason distributions (`_profile_flags`) aggregated into `metrics.json` and (compact form) optionally written into `manifest.jsonl` for sampling/viz.
- **BFS wall-clock cap introduced**: `BFS_MAX_TIME_S=0.5` (BFS internal timeout) + exporter treats BFS timeout as hard-skip (`qc_error=bfs_timeout`).
  - This avoids spending ~5s per slow BFS case and significantly improves throughput on Boston baselines.
- **DB-locality scheduling**: `EXPORT_SCHEDULE=db_grouped` implemented and validated (reduces cross-DB thrashing).

### Production (Boston 50w planned, sliced)
- Generated Boston plan with 500k planned samples: `plans/plan_v0.1_stride8_scene5000_cap100_train_boston_20260324_2254/`.
- Production strategy: split into **5 slices × 100k planned**; run each slice with `num_shards=8/12`, `db_grouped`, `BFS_MAX_TIME_S=0.5`.
- **Slice01 (N=8)**: `exports/boston50w_prod/slice01_20260326_093747/`
  - planned=100000, kept=79613, hard_skipped=20387
  - elapsed_max≈4181s, fps_kept≈19.04, size≈5.2G
- **Slice02 (N=12)**: `exports_local/boston50w_prod/slice02_N12_20260326_105143/` (exports/ is root-owned)
  - planned=100000, kept=83379, hard_skipped=16621
  - elapsed_max≈3229s, fps_kept≈25.82, size≈5.5G
- **Slice03 (N=12)**: `exports_local/boston50w_prod/slice03_N12_20260326_115618/`
  - planned=100000, kept=78522, hard_skipped=21478
  - timeout=5, bfs_timeout=0
  - elapsed_max≈3336s, fps_kept≈23.54, size≈5.2G
- **Slice04 (N=12)**: `exports_local/boston50w_prod/slice04_N12_20260328_205401/`
  - planned=100000, kept=80615, hard_skipped=19385
  - timeout=0, bfs_timeout=0
  - elapsed_max≈3026.67s, fps_kept≈26.63, size≈5.3G
  - hard_skip_reasons: route_lanes_avails_sum==0 (19385)
- **Slice05 (N=12)**: `exports_local/boston50w_prod/slice05_N12_20260328_221211/`
  - planned=100000, kept=81895, hard_skipped=18105
  - timeout=0, bfs_timeout=0
  - elapsed_max≈3029.17s, fps_kept≈27.04, size≈5.4G
  - hard_skip_reasons: route_lanes_avails_sum==0 (18105)

### Ops note
- `exports/` directory is currently root-owned (not writable by uid=1000 container runs). Using `exports_local/` until permissions are normalized.

### Next
- Restart **slice04** (delete partial outputs first), then run slice05.
- Normalize output directory permissions (`exports/` vs `exports_local/`) and standardize run metadata.
- (Optional) Add a cheap deterministic route realignment (not BFS) to reduce empty-route rate, then re-evaluate whether BFS can be made truly rare.
