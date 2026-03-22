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

### Next
- Add flat per-frame meta fields into `manifest.jsonl` (routing quality bucket + tags) to support training-time sampling/reweighting.
- Run 3k/20k export using **N=4 sharding**, then aggregate QC across shards.
