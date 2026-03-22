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

### Next
- After reboot: finalize logging policy and run full 20k export.
