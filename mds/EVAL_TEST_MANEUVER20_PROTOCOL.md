# EVAL_TEST_MANEUVER20_PROTOCOL.md

目的：用**同一套 60 个 test maneuver 场景**同时跑 open-loop 与 closed-loop，直观回答：
- 是否存在“open-loop 很好但 closed-loop 很差”的 gap（尤其在 left/right turning）。

本协议强调：**场景集固定、配置固定、产物路径固定**，后续任何 ckpt / ablation 都按同一口径对比。

---

## 1) 固定场景集（canonical）

- Rich CSV（open-loop 用）：
  - `outputs/eval/test_maneuver20/test_maneuver20_scenarios_final_local.csv`
    - 含 group(left/right/straight)、scene_token、frame_index、db_path(本机可访问) 等。
- Tokens CSV（closed-loop 用）：
  - `outputs/eval/test_maneuver20/test_maneuver20_tokens_final_local.csv`
    - 仅一列 `token`，每行一个 lidar_pc_token。

说明：open-loop 与 closed-loop 必须使用**同一份 60 场景**（以 `*_final_local.csv` 为准）。

---

## 2) Open-loop（固定配置，*replay aligned with closed-loop*）

这里的 open-loop 定义为：**运算量与 closed-loop 同级别**（逐 tick 运行 planner），
但 ego 当前状态来自 DB 的 GT（replay），不做动力学 rollout。

- 脚本：`scripts/eval/open_loop_replay_test_maneuver20.py`
- 固定：`--diffusion-steps 10`，并在每个 tick 输出 ADE/FDE@1/3/5/8s。

建议输出命名（避免覆盖）：
- frame-level：`outputs/eval/test_maneuver20/open_loop_replay_frames_sampler10_<ckpt_tag>.jsonl`
- scenario-mean：`outputs/eval/test_maneuver20/open_loop_replay_scenario_mean_sampler10_<ckpt_tag>.jsonl`

示例：
```bash
python3 scripts/eval/open_loop_replay_test_maneuver20.py \
  --scenarios-csv outputs/eval/test_maneuver20/test_maneuver20_scenarios_final_local.csv \
  --ckpt <ABS_OR_REL_CKPT_PATH> \
  --diffusion-steps 10 \
  --device cuda \
  --max-frames 80 \
  --frame-stride 2 \
  --output-frames outputs/eval/test_maneuver20/open_loop_replay_frames_sampler10_<ckpt_tag>.jsonl \
  --output-scenarios outputs/eval/test_maneuver20/open_loop_replay_scenario_mean_sampler10_<ckpt_tag>.jsonl
```

产物：
- frame-level：每个 (scenario, frame) 一行 JSONL。
- scenario-level：每个 scenario 一行 JSONL（对 frame-level 做均值聚合），用于和 closed-loop 做对比。

---

## 3) Closed-loop（固定配置）

- runner：`scripts/run_diffusion_simulation.py`（wrapper，实际实现是 `scripts/sim/run_diffusion_simulation.py`）
- 固定：`--sampling-steps 10`
- 场景来源：`--scenarios_file outputs/eval/test_maneuver20/test_maneuver20_tokens_final_local.csv`

示例：
```bash
python3 scripts/run_diffusion_simulation.py \
  --planner diffusion_planner \
  --ckpt <ABS_OR_REL_CKPT_PATH> \
  --sampling-steps 10 \
  --scenarios_file outputs/eval/test_maneuver20/test_maneuver20_tokens_final_local.csv
```

closed-loop 产物：nuPlan 默认会写到 `data/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/<timestamp>/...`
其中包含：
- `runner_report.parquet`
- `aggregator_metric/*.parquet`（用于 merge）
- `summary/summary.pdf`
- `.nuboard`

---

## 4) Open+Closed 合并报表（固定方式）

- 脚本：`scripts/eval/merge_open_closed_test_maneuver20.py`
- 输入：open-loop jsonl + closed-loop aggregator parquet
- 输出：per-scenario 与 group summary 两张 CSV

示例：
```bash
python3 scripts/eval/merge_open_closed_test_maneuver20.py \
  --open-loop-jsonl outputs/eval/test_maneuver20/open_loop_metrics_sampler10_<ckpt_tag>.jsonl \
  --closed-loop-agg <CLOSED_LOOP_DIR>/aggregator_metric/aggregator_metric.parquet \
  --out-csv outputs/eval/test_maneuver20/open_closed_per_scenario_<ckpt_tag>.csv \
  --out-summary-csv outputs/eval/test_maneuver20/open_closed_group_summary_<ckpt_tag>.csv
```

---

## 5) 第一版 baseline（dropout 之前的最大训练 ckpt）

当前工作区内可用、且在日志中被用作“较大规模全量训练”的 checkpoint：
- `outputs/training/mix3city_allcache_norm_e10_bs64_bf16_v2_resume180000_tbfix_samplerFE3000_s10_n128_tb_img5000_samplerDenoise_v3/checkpoint_step_446099.pt`

建议把它作为第一版评测 baseline，产物 tag：`ckpt446099`。
