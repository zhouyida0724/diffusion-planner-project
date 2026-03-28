# TRAINING.md (Diffusion Planner repro – using our exported features)

**Status:** draft (paper-first, but training pipeline is built around our exported sharded NPZ + manifest).

## 1) 目标与范围

### 目标
- 复现论文：**Diffusion-Based Planning for Autonomous Driving with Flexible Guidance** (ICLR 2025)
- 在 **nuPlan** 上训练可用于 closed-loop planning 的 Diffusion Planner（先以 Boston 50w 导出数据跑通训练闭环，再扩展规模/地域）。

### 当前范围（我们先做到什么）
- ✅ 数据生产：从 nuPlan 场景导出 **sharded** 数据集（`data.npz + manifest.jsonl + metrics.json`）
- 🔄 训练：实现与官方一致的训练数据流 + 训练 loop，并用“等价性测试”对齐官方实现
- ⏳ 评测：训练后在 nuPlan closed-loop（或最小 offline 指标）上 sanity check

### 非目标（暂不做 / 后续再做）
- 不在训练阶段“在线修复 routing 异常帧”：soft flag 只做 tag/分桶。
- 不把生产产物（exports/benchmarks）纳入 git。

---

## 2) 链接（权威来源）
- Paper (arxiv PDF): https://arxiv.org/pdf/2501.15564
- Official repo: https://github.com/ZhengYinan-AIR/Diffusion-Planner
- Official README training entrypoints:
  - `data_process.sh`
  - `torch_run.sh`

> 备注：我们仓库内已有 `training/` 目录**不作为参考**（历史遗留/误导），训练实现以论文 + 官方 repo 为准。

---

## 3) 环境准备（容器/依赖/路径）

### 强约束（本项目约定）
- **训练必须在 Docker `training` 容器内执行**（不要在宿主机直接跑）。
- 数据导出/可视化在 `nuplan-simulation` 容器；训练在 `training` 容器。

### 最小自检（training 容器内）
- GPU 可用：
  - `nvidia-smi`
  - `python -c "import torch; print(torch.cuda.is_available())"`
- 记录版本：torch/cuda/driver/pytorch-lightning（若使用）

### 目录约定（我们自己的）
- 训练输入（导出数据）：`/workspace/exports_local/boston50w_prod/...`（容器内路径示例；以实际挂载为准）
- 训练输出：`/workspace/training_outputs/<exp_name>/...`

---

## 4) 数据准备（以我们导出数据为主）

### 4.1 数据来源（Boston 50w，5×100k planned）
每个 slice 目录结构（示例）：
```
exports_local/boston50w_prod/sliceXX_N12_<ts>/
  shards/
    shard_000/
      data.npz
      manifest.jsonl
      metrics.json
      RUN_INFO.json
      run.stderr.log
    ...
    shard_011/
```

已完成 slice（截至当前文档编写时）：
- slice01: `exports/boston50w_prod/slice01_20260326_093747/`（N=8）
- slice02: `exports_local/boston50w_prod/slice02_N12_20260326_105143/`
- slice03: `exports_local/boston50w_prod/slice03_N12_20260326_115618/`
- slice04: `exports_local/boston50w_prod/slice04_N12_20260328_205401/`
- slice05: **running / TBD**（产出后补充路径与统计）

### 4.2 manifest 对齐与字段
- `manifest.jsonl` 每行对应一条样本（与 `data.npz` 内 batch 维通过 `t`/row_idx 对齐）。
- 强烈建议训练侧保留并读取 manifest 的关键字段：
  - `plan_row_idx`, `t`, `db_path`, `scene_token`, `frame_id`（或 sample_id）
  - QC 字段：`qc_hard_skip`, `qc_error`, `qc_flags`（soft tags）

### 4.3 hard/soft 的使用策略
- **hard skip**（例如 `route_lanes_avails_sum==0`）在 exporter 已剔除（但训练侧仍建议 assert/统计验证）。
- **soft flag**（例如 `route_min_dist_gt_30m`）不在线修复：训练时通过采样比例/分桶控制。

### 4.4 数据集切分建议（先简单可审计）
建议优先按 **db_path 或 scene_token** 做 train/val 切分，避免时间邻近泄漏：
- Train: slice01–slice04
- Val: slice05（或从 slice04/05 中抽取固定 db 子集）

> 最终切分策略要与论文设置对齐；当前先保证可复现与可审计。

---

## 5) 训练（含：预处理 + 代码实现概要 + 启动方式）

### 5.1 训练输入适配（从 sharded data.npz 读取）
我们导出的格式是 **“每 shard 一个堆叠 data.npz”**，因此训练 dataloader 推荐实现为：
- index 构建：扫描 `shards/shard_*/manifest.jsonl`，得到全局样本表（包含 shard_path + t）
- `__getitem__`：打开对应 shard 的 `data.npz`，按 t 取出单样本各字段
- 注意：避免每次 `np.load` 反复打开文件导致 IO 爆炸
  - 方案 A：每个 worker 维护 LRU cache（npz handle / mmap）
  - 方案 B：预先把 shard 转成更适合随机访问的格式（如 zarr / webdataset）——后续再做

> 这里会实现一个最小可用版本，先确保逻辑正确，再做性能优化。

### 5.2 训练代码实现概要（与官方对齐）
我们会按官方 repo 的训练入口（`torch_run.sh` -> `train_predictor.py`）梳理并对齐：
- 数据：输入字段、shape、归一化、mask/avails
- 模型：DiT-based backbone + condition fusion（按论文/官方实现）
- Diffusion：噪声注入方式、time embedding、预测目标（x0/eps/score 等）
- Loss：各项 loss 的定义与权重
- 优化器/调度器：AdamW / cosine / warmup 等（以官方为准）

### 5.3 预处理（放在训练流程里，而不是单独章节）
因为我们以导出数据为主，预处理分两层：
1) **必要预处理**（训练必须）
   - 读取 manifest，过滤/分桶/重采样
   - 构建 train/val index
2) **可选预处理**（性能/质量）
   - cache index
   - 统计每个字段的 mean/std（若论文需要）

### 5.4 与官方实现一致性的“可证明”检查
我们会做三步对齐，避免“看起来能跑但其实不一致”：
1) **Forward/Loss 对齐**：固定 seed + 固定 batch，对比 shape + 数值统计
2) **1-step update 对齐**：跑一步 optimizer，比较关键层权重 diff
3) **加载官方 ckpt（可选）**：如果要二进制兼容，需对齐 module name/state_dict

### 5.5 启动方式（先占位，待我们训练代码落地后给出最终命令）
- Sanity run：小数据 + 小 steps（验证 loss/NaN/吞吐）
- Full run：多卡 torchrun
- Resume：从 checkpoint 恢复

---

## 6) 评估与对照

最小验收（必须）：
- 能跑通训练 N steps，不 NaN
- loss 曲线合理（下降或稳定）
- 抽样可视化输入/输出（与我们已有 `visualize_npz.py` 兼容）

对照（推荐）：
- 使用官方 repo + 官方 checkpoint 做一次相同设置的 eval sanity check（确认环境无偏差）

---

## 7) 复现记录与排障

每次训练 run 需要记录：
- 数据版本：slice 列表 + 每 slice 的 `RUN_INFO.json` / metrics 汇总
- 代码版本：git commit（本 repo） + 关键依赖版本
- 配置：完整 config dump（保存到输出目录）
- 机器信息：GPU 型号、driver/cuda

常见问题：
- 路径挂载错误（/media/zhouyida/新加卷1 vs 空目录）
- 权限（exports/ root-owned；训练请读 exports_local）
- IO 瓶颈（随机访问 npz）；worker/cache 策略

---

## Appendix: slice-level metrics quick table (optional)
（后续可自动生成/追加）
