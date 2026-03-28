# TRAINING.md (Diffusion Planner repro – using our exported features)

**Status:** WIP. Paper-first, but the primary training input is **our exported sharded NPZ + manifest**.

## Checklist (where we are)

### Data production (done)
- [x] Boston 50w export **slice01–slice05** completed (sharded NPZ + manifest + metrics)

### Docs (done)
- [x] `TRAINING.md` structure agreed (data-prep is based on our exports; preprocess + implementation summary live under Training)
- [x] Expanded **5.2** into concrete implementation plan (data/model/diffusion/loss/opt/ddp/ckpt + equivalence tests)
- [x] Added **deployment to nuPlan closed-loop sim** section (planner wrapper + runtime feature extraction reuse + runner + nuboard)

### Next executable steps
- [ ] **Verify training container** from `scripts/training_docker_setup.sh` (GPU/torch/imports/data mounts/write perms) and fix the script if broken
- [ ] Implement **our** training code **inside this repo** (new clean module + clear docs structure) + equivalence tests vs official
- [ ] First sanity run: 100–1000 steps, verify no-NaN + throughput + basic qualitative checks

---

## 1) 目标与范围

### 目标
- 复现论文：**Diffusion-Based Planning for Autonomous Driving with Flexible Guidance** (ICLR 2025)
- 在 **nuPlan** 上训练可用于 closed-loop planning 的 Diffusion Planner：先用 Boston 50w 导出数据跑通“训练→部署→闭环仿真”的闭环，再扩展规模/地域。

### 当前范围（我们先做到什么）
- ✅ 数据生产：从 nuPlan 场景导出 **sharded** 数据集（`data.npz + manifest.jsonl + metrics.json`）
- 🔄 训练：实现与官方一致的训练数据流 + 训练 loop，并用“等价性测试”对齐官方实现
- ⏳ 部署与评测：训练后接入 nuPlan closed-loop simulation，跑指标与 nuboard 可视化

### 非目标（暂不做 / 后续再做）
- 不在训练阶段“在线修复 routing 异常帧”：soft flag 只做 tag/分桶。
- 不把生产产物（exports/benchmarks）纳入 git。

---

## 2) 链接（权威来源）
- Paper (arxiv PDF): https://arxiv.org/pdf/2501.15564
- Official repo: https://github.com/ZhengYinan-AIR/Diffusion-Planner

> 备注：本仓库内已有 `training/` 目录**不作为参考**（历史遗留/误导）。训练实现以论文 + 官方 repo 为准。

---

## 3) 环境准备（容器/依赖/路径）

### 强约束（本项目约定）
- **训练必须在 Docker `training` 容器内执行**（不要在宿主机直接跑）。
- 数据导出/可视化在 `nuplan-simulation` 容器；训练在 `training` 容器。

### 最小自检（training 容器内）
- GPU 可用：
  - `nvidia-smi`
  - `python -c "import torch; print(torch.cuda.is_available())"`
- 记录版本：torch/cuda/driver（以及 lightning/accelerate 等，若使用）

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
- slice05: `exports_local/boston50w_prod/slice05_N12_20260328_221211/` (planned=100000, kept=81895, hard=18105)

### 4.2 manifest 对齐与字段（训练侧必须读）
- `manifest.jsonl` 每行对应一条样本；建议训练侧把 manifest 作为“事实来源”。
- 关键字段（建议至少保留）：
  - 对齐审计：`plan_row_idx`, `t`
  - 定位回溯：`db_path`, `scene_token`, `frame_id`（或 sample_id）
  - QC：`qc_hard_skip`, `qc_error`, `qc_flags`（soft tags）

### 4.3 hard/soft 的使用策略
- hard skip（例如 `route_lanes_avails_sum==0`）在 exporter 已剔除；训练侧仍建议做统计断言（防止数据混入）。
- soft flag（例如 `route_min_dist_gt_30m`）不在线修复：训练时通过采样比例/分桶控制。

### 4.4 数据集切分建议（先简单可审计）
优先按 **db_path 或 scene_token** 做 train/val 切分，避免时间邻近泄漏：
- Train: slice01–slice04
- Val: slice05（或从 slice04/05 抽固定 db 子集）

---

## 5) 训练（含：预处理 + 代码实现概要 + 启动方式）

### 5.1 训练输入适配（从 sharded `data.npz` 随机读取）
我们的导出是“每 shard 一个堆叠 NPZ”，因此 dataloader 推荐实现为：
- **IndexBuilder**：扫描 `shards/shard_*/manifest.jsonl` → 生成全局样本表（每行至少含 `{shard_dir, t, plan_row_idx, qc_*}`），并落盘缓存（避免每次启动全量扫描）。
- **NPZBackend**：`__getitem__` 根据 `(shard_dir, t)` 从 `data.npz` 取单样本（注意 shape / avails）。
- **性能底线**：严禁每次 sample 都 `np.load` 打开/关闭。
  - 在 DataLoader worker 内做 LRU（`np.load(..., mmap_mode='r')`）或 per-worker cache。

### 5.2 训练代码实现概要（展开：每块怎么落地）
> 目标：我们自己写 training code，但能“证明”与论文/官方实现对齐。

#### 5.2.1 Data（字段/shape/归一化/mask）
- **dataset 输出统一 dict**：`{features..., target..., meta...}`，其中 `meta` 至少包含 `plan_row_idx/db_path/scene_token` 用于 debug。
- **mask/avails**：训练时必须显式使用（attention mask 或乘 mask），并在 loss 中忽略不可用 timestep/agent。
- **Normalize**：先不做神秘标准化，只做：
  - 坐标系一致性校验（ego-centric / global 的约定）
  - NaN/Inf assert + 合理范围 clamp（可选）
  - 若官方/论文要求 mean/std：提供 `compute_stats.py` 生成 `stats.json` 并在训练加载。

#### 5.2.2 Model（DiT backbone + condition fusion）
建议拆成 4 个显式模块（便于对齐/单测）：
1) `ConditionEncoder`：把动态参与者 + 静态 polyline（lanes/route_lanes/traffic lights）编码成 token。
2) `TrajectoryRepresentation`：明确要生成的未来轨迹张量定义（ego-only vs joint participants）。
3) `DiTBackbone`：输入 `x_t + cond_tokens + t_embed`，输出预测（`eps` 或 `x0`，以官方为准）。
4) `OutputHead`：把 backbone 输出映射回轨迹张量维度。

#### 5.2.3 Diffusion（训练目标与 scheduler）
- `NoiseScheduler`：管理 `betas/alphas/alpha_bar`，提供 `q_sample(x0,t,noise)`。
- **训练 target**：必须明确且与官方一致（predict `eps` / `x0` / `v`）。
- 推理 sampler（部署会用）：DDIM/DDPM 或 DPM-Solver（论文强调实时性）。

#### 5.2.4 Loss（每项 loss 的定义、mask、log）
- `loss_diffusion = MSE(pred, target)`（按 eps/x0 定义）并带 mask。
- 可选正则（以论文/官方为准）：smoothness/jerk 等。
- **必须分项 log**：`loss_total/loss_diff/loss_reg/...`，否则后期无法定位。

#### 5.2.5 Optimizer/Scheduler（以官方为准）
- AdamW +（可选）no_decay 参数组（LayerNorm/bias）。
- cosine + warmup（如官方采用）。
- gradient clip（如官方采用）。

#### 5.2.6 DDP/Checkpoint/复现信息
- 入口用 `torchrun`（DDP）。
- checkpoint 必须保存：`model/optimizer/scheduler/step` + **config snapshot** + **data index hash**。

#### 5.2.7 与官方实现一致性的“可证明”检查（强制项）
1) **Forward/Loss 对齐**：固定 seed + 固定 batch，对比：shape + 数值统计（mean/std/max/NaN）。
2) **1-step update 对齐**：跑一步 optimizer，比关键层权重 diff。
3) （可选）**state_dict 兼容**：若要加载官方 ckpt，则需对齐 module 命名/参数组织。

### 5.3 预处理（放在训练流程里）
- 必要预处理：读 manifest → 过滤/分桶/重采样 → 构建 train/val index。
- 可选预处理：cache index、统计 mean/std、把 NPZ 转换为更友好的格式（zarr/webdataset）。

### 5.4 启动方式（先给骨架；等训练代码落地后补最终命令）
- Sanity run：小数据 + 100–1000 steps（验证 loss/NaN/吞吐）。
- Full run：多卡 torchrun。
- Resume：从 checkpoint 恢复。

---

## 6) 部署到 nuPlan 闭环仿真（我们自己的 nuPlan-simulation 工程入口）

> 本节选择“后者”：优先对齐 **我们现有的 `nuplan-simulation` 工程/入口**，把训练出的模型作为一个 planner 挂进去跑 closed-loop。

### 6.1 产物约定（训练输出）
训练输出目录（建议）：
- `training_outputs/<exp_name>/`
  - `checkpoints/last.ckpt` 或 `model.pth`
  - `config.yaml`（完整快照）
  - `stats.json`（若用）

### 6.2 Inference Wrapper（模型推理封装）
实现一个最小推理类，例如 `DiffusionPlannerPolicy`：
- `load(ckpt_path, device)`
- `build_cond(features)`（把 runtime features 变成 model 输入）
- `sample(cond, num_steps, guidance_cfg) -> future_traj`

### 6.3 Runtime Feature Extraction（运行时特征提取必须与训练一致）
- **必须复用**我们已经验证过的单帧特征提取逻辑（当前在 export pipeline 里）。
- 推荐做法：把单帧 `extract_features(...)` 抽成一个可 import 的函数/模块：
  - 训练侧：dataset 读 NPZ（离线）
  - 仿真侧：planner 每 tick 调 `extract_features(...)`（在线）
- 强制断言：每个字段 shape 与训练一致（避免 silent mismatch）。

### 6.4 Planner 适配（nuPlan closed-loop）
实现/注册一个 nuPlan planner（以 nuPlan devkit 的 planner API 为准）：
- `initialize(...)`
- `compute_trajectory(observation, map_api, route, ...) -> Trajectory`

`compute_trajectory` 内部步骤：
1) 从 observation/map/route 提取 features（复用 extract_features）
2) policy.sample 生成未来轨迹
3) 转成 nuPlan 的 `Trajectory` 类型返回（包含时间步与姿态）

### 6.5 Runner（跑仿真 + nuboard）
- 提供一个 runner 脚本（bash/python），参数至少包含：
  - checkpoint 路径
  - 场景集合（val set）
  - 仿真 horizon/step
  - 输出目录
- 产物：仿真 metrics + 可用 nuboard 回放。

### 6.6 部署后最小验收
- 单场景跑通不 crash
- 轨迹输出合理（无 NaN、无爆炸）
- metrics 正常产出
- nuboard 可视化可打开

---

## 7) 评估与对照

最小验收（必须）：
- 能跑通训练 N steps，不 NaN
- loss 曲线合理
- 抽样可视化输入/输出（与 `visualize_npz.py` 兼容）

对照（推荐）：
- 用官方 repo + 官方 checkpoint 做一次 sanity（确认环境无偏差；并非要完全复刻官方 pipeline）。

---

## 8) 复现记录与排障

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
（slice05 完成后补一个简表）
