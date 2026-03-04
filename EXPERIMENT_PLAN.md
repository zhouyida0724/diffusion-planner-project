# Diffusion Planner vs IDM 对比实验计划

## 目标
- 使用 Diffusion Planner 和 IDM 各跑 200 个场景
- 对比两者在 nuPlan 场景上的闭环仿真表现

## 场景信息
- 总场景数：2200+
- 计划运行：200 个
- 数据集：nuplan-devkit (已安装在 Docker 镜像中)

## 步骤

### 1. 准备环境
```bash
# 进入 Docker 容器
./scripts/docker_setup.sh enter

# 验证 nuplan-devkit 可用
python -c "import nuplan_db; print('OK')"
```

### 2. 准备场景列表
从 2200+ 场景中随机选取 200 个，或按场景类型均匀采样。

### 3. 修改仿真配置
修改 `scripts/run_simulation.sh` 或创建新脚本：
- 支持指定 planner 类型 (IDM / Diffusion Planner)
- 支持指定输出目录

### 4. 运行 IDM 基线
```bash
# 运行 IDM planner，200 个场景
./scripts/run_simulation.sh --planner idm --scenarios 200 --output idm_results/
```

### 5. 运行 Diffusion Planner
```bash
# 运行 Diffusion Planner，200 个场景
./scripts/run_simulation.sh --planner diffusion_planner --scenarios 200 --output diffusion_results/
```

### 6. 收集指标
对比指标：
- 平均碰撞率 (collision rate)
- 平均舒适度 (comfort metrics: jerk, acceleration)
- 平均进度 (progress along route)
- 平均速度违规 (speed limit violation)
- 规划时间 (planning time)

### 7. 生成报告
使用 nuBoard 可视化或生成 CSV/图表对比。

## 状态确认

### ✅ 问题 1: Diffusion Planner 模型权重
- **状态**: ✅ 已有
- **位置**: `/home/zhouyida/.openclaw/workspace/diffusion-planner-project/checkpoints/model.pth` (48MB)
- **配置**: `checkpoints/args.json`
- **注意**: 模型存在但**尚未集成到 nuPlan 仿真中**

### ✅ 问题 2: 场景选取策略
- **建议**: 随机抽取 200 个，或使用 test split 前 200 个
- **待定**: 需确认场景类型分布

### ✅ 问题 3: GPU 支持
- **状态**: ✅ 可用
- **GPU**: NVIDIA GeForce RTX 4080 (16GB)
- **CUDA**: 12.2

### ✅ 问题 4: Diffusion Planner 集成状态
- **状态**: ❌ **需要开发集成代码**
- **当前**: 仿真脚本仅支持 IDM planner
- **TODO**: 需要编写 Diffusion Planner wrapper 接入 nuPlan

---

## 明日任务（重新规划）

### Day 1: 集成 Diffusion Planner
1. 编写 `DiffusionPlanner` 类，继承 nuPlan 的 `AbstractPlanner`
2. 加载 `checkpoints/model.pth` 权重
3. 实现 `compute_trajectory` 方法
4. 修改 `run_simulation.sh` 支持 `--planner` 参数

### Day 2: 测试运行
1. 先跑 5 个场景验证流程
2. 调试问题

### Day 3-4: 对比实验
1. 跑 200 个 IDM 场景
2. 跑 200 个 Diffusion Planner 场景
3. 对比指标

## 注意事项
- Docker 容器需要 GPU 支持（--gpus all）
- 建议先跑 5-10 个场景测试流程
- 记录每个场景的耗时
