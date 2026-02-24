# Diffusion Planner 项目总结

## 项目结构

```
diffusion-planner-project/
├── nuplan-visualization/     # 可视化代码 (从 nuplan-devkit 精简)
│   └── nuplan/
│       ├── planning/         # 仿真、评测、可视化代码
│       └── diffusion_planner/  # Diffusion Planner 推理代码 (已独立)
├── diffusion_planner_inference/  # 可删除 (已不用)
├── data/                    # 场景 token 列表
│   └── scenarios_200.txt
├── scripts/                 # 运行脚本
│   ├── run_simulation.py    # IDM 仿真
│   ├── run_diffusion_sanner.py  # Diffusion Planner 仿真
│   └── run_nuboard.sh      # nuBoard 可视化
├── docker_setup.sh         # Docker 容器管理
└── checkpoints/             # 模型权重
```

## 关键修复

### 1. nuplan-devkit 解耦
- 删除了 nuplan-devkit 依赖
- 改为使用 nuplan-visualization (精简版)

### 2. SimulationLog 修复
- 问题：pickle 保存了 planner 对象，导致 nuBoard 加载时找不到 diffusion_planner 模块
- 修复：修改 `simulation_log_callback.py`，保存时设置 `planner=None`

### 3. Diffusion Planner 导入修复
- 问题：原代码使用 `from diffusion_planner.xxx` 绝对导入
- 修复：改为相对导入 `from .xxx`，使代码完全独立

### 4. Hydra 参数问题
- 问题：命令行指定 `--scenario` 参数时引号转义问题
- 修复：修改配置文件 `one_hand_picked_scenario.yaml`

## 运行命令

```bash
# 进入容器
./docker_setup.sh enter

# 运行仿真
./scripts/run_simulation.py --scenario=310cdc079b215731
./scripts/run_diffusion_simulation.py --scenario=310cdc079b215731

# 运行可视化
./scripts/run_nuboard.sh

# 指定场景数量
./scripts/run_simulation.py --num=5
./scripts/run_diffusion_simulation.py --num=5
```

## Docker 镜像

- 镜像名：`zhouyida/nuplan-simulation:latest`
- 每次修改后需要 `./docker_setup.sh rebuild && ./docker_setup.sh push`

## 注意事项

1. 删除根目录的 `diffusion_planner/` 不影响运行（代码已在 nuplan-visualization 里）
2. `diffusion_planner_inference/` 目录可删除（已不使用）
3. nuplan-devkit 已移到项目外

## 问题排查

- 如果 nuBoard 加载 Diffusion Planner 数据转圈：检查 PYTHONPATH 是否包含 nuplan-visualization
- 如果仿真启动失败：检查 diffusion_planner 代码的相对导入是否正确
