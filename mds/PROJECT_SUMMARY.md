# Diffusion Planner 项目总结

## 项目目标

复现论文和 GitHub 源码中的算法：

- **论文**: Diffusion Planner (待补充 arxiv 链接)
- **GitHub 源码**: https://github.com/zhouyida0724/diffusion-planner-project

---

## 项目进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 1. 环境配置 | ✅ 完成 | Docker 容器、nuPlan 数据、依赖安装 |
| 2. 闭环仿真跑通 | ✅ 完成 | IDM + Diffusion Planner 仿真 |
| 3. 场景下载和分析 | ✅ 完成 | nuPlan mini 数据集分析 |
| 4. 特征提取 | 🔄 进行中 | 从场景提取训练特征 |
| 5. 模型训练 | ⏳ 待进行 | 使用提取的特征训练模型 |

---

## Git 追踪的文件目录

```
diffusion-planner-project/          # Git 仓库根目录
├── .git/                          # Git 版本控制
├── .gitignore
├── README.md                      # 项目说明
├── PROJECT_SUMMARY.md             # 本文件
├── EXPERIMENT_PLAN.md             # 实验计划
├── TRAINING_PLAN.md                # 训练计划
├── FEATURE_EXTRACTION_GUIDE.md    # 特征提取指南
├── agent.md                       # 智能体行为规范
├── nuplan-visualization/          # nuPlan 可视化代码 (精简版)
│   └── nuplan/
│       └── planning/
├── scripts/                       # 运行脚本
│   ├── extract_single_frame/      # 特征提取脚本
│   │   └── extract_single_frame.py
│   └── visualize_npz.py           # 可视化脚本
├── data/                          # 场景 token 列表
│   └── scenarios_200.txt
├── checkpoints/                   # 模型权重
│   ├── model.pth
│   └── args.json
├── docker_setup.sh               # Docker 容器管理
└── training/                      # 训练代码 (待添加)
```

---

## 生成数据的文件目录

**注意：以下目录已被 .gitignore 忽略，不追踪到 Git**

```
/home/zhouyida/.openclaw/workspace/diffusion-planner-project/
├── data/                         # 原始数据
│   └── nuplan/                  # nuPlan 数据集
│       ├── data/cache/mini/      # 场景数据库 (.db 文件)
│       └── maps/                # 地图数据
│
├── data_process/                # 处理后的数据
│   ├── npz_scenes/              # 提取的特征 (.npz 文件)
│   │   └── *.npz
│   └── run_extract.py
│
├── validation_output/            # 验证输出
│   └── *.png                    # 可视化图片
│
└── training_outputs/            # 训练输出
    ├── checkpoints/              # 训练检查点
    └── logs/                    # 训练日志
```

**映射关系**：
- 容器内 `/workspace/` → 宿主机 `/home/zhouyida/.openclaw/workspace/diffusion-planner-project/`
- 容器内 `/workspace/data_process/npz_scenes/` = 宿主机 `.../diffusion-planner-project/data_process/npz_scenes/`

---

## 核心脚本

### 特征提取 (在 nuplan-simulation 容器内执行)
```bash
# 特征提取
python3 /workspace/scripts/extract_single_frame/extract_single_frame.py

# 可视化
python3 /workspace/scripts/visualize_npz.py <npz_path> <png_path>
```

### 训练 (在 training 容器内执行)
```bash
# 待补充训练命令
```

---

## Docker 容器

| 容器名 | 用途 |
|--------|------|
| nuplan-simulation | 特征提取、可视化、仿真 |
| training | 模型训练 |

**管理命令**:
```bash
./scripts/docker_setup.sh enter   # 进入 nuplan-simulation 容器
./scripts/docker_setup.sh rebuild # 重新构建镜像
./scripts/docker_setup.sh push   # 推送镜像
```

---

## 注意事项

1. **所有脚本执行必须在 Docker 容器内进行**
2. **特征提取 → nuplan-simulation 容器**
3. **模型训练 → training 容器**
4. 生成的数据文件不追踪到 Git (已加入 .gitignore)
