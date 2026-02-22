# 数据集文档

## 目录结构

```
data/
├── nuplan/           # NuPlan 数据集
│   ├── nuplan-v1.1_mini.zip    # 已下载
│   ├── nuplan-v1.1_train/      # 训练集 (~170GB)
│   ├── nuplan-v1.1_val/        # 验证集 (~25GB)
│   └── nuplan-v1.1_test/       # 测试集 (~25GB)
│
└── bench2drive/     # Bench2Drive 数据集
    └── bench2drive-mini/       # Mini 版本
```

---

## NuPlan 数据集

### 来源
- **官网**: https://nuplan.org/
- **下载页面**: https://nuplan.org/dataset
- **HuggingFace**: https://huggingface.co/datasets/nuplan/nuplan-v1.1-mini

### 版本与大小

| 版本 | 大小 | 说明 |
|------|------|------|
| nuplan-v1.1-mini | ~1GB | ✅ 已下载 |
| nuplan-v1.1-train | ~170GB | 训练数据 |
| nuplan-v1.1-val | ~25GB | 验证数据 |
| nuplan-v1.1-test | ~25GB | 测试数据 |

### 数据结构

```
nuplan-v1.1/
├── train/                    # 训练集
│   └── nuplan-v1.1-train/
│       ├──ego_pose/         # 自车姿态
│       ├──lidarseg/         # LiDAR 语义分割
│       ├──map/              # 地图数据
│       ├──metadata/         # 元数据
│       └── ...
│
├── val/                     # 验证集
│   └── nuplan-v1.1-val/
│       └── (同train结构)
│
├── test/                    # 测试集
│   └── nuplan-v1.1-test/
│       └── (同train结构)
│
└── mini/                    # Mini 版本 (已下载)
    └── nuplan-v1.1_mini/
        └── (子集结构)
```

### 下载命令

```bash
# 方法1: 从官网下载 (需要注册)
# https://nuplan.org/dataset

# 方法2: HuggingFace CLI
huggingface-cli download nuplan/nuplan-v1.1-mini --local-dir ./data/nuplan/nuplan-v1.1_mini

# 方法3: wget/curl (需要登录获取链接)
```

---

## Bench2Drive 数据集

### 来源
- **GitHub**: https://github.com/Thinklab-SJTU/Bench2Drive
- **HuggingFace**: https://huggingface.co/datasets/Thinklab/Bench2Drive

### 版本与大小

| 版本 | 大小 | 说明 |
|------|------|------|
| Bench2Drive-Mini | ~20GB | 10 个场景 |
| Bench2Drive-Full | ~500GB+ | 1000 个场景 |

### 数据结构

```
bench2drive/
├── bench2drive-mini/
│   ├── train/               # 训练数据
│   │   └── episodes/        # 场景 episodes
│   │       ├── episode_0/
│   │       ├── episode_1/
│   │       └── ...
│   │
│   ├── val/                 # 验证数据
│   │   └── episodes/
│   │
│   └── test/                # 测试数据 (100 scenarios)
│       └── episodes/
│
└── scenarios/               # 场景定义
    └── route_*.xml
```

### 场景类型

Bench2Drive 包含多种驾驶场景:
- 车辆变道 (Lane Change)
- 车辆汇入 (Cut In)
- 行人过马路 (Pedestrian Crossing)
- 路口转弯 (Intersection Turning)
- 等等...

### 下载命令

```bash
# HuggingFace
huggingface-cli download Thinklab/Bench2Drive-Mini --local-dir ./data/bench2drive/bench2drive-mini

# 或从 GitHub release 下载
```

---

## CARLA 仿真器

### 来源
- **GitHub**: https://github.com/carla-simulator/carla
- **下载**: https://github.com/carla-simulator/carla/releases

### 版本
- **CARLA 0.9.15** - 当前使用版本

### 位置
```
CARLA/
├── CarlaUE4/            # 仿真器主程序
├── PythonAPI/           # Python 接口
├── Engine/              # Unreal 引擎
├── HDMaps/              # 高精地图
├── Plugins/             # 插件
└── Towns/               # 城市场景
    ├── Town01 - Town10
    └── Town10HD_Opt    # 高清地图
```

---

## 下载状态

- [x] nuplan-v1.1_mini.zip (~1GB)
- [ ] nuplan-v1.1_val (下载中)
- [ ] nuplan-v1.1_test (待下载)
- [ ] nuplan-v1.1_train (暂不需要)
- [x] Bench2Drive-Mini (已下载)
- [x] CARLA 0.9.15 (已安装)

---

## 自动化脚本

项目提供了两个自动化下载脚本，位于 `scripts/` 目录：

### 1. 下载 CARLA

```bash
./scripts/download_carla.sh
```

**功能：**
- 下载 CARLA 0.9.15 压缩包 (~20GB)
- 自动解压到 `data/CARLA/`
- 自动配置路径

**手动启动 CARLA：**
```bash
cd data/CARLA
./CarlaUE4.sh -prefernvidia -nosound -carla-port=2000
```

### 2. 下载 Bench2Drive

```bash
./scripts/download_bench2drive.sh
```

**功能：**
- 使用 HuggingFace CLI 下载 Bench2Drive-Mini (~20GB)
- 自动移动到 `data/bench2drive/Bench2Drive-Mini/`

**前提条件：**
```bash
pip install huggingface_hub
```

### 注意事项

- 所有数据文件都在 `data/` 目录下，不纳入 Git 版本控制
- CARLA 仿真器在 `data/CARLA/` 目录下，不纳入 Git 版本控制
- 下载脚本会自动创建必要的目录结构
