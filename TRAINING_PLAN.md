# Diffusion Planner 全量训练计划

## 1. 硬件配置

| 组件 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4080 SUPER (16GB) |
| CPU | 16 核心 |
| 内存 | 60GB |
| 磁盘 | 918GB (可用 682GB) |

---

## 2. 数据集规划

### 2.1 数据源
- **nuPlan Pittsburgh train** (~29GB, 1,560 logs)
- **nuPlan maps** (~10GB)

### 2.2 采样策略

| 参数 | 论文参考值 | 本计划建议值 |
|------|------------|--------------|
| 日志数量 | 13,180 | 1,560 (Pittsburgh) |
| 场景类型 | 70+ | 70+ |
| **总场景数** | ~7,000+ | **~3,500** |

> 目标：数据量约为论文的 50%，但保持场景类型多样性

### 2.3 数据预处理流程

#### 阶段1: 数据分析
```bash
python data_process/analyze_dataset.py \
    --data_path /data/nuplan-v1.1/train_pittsburgh \
    --visualize \
    --notion \
    --output pittsburgh_analysis.json
```
输出：
- 各场景类型数量、总时长、总帧数
- 5种可视化图表
- Notion格式表格

#### 阶段2: 场景采样
```bash
# 方式1: 均匀采样
python data_process/preprocess_pittsburgh.py \
    --data_path /data/pittsburgh \
    --map_path /data/maps \
    --save_path /data/output \
    --scenarios_per_type 50

# 方式2: 自定义分布
python data_process/preprocess_pittsburgh.py \
    --data_path /data/pittsburgh \
    --map_path /data/maps \
    --save_path /data/output \
    --distribution_file scenarios_distribution.json \
    --total_scenarios 3500
```

#### 阶段3: 抽帧 (可选)
```bash
# 每2帧抽1帧
--frame_sample_rate 2
```

### 2.4 预处理输出
- NPZ 文件数：~3,500 个
- 配置文件：`sampling_config.json`, `training_data_list.json`
- 预计数据大小：~50-100GB

---

## 3. 训练配置

### 3.1 模型架构参数 (来自 config.py)
```python
hidden_dim = 192
num_heads = 6
encoder_depth = 3
decoder_depth = 3
encoder_drop_path_rate = 0.1
decoder_drop_path_rate = 0.1

agent_num = 33      # 1 ego + 32 neighbors
static_objects_num = 5
lane_num = 70
route_num = 25

time_len = 21       # 历史轨迹长度
future_len = 80     # 未来轨迹长度
```

### 3.2 训练超参数
```python
batch_size = 2048        # 单卡 batch size
train_epochs = 500
learning_rate = 5e-4
warm_up_epoch = 5
save_utd = 20            # 每20 epoch保存

# 预计训练时间
# batch_size=2048, ~3500 samples
# 1 epoch ≈ 2-3 steps (取决于数据加载)
# 500 epochs ≈ 1000-1500 steps
```

### 3.3 资源需求
- GPU 显存：~8-12GB (16GB 足够)
- 训练时间预估：**4-8 小时** (单卡 RTX 4080 SUPER)

---

## 4. 执行步骤

### 4.1 数据准备
```bash
# 1. 确保数据已下载并解压
# /workspace/data/nuplan-v1.1/train/    (~170GB)
# /workspace/data/nuplan-v1.1/val/      (~25GB)
# /workspace/data/nuplan-v1.1/maps/      (~10GB)

# 2. 预处理训练数据
python data_process.py \
    --data_path /workspace/data/nuplan-v1.1/trainval \
    --map_path /workspace/data/nuplan-v1.1/maps \
    --save_path /workspace/data/train_output \
    --total_scenarios 3500 \
    --scenarios_per_type 50
```

### 4.2 训练
```bash
# 使用 train_simple.py (单卡)
python training/train_simple.py \
    --train_set /workspace/data/train_output \
    --train_set_list /workspace/data/train_output/training_data_list.json \
    --batch_size 2048 \
    --train_epochs 500 \
    --save_utd 20

# 或使用 train.py (Lightning，支持多卡)
python training/train.py \
    --train_set /workspace/data/train_output \
    --train_set_list /workspace/data/train_output/training_data_list.json \
    --batch_size 2048 \
    --train_epochs 500
```

### 4.3 验证
```bash
# 在验证集上评估
python run_diffusion_simulation.py \
    --checkpoint /workspace/training_outputs/best_model.ckpt \
    --scenarios_file /workspace/data/scenarios_val.csv \
    --num 100
```

---

## 5. 时间线预估

| 阶段 | 预计时间 |
|------|----------|
| 数据预处理 (~3500 scenarios) | 2-4 小时 |
| 模型训练 (500 epochs) | 4-8 小时 |
| 验证评估 | 1-2 小时 |
| **总计** | **7-14 小时** |

---

## 6. 备选方案

### 如果资源不足
- 减少 `scenarios_per_type` 到 20-30
- 减少 `train_epochs` 到 200-300
- 预计训练时间：2-4 小时

### 如果过拟合
- 增加 `scenarios_per_type` 到 100
- 增加数据增强 (augmentation)
- 使用预训练模型 fine-tune

---

## 7. 检查清单

- [ ] nuPlan train 数据 (~170GB) 已下载
- [ ] nuPlan val 数据 (~25GB) 已下载
- [ ] nuPlan maps 数据 (~10GB) 已下载
- [ ] 磁盘空间充足 (>200GB)
- [ ] GPU 显存 >= 12GB

---

*生成时间: 2026-02-28*
*基于 Diffusion Planner 官方代码和论文分析*
