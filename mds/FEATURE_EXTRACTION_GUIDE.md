# 特征提取指南

## 概述

本指南说明如何从 nuPlan 场景中提取训练特征。

---

## 环境要求

**重要：所有脚本执行必须在 Docker 容器内进行！**

- 特征提取 → nuplan-simulation 容器
- 训练 → training 容器

### 进入容器
```bash
docker exec -it nuplan-simulation bash
```

### 路径映射
- 容器内 `/workspace/` → 宿主机 `/home/zhouyida/.openclaw/workspace/diffusion-planner-project/`

---

## 数据频率

| 数据类型 | 频率 | 说明 |
|----------|------|------|
| ego_pose | 100Hz | 自车姿态数据 |
| lidar_pc | 20Hz | LiDAR 点云帧 |
| lidar_box | 20Hz | 障碍物检测框 |

---

## 特征列表

| 特征 | Shape | 说明 |
|------|-------|------|
| ego_current_state | (10,) | 自车当前状态 |
| ego_agent_future | (80, 3) | 自车未来轨迹 (8秒 @ 10Hz) |
| neighbor_agents_past | (32, 21, 11) | 邻居历史轨迹 |
| neighbor_agents_future | (32, 81, 3) | 邻居未来轨迹 |
| lanes | (70, 20, 12) | 车道特征 |
| route_lanes | (25, 20, 12) | 路线车道特征 |

---

## 提取命令

### 1. 单场景提取
```bash
python3 /workspace/scripts/extract_single_frame/extract_single_frame.py
```

脚本会自动处理以下数据库：
- 2021.06.28.16.29.11_veh-38_01415_01821.db
- 2021.10.06.07.26.10_veh-52_00006_00398.db
- ...

### 2. 可视化
```bash
python3 /workspace/scripts/visualize_npz.py <npz_path> <png_path>

# 示例
python3 /workspace/scripts/visualize_npz.py \
    /workspace/data_process/npz_scenes/scene_123.npz \
    /workspace/validation_output/scene_123.png
```

---

## 采样策略

### ego_future (未来轨迹)
- 80 点，10Hz，8秒
- 从 ego_pose (100Hz) 每 10 帧取 1 帧

### ego history (历史轨迹)
- 21 点，10Hz，2秒
- **单一真源**：`neighbor_agents_past[0]`（ego slot0）
- 从 ego_pose (100Hz) 每 10 帧取 1 帧

### neighbor_future (未来轨迹)
- 81 点，10Hz，8秒
- 从 lidar_box (20Hz) 每 2 帧取 1 帧

### neighbor_past (历史轨迹)
- 21 点，10Hz，2秒
- 从 lidar_box (20Hz) 每 2 帧取 1 帧

---

## 末尾填充

当 agent 数据不足时，分三种情况处理：

### 情况1：完全没有有效 future 点
用当前帧位置填充：
```python
# 从 center_box 获取当前位置
fill_dx, fill_dy = transform_to_ego_frame(center_box['x'], center_box['y'], ego_x, ego_y, ego_heading)
fill_heading = center_box['yaw'] - ego_heading

# 所有未来帧都用当前位置填充
for i in range(NEIGHBOR_FUTURE_LEN):
    neighbor_future[agent_idx, i] = [fill_dx, fill_dy, fill_heading]
```

### 情况2：只有 1 个有效点
用该点常值填充（不估计速度）：
```python
for i in range(1, NEIGHBOR_FUTURE_LEN):
    neighbor_future[agent_idx, i] = [last_valid_dx, last_valid_dy, last_valid_heading]
```

### 情况3：>=2 个有效点
用匀速模型填充：
```python
# 用最后两个有效点计算速度
velocity_x = last_valid_dx - prev_dx
velocity_y = last_valid_dy - prev_dy

# 按匀速模型填充
for i in range(last_valid_idx + 1, target_len):
    neighbor_future[agent_idx, i] = [
        last_valid_dx + velocity_x * (i - last_valid_idx),
        last_valid_dy + velocity_y * (i - last_valid_idx),
        last_valid_heading
    ]
```

---

## 地图选择逻辑

地图根据数据库的 `log.location` 字段自动选择：

| location | 地图名称 |
|----------|----------|
| las_vegas | us-nv-las-vegas-strip |
| boston | us-ma-boston |
| pittsburgh | us-pa-pittsburgh-hazelwood |
| singapore | sg-one-north |

如果 location 本身就是地图名称（如 us-pa-pittsburgh-hazelwood），则直接使用。

---

## 常用数据库和帧索引

### Las Vegas
| 数据库 | 帧索引 |
|--------|--------|
| 2021.06.28.16.29.11_veh-38_01415_01821.db | 29424 |
| 2021.05.12.22.28.35_veh-35_00620_01164.db | 53853 |

### Boston
| 数据库 | 帧索引 |
|--------|--------|
| 2021.08.09.17.55.59_veh-28_00021_00307.db | 17626 |

### Singapore
| 数据库 | 帧索引 |
|--------|--------|
| 2021.10.06.07.26.10_veh-52_00006_00398.db | 18701 |

---

## 输出文件

### NPZ 格式
```python
{
    'ego_current_state': (10,),
    'ego_agent_future': (80, 3),
    'neighbor_agents_past': (32, 21, 11),
    'neighbor_agents_future': (32, 81, 3),
    'lanes': (70, 20, 12),
    'route_lanes': (25, 20, 12),
    ...
}
```

### 可视化
- 蓝/红色虚线：车道边界
- 浅黄色实线：route lane 中心线
- 金色箭头：route lane 方向
- 绿色虚线：ego 历史轨迹（来自 `neighbor_agents_past[0]`）
- 蓝色实线：ego 未来轨迹
- 绿色实线：邻居未来轨迹

---

## BFS Bridge（route_lanes 断裂修复）

当 `route_lanes_avails` 全 0 且 **intersection_pruned==0** 时，说明「自车附近可用 roadblock 集合」与「pruned_route roadblock 集合」没有交集，直接按 pruned_route 抽 lane 会导致 route_lanes 全空。此时会尝试用 **BFS bridge** 在路网里补一段连接片段。

### 触发条件
- `intersection_pruned == 0`
- 且 `avails_sum_old == 0`（即旧的 route_lanes_avails 求和为 0）

### ego_rb 获取方式
- 在 `ego_pose` 的当前位置附近（固定半径搜索）收集候选 map objects：`Lane` 与 `LaneConnector`
- 计算每个候选 roadblock 的几何中心到 ego 的距离，取距离最小者作为 `ego_rb`

### BFS 到 pruned_route 前 K 个目标
- 从 `ego_rb` 作为起点，在 roadblock graph 上做 BFS
- 目标集合为 `pruned_route_roadblock_ids[:K]`（当前实现 K=10）
- 若找到任一目标 roadblock，则回溯得到一条 bridge path（包含起点与目标）

### 拼接 new_route
- 设 bridge path 为 `bridge = [rb0(=ego_rb), rb1, ..., rbt(∈ pruned_route[:K])]`
- 令 `new_route = bridge[1:] + pruned_route`（去掉 bridge 的第一个 ego_rb，避免重复）
- 后续 route_lanes 的抽取/采样基于 `new_route`

### 日志字段（/workspace/validation_output/bfs_single_case_result.json）
脚本会把关键指标落盘，便于批量排查：
- `intersection_pruned`
- `ego_rb`
- `bridge_found` / `bridge_reason`
- `bridge_len`
- `avails_sum_old` / `avails_sum_new`

> 备注：如果 BFS 未找到目标（`bridge_found=False`），则保持原 route，不会引入额外 lane。
