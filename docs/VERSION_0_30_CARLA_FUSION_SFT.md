# RadarMind v0.30：CARLA 相机—雷达 BEV 融合 SFT

## 1. 本版目标

v0.29 已把部署输入固定为 CARLA RGB 与原生 `RadarDetection` BEV，并停止生成缺乏 FMCW 物理意义的伪 RA/RD。v0.30 第一次让训练输入与在线部署输入保持一致：

```text
CARLA RGB + native RadarDetection BEV + telemetry
  -> Qwen2.5-VL-3B LoRA
  -> structured perception + longitudinal action
  -> asynchronous hybrid policy + 20 Hz safety shield
```

本版不是只有数据脚本：实际采集了 1,400 帧 CARLA 轨迹，生成 280 个对齐的多模态样本，完成三轮真实 LoRA 训练、离线基线对比和 500 帧 CARLA hybrid 闭环。

## 2. 为什么使用 CARLA privileged teacher

普通 CARLA radar detection 只有：

```text
depth / azimuth / altitude / velocity
```

它没有目标类别。相机能提供语义，但自动标注每张图代价高。模拟器可以读取 actor 的类别、位置和速度，因此训练时使用 CARLA ground truth 作为教师：

```text
actor world state
  -> ego coordinate projection
  -> class / distance / lateral offset / closing speed / TTC / in_path
  -> teacher objects + scene_summary + recommended_action
```

关键约束：`privileged_teacher` 只写进离线 trace 和 assistant label。在线 `RadarMindPolicyWorker.submit()` 只收到 camera JPEG、radar BEV JPEG、ego speed 和 radar summary，绝不收到 actor ground truth。每条数据均写入：

```json
{"privileged_input_online": false}
```

## 3. 特权教师的几何与动作标签

`carla_privileged_teacher.py` 把世界坐标差旋转到自车坐标：

```text
forward = cos(yaw) * dx + sin(yaw) * dy
lateral = -sin(yaw) * dx + cos(yaw) * dy
```

只保留车前、量程内、相机视场内的 actor。相对径向速度和 TTC 为：

```text
relative_radial_velocity = (v_actor - v_ego) dot unit_line_of_sight
closing_speed = max(0, -relative_radial_velocity)
TTC = distance / closing_speed
```

教师先采用雷达 safety action，再用 `pedestrian/cyclist` 的 `in_path`、距离和 TTC 做易受伤交通参与者语义增强。教师标签不会直接控制在线车辆。

## 4. 真实数据采集

环境：

```bash
conda activate radargym-rl
cd /path/to/radarmind-carla
export PYTHONPATH=$PWD/PKC
```

自然混合交通：

```bash
python -m radarmind.agent.carla_dynamic_demo \
  --output-dir $RADARMIND_ROOT/runs/carla_fusion_v0_30_natural \
  --policy-mode safety --max-steps 800 --save-every 5 \
  --npc-count 10 --cyclist-count 5 --pedestrian-count 20 \
  --no-realtime --web-port 7860
```

前方障碍增强：

```bash
python -m radarmind.agent.carla_dynamic_demo \
  --output-dir $RADARMIND_ROOT/runs/carla_fusion_v0_30_obstacle \
  --policy-mode safety --max-steps 600 --save-every 5 \
  --npc-count 6 --cyclist-count 3 --pedestrian-count 12 \
  --spawn-obstacle --obstacle-distance 24 --auto-reset \
  --reset-stopped-steps 20 --no-realtime --web-port 7860
```

结果：

| run | frames | vehicles/cyclists/pedestrians | snapshot pairs | ego distance |
| --- | ---: | --- | ---: | ---: |
| natural | 800 | 10/5/20 | 160 | 100.473 m |
| obstacle | 600 | 6/3/12 | 120 | 75.845 m |

每个 snapshot 只有 `camera_*.jpg` 与 `radar_bev_*.jpg`，没有 RA/RD/NPZ。

## 5. JSONL 构建与格式

构建命令：

```bash
python -m radarmind.datasets.build_carla_fusion_sft \
  --run-dir $RADARMIND_ROOT/runs/carla_fusion_v0_30_natural \
  --run-dir $RADARMIND_ROOT/runs/carla_fusion_v0_30_obstacle \
  --output-dir $RADARMIND_ROOT/datasets/carla_fusion_v0_30_action_first \
  --val-ratio 0.2
```

构建器完成：

1. 用相同 step 对齐 RGB、radar BEV、trace；
2. 调用在线同款 `compose_multimodal_frame()`，左 RGB、右 radar BEV；
3. 生成 Qwen chat messages；
4. 按 action 分层划分 train/val；
5. 生成每类等量的 `train_balanced.jsonl`；
6. 将 `recommended_action` 放在 assistant JSON 第一字段，增强自回归动作 token 的监督。

数据统计：

```json
{
  "records": 280,
  "train_records": 225,
  "balanced_train_records": 472,
  "val_records": 55,
  "balanced_train_actions": {
    "keep_speed": 118,
    "monitor": 118,
    "slow_down": 118,
    "brake": 118
  }
}
```

单条核心结构：

```json
{
  "sample_id": "carla_fusion_v0_30_natural_000720",
  "radar": {
    "radar_image_path": "$RADARMIND_ROOT/datasets/...jpg",
    "source": "carla_rgb_native_radar_bev"
  },
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>\n...telemetry..."},
    {"role": "assistant", "content": "{...}"}
  ],
  "radar_scene": {
    "recommended_action": {"type": "slow_down", "reason": "..."},
    "scene_summary": "...",
    "objects": []
  }
}
```

Qwen processor dry-run：`input_ids=[1,962]`，supervised tokens 219，`image_grid_thw=[1,3]`。

## 6. 三轮真实 LoRA 训练

基础权重：

```text
$RADARMIND_ROOT/models/Qwen2.5-VL-3B-Instruct
```

### 6.1 原始分布，64 steps

```text
train records: 128
optimizer updates: 64
trainable LoRA params: 18,576,384
mean loss: 1.850(first step) -> 0.979(cumulative)
elapsed: 67.25 s
```

它把 10 条验证的 parse rate 从 0.80 提升到 1.00、object F1 从 0.143 提升到 0.467、action accuracy 从 0.10 提升到 0.50，但 10/10 都输出 `keep_speed`，出现多数类塌缩。

### 6.2 动作过采样，96 steps

四类过采样后均为 118 条，96 次更新，累计 mean loss 降到 0.771。但模型转而 10/10 输出 `slow_down`，说明只平衡样本数不足以解决自回归目标中的 action 信号弱化。

### 6.3 动作优先 JSON，多任务 128 steps

最终训练命令：

```bash
python -m radarmind.training.qwen_vl_lora_sft \
  --model-path $RADARMIND_ROOT/models/Qwen2.5-VL-3B-Instruct \
  --train-jsonl $RADARMIND_ROOT/datasets/carla_fusion_v0_30_action_first/train_balanced.jsonl \
  --val-jsonl $RADARMIND_ROOT/datasets/carla_fusion_v0_30_action_first/val.jsonl \
  --output-dir $RADARMIND_ROOT/models/radarmind-qwen2_5-vl-3b-lora-carla-fusion-action-first-v0_30 \
  --max-train-samples 256 --batch-size 1 --epochs 2 --max-steps 128 \
  --learning-rate 5e-5 --lora-rank 8 --lora-alpha 16 \
  --device cuda:1 --dtype bf16
```

结果：

```text
optimizer updates: 128
trainable LoRA params: 18,576,384
cumulative mean loss: 2.027 -> 0.706
elapsed: 130.52 s
```

GPU3 当时只有约 11 GiB 空闲，首次尝试在第一个 forward OOM；没有杀用户进程，改用 CARLA 占用后仍有约 41 GiB 空闲的 GPU1 完成训练。复现前应先运行 `nvidia-smi`。

## 7. 公平离线对比

从 val 中固定抽取 12 条，每类 3 条：

```text
keep_speed / monitor / slow_down / brake = 3 / 3 / 3 / 3
```

| model | parse rate | object micro-F1 | action accuracy | prediction distribution |
| --- | ---: | ---: | ---: | --- |
| v0.15 best_overall | 1.000 | 0.235 | 0.250 | monitor=12 |
| v0.30 action-first | 1.000 | 0.714 | 0.333 | keep=10, monitor=2 |

v0.30 在部署域感知上明显提升，action 只从 0.25 提升到 0.333，且仍缺少 brake/slow_down 多样性。因此注册为 `carla_fusion_action_first_v0_30` candidate，不覆盖现有跨数据集 `best_overall`。这是诚实的模型选择结果。

## 8. 500 帧 CARLA hybrid 回归

```bash
python -m radarmind.agent.carla_dynamic_demo \
  --output-dir $RADARMIND_ROOT/runs/carla_fusion_v0_30_closed_loop \
  --policy-mode hybrid \
  --model-adapter-name '' \
  --model-adapter-path $RADARMIND_ROOT/models/radarmind-qwen2_5-vl-3b-lora-carla-fusion-action-first-v0_30 \
  --model-device cuda:0 --model-max-new-tokens 192 --model-interval-sec 2 \
  --max-steps 500 --save-every 50 \
  --npc-count 8 --cyclist-count 4 --pedestrian-count 16 --realtime
```

结果：

```json
{
  "records": 500,
  "ego_distance_m": 74.779,
  "max_moving_participants": 28,
  "model_fresh_frames": 298,
  "model_parse_errors": 0,
  "last_model_latency_ms": 1804.8,
  "final_actions": {"keep_speed": 113, "slow_down": 333, "brake": 54},
  "decision_sources": {
    "safety_fallback": 202,
    "radarmind": 169,
    "radarmind+safety_agree": 75,
    "safety_shield": 54
  },
  "generated_ra_rd": false
}
```

模型加载和推理不会阻塞 20 Hz world tick；加载期间 safety fallback，模型新结果到达后 hybrid fusion，54 帧由 safety shield 选择 brake。横向控制仍由 Traffic Manager 负责。

## 9. 测试与资产

9 个教师、几何、数据均衡、雷达高度、模型/安全融合单测全部通过。

```text
$RADARMIND_ROOT/datasets/carla_fusion_v0_30_action_first
$RADARMIND_ROOT/models/radarmind-qwen2_5-vl-3b-lora-carla-fusion-action-first-v0_30
$RADARMIND_ROOT/runs/carla_fusion_v0_30_eval_action_first_balanced12
$RADARMIND_ROOT/runs/carla_fusion_v0_30_closed_loop
```

## 10. 如何真正继续提高最优驾驶决策

“最优”不能只靠更长 SFT。v0.31 应分五层推进：

1. 场景课程：主动生成 cut-in、横穿行人、骑行者侵入、自车高速逼近、静止近物等场景，补齐 `emergency_brake`，按 scenario 而不是单帧随机切分；
2. 时序输入：3–5 个连续 RGB/BEV 帧或 radar track tokens，让模型学习速度趋势而不是从单帧猜运动；
3. 双头目标：感知 JSON 与离散 action head 分开，给 action 使用 class-balanced/focal loss，避免长对象 JSON 淹没动作 token；
4. grounded policy：VLM 识别类别与路径关系，雷达提供距离/closing/TTC，结构化决策工具做约束融合，安全盾保留为最后防线；
5. trajectory post-training：用碰撞、最小 TTC、路线进度、急刹 jerk、交通规则构造 episode reward，先做 teacher trajectory distillation / DPO，再做 CARLA Agentic RL。

评测必须同时报告 collision、route completion、red-light violation、min TTC、intervention rate、comfort jerk、parse rate、action confusion matrix 和跨 Town 泛化，不能只看训练 loss。

## 11. 求职技术栈对应

本版覆盖感知算法工程师与 AI Agent 工程师常见能力：CARLA 多传感器仿真、雷达坐标与 TTC、相机—雷达多模态融合、自动数据闭环、privileged teacher/distillation、Qwen2.5-VL、LoRA SFT、类别不平衡与失败分析、异步模型 serving、safety shield、闭环 trace/evaluation、model registry 和可复现工程文档。
