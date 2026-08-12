# RadarMind v0.29：回归原生 RadarDetection BEV

## 1. 版本决策

v0.27/v0.28 曾把 CARLA `RadarDetection` 点栅格化为 synthetic RA/RD occupancy tensor。这对接口联调有用，但并不具备真实 FMCW 信号形成过程，而且容易让展示者误以为 CARLA 产生了与 CARRADA 同源的雷达张量。

v0.29 按项目方向做明确收敛：

```text
CARLA RadarDetection
  -> 原生极坐标点
  -> detection BEV JPEG
  -> RadarMind + TTC safety shield
```

从本版本开始，CARLA 闭环不再：

- 生成 Range-Azimuth 数组；
- 生成 Range-Doppler 数组；
- 保存 `radar_tensors/*.npz`；
- 在网页或模型 prompt 中称其为 RA/RD。

历史 v0.27/v0.28 run 保留用于版本对比，不做破坏性删除。

## 2. 当前感知链

```text
sensor.other.radar @ 20 Hz
  -> RadarMeasurement
  -> list[depth, azimuth, altitude, velocity]
  -> native radar BEV
     x = depth * sin(azimuth)
     y = depth * cos(azimuth)
  -> colored detections
     red=closing, yellow=static, green=receding
```

同一批原始点还进入 20 Hz corridor/TTC safety shield；BEV 图与 RGB 相机拼接后进入低频 Qwen2.5-VL worker。

## 3. 代码变化

删除：

```text
detections_to_tensors
tensor_to_rgb
render_tensor_dashboard
TensorLiveState
NumPy RA/RD buffers
range_bins / azimuth_bins / doppler_bins CLI
max_abs_velocity CLI
radar_tensors output directory
```

恢复使用：

```python
live_state = LiveState(...)
radar.listen(lambda measurement: live_state.update_radar(measurement, max_range=50.0))
```

模型复合图改为：

```text
left:  RGB FOR CONTEXT
right: CARLA RADAR DETECTION BEV
```

prompt 显式声明右图不是 RA/RD，也不是 raw ADC。

## 4. 雷达安装 pitch 修正

CARLA detection 的 `altitude` 是传感器局部俯仰角。旧高度估计只使用：

```text
height = radar_z + depth * sin(local_altitude)
```

但当前雷达自身安装为 `pitch=+2°`，因此世界俯仰应为：

```text
world_elevation = local_altitude + radar_pitch
height = radar_z + depth * sin(world_elevation)
```

v0.29 同时修正 horizontal depth、forward/lateral 和 ground filter。回归测试覆盖：局部 `-2°`、安装 `+2°` 的射线实际为水平射线，20 m 处高度仍为 1 m，不应被误删为地面点。

## 5. 输出结构

```text
$RADARMIND_ROOT/runs/carla_radar_bev_agent_v0_29_verified
├── dynamic_trace.jsonl
├── dynamic_demo.report.json
├── verification_summary.json
└── snapshots
    ├── camera_000050.jpg
    └── radar_bev_000050.jpg
```

trace 中的雷达声明：

```json
{
  "radar_representation": {
    "source": "native CARLA RadarDetection polar points rendered as BEV",
    "fields": ["depth", "azimuth", "altitude", "velocity"],
    "not_generated": ["range_azimuth", "range_doppler", "raw_adc"]
  }
}
```

报告显式写入：

```json
{
  "generated_ra_rd": false
}
```

## 6. 实时运行

```bash
conda activate radargym-rl
cd /path/to/radarmind-carla

python3 -m radarmind.agent.carla_dynamic_demo \
  --output-dir $RADARMIND_ROOT/runs/carla_radar_bev_agent_v0_29_live \
  --host 127.0.0.1 \
  --port 2000 \
  --web-host 127.0.0.1 \
  --web-port 7860 \
  --max-steps 0 \
  --policy-mode hybrid \
  --model-adapter-name best_overall \
  --model-device cuda:0 \
  --npc-count 12 \
  --cyclist-count 5 \
  --pedestrian-count 20 \
  --ego-controller traffic_manager \
  --no-spawn-obstacle \
  --no-auto-reset
```

Mac：

```bash
ssh -N -L 17860:127.0.0.1:7860 USER@SERVER_IP
```

浏览器访问 `http://127.0.0.1:17860`。

## 7. 验证结果

Safety smoke：

```text
records: 200
ego distance: 42.903 m
vehicles/cyclists/pedestrians: 6/3/8
snapshots: 8
RA/RD or NPZ artifacts: 0
```

Hybrid verified run：

```text
records: 400
ego distance: 66.710 m
model fresh frames: 210
source radarmind: 84
source radarmind+safety_agree: 87
source safety_shield: 39
source safety_fallback: 190
final keep/slow/brake: 99/262/39
NPZ files: 0
RA/RD-named files: 0
radar BEV JPEG: 8
```

5 个动作融合/雷达高度单元测试全部通过。

## 8. 为什么这更适合作为后续训练输入

训练和部署必须保持输入一致。v0.28 的在线模型使用 CARLA synthetic RA/RD，但 adapter 训练来自 CARRADA pseudo-image，两者既不物理同源，视觉布局也不同。v0.29 固定部署输入为：

```text
RGB camera + CARLA RadarDetection BEV + ego/radar telemetry
```

下一版可以直接从 CARLA 轨迹保存同样的复合图，并用 privileged simulator state 生成 teacher label。训练阶段 teacher 可以读取 actor 类别/位置，推理阶段只给 VLM RGB+BEV，从而学习：

- 相机负责目标语义和道路上下文；
- 雷达 BEV 负责距离、方位和相对速度；
- telemetry 负责精确 TTC；
- 输出统一的结构化最优纵向 action。

## 9. 局限

- detection BEV 仍是 CARLA ray-cast 结果，不是真实毫米波数据；
- 当前 `best_overall` adapter 没有用 RGB+BEV 数据训练，v0.29 只是输入链迁移；
- 普通 CARLA radar detection 不含语义类别；
- 需要 privileged teacher 和动作均衡数据，才能真正训练相机—雷达融合策略；
- 横向规划仍由 Traffic Manager 负责。

这些限制由 v0.30 的融合 SFT 数据和训练 pipeline 继续解决。
