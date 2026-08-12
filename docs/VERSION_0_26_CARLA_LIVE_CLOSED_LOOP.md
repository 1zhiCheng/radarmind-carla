# RadarMind v0.26：CARLA 实时雷达闭环与浏览器看板

## 1. 版本目标

v0.25 已经能把离线 RadarMind trace 转成 `carla.VehicleControl`，但它不能直接回答两个展示问题：

1. 如何在本地电脑实时看到远程服务器上的 CARLA 画面？
2. 如何让 CARLA radar sensor 的输出实时影响下一时刻的车辆动作？

v0.26 增加真正的在线传感器闭环：

```text
CARLA RGB camera ───────────────────────────────> browser dashboard
CARLA radar detections -> radar safety agent -> ActionPolicy
                                             -> carla.VehicleControl
                                             -> vehicle.apply_control()
                                             -> next CARLA world tick
```

它同时提供一个无额外前端依赖的 HTTP 看板，实时显示：

- CARLA RGB 前视相机；
- radar detection 生成的 BEV 伪图；
- 最近障碍物距离和 radar points 数量；
- 当前 Agent action；
- throttle、brake、steer；
- 车速、CARLA frame 和动作原因。

## 2. 新增入口

主程序：

```text
radarmind/agent/carla_live_demo.py
```

默认输出：

```text
$RADARMIND_ROOT/runs/carla_live_demo_v0_26/
├── live_trace.jsonl
├── live_demo.report.json
└── snapshots/
    ├── camera_*.jpg
    └── radar_*.jpg
```

## 3. 环境要求

CARLA server 已经启动，并且 Python client 能连接：

```bash
conda activate radargym-rl

python3 - <<'PY'
import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
print(client.get_world().get_map().name)
PY
```

本次验证环境：

```text
carla Python package: 0.9.16
CARLA map: Carla/Maps/Town10HD_Opt
Python: python
```

浏览器看板只依赖 Python 标准库和 Pillow，不需要 Node.js、Gradio 或额外前端服务。

## 4. 服务器端启动实时闭环

在服务器上执行：

```bash
conda activate radargym-rl
cd /path/to/radarmind-carla

python3 -m radarmind.agent.carla_live_demo \
  --output-dir $RADARMIND_ROOT/runs/carla_live_demo_v0_26 \
  --host 127.0.0.1 \
  --port 2000 \
  --web-host 127.0.0.1 \
  --web-port 7860 \
  --max-steps 0 \
  --image-width 960 \
  --image-height 540 \
  --spawn-index 0 \
  --obstacle-distance 35 \
  --save-every 100
```

`--max-steps 0` 表示持续运行，按 `Ctrl+C` 安全退出。退出后程序会停止 sensor、销毁本次生成的自车/目标车、恢复 CARLA 原来的同步设置，并写入报告。

## 5. 本地电脑实时观看

CARLA server 和网页服务运行在远程服务器，本地电脑不需要安装 CARLA。推荐通过 SSH tunnel 转发网页端口。

在本地电脑的新终端执行：

```bash
ssh -N -L 7860:127.0.0.1:7860 zhangzongyuan@<服务器IP>
```

如果 SSH 不是 22 端口：

```bash
ssh -p <SSH端口> -N -L 7860:127.0.0.1:7860 zhangzongyuan@<服务器IP>
```

然后在本地浏览器打开：

```text
http://127.0.0.1:7860
```

不要关闭 tunnel 终端。浏览器每 180 ms 请求一次实时状态和最新 JPEG，因此看到的是当前 CARLA camera/radar/control，而不是预先生成的视频。

如需让同一局域网机器直接访问，可以在确认防火墙策略后把服务改为：

```bash
--web-host 0.0.0.0
```

这会暴露服务端口，默认方案仍建议使用 SSH tunnel。

## 6. CARLA 雷达数据是什么

CARLA `sensor.other.radar` 每帧产生一组 `RadarDetection`：

```json
{
  "depth": 18.50,
  "azimuth": 0.03,
  "altitude": -0.01,
  "velocity": -0.18
}
```

字段含义：

- `depth`：检测点与 sensor 的距离，单位为米；
- `azimuth`：水平方位角，单位为弧度；
- `altitude`：垂直角，单位为弧度；
- `velocity`：沿检测射线方向的相对速度，单位为 m/s。

它不是 CARRADA 的 RD/RA/AD FFT tensor。v0.26 先把 detection point 投影为 BEV：

```text
lateral = depth × sin(azimuth)
forward = depth × cos(azimuth)
```

这样可以实时展示 CARLA 雷达；后续若要缩小仿真和真实 tensor 的域差异，需要再实现 range-angle/range-Doppler 栅格化、噪声建模和 radar encoder。

## 7. 在线 Agent 控制逻辑

v0.26 使用高频、可解释的 radar safety agent：

```text
无 radar return              -> monitor
nearest range > 28 m         -> keep_speed
16 m < nearest range <= 28 m -> slow_down
8 m < nearest range <= 16 m  -> brake
nearest range <= 8 m         -> emergency_brake
```

高层动作交给已有的 `action_policy_default.json`：

```text
slow_down -> throttle=0.15, brake=0.25, steer=0.0
brake     -> throttle=0.00, brake=0.65, steer=0.0
```

最后转成：

```python
control = carla.VehicleControl(
    throttle=...,
    brake=...,
    steer=...,
)
vehicle.apply_control(control)
```

这条链路是在线闭环，但 v0.26 的在线决策器是规则 Agent，不是 Qwen2.5-VL。这样做是为了先验证 sensor timing、数据协议、动作安全层、CARLA 控制和实时展示。下一版可把 Qwen-VL 作为低频语义决策器，同时保留高频 radar safety shield。

## 8. 已完成的实机验证

最终 smoke：

```bash
python3 -m radarmind.agent.carla_live_demo \
  --output-dir $RADARMIND_ROOT/runs/carla_live_demo_v0_26_verified \
  --host 127.0.0.1 \
  --port 2000 \
  --web-host 127.0.0.1 \
  --web-port 7860 \
  --max-steps 120 \
  --save-every 30 \
  --image-width 640 \
  --image-height 360 \
  --obstacle-distance 35
```

结果：

```json
{
  "map": "Carla/Maps/Town10HD_Opt",
  "records": 120,
  "snapshots_saved": 8,
  "obstacle_spawned": true,
  "agent": "radar_rule_safety_agent_v0_26",
  "sync": true
}
```

trace 汇总：

```text
actions: monitor=2, slow_down=3, brake=115
radar min-range interval: 11.35–18.50 m
maximum ego speed: 10.58 km/h
final ego speed: 0.0 km/h
```

HTTP 端点也在持续运行时做了实际请求验证：

```text
GET /             -> HTTP 200, 5207 bytes
GET /state.json   -> HTTP 200, live CARLA state
GET /camera.jpg   -> HTTP 200, 50551 bytes
GET /radar.jpg    -> HTTP 200, 24776 bytes
```

## 9. trace 与 report

`live_trace.jsonl` 每行代表一个闭环 step：

```json
{
  "step": 120,
  "carla_frame": 33478096,
  "action": "brake",
  "reason": "Nearest radar return is 12.3 m: braking to preserve headway.",
  "radar": {
    "num_points": 61,
    "min_depth_m": 12.348,
    "nearest_velocity_mps": 0.0
  },
  "speed_kph": 0.0,
  "control": {
    "throttle": 0.0,
    "brake": 0.65,
    "steer": 0.0
  }
}
```

它可用于：

- 回放和错误分析；
- 从成功/失败闭环中挖掘 trajectory；
- 构造 Agentic RL reward；
- 比较 rule、SFT、DPO 和蒸馏模型的安全行为；
- 生成面试展示曲线和视频。

`live_demo.report.json` 是本次 run 的摘要，记录 map、step 数、输出路径、sensor/Agent 模式和是否正常退出。

## 10. 这版对岗位展示的价值

感知算法工程师方向：

- CARLA camera/radar sensor 接入；
- radar detection 坐标投影与 BEV 可视化；
- perception-to-control 闭环；
- 同步仿真、传感器时序和安全阈值设计。

AI Agent 工程师方向：

- 结构化 observation/action contract；
- tool/action policy；
- 高频安全 controller 与低频语义 Agent 的分层架构；
- trajectory trace、在线可观测性和后续 Agentic RL 数据入口。

## 11. 下一版建议

v0.27 推荐实现分层在线 Agent：

```text
20 Hz radar safety shield ------------------------------> final control
2–5 Hz radar pseudo-image + RGB -> Qwen-VL semantic Agent
                                  -> intent / risk / action
```

Qwen-VL 不应阻塞 CARLA world tick。应增加独立推理 worker、latest-frame queue、动作有效期和超时回退；模型超时或输出解析失败时，由 radar safety shield 接管。
