# RadarMind v0.27.1：自车正常行驶与 CARLA 雷达语义澄清

## 1. 问题

v0.27 初次 live demo 中，摄像机所在自车经常停车，看起来不像正常驾驶。排查发现有两个独立问题。

第一，自车雷达安装高度为 `1 m`，初始垂直 FOV 为 `-5°–5°`。下沿光束与地面的理论交点为：

```text
distance = 1 / sin(5°) ≈ 11.47 m
```

trace 中虚假最近距离长期为 `11–13 m`，与该几何关系一致。安全 Agent 把地面检测当作障碍物，持续制动。

第二，初版横向 controller 每帧把当前位置投影到最近 waypoint。在 Town10HD 路口，最近 waypoint 可能跳到交叉车道：

```text
end ego yaw: -30.76°
nearest driving-lane yaw: 77.81°
```

车辆已经不再沿原路线方向驾驶。

## 2. 地面回波修复

增加 detection 高度估计：

```text
estimated_height =
    radar_mount_height + depth × sin(altitude)
```

低于 `0.35 m` 的 detection 不参与安全距离决策：

```text
estimated_height < 0.35 m -> ground filtered
```

同时把雷达几何从：

```text
pitch = 0°
vertical FOV = 10°
```

调整为：

```text
pitch = 2°
vertical FOV = 4°
```

即主要覆盖 `0°–4°`，从几何上减少地面照射。

## 3. 标准道路驾驶修复

默认自车 controller 改为 CARLA Traffic Manager：

```bash
--ego-controller traffic_manager
--tm-port 8050
--cruise-speed-kph 25
```

Traffic Manager 负责：

- lane following；
- junction route；
- traffic light；
- leading-vehicle distance；
- lane change；
- longitudinal/lateral vehicle control。

RadarMind Agent 不再直接和手写横向 controller 争夺控制，而是根据雷达风险设置 Traffic Manager 目标速度：

```text
keep_speed       -> 25 km/h
monitor          -> 18 km/h
slow_down        -> 10 km/h
brake            -> 4 km/h
emergency_brake  -> 0 km/h
```

执行接口：

```python
traffic_manager.set_desired_speed(vehicle, target_speed_kph)
```

这仍然是闭环 Agent action，只是 tool 从低层 `VehicleControl` 变为更稳定的高层目标速度。实际的 throttle/brake/steer 仍从 CARLA actor 读取并写入 trace。

## 4. 无干扰长距离验证

命令：

```bash
python3 -m radarmind.agent.carla_dynamic_demo \
  --output-dir $RADARMIND_ROOT/runs/carla_ego_tm_motion_v0_27_3 \
  --host 127.0.0.1 \
  --port 2000 \
  --max-steps 600 \
  --npc-count 0 \
  --no-spawn-obstacle \
  --no-auto-reset \
  --ego-controller traffic_manager \
  --cruise-speed-kph 25 \
  --tm-port 8050
```

结果：

```json
{
  "records": 600,
  "episodes": 1,
  "max_ego_speed_kph": 25.528,
  "total_ego_distance_m": 76.146,
  "obstacle_spawned": false,
  "ego_controller": "traffic_manager",
  "cruise_speed_kph": 25.0,
  "interrupted": false
}
```

起终点：

```text
start: (-64.645,  24.471), yaw=0.159°
end:   (-45.168, -40.064), yaw=-89.595°
```

自车成功通过初版 controller 出错的路口。

后半段出现速度为 0、目标速度仍为 25 km/h 的情况，是 Traffic Manager 根据交通灯/道路规则主动制动，不是 RadarMind 虚假障碍制动。继续运行后车辆会恢复。

## 5. 带动态交通的 live 验证

正式 live：

```bash
python3 -m radarmind.agent.carla_dynamic_demo \
  --output-dir $RADARMIND_ROOT/runs/carla_normal_driving_v0_27_1_live \
  --host 127.0.0.1 \
  --port 2000 \
  --web-host 127.0.0.1 \
  --web-port 7860 \
  --max-steps 0 \
  --npc-count 12 \
  --no-spawn-obstacle \
  --no-auto-reset \
  --ego-controller traffic_manager \
  --cruise-speed-kph 25 \
  --tm-port 8050
```

实际 live 状态：

```text
step: 2081
ego speed: 24.281 km/h
target speed: 25.0 km/h
total ego distance: 356.237 m
NPC vehicles: 12
moving NPC vehicles: 7
```

这证明摄像机挂载的自车不是仅轮速变化，而是 transform 持续变化并沿 CARLA route 行驶。

## 6. CARLA 内置雷达到底输出什么

本机 CARLA 0.9.16 blueprint：

```text
sensor.other.radar
```

callback 获得：

```text
carla.RadarMeasurement
```

它可迭代为 `RadarDetection`，每个 detection 有：

```text
depth      距离，m
azimuth    水平角，rad
altitude   垂直角，rad
velocity   沿检测方向的相对速度，m/s
```

CARLA 内置 radar 不输出：

- raw ADC / IF samples；
- complex phase；
- antenna channel；
- carrier frequency；
- chirp slope；
- bandwidth；
- range FFT；
- Doppler FFT；
- RCS/amplitude；
- CFAR 前能量图。

因此它更接近 detection-level radar simulator，而不是 waveform-level FMCW radar simulator。

## 7. CARLA 雷达“频率”

需要区分两种频率。

输出帧率：

```text
sensor_tick = 0
```

默认表示每个 simulation frame 输出一次。实际 Hz 由 world FPS 决定。

本项目设置：

```text
fixed_delta_seconds = 0.05
sensor_tick = 0.05
```

因此：

```text
radar update rate = 1 / 0.05 = 20 Hz
```

点预算：

```text
points_per_second = 3000
```

在 20 Hz 下理论预算约为：

```text
3000 / 20 = 150 points/frame
```

实际命中数通常低于该值。

射频载频：

CARLA 内置 radar 没有 carrier-frequency 参数，不能说它原生工作在 77 GHz 或 79 GHz。它没有模拟真实 FMCW 波形。

本机 blueprint 默认值：

```text
points_per_second = 1500
range = 100 m
horizontal_fov = 30°
vertical_fov = 30°
sensor_tick = 0
```

## 8. “栅格化”是什么意思

是的，本项目的栅格化本质上是把稀疏 detection 投入离散 bin：

```text
(depth, azimuth) -> Range-Azimuth grid
(depth, velocity) -> Range-Doppler grid
```

它与点云投影到 BEV 类似，但坐标系不同：

```text
BEV: x/y Cartesian grid
RA:  range/azimuth polar grid
RD:  range/radial-velocity grid
```

当前使用能量累加：

```text
energy = 1 / sqrt(1 + depth)
```

这不是从 FFT 得到的真实幅值，只是 detection occupancy/intensity proxy。

## 9. 能否还原为 CARRADA 形式

不能从 CARLA detections 唯一还原真实 CARRADA tensor。原因是 detection 已经丢失原始波形、相位、天线通道、噪声、clutter 和 FFT 旁瓣；这是一个不可逆的信息损失。

可以做三种不同等级的近似。

### 9.1 Detection-level CARRADA-shaped tensor

把 CARLA detections 栅格化为与 CARRADA 相同 shape、坐标范围和归一化：

```text
RA proxy
RD proxy
AD proxy
```

再加入 Gaussian splatting、clutter、噪声、旁瓣和动态范围映射。这适合接口迁移，但不是物理还原。

### 9.2 CARRADA-style learned generator

用 CARRADA tensor 学习条件生成模型：

```text
CARLA detections + actor geometry
  -> conditional generator
  -> CARRADA-like RA/RD/AD
```

可以做 distribution alignment，但没有真实 paired ground truth 时，生成结果仍然只是 statistically CARRADA-like。

### 9.3 Waveform-level FMCW simulator

更可靠的方案是自建 CARLA radar signal layer：

```text
actor/raycast geometry
range + radial velocity + angle + material/RCS proxy
  -> configurable FMCW waveform
  -> complex ADC cube
  -> range FFT / Doppler FFT / beamforming
  -> RD / RA / AD tensor
```

显式设置：

```text
carrier frequency
bandwidth
chirp duration
chirps per frame
TX/RX antenna geometry
sampling rate
noise/clutter/multipath model
```

这是后续将 CARLA 闭环与 CARRADA 真实 tensor 模型结合的推荐路线。
