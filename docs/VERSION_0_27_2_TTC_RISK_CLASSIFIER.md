# RadarMind v0.27.2：Closing-Speed / TTC 风险分类器

## 1. 问题

v0.27.1 中自车已经可以正常驾驶，但页面可能在车辆安全停车后持续显示：

```text
emergency_brake
```

对应 trace：

```text
nearest range: 7.85–8.03 m
nearest relative velocity: 0.0 m/s
ego speed: 0.0 km/h
target speed: 0.0 km/h
```

旧规则只检查距离：

```text
range <= 8 m -> emergency_brake
```

这会把“自车已经停稳，前方有静止目标”错误描述为持续急刹车。8 m 附近的测距噪声还会造成：

```text
brake <-> emergency_brake
```

逐帧抖动。

旧实时进程退出后，浏览器曾保留最后一帧 action，仅把连接标识改为 `RECONNECTING`。v0.27.2 同时把断线 action 显式改为 `TELEMETRY STALE`，避免把缓存状态误认为实时决策。

## 2. 修复原则

急刹车不应只由距离决定，还需要目标正在快速接近，并且碰撞时间足够短。

定义 closing speed：

```text
closing_speed = max(0, -relative_radial_velocity)
```

定义 TTC：

```text
TTC = range / closing_speed
```

当 closing speed 接近 0 时，TTC 记为无穷/`None`。

## 3. 新判定

默认紧急条件：

```text
range <= 8 m
closing_speed >= 0.75 m/s
TTC <= 2.0 s
```

三个条件必须同时成立：

```text
emergency =
    close_range
    AND meaningful_closing_speed
    AND short_ttc
```

如果距离很近但目标基本静止：

```text
action = brake
reason = close target, no emergency closing condition
```

这样区分：

```text
保持停车/低速跟车 -> brake
快速逼近碰撞风险 -> emergency_brake
```

## 4. 最近 detection 可解释字段

trace 增加：

```json
{
  "nearest_detection": {
    "depth_m": 6.756,
    "azimuth_deg": 1.07,
    "altitude_deg": -1.28,
    "lateral_m": 0.126,
    "estimated_height_m": 0.849,
    "relative_velocity_mps": -0.772
  },
  "closing_speed_mps": 0.772,
  "ttc_sec": 8.751
}
```

这些字段可以帮助判断 detection 来源：

- `estimated_height` 很低：可能是地面；
- `lateral` 超出 ego corridor：可能是路边设施；
- `estimated_height≈0.5–2 m` 且 lateral 接近 0：更可能是前车；
- closing speed 和 TTC：决定是否需要紧急动作。

## 5. 单元验证

固定距离 `7.9 m`：

```text
relative velocity =  0 m/s -> brake, TTC=None
relative velocity = -1 m/s -> brake, TTC=7.9 s
relative velocity = -5 m/s -> emergency_brake, TTC=1.58 s
```

这符合风险含义：近距离本身要求谨慎，但只有快速逼近才是急刹车场景。

## 6. CARLA 实机验证

命令：

```bash
python3 -m radarmind.agent.carla_dynamic_demo \
  --output-dir $RADARMIND_ROOT/runs/carla_ttc_risk_v0_27_2_smoke \
  --host 127.0.0.1 \
  --port 2000 \
  --max-steps 500 \
  --npc-count 12 \
  --no-spawn-obstacle \
  --no-auto-reset \
  --ego-controller traffic_manager \
  --cruise-speed-kph 25 \
  --brake-speed-kph 4 \
  --tm-port 8050
```

结果：

```text
records: 500
actions:
  keep_speed: 99
  slow_down: 158
  brake: 243
  emergency_brake: 0
ego distance: 58.700 m
max ego speed: 25.528 km/h
```

末帧目标：

```text
range: 6.756 m
closing speed: 0.772 m/s
TTC: 8.751 s
height: 0.849 m
lateral: 0.126 m
```

虽然距离小于 8 m，但 TTC 远高于 2 s，因此正确输出：

```text
brake
```

而不是：

```text
emergency_brake
```

## 7. 实时启动

```bash
python3 -m radarmind.agent.carla_dynamic_demo \
  --output-dir $RADARMIND_ROOT/runs/carla_ttc_risk_v0_27_2_live \
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
  --brake-speed-kph 4 \
  --tm-port 8050
```

本地浏览器：

```text
http://127.0.0.1:17860
```

## 8. 下一步

当前 TTC 基于单帧 nearest detection。更完整的风险层应增加：

- detection temporal association；
- range-rate 平滑；
- action hysteresis；
- consecutive-frame confirmation；
- ego braking distance；
- lateral collision corridor；
- actor-level ground truth 对照评估。

这些字段也可以直接构成后续 Agentic RL 的安全 reward。
