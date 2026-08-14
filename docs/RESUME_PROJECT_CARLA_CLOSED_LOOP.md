# RadarMind-CARLA：相机—雷达融合自动驾驶闭环 Agent

## 简历可直接使用版本

**RadarMind-CARLA｜相机—雷达融合自动驾驶闭环 Agent｜独立研发**

技术栈：Python、PyTorch、Qwen2.5-VL-3B、LoRA/PEFT、CARLA 0.9.16、异步推理、TTC Safety Shield、MJPEG、Docker、Hugging Face Transformers

- 基于 CARLA 0.9.16 搭建相机—雷达融合闭环 Agent，将前视 RGB、原生 `RadarDetection` 点的 BEV、车速及 TTC 摘要组织成多模态观测，由 Qwen2.5-VL 输出 `keep_speed / monitor / slow_down / brake / emergency_brake` 结构化动作并映射为车辆纵向控制命令。
- 构建混合交通和特权教师数据流水线，在车辆、骑行者、行人动态场景中采集 1,400 帧轨迹并生成 280 组严格对齐的 RGB—Radar BEV 样本；仅离线使用 CARLA actor state 自动标注类别、距离、closing speed、TTC 与动作，在线策略不访问仿真真值。
- 完成 Qwen2.5-VL-3B LoRA SFT、动作均衡采样与 action-first JSON 监督，训练 18.58M 可学习参数；固定小规模验证集上结构化输出解析率达到 100%，目标检测 micro-F1 由 0.235 提升至 0.714，并保留动作类别塌缩的失败分析与后续时序策略改进方案。
- 设计 latest-only 异步 VLM worker，将约 1.8 s 模型延迟与 CARLA 20 Hz 同步 tick 解耦；融合确定性 TTC safety shield，在模型输出过期、解析失败或推理异常时自动降级，并选择 VLM 与安全策略中更保守的动作。
- 完成 Town10HD_Opt 500 tick 混合交通闭环回归：自车行驶 74.779 m，场景最多包含 28 个运动参与者，获得 298 个 fresh model frames、0 次 JSON 解析错误，54 tick 由 safety shield 接管制动；同时实现 RGB/Radar BEV/控制状态的浏览器 MJPEG 实时看板、SSH 隧道访问及逐帧 trace/report 审计。

项目地址：<https://github.com/1zhiCheng/radarmind-carla>

## 一页简历压缩版

如果版面只能容纳 3 条，使用下面版本：

- 基于 CARLA 0.9.16 与 Qwen2.5-VL-3B 搭建 RGB + 原生 RadarDetection BEV 自动驾驶闭环 Agent，在包含车辆、骑行者和行人的混合交通中输出结构化纵向驾驶动作并执行 `VehicleControl`。
- 构建 privileged-teacher 数据闭环，采集 1,400 帧轨迹、生成 280 组对齐多模态样本并完成 LoRA SFT；固定验证集结构化解析率 100%，目标 micro-F1 从 0.235 提升至 0.714。
- 通过 latest-only 异步推理和 TTC safety shield 解耦约 1.8 s VLM 延迟与 20 Hz 仿真 tick；500 tick 回归行驶 74.779 m、覆盖最多 28 个运动参与者、模型输出零解析错误，并实现实时 Web 看板与可审计 action trace。

## 面试时的 60 秒介绍

这个项目解决的是“大模型推理很慢，但自动驾驶闭环不能停下来等它”的工程问题。CARLA 以 20 Hz 运行，我把 RGB、CARLA 原生雷达检测点渲染成的 BEV、车速和 TTC 组成模型输入，让 Qwen2.5-VL 输出结构化纵向动作。模型单次推理约 1.8 秒，因此我实现 latest-only 异步 worker：仿真线程始终按固定 tick 前进，只消费最新可用的模型动作；同时保留确定性的 TTC safety shield，当模型结果过期、格式错误或不够保守时立即接管。训练侧使用 CARLA actor ground truth 作为离线 privileged teacher，自动生成物体语义与动作标签，但部署时模型看不到这些真值。最终完成 500 tick 的 VLM-in-the-loop 回归，并通过逐帧 trace 分析模型、安全盾和最终控制之间的关系。

## 项目架构

```text
CARLA RGB camera ─────────────┐
                             ├─> multimodal observation ─> latest-only Qwen2.5-VL worker ─┐
RadarDetection polar points  │                                                            │
        └─> native BEV ──────┘                                                            ├─> conservative fusion
Radar depth/velocity ─> TTC safety shield ────────────────────────────────────────────────┘
                                                                                           │
                                                                                           v
                                                        longitudinal VehicleControl + action trace
                                                                                           │
                                                                                           v
                                                         CARLA closed loop + MJPEG dashboard
```

离线训练链路：

```text
CARLA actor ground truth（teacher only）
        + synchronized RGB / Radar BEV / telemetry
        -> privileged labels
        -> action-balanced multimodal JSONL
        -> Qwen2.5-VL-3B LoRA SFT
        -> asynchronous hybrid policy
```

## 核心工程亮点

1. **训练—部署输入一致**：训练和在线推理均使用 RGB + 原生 RadarDetection BEV，没有使用在线不可得的 actor state。
2. **非阻塞闭环**：模型加载与生成不占用 CARLA synchronous tick；队列只保留最新观测，避免推理积压。
3. **安全降级**：模型缺失、超时、过期或 JSON 解析失败时继续由 safety shield 控制，不让主循环崩溃。
4. **可审计性**：逐 tick 保存原始模型文本、解析后 JSON、安全动作、最终控制来源、传感器摘要与聚合报告。
5. **远程可视化**：通过 MJPEG 和 SSH 端口转发实时查看 RGB、雷达 BEV、速度、油门、刹车和决策原因。

## 指标口径与真实性边界

- 500 tick 结果是工程回归，不是 CARLA Leaderboard 分数；尚未完成跨 Town 路线完成率、碰撞率、红灯违规和舒适性标准评测。
- VLM 当前负责纵向决策，横向路径跟随由 CARLA Traffic Manager 完成，不能表述为端到端转向控制。
- CARLA `RadarDetection` 是 ray-cast detection point，不是原始汽车 FMCW ADC，也没有真实 RD/RA 张量；项目验证的是传感器同步、检测点几何、语义融合和闭环决策。
- 离线 12 条均衡验证集规模较小，`0.714` object micro-F1 只作为部署域回归指标，不作为通用感知精度结论。
- action accuracy 由 0.250 提升至 0.333，但仍存在类别塌缩，因此该 LoRA 被注册为 candidate，并由 safety shield 保证闭环安全，而不是宣称已经得到最优驾驶策略。

## ATS 关键词

自动驾驶 Agent、多模态大模型、VLM、Qwen2.5-VL、CARLA、Camera-Radar Fusion、Radar BEV、LoRA、SFT、Privileged Teacher、Closed-loop Simulation、Asynchronous Inference、Safety Shield、TTC、Fallback、Structured Output、Agent Tool Use、Real-time Dashboard、MJPEG、Docker、Hugging Face、PyTorch
