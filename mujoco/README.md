# MuJoCo Pick-and-Place: Goal-Conditioned Behavior Cloning

在 **MuJoCo 4.x** 中自建的 Cartesian 双指夹爪抓取-放置环境上，完整跑通
**脚本演示采集 → 行为克隆训练 → 闭环部署评估 → 结果视频** 全链路。

这是与 `pickplace-panda-gym`（PyBullet 标准环境）对应的 **MuJoCo 版本**：
同一个抓取-放置任务，不同物理引擎与机械臂建模方式，方便对照两个仿真器的手感与结果。

---

## 1. 环境设计

为了在 CPU 上稳定产出干净的演示，机械臂不用 7-DoF 串联臂 + IK，而用
**3 轴 Prismatic（slide）Cartesian 臂 + 平行双指夹爪**，MJCF 内联在 `env_mj.py`：

- **臂**：`ee_x / ee_y / ee_z` 三个 slide 关节，position actuator（kp=200）
- **夹爪**：左/右指各一个 `slide y` 关节，对称开合
- **方块**：freejoint 立方体（边长 0.052 m），随机起始位置
- **目标**：半透明绿色 cylinder，随机位置
- **成功判据**：方块中心距目标 < 0.03 m

观测向量（策略输入，13 维）：

| 部分 | 维度 | 内容 |
|---|---|---|
| ee_xyz | 3 | 夹爪末端位置 |
| cube_xyz | 3 | 方块位置 |
| cube_yaw | 2 | 方块朝向 (cos, sin) |
| goal_xyz | 3 | 目标位置 |
| fingers_gap | 1 | 当前双指开度 |
| t_norm | 1 | 归一化任务进度（step/max_steps） |

动作（4 维，`[-1,1]`）：`[dx, dy, dz, gripper]`，xyz 每步 ~2 cm，gripper -1 闭合 / +1 张开。

> 关键工程取舍：`fingers_gap` 与 `t_norm` 两个特征是单步 BC 能否完成长程接触任务的关键。
> 没有它们时，MLP 学会了"移动→下降→闭合"，但卡在该抬升时继续输出下降（成功率 0%）；
> 加入后策略获得了"夹爪状态"和"任务进度"信号，成功率升到 95%。详见文末思考。

## 2. 复现步骤

> 解释器：Python 3.13（本环境装有 MuJoCo 4.0.0 与 torch CPU；与 panda-gym 项目的 Python 3.11 环境不互通）。

```bash
pip install mujoco torch numpy opencv-python

# 1) 采集 100 集脚本演示
python collect_mj.py --episodes 100 --out data/pickplace_mj_demos.npz

# 2) 训练 goal-conditioned BC（MLP 13 -> 128 -> 128 -> 4）
python train_bc_mj.py --epochs 200 --out checkpoints/bc_pickplace_mj.pt

# 3) 闭环评估 20 集并录视频
python eval_mj.py --episodes 20 --ckpt checkpoints/bc_pickplace_mj.pt
```

## 3. 结果（本机实测）

| 阶段 | 指标 | 数值 |
|---|---|---|
| 演示采集 | 集数 / 总步数 | 100 / 6121 |
| 演示采集 | oracle 成功率 | 1.00（100/100） |
| 训练 | 成功样本 / 最终 loss | 6121 / 0.0078 |
| 部署评估 | BC 成功率（20 集） | **0.95（19/20）** |

评估视频：`videos/eval_pickplace_mj.mp4`。

## 4. 代码结构

```
pickplace-mujoco/
├── env_mj.py        # MJCF 场景 + 环境封装（reset/step/render）
├── collect_mj.py    # 五阶段 oracle + 演示采集
├── train_bc_mj.py   # BCDataset（成功轨迹过滤+标准化）+ BCPolicy MLP
├── eval_mj.py       # 闭环评估 + mp4 录制
├── data/            # 演示数据（gitignore）
├── checkpoints/     # 权重与观测统计量（gitignore）
└── videos/          # 评估视频（gitignore）
```

## 5. 自己的思考与踩坑

**Cartesian slide 臂 vs 7-DoF 臂：**
- Cartesian 臂控制直观（xyz 直接指令），oracle 稳定，演示干净；缺点是它不是真实 Franka 的运动学。
- panda-gym 版本用真实 Panda URDF + pybullet IK，更"标准"，但 BC 在接触阶段更难（35% vs 本版 95%）。
- 两个版本对照说明：**环境简化能换得 BC 的高成功率，但泛化到真实手臂需要回到 7-DoF + 视觉**。

**MuJoCo 建模踩坑：**
1. **slide 关节 qpos 是相对 body 初始位置的位移**，不是世界坐标。最初把 `ee body pos z=0.2`，结果 qpos=0.095 时实际高度 0.295，夹爪悬浮在方块上方够不着。把 body 初始 z 设 0 后 qpos 才是绝对高度。
2. **position actuator 有稳态误差**：目标降到 0.095，受方块接触顶住实际停在 ~0.119。oracle 的阶段切换不能要求精确到目标，要按"进入合理区间"判定。
3. **单步 BC 缺时序信号**：第一版只有 11 维（无夹爪开度/进度），BC 0%——它无法判断"是否已抓住、该抬升了"。补 `fingers_gap` 和 `t_norm` 后才通。这是模仿学习里"状态可观测性"的典型教训。

**局限与扩展方向：**
- `t_norm` 是时间索引，部署时强按步数推进，泛化到不同任务时长会退化；可换成观测历史堆叠（3~5 帧）或 1D-CNN/Transformer。
- 当前纯状态输入；可接 MuJoCo `Renderer` 的 rgb 图做视觉策略（CNN/ResNet 编码）。
- 可把 MLP 换成 Diffusion Policy 动作头，在同一数据上对比 BC/DP。
- Cartesian 臂可升级为 MuJoCo Menagerie 的 Franka Panda MJCF，做 IK 抓取。

## 6. 环境

- Windows 11，Python 3.13（sandbox runtime）
- MuJoCo 4.0.0，PyTorch 2.15（CPU），OpenCV 视频写入
