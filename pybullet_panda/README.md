# Panda Pick-and-Place: Goal-Conditioned BC on a Standard Benchmark

在标准具身智能 benchmark **panda-gym `PandaPickAndPlace-v3`**（PyBullet 物理引擎 + Franka Panda 7-DoF 双臂指夹爪）上，完整复现一条
**脚本演示采集 → 行为克隆（BC）训练 → 闭环部署评估 → 结果视频** 的模仿学习链路。

这是作品集里偏「抓取 / 标准 benchmark」方向的一块：此前项目多为推方块（push），本项目补齐**抓取-放置（pick-and-place）**这类带接触、长程决策的经典任务，并跑在社区公认的注册环境上，而非自建玩具场景。

---

## 1. 任务与环境

- 环境：`PandaPickAndPlace-v3`（panda-gym 3.0.7，PyBullet 3.2.7）
- 机器人：Franka Panda，7 臂关节 + 2 指夹爪，末端笛卡尔增量控制
- 任务：把桌面绿色方块抓起，移动到半透明幽灵目标位置；方块中心距目标 **< 0.05 m** 判定成功
- 观测向量（策略输入，22 维）：

  | 部分 | 维度 | 内容 |
  |---|---|---|
  | robot obs | 7 | EE 位置(3) + EE 速度(3) + 夹爪开度(1) |
  | object obs | 12 | 物体位置(3) + 姿态(3) + 线速度(3) + 角速度(3) |
  | desired goal | 3 | 目标物体位置 |

- 动作（4 维，`[-1, 1]`）：`[dx, dy, dz, gripper]`
  - `dx/dy/dz`：末端位移增量，环境内部按 `0.05 m/步` 缩放
  - `gripper`：`>0` 张开，`<0` 闭合（环境内部按 `0.2` 缩放开度）

> 选型说明：本机存在两个互不一致的 Python 环境——Python 3.13 有 MuJoCo 4.0.0 与 torch(cpu)，但无 pybullet/panda-gym；Python 3.11 有 torch、pybullet、gymnasium、panda-gym。主流抓取 benchmark（robosuite / LIBERO / ManiSkill）在 Python 3.13 上依赖锁死、无法安装，因此标准环境侧选择轻量、可直接 `gym.make` 注册的 **panda-gym（PyBullet）**；MuJoCo 侧的推方块实验见姊妹项目。两个仿真器的对照见文末「思考」。

## 2. 方法

### 2.1 脚本演示（oracle）

`collect_data.py` 中的 `ScriptedPickPlace` 是一个五阶段状态机，产出与后续策略同构的 4 维动作：

1. **悬停方块上方**（`z = 0.12 m`）
2. **下降贴合**方块
3. **闭合夹爪**并保持 25 步完成抓取
4. **抬起**移动到目标上方
5. **下降**到目标位置后**张开**夹爪释放

位移动作由「目标点 − 当前 EE 位置」归一化得到，再裁剪到 `[-1, 1]`。

### 2.2 行为克隆（BC）

`train_bc.py`：

- 网络：MLP `22 → 128 → 128 → 4`，ReLU
- 损失：动作 MSE 回归，AdamW（lr `1e-3`，weight decay `1e-5`）
- 数据：**只保留成功 episode** 的 transitions；观测按训练集 mean/std 标准化
- 部署：观测标准化 → 网络前向 → 动作裁剪到 `[-1, 1]`，闭环 rollout

## 3. 复现步骤

> 解释器：Python 3.11（装有 pybullet / panda-gym / torch）。

```bash
pip install -r requirements.txt

# 1) 采集 100 集脚本演示（成功轨迹入训练集）
python collect_data.py --episodes 100 --out data/pickplace_demos.npz

# 2) 训练 goal-conditioned BC
python train_bc.py --epochs 200 --batch-size 64 --out checkpoints/bc_pickplace.pt

# 3) 闭环评估 20 集并录制第一条 rollout 视频
python evaluate.py --episodes 20 --max-steps 150 \
    --ckpt checkpoints/bc_pickplace.pt --out-video videos/eval_pickplace.mp4
```

## 4. 结果（本机实测）

| 阶段 | 指标 | 数值 |
|---|---|---|
| 演示采集 | 集数 / 总步数 | 100 / 3757 |
| 演示采集 | oracle 成功率 | 0.92（92/100） |
| 训练 | 成功样本数 / 最终 loss | 3357 / 0.0072 |
| 部署评估 | BC 成功率（20 集，150 步上限） | **0.35（7/20）** |

评估视频：`videos/eval_pickplace.mp4`（首条 rollout，画面叠加方块-目标距离）。

## 5. 代码结构

```
pickplace-panda-gym/
├── env_wrapper.py      # 标准环境封装：22 维观测向量、内部状态访问器
├── collect_data.py     # 五阶段 oracle + 演示采集（npz）
├── train_bc.py         # BCDataset（成功轨迹过滤 + 标准化）+ BCPolicy MLP
├── evaluate.py         # 闭环评估、成功率统计、mp4 录制
├── data/               # 演示数据（gitignore）
├── checkpoints/        # BC 权重 + 观测统计量（gitignore）
└── videos/             # 评估视频（gitignore）
```

## 6. 自己的思考与局限

**为什么单步 BC 在抓取任务上只有 35%：**

- **分布漂移（compounding error）**：BC 只模仿单步动作，一旦某步偏离演示分布，后续没有纠错机制，误差沿轨迹累积；抓取是强接触、长程任务，放大了这一问题。
- **动作是增量而非目标位姿**：策略学的是每步 `0.05 m` 的 EE 位移，MSE 回归会把多模动作平均化，夹爪闭合时机被「抹平」。
- **纯状态输入、无历史**：单帧观测不含速度之外的时序信息，接触状态判断偏弱。
- **演示规模有限**：100 集、3k+ 样本，覆盖的物体/目标随机布局有限。

**已观察到的工程细节（踩坑记录）：**

- `gym.make` 外层包了 `TimeLimit`（默认 50 步），访问内部 `robot/task` 必须走 `env.unwrapped`；默认 50 步对偏慢的策略太紧，本项目在 wrapper 里重新放宽到 150 步。
- PyTorch 2.6 的 `torch.load` 默认 `weights_only=True`，checkpoint 里存了 numpy 统计量会反序列化失败，需显式 `weights_only=False`。
- 本机双 Python 环境包不互通，必须固定 Python 3.11 解释器运行 pybullet/panda-gym 相关脚本。

**可继续迭代的方向（作品集增强点）：**

1. **动作 chunk / 时序网络**：预测一段动作序列（chunking），或用 1D-CNN / Transformer 编码观测历史，缓解单步回归平均化。
2. **Diffusion Policy**：在同一数据上把 MLP 换成扩散动作头，对比 BC 与 DP 在接触任务上的成功率（姊妹 push 项目已有 DP 复现基础）。
3. **HER（Hindsight Experience Replay）**：goal-conditioned + 稀疏奖励，用 RL 或 HER-BC 提升长程成功率，对齐 panda-gym 官方 SAC/TD3+HER 基线。
4. **视觉输入**：从 rgb_array 训练 CNN/ResNet 编码，做 state→vision 的完整升级。
5. **多物体 / 多目标泛化**：扩展到 Stack、Flip 等 panda-gym 同族任务。

## 7. 环境

- Windows 11，Python 3.11
- PyBullet 3.2.7，gymnasium 1.2.1，panda-gym 3.0.7
- PyTorch 2.6（CUDA 可用时走 GPU，否则 CPU），OpenCV 视频写入
