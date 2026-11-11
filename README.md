# Pick-and-Place across Two Simulators

同一抓取-放置任务（pick a cube, place it at a target）在两个主流仿真器上的完整模仿学习实现，
作为具身智能作品集的对照样本：**脚本演示采集 → 行为克隆训练 → 闭环部署评估 → 视频** 全链路可复现。

| 子目录 | 仿真器 | 机械臂 | 环境 | Oracle 成功率 | BC 成功率 |
|---|---|---|---|---|---|
| [`pybullet_panda/`](pybullet_panda) | PyBullet 3.2.7 | 真实 Franka Panda（7-DoF + IK） | panda-gym `PandaPickAndPlace-v3`（标准 benchmark） | 92% | **35%** |
| [`mujoco/`](mujoco) | MuJoCo 4.0 | 3-axis Cartesian slide + 双指夹爪 | 自建 MJCF | 100% | **95%** |

## 为什么做两个版本

- `pybullet_panda/` 跑在社区公认的标准注册环境上（panda-gym），机械臂是真实 Panda URDF + pybullet IK，
  结果更"标准"，但接触阶段单步 BC 较难（35%）。
- `mujoco/` 用自建 Cartesian 夹爪场景，控制直观、演示干净，单步 BC 成功率高（95%），
  但机械臂是简化建模。
- 对照本身说明一个工程经验：**环境简化能显著抬高 BC 数字，但泛化回 7-DoF + 视觉才是真实方向**。

## 快速开始

两个项目分别在不同 Python 环境下运行（依赖不互通）：

```bash
# PyBullet / panda-gym 版（Python 3.11）
cd pybullet_panda
pip install -r requirements.txt
python collect_data.py --episodes 100 --out data/pickplace_demos.npz
python train_bc.py --epochs 200
python evaluate.py --episodes 20

# MuJoCo 版（Python 3.13）
cd ../mujoco
pip install -r requirements.txt
python collect_mj.py --episodes 100 --out data/pickplace_mj_demos.npz
python train_bc_mj.py --epochs 200
python eval_mj.py --episodes 20
```

每个子目录的 `README.md` 内含环境细节、观测/动作维度、踩坑记录与扩展方向。

## 方法概览

- **演示**：五阶段脚本 oracle（悬停上方 → 下降 → 闭合抓取 → 抬起到目标上方 → 下降释放）。
- **策略**：goal-conditioned MLP，观测标准化，MSE 回归 4 维动作（xyz 位移 + 夹爪开合）。
- **评估**：20 集闭环 rollout，统计方块距目标 < 阈值的成功率，OpenCV 写 mp4。

## 局限与后续

- 单步 BC 存在分布漂移，长程接触任务成功率受限；可升级动作 chunk、观测历史堆叠、Diffusion Policy、HER。
- MuJoCo 版使用了 `t_norm`（任务进度）作为输入，部署时强按步数推进，泛化到不同时长会退化。
- 纯状态输入；可接渲染图做视觉策略。
