"""
MuJoCo pick-and-place: a Cartesian (x/y/z slide) gripper picks a cube and
places it at a ghost target.

This is the MuJoCo counterpart of the panda-gym PyBullet benchmark. Instead of a
7-DoF arm + IK we use a 3-axis prismatic arm with a two-finger parallel jaw, so
a scripted PD controller produces clean grasping demos on CPU.

Observation (state, 13-dim):
    ee_xyz(3) + cube_xyz(3) + cube_yaw_cos/sin(2) + goal_xyz(3)
    + fingers_gap(1) + t_norm(1)

Action (4-dim, in [-1, 1]):
    [dx, dy, dz, gripper]
    dx/dy/dz : EE displacement (~2 cm/step)
    gripper  : -1 close, +1 open

Success: cube centre within 0.03 m of the target.
"""
from __future__ import annotations

import numpy as np
import mujoco

XML = """
<mujoco model="pickplace">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <default>
    <geom condim="4" friction="1.0 0.1 0.01"/>
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="table" type="plane" size="0.6 0.6 0.02" pos="0 0 -0.001" rgba="0.3 0.3 0.35 1"/>
    <geom name="goal" type="cylinder" size="0.045 0.002" pos="0.25 0.25 0.002" rgba="0.2 0.9 0.3 0.5"/>

    <body name="ee" pos="0 0 0">
      <joint name="ee_x" type="slide" axis="1 0 0" range="-0.4 0.4"/>
      <joint name="ee_y" type="slide" axis="0 1 0" range="-0.4 0.4"/>
      <joint name="ee_z" type="slide" axis="0 0 1" range="0.04 0.30"/>
      <geom name="palm" type="box" size="0.02 0.02 0.01" rgba="0.2 0.2 0.9 1"/>
      <body name="lf" pos="0 -0.04 -0.06">
        <joint name="lf_y" type="slide" axis="0 1 0" range="-0.07 0.04"/>
        <geom name="lf_g" type="box" size="0.008 0.006 0.06" rgba="0.9 0.3 0.3 1"/>
      </body>
      <body name="rf" pos="0 0.04 -0.06">
        <joint name="rf_y" type="slide" axis="0 1 0" range="-0.04 0.07"/>
        <geom name="rf_g" type="box" size="0.008 0.006 0.06" rgba="0.9 0.3 0.3 1"/>
      </body>
    </body>

    <body name="cube" pos="-0.15 -0.15 0.026">
      <freejoint name="cube_joint"/>
      <geom name="cube_g" type="box" size="0.026 0.026 0.026" mass="0.05" rgba="0.1 0.9 0.1 1"/>
    </body>

    <camera name="side" pos="0.7 -0.7 0.55" xyaxes="0.707 0.707 0 -0.4 0.4 0.825" fovy="45"/>
  </worldbody>
  <actuator>
    <position name="act_x" joint="ee_x" kp="200" damping="10"/>
    <position name="act_y" joint="ee_y" kp="200" damping="10"/>
    <position name="act_z" joint="ee_z" kp="200" damping="10"/>
    <position name="act_lf" joint="lf_y" kp="80" damping="3"/>
    <position name="act_rf" joint="rf_y" kp="80" damping="3"/>
  </actuator>
</mujoco>
"""

CUBE_HALF = 0.026
GRASP_Z = 0.095      # EE height when fingers wrap the cube centre
HOVER_Z = 0.20
NEAR = 0.02
GAP_CLOSED = 0.038
GAP_OPEN = 0.12
STEP_XYZ = 0.02


class PickPlaceEnvMJ:
    def __init__(self, max_steps: int = 200, seed: int | None = None):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._renderer = None

        self.ee_x = self.model.joint("ee_x").qposadr[0]
        self.ee_y = self.model.joint("ee_y").qposadr[0]
        self.ee_z = self.model.joint("ee_z").qposadr[0]
        self.lf_y = self.model.joint("lf_y").qposadr[0]
        self.rf_y = self.model.joint("rf_y").qposadr[0]
        self.cube_q = self.model.joint("cube_joint").qposadr[0]
        self.goal = np.array([0.25, 0.25, CUBE_HALF], dtype=np.float64)
        self.obs_dim = 13
        self.action_dim = 4

    # ------------------------------------------------------------------ helpers
    def ee_pos(self) -> np.ndarray:
        return self.data.qpos[[self.ee_x, self.ee_y, self.ee_z]].copy()

    def cube_pos(self) -> np.ndarray:
        return self.data.qpos[self.cube_q: self.cube_q + 3].copy()

    def _cube_yaw(self) -> tuple[float, float]:
        q = self.data.qpos[self.cube_q + 3: self.cube_q + 7]
        w, x, y, z = q
        yaw = 2 * np.arctan2(z, w)
        return float(np.cos(yaw)), float(np.sin(yaw))

    def _set_gripper(self, gap: float):
        # lf body base y=-0.04, rf base y=+0.04; drive to ±gap/2
        self.data.ctrl[3] = -gap / 2 + 0.04
        self.data.ctrl[4] = gap / 2 - 0.04

    def render(self, size: int = 256) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=size, width=size)
        self._renderer.update_scene(self.data, camera="side")
        return self._renderer.render()

    # ------------------------------------------------------------------ gym API
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        # arm start
        self.data.qpos[self.ee_x] = -0.25
        self.data.qpos[self.ee_y] = -0.25
        self.data.qpos[self.ee_z] = HOVER_Z
        # random cube start
        cx = self._rng.uniform(-0.25, -0.05)
        cy = self._rng.uniform(-0.25, 0.25)
        self.data.qpos[self.cube_q] = cx
        self.data.qpos[self.cube_q + 1] = cy
        self.data.qpos[self.cube_q + 2] = CUBE_HALF
        # random goal (on the table, away from cube)
        gx = self._rng.uniform(0.1, 0.32)
        gy = self._rng.uniform(-0.25, 0.25)
        self.goal = np.array([gx, gy, CUBE_HALF])
        self.model.geom("goal").pos[0] = gx
        self.model.geom("goal").pos[1] = gy
        self._set_gripper(GAP_OPEN)
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        ee = self.ee_pos()
        c = self.cube_pos()
        cyc, cys = self._cube_yaw()
        gap = 0.08 + float(self.data.qpos[self.rf_y] - self.data.qpos[self.lf_y])
        t = self._step / self.max_steps
        return np.concatenate([ee, c, [cyc, cys], self.goal, [gap, t]]).astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        cur = self.ee_pos()
        target = cur + action[:3] * STEP_XYZ
        target[2] = np.clip(target[2], 0.04, 0.30)
        self.data.ctrl[0] = target[0]
        self.data.ctrl[1] = target[1]
        self.data.ctrl[2] = target[2]
        gap = GAP_CLOSED + (action[3] + 1) / 2 * (GAP_OPEN - GAP_CLOSED)
        self._set_gripper(gap)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self._step += 1
        d = float(np.linalg.norm(self.cube_pos() - self.goal))
        success = d < 0.03
        truncated = self._step >= self.max_steps
        info = {"success": success, "dist": d, "cube": self.cube_pos()}
        return self._obs(), -d, success, truncated, info

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
