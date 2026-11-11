"""
Wrapper around the standard panda-gym `PandaPickAndPlace-v3` environment.

Standard benchmark (PyBullet + Franka Panda): a 7-DoF arm with a two-finger
gripper must pick a green cube off the table and place it at a ghost target.

Observation vector fed to the policy (22-dim):
    observation (19)  = ee_xyz(3) + ee_vel(3) + fingers_width(1)
                         + object_pos(3) + object_rot(3) + object_vel(3)
                         + object_ang_vel(3)
    desired_goal (3)  = target object position

Action (4-dim, in [-1, 1]):
    [dx, dy, dz, gripper]
    dx/dy/dz : end-effector displacement (env scales by 0.05 m/step)
    gripper  : >0 open, <0 close (env scales finger width by 0.2)

Success: cube centre within 0.05 m of the target.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import panda_gym  # noqa: F401  (registers PandaPickAndPlace-v3)


class PickPlaceEnv:
    def __init__(self, render_size: int = 96, seed: int | None = None,
                 max_episode_steps: int = 150):
        raw = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")
        # unwrap the default 50-step TimeLimit and re-wrap with a looser horizon
        self.env = TimeLimit(raw.unwrapped, max_episode_steps=max_episode_steps)
        self.render_size = render_size
        self.obs_dim = 19 + 3          # observation + desired_goal
        self.action_dim = 4
        self._seed = seed

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed if seed is not None else self._seed)
        return self._vector(obs), info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._vector(obs), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self.env.render()

    def close(self):
        self.env.close()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _vector(obs: dict) -> np.ndarray:
        return np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)

    # Internal state accessors used by the scripted oracle.
    def _robot(self):
        return self.env.unwrapped.robot

    def _task(self):
        return self.env.unwrapped.task

    def ee_pos(self) -> np.ndarray:
        return np.asarray(self._robot().get_ee_position(), dtype=np.float32)

    def obj_pos(self) -> np.ndarray:
        return np.asarray(self._task().get_achieved_goal(), dtype=np.float32)

    def goal_pos(self) -> np.ndarray:
        return np.asarray(self._task().get_goal(), dtype=np.float32)

    def fingers_width(self) -> float:
        return float(self._robot().get_fingers_width())
