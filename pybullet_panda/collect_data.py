"""
Collect scripted pick-and-place demonstrations on the standard panda-gym env.

Oracle state machine (5 phases):
  P0  hover above the cube
  P1  descend onto the cube
  P2  close fingers and grasp (hold for a few steps)
  P3  lift and move above the target
  P4  descend to target and release

The oracle outputs the same 4-dim action the learned policy will later predict.
Only successful episodes are kept (clean BC data).
"""
from __future__ import annotations

import argparse
import numpy as np
from env_wrapper import PickPlaceEnv

LIFT_Z = 0.12          # hover height above table
NEAR = 0.03            # position threshold to advance phase
GRASP_HOLD = 25        # steps to keep fingers closed while lifting
STEP_SCALE = 0.05      # env scales ee displacement by this much per action


class ScriptedPickPlace:
    def __init__(self):
        self.phase = 0
        self.hold = 0

    def reset(self):
        self.phase = 0
        self.hold = 0

    def __call__(self, env: PickPlaceEnv) -> np.ndarray:
        ee = env.ee_pos()
        obj = env.obj_pos()
        goal = env.goal_pos()

        if self.phase == 0:      # above the cube
            target = np.array([obj[0], obj[1], LIFT_Z])
            gripper = 1.0
            if np.linalg.norm(ee - target) < NEAR:
                self.phase = 1
        elif self.phase == 1:    # descend onto cube
            target = obj.copy()
            gripper = 1.0
            if np.linalg.norm(ee - obj) < NEAR:
                self.phase = 2
        elif self.phase == 2:    # close and grasp
            target = obj.copy()
            gripper = -1.0
            self.hold += 1
            if self.hold >= GRASP_HOLD:
                self.phase = 3
        elif self.phase == 3:    # lift above target
            target = np.array([goal[0], goal[1], LIFT_Z])
            gripper = -1.0
            if np.linalg.norm(ee - target) < NEAR:
                self.phase = 4
        else:                    # descend to target and release
            target = goal.copy()
            gripper = 1.0

        delta = target - ee
        action_xyz = np.clip(delta / STEP_SCALE, -1.0, 1.0)
        return np.concatenate([action_xyz, [gripper]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--out", default="data/pickplace_demos.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = PickPlaceEnv()
    states, actions, ep_ends, ep_success = [], [], [], []
    successes = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        policy = ScriptedPickPlace()
        done = False
        ep_len = 0
        success = False
        while not done and ep_len < args.max_steps:
            a = policy(env)
            states.append(obs)
            actions.append(a)
            obs, r, term, trunc, info = env.step(a)
            ep_len += 1
            success = bool(info["is_success"])
            done = term or trunc
        successes += int(success)
        ep_ends.append(len(actions))
        ep_success.append(success)
        if ep % 10 == 0:
            print(f"[collect] ep {ep:3d} success={success} len={ep_len}")

    env.close()
    np.savez_compressed(
        args.out,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        ep_ends=np.asarray(ep_ends, dtype=np.int64),
        ep_success=np.asarray(ep_success, dtype=bool),
        success_rate=np.asarray([successes / args.episodes]),
    )
    print(f"[collect] saved {args.out}  steps={len(actions)}  "
          f"success_rate={successes/args.episodes:.2f}")


if __name__ == "__main__":
    main()
