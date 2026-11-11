"""Collect scripted pick-and-place demonstrations in MuJoCo."""
from __future__ import annotations

import argparse
import numpy as np
from env_mj import PickPlaceEnvMJ, GRASP_Z, HOVER_Z, NEAR, STEP_XYZ


class ScriptedPick:
    def __init__(self):
        self.phase = 0
        self.hold = 0

    def reset(self):
        self.phase = 0
        self.hold = 0

    def __call__(self, env: PickPlaceEnvMJ) -> np.ndarray:
        ee = env.ee_pos()
        cube = env.cube_pos()
        goal = env.goal

        if self.phase == 0:    # hover above cube
            target = np.array([cube[0], cube[1], HOVER_Z])
            grip = 1.0
            if np.linalg.norm(ee - target) < NEAR:
                self.phase = 1
        elif self.phase == 1:  # descend to cube
            target = np.array([cube[0], cube[1], GRASP_Z])
            grip = 1.0
            if ee[2] < 0.13 and np.linalg.norm(ee[:2] - cube[:2]) < 0.025:
                self.phase = 2
        elif self.phase == 2:  # close and hold
            target = np.array([cube[0], cube[1], GRASP_Z])
            grip = -1.0
            self.hold += 1
            if self.hold >= 20:
                self.phase = 3
        elif self.phase == 3:  # lift above target
            target = np.array([goal[0], goal[1], HOVER_Z])
            grip = -1.0
            if np.linalg.norm(ee - target) < NEAR + 0.01:
                self.phase = 4
        else:                  # descend and release
            target = np.array([goal[0], goal[1], GRASP_Z + 0.01])
            grip = 1.0

        delta = target - ee
        act_xyz = np.clip(delta / STEP_XYZ, -1.0, 1.0)
        return np.concatenate([act_xyz, [grip]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="data/pickplace_mj_demos.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = PickPlaceEnvMJ()
    states, actions, ep_ends, ep_ok = [], [], [], []
    n_ok = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        pol = ScriptedPick()
        done = False
        steps = 0
        ok = False
        while not done and steps < env.max_steps:
            a = pol(env)
            states.append(obs)
            actions.append(a)
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            ok = bool(info["success"])
            done = term or trunc
        n_ok += int(ok)
        ep_ends.append(len(actions))
        ep_ok.append(ok)
        if ep % 10 == 0:
            print(f"[collect] ep {ep:3d} ok={ok} steps={steps} d={info['dist']:.3f}")

    np.savez_compressed(
        args.out,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        ep_ends=np.asarray(ep_ends, dtype=np.int64),
        ep_success=np.asarray(ep_ok, dtype=bool),
        success_rate=np.asarray([n_ok / args.episodes]),
    )
    print(f"[collect] saved {args.out} steps={len(actions)} sr={n_ok/args.episodes:.2f}")


if __name__ == "__main__":
    main()
