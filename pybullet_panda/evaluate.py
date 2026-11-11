"""Evaluate the trained BC pick-and-place policy and save a rollout video."""
from __future__ import annotations

import argparse
import numpy as np
import torch
import cv2
from env_wrapper import PickPlaceEnv
from train_bc import BCPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bc_pickplace.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--out-video", default="videos/eval_pickplace.mp4")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = BCPolicy(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean = ckpt["mean"]; std = ckpt["std"]

    env = PickPlaceEnv()
    successes = 0
    frames = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=5000 + ep)
        done = False
        steps = 0
        while not done and steps < args.max_steps:
            s = (obs - mean) / std
            with torch.no_grad():
                t = torch.from_numpy(s).unsqueeze(0).to(device)
                act = model(t)[0].cpu().numpy()
            act = np.clip(act, -1.0, 1.0)
            obs, r, term, trunc, info = env.step(act)
            steps += 1
            done = term or trunc
            if ep == 0:
                frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
                d = float(np.linalg.norm(env.obj_pos() - env.goal_pos()))
                cv2.putText(frame, f"d={d:.2f}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                frames.append(frame)
        successes += int(info["is_success"])
        if ep % 5 == 0:
            d = float(np.linalg.norm(env.obj_pos() - env.goal_pos()))
            print(f"[eval] ep {ep:2d} success={info['is_success']} d={d:.3f} steps={steps}")

    sr = successes / args.episodes
    print(f"[eval] success rate = {sr:.2f} over {args.episodes} episodes")

    if frames:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(args.out_video, fourcc, 10.0, (256, 256))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"[eval] video saved -> {args.out_video} ({len(frames)} frames)")

    env.close()


if __name__ == "__main__":
    main()
