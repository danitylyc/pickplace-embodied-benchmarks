"""Evaluate MuJoCo BC pick-and-place policy and record a rollout video."""
from __future__ import annotations

import argparse
import numpy as np
import torch
import cv2
from env_mj import PickPlaceEnvMJ
from train_bc_mj import BCPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bc_pickplace_mj.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out-video", default="videos/eval_pickplace_mj.mp4")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = BCPolicy(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    mean, std = ckpt["mean"], ckpt["std"]

    env = PickPlaceEnvMJ()
    n_ok = 0
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=8000 + ep)
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            s = (obs - mean) / std
            with torch.no_grad():
                t = torch.from_numpy(s).unsqueeze(0).to(device)
                act = model(t)[0].cpu().numpy()
            act = np.clip(act, -1.0, 1.0)
            obs, r, term, trunc, info = env.step(act)
            steps += 1
            done = term or trunc
            if ep == 0:
                frame = cv2.cvtColor(env.render(256), cv2.COLOR_RGB2BGR)
                cv2.putText(frame, f"d={info['dist']:.2f}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                frames.append(frame)
        n_ok += int(info["success"])
        if ep % 5 == 0:
            print(f"[eval] ep {ep:2d} ok={info['success']} d={info['dist']:.3f} steps={steps}")

    sr = n_ok / args.episodes
    print(f"[eval] success rate = {sr:.2f} over {args.episodes} episodes")
    if frames:
        vw = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), 12, (256, 256))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"[eval] video -> {args.out_video} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
