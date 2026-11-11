"""
Goal-conditioned Behavior Cloning on the panda-gym pick-and-place demos.

Policy: MLP 22 -> 128 -> 128 -> 4, predicts the 4-dim action from
(observation 19 + desired goal 3). Trained with MSE on successful demos only.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class BCDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        states = d["states"].astype(np.float32)
        actions = d["actions"].astype(np.float32)
        ep_ends = list(d["ep_ends"])
        ep_success = list(d["ep_success"]) if "ep_success" in d else [True] * len(ep_ends)

        # keep only successful episodes
        keep_idx = np.zeros(len(states), dtype=bool)
        prev = 0
        for end, ok in zip(ep_ends, ep_success):
            if ok:
                keep_idx[prev:end] = True
            prev = end
        self.states = states[keep_idx]
        self.actions = actions[keep_idx]

        # observation normalization (fit on kept data)
        self.mean = self.states.mean(0)
        self.std = self.states.std(0) + 1e-6

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        s = (self.states[idx] - self.mean) / self.std
        return torch.from_numpy(s), torch.from_numpy(self.actions[idx])


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pickplace_demos.npz")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/bc_pickplace.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = BCDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    print(f"[bc] device={device}  successful samples={len(ds)}")

    model = BCPolicy(ds.states.shape[1], ds.actions.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for s, a in dl:
            s, a = s.to(device), a.to(device)
            pred = model(s)
            loss = loss_fn(pred, a)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * s.size(0)
            n += s.size(0)
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"[bc] epoch {ep:3d} loss={total/n:.4f}")

    torch.save({
        "model": model.state_dict(),
        "mean": ds.mean, "std": ds.std,
        "obs_dim": ds.states.shape[1], "action_dim": ds.actions.shape[1],
    }, args.out)
    print(f"[bc] saved -> {args.out}")


if __name__ == "__main__":
    main()
