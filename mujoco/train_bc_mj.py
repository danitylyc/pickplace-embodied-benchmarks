"""Goal-conditioned BC on MuJoCo pick-and-place demos (state 11 -> action 4)."""
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
        ep_ok = list(d["ep_success"]) if "ep_success" in d else [True] * len(ep_ends)
        keep = np.zeros(len(states), dtype=bool)
        prev = 0
        for end, ok in zip(ep_ends, ep_ok):
            if ok:
                keep[prev:end] = True
            prev = end
        self.states = states[keep]
        self.actions = actions[keep]
        self.mean = self.states.mean(0)
        self.std = self.states.std(0) + 1e-6

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        s = (self.states[i] - self.mean) / self.std
        return torch.from_numpy(s), torch.from_numpy(self.actions[i])


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pickplace_mj_demos.npz")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/bc_pickplace_mj.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = BCDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    print(f"[bc] device={device} samples={len(ds)}")
    model = BCPolicy(ds.states.shape[1], ds.actions.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    for ep in range(args.epochs):
        model.train()
        tot = n = 0.0
        for s, a in dl:
            s, a = s.to(device), a.to(device)
            pred = model(s)
            loss = loss_fn(pred, a)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * s.size(0); n += s.size(0)
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"[bc] epoch {ep:3d} loss={tot/n:.4f}")
    torch.save({"model": model.state_dict(), "mean": ds.mean, "std": ds.std,
                "obs_dim": ds.states.shape[1], "action_dim": ds.actions.shape[1]}, args.out)
    print(f"[bc] saved -> {args.out}")


if __name__ == "__main__":
    main()
