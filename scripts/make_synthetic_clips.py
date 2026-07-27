"""Generate fake landmark clips for pipeline testing.

    python scripts/make_synthetic_clips.py --out data/raw --signs 10 --reps 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FEATURE_DIM = 1662


def make_clip(gloss_id: int, rng: np.random.Generator, T: int) -> np.ndarray:
    t = np.linspace(0, 1, T)[:, None]
    seq = np.zeros((T, FEATURE_DIM), dtype=np.float32)

    pose = np.zeros((T, 33, 4), dtype=np.float32)
    pose[:, :, :3] = rng.normal(0.5, 0.02, (T, 33, 3))
    pose[:, 11, :3] = np.array([0.40, 0.50, 0.0]) + rng.normal(0, 0.005, (T, 3))
    pose[:, 12, :3] = np.array([0.60, 0.50, 0.0]) + rng.normal(0, 0.005, (T, 3))
    pose[:, :, 3] = 1.0
    seq[:, 0:132] = pose.reshape(T, -1)

    seq[:, 132:1536] = rng.normal(0.5, 0.01, (T, 1404))

    phase, freq = gloss_id * 0.7, 1.0 + 0.35 * gloss_id
    for lo, hi, sgn in ((1536, 1599, 1.0), (1599, 1662, -1.0)):
        base = np.zeros((T, 21, 3), dtype=np.float32)
        base[:, :, 0] = 0.5 + sgn * 0.12 * np.sin(2 * np.pi * freq * t + phase)
        base[:, :, 1] = 0.5 + 0.12 * np.cos(2 * np.pi * freq * t + phase)
        base[:, :, 2] = 0.02 * np.sin(4 * np.pi * freq * t)
        base += rng.normal(0, 0.012, base.shape)
        base += np.linspace(0, 1, 21)[None, :, None] * 0.03
        seq[:, lo:hi] = base.reshape(T, -1)
    return seq


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw")
    p.add_argument("--signs", type=int, default=10)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--sessions", default="A,B")
    p.add_argument("--frames", type=int, default=40)
    a = p.parse_args()

    rng = np.random.default_rng(0)
    for s in a.sessions.split(","):
        for g in range(a.signs):
            name = f"SIGN{g:02d}"
            d = Path(a.out) / s / name
            d.mkdir(parents=True, exist_ok=True)
            for r in range(a.reps):
                T = a.frames + int(rng.integers(-6, 7))
                np.save(d / f"{name}_{s}_{r:03d}.npy", make_clip(g, rng, T))
    print(f"Wrote synthetic corpus to {a.out}")


if __name__ == "__main__":
    main()
