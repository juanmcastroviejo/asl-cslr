"""Clip store, synthetic phrase construction, and CTC-ready datasets.

WLASL and self-recorded corpora both contain *isolated* signs, so phrase-level
sequences are built by concatenating isolated clips with interpolated
transition frames. Train phrases are sampled on the fly (unlimited
augmentation); validation and test phrases are generated once from a fixed
seed so every model variant is scored on identical sequences.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import FEATURE_DIM, augment_sequence, normalize_sequence, select_features

BLANK = 0   # CTC blank occupies index 0; glosses are 1..V


# --------------------------------------------------------------------------
@dataclass
class ClipStore:
    """All isolated clips, normalized, grouped by gloss and session."""
    clips: dict[str, list[np.ndarray]]
    vocab: list[str]
    sessions: dict[str, list[str]]   # gloss -> session id per clip

    @classmethod
    def load(cls, root: str | Path, group: str = "full",
             sessions: list[str] | None = None) -> "ClipStore":
        root = Path(root)
        clips: dict[str, list[np.ndarray]] = {}
        sess: dict[str, list[str]] = {}
        session_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        if sessions:
            session_dirs = [d for d in session_dirs if d.name in sessions]
        for sdir in session_dirs:
            for gdir in sorted(d for d in sdir.iterdir() if d.is_dir()):
                for f in sorted(gdir.glob("*.npy")):
                    arr = np.load(f)
                    if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM or arr.shape[0] < 4:
                        continue
                    arr = select_features(normalize_sequence(arr), group)
                    clips.setdefault(gdir.name, []).append(arr.astype(np.float32))
                    sess.setdefault(gdir.name, []).append(sdir.name)
        vocab = sorted(clips)
        if not vocab:
            raise SystemExit(f"No clips found under {root}")
        return cls(clips, vocab, sess)

    @property
    def feature_dim(self) -> int:
        return next(iter(self.clips.values()))[0].shape[1]

    def counts(self) -> dict[str, int]:
        return {g: len(v) for g, v in self.clips.items()}


# --------------------------------------------------------------------------
def make_phrase(store: ClipStore, gloss_ids: list[int], rng: np.random.Generator,
                transition_frames: int = 5, pool: list[list[int]] | None = None
                ) -> tuple[np.ndarray, list[int]]:
    """Concatenate one clip per gloss, blending across sign boundaries."""
    segments: list[np.ndarray] = []
    for k, gid in enumerate(gloss_ids):
        gloss = store.vocab[gid - 1]
        idxs = pool[gid - 1] if pool is not None else range(len(store.clips[gloss]))
        clip = store.clips[gloss][int(rng.choice(list(idxs)))]
        if segments and transition_frames > 0:
            a, b = segments[-1][-1], clip[0]
            w = np.linspace(0, 1, transition_frames + 2, dtype=np.float32)[1:-1, None]
            segments.append((1 - w) * a[None, :] + w * b[None, :])
        segments.append(clip)
    return np.concatenate(segments, axis=0), list(gloss_ids)


def sample_lengths(rng: np.random.Generator, lengths: list[int]) -> int:
    return int(rng.choice(lengths))


# --------------------------------------------------------------------------
class PhraseDataset(Dataset):
    """On-the-fly phrase sampler used for training."""

    def __init__(self, store: ClipStore, clip_pool: list[list[int]], n_samples: int,
                 lengths=(2, 3, 4, 5, 6), seed: int = 0, augment: bool = True,
                 transition_frames: int = 5, stats: tuple[np.ndarray, np.ndarray] | None = None):
        self.store, self.pool, self.n = store, clip_pool, n_samples
        self.lengths, self.augment = list(lengths), augment
        self.transition_frames, self.stats, self.seed = transition_frames, stats, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        V = len(self.store.vocab)
        L = sample_lengths(rng, self.lengths)
        ids = [int(x) + 1 for x in rng.choice(V, size=min(L, V), replace=False)]
        X, y = make_phrase(self.store, ids, rng, self.transition_frames, self.pool)
        if self.augment:
            X = _augment_flat(X, rng)
        if self.stats is not None:
            X = (X - self.stats[0]) / self.stats[1]
        return torch.from_numpy(np.ascontiguousarray(X)), torch.tensor(y, dtype=torch.long)


class FixedPhraseDataset(Dataset):
    """Deterministic phrase set for validation / test."""

    def __init__(self, store: ClipStore, items: list[dict], seed: int = 1234,
                 transition_frames: int = 5, stats=None):
        self.store, self.items = store, items
        self.transition_frames, self.stats, self.seed = transition_frames, stats, seed
        self._cache: dict[int, tuple[np.ndarray, list[int]]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        if i not in self._cache:
            it = self.items[i]
            rng = np.random.default_rng(self.seed + i)
            segs = []
            for gid, ci in zip(it["ids"], it["clip_idx"]):
                clip = self.store.clips[self.store.vocab[gid - 1]][ci]
                if segs and self.transition_frames > 0:
                    a, b = segs[-1][-1], clip[0]
                    w = np.linspace(0, 1, self.transition_frames + 2, dtype=np.float32)[1:-1, None]
                    segs.append((1 - w) * a[None, :] + w * b[None, :])
                segs.append(clip)
            X = np.concatenate(segs, 0)
            if self.stats is not None:
                X = (X - self.stats[0]) / self.stats[1]
            self._cache[i] = (X.astype(np.float32), list(it["ids"]))
        X, y = self._cache[i]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)


def _augment_flat(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augmentation on already-sliced features (jitter + global scale)."""
    s = rng.uniform(0.9, 1.1)
    return (X * s + rng.normal(0, 0.01, X.shape).astype(np.float32)).astype(np.float32)


# --------------------------------------------------------------------------
def build_fixed_items(store: ClipStore, clip_pool: list[list[int]], n: int,
                      lengths=(2, 3, 4, 5, 6), seed: int = 1234) -> list[dict]:
    rng = np.random.default_rng(seed)
    V = len(store.vocab)
    items = []
    for _ in range(n):
        L = sample_lengths(rng, list(lengths))
        ids = [int(x) + 1 for x in rng.choice(V, size=min(L, V), replace=False)]
        clip_idx = [int(rng.choice(clip_pool[g - 1])) for g in ids]
        items.append({"ids": ids, "clip_idx": clip_idx, "length": len(ids)})
    return items


def compute_stats(store: ClipStore, clip_pool: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Feature-wise mean/std over the training clips only (no leakage)."""
    frames = [store.clips[g][i] for gi, g in enumerate(store.vocab) for i in clip_pool[gi]]
    allf = np.concatenate(frames, axis=0)
    mean = allf.mean(0)
    std = allf.std(0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def collate(batch):
    """Pad to the longest sequence and emit CTC-shaped tensors."""
    xs, ys = zip(*batch)
    in_lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    tgt_lens = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)
    T, D = int(in_lens.max()), xs[0].shape[1]
    padded = torch.zeros(len(xs), T, D, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, :x.shape[0]] = x
    return padded, torch.cat(ys), in_lens, tgt_lens


def split_clip_pools(store: ClipStore, val_frac: float = 0.15, seed: int = 0,
                     holdout_session: str | None = None):
    """Return (train_pool, val_pool, test_pool) as per-gloss clip index lists.

    If `holdout_session` is given, every clip from that session becomes the
    test set -- a signer/session-independent split, which is far stricter
    than a random split and is what the paper reports.
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for g in store.vocab:
        n = len(store.clips[g])
        idx = np.arange(n)
        if holdout_session:
            sess = np.array(store.sessions[g])
            test_i = idx[sess == holdout_session]
            rest = idx[sess != holdout_session]
        else:
            rng.shuffle(idx)
            cut = max(1, int(0.15 * n))
            test_i, rest = idx[:cut], idx[cut:]
        rest = rest.copy()
        rng.shuffle(rest)
        vcut = max(1, int(val_frac * len(rest)))
        val.append(rest[:vcut].tolist())
        train.append(rest[vcut:].tolist())
        test.append(test_i.tolist())
        if not train[-1] or not test[-1]:
            raise SystemExit(f"Gloss {g} has too few clips to split ({n}).")
    return train, val, test


def save_manifest(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
