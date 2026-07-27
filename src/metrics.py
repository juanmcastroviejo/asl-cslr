"""Sequence-level evaluation metrics for CSLR."""
from __future__ import annotations

from collections import Counter

import numpy as np


def edit_ops(ref: list[int], hyp: list[int]) -> tuple[int, int, int, list[tuple]]:
    """Levenshtein alignment. Returns (subs, dels, ins, alignment ops)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)

    i, j, ops = n, m, []
    s = dl = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] == hyp[j - 1]:
                ops.append(("ok", ref[i - 1], hyp[j - 1]))
            else:
                ops.append(("sub", ref[i - 1], hyp[j - 1]))
                s += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], None))
            dl += 1
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            ins += 1
            j -= 1
    return s, dl, ins, ops[::-1]


def wer(refs: list[list[int]], hyps: list[list[int]]) -> dict:
    """Corpus-level word error rate plus its S/D/I decomposition."""
    S = D = I = N = 0
    exact = 0
    for r, h in zip(refs, hyps):
        s, d, i, _ = edit_ops(r, h)
        S, D, I, N = S + s, D + d, I + i, N + len(r)
        exact += int(r == h)
    N = max(N, 1)
    return {"wer": (S + D + I) / N, "sub": S / N, "del": D / N, "ins": I / N,
            "sentence_acc": exact / max(len(refs), 1), "n_tokens": N}


def wer_by_length(refs: list[list[int]], hyps: list[list[int]]) -> dict[int, dict]:
    """WER bucketed by reference phrase length -> answers RQ1."""
    buckets: dict[int, tuple[list, list]] = {}
    for r, h in zip(refs, hyps):
        buckets.setdefault(len(r), ([], []))
        buckets[len(r)][0].append(r)
        buckets[len(r)][1].append(h)
    return {L: {**wer(r, h), "n_phrases": len(r)} for L, (r, h) in sorted(buckets.items())}


def confusion_pairs(refs: list[list[int]], hyps: list[list[int]]) -> Counter:
    """Count substitution pairs (reference gloss -> predicted gloss)."""
    c: Counter = Counter()
    for r, h in zip(refs, hyps):
        for op, a, b in edit_ops(r, h)[3]:
            if op == "sub":
                c[(a, b)] += 1
    return c


def confusion_matrix(refs: list[list[int]], hyps: list[list[int]], vocab_size: int) -> np.ndarray:
    """(V+1) x (V+1) matrix; last row/col are insertions/deletions."""
    M = np.zeros((vocab_size + 1, vocab_size + 1), dtype=np.int32)
    for r, h in zip(refs, hyps):
        for op, a, b in edit_ops(r, h)[3]:
            if op in ("ok", "sub"):
                M[a - 1, b - 1] += 1
            elif op == "del":
                M[a - 1, vocab_size] += 1
            elif op == "ins":
                M[vocab_size, b - 1] += 1
    return M
