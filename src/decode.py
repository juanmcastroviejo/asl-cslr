"""CTC decoding: greedy collapse and prefix beam search."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

NEG_INF = -1e30
BLANK = 0


def greedy_decode(log_probs: np.ndarray, blank: int = BLANK) -> list[int]:
    """Argmax per frame, collapse repeats, drop blanks. log_probs: (T, C)."""
    best = log_probs.argmax(axis=1)
    out, prev = [], -1
    for k in best:
        k = int(k)
        if k != prev and k != blank:
            out.append(k)
        prev = k
    return out


def beam_search_decode(log_probs: np.ndarray, beam_width: int = 10,
                       blank: int = BLANK, topk: int = 8) -> list[int]:
    """Standard CTC prefix beam search (Graves et al., 2006).

    Beams are scored in log space as (p_blank, p_nonblank) per prefix.
    """
    T, C = log_probs.shape
    topk = min(topk, C)
    beams: dict[tuple, tuple[float, float]] = {(): (0.0, NEG_INF)}

    for t in range(T):
        cand: dict[tuple, list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF])
        symbols = np.argpartition(log_probs[t], -topk)[-topk:]
        for prefix, (pb, pnb) in beams.items():
            ptot = np.logaddexp(pb, pnb)
            # extend with blank -> prefix unchanged
            e = cand[prefix]
            e[0] = np.logaddexp(e[0], ptot + log_probs[t, blank])
            for c in symbols:
                c = int(c)
                if c == blank:
                    continue
                p = log_probs[t, c]
                if prefix and c == prefix[-1]:
                    # repeat of the last symbol: stays in place unless a blank
                    # separated the two emissions
                    e = cand[prefix]
                    e[1] = np.logaddexp(e[1], pnb + p)
                    e2 = cand[prefix + (c,)]
                    e2[1] = np.logaddexp(e2[1], pb + p)
                else:
                    e2 = cand[prefix + (c,)]
                    e2[1] = np.logaddexp(e2[1], ptot + p)
        beams = dict(sorted(cand.items(),
                            key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))[:beam_width])
        beams = {k: (v[0], v[1]) for k, v in beams.items()}

    best = max(beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))[0]
    return list(best)


def decode_batch(log_probs: np.ndarray, lengths, method: str = "greedy",
                 beam_width: int = 10) -> list[list[int]]:
    """log_probs: (B, T, C) -> list of gloss id sequences."""
    out = []
    for i, L in enumerate(lengths):
        lp = log_probs[i, :int(L)]
        out.append(greedy_decode(lp) if method == "greedy"
                   else beam_search_decode(lp, beam_width))
    return out
