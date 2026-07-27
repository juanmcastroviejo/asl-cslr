"""Measure per-frame inference latency on CPU.

Reports the two costs separately, because they behave very differently:
landmark extraction runs on every frame, while the sequence model runs once
per sliding window.

    python -m src.benchmark --runs results/full_bilstm results/full_transformer
    python -m src.benchmark --runs ... --with-mediapipe   # adds extraction cost
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .models import CSLRModel, count_params


def bench_model(ckpt: Path, window: int, iters: int, warmup: int) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = CSLRModel(ck["feature_dim"], len(ck["vocab"]), head=a["head"],
                      d_model=a.get("d_model", 256), hidden=a.get("hidden", 256),
                      layers=a.get("layers", 2))
    model.load_state_dict(ck["model"])
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    x = torch.randn(1, window, ck["feature_dim"])
    ln = torch.tensor([window])
    with torch.no_grad():
        for _ in range(warmup):
            model(x, ln)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x, ln)
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return {"model_ms_mean": float(ts.mean()), "model_ms_p50": float(np.percentile(ts, 50)),
            "model_ms_p95": float(np.percentile(ts, 95)), "model_ms_std": float(ts.std()),
            "window_frames": window, "params": count_params(model),
            "per_frame_amortized_ms": float(ts.mean() / window)}


def bench_mediapipe(n: int = 200) -> dict:
    """Cost of extracting landmarks from one frame (synthetic 640x480 input)."""
    from .features import landmarks_to_vector, make_holistic
    h = make_holistic()
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
    for f in frames[:3]:
        h.process(f)
    ts = []
    for i in range(n):
        f = frames[i % len(frames)]
        t0 = time.perf_counter()
        landmarks_to_vector(h.process(f))
        ts.append((time.perf_counter() - t0) * 1000.0)
    h.close()
    ts = np.array(ts)
    return {"mediapipe_ms_mean": float(ts.mean()),
            "mediapipe_ms_p95": float(np.percentile(ts, 95))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--with-mediapipe", action="store_true")
    p.add_argument("--out", default="results/latency.json")
    a = p.parse_args()

    results = {}
    mp_cost = bench_mediapipe() if a.with_mediapipe else {}
    for d in a.runs:
        d = Path(d)
        ck = d / "best.pt"
        if not ck.exists():
            print(f"  (skipping {d}: no checkpoint)")
            continue
        r = bench_model(ck, a.window, a.iters, a.warmup)
        r.update(mp_cost)
        if mp_cost:
            r["end_to_end_per_frame_ms"] = (mp_cost["mediapipe_ms_mean"]
                                            + r["per_frame_amortized_ms"])
        results[d.name] = r
        print(f"{d.name}: model {r['model_ms_mean']:.1f} ms/window "
              f"({r['per_frame_amortized_ms']:.2f} ms/frame amortized)")

    results["_environment"] = {"platform": platform.platform(),
                               "processor": platform.processor(),
                               "python": platform.python_version(),
                               "torch": torch.__version__,
                               "torch_threads": torch.get_num_threads()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print("Wrote", a.out)


if __name__ == "__main__":
    main()
