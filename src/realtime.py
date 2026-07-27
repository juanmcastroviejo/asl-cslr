"""Live webcam CSLR demo.

Maintains a sliding buffer of landmark frames, re-decodes every N milliseconds,
and overlays the predicted gloss sequence on the video feed. Record your screen
running this for the presentation video.

    python -m src.realtime --ckpt results/full_bilstm/best.pt
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

from .decode import beam_search_decode, greedy_decode
from .features import (draw_landmarks, landmarks_to_vector, make_holistic,
                       normalize_sequence, select_features)
from .models import CSLRModel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--buffer", type=int, default=64)
    p.add_argument("--interval-ms", type=int, default=500)
    p.add_argument("--decode", default="beam", choices=["greedy", "beam"])
    p.add_argument("--beam-width", type=int, default=10)
    p.add_argument("--camera", type=int, default=0)
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, vocab = ck["args"], ck["vocab"]
    group = cfg.get("group", "full")
    mean, std = ck["stats"]
    model = CSLRModel(ck["feature_dim"], len(vocab), head=cfg["head"],
                      d_model=cfg.get("d_model", 256), hidden=cfg.get("hidden", 256),
                      layers=cfg.get("layers", 2))
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded {a.ckpt}: {len(vocab)} glosses, features={group}, head={cfg['head']}")

    holistic = make_holistic()
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    buf: deque = deque(maxlen=a.buffer)
    prediction, last_decode, infer_ms, fps_ema = "", 0.0, 0.0, 0.0

    try:
        while True:
            t_frame = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            buf.append(landmarks_to_vector(res))
            draw_landmarks(frame, res)

            now = time.time() * 1000
            if len(buf) >= 16 and now - last_decode >= a.interval_ms:
                last_decode = now
                seq = normalize_sequence(np.asarray(buf, dtype=np.float32))
                X = (select_features(seq, group) - mean) / std
                xt = torch.from_numpy(X[None].astype(np.float32))
                t0 = time.perf_counter()
                with torch.no_grad():
                    lp = model(xt, torch.tensor([X.shape[0]]))[0].numpy()
                ids = (greedy_decode(lp) if a.decode == "greedy"
                       else beam_search_decode(lp, a.beam_width))
                infer_ms = (time.perf_counter() - t0) * 1000
                prediction = " ".join(vocab[i - 1] for i in ids) or "..."

            dt = time.perf_counter() - t_frame
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(dt, 1e-6))
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 74), (0, 0, 0), -1)
            cv2.putText(frame, prediction[:60], (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"decode {infer_ms:.0f} ms | {fps_ema:.1f} fps | buffer {len(buf)}",
                        (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("ASL CSLR (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()


if __name__ == "__main__":
    main()
