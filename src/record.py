"""Guided webcam recorder for isolated ASL sign clips.

Landmarks are extracted live and saved as .npy arrays of shape (T, 1662).
Raw video is never written to disk, which keeps the corpus small and avoids
storing identifiable footage.

Usage
-----
    python -m src.record --session A --reps 25
    python -m src.record --session B --reps 15 --signs HELLO,PLEASE,SORRY

Controls: SPACE = start/advance, R = redo last take, S = skip sign, Q = quit.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .features import draw_landmarks, landmarks_to_vector, make_holistic

DEFAULT_SIGNS = [
    "HELLO", "GOODBYE", "PLEASE", "SORRY", "THANK-YOU", "YES", "NO", "HELP",
    "WANT", "NEED", "EAT", "DRINK", "MOTHER", "FATHER", "MORE", "FINISH",
]


def _banner(frame, lines, color=(0, 255, 0)):
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, 34 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, 34 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
    return frame


def record(args) -> None:
    signs = args.signs.split(",") if args.signs else DEFAULT_SIGNS
    out_root = Path(args.out) / args.session
    out_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Check camera permissions.")

    holistic = make_holistic()
    print(f"Recording session '{args.session}': {len(signs)} signs x {args.reps} reps")

    try:
        for sign in signs:
            sign_dir = out_root / sign
            sign_dir.mkdir(exist_ok=True)
            rep = len(list(sign_dir.glob("*.npy")))   # resume where we left off
            while rep < args.reps:
                # --- idle / ready screen -------------------------------
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    _banner(frame, [f"SIGN: {sign}", f"take {rep + 1} / {args.reps}",
                                    "SPACE=record  S=skip sign  Q=quit"])
                    cv2.imshow("recorder", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        break
                    if key == ord("s"):
                        rep = args.reps
                        break
                    if key == ord("q"):
                        return
                if rep >= args.reps:
                    break

                # --- countdown -----------------------------------------
                t0 = time.time()
                while time.time() - t0 < args.countdown:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    left = args.countdown - (time.time() - t0)
                    _banner(frame, [f"{sign}", f"{left:.1f}"], (0, 200, 255))
                    cv2.imshow("recorder", frame)
                    cv2.waitKey(1)

                # --- capture -------------------------------------------
                seq, t0 = [], time.time()
                while time.time() - t0 < args.seconds:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    res = holistic.process(rgb)
                    seq.append(landmarks_to_vector(res))
                    draw_landmarks(frame, res)
                    _banner(frame, [f"REC {sign}", f"{len(seq)} frames"], (0, 0, 255))
                    cv2.imshow("recorder", frame)
                    cv2.waitKey(1)

                arr = np.asarray(seq, dtype=np.float32)
                hands_seen = float(np.mean(np.any(arr[:, 1536:], axis=1)))
                # --- confirm -------------------------------------------
                ok_take = True
                if hands_seen < 0.5:
                    print(f"  !! only {hands_seen:.0%} of frames saw a hand - redo suggested")
                    ok_take = False
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    _banner(frame,
                            [f"{sign}: {arr.shape[0]} frames, hands in {hands_seen:.0%}",
                             "SPACE=keep   R=redo"],
                            (0, 255, 0) if ok_take else (0, 0, 255))
                    cv2.imshow("recorder", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        np.save(sign_dir / f"{sign}_{args.session}_{rep:03d}.npy", arr)
                        rep += 1
                        break
                    if key == ord("r"):
                        break
                    if key == ord("q"):
                        return
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        print("Done. Clips written to", out_root)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", required=True, help="session id, e.g. A or B")
    p.add_argument("--signs", default="", help="comma-separated glosses (default: 16-sign set)")
    p.add_argument("--reps", type=int, default=25)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--countdown", type=float, default=2.0)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", default="data/raw")
    record(p.parse_args())


if __name__ == "__main__":
    main()
