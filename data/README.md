# Data

Landmark clips are **not** committed to this repository.

## Self-recorded corpus

Generate synthetic data for testing:

```bash
python scripts/make_synthetic_clips.py --out data/raw --signs 10 --reps 20
```

This creates:
Each `.npy` is a float32 array of shape (T, 1662) — landmark sequences only, no video.
