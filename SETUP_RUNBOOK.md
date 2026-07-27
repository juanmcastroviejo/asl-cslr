# 5-Day Runbook — CAI2840C Final Project

## Wednesday 22 (today)

**1. Environment (20 min).** MediaPipe is the only fragile dependency. Do this first — if it
fails, everything downstream stalls.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import mediapipe, cv2, torch; print('ok')"
```

On Apple Silicon, if `mediapipe==0.10.14` will not install, try `pip install mediapipe` for the
newest wheel and tell me — the legacy `solutions.holistic` API was deprecated and newer builds
may require switching to the Tasks API.

**2. Verify the pipeline before recording anything (10 min).**

```bash
python scripts/make_synthetic_clips.py --out data/synthetic --signs 6 --reps 12
python -m src.train --data data/synthetic --group hands --head bilstm \
    --epochs 30 --holdout-session B --run smoke
```

Validation WER should fall to roughly 0 by epoch 15. If it does, the whole chain works and the
only remaining variable is your data.

**3. Learn the 16 signs (45 min).** Use Handspeak or Lifeprint. Practice each until it is
consistent — *your* consistency is the ceiling on model accuracy, and inconsistency will show
up as noise you cannot fix later.

**4. Record session A (45 min).**

```bash
python -m src.record --session A --reps 25
```

Then **change something** — different shirt, different lighting, sit further back — and record
session B with fewer reps:

```bash
python -m src.record --session B --reps 12
```

Session B becomes the held-out test set. This matters: it is the difference between "we scored
our model on data it effectively memorized" and a defensible generalization claim.

**5. Initialize git and commit (10 min).**

```bash
git init && git add -A && git commit -m "Pipeline: extraction, models, training, evaluation"
gh repo create asl-cslr --public --source=. --push    # or create on github.com and push
```

Commit after every meaningful step from here on.

## Thursday 23

```bash
EPOCHS=60 ./scripts/run_experiments.sh data/raw B
```

Six runs, unattended. Start it and go do something else. When it finishes you will have
`results/tables/table_main_results.md` and every figure the paper needs.

Send me the contents of `results/tables/` and I will write the Results and Discussion sections
around your actual numbers.

## Friday 24 — paper draft
## Saturday 25 — paper revision, midpoint check-in
## Sunday 26 — slides, record the video (use `src/realtime.py` for the live demo segment)
## Monday 27 — submit

## If something breaks

Send me the full traceback. Do not spend more than 15 minutes stuck on any single error —
your bottleneck is time, not difficulty.
