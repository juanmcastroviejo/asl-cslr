# Continuous Sign Language Recognition for Phrase-Level ASL Understanding

**Course:** CAI2840C — Introduction to Computer Vision, Miami Dade College
**Team member:** Juan Castroviejo
**Instructor:** Professor Desiree Dominguez

## Description

This project implements a continuous sign language recognition (CSLR) pipeline that decodes
*sequences* of American Sign Language glosses from video, rather than classifying isolated
signs one at a time. Each frame is reduced to a 1,662-dimensional MediaPipe Holistic landmark
vector covering hand, body-pose, and facial keypoints; those landmark sequences are normalized
against shoulder position and width, then passed through a dilated temporal convolutional
encoder followed by either a bidirectional LSTM or a Transformer encoder. Both variants are
trained with Connectionist Temporal Classification (CTC) loss, which learns sign boundaries
from sequence-level supervision alone and therefore requires no frame-by-frame annotation.
Because the source corpora contain only isolated signs, phrase-level sequences are constructed
synthetically by concatenating isolated clips with interpolated transition frames. The system
is evaluated on word error rate, its substitution/deletion/insertion decomposition, degradation
as phrase length grows, the marginal contribution of pose and facial landmarks, and per-frame
inference latency on consumer CPU hardware.

## Research questions

1. How does word error rate change as phrase length grows from two glosses to six?
2. How much do facial and body-pose landmarks contribute to accuracy relative to hands alone?
3. How does inference latency compare between the LSTM and Transformer decoder variants at
   equivalent accuracy?

## Repository layout

```
src/
  features.py     MediaPipe Holistic extraction, normalization, augmentation, feature groups
  record.py       guided webcam recorder -> per-clip landmark arrays
  dataset.py      clip store, synthetic phrase construction, CTC collation, data splits
  models.py       TCN encoder with BiLSTM / Transformer heads
  decode.py       CTC greedy decoding and prefix beam search
  metrics.py      WER, S/D/I decomposition, per-length breakdown, confusion matrices
  train.py        CTC training loop with early stopping on validation WER
  evaluate.py     aggregates runs into paper-ready tables and figures
  benchmark.py    per-frame latency measurement (extraction vs. model cost)
  realtime.py     live webcam demo with sliding buffer and gloss overlay
scripts/
  make_synthetic_clips.py   fake corpus for pipeline testing only
  run_experiments.sh        full 3 x 2 experiment grid
data/                       corpus (see data/README.md)
results/                    checkpoints, metrics, figures, tables
```

## Setup

Python 3.11 is required — MediaPipe does not publish wheels for every newer version.

```bash
git clone <this-repo-url>
cd asl-cslr
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the code

**1. Verify the pipeline** (no camera or dataset needed; synthetic data only):

```bash
python scripts/make_synthetic_clips.py --out data/synthetic --signs 6 --reps 12
python -m src.train --data data/synthetic --group hands --head bilstm \
    --epochs 30 --holdout-session B --run smoke
```

**2. Record the corpus.** Two sessions, varying lighting/clothing/position between them, so
that session B can be held out as a signer-condition-independent test set:

```bash
python -m src.record --session A --reps 25
python -m src.record --session B --reps 12
```

**3. Run the full experiment grid** (three feature sets x two architectures, then latency
benchmarking and figure generation):

```bash
./scripts/run_experiments.sh data/raw B
```

**4. Live demo:**

```bash
python -m src.realtime --ckpt results/full_bilstm/best.pt
```

## Dependencies

`mediapipe`, `opencv-python`, `numpy<2`, `torch`, `matplotlib`, `tqdm`. Exact pins are in
`requirements.txt`. Training runs on CPU by default; the models are small enough that this is
practical, and CTC loss is not reliably supported on Apple's MPS backend.

## Reproducibility

Random seeds are fixed for data splits, phrase construction, and model initialization.
Validation and test phrase sets are generated once from fixed seeds and are byte-identical
across all six runs, so architecture and feature comparisons are scored on the same sequences.
Normalization statistics are computed on training clips only.

## License and data use

Code is released for academic use. Self-recorded landmark data contains no raw video and no
personally identifying information. If WLASL is used, it remains subject to its own academic
license and is not redistributed here.
