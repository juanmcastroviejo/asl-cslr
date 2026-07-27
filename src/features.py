"""MediaPipe Holistic landmark extraction and feature engineering.

Feature vector layout (1662 dims total), fixed ordering used everywhere:
    [   0 : 132 ]  pose   -> 33 landmarks x (x, y, z, visibility)
    [ 132 :1536 ]  face   -> 468 landmarks x (x, y, z)
    [1536 :1599 ]  left   -> 21 landmarks x (x, y, z)
    [1599 :1662 ]  right  -> 21 landmarks x (x, y, z)
"""
from __future__ import annotations

import numpy as np

POSE_SLICE = (0, 132)
FACE_SLICE = (132, 1536)
LH_SLICE = (1536, 1599)
RH_SLICE = (1599, 1662)
FEATURE_DIM = 1662

# Ablation groups -> answers RQ2 (contribution of pose / face landmarks).
FEATURE_GROUPS: dict[str, list[tuple[int, int]]] = {
    "hands": [LH_SLICE, RH_SLICE],                          # 126 dims
    "hands_pose": [POSE_SLICE, LH_SLICE, RH_SLICE],         # 258 dims
    "full": [POSE_SLICE, FACE_SLICE, LH_SLICE, RH_SLICE],   # 1662 dims
}

# MediaPipe pose landmark indices for the shoulders.
L_SHOULDER, R_SHOULDER = 11, 12


def group_dim(group: str) -> int:
    return sum(b - a for a, b in FEATURE_GROUPS[group])


def select_features(seq: np.ndarray, group: str) -> np.ndarray:
    """Slice a (T, 1662) sequence down to one ablation group."""
    return np.concatenate([seq[:, a:b] for a, b in FEATURE_GROUPS[group]], axis=1)


# --------------------------------------------------------------------------
# Extraction (requires mediapipe; imported lazily so training works headless)
# --------------------------------------------------------------------------
def make_holistic(min_detection_confidence: float = 0.5,
                  min_tracking_confidence: float = 0.5):
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def landmarks_to_vector(results) -> np.ndarray:
    """Flatten one MediaPipe Holistic result into the 1662-dim vector.

    Absent body parts become all-zeros; downstream code treats an all-zero
    block as 'missing' and repairs it by interpolation.
    """
    pose = (np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark],
                     dtype=np.float32).ravel()
            if results.pose_landmarks else np.zeros(132, np.float32))
    face = (np.array([[p.x, p.y, p.z] for p in results.face_landmarks.landmark],
                     dtype=np.float32).ravel()
            if results.face_landmarks else np.zeros(1404, np.float32))
    lh = (np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark],
                   dtype=np.float32).ravel()
          if results.left_hand_landmarks else np.zeros(63, np.float32))
    rh = (np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark],
                   dtype=np.float32).ravel()
          if results.right_hand_landmarks else np.zeros(63, np.float32))
    return np.concatenate([pose, face, lh, rh])


def draw_landmarks(image, results):
    """Overlay landmarks on a BGR frame (for the recorder / live demo)."""
    import mediapipe as mp
    mp_d, mp_h = mp.solutions.drawing_utils, mp.solutions.holistic
    mp_d.draw_landmarks(image, results.pose_landmarks, mp_h.POSE_CONNECTIONS)
    mp_d.draw_landmarks(image, results.left_hand_landmarks, mp_h.HAND_CONNECTIONS)
    mp_d.draw_landmarks(image, results.right_hand_landmarks, mp_h.HAND_CONNECTIONS)
    return image


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def _block_missing(seq: np.ndarray, sl: tuple[int, int]) -> np.ndarray:
    """Boolean (T,) mask: True where this landmark block is entirely zero."""
    return ~np.any(seq[:, sl[0]:sl[1]], axis=1)


def interpolate_missing(seq: np.ndarray) -> np.ndarray:
    """Linearly interpolate frames where a hand block dropped out.

    Implements the proposal's requirement that low-confidence frames are
    repaired rather than dropped, so the temporal sequence stays intact.
    """
    seq = seq.copy()
    T = seq.shape[0]
    for sl in (LH_SLICE, RH_SLICE):
        missing = _block_missing(seq, sl)
        present = np.flatnonzero(~missing)
        if present.size == 0 or present.size == T:
            continue
        block = seq[:, sl[0]:sl[1]]
        for j in range(block.shape[1]):
            block[missing, j] = np.interp(np.flatnonzero(missing), present, block[present, j])
        seq[:, sl[0]:sl[1]] = block
    return seq


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Translation- and scale-invariant normalization.

    Every coordinate is re-expressed relative to the shoulder midpoint and
    divided by shoulder width, so the model cannot cheat off where the signer
    happened to stand or how close they sat to the camera.
    """
    seq = interpolate_missing(seq).copy()
    T = seq.shape[0]

    pose = seq[:, POSE_SLICE[0]:POSE_SLICE[1]].reshape(T, 33, 4)
    ls, rs = pose[:, L_SHOULDER, :3], pose[:, R_SHOULDER, :3]
    center = (ls + rs) / 2.0                                     # (T, 3)
    scale = np.linalg.norm(ls - rs, axis=1, keepdims=True)       # (T, 1)
    scale = np.where(scale < 1e-3, 1.0, scale)

    def _apply(sl: tuple[int, int], n_pts: int, stride: int) -> None:
        was_missing = _block_missing(seq, sl)
        blk = seq[:, sl[0]:sl[1]].reshape(T, n_pts, stride)
        blk[..., :3] = (blk[..., :3] - center[:, None, :]) / scale[:, None, :]
        blk[was_missing] = 0.0        # keep 'missing' encoded as exact zeros
        seq[:, sl[0]:sl[1]] = blk.reshape(T, -1)

    _apply(POSE_SLICE, 33, 4)
    _apply(FACE_SLICE, 468, 3)
    _apply(LH_SLICE, 21, 3)
    _apply(RH_SLICE, 21, 3)
    return seq.astype(np.float32)


# --------------------------------------------------------------------------
# Landmark-space augmentation (training only)
# --------------------------------------------------------------------------
def augment_sequence(seq: np.ndarray, rng: np.random.Generator,
                     max_rot_deg: float = 15.0,
                     scale_range: tuple[float, float] = (0.8, 1.2),
                     jitter_std: float = 0.01) -> np.ndarray:
    """Random in-plane rotation, isotropic scaling, and Gaussian jitter."""
    seq = seq.copy()
    T = seq.shape[0]
    theta = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
    s = rng.uniform(*scale_range)
    c, si = np.cos(theta) * s, np.sin(theta) * s
    R = np.array([[c, -si], [si, c]], dtype=np.float32)

    for sl, n_pts, stride in ((POSE_SLICE, 33, 4), (FACE_SLICE, 468, 3),
                              (LH_SLICE, 21, 3), (RH_SLICE, 21, 3)):
        was_missing = _block_missing(seq, sl)
        blk = seq[:, sl[0]:sl[1]].reshape(T, n_pts, stride)
        blk[..., :2] = blk[..., :2] @ R.T
        blk[..., 2] *= s
        blk[..., :3] += rng.normal(0, jitter_std, blk[..., :3].shape).astype(np.float32)
        blk[was_missing] = 0.0
        seq[:, sl[0]:sl[1]] = blk.reshape(T, -1)
    return seq
