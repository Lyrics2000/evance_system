"""
hmm_wlasl_fixed.py
==================
Corrected HMM baseline for WLASL sign recognition.

Replaces the feature/training/prediction/evaluation logic of
`02_hmm_wlasl_baseline.ipynb`, which collapsed to a single-class predictor
(accuracy 0.2353 == majority-class baseline).

KEY POINT: this script rebuilds features FROM THE EXISTING CACHE
(`ksl_project_data/features/wlasl_hmm_sequences.joblib`). MediaPipe does NOT
need to be re-run. The cached frames are 1629-D with the known layout:

    [   0 :   99 ]  pose        33 landmarks x (x,y,z)
    [  99 :  162 ]  left hand   21 landmarks x (x,y,z)
    [ 162 :  225 ]  right hand  21 landmarks x (x,y,z)
    [ 225 : 1629 ]  face       468 landmarks x (x,y,z)   <-- DISCARDED

--------------------------------------------------------------------------
ADR-001: Why discard the face block
    Face is 1404 / 1629 = 86.2% of every frame vector. StandardScaler gives
    each dimension equal weight, so PCA's leading components are dominated by
    facial geometry and head pose, which carry no information distinguishing
    'before' / 'computer' / 'cool' / 'cousin' / 'drink'. Those glosses are
    disambiguated by HAND SHAPE and HAND TRAJECTORY. Dropping face raises the
    signal fraction of the input from ~14% to ~100%.
    Rejected alternative: weighting face down. Rejected because it introduces a
    hyperparameter with no data to tune it on (50 training sequences).

ADR-002: Why explicit presence flags instead of zero-fill
    The original `flatten_landmarks` returns a zero vector when MediaPipe
    fails to detect a hand. Downstream, StandardScaler maps 0.0 to
    (0 - mean)/std, and since MediaPipe x/y coordinates are image-normalised
    with mean ~0.5, a missing hand becomes a LARGE-MAGNITUDE outlier rather
    than a neutral value. PCA then spends its leading components modelling
    detector dropout. We instead normalise only observed blocks, leave missing
    blocks at exact zero AFTER centring (so zero == "at body centre", a neutral
    point), and expose missingness as 3 explicit binary channels.

ADR-003: Why body-centred, shoulder-scaled coordinates
    Raw MediaPipe x/y are image-relative. Signer position in frame and camera
    distance therefore vary more across WLASL clips than the signs themselves.
    We translate to the mid-shoulder point and divide by inter-shoulder
    distance, making features invariant to translation and scale.

ADR-004: Why velocity (delta) features
    A Gaussian HMM emission is a static description of one frame. Sign identity
    lives substantially in motion. First-order differences are the standard
    cheap fix (cf. delta-cepstra in speech HMMs). Second-order (acceleration)
    deltas are omitted: at ~35 frames/sequence they are too noisy.

ADR-005: Why a fixed left-to-right (Bakis) transition matrix
    Signs are monotonic: you do not return to the preparation phase after the
    stroke. An ergodic 5-state HMM must estimate 25 transition probabilities
    from ~350 highly autocorrelated frames per class. The original run proves
    this fails -- hmmlearn emitted "Some rows of transmat_ have zero sum because
    no transition from the state was ever observed" for 4 of 5 classes, which
    makes model.score() return -inf and is the direct cause of the collapse.
    We fix startprob_ and transmat_ to a left-to-right prior and train ONLY the
    Gaussian emissions (params="mc"). This removes 30 free parameters per class
    and makes zero-sum rows structurally impossible.
    Set TRAIN_TRANSITIONS = True to re-enable transition learning; a sanitiser
    then repairs any zero rows rather than letting them poison score().

ADR-006: Why StratifiedKFold instead of a single 25% holdout
    The original test set is 17 sequences. One sample flipping moves accuracy
    by 5.9 points, so the reported number carries no usable signal. 5-fold
    stratified CV over all 67 sequences yields out-of-fold predictions for
    every sample plus a fold-to-fold standard deviation.

--------------------------------------------------------------------------
RUN
    pip install --break-system-packages numpy scipy scikit-learn hmmlearn joblib
    cd "<project>/dataset/ipynb_checkpoints"
    python3 hmm_wlasl_fixed.py

    Optional:
        python3 hmm_wlasl_fixed.py --diagnose-only   # explain the old failure
        python3 hmm_wlasl_fixed.py --states 4 --pca 20
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path("ksl_project_data").resolve()
FEATURE_DIR = PROJECT_DIR / "features"
MODEL_DIR = PROJECT_DIR / "models"
LOG_DIR = PROJECT_DIR / "logs"
REPORT_DIR = PROJECT_DIR / "reports"

for _d in (PROJECT_DIR, FEATURE_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEQUENCE_CACHE = FEATURE_DIR / "wlasl_hmm_sequences.joblib"

# Frame layout of the cached 1629-D vectors.
POSE_OFF, POSE_D = 0, 99      # 33 landmarks * 3
LH_OFF, LH_D = 99, 63         # 21 landmarks * 3
RH_OFF, RH_D = 162, 63        # 21 landmarks * 3
FACE_OFF, FACE_D = 225, 1404  # 468 landmarks * 3
EXPECTED_RAW_DIM = 1629

# MediaPipe Pose landmark indices we keep (upper body only).
# 0 nose | 11/12 shoulders | 13/14 elbows | 15/16 wrists | 23/24 hips
UPPER_POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
L_SHOULDER, R_SHOULDER = 11, 12

N_COORD_DIMS = (len(UPPER_POSE_IDX) * 3) + LH_D + RH_D   # 27 + 63 + 63 = 153
N_FLAG_DIMS = 3                                          # pose / LH / RH present
N_BASE_DIMS = N_COORD_DIMS + N_FLAG_DIMS                 # 156
N_FULL_DIMS = N_BASE_DIMS + N_COORD_DIMS                 # 156 + 153 = 309

# Defaults (override on the command line).
DEFAULT_PCA_COMPONENTS = 16
DEFAULT_HMM_STATES = 3
DEFAULT_N_FOLDS = 5
DEFAULT_N_ITER = 200
MIN_COVAR = 1e-3
TRAIN_TRANSITIONS = False   # see ADR-005
RANDOM_STATE = 42

logger = logging.getLogger("hmm_fixed")


def configure_logging() -> Path:
    log_file = LOG_DIR / f"hmm_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # hmmlearn warns via the warnings module; surface them as log records once.
    logging.captureWarnings(True)
    return log_file


# ---------------------------------------------------------------------------
# 1. Errors
# ---------------------------------------------------------------------------

class PipelineError(RuntimeError):
    """Base error carrying a stable code for log grepping."""

    code = "HMM-0000"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.code}] {message}")


class CacheMissingError(PipelineError):
    code = "HMM-1001"


class CacheShapeError(PipelineError):
    code = "HMM-1002"


class InsufficientDataError(PipelineError):
    code = "HMM-1003"


# ---------------------------------------------------------------------------
# 2. Load and validate the cached sequences
# ---------------------------------------------------------------------------

def load_cached_sequences(cache_path: Path = SEQUENCE_CACHE):
    """Return (sequences, labels) from the joblib cache, validating shape."""
    if not cache_path.exists():
        raise CacheMissingError(
            f"Sequence cache not found at {cache_path}. Run cell 14 of "
            f"02_hmm_wlasl_baseline.ipynb first to build it."
        )

    payload = joblib.load(cache_path)
    if not isinstance(payload, dict) or "sequences" not in payload or "labels" not in payload:
        raise CacheShapeError(
            f"Cache at {cache_path} has keys {list(payload)!r}; expected "
            f"'sequences' and 'labels'."
        )

    sequences = list(payload["sequences"])
    labels = list(payload["labels"])

    if len(sequences) != len(labels):
        raise CacheShapeError(
            f"Cache is inconsistent: {len(sequences)} sequences vs {len(labels)} labels."
        )
    if not sequences:
        raise InsufficientDataError("Cache contains zero sequences.")

    for i, seq in enumerate(sequences):
        arr = np.asarray(seq)
        if arr.ndim != 2:
            raise CacheShapeError(f"Sequence {i} has ndim={arr.ndim}, expected 2.")
        if arr.shape[1] != EXPECTED_RAW_DIM:
            raise CacheShapeError(
                f"Sequence {i} has {arr.shape[1]} features, expected "
                f"{EXPECTED_RAW_DIM} (pose 99 + LH 63 + RH 63 + face 1404)."
            )
        if arr.shape[0] < 2:
            raise CacheShapeError(
                f"Sequence {i} has only {arr.shape[0]} frame(s); need >= 2 for deltas."
            )

    logger.info("Loaded %d sequences from %s", len(sequences), cache_path)
    logger.info("Class distribution: %s", dict(Counter(labels)))
    return [np.asarray(s, dtype=np.float64) for s in sequences], labels


# ---------------------------------------------------------------------------
# 3. Feature engineering  (ADR-001..004)
# ---------------------------------------------------------------------------

def _block_present(block: np.ndarray) -> bool:
    """MediaPipe misses are encoded as an exact all-zero block by the extractor."""
    return bool(np.any(block != 0.0))


def normalise_frame(raw_frame: np.ndarray, fallback):
    """
    Convert one raw 1629-D frame to a 156-D body-centred descriptor.

    Returns (feature_156, new_fallback) where new_fallback is the
    (origin_xy, scale) pair to reuse if the next frame has no pose.
    """
    pose = raw_frame[POSE_OFF:POSE_OFF + POSE_D].reshape(33, 3)
    lh = raw_frame[LH_OFF:LH_OFF + LH_D].reshape(21, 3)
    rh = raw_frame[RH_OFF:RH_OFF + RH_D].reshape(21, 3)
    # face block deliberately ignored -- ADR-001

    pose_present = _block_present(pose)
    lh_present = _block_present(lh)
    rh_present = _block_present(rh)

    # --- derive the body frame of reference (ADR-003) ---
    if pose_present:
        ls_xy = pose[L_SHOULDER, :2]
        rs_xy = pose[R_SHOULDER, :2]
        origin_xy = (ls_xy + rs_xy) / 2.0
        scale = float(np.linalg.norm(ls_xy - rs_xy))
        if not np.isfinite(scale) or scale < 1e-6:
            # Degenerate: shoulders coincide (profile view / bad detection).
            origin_xy, scale = fallback
        new_fallback = (origin_xy, scale)
    else:
        origin_xy, scale = fallback
        new_fallback = fallback

    def _norm(block: np.ndarray, present: bool) -> np.ndarray:
        # ADR-002: missing blocks stay at exact zero, which after centring
        # means "at body origin" -- a neutral value, not an outlier.
        if not present:
            return np.zeros_like(block)
        out = np.empty_like(block)
        out[:, 0] = (block[:, 0] - origin_xy[0]) / scale
        out[:, 1] = (block[:, 1] - origin_xy[1]) / scale
        out[:, 2] = block[:, 2] / scale
        return out

    pose_n = _norm(pose, pose_present)[UPPER_POSE_IDX]   # (9, 3)
    lh_n = _norm(lh, lh_present)                          # (21, 3)
    rh_n = _norm(rh, rh_present)                          # (21, 3)

    coords = np.concatenate([pose_n.ravel(), lh_n.ravel(), rh_n.ravel()])
    flags = np.array(
        [float(pose_present), float(lh_present), float(rh_present)], dtype=np.float64
    )
    feature = np.concatenate([coords, flags])

    assert feature.shape[0] == N_BASE_DIMS, (feature.shape, N_BASE_DIMS)
    return feature, new_fallback


def build_sequence_features(raw_sequence: np.ndarray) -> np.ndarray:
    """Raw (T, 1629) -> engineered (T, 309) with velocity channels."""
    fallback = (np.array([0.5, 0.5]), 1.0)  # image centre, unit scale
    base = np.empty((raw_sequence.shape[0], N_BASE_DIMS), dtype=np.float64)

    for t in range(raw_sequence.shape[0]):
        base[t], fallback = normalise_frame(raw_sequence[t], fallback)

    # ADR-004: first-order differences over the coordinate channels only.
    coords = base[:, :N_COORD_DIMS]
    delta = np.zeros_like(coords)
    delta[1:] = np.diff(coords, axis=0)

    full = np.concatenate([base, delta], axis=1)
    assert full.shape[1] == N_FULL_DIMS, (full.shape, N_FULL_DIMS)

    if not np.all(np.isfinite(full)):
        n_bad = int((~np.isfinite(full)).sum())
        logger.warning("Sequence contained %d non-finite values; zeroing them.", n_bad)
        full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)

    return full


# ---------------------------------------------------------------------------
# 4. HMM construction  (ADR-005)
# ---------------------------------------------------------------------------

def left_to_right_transmat(n_states: int, self_loop: float = 0.7) -> np.ndarray:
    """Bakis topology: stay in state i, or advance to i+1. Absorbing last state."""
    if n_states < 1:
        raise ValueError("n_states must be >= 1")
    t = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states - 1):
        t[i, i] = self_loop
        t[i, i + 1] = 1.0 - self_loop
    t[n_states - 1, n_states - 1] = 1.0
    return t


def build_hmm(n_states: int, n_iter: int, train_transitions: bool) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",   # full covariances are singular at this data scale
        min_covar=MIN_COVAR,      # variance floor -> prevents +inf densities
        n_iter=n_iter,
        tol=1e-4,
        init_params="mc",                              # init means/covars by kmeans
        params="mct" if train_transitions else "mc",   # ADR-005
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.startprob_ = np.eye(n_states)[0]             # always begin in state 0
    model.transmat_ = left_to_right_transmat(n_states)
    return model


def sanitise_hmm(model: GaussianHMM) -> GaussianHMM:
    """
    Repair a fitted model so score() can never return -inf/NaN for structural
    reasons. Only relevant when TRAIN_TRANSITIONS is True.
    """
    n = model.n_components

    start = np.asarray(model.startprob_, dtype=np.float64)
    if not np.all(np.isfinite(start)) or start.sum() <= 0:
        start = np.full(n, 1.0 / n)
    start = np.clip(start, 1e-8, None)
    model.startprob_ = start / start.sum()

    trans = np.asarray(model.transmat_, dtype=np.float64).copy()
    trans[~np.isfinite(trans)] = 0.0
    for i in range(n):
        if trans[i].sum() <= 0:
            logger.warning("Repairing zero-sum transmat row %d (uniform fallback).", i)
            trans[i] = 1.0 / n
    trans = np.clip(trans, 1e-8, None)
    model.transmat_ = trans / trans.sum(axis=1, keepdims=True)

    covars = model._covars_ if hasattr(model, "_covars_") else None
    if covars is not None:
        model._covars_ = np.clip(np.asarray(covars), MIN_COVAR, None)

    return model


def fit_class_models(train_seqs, train_labels, class_ids, n_states, n_iter):
    """Train one HMM per class. Returns {class_id: fitted model}."""
    models = {}
    for cid in class_ids:
        member = [s for s, y in zip(train_seqs, train_labels) if y == cid]
        if len(member) < 2:
            logger.warning("Class %d skipped: only %d training sequence(s).", cid, len(member))
            continue

        X = np.vstack(member)
        lengths = [len(s) for s in member]
        model = build_hmm(n_states, n_iter, TRAIN_TRANSITIONS)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X, lengths)
            except Exception as exc:  # noqa: BLE001 - we want the class name + trace
                logger.exception("[HMM-2001] Fit failed for class %d: %s", cid, exc)
                continue

        model = sanitise_hmm(model)
        converged = bool(getattr(model.monitor_, "converged", False))
        if not converged:
            logger.warning(
                "Class %d did not converge in %d iterations (delta=%.4g).",
                cid, n_iter, getattr(model.monitor_, "history", [float("nan")])[-1]
                if getattr(model.monitor_, "history", None) else float("nan"),
            )
        models[cid] = model
    return models


# ---------------------------------------------------------------------------
# 5. Prediction  (the direct bug fix)
# ---------------------------------------------------------------------------

def score_matrix(sequences, models, class_ids) -> np.ndarray:
    """
    (n_sequences, n_classes) of LENGTH-NORMALISED log-likelihoods.
    Non-finite entries are preserved as -inf so callers can detect collapse.
    """
    S = np.full((len(sequences), len(class_ids)), -np.inf, dtype=np.float64)
    for i, seq in enumerate(sequences):
        T = max(len(seq), 1)
        for j, cid in enumerate(class_ids):
            model = models.get(cid)
            if model is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ll = model.score(seq)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[HMM-3001] score() failed (seq %d, class %d): %s", i, cid, exc)
                continue
            if np.isfinite(ll):
                S[i, j] = ll / T
    return S


def predict_from_scores(S: np.ndarray, class_ids) -> np.ndarray:
    """
    argmax with an explicit abstain. Returns -1 where every class scored
    non-finite.

    THIS IS THE BUG FIX. The original code used
        max(scores, key=scores.get)
    on a dict whose values were all -inf, so Python returned the FIRST INSERTED
    KEY -- class_id 0 -- which LabelEncoder had assigned to 'before'. That is
    why 'before' had recall 1.00 and every other class had recall 0.00.
    """
    ids = np.asarray(class_ids)
    out = np.full(S.shape[0], -1, dtype=int)
    usable = np.isfinite(S).any(axis=1)
    if usable.any():
        out[usable] = ids[np.argmax(np.where(np.isfinite(S), S, -np.inf), axis=1)[usable]]
    return out


# ---------------------------------------------------------------------------
# 6. DTW + 1-NN reference baseline
# ---------------------------------------------------------------------------

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard DTW with Euclidean local cost. O(len(a)*len(b))."""
    n, m = len(a), len(b)
    # Squared-euclidean cost matrix via the expansion |x-y|^2 = |x|^2 - 2xy + |y|^2
    cost = np.sqrt(
        np.maximum(
            (a ** 2).sum(1)[:, None] - 2.0 * (a @ b.T) + (b ** 2).sum(1)[None, :],
            0.0,
        )
    )
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        ci = cost[i - 1]
        for j in range(1, m + 1):
            acc[i, j] = ci[j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


def dtw_1nn_predict(train_seqs, train_labels, test_seqs) -> np.ndarray:
    preds = np.empty(len(test_seqs), dtype=int)
    for i, q in enumerate(test_seqs):
        dists = [dtw_distance(q, r) for r in train_seqs]
        preds[i] = train_labels[int(np.argmin(dists))]
    return preds


# ---------------------------------------------------------------------------
# 7. Diagnosis of the ORIGINAL failure
# ---------------------------------------------------------------------------

def diagnose_original(sequences, labels, n_states=5, n_pca=30):
    """Reproduce the original configuration and print its score matrix."""
    logger.info("=" * 74)
    logger.info("DIAGNOSIS: reproducing the original 1629-D / ergodic-5-state setup")
    logger.info("=" * 74)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_ids = list(range(len(le.classes_)))

    skf = StratifiedKFold(n_splits=DEFAULT_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(iter(skf.split(np.zeros(len(y)), y)))

    scaler = StandardScaler().fit(np.vstack([sequences[i] for i in tr_idx]))
    pca = PCA(n_components=n_pca, random_state=RANDOM_STATE).fit(
        scaler.transform(np.vstack([sequences[i] for i in tr_idx]))
    )
    tf = lambda s: pca.transform(scaler.transform(s)).astype(np.float64)  # noqa: E731

    tr_seqs = [tf(sequences[i]) for i in tr_idx]
    te_seqs = [tf(sequences[i]) for i in te_idx]

    models = {}
    for cid in class_ids:
        member = [s for s, yy in zip(tr_seqs, y[tr_idx]) if yy == cid]
        if len(member) < 2:
            continue
        m = GaussianHMM(
            n_components=n_states, covariance_type="diag",
            n_iter=100, random_state=RANDOM_STATE,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m.fit(np.vstack(member), [len(s) for s in member])
                models[cid] = m
            except Exception as exc:  # noqa: BLE001
                logger.warning("original-config fit failed for class %d: %s", cid, exc)

    logger.info("Raw (UN-normalised) log-likelihood matrix, original config:")
    header = "  true          " + "".join(f"{le.classes_[c]:>14s}" for c in class_ids)
    logger.info(header)

    n_nonfinite_rows = 0
    for seq, true_id in zip(te_seqs, y[te_idx]):
        row = []
        for cid in class_ids:
            m = models.get(cid)
            if m is None:
                row.append(-np.inf)
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    row.append(m.score(seq))
            except Exception:  # noqa: BLE001
                row.append(-np.inf)
        row = np.array(row)
        if not np.isfinite(row).any():
            n_nonfinite_rows += 1
        logger.info(
            "  %-14s" % le.classes_[true_id]
            + "".join(f"{v:14.2f}" if np.isfinite(v) else f"{'-inf':>14s}" for v in row)
        )

    logger.info("-" * 74)
    logger.info("Rows where EVERY class scored non-finite: %d / %d",
                n_nonfinite_rows, len(te_seqs))
    logger.info(
        "Any such row makes `max(scores, key=scores.get)` return the first "
        "inserted key = class 0 = %r.", le.classes_[0]
    )
    logger.info("=" * 74)


# ---------------------------------------------------------------------------
# 8. Cross-validated evaluation
# ---------------------------------------------------------------------------

def run_cv(features, labels, n_pca, n_states, n_folds, n_iter, with_dtw=True):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_ids = list(range(len(le.classes_)))
    n_samples = len(features)

    counts = Counter(y)
    min_count = min(counts.values())
    if min_count < n_folds:
        n_folds = max(2, min_count)
        logger.warning("Reducing n_folds to %d (smallest class has %d samples).",
                       n_folds, min_count)

    oof_hmm = np.full(n_samples, -1, dtype=int)
    oof_dtw = np.full(n_samples, -1, dtype=int)
    fold_acc, fold_abstain = [], []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(n_samples), y), start=1):
        train_frames = np.vstack([features[i] for i in tr_idx])

        scaler = StandardScaler().fit(train_frames)
        max_components = min(n_pca, train_frames.shape[0], train_frames.shape[1])
        pca = PCA(n_components=max_components, random_state=RANDOM_STATE)
        pca.fit(scaler.transform(train_frames))

        def tf(seq):
            return pca.transform(scaler.transform(seq)).astype(np.float64)

        tr_seqs = [tf(features[i]) for i in tr_idx]
        te_seqs = [tf(features[i]) for i in te_idx]

        models = fit_class_models(tr_seqs, y[tr_idx], class_ids, n_states, n_iter)
        S = score_matrix(te_seqs, models, class_ids)
        preds = predict_from_scores(S, class_ids)
        oof_hmm[te_idx] = preds

        n_abstain = int((preds == -1).sum())
        decided = preds != -1
        acc = accuracy_score(y[te_idx][decided], preds[decided]) if decided.any() else 0.0
        fold_acc.append(acc)
        fold_abstain.append(n_abstain)

        logger.info(
            "fold %d/%d | train=%d test=%d | models=%d | evr=%.3f | acc=%.4f | abstain=%d",
            fold, n_folds, len(tr_idx), len(te_idx), len(models),
            float(pca.explained_variance_ratio_.sum()), acc, n_abstain,
        )

        if with_dtw:
            oof_dtw[te_idx] = dtw_1nn_predict(tr_seqs, y[tr_idx], te_seqs)

    return {
        "label_encoder": le,
        "y_true": y,
        "oof_hmm": oof_hmm,
        "oof_dtw": oof_dtw if with_dtw else None,
        "fold_acc": np.array(fold_acc),
        "fold_abstain": np.array(fold_abstain),
        "n_folds": n_folds,
    }


def report(name, y_true, y_pred, class_names):
    decided = y_pred != -1
    n_abstain = int((~decided).sum())
    yt, yp = y_true[decided], y_pred[decided]

    acc = accuracy_score(yt, yp) if len(yt) else 0.0
    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, average="weighted", zero_division=0
    )
    n_predicted_classes = len(set(yp.tolist()))

    print()
    print("=" * 74)
    print(f"{name}   (n={len(y_true)}, abstained={n_abstain})")
    print("=" * 74)
    print(f"Accuracy            : {acc:.4f}")
    print(f"Weighted Precision  : {p:.4f}")
    print(f"Weighted Recall     : {r:.4f}")
    print(f"Weighted F1-score   : {f1:.4f}")
    print(f"Distinct classes predicted : {n_predicted_classes} / {len(class_names)}")
    if n_predicted_classes <= 1:
        print("  ** COLLAPSED: the model is a constant predictor. **")
    print()
    print(classification_report(
        yt, yp,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        zero_division=0,
    ))
    print("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(yt, yp, labels=list(range(len(class_names))))
    width = max(len(c) for c in class_names) + 2
    print(" " * width + "".join(f"{c[:8]:>9s}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"{c:<{width}}" + "".join(f"{v:9d}" for v in cm[i]))
    return {
        "accuracy": float(acc),
        "weighted_precision": float(p),
        "weighted_recall": float(r),
        "weighted_f1": float(f1),
        "n_abstain": n_abstain,
        "distinct_classes_predicted": n_predicted_classes,
    }


# ---------------------------------------------------------------------------
# 9. Final model on all data + persistence
# ---------------------------------------------------------------------------

def fit_final_model(features, labels, n_pca, n_states, n_iter):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_ids = list(range(len(le.classes_)))

    all_frames = np.vstack(features)
    scaler = StandardScaler().fit(all_frames)
    n_comp = min(n_pca, all_frames.shape[0], all_frames.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(scaler.transform(all_frames))

    seqs = [pca.transform(scaler.transform(s)).astype(np.float64) for s in features]
    models = fit_class_models(seqs, y, class_ids, n_states, n_iter)

    return {
        "class_models": models,
        "scaler": scaler,
        "pca": pca,
        "label_encoder": le,
        "n_hmm_states": n_states,
        "n_pca_components": n_comp,
        "feature_spec": {
            "upper_pose_idx": UPPER_POSE_IDX,
            "n_base_dims": N_BASE_DIMS,
            "n_full_dims": N_FULL_DIMS,
            "face_included": False,
            "body_centred": True,
            "shoulder_scaled": True,
            "delta_features": True,
        },
        "train_transitions": TRAIN_TRANSITIONS,
    }


# ---------------------------------------------------------------------------
# 10. Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Corrected WLASL HMM baseline")
    ap.add_argument("--pca", type=int, default=DEFAULT_PCA_COMPONENTS)
    ap.add_argument("--states", type=int, default=DEFAULT_HMM_STATES)
    ap.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--iters", type=int, default=DEFAULT_N_ITER)
    ap.add_argument("--no-dtw", action="store_true", help="skip the DTW 1-NN baseline")
    ap.add_argument("--diagnose-only", action="store_true",
                    help="only reproduce and explain the original failure")
    args = ap.parse_args()

    log_file = configure_logging()
    logger.info("Log file: %s", log_file)

    sequences, labels = load_cached_sequences()

    lengths = [len(s) for s in sequences]
    logger.info("Sequence length: min=%d median=%d max=%d",
                min(lengths), int(np.median(lengths)), max(lengths))

    if args.diagnose_only:
        diagnose_original(sequences, labels)
        return 0

    diagnose_original(sequences, labels)

    logger.info("Building corrected features (1629 -> %d dims per frame)...", N_FULL_DIMS)
    features = [build_sequence_features(s) for s in sequences]

    # Report how often MediaPipe actually saw each hand -- key data-quality metric.
    all_flags = np.vstack([f[:, N_COORD_DIMS:N_COORD_DIMS + N_FLAG_DIMS] for f in features])
    logger.info(
        "Landmark detection rate across all frames: pose=%.1f%%  left-hand=%.1f%%  right-hand=%.1f%%",
        100 * all_flags[:, 0].mean(), 100 * all_flags[:, 1].mean(), 100 * all_flags[:, 2].mean(),
    )

    results = run_cv(
        features, labels,
        n_pca=args.pca, n_states=args.states,
        n_folds=args.folds, n_iter=args.iters,
        with_dtw=not args.no_dtw,
    )

    le = results["label_encoder"]
    class_names = list(le.classes_)
    y_true = results["y_true"]

    counts = Counter(y_true.tolist())
    majority_acc = max(counts.values()) / len(y_true)

    print()
    print("#" * 74)
    print(f"# Out-of-fold results over {len(y_true)} sequences, "
          f"{results['n_folds']}-fold stratified CV")
    print(f"# Majority-class baseline accuracy = {majority_acc:.4f}  "
          f"(this is what the old notebook scored)")
    print("#" * 74)

    hmm_metrics = report("HMM (corrected)", y_true, results["oof_hmm"], class_names)
    print(f"\nPer-fold accuracy: "
          f"{np.round(results['fold_acc'], 4).tolist()}")
    print(f"Mean +/- SD      : {results['fold_acc'].mean():.4f} "
          f"+/- {results['fold_acc'].std():.4f}")

    dtw_metrics = None
    if results["oof_dtw"] is not None:
        dtw_metrics = report("DTW 1-NN (reference)", y_true, results["oof_dtw"], class_names)

    print()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  majority baseline : {majority_acc:.4f}")
    print(f"  HMM (corrected)   : {hmm_metrics['accuracy']:.4f}")
    if dtw_metrics:
        print(f"  DTW 1-NN          : {dtw_metrics['accuracy']:.4f}")
    print(f"  old notebook      : 0.2353  (collapsed to a single class)")

    logger.info("Fitting final model on all %d sequences...", len(features))
    package = fit_final_model(features, labels, args.pca, args.states, args.iters)

    pkg_path = MODEL_DIR / "hmm_wlasl_fixed_package.joblib"
    joblib.dump(package, pkg_path)

    metadata = {
        "model": "GaussianHMM per class, left-to-right topology, fixed transitions",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "classes": class_names,
        "n_sequences": len(features),
        "frame_feature_dim": N_FULL_DIMS,
        "face_landmarks_used": False,
        "n_pca_components": package["n_pca_components"],
        "n_hmm_states": args.states,
        "cv_folds": results["n_folds"],
        "majority_baseline_accuracy": float(majority_acc),
        "hmm": hmm_metrics,
        "hmm_fold_accuracies": results["fold_acc"].tolist(),
        "hmm_fold_accuracy_mean": float(results["fold_acc"].mean()),
        "hmm_fold_accuracy_std": float(results["fold_acc"].std()),
        "dtw_1nn": dtw_metrics,
        "previous_notebook_accuracy": 0.2353,
    }
    meta_path = MODEL_DIR / "hmm_fixed_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Saved model package : %s", pkg_path)
    logger.info("Saved metadata      : %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
