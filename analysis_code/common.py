"""
Shared utilities for the KSL model comparison.
Real training on the cached MediaPipe landmark features (no MediaPipe re-run).

Landmark layout of the 1629-D per-frame vector (MediaPipe Holistic):
  pose      : indices    0 : 99    (33 landmarks x 3)
  left hand : indices   99 : 162   (21 landmarks x 3)
  right hand: indices  162 : 225   (21 landmarks x 3)
  face      : indices  225 : 1629  (468 landmarks x 3)
"""
import numpy as np
import joblib
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score, confusion_matrix)

SEED = 42
DATA = "/mnt/user-data/uploads/evanceaiproject/dataset/ksl_project_data/features"

POSE = slice(0, 99)
LH   = slice(99, 162)
RH   = slice(162, 225)
FACE = slice(225, 1629)

# MediaPipe pose landmark indices for shoulders
L_SHOULDER, R_SHOULDER = 11, 12


def load_sequences():
    d = joblib.load(f"{DATA}/wlasl_lstm_sequences_fixed_classes.joblib")
    X = np.asarray(d["X_sequences"], dtype=np.float32)   # (67,60,1629)
    y = np.asarray(d["y_labels"])
    classes = sorted(list(d["fixed_classes"]))
    return X, y, classes


def load_graph():
    d = joblib.load(f"{DATA}/wlasl_stgcn_graph_sequences_fixed_classes.joblib")
    Xg = np.asarray(d["X_graph"], dtype=np.float32)       # (67,60,75,3)
    y = np.asarray(d["y_labels"])
    classes = sorted(list(d["fixed_classes"]))
    return Xg, y, classes


def seq_lengths(X):
    """Number of non-padding frames per sample (padding frames are all-zero)."""
    nz = np.any(X != 0.0, axis=2)          # (N,T) True where frame has any signal
    lens = nz.sum(axis=1).astype(int)
    lens = np.clip(lens, 1, X.shape[1])
    return lens


def graph_lengths(Xg):
    nz = np.any(Xg.reshape(Xg.shape[0], Xg.shape[1], -1) != 0.0, axis=2)
    lens = nz.sum(axis=1).astype(int)
    return np.clip(lens, 1, Xg.shape[1])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, labels):
    """Return the full metric bundle for an isolated-sign recogniser.

    WER: for isolated single-gloss recognition each utterance is one word, so
    Word Error Rate reduces to the substitution rate = 1 - accuracy. Reported
    for parity with continuous-recognition literature.
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None,
                         zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_pred_classes = len(set(y_pred))
    return {
        "accuracy": float(acc),
        "weighted_precision": float(p),
        "weighted_recall": float(r),
        "weighted_f1": float(f),
        "wer": float(1.0 - acc),
        "per_class_f1": {lab: float(v) for lab, v in zip(labels, per_class)},
        "confusion_matrix": cm.tolist(),
        "distinct_classes_predicted": int(n_pred_classes),
        "collapsed": bool(n_pred_classes <= 1),
    }
