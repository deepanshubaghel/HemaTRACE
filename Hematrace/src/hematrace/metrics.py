"""Regression, classification, threshold selection, and calibration metrics."""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .config import cfg
from .optional_dependencies import spearmanr

def rmse(y, p):
    return float(math.sqrt(mean_squared_error(y, p)))


def sigmoid_np(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def logit_np(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def clean_prob(p):
    p = np.asarray(p, dtype=float)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(p, 0.0, 1.0)


def specificity_score(y, yhat):
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp + 1e-8))


def ece_score(y, prob, n_bins=10):
    y = np.asarray(y).astype(int)
    prob = clean_prob(prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (prob >= lo) & (prob <= hi)
        else:
            m = (prob >= lo) & (prob < hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(prob[m].mean() - y[m].mean())
    return float(ece)


def regression_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    out = {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": rmse(y, p),
        "R2": float(r2_score(y, p)),
        "N": int(len(y)),
        "MAE_Hb_lt_8": float(mean_absolute_error(y[y < 8], p[y < 8])) if (y < 8).sum() > 0 else np.nan,
        "MAE_Hb_lt_7": float(mean_absolute_error(y[y < 7], p[y < 7])) if (y < 7).sum() > 0 else np.nan,
    }
    if spearmanr is not None:
        try:
            out["Spearman"] = float(spearmanr(y, p).correlation)
        except Exception:
            out["Spearman"] = np.nan
    else:
        out["Spearman"] = np.nan
    return out


def classification_metrics(y, prob, threshold=0.5):
    y = np.asarray(y).astype(int)
    prob = clean_prob(prob)
    pred = (prob >= threshold).astype(int)
    out = {
        "Accuracy": float(accuracy_score(y, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, pred)),
        "Macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Sensitivity": float(recall_score(y, pred, zero_division=0)),
        "Specificity": specificity_score(y, pred),
        "Brier": float(brier_score_loss(y, prob)),
        "ECE": ece_score(y, prob),
        "N": int(len(y)),
        "Positive_Rate": float(y.mean()),
        "threshold": float(threshold),
    }
    try:
        out["AUROC"] = float(roc_auc_score(y, prob))
    except Exception:
        out["AUROC"] = np.nan
    try:
        out["AUPRC"] = float(average_precision_score(y, prob))
    except Exception:
        out["AUPRC"] = np.nan
    return out


def tune_threshold(y, prob):
    y = np.asarray(y).astype(int)
    prob = clean_prob(prob)
    best_thr, best_score = 0.5, -1.0
    for thr in np.arange(cfg.THRESHOLD_GRID_MIN, cfg.THRESHOLD_GRID_MAX + 1e-9, cfg.THRESHOLD_GRID_STEP):
        score = f1_score(y, (prob >= thr).astype(int), average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, float(best_score)


def calibration_table(y, prob, n_bins=10):
    y = np.asarray(y).astype(int)
    prob = clean_prob(prob)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (prob >= lo) & (prob <= hi)
        else:
            m = (prob >= lo) & (prob < hi)
        rows.append({
            "bin": i,
            "prob_low": lo,
            "prob_high": hi,
            "n": int(m.sum()),
            "mean_predicted_probability": float(prob[m].mean()) if m.sum() else np.nan,
            "observed_rate": float(y[m].mean()) if m.sum() else np.nan,
        })
    return pd.DataFrame(rows)
