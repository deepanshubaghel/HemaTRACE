"""Full DL comparator suite and late manuscript-ready overrides."""

import copy
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
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
from torch.utils.data import DataLoader, Dataset

from ..config import (
    DATA_DIR,
    FIGURE_PDF_DIR,
    FIGURE_PNG_DIR,
    LOGS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    RUN_DIR,
    TABLES_DIR,
    cfg,
)
from ..utils import save_current_fig_both, save_df, save_json

# ============================================================
# 0. SAFE FALLBACK HELPERS
# These are used only if your main script does not already define them.
# ============================================================

def _dl_json_converter(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        if not np.isfinite(o):
            return None
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def _dl_save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, default=_dl_json_converter)
    print(f"[SAVED] {path}")


def _dl_save_df(df: pd.DataFrame, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVED] {path} shape={df.shape}")


def _dl_clean_prob(p):
    p = np.asarray(p, dtype=float)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(p, 0.0, 1.0)


def _dl_sigmoid_np(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _dl_rmse(y, p):
    return float(math.sqrt(mean_squared_error(y, p)))


def _dl_specificity(y, yhat):
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp + 1e-8))


def _dl_ece(y, prob, n_bins=10):
    y = np.asarray(y).astype(int)
    prob = _dl_clean_prob(prob)
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


def _dl_regression_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": _dl_rmse(y, p),
        "R2": float(r2_score(y, p)),
        "N": int(len(y)),
    }


def _dl_classification_metrics(y, prob, threshold=0.5):
    y = np.asarray(y).astype(int)
    prob = _dl_clean_prob(prob)
    pred = (prob >= threshold).astype(int)
    out = {
        "Accuracy": float(accuracy_score(y, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, pred)),
        "Macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Sensitivity": float(recall_score(y, pred, zero_division=0)),
        "Specificity": _dl_specificity(y, pred),
        "Brier": float(brier_score_loss(y, prob)),
        "ECE": _dl_ece(y, prob),
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


def _dl_tune_threshold(y, prob, grid_min=0.05, grid_max=0.95, step=0.01):
    y = np.asarray(y).astype(int)
    prob = _dl_clean_prob(prob)
    best_thr, best_score = 0.5, -1.0
    for thr in np.arange(grid_min, grid_max + 1e-9, step):
        score = f1_score(y, (prob >= thr).astype(int), average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, float(best_score)


def _dl_get_global(name, default=None):
    return globals().get(name, default)


def _dl_save_current_fig_both(stem: str):
    png_dir = Path(_dl_get_global("FIGURE_PNG_DIR", Path("figures/png")))
    pdf_dir = Path(_dl_get_global("FIGURE_PDF_DIR", Path("figures/pdf")))
    dpi = int(getattr(_dl_get_global("cfg", object()), "DPI", 300))
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{stem}.png"
    pdf_path = pdf_dir / f"{stem}.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")
    return png_path, pdf_path


# ============================================================
# 1. CONFIG FOR DL COMPARISON
# ============================================================

@dataclass
class DLCompareConfig:
    max_epochs: int = 60
    patience: int = 10
    batch_size: int = 512
    num_workers: int = 2
    lr: float = 8e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    dropout: float = 0.25
    hidden_dim: int = 128
    static_hidden_dim: int = 96
    static_embed_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    tcn_levels: int = 3
    loss_reg_weight: float = 1.0
    loss_severe_weight: float = 0.7
    loss_onset_weight: float = 0.5
    score_mae_weight: float = 0.60
    score_onset_weight: float = 0.30
    use_amp: bool = True
    seed: int = 42
    threshold_grid_min: float = 0.05
    threshold_grid_max: float = 0.95
    threshold_grid_step: float = 0.01


def _make_dl_cfg(overrides: Optional[dict] = None) -> DLCompareConfig:
    main_cfg = _dl_get_global("cfg", None)
    c = DLCompareConfig()
    if main_cfg is not None:
        c.max_epochs = int(getattr(main_cfg, "DL_COMPARE_MAX_EPOCHS", getattr(main_cfg, "MAX_EPOCHS", c.max_epochs)))
        c.patience = int(getattr(main_cfg, "DL_COMPARE_PATIENCE", getattr(main_cfg, "PATIENCE", c.patience)))
        c.batch_size = int(getattr(main_cfg, "DL_COMPARE_BATCH_SIZE", getattr(main_cfg, "BATCH_SIZE", c.batch_size)))
        c.num_workers = int(getattr(main_cfg, "NUM_WORKERS", c.num_workers))
        c.lr = float(getattr(main_cfg, "DL_COMPARE_LR", getattr(main_cfg, "LR", c.lr)))
        c.weight_decay = float(getattr(main_cfg, "DL_COMPARE_WEIGHT_DECAY", getattr(main_cfg, "WEIGHT_DECAY", c.weight_decay)))
        c.grad_clip = float(getattr(main_cfg, "GRAD_CLIP", c.grad_clip))
        c.dropout = float(getattr(main_cfg, "DROPOUT", c.dropout))
        c.hidden_dim = int(getattr(main_cfg, "DL_COMPARE_HIDDEN_DIM", getattr(main_cfg, "GRU_HIDDEN_DIM", c.hidden_dim)))
        c.use_amp = bool(getattr(main_cfg, "USE_AMP", c.use_amp))
        c.seed = int(getattr(main_cfg, "RANDOM_STATE", c.seed))
        c.threshold_grid_min = float(getattr(main_cfg, "THRESHOLD_GRID_MIN", c.threshold_grid_min))
        c.threshold_grid_max = float(getattr(main_cfg, "THRESHOLD_GRID_MAX", c.threshold_grid_max))
        c.threshold_grid_step = float(getattr(main_cfg, "THRESHOLD_GRID_STEP", c.threshold_grid_step))
    if overrides:
        for k, v in overrides.items():
            if hasattr(c, k):
                setattr(c, k, v)
    return c


def _dl_seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. DATA LOADING AND NORMALIZATION
# ============================================================

def _dl_load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _dl_compute_norm(train):
    x_value, x_mask, x_count = train["x_value"], train["x_mask"], train["x_count"]

    obs = x_mask.sum(axis=(0, 1))
    mean = (x_value * x_mask).sum(axis=(0, 1)) / np.maximum(obs, 1)
    var = (((x_value - mean.reshape(1, 1, -1)) ** 2) * x_mask).sum(axis=(0, 1)) / np.maximum(obs, 1)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0

    delta = train["x_delta"].astype(np.float32)
    delta_mean = delta.mean(axis=(0, 1))
    delta_std = delta.std(axis=(0, 1))
    delta_std[delta_std < 1e-6] = 1.0

    cnt_log = np.log1p(x_count)
    cnt_mean = cnt_log.mean(axis=(0, 1))
    cnt_std = cnt_log.std(axis=(0, 1))
    cnt_std[cnt_std < 1e-6] = 1.0

    static_value = train["static_value"].astype(np.float32)
    static_mask = train["static_mask"].astype(np.float32)
    smean = (static_value * static_mask).sum(axis=0) / np.maximum(static_mask.sum(axis=0), 1)
    svar = (((static_value - smean.reshape(1, -1)) ** 2) * static_mask).sum(axis=0) / np.maximum(static_mask.sum(axis=0), 1)
    sstd = np.sqrt(svar)
    sstd[sstd < 1e-6] = 1.0

    delta_y = train["y_future_min_hb"].astype(np.float32) - train["early_last_hb"].astype(np.float32)
    target_delta_mean = float(np.nanmean(delta_y))
    target_delta_std = float(np.nanstd(delta_y))
    if not np.isfinite(target_delta_std) or target_delta_std < 1e-6:
        target_delta_std = 1.0

    return {
        "value_mean": mean.astype(np.float32),
        "value_std": std.astype(np.float32),
        "delta_mean": delta_mean.astype(np.float32),
        "delta_std": delta_std.astype(np.float32),
        "count_mean": cnt_mean.astype(np.float32),
        "count_std": cnt_std.astype(np.float32),
        "static_mean": smean.astype(np.float32),
        "static_std": sstd.astype(np.float32),
        "target_delta_mean": target_delta_mean,
        "target_delta_std": target_delta_std,
    }


def _dl_apply_norm(data, norm):
    d = dict(data)

    d["x_value_norm"] = (
        (d["x_value"].astype(np.float32) - norm["value_mean"].reshape(1, 1, -1))
        / norm["value_std"].reshape(1, 1, -1)
    )
    d["x_value_norm"] = np.where(d["x_mask"] > 0, d["x_value_norm"], 0.0).astype(np.float32)

    d["x_delta_norm"] = (
        (d["x_delta"].astype(np.float32) - norm["delta_mean"].reshape(1, 1, -1))
        / norm["delta_std"].reshape(1, 1, -1)
    ).astype(np.float32)

    d["x_count_norm"] = (
        (np.log1p(d["x_count"].astype(np.float32)) - norm["count_mean"].reshape(1, 1, -1))
        / norm["count_std"].reshape(1, 1, -1)
    ).astype(np.float32)

    d["static_norm"] = (
        (d["static_value"].astype(np.float32) - norm["static_mean"].reshape(1, -1))
        / norm["static_std"].reshape(1, -1)
    )
    d["static_norm"] = np.where(d["static_mask"] > 0, d["static_norm"], 0.0).astype(np.float32)

    delta_y = d["y_future_min_hb"].astype(np.float32) - d["early_last_hb"].astype(np.float32)
    d["target_delta_z"] = (
        (delta_y - float(norm["target_delta_mean"]))
        / float(norm["target_delta_std"])
    ).astype(np.float32)

    for key in ["x_value_norm", "x_delta_norm", "x_count_norm", "static_norm", "target_delta_z"]:
        d[key] = np.nan_to_num(d[key], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return d


class DLTemporalDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], mode: str = "all"):
        self.data = data
        if mode == "onset_only":
            self.idx = np.where(data["new_onset_mask"] > 0)[0]
        else:
            self.idx = np.arange(len(data["y_future_min_hb"]))

    def __len__(self):
        return len(self.idx)

    def _get_int(self, key, i):
        if key in self.data:
            return int(self.data[key][i])
        return -1

    def __getitem__(self, j):
        i = int(self.idx[j])
        d = self.data

        return {
            "x_value": torch.tensor(d["x_value_norm"][i], dtype=torch.float32),
            "x_mask": torch.tensor(d["x_mask"][i], dtype=torch.float32),
            "x_delta": torch.tensor(d["x_delta_norm"][i], dtype=torch.float32),
            "x_count": torch.tensor(d["x_count_norm"][i], dtype=torch.float32),
            "static_value": torch.tensor(d["static_norm"][i], dtype=torch.float32),
            "static_mask": torch.tensor(d["static_mask"][i], dtype=torch.float32),

            "base_hb": torch.tensor(d["early_last_hb"][i], dtype=torch.float32),
            "target_delta_z": torch.tensor(d["target_delta_z"][i], dtype=torch.float32),

            "y_reg": torch.tensor(d["y_future_min_hb"][i], dtype=torch.float32),
            "y_sev": torch.tensor(d["y_future_severe_anemia"][i], dtype=torch.float32),
            "y_on": torch.tensor(d["y_new_onset_anemia"][i], dtype=torch.float32),
            "on_mask": torch.tensor(d["new_onset_mask"][i], dtype=torch.float32),

            "subject_id": torch.tensor(self._get_int("subject_id", i), dtype=torch.long),
            "hadm_id": torch.tensor(self._get_int("hadm_id", i), dtype=torch.long),
            "stay_id": torch.tensor(self._get_int("stay_id", i), dtype=torch.long),
        }


def _dl_make_loader(data, c: DLCompareConfig, shuffle=False):
    return DataLoader(
        DLTemporalDataset(data),
        batch_size=c.batch_size,
        shuffle=shuffle,
        num_workers=c.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _dl_move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ============================================================
# 3. MODEL BUILDING BLOCKS
# ============================================================

class DLStaticEncoder(nn.Module):
    def __init__(self, static_dim, hidden_dim, embed_dim, dropout):
        super().__init__()
        inp = max(1, static_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(inp, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, static_value, static_mask):
        x = torch.cat([static_value, static_mask], dim=-1)
        return self.net(x)


class DLHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg = nn.Linear(hidden_dim, 1)
        self.sev = nn.Linear(hidden_dim, 1)
        self.on = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        h = self.shared(z)
        return {
            "delta_z": self.reg(h).squeeze(-1),
            "severe_logit": self.sev(h).squeeze(-1),
            "onset_logit": self.on(h).squeeze(-1),
        }


class DLBaseComparator(nn.Module):
    def __init__(self, temporal_dim, static_dim, c: DLCompareConfig):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.static_dim = static_dim
        self.c = c

    def make_temporal_input(self, x_value, x_mask, x_delta, x_count):
        # input shape: B x T x F
        # concatenated channel shape: B x T x 4F
        return torch.cat([x_value, x_mask, x_delta, x_count], dim=-1)


class StaticMLPComparator(DLBaseComparator):
    def __init__(self, temporal_dim, static_dim, c):
        super().__init__(temporal_dim, static_dim, c)
        self.static = DLStaticEncoder(static_dim, c.static_hidden_dim, c.static_embed_dim, c.dropout)
        self.head = DLHead(c.static_embed_dim, c.hidden_dim, c.dropout)

    def forward(self, x_value, x_mask, x_delta, x_count, static_value, static_mask):
        z = self.static(static_value, static_mask)
        out = self.head(z)
        out["attn"] = None
        return out


class TemporalFlattenMLPComparator(DLBaseComparator):
    def __init__(self, temporal_dim, static_dim, c):
        super().__init__(temporal_dim, static_dim, c)
        self.temporal_net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.GELU(),
        )
        self.static = DLStaticEncoder(static_dim, c.static_hidden_dim, c.static_embed_dim, c.dropout)
        self.head = DLHead(c.hidden_dim + c.static_embed_dim, c.hidden_dim, c.dropout)

    def forward(self, x_value, x_mask, x_delta, x_count, static_value, static_mask):
        xt = self.make_temporal_input(x_value, x_mask, x_delta, x_count)
        zt = self.temporal_net(xt)
        zs = self.static(static_value, static_mask)
        out = self.head(torch.cat([zt, zs], dim=-1))
        out["attn"] = None
        return out


class CNN1DComparator(DLBaseComparator):
    def __init__(self, temporal_dim, static_dim, c):
        super().__init__(temporal_dim, static_dim, c)
        in_ch = temporal_dim * 4
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, c.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Conv1d(c.hidden_dim, c.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(c.hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.static = DLStaticEncoder(static_dim, c.static_hidden_dim, c.static_embed_dim, c.dropout)
        self.head = DLHead(c.hidden_dim + c.static_embed_dim, c.hidden_dim, c.dropout)

    def forward(self, x_value, x_mask, x_delta, x_count, static_value, static_mask):
        xt = self.make_temporal_input(x_value, x_mask, x_delta, x_count).transpose(1, 2)
        zt = self.conv(xt).squeeze(-1)
        zs = self.static(static_value, static_mask)
        out = self.head(torch.cat([zt, zs], dim=-1))
        out["attn"] = None
        return out


class RNNComparator(DLBaseComparator):
    def __init__(self, temporal_dim, static_dim, c, rnn_type="GRU", bidirectional=False, attention=False):
        super().__init__(temporal_dim, static_dim, c)
        in_dim = temporal_dim * 4
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )
        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=c.hidden_dim,
            hidden_size=c.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        self.attention = attention
        rnn_out_dim = c.hidden_dim * (2 if bidirectional else 1)

        if attention:
            self.attn = nn.Sequential(
                nn.Linear(rnn_out_dim, c.hidden_dim),
                nn.Tanh(),
                nn.Linear(c.hidden_dim, 1),
            )
        else:
            self.attn = None

        self.static = DLStaticEncoder(static_dim, c.static_hidden_dim, c.static_embed_dim, c.dropout)
        self.head = DLHead(rnn_out_dim + c.static_embed_dim, c.hidden_dim, c.dropout)

    def forward(self, x_value, x_mask, x_delta, x_count, static_value, static_mask):
        xt = self.make_temporal_input(x_value, x_mask, x_delta, x_count)
        xt = self.input_proj(xt)
        h, _ = self.rnn(xt)

        if self.attention:
            score = self.attn(h).squeeze(-1)
            weight = torch.softmax(score, dim=1)
            zt = torch.sum(h * weight.unsqueeze(-1), dim=1)
            attn_out = weight
        else:
            zt = h[:, -1, :]
            attn_out = None

        zs = self.static(static_value, static_mask)
        out = self.head(torch.cat([zt, zs], dim=-1))
        out["attn"] = attn_out
        return out


class TransformerComparator(DLBaseComparator):
    def __init__(self, temporal_dim, static_dim, c, max_len=64):
        super().__init__(temporal_dim, static_dim, c)
        in_dim = temporal_dim * 4
        self.proj = nn.Linear(in_dim, c.hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, c.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=c.hidden_dim,
            nhead=c.transformer_heads,
            dim_feedforward=c.hidden_dim * 4,
            dropout=c.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=c.transformer_layers)
        self.pool = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.Tanh(),
            nn.Linear(c.hidden_dim, 1),
        )
        self.static = DLStaticEncoder(static_dim, c.static_hidden_dim, c.static_embed_dim, c.dropout)
        self.head = DLHead(c.hidden_dim + c.static_embed_dim, c.hidden_dim, c.dropout)

    def forward(self, x_value, x_mask, x_delta, x_count, static_value, static_mask):
        xt = self.make_temporal_input(x_value, x_mask, x_delta, x_count)
        T = xt.shape[1]
        h = self.proj(xt) + self.pos[:, :T, :]
        h = self.encoder(h)
        score = self.pool(h).squeeze(-1)
        weight = torch.softmax(score, dim=1)
        zt = torch.sum(h * weight.unsqueeze(-1), dim=1)
        zs = self.static(static_value, static_mask)
        out = self.head(torch.cat([zt, zs], dim=-1))
        out["attn"] = weight
        return out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.downsample(x)


class TCNComparator(DLBaseComparator):
    def __init__(self, temporal_dim, static_dim, c):
        super().__init__(temporal_dim, static_dim, c)
        in_ch = temporal_dim * 4
        layers = []
        ch = in_ch
        for i in range(c.tcn_levels):
            layers.append(TemporalBlock(ch, c.hidden_dim, kernel_size=3, dilation=2 ** i, dropout=c.dropout))
            ch = c.hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.static = DLStaticEncoder(static_dim, c.static_hidden_dim, c.static_embed_dim, c.dropout)
        self.head = DLHead(c.hidden_dim + c.static_embed_dim, c.hidden_dim, c.dropout)

    def forward(self, x_value, x_mask, x_delta, x_count, static_value, static_mask):
        xt = self.make_temporal_input(x_value, x_mask, x_delta, x_count).transpose(1, 2)
        h = self.tcn(xt)
        zt = self.pool(h).squeeze(-1)
        zs = self.static(static_value, static_mask)
        out = self.head(torch.cat([zt, zs], dim=-1))
        out["attn"] = None
        return out


def _dl_build_model(model_name: str, temporal_dim: int, static_dim: int, c: DLCompareConfig):
    if model_name == "DL_MLP_Static":
        return StaticMLPComparator(temporal_dim, static_dim, c)
    if model_name == "DL_MLP_TemporalFlatten":
        return TemporalFlattenMLPComparator(temporal_dim, static_dim, c)
    if model_name == "DL_CNN1D":
        return CNN1DComparator(temporal_dim, static_dim, c)
    if model_name == "DL_LSTM":
        return RNNComparator(temporal_dim, static_dim, c, rnn_type="LSTM", bidirectional=False, attention=False)
    if model_name == "DL_GRU":
        return RNNComparator(temporal_dim, static_dim, c, rnn_type="GRU", bidirectional=False, attention=False)
    if model_name == "DL_BiGRU":
        return RNNComparator(temporal_dim, static_dim, c, rnn_type="GRU", bidirectional=True, attention=False)
    if model_name == "DL_BiLSTM_Attention":
        return RNNComparator(temporal_dim, static_dim, c, rnn_type="LSTM", bidirectional=True, attention=True)
    if model_name == "DL_Transformer":
        return TransformerComparator(temporal_dim, static_dim, c)
    if model_name == "DL_TCN":
        return TCNComparator(temporal_dim, static_dim, c)

    raise ValueError(f"Unknown DL comparator model: {model_name}")


# ============================================================
# 4. TRAINING, PREDICTION, METRICS
# ============================================================

def _dl_multitask_loss(out, batch, c: DLCompareConfig):
    reg_loss = nn.functional.smooth_l1_loss(out["delta_z"], batch["target_delta_z"])

    sev_loss = nn.functional.binary_cross_entropy_with_logits(
        out["severe_logit"],
        batch["y_sev"],
    )

    mask = batch["on_mask"] > 0
    if mask.sum() > 0:
        on_loss = nn.functional.binary_cross_entropy_with_logits(
            out["onset_logit"][mask],
            batch["y_on"][mask],
        )
    else:
        on_loss = torch.tensor(0.0, device=out["delta_z"].device)

    return (
        c.loss_reg_weight * reg_loss
        + c.loss_severe_weight * sev_loss
        + c.loss_onset_weight * on_loss
    ), {
        "reg_loss": float(reg_loss.detach().cpu()),
        "sev_loss": float(sev_loss.detach().cpu()),
        "on_loss": float(on_loss.detach().cpu()),
    }


def _dl_validation_score(pred):
    reg = _dl_regression_metrics(pred["y_reg"], pred["pred_hb"])

    sev_auc = np.nan
    try:
        sev_auc = float(roc_auc_score(pred["y_sev"].astype(int), pred["sev_prob"]))
    except Exception:
        pass

    onset_auc = np.nan
    m = pred["on_mask"] > 0
    try:
        if m.sum() > 0:
            onset_auc = float(roc_auc_score(pred["y_on"][m].astype(int), pred["on_prob"][m]))
    except Exception:
        pass

    # Lower is better.
    # Main target: low MAE, with tie-breaker by onset AUROC and severe AUROC.
    score = reg["MAE"]
    if np.isfinite(onset_auc):
        score -= 0.05 * onset_auc
    if np.isfinite(sev_auc):
        score -= 0.03 * sev_auc
    return float(score)


def _dl_predict(model, loader, device, norm):
    model.eval()
    rows = {
        "subject_id": [],
        "hadm_id": [],
        "stay_id": [],
        "pred_hb": [],
        "sev_logit": [],
        "sev_prob": [],
        "on_logit": [],
        "on_prob": [],
        "y_reg": [],
        "y_sev": [],
        "y_on": [],
        "on_mask": [],
        "base_hb": [],
        "attn": [],
    }

    with torch.no_grad():
        for batch in loader:
            ids = {
                "subject_id": batch["subject_id"].detach().cpu().numpy(),
                "hadm_id": batch["hadm_id"].detach().cpu().numpy(),
                "stay_id": batch["stay_id"].detach().cpu().numpy(),
            }
            batch = _dl_move_batch(batch, device)

            out = model(
                batch["x_value"],
                batch["x_mask"],
                batch["x_delta"],
                batch["x_count"],
                batch["static_value"],
                batch["static_mask"],
            )

            pred_delta = (
                out["delta_z"].detach().cpu().numpy()
                * float(norm["target_delta_std"])
                + float(norm["target_delta_mean"])
            )
            base_hb = batch["base_hb"].detach().cpu().numpy()
            pred_hb = base_hb + pred_delta

            rows["subject_id"].append(ids["subject_id"])
            rows["hadm_id"].append(ids["hadm_id"])
            rows["stay_id"].append(ids["stay_id"])
            rows["pred_hb"].append(pred_hb)
            rows["sev_logit"].append(out["severe_logit"].detach().cpu().numpy())
            rows["sev_prob"].append(torch.sigmoid(out["severe_logit"]).detach().cpu().numpy())
            rows["on_logit"].append(out["onset_logit"].detach().cpu().numpy())
            rows["on_prob"].append(torch.sigmoid(out["onset_logit"]).detach().cpu().numpy())
            rows["y_reg"].append(batch["y_reg"].detach().cpu().numpy())
            rows["y_sev"].append(batch["y_sev"].detach().cpu().numpy())
            rows["y_on"].append(batch["y_on"].detach().cpu().numpy())
            rows["on_mask"].append(batch["on_mask"].detach().cpu().numpy())
            rows["base_hb"].append(base_hb)

            if out.get("attn") is not None:
                rows["attn"].append(out["attn"].detach().cpu().numpy())

    flat = {}
    for k, parts in rows.items():
        if k == "attn":
            flat[k] = np.concatenate(parts, axis=0) if len(parts) > 0 else None
        else:
            flat[k] = np.concatenate(parts, axis=0) if len(parts) > 0 else np.asarray([])
    return flat


def _dl_prediction_dataframe(pred, model_name, split_name):
    df = pd.DataFrame({
        "model": model_name,
        "split": split_name,
        "subject_id": pred["subject_id"],
        "hadm_id": pred["hadm_id"],
        "stay_id": pred["stay_id"],
        "true_future_min_hb": pred["y_reg"],
        "true_future_severe_anemia": pred["y_sev"].astype(int),
        "true_new_onset_anemia": pred["y_on"].astype(int),
        "new_onset_mask": pred["on_mask"].astype(int),
        "early_last_hb": pred["base_hb"],
        "dl_pred_future_min_hb": pred["pred_hb"],
        "dl_pred_severe_logit": pred["sev_logit"],
        "dl_pred_severe_prob": pred["sev_prob"],
        "dl_pred_onset_logit": pred["on_logit"],
        "dl_pred_onset_prob": pred["on_prob"],
    })
    return df


def _dl_evaluate_prediction(pred, model_name, label, split_name, thresholds):
    rows = []

    reg = _dl_regression_metrics(pred["y_reg"], pred["pred_hb"])
    row = {
        "model": model_name,
        "label": label,
        "split": split_name,
        "task": "future_min_hb_regression",
    }
    row.update(reg)
    rows.append(row)

    sev = _dl_classification_metrics(
        pred["y_sev"].astype(int),
        pred["sev_prob"],
        threshold=thresholds.get("severe", 0.5),
    )
    row = {
        "model": model_name,
        "label": label,
        "split": split_name,
        "task": "severe_anemia_classification",
    }
    row.update({f"Severe_{k}": v for k, v in sev.items()})
    rows.append(row)

    m = pred["on_mask"] > 0
    if m.sum() > 0:
        onset = _dl_classification_metrics(
            pred["y_on"][m].astype(int),
            pred["on_prob"][m],
            threshold=thresholds.get("onset", 0.5),
        )
        row = {
            "model": model_name,
            "label": label,
            "split": split_name,
            "task": "new_onset_anemia_classification",
        }
        row.update({f"Onset_{k}": v for k, v in onset.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def _dl_wide_metrics(metrics_long):
    base = metrics_long[["model", "label", "split"]].drop_duplicates().copy()
    out = base.copy()

    reg = metrics_long[metrics_long.task == "future_min_hb_regression"]
    reg_cols = ["model", "split", "MAE", "RMSE", "R2", "N"]
    if not reg.empty:
        out = out.merge(reg[reg_cols], on=["model", "split"], how="left")

    sev = metrics_long[metrics_long.task == "severe_anemia_classification"]
    sev_keep = [
        "model", "split",
        "Severe_AUROC", "Severe_AUPRC", "Severe_Accuracy", "Severe_Balanced_Accuracy",
        "Severe_Macro_F1", "Severe_Sensitivity", "Severe_Specificity",
        "Severe_Brier", "Severe_ECE", "Severe_threshold",
        "Severe_N", "Severe_Positive_Rate",
    ]
    sev_keep = [c for c in sev_keep if c in sev.columns]
    if not sev.empty:
        out = out.merge(sev[sev_keep], on=["model", "split"], how="left")

    onset = metrics_long[metrics_long.task == "new_onset_anemia_classification"]
    onset_keep = [
        "model", "split",
        "Onset_AUROC", "Onset_AUPRC", "Onset_Accuracy", "Onset_Balanced_Accuracy",
        "Onset_Macro_F1", "Onset_Sensitivity", "Onset_Specificity",
        "Onset_Brier", "Onset_ECE", "Onset_threshold",
        "Onset_N", "Onset_Positive_Rate",
    ]
    onset_keep = [c for c in onset_keep if c in onset.columns]
    if not onset.empty:
        out = out.merge(onset[onset_keep], on=["model", "split"], how="left")

    return out


def _dl_train_one_model(model_name, train_data, val_data, test_data, norm, c, out_root):
    _dl_seed_everything(c.seed)

    model_dir = Path(out_root) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temporal_dim = int(train_data["x_value"].shape[-1])
    static_dim = int(train_data["static_value"].shape[-1])

    model = _dl_build_model(model_name, temporal_dim, static_dim, c).to(device)

    train_loader = _dl_make_loader(train_data, c, shuffle=True)
    val_loader = _dl_make_loader(val_data, c, shuffle=False)
    test_loader = _dl_make_loader(test_data, c, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, c.patience // 3),
    )

    use_amp = bool(c.use_amp and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = np.inf
    best_state = None
    best_epoch = -1
    bad_epochs = 0
    history = []

    print(f"\n[DL COMPARISON] Training {model_name} on {device}")

    for epoch in range(1, c.max_epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            batch = _dl_move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(
                    batch["x_value"],
                    batch["x_mask"],
                    batch["x_delta"],
                    batch["x_count"],
                    batch["static_value"],
                    batch["static_mask"],
                )
                loss, loss_parts = _dl_multitask_loss(out, batch, c)

            scaler.scale(loss).backward()
            if c.grad_clip is not None and c.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu()))

        val_pred = _dl_predict(model, val_loader, device, norm)
        val_score = _dl_validation_score(val_pred)
        scheduler.step(val_score)

        val_reg = _dl_regression_metrics(val_pred["y_reg"], val_pred["pred_hb"])
        try:
            val_sev_auc = float(roc_auc_score(val_pred["y_sev"].astype(int), val_pred["sev_prob"]))
        except Exception:
            val_sev_auc = np.nan

        m = val_pred["on_mask"] > 0
        try:
            val_on_auc = float(roc_auc_score(val_pred["y_on"][m].astype(int), val_pred["on_prob"][m])) if m.sum() > 0 else np.nan
        except Exception:
            val_on_auc = np.nan

        hist_row = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if len(losses) else np.nan,
            "val_score": float(val_score),
            "val_MAE": val_reg["MAE"],
            "val_RMSE": val_reg["RMSE"],
            "val_R2": val_reg["R2"],
            "val_Severe_AUROC": val_sev_auc,
            "val_Onset_AUROC": val_on_auc,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(hist_row)

        print(
            f"[{model_name}] epoch={epoch:03d} "
            f"loss={hist_row['train_loss']:.4f} "
            f"val_score={val_score:.4f} "
            f"val_MAE={val_reg['MAE']:.4f} "
            f"sev_auc={val_sev_auc:.4f} "
            f"on_auc={val_on_auc:.4f}"
        )

        if val_score < best_score - 1e-6:
            best_score = float(val_score)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            torch.save(best_state, model_dir / "best_model.pt")
        else:
            bad_epochs += 1

        if bad_epochs >= c.patience:
            print(f"[{model_name}] early stopping at epoch={epoch}, best_epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred = _dl_predict(model, val_loader, device, norm)
    test_pred = _dl_predict(model, test_loader, device, norm)

    sev_thr, sev_thr_score = _dl_tune_threshold(
        val_pred["y_sev"].astype(int),
        val_pred["sev_prob"],
        c.threshold_grid_min,
        c.threshold_grid_max,
        c.threshold_grid_step,
    )

    m = val_pred["on_mask"] > 0
    if m.sum() > 0:
        on_thr, on_thr_score = _dl_tune_threshold(
            val_pred["y_on"][m].astype(int),
            val_pred["on_prob"][m],
            c.threshold_grid_min,
            c.threshold_grid_max,
            c.threshold_grid_step,
        )
    else:
        on_thr, on_thr_score = 0.5, np.nan

    thresholds = {
        "severe": float(sev_thr),
        "severe_val_macro_f1": float(sev_thr_score),
        "onset": float(on_thr),
        "onset_val_macro_f1": float(on_thr_score) if np.isfinite(on_thr_score) else None,
    }

    val_df = _dl_prediction_dataframe(val_pred, model_name, "validation")
    test_df = _dl_prediction_dataframe(test_pred, model_name, "test")

    save_df_func = _dl_get_global("save_df", _dl_save_df)
    save_json_func = _dl_get_global("save_json", _dl_save_json)

    data_dir = Path(_dl_get_global("DATA_DIR", Path("data")))
    save_df_func(val_df, data_dir / f"dl_predictions_validation_{model_name}.csv")
    save_df_func(test_df, data_dir / f"dl_predictions_test_{model_name}.csv")

    hist_df = pd.DataFrame(history)
    save_df_func(hist_df, model_dir / f"history_{model_name}.csv")

    val_metrics = _dl_evaluate_prediction(val_pred, model_name, model_name.replace("_", " "), "validation", thresholds)
    test_metrics = _dl_evaluate_prediction(test_pred, model_name, model_name.replace("_", " "), "test", thresholds)
    metrics = pd.concat([val_metrics, test_metrics], ignore_index=True)

    summary = {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "thresholds": thresholds,
        "model_dir": str(model_dir),
        "n_train": int(len(train_data["y_future_min_hb"])),
        "n_validation": int(len(val_data["y_future_min_hb"])),
        "n_test": int(len(test_data["y_future_min_hb"])),
        "temporal_dim": temporal_dim,
        "static_dim": static_dim,
    }
    save_json_func(summary, model_dir / "summary.json")

    return {
        "model": model,
        "model_name": model_name,
        "history": hist_df,
        "thresholds": thresholds,
        "val_predictions": val_df,
        "test_predictions": test_df,
        "metrics_long": metrics,
        "summary": summary,
    }


# ============================================================
# 5. HEMATRACE METRICS MERGE
# ============================================================

def _dl_extract_hematrace_test_metrics(hema_metrics_path):
    path = Path(hema_metrics_path) if hema_metrics_path is not None else None
    if path is None or not path.exists():
        return pd.DataFrame()

    m = pd.read_csv(path)
    if "split" in m.columns:
        m = m[m["split"].astype(str).str.lower().eq("test")].copy()

    # If already wide, keep relevant fields.
    if "task" not in m.columns:
        keep = [
            "model", "label", "MAE", "RMSE", "R2",
            "Severe_AUROC", "Severe_AUPRC", "Severe_ECE",
            "Onset_AUROC", "Onset_AUPRC", "Onset_ECE",
        ]
        keep = [c for c in keep if c in m.columns]
        out = m[keep].copy()
        out["comparison_group"] = "HemaTRACE"
        return out

    rows = []
    for model, g in m.groupby("model"):
        label = g["label"].iloc[0] if "label" in g.columns else model
        row = {"model": model, "label": label}

        reg = g[g.task.astype(str).str.contains("regression", case=False, na=False)]
        if not reg.empty:
            for col in ["MAE", "RMSE", "R2"]:
                if col in reg.columns:
                    row[col] = reg[col].iloc[0]

        sev = g[g.task.astype(str).str.contains("severe", case=False, na=False)]
        if not sev.empty:
            for col in ["AUROC", "AUPRC", "ECE", "Brier", "threshold"]:
                if col in sev.columns:
                    row[f"Severe_{col}"] = sev[col].iloc[0]
                elif f"Severe_{col}" in sev.columns:
                    row[f"Severe_{col}"] = sev[f"Severe_{col}"].iloc[0]

        onset = g[g.task.astype(str).str.contains("onset", case=False, na=False)]
        if not onset.empty:
            for col in ["AUROC", "AUPRC", "ECE", "Brier", "threshold"]:
                if col in onset.columns:
                    row[f"Onset_{col}"] = onset[col].iloc[0]
                elif f"Onset_{col}" in onset.columns:
                    row[f"Onset_{col}"] = onset[f"Onset_{col}"].iloc[0]

        row["comparison_group"] = "HemaTRACE"
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 6. TABLES AND PLOTS
# ============================================================

def _dl_normalize_for_combined_plot(df):
    d = df.copy()

    metrics = [
        ("MAE", False),
        ("RMSE", False),
        ("R2", True),
        ("Severe_AUROC", True),
        ("Severe_AUPRC", True),
        ("Onset_AUROC", True),
        ("Onset_AUPRC", True),
    ]

    rows = []
    for _, r in d.iterrows():
        for metric, higher_better in metrics:
            if metric not in d.columns:
                continue
            val = r.get(metric, np.nan)
            if not np.isfinite(val):
                continue
            col = d[metric].astype(float)
            mn, mx = np.nanmin(col.values), np.nanmax(col.values)
            if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
                score = 1.0
            else:
                score = (val - mn) / (mx - mn)
                if not higher_better:
                    score = 1.0 - score
            rows.append({
                "model": r["model"],
                "label": r.get("label", r["model"]),
                "comparison_group": r.get("comparison_group", "DL comparator"),
                "metric": metric,
                "raw_value": float(val),
                "normalized_score_higher_is_better": float(score),
            })
    return pd.DataFrame(rows)


def _dl_plot_metric_bar(df, metric, stem, title, ylabel=None, top_n=None):
    if metric not in df.columns:
        return None

    d = df[["model", "label", metric]].dropna().copy()
    if d.empty:
        return None

    ascending = metric in ["MAE", "RMSE", "Severe_ECE", "Onset_ECE"]
    d = d.sort_values(metric, ascending=ascending)
    if top_n is not None:
        d = d.head(top_n)

    plt.figure(figsize=(max(10, 0.55 * len(d)), 5.5))
    x = np.arange(len(d))
    plt.bar(x, d[metric].values)
    plt.xticks(x, d["label"].astype(str).values, rotation=45, ha="right")
    plt.ylabel(ylabel or metric)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    return _dl_save_current_fig_both(stem)


def _dl_plot_combined_normalized(norm_df, stem="final_dl_comparison_combined_normalized"):
    if norm_df.empty:
        return None

    pivot = norm_df.pivot_table(
        index="label",
        columns="metric",
        values="normalized_score_higher_is_better",
        aggfunc="mean",
    )

    # Sort by average score.
    pivot["Average"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("Average", ascending=False)
    plot_df = pivot.drop(columns=["Average"])

    plt.figure(figsize=(max(11, 0.55 * len(plot_df)), 6.5))
    x = np.arange(len(plot_df))
    width = 0.8 / max(1, len(plot_df.columns))

    for i, metric in enumerate(plot_df.columns):
        plt.bar(x + i * width, plot_df[metric].values, width=width, label=metric)

    plt.xticks(x + width * (len(plot_df.columns) - 1) / 2, plot_df.index, rotation=45, ha="right")
    plt.ylabel("Normalized score, higher is better")
    plt.title("Deep-learning comparator performance across metrics")
    plt.ylim(0, 1.05)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    return _dl_save_current_fig_both(stem)


def _dl_plot_3d_metric_surface(df, stem="final_dl_comparison_3d_metric_surface"):
    metrics = ["MAE", "RMSE", "R2", "Severe_AUROC", "Severe_AUPRC", "Onset_AUROC", "Onset_AUPRC"]
    metrics = [m for m in metrics if m in df.columns and df[m].notna().sum() > 0]
    if len(metrics) < 3 or df.empty:
        return None

    d = df[["label"] + metrics].copy()
    d = d.dropna(how="all", subset=metrics)
    if d.empty:
        return None

    # Normalize so 3D surface is interpretable.
    Z = []
    for _, r in d.iterrows():
        row = []
        for metric in metrics:
            vals = d[metric].astype(float).values
            mn, mx = np.nanmin(vals), np.nanmax(vals)
            v = r[metric]
            if not np.isfinite(v) or abs(mx - mn) < 1e-12:
                score = np.nan
            else:
                score = (v - mn) / (mx - mn)
                if metric in ["MAE", "RMSE"]:
                    score = 1 - score
            row.append(score)
        Z.append(row)

    Z = np.asarray(Z, dtype=float)
    Z = np.nan_to_num(Z, nan=np.nanmean(Z) if np.isfinite(np.nanmean(Z)) else 0.0)

    X, Y = np.meshgrid(np.arange(len(metrics)), np.arange(len(d)))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, alpha=0.85, edgecolor="k", linewidth=0.2)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["label"].astype(str).values)
    ax.set_zlabel("Normalized score")
    ax.set_title("3D surface of DL model comparison metrics")
    ax.view_init(elev=25, azim=-125)

    return _dl_save_current_fig_both(stem)


def _dl_make_pdf_report(fig_stems, output_name="final_dl_comparison_combined_figures.pdf"):
    pdf_pages_cls = _dl_get_global("PdfPages", None)
    if pdf_pages_cls is None:
        try:
            from matplotlib.backends.backend_pdf import PdfPages as pdf_pages_cls
        except Exception:
            return None

    report_dir = Path(_dl_get_global("REPORTS_DIR", Path("reports")))
    fig_png_dir = Path(_dl_get_global("FIGURE_PNG_DIR", Path("figures/png")))
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / output_name

    image_paths = []
    for stem in fig_stems:
        p = fig_png_dir / f"{stem}.png"
        if p.exists():
            image_paths.append(p)

    if not image_paths:
        return None

    with pdf_pages_cls(out) as pdf:
        for p in image_paths:
            img = plt.imread(str(p))
            plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis("off")
            plt.title(p.stem)
            pdf.savefig(bbox_inches="tight")
            plt.close()

    print(f"[SAVED] {out}")
    return out


# ============================================================
# 7. MAIN PUBLIC FUNCTION
# ============================================================

def run_deep_learning_model_comparison(
    temporal_paths: Dict[str, str],
    hema_metrics_path=None,
    model_names: Optional[List[str]] = None,
    config_overrides: Optional[dict] = None,
):
    """
    Train and evaluate DL comparator models using the same temporal NPZ files.

    Required:
        temporal_paths:
            {
              "train": ".../temporal_train.npz",
              "validation": ".../temporal_validation.npz",
              "test": ".../temporal_test.npz"
            }

    Optional:
        hema_metrics_path:
            usually TABLES_DIR / "final_all_model_metrics.csv"
            If provided, this function merges HemaTRACE final rows with DL comparator rows.

        model_names:
            subset of supported models.

        config_overrides:
            dict for changing max_epochs, patience, batch_size, lr, etc.

    Saves:
        tables/final_dl_model_comparison_long.csv
        tables/final_dl_model_comparison_table.csv
        tables/final_dl_model_comparison_manuscript_table.csv
        tables/final_dl_model_comparison_with_hematrace.csv
        tables/final_dl_model_comparison_normalized_scores.csv
        tables/final_dl_model_training_history.csv
        tables/final_dl_model_thresholds.csv
        models/dl_comparators/<model>/best_model.pt
        data/dl_predictions_validation_<model>.csv
        data/dl_predictions_test_<model>.csv
        figures/png/pdf/final_dl_comparison_*.png/pdf
        reports/final_dl_comparison_combined_figures.pdf
    """

    c = _make_dl_cfg(config_overrides)
    _dl_seed_everything(c.seed)

    save_df_func = _dl_get_global("save_df", _dl_save_df)
    save_json_func = _dl_get_global("save_json", _dl_save_json)

    tables_dir = Path(_dl_get_global("TABLES_DIR", Path("tables")))
    models_dir = Path(_dl_get_global("MODELS_DIR", Path("models")))
    logs_dir = Path(_dl_get_global("LOGS_DIR", Path("logs")))

    dl_model_root = models_dir / "dl_comparators"
    dl_model_root.mkdir(parents=True, exist_ok=True)

    if model_names is None:
        model_names = [
            "DL_MLP_Static",
            "DL_MLP_TemporalFlatten",
            "DL_CNN1D",
            "DL_LSTM",
            "DL_GRU",
            "DL_BiGRU",
            "DL_BiLSTM_Attention",
            "DL_Transformer",
            "DL_TCN",
        ]

    manifest = pd.DataFrame([
        {
            "model": "DL_MLP_Static",
            "label": "Static MLP",
            "description": "MLP using static/early-Hb context only.",
        },
        {
            "model": "DL_MLP_TemporalFlatten",
            "label": "Temporal flatten MLP",
            "description": "Flattened temporal tensor plus static context.",
        },
        {
            "model": "DL_CNN1D",
            "label": "1D CNN",
            "description": "Convolutional temporal baseline over 0-24 h bins.",
        },
        {
            "model": "DL_LSTM",
            "label": "LSTM",
            "description": "Unidirectional LSTM temporal baseline.",
        },
        {
            "model": "DL_GRU",
            "label": "GRU",
            "description": "Unidirectional GRU temporal baseline.",
        },
        {
            "model": "DL_BiGRU",
            "label": "BiGRU",
            "description": "Bidirectional GRU temporal baseline.",
        },
        {
            "model": "DL_BiLSTM_Attention",
            "label": "BiLSTM-Attention",
            "description": "Bidirectional LSTM with attention pooling.",
        },
        {
            "model": "DL_Transformer",
            "label": "Transformer encoder",
            "description": "Transformer encoder with attention pooling.",
        },
        {
            "model": "DL_TCN",
            "label": "TCN",
            "description": "Temporal convolutional network baseline.",
        },
    ])
    manifest = manifest[manifest["model"].isin(model_names)].copy()
    save_df_func(manifest, tables_dir / "final_dl_model_comparison_manifest.csv")

    train_raw = _dl_load_npz(temporal_paths["train"])
    val_raw = _dl_load_npz(temporal_paths["validation"])
    test_raw = _dl_load_npz(temporal_paths["test"])

    norm = _dl_compute_norm(train_raw)
    save_json_func(norm, logs_dir / "final_dl_model_comparison_normalization.json")

    train = _dl_apply_norm(train_raw, norm)
    val = _dl_apply_norm(val_raw, norm)
    test = _dl_apply_norm(test_raw, norm)

    all_metrics = []
    all_histories = []
    all_thresholds = []

    for model_name in model_names:
        result = _dl_train_one_model(
            model_name=model_name,
            train_data=train,
            val_data=val,
            test_data=test,
            norm=norm,
            c=c,
            out_root=dl_model_root,
        )
        all_metrics.append(result["metrics_long"])
        all_histories.append(result["history"])
        th = dict(result["thresholds"])
        th["model"] = model_name
        all_thresholds.append(th)

        # free GPU memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_long = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    histories = pd.concat(all_histories, ignore_index=True) if all_histories else pd.DataFrame()
    thresholds = pd.DataFrame(all_thresholds)

    save_df_func(metrics_long, tables_dir / "final_dl_model_comparison_long.csv")
    save_df_func(histories, tables_dir / "final_dl_model_training_history.csv")
    save_df_func(thresholds, tables_dir / "final_dl_model_thresholds.csv")

    wide = _dl_wide_metrics(metrics_long)
    wide["comparison_group"] = "DL comparator"

    # Attach nice labels from manifest.
    label_map = dict(zip(manifest["model"], manifest["label"]))
    wide["label"] = wide["model"].map(label_map).fillna(wide["label"])

    save_df_func(wide, tables_dir / "final_dl_model_comparison_table.csv")

    test_wide = wide[wide["split"].astype(str).str.lower().eq("test")].copy()
    keep_cols = [
        "model", "label",
        "MAE", "RMSE", "R2",
        "Severe_AUROC", "Severe_AUPRC", "Severe_ECE",
        "Onset_AUROC", "Onset_AUPRC", "Onset_ECE",
        "Severe_threshold", "Onset_threshold",
    ]
    keep_cols = [c for c in keep_cols if c in test_wide.columns]
    manuscript = test_wide[keep_cols].copy()

    sort_cols = [c for c in ["MAE", "Onset_AUROC", "Severe_AUROC"] if c in manuscript.columns]
    if "MAE" in sort_cols:
        manuscript = manuscript.sort_values(["MAE"], ascending=True)
    save_df_func(manuscript, tables_dir / "final_dl_model_comparison_manuscript_table.csv")

    hema = _dl_extract_hematrace_test_metrics(hema_metrics_path)
    if not hema.empty:
        common_cols = sorted(set(hema.columns).union(set(test_wide.columns)))
        hema2 = hema.reindex(columns=common_cols)
        dl2 = test_wide.reindex(columns=common_cols)
        combined = pd.concat([hema2, dl2], ignore_index=True)
    else:
        combined = test_wide.copy()

    if "comparison_group" not in combined.columns:
        combined["comparison_group"] = np.where(
            combined["model"].astype(str).str.startswith("DL_"),
            "DL comparator",
            "HemaTRACE",
        )

    save_df_func(combined, tables_dir / "final_dl_model_comparison_with_hematrace.csv")

    norm_scores = _dl_normalize_for_combined_plot(combined)
    save_df_func(norm_scores, tables_dir / "final_dl_model_comparison_normalized_scores.csv")

    # Individual and combined plots.
    fig_stems = []
    plot_specs = [
        ("MAE", "final_dl_comparison_mae", "Held-out test MAE by model", "MAE, lower is better"),
        ("RMSE", "final_dl_comparison_rmse", "Held-out test RMSE by model", "RMSE, lower is better"),
        ("R2", "final_dl_comparison_r2", "Held-out test R_square by model", "R_square, higher is better"),
        ("Severe_AUROC", "final_dl_comparison_severe_auroc", "Severe anemia AUROC by model", "AUROC"),
        ("Severe_AUPRC", "final_dl_comparison_severe_auprc", "Severe anemia AUPRC by model", "AUPRC"),
        ("Onset_AUROC", "final_dl_comparison_onset_auroc", "New-onset anemia AUROC by model", "AUROC"),
        ("Onset_AUPRC", "final_dl_comparison_onset_auprc", "New-onset anemia AUPRC by model", "AUPRC"),
        ("Onset_ECE", "final_dl_comparison_onset_ece", "New-onset anemia calibration error by model", "ECE, lower is better"),
    ]
    for metric, stem, title, ylabel in plot_specs:
        res = _dl_plot_metric_bar(combined, metric, stem, title, ylabel)
        if res is not None:
            fig_stems.append(stem)

    if _dl_plot_combined_normalized(norm_scores, "final_dl_comparison_combined_normalized") is not None:
        fig_stems.append("final_dl_comparison_combined_normalized")

    if _dl_plot_3d_metric_surface(combined, "final_dl_comparison_3d_metric_surface") is not None:
        fig_stems.append("final_dl_comparison_3d_metric_surface")

    _dl_make_pdf_report(fig_stems, "final_dl_comparison_combined_figures.pdf")

    summary = {
        "n_models": len(model_names),
        "models": model_names,
        "config": c.__dict__,
        "outputs": {
            "long_metrics": str(tables_dir / "final_dl_model_comparison_long.csv"),
            "wide_metrics": str(tables_dir / "final_dl_model_comparison_table.csv"),
            "manuscript_table": str(tables_dir / "final_dl_model_comparison_manuscript_table.csv"),
            "combined_with_hematrace": str(tables_dir / "final_dl_model_comparison_with_hematrace.csv"),
            "normalized_scores": str(tables_dir / "final_dl_model_comparison_normalized_scores.csv"),
            "model_root": str(dl_model_root),
        },
    }
    save_json_func(summary, tables_dir / "final_dl_model_comparison_summary.json")

    print("\n[DL COMPARISON COMPLETE]")
    print(f"Saved manuscript table: {tables_dir / 'final_dl_model_comparison_manuscript_table.csv'}")
    print(f"Saved combined HemaTRACE + DL table: {tables_dir / 'final_dl_model_comparison_with_hematrace.csv'}")
    print(f"Saved model directory: {dl_model_root}")

    return {
        "metrics_long": metrics_long,
        "wide_metrics": wide,
        "manuscript_table": manuscript,
        "combined_with_hematrace": combined,
        "normalized_scores": norm_scores,
        "histories": histories,
        "thresholds": thresholds,
        "summary": summary,
    }


# Backward-compatible alias if you used a previous function name.
def run_dl_model_comparison(
    temporal_paths: Dict[str, str],
    hema_metrics_path=None,
    model_names: Optional[List[str]] = None,
    config_overrides: Optional[dict] = None,
):
    return run_deep_learning_model_comparison(
        temporal_paths=temporal_paths,
        hema_metrics_path=hema_metrics_path,
        model_names=model_names,
        config_overrides=config_overrides,
    )

# ============================================================
# END HemaTRACE DEEP-LEARNING COMPARATOR ADD-ON

_dl_train_one_model_original = _dl_train_one_model
_run_deep_learning_model_comparison_original = run_deep_learning_model_comparison
_dl_plot_combined_normalized_original = _dl_plot_combined_normalized
_dl_plot_3d_metric_surface_original = _dl_plot_3d_metric_surface

STANDARD_MODEL_NAMES = {
    "Tabular": "Tabular",
    "HemaTRACE-Temporal": "HemaTRACE-Temporal",
    "HemaTRACE-OnsetSpecialized": "HemaTRACE-OnsetSpecialized",
    "HemaTRACE-Fusion": "HemaTRACE-Fusion",
    "DL_MLP_Static": "Static MLP",
    "DL_MLP_TemporalFlatten": "Temporal Flatten MLP",
    "DL_CNN1D": "1D CNN",
    "DL_LSTM": "LSTM",
    "DL_GRU": "GRU",
    "DL_BiGRU": "BiGRU",
    "DL_BiLSTM_Attention": "BiLSTM-Attention",
    "DL_Transformer": "Transformer",
    "DL_TCN": "TCN",
    "Static MLP": "Static MLP",
    "Temporal flatten MLP": "Temporal Flatten MLP",
    "1D CNN": "1D CNN",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "BiGRU": "BiGRU",
    "BiLSTM-Attention": "BiLSTM-Attention",
    "Transformer encoder": "Transformer",
    "TCN": "TCN",
}

MODEL_ORDER = [
    "HemaTRACE-Fusion", "Tabular", "HemaTRACE-Temporal", "HemaTRACE-OnsetSpecialized",
    "GRU", "BiGRU", "LSTM", "BiLSTM-Attention", "TCN", "1D CNN",
    "Transformer", "Temporal Flatten MLP", "Static MLP",
]

MANUSCRIPT_METRICS = [
    "MAE", "RMSE", "R2", "Severe_AUROC", "Severe_AUPRC", "Severe_ECE", "Onset_AUROC", "Onset_AUPRC", "Onset_ECE"
]
LOWER_IS_BETTER = {"MAE", "RMSE", "Severe_ECE", "Onset_ECE"}


def _std_model_label(x):
    return STANDARD_MODEL_NAMES.get(str(x), str(x).replace("DL_", "").replace("_", " "))


def _ensure_manuscript_dirs():
    base = Path(_dl_get_global("RUN_DIR", Path("."))) / "manuscript"
    dirs = {
        "base": base,
        "tables": base / "tables",
        "figures": base / "figures",
        "png": base / "figures" / "png",
        "pdf": base / "figures" / "pdf",
        "reports": base / "reports",
        "data": base / "data",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_fig_all_locations(stem, manuscript=True, tight=True):
    if tight:
        plt.tight_layout()
    save_current_fig_both = _dl_get_global("save_current_fig_both", None)
    saved = None
    if save_current_fig_both is not None:
        try:
            saved = save_current_fig_both(stem)
        except Exception:
            saved = None
    else:
        fig_png_dir = Path(_dl_get_global("FIGURE_PNG_DIR", Path("figures/png")))
        fig_pdf_dir = Path(_dl_get_global("FIGURE_PDF_DIR", Path("figures/pdf")))
        fig_png_dir.mkdir(parents=True, exist_ok=True)
        fig_pdf_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_png_dir / f"{stem}.png", dpi=int(_dl_get_global("cfg", cfg).DPI), bbox_inches="tight")
        plt.savefig(fig_pdf_dir / f"{stem}.pdf", bbox_inches="tight")
        saved = (fig_png_dir / f"{stem}.png", fig_pdf_dir / f"{stem}.pdf")
    if manuscript:
        md = _ensure_manuscript_dirs()
        plt.savefig(md["png"] / f"{stem}.png", dpi=int(_dl_get_global("cfg", cfg).DPI), bbox_inches="tight")
        plt.savefig(md["pdf"] / f"{stem}.pdf", bbox_inches="tight")
    return saved


def _safe_numeric_series(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _prepare_comparison_table_for_manuscript(df):
    d = df.copy()
    if "label" not in d.columns:
        d["label"] = d["model"].map(_std_model_label)
    else:
        d["label"] = d["label"].fillna(d["model"]).map(_std_model_label)
    if "comparison_group" not in d.columns:
        d["comparison_group"] = np.where(d["model"].astype(str).str.startswith("DL_"), "DL comparator", "HemaTRACE")
    for m in MANUSCRIPT_METRICS:
        if m in d.columns:
            d[m] = pd.to_numeric(d[m], errors="coerce")
    d["_order"] = d["label"].map({n: i for i, n in enumerate(MODEL_ORDER)}).fillna(999).astype(int)
    d = d.sort_values(["_order", "label"]).drop(columns=["_order"])
    return d


def _normalized_scores_wide(df, metrics=None):
    d = _prepare_comparison_table_for_manuscript(df)
    metrics = metrics or [m for m in MANUSCRIPT_METRICS if m in d.columns]
    rows = []
    for metric in metrics:
        vals = pd.to_numeric(d[metric], errors="coerce")
        good = vals.notna() & np.isfinite(vals)
        if good.sum() == 0:
            continue
        mn, mx = vals[good].min(), vals[good].max()
        if abs(mx - mn) < 1e-12:
            score = pd.Series(1.0, index=d.index)
        else:
            score = (vals - mn) / (mx - mn)
            if metric in LOWER_IS_BETTER:
                score = 1.0 - score
        for idx, r in d.iterrows():
            if pd.notna(score.loc[idx]) and np.isfinite(score.loc[idx]):
                rows.append({
                    "model": r["model"],
                    "label": r["label"],
                    "comparison_group": r.get("comparison_group", ""),
                    "metric": metric,
                    "raw_value": float(r[metric]) if pd.notna(r[metric]) else np.nan,
                    "normalized_score_higher_is_better": float(score.loc[idx]),
                })
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index="label", columns="metric", values="normalized_score_higher_is_better", aggfunc="mean")
    if not wide.empty:
        wide["Average"] = wide.mean(axis=1)
        wide = wide.sort_values("Average", ascending=False)
    return long, wide


def _plot_manuscript_metric_heatmap(df, stem="manuscript_all_models_metric_heatmap"):
    metrics = [m for m in ["MAE", "RMSE", "R2", "Severe_AUROC", "Severe_AUPRC", "Onset_AUROC", "Onset_AUPRC"] if m in df.columns]
    long, wide = _normalized_scores_wide(df, metrics)
    if wide.empty:
        return None
    plot_df = wide.drop(columns=["Average"], errors="ignore")
    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 0.35 * len(plot_df))))
    im = ax.imshow(plot_df.values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_xticklabels([c.replace("_", " ") for c in plot_df.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Normalized performance across all models")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Normalized score (higher is better)")
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            val = plot_df.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    _save_fig_all_locations(stem)
    plt.close(fig)
    return stem


def _plot_manuscript_all_models_bar(df, stem="manuscript_all_models_primary_metrics"):
    d = _prepare_comparison_table_for_manuscript(df)
    metrics = [m for m in ["MAE", "Severe_AUROC", "Severe_AUPRC", "Onset_AUROC", "Onset_AUPRC"] if m in d.columns]
    long, wide = _normalized_scores_wide(d, metrics)
    if wide.empty:
        return None
    plot_df = wide.drop(columns=["Average"], errors="ignore")
    plot_df = plot_df.loc[[x for x in MODEL_ORDER if x in plot_df.index] + [x for x in plot_df.index if x not in MODEL_ORDER]]
    x = np.arange(len(plot_df))
    width = 0.82 / max(1, len(plot_df.columns))
    plt.figure(figsize=(max(11.0, 0.62 * len(plot_df)), 6.2))
    for j, metric in enumerate(plot_df.columns):
        plt.bar(x + j * width, plot_df[metric].values, width=width, label=metric.replace("_", " "))
    plt.xticks(x + width * (len(plot_df.columns) - 1) / 2, plot_df.index, rotation=38, ha="right")
    plt.ylabel("Normalized score (higher is better)")
    plt.ylim(0, 1.05)
    plt.title("All-model comparison using primary manuscript metrics")
    plt.grid(axis="y", alpha=0.22)
    plt.legend(ncol=3, fontsize=8)
    _save_fig_all_locations(stem)
    plt.close()
    return stem


def _plot_metric_bars_standard(df, stem_prefix="manuscript_metric"):
    d = _prepare_comparison_table_for_manuscript(df)
    stems = []
    specs = [
        ("MAE", "Future minimum Hb MAE", "MAE (g/dL), lower is better", True),
        ("RMSE", "Future minimum Hb RMSE", "RMSE (g/dL), lower is better", True),
        ("R2", "Future minimum Hb R_square", "R_square, higher is better", False),
        ("Severe_AUROC", "Future severe anemia AUROC", "AUROC", False),
        ("Severe_AUPRC", "Future severe anemia AUPRC", "AUPRC", False),
        ("Onset_AUROC", "New-onset anemia AUROC", "AUROC", False),
        ("Onset_AUPRC", "New-onset anemia AUPRC", "AUPRC", False),
    ]
    for metric, title, ylabel, asc in specs:
        if metric not in d.columns or d[metric].notna().sum() == 0:
            continue
        p = d[["label", metric]].dropna().sort_values(metric, ascending=asc)
        plt.figure(figsize=(max(9, 0.55 * len(p)), 5.0))
        plt.bar(np.arange(len(p)), p[metric].values)
        plt.xticks(np.arange(len(p)), p["label"].values, rotation=38, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", alpha=0.25)
        stem = f"{stem_prefix}_{metric.lower()}"
        _save_fig_all_locations(stem)
        plt.close()
        stems.append(stem)
    return stems


def _plot_performance_efficiency(df, complexity_df=None, stem="manuscript_performance_efficiency_tradeoff"):
    d = _prepare_comparison_table_for_manuscript(df)
    if "MAE" not in d.columns:
        return None
    if complexity_df is not None and not complexity_df.empty:
        c = complexity_df.copy()
        if "label" not in c.columns and "model" in c.columns:
            c["label"] = c["model"].map(_std_model_label)
        d = d.merge(c[[col for col in ["label", "trainable_parameters", "total_training_time_sec", "checkpoint_size_mb"] if col in c.columns]], on="label", how="left")
    xcol = "trainable_parameters" if "trainable_parameters" in d.columns and d["trainable_parameters"].notna().sum() else None
    ycol = "MAE"
    if xcol is None:
        return None
    p = d.dropna(subset=[xcol, ycol]).copy()
    if p.empty:
        return None
    plt.figure(figsize=(8.5, 6.0))
    sizes = 80
    if "checkpoint_size_mb" in p.columns and p["checkpoint_size_mb"].notna().sum():
        ck = p["checkpoint_size_mb"].fillna(p["checkpoint_size_mb"].median())
        sizes = 60 + 220 * (ck - ck.min()) / max(ck.max() - ck.min(), 1e-9)
    plt.scatter(p[xcol].values, p[ycol].values, s=sizes, alpha=0.75)
    for _, r in p.iterrows():
        plt.annotate(r["label"], (r[xcol], r[ycol]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    plt.xscale("log")
    plt.xlabel("Trainable parameters (log scale)")
    plt.ylabel("MAE (g/dL), lower is better")
    plt.title("Performance--complexity trade-off")
    plt.grid(alpha=0.25)
    _save_fig_all_locations(stem)
    plt.close()
    return stem


def _plot_3d_metric_surface_clear(df, stem="manuscript_3d_model_metric_surface"):
    d = _prepare_comparison_table_for_manuscript(df)
    metrics = [m for m in ["MAE", "RMSE", "R2", "Severe_AUROC", "Severe_AUPRC", "Onset_AUROC", "Onset_AUPRC"] if m in d.columns]
    if len(metrics) < 3 or d.empty:
        return None
    _, wide = _normalized_scores_wide(d, metrics)
    if wide.empty:
        return None
    plot_df = wide.drop(columns=["Average"], errors="ignore")
    plot_df = plot_df.loc[[x for x in MODEL_ORDER if x in plot_df.index] + [x for x in plot_df.index if x not in MODEL_ORDER]]
    Z = plot_df.values.astype(float)
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    fig = plt.figure(figsize=(11.5, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.4, edgecolor="k", alpha=0.88)
    ax.set_xticks(np.arange(len(plot_df.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in plot_df.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=8)
    ax.set_zlabel("Normalized score")
    ax.set_title("3D model--metric performance surface")
    ax.view_init(elev=28, azim=-132)
    _save_fig_all_locations(stem)
    plt.close(fig)
    return stem


def _plot_3d_metric_bars_clear(df, stem="manuscript_3d_model_metric_bars"):
    d = _prepare_comparison_table_for_manuscript(df)
    metrics = [m for m in ["MAE", "R2", "Severe_AUROC", "Severe_AUPRC", "Onset_AUROC", "Onset_AUPRC"] if m in d.columns]
    if len(metrics) < 3:
        return None
    _, wide = _normalized_scores_wide(d, metrics)
    if wide.empty:
        return None
    plot_df = wide.drop(columns=["Average"], errors="ignore")
    plot_df = plot_df.head(12)
    xpos, ypos = np.meshgrid(np.arange(plot_df.shape[1]), np.arange(plot_df.shape[0]))
    xpos = xpos.ravel(); ypos = ypos.ravel(); zpos = np.zeros_like(xpos, dtype=float)
    dx = 0.55 * np.ones_like(xpos, dtype=float); dy = 0.55 * np.ones_like(ypos, dtype=float)
    dz = plot_df.values.ravel()
    fig = plt.figure(figsize=(11.5, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, alpha=0.86)
    ax.set_xticks(np.arange(plot_df.shape[1]) + 0.25)
    ax.set_xticklabels([c.replace("_", " ") for c in plot_df.columns], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(np.arange(plot_df.shape[0]) + 0.25)
    ax.set_yticklabels(plot_df.index, fontsize=8)
    ax.set_zlabel("Normalized score")
    ax.set_title("3D normalized metric bars for top models")
    ax.view_init(elev=27, azim=-125)
    _save_fig_all_locations(stem)
    plt.close(fig)
    return stem


def _dl_plot_combined_normalized(norm_df, stem="final_dl_comparison_combined_normalized"):
    # Replacement fixes crowded grouped bars and Static MLP labeling.
    if norm_df is None or norm_df.empty:
        return None
    d = norm_df.copy()
    d["label"] = d["label"].map(_std_model_label)
    pivot = d.pivot_table(index="label", columns="metric", values="normalized_score_higher_is_better", aggfunc="mean")
    if pivot.empty:
        return None
    pivot["Average"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("Average", ascending=False)
    plot_df = pivot.drop(columns=["Average"], errors="ignore")
    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 0.35 * len(plot_df))))
    im = ax.imshow(plot_df.values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_xticklabels([c.replace("_", " ") for c in plot_df.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Deep-learning and HemaTRACE comparison across normalized metrics")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Normalized score (higher is better)")
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            ax.text(j, i, f"{plot_df.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    _save_fig_all_locations(stem)
    plt.close(fig)
    return stem


def _dl_plot_3d_metric_surface(df, stem="final_dl_comparison_3d_metric_surface"):
    return _plot_3d_metric_surface_clear(df, stem)


def _parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _checkpoint_mb(model_dir):
    p = Path(model_dir) / "best_model.pt"
    return float(p.stat().st_size / (1024 ** 2)) if p.exists() else np.nan


def _benchmark_model_inference(model, data, c, norm, n_batches=20):
    try:
        device = next(model.parameters()).device
        loader = _dl_make_loader(data, c, shuffle=False)
        model.eval()
        n = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                batch = _dl_move_batch(batch, device)
                _ = model(batch["x_value"], batch["x_mask"], batch["x_delta"], batch["x_count"], batch["static_value"], batch["static_mask"])
                n += int(batch["y_reg"].shape[0])
                if bi + 1 >= n_batches:
                    break
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return {
            "benchmark_samples": int(n),
            "inference_time_sec": float(elapsed),
            "samples_per_sec": float(n / elapsed) if elapsed > 0 else np.nan,
            "latency_ms_per_sample": float(1000.0 * elapsed / n) if n > 0 else np.nan,
        }
    except Exception as e:
        return {"benchmark_error": str(e)}


def _estimate_model_macs_per_sample(model_name, temporal_dim, static_dim, c):
    # Coarse manuscript-level estimate, not hardware-specific profiling.
    T = int(_dl_get_global("cfg", cfg).N_BINS)
    F4 = int(temporal_dim * 4)
    H = int(c.hidden_dim)
    S = int(static_dim * 2)
    if model_name == "DL_MLP_Static":
        macs = S * c.static_hidden_dim + c.static_hidden_dim * c.static_embed_dim + c.static_embed_dim * H
    elif model_name == "DL_MLP_TemporalFlatten":
        macs = T * F4 * H + S * c.static_hidden_dim + H * H
    elif model_name == "DL_CNN1D":
        macs = T * F4 * H * 3 + S * c.static_hidden_dim + H * H
    elif model_name in ["DL_LSTM", "DL_GRU", "DL_BiGRU", "DL_BiLSTM_Attention"]:
        gates = 4 if "LSTM" in model_name else 3
        direction = 2 if "Bi" in model_name else 1
        macs = direction * T * gates * (F4 * H + H * H) + S * c.static_hidden_dim + H * H
    elif model_name == "DL_Transformer":
        macs = T * F4 * H + c.transformer_layers * (T * T * H + T * H * H * 4) + S * c.static_hidden_dim
    elif model_name == "DL_TCN":
        macs = c.tcn_levels * T * F4 * H * 3 + S * c.static_hidden_dim + H * H
    else:
        macs = np.nan
    return float(macs), float(macs / 1e6) if np.isfinite(macs) else np.nan


def _dl_train_one_model(model_name, train_data, val_data, test_data, norm, c, out_root):
    t0 = time.perf_counter()
    result = _dl_train_one_model_original(model_name, train_data, val_data, test_data, norm, c, out_root)
    total_sec = time.perf_counter() - t0
    model = result.get("model", None)
    model_dir = Path(out_root) / model_name
    total_params, trainable_params = _parameter_count(model) if model is not None else (np.nan, np.nan)
    temporal_dim = int(train_data["x_value"].shape[-1])
    static_dim = int(train_data["static_value"].shape[-1])
    macs, mflops = _estimate_model_macs_per_sample(model_name, temporal_dim, static_dim, c)
    bench = _benchmark_model_inference(model, test_data, c, norm) if model is not None else {}
    n_epochs = int(len(result["history"])) if "history" in result else np.nan
    complexity = {
        "model": model_name,
        "label": _std_model_label(model_name),
        "n_epochs_completed": n_epochs,
        "best_epoch": result.get("summary", {}).get("best_epoch", None),
        "total_training_time_sec": float(total_sec),
        "mean_time_sec_per_epoch": float(total_sec / n_epochs) if n_epochs and n_epochs > 0 else np.nan,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "checkpoint_size_mb": _checkpoint_mb(model_dir),
        "estimated_macs_per_sample": macs,
        "estimated_mflops_per_sample": mflops,
        "temporal_features": temporal_dim,
        "static_features": static_dim,
        "time_bins": int(_dl_get_global("cfg", cfg).N_BINS),
        "batch_size": int(c.batch_size),
        "device": str(next(model.parameters()).device) if model is not None else "unknown",
    }
    complexity.update(bench)
    save_df_func = _dl_get_global("save_df", _dl_save_df)
    save_json_func = _dl_get_global("save_json", _dl_save_json)
    tables_dir = Path(_dl_get_global("TABLES_DIR", Path("tables")))
    save_df_func(pd.DataFrame([complexity]), tables_dir / f"final_dl_time_complexity_{model_name}.csv")
    save_json_func(complexity, model_dir / "time_complexity_summary.json")
    md = _ensure_manuscript_dirs()
    save_df_func(pd.DataFrame([complexity]), md["tables"] / f"time_complexity_{model_name}.csv")
    result["complexity"] = complexity
    result["summary"]["time_complexity"] = complexity
    return result


def _make_manuscript_tables_and_plots(combined, norm_scores=None, complexity=None):
    save_df_func = _dl_get_global("save_df", _dl_save_df)
    save_json_func = _dl_get_global("save_json", _dl_save_json)
    md = _ensure_manuscript_dirs()
    d = _prepare_comparison_table_for_manuscript(combined)
    metrics = [m for m in MANUSCRIPT_METRICS if m in d.columns]
    manuscript = d[[c for c in ["model", "label", "comparison_group"] + metrics if c in d.columns]].copy()
    # Rank table.
    long_norm, wide_norm = _normalized_scores_wide(d, [m for m in metrics if m not in ["Severe_ECE", "Onset_ECE"]])
    if not wide_norm.empty:
        rank = wide_norm[["Average"]].reset_index().rename(columns={"label": "model_label", "Average": "mean_normalized_score"})
        rank["rank"] = np.arange(1, len(rank) + 1)
    else:
        rank = pd.DataFrame()
    save_df_func(manuscript, md["tables"] / "manuscript_all_models_clean_metrics.csv")
    save_df_func(long_norm, md["tables"] / "manuscript_all_models_normalized_scores_long.csv")
    save_df_func(wide_norm.reset_index(), md["tables"] / "manuscript_all_models_normalized_scores_wide.csv")
    save_df_func(rank, md["tables"] / "manuscript_all_models_ranked.csv")
    # Ablation-style table from actual available components.
    abl_models = ["Tabular", "HemaTRACE-Temporal", "HemaTRACE-OnsetSpecialized", "HemaTRACE-Fusion"]
    ablation = manuscript[manuscript["label"].isin(abl_models)].copy()
    if not ablation.empty:
        best = ablation[ablation["label"].eq("HemaTRACE-Fusion")]
        if not best.empty:
            for col in ["MAE", "Severe_AUROC", "Severe_AUPRC", "Onset_AUROC", "Onset_AUPRC"]:
                if col in ablation.columns:
                    base_val = float(best[col].iloc[0]) if pd.notna(best[col].iloc[0]) else np.nan
                    ablation[f"delta_vs_fusion_{col}"] = pd.to_numeric(ablation[col], errors="coerce") - base_val
        save_df_func(ablation, md["tables"] / "manuscript_architecture_ablation_clean.csv")
    if complexity is not None and not complexity.empty:
        c = complexity.copy()
        c["label"] = c["label"].map(_std_model_label)
        save_df_func(c, md["tables"] / "manuscript_model_time_complexity.csv")
    stems = []
    for f in [
        _plot_manuscript_metric_heatmap,
        _plot_manuscript_all_models_bar,
        _plot_3d_metric_surface_clear,
        _plot_3d_metric_bars_clear,
    ]:
        try:
            s = f(d)
            if s:
                stems.append(s)
        except Exception as e:
            print(f"[WARN] manuscript plot failed: {f.__name__}: {e}")
    try:
        stems.extend(_plot_metric_bars_standard(d))
    except Exception as e:
        print(f"[WARN] standard metric bars failed: {e}")
    try:
        s = _plot_performance_efficiency(d, complexity)
        if s:
            stems.append(s)
    except Exception as e:
        print(f"[WARN] efficiency plot failed: {e}")
    # PDF with all manuscript plots.
    try:
        report_path = md["reports"] / "manuscript_model_comparison_figures.pdf"
        with PdfPages(report_path) as pdf:
            for p in sorted(md["png"].glob("manuscript_*.png")):
                img = plt.imread(str(p))
                plt.figure(figsize=(11, 8.5))
                plt.imshow(img)
                plt.axis("off")
                plt.title(p.stem)
                pdf.savefig(bbox_inches="tight")
                plt.close()
        print(f"[SAVED] {report_path}")
    except Exception as e:
        print(f"[WARN] Could not create manuscript figure PDF: {e}")
    save_json_func({
        "manuscript_dir": str(md["base"]),
        "tables": str(md["tables"]),
        "figures_png": str(md["png"]),
        "figures_pdf": str(md["pdf"]),
        "reports": str(md["reports"]),
        "generated_plot_stems": stems,
    }, md["reports"] / "manuscript_outputs_manifest.json")
    return {"manuscript_metrics": manuscript, "ranked": rank, "normalized_long": long_norm, "normalized_wide": wide_norm}


def run_deep_learning_model_comparison(temporal_paths: Dict[str, str], hema_metrics_path=None, model_names: Optional[List[str]] = None, config_overrides: Optional[dict] = None):
    result = _run_deep_learning_model_comparison_original(
        temporal_paths=temporal_paths,
        hema_metrics_path=hema_metrics_path,
        model_names=model_names,
        config_overrides=config_overrides,
    )
    save_df_func = _dl_get_global("save_df", _dl_save_df)
    save_json_func = _dl_get_global("save_json", _dl_save_json)
    tables_dir = Path(_dl_get_global("TABLES_DIR", Path("tables")))
    models_dir = Path(_dl_get_global("MODELS_DIR", Path("models")))
    combined = result.get("combined_with_hematrace", pd.DataFrame())
    if combined is None or combined.empty:
        return result
    # Load all individual time/complexity files produced by the overridden trainer.
    complexity_parts = []
    for p in sorted(tables_dir.glob("final_dl_time_complexity_*.csv")):
        try:
            complexity_parts.append(pd.read_csv(p))
        except Exception:
            pass
    complexity = pd.concat(complexity_parts, ignore_index=True) if complexity_parts else pd.DataFrame()
    if not complexity.empty:
        complexity["label"] = complexity["model"].map(_std_model_label)
        complexity = complexity.sort_values(["mean_time_sec_per_epoch", "trainable_parameters"], na_position="last")
        save_df_func(complexity, tables_dir / "final_dl_time_complexity_all_models.csv")
    # Standardize combined labels and rewrite clean outputs.
    combined_clean = _prepare_comparison_table_for_manuscript(combined)
    save_df_func(combined_clean, tables_dir / "final_all_models_existing_proposed_comparison_clean.csv")
    norm_long, norm_wide = _normalized_scores_wide(combined_clean)
    save_df_func(norm_long, tables_dir / "final_all_models_existing_proposed_normalized_scores_long.csv")
    save_df_func(norm_wide.reset_index(), tables_dir / "final_all_models_existing_proposed_normalized_scores_wide.csv")
    manuscript_outputs = _make_manuscript_tables_and_plots(combined_clean, norm_long, complexity)
    summary = {
        "combined_table": str(tables_dir / "final_all_models_existing_proposed_comparison_clean.csv"),
        "normalized_scores_long": str(tables_dir / "final_all_models_existing_proposed_normalized_scores_long.csv"),
        "normalized_scores_wide": str(tables_dir / "final_all_models_existing_proposed_normalized_scores_wide.csv"),
        "time_complexity_all_models": str(tables_dir / "final_dl_time_complexity_all_models.csv"),
        "manuscript_dir": str(_ensure_manuscript_dirs()["base"]),
        "notes": [
            "This block treats the HemaTRACE component comparison and DL architecture comparison as manuscript ablation/comparator analysis.",
            "Static MLP label handling and crowded grouped-bar plots are replaced by heatmaps, ranked normalized tables, and clearer 3D plots.",
        ],
    }
    save_json_func(summary, tables_dir / "final_manuscript_ablation_comparison_summary.json")
    result["manuscript_outputs"] = manuscript_outputs
    result["time_complexity"] = complexity
    result["clean_combined_comparison"] = combined_clean
    return result
