"""Original full-component HemaTRACE ablation study."""

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, roc_auc_score

from ..config import (
    DATA_DIR,
    FIGURE_PNG_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    cfg,
)
from ..evaluation.statistics import evaluate_master, tune_thresholds_on_validation
from ..metrics import (
    classification_metrics,
    clean_prob,
    logit_np,
    regression_metrics,
    sigmoid_np,
    tune_threshold,
)
from ..models.fusion import fit_temperature
from ..models.temporal import (
    AttentionPooling,
    apply_norm,
    compute_norm,
    load_npz,
    make_loader,
    move_batch,
)
from ..utils import (
    json_converter,
    safe_name,
    save_current_fig_both,
    save_df,
    save_json,
    seed_everything,
)

ABLATION_MODEL_DIR = MODELS_DIR / "ablation"
ABLATION_TABLE_DIR = TABLES_DIR
ABLATION_DATA_DIR = DATA_DIR
ABLATION_FIGURE_PREFIX = "final_ablation"
ABLATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AblationSpec:
    name: str
    label: str
    use_temporal: bool = True
    use_static: bool = True
    use_attention: bool = True
    use_value: bool = True
    use_mask: bool = True
    use_delta: bool = True
    use_count: bool = True
    zero_treatment_features: bool = False
    zero_hb_trend_static_features: bool = False
    max_epochs: Optional[int] = None
    patience: Optional[int] = None
    lr: Optional[float] = None


class AblationHemaTRACEModel(nn.Module):
    """
    Flexible HemaTRACE ablation model.

    Supported component removals:
        - temporal branch removed -> static-only neural model
        - static branch removed -> temporal-only model
        - attention removed -> last GRU hidden state pooling
        - missingness/time/count removed -> selected temporal channels only
        - treatment features zeroed in data -> no transfusion features
        - Hb-trend static features zeroed in data -> no derived Hb-margin/slope variables
    """

    def __init__(self, n_temp: int, n_static: int, spec: AblationSpec):
        super().__init__()
        self.spec = spec
        self.channel_names = []
        if spec.use_value:
            self.channel_names.append("value")
        if spec.use_mask:
            self.channel_names.append("mask")
        if spec.use_delta:
            self.channel_names.append("delta")
        if spec.use_count:
            self.channel_names.append("count")

        if not spec.use_temporal and not spec.use_static:
            raise ValueError(f"Invalid ablation {spec.name}: both temporal and static branches are disabled.")
        if spec.use_temporal and len(self.channel_names) == 0:
            raise ValueError(f"Invalid ablation {spec.name}: temporal branch has no input channels.")

        self.use_temporal = spec.use_temporal
        self.use_static = spec.use_static
        self.use_attention = spec.use_attention and spec.use_temporal

        if self.use_temporal:
            tin = n_temp * len(self.channel_names)
            self.temporal_encoder = nn.Sequential(
                nn.Linear(tin, cfg.TEMPORAL_EMBED_DIM),
                nn.LayerNorm(cfg.TEMPORAL_EMBED_DIM),
                nn.GELU(),
                nn.Dropout(cfg.DROPOUT),
                nn.Linear(cfg.TEMPORAL_EMBED_DIM, cfg.TEMPORAL_EMBED_DIM),
                nn.LayerNorm(cfg.TEMPORAL_EMBED_DIM),
                nn.GELU(),
                nn.Dropout(cfg.DROPOUT),
            )
            self.gru = nn.GRU(
                cfg.TEMPORAL_EMBED_DIM,
                cfg.GRU_HIDDEN_DIM,
                num_layers=cfg.GRU_LAYERS,
                batch_first=True,
                bidirectional=cfg.GRU_BIDIRECTIONAL,
            )
            self.temporal_out_dim = cfg.GRU_HIDDEN_DIM * (2 if cfg.GRU_BIDIRECTIONAL else 1)
            if self.use_attention:
                self.attn = AttentionPooling(self.temporal_out_dim)
            else:
                self.attn = None
        else:
            self.temporal_out_dim = 0
            self.temporal_encoder = None
            self.gru = None
            self.attn = None

        if self.use_static:
            sin = n_static * 2
            self.static_encoder = nn.Sequential(
                nn.Linear(sin, cfg.STATIC_HIDDEN_DIM),
                nn.LayerNorm(cfg.STATIC_HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(cfg.DROPOUT),
                nn.Linear(cfg.STATIC_HIDDEN_DIM, cfg.STATIC_EMBED_DIM),
                nn.LayerNorm(cfg.STATIC_EMBED_DIM),
                nn.GELU(),
                nn.Dropout(cfg.DROPOUT),
            )
            self.static_out_dim = cfg.STATIC_EMBED_DIM
        else:
            self.static_encoder = None
            self.static_out_dim = 0

        shared_in = self.temporal_out_dim + self.static_out_dim
        self.shared = nn.Sequential(
            nn.Linear(shared_in, cfg.SHARED_HIDDEN_DIM),
            nn.LayerNorm(cfg.SHARED_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.SHARED_HIDDEN_DIM, cfg.SHARED_HIDDEN_DIM),
            nn.LayerNorm(cfg.SHARED_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
        )
        self.delta_head = nn.Linear(cfg.SHARED_HIDDEN_DIM, 1)
        self.severe_head = nn.Linear(cfg.SHARED_HIDDEN_DIM, 1)
        self.onset_head = nn.Linear(cfg.SHARED_HIDDEN_DIM, 1)

    def _make_temporal_input(self, xv, xm, xd, xc):
        xs = []
        if self.spec.use_value:
            xs.append(xv)
        if self.spec.use_mask:
            xs.append(xm)
        if self.spec.use_delta:
            xs.append(xd)
        if self.spec.use_count:
            xs.append(xc)
        return torch.cat(xs, dim=-1)

    def forward(self, xv, xm, xd, xc, sv, sm):
        reps = []
        attn_w = None

        if self.use_temporal:
            x = self._make_temporal_input(xv, xm, xd, xc)
            x = self.temporal_encoder(x)
            h, _ = self.gru(x)
            if self.use_attention:
                pooled, attn_w = self.attn(h)
            else:
                pooled = h[:, -1, :]
            reps.append(pooled)

        if self.use_static:
            s = self.static_encoder(torch.cat([sv, sm], dim=-1))
            reps.append(s)

        z = self.shared(torch.cat(reps, dim=-1))
        return {
            "delta_z": self.delta_head(z).squeeze(-1),
            "severe_logit": self.severe_head(z).squeeze(-1),
            "onset_logit": self.onset_head(z).squeeze(-1),
            "attn": attn_w,
        }


def ablation_specs() -> List[AblationSpec]:
    """Journal-ready ablation set. Keep FullTemporal first because deltas are computed against it."""
    return [
        AblationSpec(
            name="FullTemporal_ReRun",
            label="Full temporal model rerun",
            use_temporal=True,
            use_static=True,
            use_attention=True,
            use_value=True,
            use_mask=True,
            use_delta=True,
            use_count=True,
        ),
        AblationSpec(
            name="NoAttention",
            label="No attention pooling",
            use_temporal=True,
            use_static=True,
            use_attention=False,
            use_value=True,
            use_mask=True,
            use_delta=True,
            use_count=True,
        ),
        AblationSpec(
            name="TemporalOnly_NoStatic",
            label="Temporal branch only",
            use_temporal=True,
            use_static=False,
            use_attention=True,
            use_value=True,
            use_mask=True,
            use_delta=True,
            use_count=True,
        ),
        AblationSpec(
            name="StaticOnly_NoTemporal",
            label="Static/aggregate branch only",
            use_temporal=False,
            use_static=True,
            use_attention=False,
            use_value=False,
            use_mask=False,
            use_delta=False,
            use_count=False,
        ),
        AblationSpec(
            name="ValueOnly_NoMissingnessTimeCount",
            label="Value only: no mask, delta, count",
            use_temporal=True,
            use_static=True,
            use_attention=True,
            use_value=True,
            use_mask=False,
            use_delta=False,
            use_count=False,
        ),
        AblationSpec(
            name="NoTimeDelta",
            label="No time-delta channel",
            use_temporal=True,
            use_static=True,
            use_attention=True,
            use_value=True,
            use_mask=True,
            use_delta=False,
            use_count=True,
        ),
        AblationSpec(
            name="NoObservationCount",
            label="No observation-count channel",
            use_temporal=True,
            use_static=True,
            use_attention=True,
            use_value=True,
            use_mask=True,
            use_delta=True,
            use_count=False,
        ),
        AblationSpec(
            name="NoTransfusionFeatures",
            label="No RBC transfusion features",
            use_temporal=True,
            use_static=True,
            use_attention=True,
            use_value=True,
            use_mask=True,
            use_delta=True,
            use_count=True,
            zero_treatment_features=True,
        ),
        AblationSpec(
            name="NoHbTrendStaticFeatures",
            label="No Hb trend/margin static features",
            use_temporal=True,
            use_static=True,
            use_attention=True,
            use_value=True,
            use_mask=True,
            use_delta=True,
            use_count=True,
            zero_hb_trend_static_features=True,
        ),
    ]


def clone_npz_dict(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in d.items()}


def get_temporal_feature_indices_for_ablation(patterns: List[str]) -> List[int]:
    path = TABLES_DIR / "final_temporal_feature_names.csv"
    if not path.exists():
        return []
    f = pd.read_csv(path)
    name_col = "temporal_feature" if "temporal_feature" in f.columns else f.columns[-1]
    idx_col = "feature_idx" if "feature_idx" in f.columns else f.columns[0]
    idx = []
    for _, r in f.iterrows():
        nm = str(r[name_col]).lower()
        if any(p.lower() in nm for p in patterns):
            idx.append(int(r[idx_col]))
    return sorted(set(idx))


def get_static_feature_indices_for_ablation(patterns: List[str]) -> List[int]:
    path = TABLES_DIR / "final_static_feature_names.csv"
    if not path.exists():
        return []
    f = pd.read_csv(path)
    name_col = "static_feature" if "static_feature" in f.columns else f.columns[-1]
    idx = []
    for i, nm in enumerate(f[name_col].astype(str).tolist()):
        low = nm.lower()
        if any(p.lower() in low for p in patterns):
            idx.append(i)
    return sorted(set(idx))


def apply_ablation_data_masks(data: Dict[str, np.ndarray], spec: AblationSpec) -> Dict[str, np.ndarray]:
    """Zero selected feature groups after normalization while preserving tensor shapes."""
    out = clone_npz_dict(data)

    if spec.zero_treatment_features:
        treat_idx = get_temporal_feature_indices_for_ablation(["treat_rbc", "transfusion"])
        if len(treat_idx):
            out["x_value_norm"][:, :, treat_idx] = 0.0
            out["x_mask"][:, :, treat_idx] = 0.0
            out["x_delta_norm"][:, :, treat_idx] = 0.0
            out["x_count_norm"][:, :, treat_idx] = 0.0

    if spec.zero_hb_trend_static_features:
        hb_patterns = [
            "hb_slope", "hemoglobin_slope", "hb_margin", "early_last_hb", "early_min_hb",
            "hb_range", "last_minus_min", "sex_specific_anemia_threshold",
        ]
        sidx = get_static_feature_indices_for_ablation(hb_patterns)
        if len(sidx):
            out["static_norm"][:, sidx] = 0.0
            out["static_mask"][:, sidx] = 0.0

    return out


def ablation_loss(out, batch):
    reg_loss = nn.SmoothL1Loss()(out["delta_z"], batch["target_delta_z"])
    sev_loss = nn.BCEWithLogitsLoss()(out["severe_logit"], batch["y_sev"])
    onset_raw = nn.BCEWithLogitsLoss(reduction="none")(out["onset_logit"], batch["y_on"])
    onset_loss = (onset_raw * batch["on_mask"]).sum() / batch["on_mask"].sum().clamp_min(1.0)
    return (
        cfg.REG_LOSS_WEIGHT * reg_loss
        + cfg.SEVERE_LOSS_WEIGHT * sev_loss
        + cfg.ONSET_LOSS_WEIGHT * onset_loss
    )


# ============================================================
# FIXED ABLATION PREDICTION
# Replace your existing predict_ablation() with this block
# ============================================================

def predict_ablation(model, loader, device, norm):
    model.eval()

    rows = {
        "pred_hb": [],
        "sev_prob": [],
        "on_prob": [],
        "sev_logit": [],
        "on_logit": [],
        "y_reg": [],
        "y_sev": [],
        "y_on": [],
        "on_mask": [],
        "subject_id": [],
        "hadm_id": [],
        "stay_id": [],
        "attn": [],
    }

    with torch.no_grad():
        for batch in loader:
            # Keep IDs on CPU before moving the rest.
            subject_id = batch.get("subject_id", None)
            hadm_id = batch.get("hadm_id", None)
            stay_id = batch.get("stay_id", None)

            batch = move_batch(batch, device)

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

            pred_hb = batch["base_hb"].detach().cpu().numpy() + pred_delta

            rows["pred_hb"].append(pred_hb)
            rows["sev_logit"].append(out["severe_logit"].detach().cpu().numpy())
            rows["on_logit"].append(out["onset_logit"].detach().cpu().numpy())
            rows["sev_prob"].append(torch.sigmoid(out["severe_logit"]).detach().cpu().numpy())
            rows["on_prob"].append(torch.sigmoid(out["onset_logit"]).detach().cpu().numpy())

            rows["y_reg"].append(batch["y_reg"].detach().cpu().numpy())
            rows["y_sev"].append(batch["y_sev"].detach().cpu().numpy())
            rows["y_on"].append(batch["y_on"].detach().cpu().numpy())
            rows["on_mask"].append(batch["on_mask"].detach().cpu().numpy())

            if subject_id is not None:
                rows["subject_id"].append(subject_id.detach().cpu().numpy())
            else:
                rows["subject_id"].append(np.full(len(pred_hb), -1, dtype=np.int64))

            if hadm_id is not None:
                rows["hadm_id"].append(hadm_id.detach().cpu().numpy())
            else:
                rows["hadm_id"].append(np.full(len(pred_hb), -1, dtype=np.int64))

            if stay_id is not None:
                rows["stay_id"].append(stay_id.detach().cpu().numpy())
            else:
                rows["stay_id"].append(np.full(len(pred_hb), -1, dtype=np.int64))

            if out.get("attn") is not None:
                rows["attn"].append(out["attn"].detach().cpu().numpy())

    flat = {}

    for k, parts in rows.items():
        if k == "attn":
            flat[k] = np.concatenate(parts, axis=0) if len(parts) > 0 else None
        else:
            if len(parts) == 0:
                flat[k] = np.asarray([])
            else:
                flat[k] = np.concatenate(parts, axis=0)

    return flat


def ablation_pred_df(pred: Dict[str, np.ndarray], prefix: str) -> pd.DataFrame:
    df = pd.DataFrame({
        "subject_id": pred["subject_id"].astype(np.int64),
        "hadm_id": pred["hadm_id"].astype(np.int64),
        "stay_id": pred["stay_id"].astype(np.int64),
        "true_future_min_hb": pred["y_reg"].astype(float),
        "true_future_severe_anemia": pred["y_sev"].astype(int),
        "true_new_onset_anemia": pred["y_on"].astype(int),
        "new_onset_mask": pred["on_mask"].astype(int),
        f"{prefix}_pred_future_min_hb": pred["pred_hb"].astype(float),
        f"{prefix}_pred_severe_prob": clean_prob(pred["sev_prob"]),
        f"{prefix}_pred_onset_prob": clean_prob(pred["on_prob"]),
        f"{prefix}_severe_logit": pred["sev_logit"].astype(float),
        f"{prefix}_onset_logit": pred["on_logit"].astype(float),
    })
    if pred.get("attn") is not None:
        attn = pred["attn"]
        for b in range(attn.shape[1]):
            df[f"{prefix}_attn_bin_{b}"] = attn[:, b]
    return df


def score_ablation_validation(pred: Dict[str, np.ndarray]) -> float:
    mae = mean_absolute_error(pred["y_reg"], pred["pred_hb"])
    try:
        sev_auc = roc_auc_score(pred["y_sev"].astype(int), pred["sev_prob"])
    except Exception:
        sev_auc = 0.5
    m = pred["on_mask"] > 0
    try:
        on_auc = roc_auc_score(pred["y_on"][m].astype(int), pred["on_prob"][m]) if m.sum() > 10 else 0.5
    except Exception:
        on_auc = 0.5
    # Higher is better. MAE is converted to a bounded negative component.
    return float(sev_auc + cfg.SCORE_ONSET_WEIGHT * on_auc - cfg.SCORE_MAE_WEIGHT * mae)


def train_one_ablation_model(temporal_paths: Dict[str, str], spec: AblationSpec) -> Dict[str, object]:
    seed_everything(cfg.RANDOM_STATE + abs(hash(spec.name)) % 10000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = ABLATION_MODEL_DIR / safe_name(spec.name)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_npz(temporal_paths["train"])
    val_raw = load_npz(temporal_paths["validation"])
    test_raw = load_npz(temporal_paths["test"])

    norm = compute_norm(train_raw)
    train = apply_ablation_data_masks(apply_norm(train_raw, norm), spec)
    val = apply_ablation_data_masks(apply_norm(val_raw, norm), spec)
    test = apply_ablation_data_masks(apply_norm(test_raw, norm), spec)

    train_loader = make_loader(train, shuffle=True)
    val_loader = make_loader(val, shuffle=False)
    test_loader = make_loader(test, shuffle=False)

    n_temp = train["x_value_norm"].shape[-1]
    n_static = train["static_norm"].shape[-1]
    model = AblationHemaTRACEModel(n_temp=n_temp, n_static=n_static, spec=spec).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec.lr or cfg.LR),
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

    max_epochs = int(spec.max_epochs or cfg.MAX_EPOCHS)
    patience = int(spec.patience or cfg.PATIENCE)
    best_score = -np.inf
    bad = 0
    hist = []
    ckpt_path = model_dir / "best_model.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        nb = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(
                batch["x_value"], batch["x_mask"], batch["x_delta"], batch["x_count"],
                batch["static_value"], batch["static_mask"],
            )
            loss = ablation_loss(out, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
            loss_sum += float(loss.item())
            nb += 1

        val_pred = predict_ablation(model, val_loader, device, norm)
        score = score_ablation_validation(val_pred)
        scheduler.step(score)
        val_mae = mean_absolute_error(val_pred["y_reg"], val_pred["pred_hb"])
        try:
            val_sev_auc = roc_auc_score(val_pred["y_sev"].astype(int), val_pred["sev_prob"])
        except Exception:
            val_sev_auc = np.nan
        m = val_pred["on_mask"] > 0
        try:
            val_on_auc = roc_auc_score(val_pred["y_on"][m].astype(int), val_pred["on_prob"][m]) if m.sum() > 10 else np.nan
        except Exception:
            val_on_auc = np.nan

        hist.append({
            "ablation": spec.name,
            "label": spec.label,
            "epoch": epoch,
            "train_loss": loss_sum / max(nb, 1),
            "validation_score": score,
            "validation_MAE": float(val_mae),
            "validation_severe_AUROC": float(val_sev_auc) if np.isfinite(val_sev_auc) else np.nan,
            "validation_onset_AUROC": float(val_on_auc) if np.isfinite(val_on_auc) else np.nan,
            "lr": float(opt.param_groups[0]["lr"]),
        })

        if score > best_score:
            best_score = score
            bad = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm": norm,
                "spec": asdict(spec),
                "epoch": epoch,
                "score": score,
            }, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                break

    hist_df = pd.DataFrame(hist)
    save_df(hist_df, ABLATION_TABLE_DIR / f"final_ablation_history_{safe_name(spec.name)}.csv")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_pred = predict_ablation(model, val_loader, device, norm)
    test_pred = predict_ablation(model, test_loader, device, norm)

    prefix = "ablation_" + safe_name(spec.name)
    val_df = ablation_pred_df(val_pred, prefix)
    test_df = ablation_pred_df(test_pred, prefix)

    # Temperature calibration is fitted on validation only and then applied to validation/test.
    sev_temp, val_sev_cal = fit_temperature(val_df["true_future_severe_anemia"].values, val_df[f"{prefix}_pred_severe_prob"].values)
    test_sev_cal = sigmoid_np(logit_np(test_df[f"{prefix}_pred_severe_prob"].values) / sev_temp)
    val_df[f"{prefix}_pred_severe_prob_cal"] = clean_prob(val_sev_cal)
    test_df[f"{prefix}_pred_severe_prob_cal"] = clean_prob(test_sev_cal)

    on_temp = np.nan
    val_df[f"{prefix}_pred_onset_prob_cal"] = np.nan
    test_df[f"{prefix}_pred_onset_prob_cal"] = np.nan
    mv = val_df["new_onset_mask"].astype(bool).values
    mt = test_df["new_onset_mask"].astype(bool).values
    if mv.sum() > 10 and len(np.unique(val_df.loc[mv, "true_new_onset_anemia"].values)) > 1:
        on_temp, val_on_cal = fit_temperature(
            val_df.loc[mv, "true_new_onset_anemia"].values,
            val_df.loc[mv, f"{prefix}_pred_onset_prob"].values,
        )
        val_df.loc[mv, f"{prefix}_pred_onset_prob_cal"] = clean_prob(val_on_cal)
        test_df.loc[mt, f"{prefix}_pred_onset_prob_cal"] = clean_prob(
            sigmoid_np(logit_np(test_df.loc[mt, f"{prefix}_pred_onset_prob"].values) / on_temp)
        )

    save_df(val_df, ABLATION_DATA_DIR / f"ablation_predictions_validation_{safe_name(spec.name)}.csv")
    save_df(test_df, ABLATION_DATA_DIR / f"ablation_predictions_test_{safe_name(spec.name)}.csv")

    save_json({
        "ablation": spec.name,
        "label": spec.label,
        "checkpoint": str(ckpt_path),
        "best_epoch": int(ckpt.get("epoch", -1)),
        "best_validation_score": float(ckpt.get("score", np.nan)),
        "severe_temperature": float(sev_temp),
        "onset_temperature": float(on_temp) if np.isfinite(on_temp) else None,
        "spec": asdict(spec),
    }, model_dir / "ablation_summary.json")

    return {
        "spec": spec,
        "model_path": ckpt_path,
        "history": hist_df,
        "val_df": val_df,
        "test_df": test_df,
        "prefix": prefix,
    }


def evaluate_single_ablation_df(df: pd.DataFrame, split: str, spec: AblationSpec, prefix: str) -> pd.DataFrame:
    rows = []
    reg_col = f"{prefix}_pred_future_min_hb"
    sev_col = f"{prefix}_pred_severe_prob_cal"
    onset_col = f"{prefix}_pred_onset_prob_cal"

    reg = regression_metrics(df["true_future_min_hb"].values, df[reg_col].values)
    rows.append({"split": split, "model": spec.name, "label": spec.label, "task": "future_min_hb_regression", **reg})

    sev_thr, sev_score = tune_threshold(df["true_future_severe_anemia"].astype(int).values, df[sev_col].values)
    sev = classification_metrics(df["true_future_severe_anemia"].astype(int).values, df[sev_col].values, threshold=sev_thr)
    rows.append({"split": split, "model": spec.name, "label": spec.label, "task": "severe_anemia_classification", "validation_or_internal_threshold_score": sev_score, **sev})

    m = df["new_onset_mask"].astype(bool).values
    if m.sum() > 10 and df.loc[m, onset_col].notna().sum() > 10 and len(np.unique(df.loc[m, "true_new_onset_anemia"].values)) > 1:
        on_thr, on_score = tune_threshold(df.loc[m, "true_new_onset_anemia"].astype(int).values, df.loc[m, onset_col].values)
        onset = classification_metrics(df.loc[m, "true_new_onset_anemia"].astype(int).values, df.loc[m, onset_col].values, threshold=on_thr)
        rows.append({"split": split, "model": spec.name, "label": spec.label, "task": "new_onset_anemia_classification", "validation_or_internal_threshold_score": on_score, **onset})

    return pd.DataFrame(rows)


def compact_ablation_table(metrics_df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    rows = []
    d = metrics_df[metrics_df["split"] == split].copy()
    for model, g in d.groupby("model", sort=False):
        label = g["label"].dropna().iloc[0] if "label" in g.columns and g["label"].notna().any() else model
        reg = g[g.task == "future_min_hb_regression"]
        sev = g[g.task == "severe_anemia_classification"]
        onset = g[g.task == "new_onset_anemia_classification"]
        rows.append({
            "model": model,
            "label": label,
            "MAE": float(reg["MAE"].iloc[0]) if len(reg) else np.nan,
            "RMSE": float(reg["RMSE"].iloc[0]) if len(reg) else np.nan,
            "R2": float(reg["R2"].iloc[0]) if len(reg) else np.nan,
            "Severe_AUROC": float(sev["AUROC"].iloc[0]) if len(sev) else np.nan,
            "Severe_AUPRC": float(sev["AUPRC"].iloc[0]) if len(sev) else np.nan,
            "Severe_ECE": float(sev["ECE"].iloc[0]) if len(sev) else np.nan,
            "Onset_AUROC": float(onset["AUROC"].iloc[0]) if len(onset) else np.nan,
            "Onset_AUPRC": float(onset["AUPRC"].iloc[0]) if len(onset) else np.nan,
            "Onset_ECE": float(onset["ECE"].iloc[0]) if len(onset) else np.nan,
        })
    out = pd.DataFrame(rows)
    return out


def add_existing_main_models_to_ablation_metrics(metrics_df: pd.DataFrame, threshold_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add already trained Tabular/Temporal/OnsetSpecialized/Fusion rows if final master files exist."""
    parts = [metrics_df]
    val_path = DATA_DIR / "final_master_predictions_validation.csv"
    test_path = DATA_DIR / "final_master_predictions_test.csv"
    if val_path.exists() and test_path.exists():
        val_master = pd.read_csv(val_path)
        test_master = pd.read_csv(test_path)
        if threshold_df is None or threshold_df.empty:
            threshold_df = tune_thresholds_on_validation(val_master)
        main_val = evaluate_master(val_master, "validation", threshold_df)
        main_test = evaluate_master(test_master, "test", threshold_df)
        main = pd.concat([main_val, main_test], ignore_index=True)
        main["label"] = main["model"]
        parts.append(main)
    return pd.concat(parts, ignore_index=True, sort=False)


def compute_ablation_deltas(clean_df: pd.DataFrame, reference_model: str = "FullTemporal_ReRun") -> pd.DataFrame:
    d = clean_df.copy()
    ref = d[d["model"] == reference_model]
    if ref.empty and "HemaTRACE-Temporal" in d["model"].values:
        reference_model = "HemaTRACE-Temporal"
        ref = d[d["model"] == reference_model]
    if ref.empty:
        return pd.DataFrame()
    ref = ref.iloc[0]
    rows = []
    for _, r in d.iterrows():
        rows.append({
            "reference_model": reference_model,
            "model": r["model"],
            "label": r.get("label", r["model"]),
            "delta_MAE_positive_worse": float(r["MAE"] - ref["MAE"]) if pd.notna(r.get("MAE")) and pd.notna(ref.get("MAE")) else np.nan,
            "delta_Severe_AUROC_positive_better": float(r["Severe_AUROC"] - ref["Severe_AUROC"]) if pd.notna(r.get("Severe_AUROC")) and pd.notna(ref.get("Severe_AUROC")) else np.nan,
            "delta_Onset_AUROC_positive_better": float(r["Onset_AUROC"] - ref["Onset_AUROC"]) if pd.notna(r.get("Onset_AUROC")) and pd.notna(ref.get("Onset_AUROC")) else np.nan,
            "delta_Severe_ECE_positive_worse": float(r["Severe_ECE"] - ref["Severe_ECE"]) if pd.notna(r.get("Severe_ECE")) and pd.notna(ref.get("Severe_ECE")) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_ablation_metric_bars(clean_df: pd.DataFrame):
    d = clean_df.copy()
    d = d[d["model"].notna()].copy()
    metrics = ["MAE", "Severe_AUROC", "Onset_AUROC", "Severe_AUPRC", "Onset_AUPRC"]
    for metric in metrics:
        if metric not in d.columns or d[metric].notna().sum() == 0:
            continue
        dd = d[["label", metric]].dropna().copy()
        dd = dd.sort_values(metric, ascending=(metric == "MAE"))
        plt.figure(figsize=(12, max(5, 0.45 * len(dd))))
        plt.barh(dd["label"].astype(str), dd[metric].astype(float))
        plt.xlabel(metric)
        plt.ylabel("Ablation variant")
        plt.title(f"Ablation comparison: {metric}")
        save_current_fig_both(f"{ABLATION_FIGURE_PREFIX}_{metric}_bar")
        plt.close()

    # Combined compact panel as one figure without subplots? Requirement says all ablation combined.
    long_rows = []
    for _, r in d.iterrows():
        for metric in ["MAE", "Severe_AUROC", "Onset_AUROC"]:
            if metric in r and pd.notna(r[metric]):
                long_rows.append({"label": r["label"], "metric": metric, "value": float(r[metric])})
    long_df = pd.DataFrame(long_rows)
    if not long_df.empty:
        pivot = long_df.pivot_table(index="label", columns="metric", values="value", aggfunc="first")
        save_df(pivot.reset_index(), ABLATION_TABLE_DIR / "final_ablation_combined_plot_source.csv")
        plt.figure(figsize=(12, max(6, 0.45 * len(pivot))))
        x = np.arange(len(pivot.index))
        width = 0.25
        plotted = 0
        for metric in ["MAE", "Severe_AUROC", "Onset_AUROC"]:
            if metric in pivot.columns:
                vals = pivot[metric].values.astype(float)
                # Normalize MAE inversely for visual combined comparison only.
                if metric == "MAE":
                    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                    vals_plot = 1.0 - (vals - vmin) / (vmax - vmin + 1e-8)
                    lab = "MAE normalized, higher better"
                else:
                    vals_plot = vals
                    lab = metric
                plt.bar(x + (plotted - 1) * width, vals_plot, width=width, label=lab)
                plotted += 1
        plt.xticks(x, pivot.index.astype(str), rotation=70, ha="right")
        plt.ylabel("Comparable score")
        plt.title("Combined ablation plot: regression + classification")
        plt.legend()
        save_current_fig_both(f"{ABLATION_FIGURE_PREFIX}_combined_all_ablation_metrics")
        plt.close()


def plot_ablation_delta_bars(delta_df: pd.DataFrame):
    if delta_df.empty:
        return
    for metric in ["delta_MAE_positive_worse", "delta_Severe_AUROC_positive_better", "delta_Onset_AUROC_positive_better"]:
        if metric not in delta_df.columns or delta_df[metric].notna().sum() == 0:
            continue
        dd = delta_df[["label", metric]].dropna().copy()
        dd = dd.sort_values(metric)
        plt.figure(figsize=(12, max(5, 0.45 * len(dd))))
        plt.barh(dd["label"].astype(str), dd[metric].astype(float))
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel(metric)
        plt.ylabel("Ablation variant")
        plt.title(f"Ablation delta relative to full model: {metric}")
        save_current_fig_both(f"{ABLATION_FIGURE_PREFIX}_{metric}")
        plt.close()


def plot_ablation_3d_metric_surface(clean_df: pd.DataFrame):
    d = clean_df.copy().reset_index(drop=True)
    metric_cols = ["MAE", "Severe_AUROC", "Severe_AUPRC", "Onset_AUROC", "Onset_AUPRC", "Severe_ECE"]
    metric_cols = [c for c in metric_cols if c in d.columns and d[c].notna().sum() > 0]
    if len(d) == 0 or len(metric_cols) == 0:
        return

    Z = d[metric_cols].copy()
    # Put all metrics on comparable 0-1 visual scale. For MAE/ECE lower is better, invert.
    for c in metric_cols:
        vals = Z[c].astype(float).values
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        scaled = (vals - vmin) / (vmax - vmin + 1e-8)
        if c in ["MAE", "Severe_ECE", "Onset_ECE"]:
            scaled = 1.0 - scaled
        Z[c] = scaled
    arr = Z.values.astype(float)
    X, Y = np.meshgrid(np.arange(len(metric_cols)), np.arange(len(d)))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, arr)
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["label"].astype(str).tolist(), fontsize=7)
    ax.set_zlabel("Normalized performance, higher better")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Ablation variant")
    ax.set_title("3D ablation performance surface")
    save_current_fig_both(f"{ABLATION_FIGURE_PREFIX}_3d_metric_surface")
    plt.close()


def plot_ablation_3d_attention_surfaces(results: List[Dict[str, object]]):
    for res in results:
        spec = res["spec"]
        test_df = res["test_df"]
        prefix = res["prefix"]
        attn_cols = [c for c in test_df.columns if c.startswith(f"{prefix}_attn_bin_")]
        risk_col = f"{prefix}_pred_severe_prob_cal"
        if len(attn_cols) == 0 or risk_col not in test_df.columns:
            continue
        risk_bins = pd.qcut(test_df[risk_col], q=5, labels=False, duplicates="drop")
        surf = []
        y_labels = []
        for rb in sorted(pd.Series(risk_bins).dropna().unique()):
            m = risk_bins == rb
            if m.sum() == 0:
                continue
            surf.append([test_df.loc[m, c].mean() for c in attn_cols])
            y_labels.append(str(int(rb)))
        if len(surf) == 0:
            continue
        Z = np.asarray(surf, dtype=float)
        X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel("Input time bin")
        ax.set_ylabel("Severe-risk quantile")
        ax.set_zlabel("Attention weight")
        ax.set_title(f"3D attention-risk surface: {spec.label}")
        save_current_fig_both(f"{ABLATION_FIGURE_PREFIX}_3d_attention_risk_surface_{safe_name(spec.name)}")
        plt.close()


def make_ablation_combined_pdf():
    files = sorted(FIGURE_PNG_DIR.glob(f"{ABLATION_FIGURE_PREFIX}_*.png"))
    out = REPORTS_DIR / "final_ablation_combined_figures.pdf"
    if len(files) == 0:
        return None
    with PdfPages(out) as pdf:
        for p in files:
            img = plt.imread(p)
            plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis("off")
            plt.title(p.stem, fontsize=10)
            pdf.savefig(bbox_inches="tight")
            plt.close()
    print(f"[SAVED] {out}")
    return out


def run_full_ablation_study(
    temporal_paths: Dict[str, str],
    val_master: Optional[pd.DataFrame] = None,
    test_master: Optional[pd.DataFrame] = None,
    threshold_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Run full ablation experiments and save all results.

    Call inside main() AFTER:
        temporal_result = train_temporal_model(...)
        val_master, test_master = merge_predictions(...)
        val_master, test_master = train_fusion(...)
        threshold_df = tune_thresholds_on_validation(val_master)

    Recommended main() insertion:
        ablation_result = run_full_ablation_study(
            temporal_paths=temporal_paths,
            val_master=val_master,
            test_master=test_master,
            threshold_df=threshold_df,
        )
    """
    print("=" * 100)
    print("RUNNING FULL HEMATRACE ABLATION STUDY")
    print("=" * 100)

    specs = ablation_specs()
    save_df(pd.DataFrame([asdict(s) for s in specs]), ABLATION_TABLE_DIR / "final_ablation_manifest.csv")

    results = []
    metrics_parts = []
    for spec in specs:
        print(f"\n[ABLATION] {spec.name}: {spec.label}")
        res = train_one_ablation_model(temporal_paths, spec)
        results.append(res)
        metrics_parts.append(evaluate_single_ablation_df(res["val_df"], "validation", spec, res["prefix"]))
        metrics_parts.append(evaluate_single_ablation_df(res["test_df"], "test", spec, res["prefix"]))

    metrics_df = pd.concat(metrics_parts, ignore_index=True, sort=False)
    metrics_df = add_existing_main_models_to_ablation_metrics(metrics_df, threshold_df)
    save_df(metrics_df, ABLATION_TABLE_DIR / "final_ablation_all_model_metrics_long.csv")

    clean_test = compact_ablation_table(metrics_df, split="test")
    # Put clinically important rows first if present.
    preferred = [
        "Tabular", "HemaTRACE-Temporal", "HemaTRACE-OnsetSpecialized", "HemaTRACE-Fusion",
        "FullTemporal_ReRun", "NoAttention", "TemporalOnly_NoStatic", "StaticOnly_NoTemporal",
        "ValueOnly_NoMissingnessTimeCount", "NoTimeDelta", "NoObservationCount",
        "NoTransfusionFeatures", "NoHbTrendStaticFeatures",
    ]
    clean_test["_order"] = clean_test["model"].apply(lambda x: preferred.index(x) if x in preferred else len(preferred))
    clean_test = clean_test.sort_values(["_order", "model"]).drop(columns=["_order"])
    save_df(clean_test, ABLATION_TABLE_DIR / "final_ablation_clean_table_for_manuscript.csv")

    delta_df = compute_ablation_deltas(clean_test, reference_model="FullTemporal_ReRun")
    save_df(delta_df, ABLATION_TABLE_DIR / "final_ablation_delta_vs_full.csv")

    plot_ablation_metric_bars(clean_test)
    plot_ablation_delta_bars(delta_df)
    plot_ablation_3d_metric_surface(clean_test)
    plot_ablation_3d_attention_surfaces(results)
    ablation_pdf = make_ablation_combined_pdf()

    summary = {
        "n_ablation_models_trained": len(results),
        "ablation_models": [r["spec"].name for r in results],
        "tables": {
            "manifest": str(ABLATION_TABLE_DIR / "final_ablation_manifest.csv"),
            "long_metrics": str(ABLATION_TABLE_DIR / "final_ablation_all_model_metrics_long.csv"),
            "clean_manuscript_table": str(ABLATION_TABLE_DIR / "final_ablation_clean_table_for_manuscript.csv"),
            "delta_vs_full": str(ABLATION_TABLE_DIR / "final_ablation_delta_vs_full.csv"),
        },
        "figures_pdf": str(ablation_pdf) if ablation_pdf is not None else None,
        "models_dir": str(ABLATION_MODEL_DIR),
    }
    save_json(summary, REPORTS_DIR / "final_ablation_summary.json")
    print(json.dumps(summary, indent=4, default=json_converter))
    return {
        "results": results,
        "metrics_long": metrics_df,
        "clean_table": clean_test,
        "delta_table": delta_df,
        "summary": summary,
    }
