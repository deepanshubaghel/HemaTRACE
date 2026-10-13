"""Temperature scaling, prediction merging, and model fusion."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import DATA_DIR, FUSION_MODEL_DIR, TABLES_DIR
from ..metrics import clean_prob, logit_np, sigmoid_np
from ..utils import save_df, save_pickle

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__(); self.log_temp=nn.Parameter(torch.zeros(1))
    def forward(self, logits):
        return logits / torch.exp(self.log_temp).clamp(0.05, 20.0)


def fit_temperature(y, prob):
    y=torch.tensor(np.asarray(y).astype(np.float32))
    logits=torch.tensor(logit_np(prob).astype(np.float32))
    scaler=TemperatureScaler()
    opt=torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=100)
    bce=nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad(); loss=bce(scaler(logits), y); loss.backward(); return loss
    try: opt.step(closure)
    except Exception: pass
    temp=float(torch.exp(scaler.log_temp).detach().cpu().item())
    cal=sigmoid_np(logit_np(prob)/temp)
    return temp, cal


def merge_predictions(tab_val, tab_test, temp_val, temp_test, onset_val, onset_test):
    """
    Merge tabular, temporal, and onset-specialized predictions.

    Fix:
        Pandas can raise LossySetitemError / invalid value for dtype float32
        when assigning calibrated float64 arrays into an existing float32
        probability column using .loc.

        Therefore calibrated columns are created as fresh float64 numpy arrays,
        updated in numpy, and then assigned once to the dataframe.
    """

    keys = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "true_future_min_hb",
        "true_future_severe_anemia",
        "true_new_onset_anemia",
        "new_onset_mask",
        "early_last_hb",
        "early_min_hb",
        "hb_slope",
        "hb_margin_to_8",
        "hb_margin_to_sex_threshold",
    ]

    val = tab_val.merge(
        temp_val.drop(
            columns=[
                c for c in keys
                if c in temp_val.columns and c not in ["subject_id", "hadm_id", "stay_id"]
            ]
        ),
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
    )

    val = val.merge(
        onset_val[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "onset_specialized_pred_onset_prob",
                "onset_specialized_pred_severe_prob",
                "onset_specialized_pred_future_min_hb",
            ]
        ],
        on=["subject_id", "hadm_id", "stay_id"],
        how="left",
    )

    test = tab_test.merge(
        temp_test.drop(
            columns=[
                c for c in keys
                if c in temp_test.columns and c not in ["subject_id", "hadm_id", "stay_id"]
            ]
        ),
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
    )

    test = test.merge(
        onset_test[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "onset_specialized_pred_onset_prob",
                "onset_specialized_pred_severe_prob",
                "onset_specialized_pred_future_min_hb",
            ]
        ],
        on=["subject_id", "hadm_id", "stay_id"],
        how="left",
    )

    calibration_rows = []

    calibration_jobs = [
        ("tabular_pred_severe_prob", "true_future_severe_anemia", None),
        ("temporal_pred_severe_prob", "true_future_severe_anemia", None),
        ("tabular_pred_onset_prob", "true_new_onset_anemia", "new_onset_mask"),
        ("temporal_pred_onset_prob", "true_new_onset_anemia", "new_onset_mask"),
        ("onset_specialized_pred_onset_prob", "true_new_onset_anemia", "new_onset_mask"),
    ]

    for prob_col, target_col, mask_col in calibration_jobs:
        if prob_col not in val.columns:
            continue

        if target_col not in val.columns:
            continue

        # Validation calibration mask.
        if mask_col is None:
            m_val = np.ones(len(val), dtype=bool)
        else:
            m_val = val[mask_col].fillna(0).values.astype(bool)

        # Test calibration mask.
        if mask_col is None:
            m_test = np.ones(len(test), dtype=bool)
        else:
            m_test = test[mask_col].fillna(0).values.astype(bool)

        # Base calibrated arrays are fresh float64 arrays.
        val_base = clean_prob(val[prob_col].values).astype(np.float64)
        test_base = clean_prob(test[prob_col].values).astype(np.float64)

        val_cal_full = val_base.copy()
        test_cal_full = test_base.copy()

        # Fit temperature only on valid validation rows with finite probabilities and labels.
        fit_mask = (
            m_val
            & np.isfinite(val_base)
            & np.isfinite(pd.to_numeric(val[target_col], errors="coerce").values)
        )

        # Need both classes for meaningful temperature calibration.
        y_fit = pd.to_numeric(val.loc[fit_mask, target_col], errors="coerce").values.astype(int)
        p_fit = val_base[fit_mask]

        if fit_mask.sum() >= 10 and len(np.unique(y_fit)) == 2:
            temp, val_cal = fit_temperature(y_fit, p_fit)

            val_cal_full[fit_mask] = clean_prob(val_cal).astype(np.float64)

            test_apply_mask = m_test & np.isfinite(test_base)
            test_cal_full[test_apply_mask] = clean_prob(
                sigmoid_np(logit_np(test_base[test_apply_mask]) / temp)
            ).astype(np.float64)
        else:
            temp = 1.0

        # Assign once. This avoids pandas float32 in-place assignment errors.
        val[prob_col + "_cal"] = val_cal_full.astype(np.float64)
        test[prob_col + "_cal"] = test_cal_full.astype(np.float64)

        calibration_rows.append(
            {
                "probability_column": prob_col,
                "target_column": target_col,
                "mask_column": mask_col if mask_col is not None else "",
                "temperature": float(temp),
                "n_validation_used": int(fit_mask.sum()),
            }
        )

    if len(calibration_rows) > 0:
        save_df(
            pd.DataFrame(calibration_rows),
            TABLES_DIR / "final_temperature_calibration_summary.csv",
        )

    save_df(val, DATA_DIR / "final_master_predictions_validation.csv")
    save_df(test, DATA_DIR / "final_master_predictions_test.csv")

    return val, test


def train_fusion(val, test):
    # regression fusion
    reg_feats=["tabular_pred_future_min_hb","temporal_pred_future_min_hb","early_last_hb","early_min_hb","hb_slope"]
    reg_feats=[c for c in reg_feats if c in val.columns]
    reg=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("ridge",Ridge(alpha=1.0))])
    reg.fit(val[reg_feats], val["true_future_min_hb"])
    val["fusion_pred_future_min_hb"]=reg.predict(val[reg_feats]); test["fusion_pred_future_min_hb"]=reg.predict(test[reg_feats])
    save_pickle(reg, FUSION_MODEL_DIR / "fusion_regression_ridge.pkl")
    # severe fusion
    sev_feats=["tabular_pred_severe_prob_cal","temporal_pred_severe_prob_cal","temporal_pred_future_min_hb","early_last_hb","hb_margin_to_8"]
    sev_feats=[c for c in sev_feats if c in val.columns]
    sev=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("logit",LogisticRegression(max_iter=1000))])
    sev.fit(val[sev_feats], val["true_future_severe_anemia"].astype(int))
    val["fusion_pred_severe_prob"]=sev.predict_proba(val[sev_feats])[:,1]; test["fusion_pred_severe_prob"]=sev.predict_proba(test[sev_feats])[:,1]
    save_pickle(sev, FUSION_MODEL_DIR / "fusion_severe_logistic.pkl")
    # onset fusion
    on_feats=["tabular_pred_onset_prob_cal","temporal_pred_onset_prob_cal","onset_specialized_pred_onset_prob_cal","hb_margin_to_sex_threshold","hb_slope"]
    on_feats=[c for c in on_feats if c in val.columns]
    mv=val.new_onset_mask.astype(bool); mt=test.new_onset_mask.astype(bool)
    on=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("logit",LogisticRegression(max_iter=1000))])
    on.fit(val.loc[mv,on_feats], val.loc[mv,"true_new_onset_anemia"].astype(int))
    val["fusion_pred_onset_prob"]=np.nan; test["fusion_pred_onset_prob"]=np.nan
    val.loc[mv,"fusion_pred_onset_prob"]=on.predict_proba(val.loc[mv,on_feats])[:,1]
    test.loc[mt,"fusion_pred_onset_prob"]=on.predict_proba(test.loc[mt,on_feats])[:,1]
    save_pickle(on, FUSION_MODEL_DIR / "fusion_onset_logistic.pkl")
    coef_rows=[]
    for name, model, feats in [("regression",reg,reg_feats),("severe",sev,sev_feats),("onset",on,on_feats)]:
        est=model[-1]
        coef=np.ravel(est.coef_) if hasattr(est,"coef_") else np.ravel(est.coef_ if hasattr(est,"coef_") else [])
        if len(coef)==len(feats):
            for f,c in zip(feats,coef): coef_rows.append({"fusion_task":name,"feature":f,"coefficient":float(c)})
    save_df(pd.DataFrame(coef_rows), TABLES_DIR / "final_fusion_coefficients.csv")
    save_df(val, DATA_DIR / "final_master_predictions_validation.csv")
    save_df(test, DATA_DIR / "final_master_predictions_test.csv")
    return val, test
