"""Held-out evaluation, confidence intervals, and subgroup analyses."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

from ..config import TABLES_DIR, cfg
from ..metrics import classification_metrics, regression_metrics, tune_threshold
from ..utils import save_df

def get_model_prediction_columns():
    return {
        "Tabular": (
            "tabular_pred_future_min_hb",
            "tabular_pred_severe_prob_cal",
            "tabular_pred_onset_prob_cal",
        ),
        "HemaTRACE-Temporal": (
            "temporal_pred_future_min_hb",
            "temporal_pred_severe_prob_cal",
            "temporal_pred_onset_prob_cal",
        ),
        "HemaTRACE-OnsetSpecialized": (
            "onset_specialized_pred_future_min_hb",
            "onset_specialized_pred_severe_prob",
            "onset_specialized_pred_onset_prob_cal",
        ),
        "HemaTRACE-Fusion": (
            "fusion_pred_future_min_hb",
            "fusion_pred_severe_prob",
            "fusion_pred_onset_prob",
        ),
    }


def tune_thresholds_on_validation(val_df: pd.DataFrame) -> pd.DataFrame:
    """Tune classification thresholds on validation only."""
    rows = []
    for model, (_, severe_col, onset_col) in get_model_prediction_columns().items():
        if severe_col in val_df.columns and val_df[severe_col].notna().sum() > 0:
            thr, score = tune_threshold(
                val_df["true_future_severe_anemia"].astype(int).values,
                val_df[severe_col].values,
            )
            rows.append({
                "model": model,
                "task": "severe",
                "threshold": float(thr),
                "validation_macro_f1": float(score),
            })

        if onset_col in val_df.columns and val_df[onset_col].notna().sum() > 0:
            mask = val_df["new_onset_mask"].astype(bool).values
            if mask.sum() > 0:
                thr, score = tune_threshold(
                    val_df.loc[mask, "true_new_onset_anemia"].astype(int).values,
                    val_df.loc[mask, onset_col].values,
                )
                rows.append({
                    "model": model,
                    "task": "onset",
                    "threshold": float(thr),
                    "validation_macro_f1": float(score),
                })
    return pd.DataFrame(rows)


def threshold_lookup(threshold_df: pd.DataFrame, model: str, task: str, default: float = 0.5) -> float:
    if threshold_df is None or threshold_df.empty:
        return float(default)
    m = threshold_df[(threshold_df["model"] == model) & (threshold_df["task"] == task)]
    if len(m) == 0:
        return float(default)
    return float(m.iloc[0]["threshold"])


def evaluate_master(df: pd.DataFrame, split: str, threshold_df: pd.DataFrame):
    """
    Evaluate models using thresholds selected on validation.

    AUROC/AUPRC are threshold-free. Accuracy/F1/sensitivity/specificity use
    validation-selected thresholds, including on the test split.
    """
    rows = []
    for model, (rcol, scol, ocol) in get_model_prediction_columns().items():
        if rcol in df.columns and df[rcol].notna().sum() > 0:
            m = regression_metrics(df["true_future_min_hb"].values, df[rcol].values)
            rows.append({"split": split, "model": model, "task": "future_min_hb_regression", **m})

        if scol in df.columns and df[scol].notna().sum() > 0:
            thr = threshold_lookup(threshold_df, model, "severe", default=0.5)
            m = classification_metrics(
                df["true_future_severe_anemia"].astype(int).values,
                df[scol].values,
                threshold=thr,
            )
            rows.append({"split": split, "model": model, "task": "future_severe_anemia_classification", **m})

        if ocol in df.columns and df[ocol].notna().sum() > 0:
            mask = df["new_onset_mask"].astype(bool).values
            if mask.sum() > 0:
                thr = threshold_lookup(threshold_df, model, "onset", default=0.5)
                m = classification_metrics(
                    df.loc[mask, "true_new_onset_anemia"].astype(int).values,
                    df.loc[mask, ocol].values,
                    threshold=thr,
                )
                rows.append({"split": split, "model": model, "task": "new_onset_anemia_classification", **m})

    return pd.DataFrame(rows)

def bootstrap_ci(df: pd.DataFrame, n_boot: int):
    if cfg.FAST_DEBUG: n_boot=cfg.FAST_DEBUG_BOOTSTRAP_N
    rng=np.random.default_rng(cfg.BOOTSTRAP_SEED)
    models={"Tabular":("tabular_pred_future_min_hb","tabular_pred_severe_prob_cal","tabular_pred_onset_prob_cal"),"HemaTRACE-Temporal":("temporal_pred_future_min_hb","temporal_pred_severe_prob_cal","temporal_pred_onset_prob_cal"),"HemaTRACE-OnsetSpecialized":("onset_specialized_pred_future_min_hb","onset_specialized_pred_severe_prob","onset_specialized_pred_onset_prob_cal"),"HemaTRACE-Fusion":("fusion_pred_future_min_hb","fusion_pred_severe_prob","fusion_pred_onset_prob")}
    rows=[]
    n=len(df); idx_all=np.arange(n)
    onset_idx=np.where(df.new_onset_mask.astype(bool).values)[0]
    for model,(rcol,scol,ocol) in models.items():
        vals={"MAE":[],"severe_AUROC":[],"severe_AUPRC":[],"onset_AUROC":[],"onset_AUPRC":[]}
        for b in range(n_boot):
            idx=rng.choice(idx_all, size=n, replace=True)
            if rcol in df.columns:
                vals["MAE"].append(mean_absolute_error(df.true_future_min_hb.values[idx], df[rcol].values[idx]))
            if scol in df.columns and len(np.unique(df.true_future_severe_anemia.values[idx]))>1:
                vals["severe_AUROC"].append(roc_auc_score(df.true_future_severe_anemia.values[idx], df[scol].values[idx]))
                vals["severe_AUPRC"].append(average_precision_score(df.true_future_severe_anemia.values[idx], df[scol].values[idx]))
            if ocol in df.columns and len(onset_idx)>10:
                oi=rng.choice(onset_idx, size=len(onset_idx), replace=True)
                if len(np.unique(df.true_new_onset_anemia.values[oi]))>1:
                    vals["onset_AUROC"].append(roc_auc_score(df.true_new_onset_anemia.values[oi], df[ocol].values[oi]))
                    vals["onset_AUPRC"].append(average_precision_score(df.true_new_onset_anemia.values[oi], df[ocol].values[oi]))
        for metric, arr in vals.items():
            arr=np.asarray(arr,dtype=float); arr=arr[np.isfinite(arr)]
            if len(arr): rows.append({"model":model,"metric":metric,"point":float(np.mean(arr)),"ci_low":float(np.quantile(arr,0.025)),"ci_high":float(np.quantile(arr,0.975)),"n_boot":len(arr)})
    out=pd.DataFrame(rows); save_df(out, TABLES_DIR / "final_bootstrap_ci.csv"); return out


def paired_tests(df: pd.DataFrame):
    pairs=[]; model_cols={"Tabular":("tabular_pred_future_min_hb","tabular_pred_severe_prob_cal","tabular_pred_onset_prob_cal"),"HemaTRACE-Temporal":("temporal_pred_future_min_hb","temporal_pred_severe_prob_cal","temporal_pred_onset_prob_cal"),"HemaTRACE-Fusion":("fusion_pred_future_min_hb","fusion_pred_severe_prob","fusion_pred_onset_prob")}
    rng=np.random.default_rng(cfg.BOOTSTRAP_SEED+99); n=len(df); idx_all=np.arange(n); B=cfg.FAST_DEBUG_BOOTSTRAP_N if cfg.FAST_DEBUG else cfg.BOOTSTRAP_N
    names=list(model_cols.keys())
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            a,b=names[i],names[j]
            ar,asv,aon=model_cols[a]; br,bsv,bon=model_cols[b]
            diffs=[]
            for _ in range(B):
                idx=rng.choice(idx_all,size=n,replace=True)
                diffs.append(mean_absolute_error(df.true_future_min_hb.values[idx], df[ar].values[idx])-mean_absolute_error(df.true_future_min_hb.values[idx], df[br].values[idx]))
            diffs=np.asarray(diffs); pairs.append({"task":"regression_MAE_diff", "model_a":a,"model_b":b,"diff_a_minus_b":float(np.mean(diffs)),"ci_low":float(np.quantile(diffs,0.025)),"ci_high":float(np.quantile(diffs,0.975)),"p_two_sided":float(2*min((diffs<=0).mean(),(diffs>=0).mean()))})
            # AUROC severe diff
            diffs=[]
            for _ in range(B):
                idx=rng.choice(idx_all,size=n,replace=True)
                if len(np.unique(df.true_future_severe_anemia.values[idx]))<2: continue
                diffs.append(roc_auc_score(df.true_future_severe_anemia.values[idx], df[asv].values[idx])-roc_auc_score(df.true_future_severe_anemia.values[idx], df[bsv].values[idx]))
            if diffs:
                diffs=np.asarray(diffs); pairs.append({"task":"severe_AUROC_diff", "model_a":a,"model_b":b,"diff_a_minus_b":float(np.mean(diffs)),"ci_low":float(np.quantile(diffs,0.025)),"ci_high":float(np.quantile(diffs,0.975)),"p_two_sided":float(2*min((diffs<=0).mean(),(diffs>=0).mean()))})
    out=pd.DataFrame(pairs); save_df(out, TABLES_DIR / "final_paired_bootstrap_tests.csv"); return out


def subgroup_metrics(df: pd.DataFrame):
    d=df.copy()
    d["age_group"]=pd.cut(d.get("age", pd.Series(np.nan,index=d.index)), bins=[17,50,65,80,200], labels=["18-50","51-65","66-80",">80"])
    d["baseline_hb_group"]=pd.cut(d.early_last_hb, bins=[-np.inf,8,10,12,14,np.inf], labels=["<=8","8-10","10-12","12-14",">14"])
    d["future_hb_lt8"]=np.where(d.true_future_min_hb<8,"Hb<8","Hb>=8")
    rows=[]
    for group_col in ["age_group","baseline_hb_group","future_hb_lt8"]:
        for gval,g in d.groupby(group_col, dropna=False):
            if len(g)<50: continue
            for model,rcol,scol,ocol in [("Tabular","tabular_pred_future_min_hb","tabular_pred_severe_prob_cal","tabular_pred_onset_prob_cal"),("HemaTRACE-Temporal","temporal_pred_future_min_hb","temporal_pred_severe_prob_cal","temporal_pred_onset_prob_cal"),("HemaTRACE-Fusion","fusion_pred_future_min_hb","fusion_pred_severe_prob","fusion_pred_onset_prob")]:
                if rcol in g.columns:
                    rows.append({"subgroup_variable":group_col,"subgroup":str(gval),"model":model,"task":"regression","N":len(g),"MAE":mean_absolute_error(g.true_future_min_hb,g[rcol])})
                if scol in g.columns and len(np.unique(g.true_future_severe_anemia))>1:
                    rows.append({"subgroup_variable":group_col,"subgroup":str(gval),"model":model,"task":"severe","N":len(g),"AUROC":roc_auc_score(g.true_future_severe_anemia,g[scol]),"AUPRC":average_precision_score(g.true_future_severe_anemia,g[scol])})
    out=pd.DataFrame(rows); save_df(out, TABLES_DIR / "final_subgroup_metrics.csv"); return out
