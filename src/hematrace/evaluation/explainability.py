"""Manuscript figures, occlusion analysis, and combined figure reports."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..config import FIGURE_PNG_DIR, REPORTS_DIR, XAI_DIR
from ..metrics import calibration_table
from ..utils import safe_name, save_current_fig_both, save_df

def plot_training_history(hist):
    for stem,tc,vc,title in [("total_loss","train_total_loss","val_total_loss","Total Loss"),("reg_loss","train_reg_loss","val_reg_loss","Regression Loss"),("severe_loss","train_severe_loss","val_severe_loss","Severe Loss"),("onset_loss","train_onset_loss","val_onset_loss","Onset Loss")]:
        plt.figure(figsize=(8,5)); plt.plot(hist.epoch,hist[tc],label="train"); plt.plot(hist.epoch,hist[vc],label="validation"); plt.xlabel("Epoch"); plt.ylabel(title); plt.title(f"Train vs Validation {title}"); plt.legend(); save_current_fig_both(f"final_train_validation_{stem}"); plt.close()
    for stem,cols,title in [("regression_metrics",["val_mae","val_rmse"],"Validation Regression Metrics"),("severe_auc_pr",["val_severe_auroc","val_severe_auprc"],"Validation Severe AUROC/AUPRC"),("onset_auc_pr",["val_onset_auroc","val_onset_auprc"],"Validation Onset AUROC/AUPRC")]:
        plt.figure(figsize=(8,5));
        for c in cols: plt.plot(hist.epoch,hist[c],label=c)
        plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.title(title); plt.legend(); save_current_fig_both(f"final_validation_{stem}"); plt.close()
    plt.figure(figsize=(8,5)); plt.plot(hist.epoch,hist.lr,label="learning rate"); plt.title("Learning Rate Curve"); plt.xlabel("Epoch"); plt.ylabel("LR"); plt.legend(); save_current_fig_both("final_learning_rate"); plt.close()


def plot_model_overlays(df):
    cls_specs=[("future_severe_anemia","true_future_severe_anemia",{"Tabular":"tabular_pred_severe_prob_cal","HemaTRACE-Temporal":"temporal_pred_severe_prob_cal","HemaTRACE-Fusion":"fusion_pred_severe_prob"}), ("new_onset_anemia","true_new_onset_anemia",{"Tabular":"tabular_pred_onset_prob_cal","HemaTRACE-Temporal":"temporal_pred_onset_prob_cal","HemaTRACE-OnsetSpecialized":"onset_specialized_pred_onset_prob_cal","HemaTRACE-Fusion":"fusion_pred_onset_prob"})]
    for task,ycol,models in cls_specs:
        mask=np.ones(len(df),dtype=bool) if task=="future_severe_anemia" else df.new_onset_mask.astype(bool).values
        y=df.loc[mask,ycol].astype(int).values
        plt.figure(figsize=(7,6))
        for name,col in models.items():
            if col in df.columns and df.loc[mask,col].notna().sum()>0 and len(np.unique(y))>1:
                fpr,tpr,_=roc_curve(y, df.loc[mask,col].values); auc=roc_auc_score(y,df.loc[mask,col].values); plt.plot(fpr,tpr,label=f"{name} AUROC={auc:.3f}")
        plt.plot([0,1],[0,1],"--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(f"ROC Overlay: {task}"); plt.legend(); save_current_fig_both(f"final_roc_overlay_{task}"); plt.close()
        plt.figure(figsize=(7,6))
        for name,col in models.items():
            if col in df.columns and df.loc[mask,col].notna().sum()>0 and len(np.unique(y))>1:
                pr,rc,_=precision_recall_curve(y, df.loc[mask,col].values); ap=average_precision_score(y,df.loc[mask,col].values); plt.plot(rc,pr,label=f"{name} AUPRC={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Overlay: {task}"); plt.legend(); save_current_fig_both(f"final_pr_overlay_{task}"); plt.close()
        plt.figure(figsize=(7,6)); plt.plot([0,1],[0,1],"--",label="perfect")
        for name,col in models.items():
            if col in df.columns and df.loc[mask,col].notna().sum()>0:
                tab=calibration_table(y,df.loc[mask,col].values); plt.plot(tab.mean_predicted_probability,tab.observed_rate,marker="o",label=name)
        plt.xlabel("Mean predicted probability"); plt.ylabel("Observed event rate"); plt.title(f"Calibration Overlay: {task}"); plt.legend(); save_current_fig_both(f"final_calibration_overlay_{task}"); plt.close()
    # regression scatter and Bland-Altman for fusion and temporal
    for name,col in {"Tabular":"tabular_pred_future_min_hb","HemaTRACE-Temporal":"temporal_pred_future_min_hb","HemaTRACE-Fusion":"fusion_pred_future_min_hb"}.items():
        if col not in df.columns: continue
        y=df.true_future_min_hb.values; p=df[col].values; res=p-y
        plt.figure(figsize=(6,6)); plt.scatter(y,p,s=5,alpha=0.25); mn=min(y.min(),p.min()); mx=max(y.max(),p.max()); plt.plot([mn,mx],[mn,mx],"--"); plt.xlabel("True future minimum Hb"); plt.ylabel("Predicted future minimum Hb"); plt.title(f"Regression Scatter: {name}"); save_current_fig_both(f"final_regression_scatter_{safe_name(name)}"); plt.close()
        plt.figure(figsize=(7,5)); plt.scatter(y,res,s=5,alpha=0.25); plt.axhline(0,linestyle="--"); plt.xlabel("True future minimum Hb"); plt.ylabel("Residual predicted - true"); plt.title(f"Residual Plot: {name}"); save_current_fig_both(f"final_residual_plot_{safe_name(name)}"); plt.close()
        mean_pair=(y+p)/2; plt.figure(figsize=(7,5)); plt.scatter(mean_pair,res,s=5,alpha=0.25); mr=res.mean(); sr=res.std(); plt.axhline(mr); plt.axhline(mr+1.96*sr,linestyle="--"); plt.axhline(mr-1.96*sr,linestyle="--"); plt.xlabel("Mean predicted and true Hb"); plt.ylabel("Predicted - true"); plt.title(f"Bland-Altman: {name}"); save_current_fig_both(f"final_bland_altman_{safe_name(name)}"); plt.close()


def plot_decision_curve(df, ycol, models, stem):
    thresholds=np.linspace(0.05,0.8,60); y=df[ycol].astype(int).values; N=len(y); rows=[]
    plt.figure(figsize=(7,5))
    for name,col in models.items():
        if col not in df.columns: continue
        nb=[]
        for pt in thresholds:
            pred=(df[col].values>=pt).astype(int); tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum(); net=tp/N - fp/N*(pt/(1-pt)); nb.append(net); rows.append({"model":name,"threshold":pt,"net_benefit":net})
        plt.plot(thresholds,nb,label=name)
    prev=y.mean(); treat_all=prev-(1-prev)*(thresholds/(1-thresholds)); plt.plot(thresholds,treat_all,"--",label="treat all"); plt.axhline(0,linestyle=":",label="treat none"); plt.xlabel("Threshold probability"); plt.ylabel("Net benefit"); plt.title(f"Decision Curve: {stem}"); plt.legend(); save_current_fig_both(f"final_decision_curve_{stem}"); plt.close(); return pd.DataFrame(rows)


def xai_occlusion(model_info, temporal_result, test_master):
    # Simple XAI from saved predictions and attention + optional occlusion with loaded model skipped if GPU memory tight.
    # Attention summary from temporal predictions columns.
    attn_cols=[c for c in test_master.columns if c.startswith("temporal_attention_bin_")]
    rows=[]
    for c in attn_cols:
        rows.append({"bin":c,"mean_attention":float(test_master[c].mean()),"severe0_attention":float(test_master.loc[test_master.true_future_severe_anemia==0,c].mean()),"severe1_attention":float(test_master.loc[test_master.true_future_severe_anemia==1,c].mean())})
    attn_df=pd.DataFrame(rows); save_df(attn_df, XAI_DIR / "final_attention_summary.csv")
    if len(attn_df):
        bins=["0-4h","4-8h","8-12h","12-16h","16-20h","20-24h"][:len(attn_df)]
        plt.figure(figsize=(7,4)); plt.bar(bins, attn_df.mean_attention); plt.xlabel("Input time bin"); plt.ylabel("Mean attention"); plt.title("HemaTRACE Mean Temporal Attention"); save_current_fig_both("final_xai_mean_temporal_attention"); plt.close()
        plt.figure(figsize=(7,4)); plt.plot(bins,attn_df.severe0_attention,marker="o",label="severe=0"); plt.plot(bins,attn_df.severe1_attention,marker="o",label="severe=1"); plt.xlabel("Input time bin"); plt.ylabel("Mean attention"); plt.title("Attention by Severe Anemia Class"); plt.legend(); save_current_fig_both("final_xai_attention_by_severe_class"); plt.close()
    # 3D attention-risk surface from attention bins and severe risk quantiles
    if len(attn_cols):
        risk_bins=pd.qcut(test_master.temporal_pred_severe_prob_cal, q=5, labels=False, duplicates="drop")
        surf=[]
        for rb in sorted(pd.Series(risk_bins).dropna().unique()):
            m=risk_bins==rb
            surf.append([test_master.loc[m,c].mean() for c in attn_cols])
        Z=np.asarray(surf); X,Y=np.meshgrid(np.arange(Z.shape[1]),np.arange(Z.shape[0]))
        fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection="3d"); ax.plot_surface(X,Y,Z); ax.set_xlabel("Time bin"); ax.set_ylabel("Risk quantile"); ax.set_zlabel("Attention"); ax.set_title("3D Attention-Risk Surface"); save_current_fig_both("final_xai_3d_attention_risk_surface"); plt.close()


def create_combined_pdf():
    files=sorted(FIGURE_PNG_DIR.glob("*.png")); out=REPORTS_DIR/"final_combined_all_figures.pdf"
    with PdfPages(out) as pdf:
        for p in files:
            img=plt.imread(p); plt.figure(figsize=(11,8.5)); plt.imshow(img); plt.axis("off"); plt.title(p.stem,fontsize=10); pdf.savefig(bbox_inches="tight"); plt.close()
    print(f"[SAVED] {out}"); return out
