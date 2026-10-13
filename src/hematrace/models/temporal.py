"""Original HemaTRACE recurrent temporal architecture and training."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from ..config import (
    DATA_DIR,
    LOGS_DIR,
    ONSET_MODEL_DIR,
    TABLES_DIR,
    TEMPORAL_MODEL_DIR,
    cfg,
)
from ..metrics import classification_metrics, regression_metrics
from ..utils import save_df, save_json

def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def compute_norm(train):
    x_value, x_mask, x_count = train["x_value"], train["x_mask"], train["x_count"]
    obs = x_mask.sum(axis=(0,1))
    mean = (x_value * x_mask).sum(axis=(0,1)) / np.maximum(obs, 1)
    var = (((x_value - mean.reshape(1,1,-1))**2) * x_mask).sum(axis=(0,1)) / np.maximum(obs, 1)
    std = np.sqrt(var); std[std < 1e-6] = 1.0
    cnt_log = np.log1p(x_count)
    cnt_mean = cnt_log.mean(axis=(0,1)); cnt_std = cnt_log.std(axis=(0,1)); cnt_std[cnt_std < 1e-6] = 1.0
    sm = train["static_mask"]; sv = train["static_value"]
    smean = (sv*sm).sum(axis=0)/np.maximum(sm.sum(axis=0),1)
    svar = (((sv-smean.reshape(1,-1))**2)*sm).sum(axis=0)/np.maximum(sm.sum(axis=0),1)
    sstd = np.sqrt(svar); sstd[sstd<1e-6]=1.0
    delta_raw = train["y_future_min_hb"] - train["early_last_hb"]
    dmean, dstd = np.nanmean(delta_raw).astype(np.float32), np.nanstd(delta_raw).astype(np.float32)
    if dstd < 1e-6: dstd = np.float32(1.0)
    norm = {"value_mean": mean.astype(np.float32), "value_std": std.astype(np.float32), "count_mean": cnt_mean.astype(np.float32), "count_std": cnt_std.astype(np.float32), "static_mean": smean.astype(np.float32), "static_std": sstd.astype(np.float32), "target_delta_mean": float(dmean), "target_delta_std": float(dstd)}
    save_json({k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in norm.items()}, LOGS_DIR / "final_temporal_normalization.json")
    return norm


def apply_norm(data, norm):
    out = dict(data)
    x = (data["x_value"] - norm["value_mean"].reshape(1,1,-1)) / norm["value_std"].reshape(1,1,-1)
    x = np.where(data["x_mask"] > 0, x, 0.0)
    out["x_value_norm"] = np.clip(x, -10, 10).astype(np.float32)
    out["x_delta_norm"] = np.clip(data["x_delta"] / 24.0, 0, 1).astype(np.float32)
    xc = (np.log1p(data["x_count"]) - norm["count_mean"].reshape(1,1,-1)) / norm["count_std"].reshape(1,1,-1)
    out["x_count_norm"] = np.clip(xc, -10, 10).astype(np.float32)
    s = (data["static_value"] - norm["static_mean"].reshape(1,-1)) / norm["static_std"].reshape(1,-1)
    s = np.where(data["static_mask"] > 0, s, 0.0)
    out["static_norm"] = np.clip(s, -10, 10).astype(np.float32)
    out["target_delta_z"] = ((data["y_future_min_hb"] - data["early_last_hb"]) - norm["target_delta_mean"]) / norm["target_delta_std"]
    return out


class TemporalDS(Dataset):
    def __init__(self, data, onset_only=False):
        self.data = data
        if onset_only:
            self.idx = np.where(data["new_onset_mask"] > 0)[0]
        else:
            self.idx = np.arange(len(data["y_future_min_hb"]))
    def __len__(self): return len(self.idx)
    def __getitem__(self, j):
        i = self.idx[j]
        d = self.data
        return {"x_value": torch.tensor(d["x_value_norm"][i], dtype=torch.float32), "x_mask": torch.tensor(d["x_mask"][i], dtype=torch.float32), "x_delta": torch.tensor(d["x_delta_norm"][i], dtype=torch.float32), "x_count": torch.tensor(d["x_count_norm"][i], dtype=torch.float32), "static_value": torch.tensor(d["static_norm"][i], dtype=torch.float32), "static_mask": torch.tensor(d["static_mask"][i], dtype=torch.float32), "base_hb": torch.tensor(d["early_last_hb"][i], dtype=torch.float32), "target_delta_z": torch.tensor(d["target_delta_z"][i], dtype=torch.float32), "y_reg": torch.tensor(d["y_future_min_hb"][i], dtype=torch.float32), "y_sev": torch.tensor(d["y_future_severe_anemia"][i], dtype=torch.float32), "y_on": torch.tensor(d["y_new_onset_anemia"][i], dtype=torch.float32), "on_mask": torch.tensor(d["new_onset_mask"][i], dtype=torch.float32)}


def make_loader(data, shuffle=False, onset_only=False):
    return DataLoader(TemporalDS(data, onset_only=onset_only), batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.NUM_WORKERS, pin_memory=torch.cuda.is_available())


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim,1))
    def forward(self, h):
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return (h*w.unsqueeze(-1)).sum(dim=1), w


class HemaTRACEModel(nn.Module):
    def __init__(self, n_temp, n_static):
        super().__init__()
        tin = n_temp * 4; sin = n_static * 2
        self.temporal_encoder = nn.Sequential(nn.Linear(tin, cfg.TEMPORAL_EMBED_DIM), nn.LayerNorm(cfg.TEMPORAL_EMBED_DIM), nn.GELU(), nn.Dropout(cfg.DROPOUT), nn.Linear(cfg.TEMPORAL_EMBED_DIM, cfg.TEMPORAL_EMBED_DIM), nn.LayerNorm(cfg.TEMPORAL_EMBED_DIM), nn.GELU(), nn.Dropout(cfg.DROPOUT))
        self.gru = nn.GRU(cfg.TEMPORAL_EMBED_DIM, cfg.GRU_HIDDEN_DIM, num_layers=cfg.GRU_LAYERS, batch_first=True, bidirectional=cfg.GRU_BIDIRECTIONAL)
        gout = cfg.GRU_HIDDEN_DIM * (2 if cfg.GRU_BIDIRECTIONAL else 1)
        self.attn = AttentionPooling(gout)
        self.static_encoder = nn.Sequential(nn.Linear(sin, cfg.STATIC_HIDDEN_DIM), nn.LayerNorm(cfg.STATIC_HIDDEN_DIM), nn.GELU(), nn.Dropout(cfg.DROPOUT), nn.Linear(cfg.STATIC_HIDDEN_DIM, cfg.STATIC_EMBED_DIM), nn.LayerNorm(cfg.STATIC_EMBED_DIM), nn.GELU(), nn.Dropout(cfg.DROPOUT))
        self.shared = nn.Sequential(nn.Linear(gout + cfg.STATIC_EMBED_DIM, cfg.SHARED_HIDDEN_DIM), nn.LayerNorm(cfg.SHARED_HIDDEN_DIM), nn.GELU(), nn.Dropout(cfg.DROPOUT), nn.Linear(cfg.SHARED_HIDDEN_DIM, cfg.SHARED_HIDDEN_DIM), nn.LayerNorm(cfg.SHARED_HIDDEN_DIM), nn.GELU(), nn.Dropout(cfg.DROPOUT))
        self.delta_head = nn.Linear(cfg.SHARED_HIDDEN_DIM, 1)
        self.severe_head = nn.Linear(cfg.SHARED_HIDDEN_DIM, 1)
        self.onset_head = nn.Linear(cfg.SHARED_HIDDEN_DIM, 1)
    def forward(self, xv, xm, xd, xc, sv, sm):
        x = torch.cat([xv,xm,xd,xc], dim=-1)
        x = self.temporal_encoder(x)
        h,_ = self.gru(x)
        pooled,w = self.attn(h)
        s = self.static_encoder(torch.cat([sv,sm], dim=-1))
        z = self.shared(torch.cat([pooled,s], dim=-1))
        return {"delta_z": self.delta_head(z).squeeze(-1), "severe_logit": self.severe_head(z).squeeze(-1), "onset_logit": self.onset_head(z).squeeze(-1), "attention": w}


def move_batch(b, device): return {k: v.to(device, non_blocking=True) for k,v in b.items()}


def compute_loss(batch, out, reg_fn, bce_fn):
    reg = reg_fn(out["delta_z"], batch["target_delta_z"])
    sev = bce_fn(out["severe_logit"], batch["y_sev"])
    on_raw = nn.functional.binary_cross_entropy_with_logits(out["onset_logit"], batch["y_on"], reduction="none")
    on = (on_raw * batch["on_mask"]).sum() / torch.clamp(batch["on_mask"].sum(), min=1.0)
    total = cfg.REG_LOSS_WEIGHT*reg + cfg.SEVERE_LOSS_WEIGHT*sev + cfg.ONSET_LOSS_WEIGHT*on
    return total, {"total": total.item(), "reg": reg.item(), "severe": sev.item(), "onset": on.item()}


def train_temporal_model(npz_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_raw, val_raw, test_raw = load_npz(npz_paths["train"]), load_npz(npz_paths["validation"]), load_npz(npz_paths["test"])
    norm = compute_norm(train_raw)
    train, val, test = apply_norm(train_raw, norm), apply_norm(val_raw, norm), apply_norm(test_raw, norm)
    tr_loader, val_loader, test_loader = make_loader(train, True), make_loader(val, False), make_loader(test, False)
    model = HemaTRACEModel(train["x_value"].shape[-1], train["static_value"].shape[-1]).to(device)
    reg_fn = nn.SmoothL1Loss(beta=0.5)
    bce_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, min_lr=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP and torch.cuda.is_available())
    amp = cfg.USE_AMP and torch.cuda.is_available()
    best = {"joint": -1e18, "regression": -1e18, "severe": -1e18, "onset": -1e18}
    best_epoch = {k: -1 for k in best}
    paths = {k: TEMPORAL_MODEL_DIR / f"best_{k}_model.pt" for k in best}
    hist=[]; bad=0
    for epoch in range(1, cfg.MAX_EPOCHS+1):
        model.train(); sums={"total":0,"reg":0,"severe":0,"onset":0}; nb=0
        for batch in tr_loader:
            batch=move_batch(batch, device); opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                out=model(batch["x_value"],batch["x_mask"],batch["x_delta"],batch["x_count"],batch["static_value"],batch["static_mask"])
                loss, parts=compute_loss(batch,out,reg_fn,bce_fn)
            if amp:
                scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP); scaler.step(opt); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP); opt.step()
            for k in sums: sums[k]+=parts[k]
            nb+=1
        pred_val = predict_temporal(model, val_loader, device, norm, with_loss=(reg_fn,bce_fn))
        m = all_metrics_from_pred(pred_val)
        joint = m["severe"]["AUROC"] + cfg.SCORE_ONSET_WEIGHT*m["onset"]["AUROC"] - cfg.SCORE_MAE_WEIGHT*m["regression"]["MAE"]
        scores = {"joint": joint, "regression": -m["regression"]["MAE"], "severe": m["severe"]["AUROC"], "onset": m["onset"]["AUROC"]}
        row={"epoch":epoch,"lr":opt.param_groups[0]["lr"],"train_total_loss":sums["total"]/nb,"train_reg_loss":sums["reg"]/nb,"train_severe_loss":sums["severe"]/nb,"train_onset_loss":sums["onset"]/nb,"val_total_loss":pred_val["loss_total"],"val_reg_loss":pred_val["loss_reg"],"val_severe_loss":pred_val["loss_severe"],"val_onset_loss":pred_val["loss_onset"],"val_mae":m["regression"]["MAE"],"val_rmse":m["regression"]["RMSE"],"val_r2":m["regression"]["R2"],"val_severe_auroc":m["severe"]["AUROC"],"val_severe_auprc":m["severe"]["AUPRC"],"val_severe_accuracy":m["severe"]["Accuracy"],"val_severe_macro_f1":m["severe"]["Macro_F1"],"val_onset_auroc":m["onset"]["AUROC"],"val_onset_auprc":m["onset"]["AUPRC"],"val_onset_accuracy":m["onset"]["Accuracy"],"val_onset_macro_f1":m["onset"]["Macro_F1"],"score_joint":scores["joint"],"score_regression":scores["regression"],"score_severe":scores["severe"],"score_onset":scores["onset"]}
        hist.append(row); print(f"Epoch {epoch}: val MAE={row['val_mae']:.4f} severeAUROC={row['val_severe_auroc']:.4f} onsetAUROC={row['val_onset_auroc']:.4f} joint={joint:.4f}")
        improved=False
        for k,s in scores.items():
            if s>best[k]:
                best[k]=s; best_epoch[k]=epoch; improved=True
                torch.save({"model_state_dict":model.state_dict(),"norm":norm,"cfg":asdict(cfg),"epoch":epoch,"score":s}, paths[k])
        if improved and scores["joint"]>=best["joint"]: bad=0
        else: bad+=1
        sched.step(scores["joint"])
        if bad>=cfg.PATIENCE: break
    hist_df=pd.DataFrame(hist); save_df(hist_df, TABLES_DIR / "final_temporal_epoch_history.csv")
    # load joint and make predictions
    ckpt=torch.load(paths["joint"], map_location=device); model.load_state_dict(ckpt["model_state_dict"])
    val_pred=predict_temporal(model,val_loader,device,norm); test_pred=predict_temporal(model,test_loader,device,norm)
    val_df=temporal_pred_df(val_raw,val_pred,"temporal")
    test_df=temporal_pred_df(test_raw,test_pred,"temporal")
    save_df(val_df, DATA_DIR / "temporal_predictions_validation.csv")
    save_df(test_df, DATA_DIR / "temporal_predictions_test.csv")
    # onset fine-tune from best onset checkpoint
    onset_path, onset_val_df, onset_test_df, onset_hist = finetune_onset(model, paths["onset"], norm, train, val, test, train_raw, val_raw, test_raw, device)
    return {"model": model, "norm": norm, "paths": paths, "best_scores": best, "best_epochs": best_epoch, "history": hist_df, "val_pred_df": val_df, "test_pred_df": test_df, "onset_val_df": onset_val_df, "onset_test_df": onset_test_df, "train_data": train, "val_data": val, "test_data": test}


@torch.no_grad()
def predict_temporal(model, loader, device, norm, with_loss=None):
    model.eval(); outs=[]; losses={"total":0,"reg":0,"severe":0,"onset":0}; nb=0
    reg_fn,bce_fn = with_loss if with_loss is not None else (None,None)
    for batch in loader:
        batch=move_batch(batch, device)
        out=model(batch["x_value"],batch["x_mask"],batch["x_delta"],batch["x_count"],batch["static_value"],batch["static_mask"])
        if with_loss is not None:
            _, parts=compute_loss(batch,out,reg_fn,bce_fn)
            for k in losses: losses[k]+=parts[k]
            nb+=1
        delta = out["delta_z"].detach().cpu().numpy()*norm["target_delta_std"]+norm["target_delta_mean"]
        base = batch["base_hb"].detach().cpu().numpy()
        outs.append({"pred_hb":base+delta,"pred_delta":delta,"sev_prob":torch.sigmoid(out["severe_logit"]).detach().cpu().numpy(),"on_prob":torch.sigmoid(out["onset_logit"]).detach().cpu().numpy(),"y_hb":batch["y_reg"].detach().cpu().numpy(),"y_sev":batch["y_sev"].detach().cpu().numpy(),"y_on":batch["y_on"].detach().cpu().numpy(),"on_mask":batch["on_mask"].detach().cpu().numpy(),"base":base,"attn":out["attention"].detach().cpu().numpy()})
    pred={k:np.concatenate([o[k] for o in outs]) for k in outs[0]}
    if with_loss is not None:
        pred.update({f"loss_{k}":v/max(nb,1) for k,v in losses.items()})
    return pred


def all_metrics_from_pred(pred):
    reg=regression_metrics(pred["y_hb"], pred["pred_hb"])
    sev=classification_metrics(pred["y_sev"].astype(int), pred["sev_prob"], 0.5)
    m=pred["on_mask"]>0
    onset=classification_metrics(pred["y_on"][m].astype(int), pred["on_prob"][m], 0.5) if m.sum()>0 else {"AUROC":np.nan,"AUPRC":np.nan,"Accuracy":np.nan,"Macro_F1":np.nan,"N":0}
    return {"regression":reg,"severe":sev,"onset":onset}


def temporal_pred_df(raw, pred, prefix):
    return pd.DataFrame({"subject_id":raw["subject_id"],"hadm_id":raw["hadm_id"],"stay_id":raw["stay_id"],"true_future_min_hb":pred["y_hb"],"true_future_severe_anemia":pred["y_sev"].astype(int),"true_new_onset_anemia":pred["y_on"].astype(int),"new_onset_mask":pred["on_mask"].astype(int),"early_last_hb":raw["early_last_hb"],"early_min_hb":raw["early_min_hb"],"hb_slope":raw["hb_slope"],"hb_margin_to_8":raw["hb_margin_to_8"],"hb_margin_to_sex_threshold":raw["hb_margin_to_sex_threshold"],f"{prefix}_pred_future_min_hb":pred["pred_hb"],f"{prefix}_pred_delta_hb":pred["pred_delta"],f"{prefix}_pred_severe_prob":pred["sev_prob"],f"{prefix}_pred_onset_prob":pred["on_prob"],**{f"{prefix}_attention_bin_{i}":pred["attn"][:,i] for i in range(pred["attn"].shape[1])}})


def finetune_onset(model, onset_ckpt_path, norm, train, val, test, train_raw, val_raw, test_raw, device):
    ckpt=torch.load(onset_ckpt_path, map_location=device); model.load_state_dict(ckpt["model_state_dict"])
    for p in model.temporal_encoder.parameters(): p.requires_grad=False
    for p in model.gru.parameters(): p.requires_grad=False
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.ONSET_FINETUNE_LR, weight_decay=cfg.WEIGHT_DECAY)
    train_loader=make_loader(train, True, onset_only=True); val_loader=make_loader(val, False); test_loader=make_loader(test, False)
    best=-1; bad=0; hist=[]; path=ONSET_MODEL_DIR/"best_onset_finetuned_model.pt"
    bce=nn.BCEWithLogitsLoss()
    for epoch in range(1,cfg.ONSET_FINETUNE_EPOCHS+1):
        model.train(); loss_sum=0; nb=0
        for batch in train_loader:
            batch=move_batch(batch, device); opt.zero_grad(set_to_none=True)
            out=model(batch["x_value"],batch["x_mask"],batch["x_delta"],batch["x_count"],batch["static_value"],batch["static_mask"])
            loss=bce(out["onset_logit"], batch["y_on"])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP); opt.step()
            loss_sum+=loss.item(); nb+=1
        pred_val=predict_temporal(model,val_loader,device,norm); m=pred_val["on_mask"]>0
        auc=roc_auc_score(pred_val["y_on"][m], pred_val["on_prob"][m]) if m.sum()>0 else np.nan
        hist.append({"epoch":epoch,"train_onset_loss":loss_sum/max(nb,1),"val_onset_auroc":auc})
        if auc>best:
            best=auc; bad=0; torch.save({"model_state_dict":model.state_dict(),"norm":norm,"epoch":epoch,"score":auc}, path)
        else: bad+=1
        if bad>=cfg.ONSET_FINETUNE_PATIENCE: break
    save_df(pd.DataFrame(hist), TABLES_DIR / "final_onset_finetune_history.csv")
    ckpt=torch.load(path,map_location=device); model.load_state_dict(ckpt["model_state_dict"])
    val_pred=predict_temporal(model,val_loader,device,norm); test_pred=predict_temporal(model,test_loader,device,norm)
    val_df=temporal_pred_df(val_raw,val_pred,"onset_specialized")
    test_df=temporal_pred_df(test_raw,test_pred,"onset_specialized")
    save_df(val_df, DATA_DIR / "onset_specialized_predictions_validation.csv")
    save_df(test_df, DATA_DIR / "onset_specialized_predictions_test.csv")
    return path, val_df, test_df, pd.DataFrame(hist)
