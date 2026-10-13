"""Feature engineering, future Hb targets, subject splits, temporal tensors."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import DATA_DIR, TABLES_DIR, cfg
from ..data.item_maps import LAB_VARIABLES, TEMPORAL_FEATURES, TREAT_VARIABLES, VITAL_VARIABLES
from ..utils import safe_numeric, save_df

def aggregate_events(events: pd.DataFrame, prefix: str, features: List[str]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for (stay_id, feat), g in events.groupby(["stay_id", "feature"]):
        vals = pd.to_numeric(g["value"], errors="coerce").dropna()
        if len(vals) == 0: continue
        gg = g.loc[vals.index].sort_values("event_hour")
        hours = gg["event_hour"].values.astype(float)
        vv = gg["value"].values.astype(float)
        slope = 0.0
        if len(vv) >= 2 and np.nanstd(hours) > 1e-6:
            try:
                slope = float(np.polyfit(hours, vv, 1)[0])
            except Exception:
                slope = 0.0
        rows.append({
            "stay_id": stay_id,
            f"{prefix}_{feat}_count": len(vv),
            f"{prefix}_{feat}_mean": float(np.nanmean(vv)),
            f"{prefix}_{feat}_std": float(np.nanstd(vv)) if len(vv) > 1 else 0.0,
            f"{prefix}_{feat}_min": float(np.nanmin(vv)),
            f"{prefix}_{feat}_max": float(np.nanmax(vv)),
            f"{prefix}_{feat}_first": float(vv[0]),
            f"{prefix}_{feat}_last": float(vv[-1]),
            f"{prefix}_{feat}_slope_per_hour": slope,
        })
    wide = pd.DataFrame(rows)
    if wide.empty:
        return pd.DataFrame({"stay_id": []})
    wide = wide.groupby("stay_id", as_index=False).first()
    for feat in features:
        for stat in ["count", "mean", "std", "min", "max", "first", "last", "slope_per_hour"]:
            col = f"{prefix}_{feat}_{stat}"
            if col not in wide.columns:
                wide[col] = np.nan
        wide[f"{prefix}_{feat}_missing"] = wide[f"{prefix}_{feat}_count"].isna().astype(int)
    return wide


def build_targets_and_features(cohort: pd.DataFrame, lab_path: Path, vital_path: Path, input_path: Path):
    labs_all = pd.read_csv(lab_path, low_memory=False)
    labs_all["event_hour"] = safe_numeric(labs_all["event_hour"])
    labs_all["value"] = safe_numeric(labs_all["value"])
    labs_early = labs_all[(labs_all.event_hour >= 0) & (labs_all.event_hour < cfg.INPUT_END_HOURS)].copy()
    labs_future_hb = labs_all[(labs_all.feature == "lab_hemoglobin") & (labs_all.event_hour >= cfg.TARGET_START_HOURS) & (labs_all.event_hour < cfg.TARGET_END_HOURS)].copy()

    # target stats
    target_rows = []
    early_hb_rows = []
    for stay_id, g in labs_future_hb.groupby("stay_id"):
        g = g.dropna(subset=["value"]).sort_values("event_hour")
        if g.empty: continue
        vals = g.value.values.astype(float)
        hours = g.event_hour.values.astype(float)
        slope = float(np.polyfit(hours, vals, 1)[0]) if len(vals) >= 2 and np.std(hours) > 1e-6 else 0.0
        target_rows.append({
            "stay_id": int(stay_id),
            "future_hb_count_24_72h": len(vals),
            "future_mean_hb_24_72h": float(np.mean(vals)),
            "future_std_hb_24_72h": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "future_min_hb_24_72h": float(np.min(vals)),
            "future_max_hb_24_72h": float(np.max(vals)),
            "future_first_hb_24_72h": float(vals[0]),
            "future_last_hb_24_72h": float(vals[-1]),
            "future_hb_slope_24_72h": slope,
        })
    hb_early = labs_early[labs_early.feature == "lab_hemoglobin"].copy()
    for stay_id, g in hb_early.groupby("stay_id"):
        g = g.dropna(subset=["value"]).sort_values("event_hour")
        if g.empty: continue
        vals = g.value.values.astype(float)
        hours = g.event_hour.values.astype(float)
        slope = float(np.polyfit(hours, vals, 1)[0]) if len(vals) >= 2 and np.std(hours) > 1e-6 else 0.0
        early_hb_rows.append({
            "stay_id": int(stay_id),
            "early_last_hb_0_24h": float(vals[-1]),
            "early_min_hb_0_24h": float(np.min(vals)),
            "lab0_24h_hemoglobin_slope_per_hour": slope,
        })
    targets = pd.DataFrame(target_rows)
    early_hb = pd.DataFrame(early_hb_rows)

    df = cohort.merge(targets, on="stay_id", how="inner").merge(early_hb, on="stay_id", how="left")
    df["sex_specific_anemia_threshold"] = np.where(df["gender"].astype(str).str.upper().str.startswith("M"), cfg.MALE_ANEMIA_THRESHOLD, cfg.FEMALE_ANEMIA_THRESHOLD)
    df["future_severe_anemia_24_72h"] = (df["future_min_hb_24_72h"] < cfg.SEVERE_HB_THRESHOLD).astype(int)
    df["future_anemia_24_72h"] = (df["future_min_hb_24_72h"] < df["sex_specific_anemia_threshold"]).astype(int)
    df["future_hb_drop_24_72h"] = df["early_last_hb_0_24h"] - df["future_min_hb_24_72h"]

    # New-onset anemia is only defined when early 0-24h Hb is observed.
    # Missing early Hb must not be treated as "not anemic".
    df["early_anemia_status_0_24h"] = np.nan
    has_early_hb = df["early_min_hb_0_24h"].notna()
    df.loc[has_early_hb, "early_anemia_status_0_24h"] = (
        df.loc[has_early_hb, "early_min_hb_0_24h"]
        < df.loc[has_early_hb, "sex_specific_anemia_threshold"]
    ).astype(float)
    eligible = df["early_anemia_status_0_24h"] == 0
    df["new_onset_anemia_24_72h"] = np.where(eligible, df["future_anemia_24_72h"], np.nan)

    df["hb_margin_to_8"] = df["early_last_hb_0_24h"] - cfg.SEVERE_HB_THRESHOLD
    df["hb_margin_to_sex_threshold"] = df["early_last_hb_0_24h"] - df["sex_specific_anemia_threshold"]
    df["hb_range_0_24h_last_minus_min"] = df["early_last_hb_0_24h"] - df["early_min_hb_0_24h"]

    # tabular aggregate features
    lab_agg = aggregate_events(labs_early, "lab0_24h", [f"lab_{x}" for x in LAB_VARIABLES])
    vitals = pd.read_csv(vital_path, low_memory=False)
    vital_agg = aggregate_events(vitals, "vital0_24h", [f"vital_{x}" for x in VITAL_VARIABLES])
    inputs = pd.read_csv(input_path, low_memory=False)
    treat_agg = aggregate_events(inputs, "treat0_24h", [f"treat_{x}" for x in TREAT_VARIABLES])

    for w in [lab_agg, vital_agg, treat_agg]:
        if not w.empty:
            df = df.merge(w, on="stay_id", how="left")

    # clean high-level static categorical features
    keep_static = ["subject_id", "hadm_id", "stay_id", "gender", "age", "race", "insurance", "admission_type", "admission_location", "first_careunit", "intime"]
    keep_static = [c for c in keep_static if c in df.columns]

    flow = pd.DataFrame([
        {"stage": "adult_icu_los_ge_24h", "n": len(cohort), "subjects": cohort.subject_id.nunique()},
        {"stage": "future_hb_available_24_72h", "n": len(df), "subjects": df.subject_id.nunique()},
    ])
    save_df(flow, TABLES_DIR / "final_cohort_flow.csv")
    save_df(df, DATA_DIR / "final_cohort_all.csv")
    return df


def split_subject_level(df: pd.DataFrame) -> pd.DataFrame:
    subjects = df["subject_id"].drop_duplicates().values
    train_sub, temp_sub = train_test_split(subjects, test_size=(1 - cfg.TRAIN_FRAC), random_state=cfg.RANDOM_STATE)
    val_frac_temp = cfg.VAL_FRAC / (cfg.VAL_FRAC + cfg.TEST_FRAC)
    val_sub, test_sub = train_test_split(temp_sub, test_size=(1 - val_frac_temp), random_state=cfg.RANDOM_STATE)
    split = pd.Series("", index=df.index)
    split[df.subject_id.isin(train_sub)] = "train"
    split[df.subject_id.isin(val_sub)] = "validation"
    split[df.subject_id.isin(test_sub)] = "test"
    out = df.copy()
    out["split"] = split.values
    # verify overlap
    rows = []
    for s, g in out.groupby("split"):
        rows.append({
            "split": s,
            "n_rows": len(g),
            "n_subjects": g.subject_id.nunique(),
            "future_severe_rate": g.future_severe_anemia_24_72h.mean(),
            "new_onset_n": int(g.new_onset_anemia_24_72h.notna().sum()),
            "new_onset_rate": float(g.new_onset_anemia_24_72h.mean()),
            "mean_future_min_hb": float(g.future_min_hb_24_72h.mean()),
        })
    save_df(pd.DataFrame(rows), TABLES_DIR / "final_split_summary.csv")
    for s in ["train", "validation", "test"]:
        save_df(out[out.split == s].copy(), DATA_DIR / f"final_{s}.csv")
    return out


def build_temporal_npz(df: pd.DataFrame, lab_path: Path, vital_path: Path, input_path: Path):
    df = df.reset_index(drop=True).copy()
    df["row_idx"] = np.arange(len(df), dtype=np.int64)
    stay_to_row = dict(zip(df.stay_id.astype(int), df.row_idx.astype(int)))
    n, T, F = len(df), cfg.N_BINS, len(TEMPORAL_FEATURES)
    f_to_idx = {f: i for i, f in enumerate(TEMPORAL_FEATURES)}
    x_sum = np.zeros((n, T, F), dtype=np.float32)
    x_count = np.zeros((n, T, F), dtype=np.float32)
    last_hour = np.full((n, T, F), -1.0, dtype=np.float32)

    def update_from_df(ev: pd.DataFrame):
        if ev.empty: return
        ev = ev.copy()
        ev["row_idx"] = ev["stay_id"].astype(int).map(stay_to_row)
        ev["feature_idx"] = ev["feature"].map(f_to_idx)
        ev = ev.dropna(subset=["row_idx", "feature_idx", "event_hour", "value"])
        ev = ev[(ev.event_hour >= 0) & (ev.event_hour < cfg.INPUT_END_HOURS)].copy()
        if ev.empty: return
        ev["bin_idx"] = np.floor(ev.event_hour / cfg.BIN_SIZE_HOURS).astype(int)
        ev = ev[(ev.bin_idx >= 0) & (ev.bin_idx < T)]
        grouped = ev.groupby(["row_idx", "bin_idx", "feature_idx"], as_index=False).agg(value_sum=("value", "sum"), value_count=("value", "count"), event_hour_max=("event_hour", "max"))
        r = grouped.row_idx.values.astype(np.int64)
        b = grouped.bin_idx.values.astype(np.int64)
        f = grouped.feature_idx.values.astype(np.int64)
        np.add.at(x_sum, (r, b, f), grouped.value_sum.values.astype(np.float32))
        np.add.at(x_count, (r, b, f), grouped.value_count.values.astype(np.float32))
        np.maximum.at(last_hour, (r, b, f), grouped.event_hour_max.values.astype(np.float32))

    update_from_df(pd.read_csv(lab_path, low_memory=False))
    update_from_df(pd.read_csv(vital_path, low_memory=False))
    update_from_df(pd.read_csv(input_path, low_memory=False))

    x_mask = (x_count > 0).astype(np.float32)
    x_value = np.zeros_like(x_sum)
    m = x_count > 0
    x_value[m] = x_sum[m] / np.maximum(x_count[m], 1)
    # treatment sum features use sum, not mean
    for feat in ["treat_rbc_transfusion_count", "treat_rbc_transfusion_amount"]:
        if feat in f_to_idx:
            idx = f_to_idx[feat]
            x_value[:, :, idx] = x_sum[:, :, idx]
    x_delta = np.full_like(x_sum, cfg.INPUT_END_HOURS)
    for i in range(n):
        for f in range(F):
            prev = None
            for b in range(T):
                if x_mask[i, b, f] > 0:
                    x_delta[i, b, f] = 0.0
                    prev = last_hour[i, b, f]
                else:
                    end = (b + 1) * cfg.BIN_SIZE_HOURS
                    x_delta[i, b, f] = cfg.INPUT_END_HOURS if prev is None or prev < 0 else max(0.0, end - prev)
    x_value = np.nan_to_num(x_value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x_delta = np.nan_to_num(x_delta, nan=cfg.INPUT_END_HOURS, posinf=cfg.INPUT_END_HOURS, neginf=cfg.INPUT_END_HOURS).astype(np.float32)
    x_count = np.clip(x_count, 0, 1000).astype(np.float32)

    # static features for temporal model.
    # Only information known by 24h is allowed. Total ICU LOS, discharge/death fields,
    # and future-window columns are excluded to prevent leakage.
    unsafe = {
        "hospital_expire_flag",
        "los",
        "icu_los_hours_calc",
        "icu_los_hours_calc_internal",
    }
    target_cols = (
        [c for c in df.columns if c.startswith("future_")]
        + ["new_onset_anemia_24_72h", "early_anemia_status_0_24h"]
    )
    id_cols = [
        "subject_id", "hadm_id", "stay_id", "row_idx", "split",
        "intime", "outtime", "admittime", "dischtime", "deathtime",
        "edregtime", "edouttime", "dod",
    ]
    unsafe_substrings = [
        "24_72h", "outtime", "dischtime", "deathtime", "expire",
        "mortality", "death", "dod", "los",
    ]
    temporal_aggregate_prefixes = ("lab0_24h_", "vital0_24h_", "treat0_24h_")
    static_cols = []
    for c in df.columns:
        cl = c.lower()
        if c in unsafe or c in target_cols or c in id_cols:
            continue
        if any(s in cl for s in unsafe_substrings):
            continue
        if c.startswith(temporal_aggregate_prefixes):
            continue
        if c.startswith("last_careunit"):
            continue
        if df[c].dtype == "object":
            continue
        if str(df[c].dtype).startswith("datetime"):
            continue
        if df[c].notna().sum() == 0:
            continue
        static_cols.append(c)
    static_raw = df[static_cols].apply(pd.to_numeric, errors="coerce")
    static_mask = static_raw.notna().astype(np.float32).values
    static_value = static_raw.fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    save_df(pd.DataFrame({"feature_idx": range(F), "temporal_feature": TEMPORAL_FEATURES}), TABLES_DIR / "final_temporal_feature_names.csv")
    save_df(pd.DataFrame({"static_feature": static_cols}), TABLES_DIR / "final_static_feature_names.csv")

    availability = []
    for fi, feat in enumerate(TEMPORAL_FEATURES):
        any_obs = (x_mask[:, :, fi].sum(axis=1) > 0)
        availability.append({"feature_idx": fi, "feature": feat, "rate_stays_observed": float(any_obs.mean()), "rate_bins_observed": float(x_mask[:, :, fi].mean()), "total_count": float(x_count[:, :, fi].sum())})
    save_df(pd.DataFrame(availability), TABLES_DIR / "final_temporal_feature_availability.csv")

    paths = {}
    for split in ["train", "validation", "test"]:
        idx = df.index[df.split == split].values
        arr = {
            "x_value": x_value[idx], "x_mask": x_mask[idx], "x_delta": x_delta[idx], "x_count": x_count[idx],
            "static_value": static_value[idx], "static_mask": static_mask[idx],
            "y_future_min_hb": df.iloc[idx].future_min_hb_24_72h.values.astype(np.float32),
            "y_future_severe_anemia": df.iloc[idx].future_severe_anemia_24_72h.values.astype(np.float32),
            "y_new_onset_anemia": df.iloc[idx].new_onset_anemia_24_72h.fillna(0).values.astype(np.float32),
            "new_onset_mask": df.iloc[idx].new_onset_anemia_24_72h.notna().astype(np.float32).values,
            "early_last_hb": df.iloc[idx].early_last_hb_0_24h.fillna(df.early_last_hb_0_24h.median()).values.astype(np.float32),
            "early_min_hb": df.iloc[idx].early_min_hb_0_24h.fillna(df.early_min_hb_0_24h.median()).values.astype(np.float32),
            "hb_slope": df.iloc[idx].lab0_24h_hemoglobin_slope_per_hour.fillna(0).values.astype(np.float32),
            "hb_margin_to_8": df.iloc[idx].hb_margin_to_8.fillna(0).values.astype(np.float32),
            "hb_margin_to_sex_threshold": df.iloc[idx].hb_margin_to_sex_threshold.fillna(0).values.astype(np.float32),
            "subject_id": df.iloc[idx].subject_id.values.astype(np.int64),
            "hadm_id": df.iloc[idx].hadm_id.values.astype(np.int64),
            "stay_id": df.iloc[idx].stay_id.values.astype(np.int64),
        }
        path = DATA_DIR / f"temporal_{split}.npz"
        np.savez_compressed(path, **arr)
        paths[split] = str(path)
        print(f"[SAVED] {path}")
    shape_tbl = pd.DataFrame([{"split": s, "npz": p} for s, p in paths.items()])
    save_df(shape_tbl, TABLES_DIR / "final_temporal_npz_paths.csv")
    return paths, static_cols
