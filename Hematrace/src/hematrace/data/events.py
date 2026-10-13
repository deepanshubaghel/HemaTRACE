"""Chunked extraction of laboratory, chart, and ICU input events."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TMP_DIR, cfg
from ..utils import available_usecols, hours_between, safe_numeric

def append_or_write(df: pd.DataFrame, path: Path):
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def extract_labevents(root: Path, cohort: pd.DataFrame, lab_map: pd.DataFrame):
    path = root / "hosp" / "labevents.csv"
    out_path = TMP_DIR / "filtered_labevents_0_72.csv"
    if out_path.exists(): out_path.unlink()
    item_map = lab_map[["itemid", "feature"]].drop_duplicates()
    itemids = set(item_map.itemid.astype(int))
    coh = cohort[["subject_id", "hadm_id", "stay_id", "intime"]].copy()
    total = 0
    usecols = available_usecols(path, ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"])
    for ci, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=cfg.CHUNKSIZE_LABEVENTS, low_memory=False), 1):
        chunk["itemid"] = safe_numeric(chunk["itemid"])
        chunk = chunk[chunk.itemid.isin(itemids)].copy()
        if chunk.empty:
            if cfg.FAST_DEBUG and ci >= cfg.FAST_DEBUG_MAX_CHUNKS: break
            continue
        chunk["valuenum"] = safe_numeric(chunk["valuenum"])
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk.merge(item_map, on="itemid", how="inner")
        merged = chunk.merge(coh, on=["subject_id", "hadm_id"], how="inner") if "subject_id" in chunk.columns else chunk.merge(coh, on="hadm_id", how="inner")
        if merged.empty:
            if cfg.FAST_DEBUG and ci >= cfg.FAST_DEBUG_MAX_CHUNKS: break
            continue
        merged["charttime"] = pd.to_datetime(merged["charttime"], errors="coerce")
        merged["event_hour"] = hours_between(merged["charttime"], merged["intime"])
        merged = merged[(merged.event_hour >= 0) & (merged.event_hour < cfg.TARGET_END_HOURS)].copy()
        if merged.empty:
            if cfg.FAST_DEBUG and ci >= cfg.FAST_DEBUG_MAX_CHUNKS: break
            continue
        keep = merged[["subject_id", "hadm_id", "stay_id", "feature", "event_hour", "valuenum"]].rename(columns={"valuenum": "value"})
        append_or_write(keep, out_path)
        total += len(keep)
        print(f"LAB chunk {ci}: kept={len(keep):,} total={total:,}")
        del chunk, merged, keep
        gc.collect()
        if cfg.FAST_DEBUG and ci >= cfg.FAST_DEBUG_MAX_CHUNKS: break
    if not out_path.exists():
        pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "feature", "event_hour", "value"]).to_csv(out_path, index=False)
    return out_path


def extract_chartevents(root: Path, cohort: pd.DataFrame, vital_map: pd.DataFrame):
    path = root / "icu" / "chartevents.csv"
    out_path = TMP_DIR / "filtered_chartevents_0_24.csv"
    if out_path.exists(): out_path.unlink()
    item_map = vital_map[["itemid", "feature"]].drop_duplicates()
    itemids = set(item_map.itemid.astype(int))
    stayids = set(cohort.stay_id.astype(int))
    coh = cohort[["stay_id", "intime"]].copy()
    total = 0
    usecols = available_usecols(path, ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"])
    for ci, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=cfg.CHUNKSIZE_CHARTEVENTS, low_memory=False), 1):
        chunk["itemid"] = safe_numeric(chunk["itemid"])
        chunk["stay_id"] = safe_numeric(chunk["stay_id"])
        chunk = chunk[chunk.itemid.isin(itemids) & chunk.stay_id.isin(stayids)].copy()
        if chunk.empty:
            if cfg.FAST_DEBUG and ci >= cfg.FAST_DEBUG_MAX_CHUNKS: break
            continue
        chunk["valuenum"] = safe_numeric(chunk["valuenum"])
        chunk = chunk.dropna(subset=["stay_id", "itemid", "charttime", "valuenum"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk.merge(item_map, on="itemid", how="inner")
        merged = chunk.merge(coh, on="stay_id", how="inner")
        merged["charttime"] = pd.to_datetime(merged["charttime"], errors="coerce")
        merged["event_hour"] = hours_between(merged["charttime"], merged["intime"])
        merged = merged[(merged.event_hour >= 0) & (merged.event_hour < cfg.INPUT_END_HOURS)].copy()
        if merged.empty:
            if cfg.FAST_DEBUG and ci >= cfg.FAST_DEBUG_MAX_CHUNKS: break
            continue
        keep_cols = [c for c in ["subject_id", "hadm_id", "stay_id"] if c in merged.columns]
        keep = merged[keep_cols + ["feature", "event_hour", "valuenum"]].rename(columns={"valuenum": "value"})
        append_or_write(keep, out_path)
        total += len(keep)
        print(f"CHARTEVENTS chunk {ci}: kept={len(keep):,} total={total:,}")
        del chunk, merged, keep
        gc.collect()
        if cfg.FAST_DEBUG and ci >= cfg.FAST_DEBUG_MAX_CHUNKS: break
    if not out_path.exists():
        pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "feature", "event_hour", "value"]).to_csv(out_path, index=False)
    return out_path


def extract_inputevents(root: Path, cohort: pd.DataFrame, input_map: pd.DataFrame):
    path = root / "icu" / "inputevents.csv"
    out_path = TMP_DIR / "filtered_inputevents_0_24.csv"
    if out_path.exists(): out_path.unlink()
    if input_map.empty or not path.exists():
        pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "feature", "event_hour", "value"]).to_csv(out_path, index=False)
        return out_path
    itemids = set(input_map.itemid.astype(int))
    stayids = set(cohort.stay_id.astype(int))
    coh = cohort[["stay_id", "intime"]].copy()
    usecols = available_usecols(path, ["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime", "amount", "totalamount"])
    time_col = "starttime" if "starttime" in usecols else "endtime"
    total = 0
    for ci, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=cfg.CHUNKSIZE_INPUTEVENTS, low_memory=False), 1):
        chunk["itemid"] = safe_numeric(chunk["itemid"])
        chunk["stay_id"] = safe_numeric(chunk["stay_id"])
        chunk = chunk[chunk.itemid.isin(itemids) & chunk.stay_id.isin(stayids)].copy()
        if chunk.empty:
            if cfg.FAST_DEBUG and ci >= cfg.FAST_DEBUG_MAX_CHUNKS: break
            continue
        chunk = chunk.dropna(subset=["stay_id", time_col])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        merged = chunk.merge(coh, on="stay_id", how="inner")
        merged[time_col] = pd.to_datetime(merged[time_col], errors="coerce")
        merged["event_hour"] = hours_between(merged[time_col], merged["intime"])
        merged = merged[(merged.event_hour >= 0) & (merged.event_hour < cfg.INPUT_END_HOURS)].copy()
        if merged.empty:
            if cfg.FAST_DEBUG and ci >= cfg.FAST_DEBUG_MAX_CHUNKS: break
            continue
        amount = safe_numeric(merged["amount"]) if "amount" in merged.columns else pd.Series(np.nan, index=merged.index)
        if amount.notna().sum() == 0 and "totalamount" in merged.columns:
            amount = safe_numeric(merged["totalamount"])
        amount = amount.fillna(0).clip(lower=0)
        base_cols = [c for c in ["subject_id", "hadm_id", "stay_id"] if c in merged.columns]
        count_df = merged[base_cols + ["event_hour"]].copy()
        count_df["feature"] = "treat_rbc_transfusion_count"
        count_df["value"] = 1.0
        amount_df = merged[base_cols + ["event_hour"]].copy()
        amount_df["feature"] = "treat_rbc_transfusion_amount"
        amount_df["value"] = amount.values
        keep = pd.concat([count_df, amount_df], ignore_index=True)
        append_or_write(keep, out_path)
        total += len(merged)
        print(f"INPUTEVENTS chunk {ci}: kept_events={len(merged):,} total={total:,}")
        del chunk, merged, keep, count_df, amount_df
        gc.collect()
        if cfg.FAST_DEBUG and ci >= cfg.FAST_DEBUG_MAX_CHUNKS: break
    if not out_path.exists():
        pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "feature", "event_hour", "value"]).to_csv(out_path, index=False)
    return out_path
