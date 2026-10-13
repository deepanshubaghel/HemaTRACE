"""Adult ICU cohort construction and raw-file validation."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..config import DATA_DIR, TABLES_DIR, cfg
from ..utils import save_df

def raw_file_audit(root: Path):
    required = [
        "hosp/patients.csv", "hosp/admissions.csv", "hosp/labevents.csv", "hosp/d_labitems.csv",
        "icu/icustays.csv", "icu/chartevents.csv", "icu/d_items.csv", "icu/inputevents.csv",
    ]
    optional = [
        "hosp/diagnoses_icd.csv", "hosp/d_icd_diagnoses.csv", "hosp/prescriptions.csv", "hosp/pharmacy.csv",
        "hosp/emar.csv", "hosp/emar_detail.csv", "icu/procedureevents.csv",
    ]
    rows = []
    for rel in required + optional:
        p = root / rel
        rows.append({"file": rel, "required": rel in required, "exists": p.exists(), "size_mb": p.stat().st_size / 1024**2 if p.exists() else np.nan})
    audit = pd.DataFrame(rows)
    save_df(audit, TABLES_DIR / "final_raw_file_audit.csv")
    missing_required = audit[(audit.required) & (~audit.exists)]["file"].tolist()
    if missing_required:
        raise FileNotFoundError(f"Missing required raw MIMIC files: {missing_required}")
    return audit


def load_static_raw(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = pd.read_csv(root / "hosp" / "patients.csv", low_memory=False)
    admissions = pd.read_csv(root / "hosp" / "admissions.csv", low_memory=False)
    icu = pd.read_csv(root / "icu" / "icustays.csv", low_memory=False)
    return patients, admissions, icu


def compute_age(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    adm = admissions.copy()
    pat = patients.copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    if "anchor_age" in pat.columns and "anchor_year" in pat.columns:
        tmp = adm[["subject_id", "hadm_id", "admittime"]].merge(pat[["subject_id", "gender", "anchor_age", "anchor_year"]], on="subject_id", how="left")
        tmp["age"] = tmp["anchor_age"] + (tmp["admittime"].dt.year - tmp["anchor_year"])
        tmp.loc[tmp["anchor_age"] >= 89, "age"] = 91
        return tmp[["subject_id", "hadm_id", "gender", "age"]]
    elif "anchor_age" in pat.columns:
        tmp = adm[["subject_id", "hadm_id"]].merge(pat[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
        tmp["age"] = tmp["anchor_age"]
        return tmp[["subject_id", "hadm_id", "gender", "age"]]
    else:
        raise ValueError("patients.csv must contain anchor_age")


def base_icu_cohort(root: Path) -> pd.DataFrame:
    patients, admissions, icu = load_static_raw(root)
    age_df = compute_age(patients, admissions)
    adm_cols = [c for c in ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "admission_location", "insurance", "race", "hospital_expire_flag"] if c in admissions.columns]
    adm = admissions[adm_cols].copy()
    for c in ["admittime", "dischtime", "deathtime"]:
        if c in adm.columns:
            adm[c] = pd.to_datetime(adm[c], errors="coerce")

    cohort = icu.merge(age_df, on=["subject_id", "hadm_id"], how="left")
    cohort = cohort.merge(adm, on=["subject_id", "hadm_id"], how="left", suffixes=("", "_adm"))
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort["outtime"] = pd.to_datetime(cohort["outtime"], errors="coerce")
    if "los" in cohort.columns:
        cohort["icu_los_hours_calc_internal"] = pd.to_numeric(cohort["los"], errors="coerce") * 24.0
    else:
        cohort["icu_los_hours_calc_internal"] = (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600.0
    cohort = cohort[(cohort["age"] >= 18) & (cohort["icu_los_hours_calc_internal"] >= cfg.INPUT_END_HOURS)].copy()
    cohort = cohort.dropna(subset=["subject_id", "hadm_id", "stay_id", "intime"])
    cohort["subject_id"] = cohort["subject_id"].astype(int)
    cohort["hadm_id"] = cohort["hadm_id"].astype(int)
    cohort["stay_id"] = cohort["stay_id"].astype(int)
    if cfg.FAST_DEBUG:
        sub = cohort["subject_id"].drop_duplicates().sample(min(cfg.FAST_DEBUG_MAX_SUBJECTS, cohort.subject_id.nunique()), random_state=cfg.RANDOM_STATE)
        cohort = cohort[cohort.subject_id.isin(sub)].copy()
    save_df(cohort, DATA_DIR / "final_base_adult_icu_cohort.csv")
    return cohort
