"""Curated MIMIC-IV item identifier maps and feature definitions."""

from pathlib import Path

import pandas as pd

from ..config import TABLES_DIR
from ..utils import available_usecols, normalize_label, save_df

LAB_VARIABLES = [
    "hemoglobin", "hematocrit", "rbc", "mcv", "mchc", "mch", "rdw", "platelet", "wbc",
    "creatinine", "bun", "lactate", "albumin", "bilirubin", "inr", "pt", "ptt",
    "ferritin", "iron", "tibc", "transferrin", "b12", "folate",
]
VITAL_VARIABLES = [
    "heart_rate", "systolic_bp", "diastolic_bp", "mean_bp", "respiratory_rate", "spo2",
    "temperature", "gcs_eye", "gcs_motor", "gcs_verbal",
]
TREAT_VARIABLES = ["rbc_transfusion_count", "rbc_transfusion_amount"]
TEMPORAL_FEATURES = [f"lab_{x}" for x in LAB_VARIABLES] + [f"vital_{x}" for x in VITAL_VARIABLES] + [f"treat_{x}" for x in TREAT_VARIABLES]


def build_lab_itemid_map(root: Path) -> pd.DataFrame:
    path = root / "hosp" / "d_labitems.csv"
    d = pd.read_csv(path, usecols=available_usecols(path, ["itemid", "label", "fluid", "category"]), low_memory=False)
    curated = {
        "lab_hemoglobin": (["hemoglobin"], ["a1c", "glycated", "fetal", "hemoglobin a", "hemoglobin a2", "hemoglobin c", "hemoglobin f", "hemoglobin s", "reticulocyte", "plasma free", "urine"]),
        "lab_hematocrit": (["hematocrit", "hematocrit calculated"], ["urine"]),
        "lab_rbc": (["red blood cells"], ["urine", "csf", "fluid", "bacteria"]),
        "lab_mcv": (["mcv"], []),
        "lab_mchc": (["mchc"], []),
        "lab_mch": (["mch"], ["mchc"]),
        "lab_rdw": (["rdw", "rdw cv", "rdw sd"], []),
        "lab_platelet": (["platelet count"], ["platelet clumps", "platelet smear", "mean platelet volume", "mpv", "platelet aggregation"]),
        "lab_wbc": (["white blood cells", "wbc"], ["urine", "csf", "fluid"]),
        "lab_creatinine": (["creatinine"], ["urine", "clearance"]),
        "lab_bun": (["urea nitrogen", "bun"], ["urine"]),
        "lab_lactate": (["lactate"], ["lactate dehydrogenase", "ld", "ldh"]),
        "lab_albumin": (["albumin"], ["urine", "albumin creatinine"]),
        "lab_bilirubin": (["bilirubin total", "bilirubin direct", "bilirubin indirect", "bilirubin"], ["urine"]),
        "lab_inr": (["inr pt", "inr"], ["control"]),
        "lab_pt": (["pt", "prothrombin time"], ["ptt", "control", "mean", "inr"]),
        "lab_ptt": (["ptt", "partial thromboplastin time"], ["control", "mean", "la", "lupus"]),
        "lab_ferritin": (["ferritin"], []),
        "lab_iron": (["iron"], ["iron binding capacity", "tibc", "transferrin"]),
        "lab_tibc": (["iron binding capacity total", "total iron binding capacity", "tibc"], []),
        "lab_transferrin": (["transferrin"], ["saturation"]),
        "lab_b12": (["vitamin b12", "b12"], []),
        "lab_folate": (["folate"], []),
    }
    bad_fluids = ["urine", "csf", "pleural", "ascites", "joint fluid", "body fluid", "stool", "sputum"]
    rows = []
    for _, r in d.iterrows():
        if not pd.notna(r.get("itemid")):
            continue
        label = normalize_label(r.get("label", ""))
        fluid = normalize_label(r.get("fluid", ""))
        if any(bf in fluid for bf in bad_fluids):
            continue
        for feat, (incs, excs) in curated.items():
            if any(inc == label or inc in label for inc in incs) and not any(exc in label for exc in excs):
                rows.append({
                    "itemid": int(r["itemid"]),
                    "label": r.get("label", ""),
                    "fluid": r.get("fluid", ""),
                    "category": r.get("category", ""),
                    "feature": feat,
                    "variable": feat.replace("lab_", ""),
                })
                break
    out = pd.DataFrame(rows).drop_duplicates(subset=["itemid", "feature"])
    save_df(out, TABLES_DIR / "final_lab_itemid_map.csv")
    return out


def build_vital_itemid_map(root: Path) -> pd.DataFrame:
    path = root / "icu" / "d_items.csv"
    d = pd.read_csv(path, usecols=available_usecols(path, ["itemid", "label", "category", "unitname"]), low_memory=False)
    curated = {
        "vital_heart_rate": (["heart rate"], ["alarm", "high", "low"]),
        "vital_systolic_bp": (["arterial blood pressure systolic", "non invasive blood pressure systolic", "nbp systolic", "art bp systolic", "manual blood pressure systolic left", "manual blood pressure systolic right"], ["pulmonary", "pap", "left ventricular", "lv", "impella", "signal", "alarm", "high", "low"]),
        "vital_diastolic_bp": (["arterial blood pressure diastolic", "non invasive blood pressure diastolic", "nbp diastolic", "art bp diastolic", "manual blood pressure diastolic left", "manual blood pressure diastolic right"], ["pulmonary", "pap", "left ventricular", "lv", "impella", "signal", "alarm", "high", "low"]),
        "vital_mean_bp": (["arterial blood pressure mean", "non invasive blood pressure mean", "nbp mean", "art bp mean"], ["pulmonary", "pap", "left ventricular", "lv", "impella", "signal", "alarm", "high", "low"]),
        "vital_respiratory_rate": (["respiratory rate"], ["set", "setting", "alarm", "high", "low"]),
        "vital_spo2": (["o2 saturation pulseoxymetry", "spo2", "arterial o2 saturation"], ["desat", "limit", "sensor", "pa ", "svr", "mixed venous", "venous", "goal", "alarm"]),
        "vital_temperature": (["temperature fahrenheit", "temperature celsius", "temperature f", "temperature c"], ["site", "change", "changes", "alarm", "high", "low"]),
        "vital_gcs_eye": (["gcs eye opening", "gcs eye"], []),
        "vital_gcs_motor": (["gcs motor response", "gcs motor"], []),
        "vital_gcs_verbal": (["gcs verbal response", "gcs verbal"], []),
    }
    rows = []
    for _, r in d.iterrows():
        if not pd.notna(r.get("itemid")):
            continue
        label = normalize_label(r.get("label", ""))
        for feat, (incs, excs) in curated.items():
            if any(inc == label or inc in label for inc in incs) and not any(exc in label for exc in excs):
                rows.append({
                    "itemid": int(r["itemid"]),
                    "label": r.get("label", ""),
                    "category": r.get("category", ""),
                    "unitname": r.get("unitname", ""),
                    "feature": feat,
                    "variable": feat.replace("vital_", ""),
                })
                break
    out = pd.DataFrame(rows).drop_duplicates(subset=["itemid", "feature"])
    save_df(out, TABLES_DIR / "final_vital_itemid_map.csv")
    return out


def build_input_itemid_map(root: Path) -> pd.DataFrame:
    path = root / "icu" / "d_items.csv"
    d = pd.read_csv(path, usecols=available_usecols(path, ["itemid", "label", "category", "unitname"]), low_memory=False)
    allowed = ["packed red blood cells", "packed rbc", "or packed rbc intake", "pacu packed rbc intake", "red blood cells"]
    excludes = ["consent", "order", "documentation", "comment", "status", "reason"]
    rows = []
    for _, r in d.iterrows():
        if not pd.notna(r.get("itemid")):
            continue
        label = normalize_label(r.get("label", ""))
        if any(a == label or a in label for a in allowed) and not any(e in label for e in excludes):
            rows.append({
                "itemid": int(r["itemid"]),
                "label": r.get("label", ""),
                "category": r.get("category", ""),
                "unitname": r.get("unitname", ""),
            })
    out = pd.DataFrame(rows).drop_duplicates(subset=["itemid"])
    save_df(out, TABLES_DIR / "final_inputevents_rbc_itemid_map.csv")
    return out
