"""Shared filesystem, serialization, randomness, and processing helpers."""

import json
import pickle
import random
import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import FIGURE_PDF_DIR, FIGURE_PNG_DIR, cfg

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(cfg.RANDOM_STATE)



def mimic_root() -> Path:
    root = Path(cfg.DATA_ROOT) / cfg.MIMIC_SUBDIR
    if not root.exists():
        alt = Path("/home/phdcs10010.25/HemaTrace") / cfg.DATA_ROOT / cfg.MIMIC_SUBDIR
        if alt.exists():
            return alt
        raise FileNotFoundError(f"MIMIC root not found: {root} or {alt}")
    return root


def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVED] {path} shape={df.shape}")


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVED] {path}")


def json_converter(o):
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


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, default=json_converter)
    print(f"[SAVED] {path}")


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_label(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def available_usecols(path: Path, wanted: List[str]) -> List[str]:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    return [c for c in wanted if c in cols]


def hours_between(t, intime):
    return (t - intime).dt.total_seconds() / 3600.0


def safe_name(x):
    x = str(x)
    for ch in [" ", "/", "\\", ":", "(", ")", "+", "|", ".", ",", ";", "<", ">", "="]:
        x = x.replace(ch, "_")
    return x


def save_current_fig_both(stem: str):
    png_path = FIGURE_PNG_DIR / f"{stem}.png"
    pdf_path = FIGURE_PDF_DIR / f"{stem}.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def read_csv_maybe(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path, **kwargs)
    return None
