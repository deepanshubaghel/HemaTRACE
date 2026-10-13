"""Verify split isolation by subject using only synthetic records."""

import numpy as np
import pandas as pd

from hematrace.data import features


def test_subject_level_split_is_disjoint(monkeypatch, tmp_path):
    monkeypatch.setattr(features, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(features, "TABLES_DIR", tmp_path / "tables")
    rows = []
    for subject_id in range(100):
        for visit in range(2):
            rows.append({
                "subject_id": subject_id,
                "stay_id": subject_id * 2 + visit,
                "future_severe_anemia_24_72h": subject_id % 2,
                "new_onset_anemia_24_72h": float(subject_id % 3 == 0),
                "future_min_hb_24_72h": 7.0 + subject_id / 20,
            })
    split = features.split_subject_level(pd.DataFrame(rows))
    sets = {
        name: set(split.loc[split.split == name, "subject_id"])
        for name in ("train", "validation", "test")
    }
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert sum(map(len, sets.values())) == 100
