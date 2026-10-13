"""Original leakage-safe tabular baselines and preprocessing."""

from typing import List, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..config import DATA_DIR, TABLES_DIR, TABULAR_MODEL_DIR, cfg
from ..metrics import sigmoid_np
from ..optional_dependencies import lgb, xgb
from ..utils import save_df, save_pickle



class NumericCaster(BaseEstimator, TransformerMixin):
    """
    Pickle-safe numeric caster.

    Important:
        This class is defined at module level, not inside make_preprocessor.
        Therefore sklearn pipelines containing it can be saved with pickle.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return X


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """
    Pickle-safe categorical caster.

    Important:
        This class is defined at module level, not inside make_preprocessor.
        Therefore sklearn pipelines containing it can be saved with pickle.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].astype("object")
        return X


def get_tabular_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Leakage-safe tabular feature selector.

    Rules:
        - Use only features known by ICU hour 24.
        - Exclude IDs, targets, future-window variables, discharge/death fields,
          raw ICU LOS, and derived total LOS.
        - Known categorical variables are forced into categorical branch.
        - Numeric features are selected using pandas is_numeric_dtype.
    """

    target_like = (
        [c for c in df.columns if c.startswith("future_")]
        + ["new_onset_anemia_24_72h", "early_anemia_status_0_24h"]
    )

    exact_exclude = set(
        target_like
        + [
            "subject_id", "hadm_id", "stay_id", "split",
            "intime", "outtime", "admittime", "dischtime", "deathtime",
            "edregtime", "edouttime", "dod",
            "los", "icu_los_hours_calc", "icu_los_hours_calc_internal",
            "hospital_expire_flag",
        ]
    )

    known_categorical = {
        "gender", "race", "insurance", "admission_type",
        "admission_location", "first_careunit",
    }

    unsafe_prefixes = ["last_careunit", "future_"]
    unsafe_substrings = [
        "24_72h", "outtime", "deathtime", "dischtime", "expire",
        "mortality", "death", "dod", "los",
    ]

    numeric, categorical = [], []

    for c in df.columns:
        cl = c.lower()
        if c in exact_exclude:
            continue
        if any(c.startswith(pref) for pref in unsafe_prefixes):
            continue
        if any(s in cl for s in unsafe_substrings):
            continue
        if c in known_categorical:
            if df[c].notna().sum() > 0:
                categorical.append(c)
            continue
        if is_numeric_dtype(df[c]):
            if df[c].notna().sum() > 0:
                numeric.append(c)
        else:
            # Allow only bounded-cardinality categorical variables. This avoids
            # accidental free text or datetime strings.
            nunique = df[c].nunique(dropna=True)
            if 1 <= nunique <= 100:
                categorical.append(c)

    numeric = list(dict.fromkeys(numeric))
    categorical = list(dict.fromkeys([c for c in categorical if c not in numeric]))
    return numeric, categorical

def make_preprocessor(numeric, categorical):
    """
    Pickle-safe sklearn preprocessor.

    Numeric:
        NumericCaster -> median imputation

    Categorical:
        CategoricalCaster -> most frequent imputation -> one-hot encoding
    """

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []

    if len(numeric) > 0:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("cast", NumericCaster()),
                        ("imp", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric,
            )
        )

    if len(categorical) > 0:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("cast", CategoricalCaster()),
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", onehot),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

def choose_regressor():
    if lgb is not None:
        return lgb.LGBMRegressor(n_estimators=900, learning_rate=0.03, num_leaves=31, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, random_state=cfg.RANDOM_STATE, n_jobs=-1)
    if xgb is not None:
        return xgb.XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=4, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, random_state=cfg.RANDOM_STATE, n_jobs=-1, tree_method="hist")
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=31, random_state=cfg.RANDOM_STATE)


def choose_classifier(pos_weight=None):
    if lgb is not None:
        return lgb.LGBMClassifier(n_estimators=900, learning_rate=0.03, num_leaves=31, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, random_state=cfg.RANDOM_STATE, n_jobs=-1)
    if xgb is not None:
        return xgb.XGBClassifier(n_estimators=700, learning_rate=0.03, max_depth=4, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, random_state=cfg.RANDOM_STATE, n_jobs=-1, tree_method="hist", eval_metric="logloss")
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, max_leaf_nodes=31, random_state=cfg.RANDOM_STATE)


def train_tabular_models(df: pd.DataFrame):
    train = df[df.split == "train"].copy()
    val = df[df.split == "validation"].copy()
    test = df[df.split == "test"].copy()
    numeric, categorical = get_tabular_feature_columns(df)
    save_df(pd.DataFrame({"numeric_feature": numeric}), TABLES_DIR / "final_tabular_numeric_features.csv")
    save_df(pd.DataFrame({"categorical_feature": categorical}), TABLES_DIR / "final_tabular_categorical_features.csv")
    pre = make_preprocessor(numeric, categorical)
    X_train = train[numeric + categorical]
    X_val = val[numeric + categorical]
    X_test = test[numeric + categorical]

    reg_pipe = Pipeline([("pre", pre), ("model", choose_regressor())])
    reg_pipe.fit(X_train, train.future_min_hb_24_72h)
    save_pickle(reg_pipe, TABULAR_MODEL_DIR / "tabular_regressor.pkl")

    sev_pipe = Pipeline([("pre", make_preprocessor(numeric, categorical)), ("model", choose_classifier())])
    sev_pipe.fit(X_train, train.future_severe_anemia_24_72h.astype(int))
    save_pickle(sev_pipe, TABULAR_MODEL_DIR / "tabular_severe_classifier.pkl")

    onset_train = train[train.new_onset_anemia_24_72h.notna()].copy()
    onset_pipe = Pipeline([("pre", make_preprocessor(numeric, categorical)), ("model", choose_classifier())])
    onset_pipe.fit(onset_train[numeric + categorical], onset_train.new_onset_anemia_24_72h.astype(int))
    save_pickle(onset_pipe, TABULAR_MODEL_DIR / "tabular_onset_classifier.pkl")

    def proba(pipe, X):
        if hasattr(pipe[-1], "predict_proba"):
            return pipe.predict_proba(X)[:, 1]
        if hasattr(pipe[-1], "decision_function"):
            return sigmoid_np(pipe.decision_function(X))
        return pipe.predict(X)

    preds = {}
    for split_name, d, X in [("validation", val, X_val), ("test", test, X_test)]:
        p_reg = reg_pipe.predict(X)
        p_sev = proba(sev_pipe, X)
        p_on = proba(onset_pipe, X)
        pred_df = pd.DataFrame({
            "subject_id": d.subject_id.values, "hadm_id": d.hadm_id.values, "stay_id": d.stay_id.values,
            "true_future_min_hb": d.future_min_hb_24_72h.values,
            "true_future_severe_anemia": d.future_severe_anemia_24_72h.values.astype(int),
            "true_new_onset_anemia": d.new_onset_anemia_24_72h.fillna(0).values.astype(int),
            "new_onset_mask": d.new_onset_anemia_24_72h.notna().astype(int).values,
            "early_last_hb": d.early_last_hb_0_24h.values,
            "early_min_hb": d.early_min_hb_0_24h.values,
            "hb_slope": d.lab0_24h_hemoglobin_slope_per_hour.fillna(0).values,
            "hb_margin_to_8": d.hb_margin_to_8.fillna(0).values,
            "hb_margin_to_sex_threshold": d.hb_margin_to_sex_threshold.fillna(0).values,
            "tabular_pred_future_min_hb": p_reg,
            "tabular_pred_severe_prob": p_sev,
            "tabular_pred_onset_prob": p_on,
        })
        save_df(pred_df, DATA_DIR / f"tabular_predictions_{split_name}.csv")
        preds[split_name] = pred_df
    return preds, numeric, categorical
