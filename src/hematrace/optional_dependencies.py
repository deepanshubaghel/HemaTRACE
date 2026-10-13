"""Optional scientific/statistical learners with original fallback behavior."""

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

try:
    from statsmodels.stats.contingency_tables import mcnemar
except Exception:
    mcnemar = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None
