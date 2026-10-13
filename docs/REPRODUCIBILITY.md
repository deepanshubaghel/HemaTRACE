# Reproducibility notes

## Source-preserved settings

`CFG.RANDOM_STATE = 42`; split proportions are 70/10/20 at the subject level; input 0–24 h; target 24–72 h; temporal discretization 6 × 4 h; main-model training max 80 epochs with patience 12; default 512 batches/12 workers; 1000 bootstrap replications. Research model details and fine-tuning settings are in `src/hematrace/config.py`.

## Expected environment information to record

- Python version, OS, CPU/GPU type, PyTorch/CUDA/cuDNN versions
- MIMIC-IV release version and exactly which raw CSV files were used
- Installed versions of pandas, NumPy, scikit-learn, Matplotlib, SciPy, statsmodels, LightGBM, XGBoost
- Presence/absence of LightGBM and XGBoost (existing learners have preference/fallback behavior)
- `logs/final_config.json`, outputs from `final_split_summary.csv`, and preprocessing/itemid map tables
- Git commit SHA for the modular implementation and any local modifications

### CUDA determinism

The source calls `torch.backends.cudnn.benchmark = True` and seeds Python, NumPy and torch. GPU training can still be non-deterministic across devices, kernels, library versions and worker scheduling. This setting is deliberately preserved, rather than quietly changing it to a potentially different experimental configuration.

## No-raw-data checks provided

`python -m compileall -q src main.py` checks syntax. `python -m pytest -q` runs fast numerical, subject-split, comparator-override, and shape smoke tests. Neither validates raw-data lab mapping accuracy, empirical study metrics, or full MIMIC-IV runtime.

## Scope and inherited limitations

The structural refactor did not audit scientific assumptions or repair original data handling. In particular, raw event filtering reads large CSV chunks and then parts of the filtering/aggregation process load full filtered tables into memory; adequate RAM/disk are needed. The original source creates timestamped output directories at import. If a method or bias issue is found in the underlying research script, any scientific code fix should be made intentionally, tracked as a new version, and empirically revalidated; the present ZIP should not be described as a scientifically improved model.
