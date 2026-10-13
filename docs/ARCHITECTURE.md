# HemaTRACE architecture and module responsibilities

This repository reproduces the source-script stage orchestration while replacing one Python file with explicit importable modules. No model design, threshold or core stage ordering was changed.

## Research workflow

| Source-script section | Python module | Responsibility |
| --- | --- | --- |
| 1. Config | `hematrace.config` | CFG and run/output directories |
| 2. Utilities | `hematrace.utils` | Reproducibility, CSV/pickle/JSON, label normalization, plots |
| 3. Metrics | `hematrace.metrics` | Regression, classification, calibration, thresholds |
| 4. Curated variables | `hematrace.data.item_maps` | Curated lab/vital/treatment item IDs |
| 5. Raw audit/cohort | `hematrace.data.cohort` | Required files, adult ICU stays |
| 6. Event extraction | `hematrace.data.events` | Chunked labevents, chartevents, inputevents |
| 7. Features/targets/splits | `hematrace.data.features` | Outcomes, aggregates, subject split, temporal NPZ |
| 8. Tabular | `hematrace.models.tabular` | Preprocessing and learners |
| 9. Temporal | `hematrace.models.temporal` | Dataset, HemaTRACEModel, training, fine-tune |
| 10. Calibration/fusion | `hematrace.models.fusion` | Temperature scaling and calibrated ensemble |
| 11. Evaluation | `hematrace.evaluation.statistics` | Metrics and statistical testing |
| 12. XAI | `hematrace.evaluation.explainability` | Graphics, occlusion and PDF |
| Deep-learning add-on | `hematrace.comparisons.deep_learning` | Comparator architectures and paper plots |
| 12B. Complexity | `hematrace.reports.complexity` | Complexity tables, manifests, methods documentation |
| 13A. Ablation | `hematrace.studies.ablation` | Component-drop/mask ablations and metrics |
| 13. Main | `hematrace.pipeline` | Original stage calls, results summary |
| Late manuscript extension | `hematrace.comparisons.deep_learning` | Preserves later comparator function overrides |

## Why the appended extension shares the comparison module

The input script defines four comparison helper names more than once, with late manuscript code retaining original callables in `_..._original` aliases before overriding them. Splitting those overrides into a separate module would change Python global name resolution unless wrappers were inserted. Therefore `comparisons.deep_learning` deliberately keeps both original and later comparison code **in source order** with the same alias capture behavior, avoiding a behavioral rewrite. The script's main calls the later exposed `run_deep_learning_model_comparison`.

## Main neural model

Each of six 4-hour time bins includes four channels per variable: `x_value`, `x_mask`, `x_delta`, and `x_count`. Temporal input flows through an MLP encoder and GRU/attention pooling. Static scalar values/missingness indicators are separately encoded. Concatenated branches go to a shared MLP and three output heads (`delta_z`, `severe_logit`, `onset_logit`). Loss weights and optimization parameters are read from the original `CFG`.

## Package boundaries

- `data/` functions own raw data ingestion and feature construction.
- `models/` owns training/prediction of the primary model families.
- `comparisons/` owns benchmark architectures and manuscript-ready comparison extensions.
- `studies/` owns controlled retraining of HemaTRACE ablations.
- `evaluation/` and `reports/` own plots, statistical analysis, and model/pipeline documentation.
- `pipeline.py` is the only stage-level orchestrator.

Run root entry points are thin Python launchers, **not duplicated copies** of the research algorithms.
