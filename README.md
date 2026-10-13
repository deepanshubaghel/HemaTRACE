# HemaTRACE

**A modular, GitHub-ready implementation of the complete HemaTRACE MIMIC-IV research pipeline.**

HemaTRACE predicts future hemoglobin and anemia outcomes for adult ICU stays using **measurements collected during ICU hours 0–24** to predict **outcomes observed during ICU hours 24–72**. This repository is a structural refactor of the supplied single-file research implementation; it keeps the original algorithms, data rules, training configurations, thresholds, model types, study stages, filenames, and plotting behavior. It also retains the originally appended **deep-learning model comparison**, **full ablation study**, **model/pipeline complexity analysis**, and **manuscript-ready plotting extensions**.

> **Research software only.** The code has not been clinically validated and should not be used for treatment, transfusion, or patient-management decisions. This repository does **not** include MIMIC-IV data, model checkpoints, trained weights, or empirical results.

## Contents

- [1. Project layout](#1-project-layout)
- [2. Pipeline design](#2-pipeline-design)
- [3. Prerequisites](#3-prerequisites)
- [4. Installation](#4-installation)
- [5. Input dataset layout](#5-input-dataset-layout)
- [6. How to run](#6-how-to-run)
- [7. Configuration](#7-configuration)
- [8. Model families](#8-model-families)
- [9. Evaluation and generated outputs](#9-evaluation-and-generated-outputs)
- [10. Reproducibility and leakage control](#10-reproducibility-and-leakage-control)
- [11. Tests and code maintenance](#11-tests-and-code-maintenance)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Data access, citation, and use](#13-data-access-citation-and-use)
  
## 1. Project layout

```text
hematrace/
├── README.md
├── main.py                         # Run the complete original workflow
├── pyproject.toml                  # Editable install + CLI entry point
├── requirements.txt                # Required runtime dependencies
├── requirements-optional.txt       # Optional statistical/boosting dependencies
├── requirements-dev.txt            # Test dependencies
├── Makefile
├── MANIFEST.in
├── .gitignore
├── Dataset/
│   └── Mimic4/
│       ├── hosp/                   # Place raw MIMIC-IV hospital CSVs here
│       └── icu/                    # Place raw MIMIC-IV ICU CSVs here
├── src/
│   └── hematrace/
│       ├── __init__.py
│       ├── __main__.py              # python -m hematrace
│       ├── config.py                # Original CFG, run directories
│       ├── utils.py                 # Helpers and random seeding
│       ├── metrics.py               # Regression and classification metrics
│       ├── optional_dependencies.py # Optional imports with existing fallbacks
│       ├── pipeline.py              # The complete main() orchestration
│       ├── data/
│       │   ├── item_maps.py         # Curated MIMIC itemid mappings
│       │   ├── cohort.py            # Raw file audit + ICU cohort
│       │   ├── events.py            # Chunked event extraction
│       │   └── features.py          # Targets, aggregate features, NPZ, splits
│       ├── models/
│       │   ├── tabular.py           # Tabular learners
│       │   ├── temporal.py          # HemaTRACE GRU-attention multitask model
│       │   └── fusion.py            # Calibration and validation-trained fusion
│       ├── evaluation/
│       │   ├── statistics.py        # Metrics, bootstrap, tests, subgroups
│       │   └── explainability.py    # XAI and original plots
│       ├── studies/
│       │   └── ablation.py          # Full controlled component ablations
│       ├── comparisons/
│       │   └── deep_learning.py     # All DL baselines + manuscript overrides
│       └── reports/
│           └── complexity.py       # Runtime/model complexity and manifests
├── scripts/
│   └── check_data.py               # Filename-only raw dataset check
├── tests/                          # Synthetic smoke tests; no clinical data
└── docs/
    ├── ARCHITECTURE.md
    ├── DATA_AND_TARGETS.md
    ├── REPRODUCIBILITY.md

```

`HemaTrace_results/` is created automatically when the pipeline modules are imported and is excluded from Git. No original output datasets or previously trained models are loaded by the primary pipeline.

## 2. Pipeline design

```text
Raw MIMIC-IV hosp/ + icu/ CSVs
      |
      v
Required-file audit --> age/ICU-stay cohort --> curated lab/vital/RBC mappings
      |
      v
Filtered labevents (0–72 h) + chartevents (0–24 h)
                              + inputevents (0–24 h)
      |
      +--> future hemoglobin targets (24–72 h)
      +--> observed Hb / aggregate features (0–24 h)
      +--> 4-hour temporal tensors (6 bins)
      |
      v
Subject-level train / validation / test (70% / 10% / 20%)
      |
      +--> tabular baselines
      +--> HemaTRACE GRU-attention regression + two classification heads
      +--> onset-specialized fine-tuning
      +--> calibration / validation-trained fusion
      +--> component ablation re-training
      +--> nine deep-learning comparator architectures
      |
      v
Test metrics + confidence intervals + paired tests + subgroup analysis
      + decision curves + XAI + manuscript plots + PDFs + complexity manifests
```

The main prediction targets are:

| Task | Source-derived definition |
| --- | --- |
| Future minimum Hb | Minimum available hemoglobin in ICU hours **24–72** (regression) |
| Future severe anemia | Future minimum Hb **< 8.0** (binary classification) |
| New-onset anemia | Anemia in hours 24–72 among stays **without observed anemia in hours 0–24**; threshold is Hb **< 13.0** for recorded male, **< 12.0** for recorded female |

The code requires an observed future Hb value to include a stay in the modeled cohort. Missing early Hb does **not** qualify a stay as new-onset-negative. See [Data and targets](docs/DATA_AND_TARGETS.md).

## 3. Prerequisites

- Python **3.10+** with compatible library wheels (recommended: 3.11 or 3.12).
- An environment with enough disk space and RAM to process large MIMIC-IV CSVs and preserve intermediate events/arrays; a CUDA GPU is highly recommended for full experiments. CPU execution is supported by the underlying PyTorch branches, but can be very slow.
- Authorized access to the **MIMIC-IV** raw dataset and its data-use requirements. Do not include hospital data in a public GitHub repository.
- The exact raw CSV structure below. Compressed `.csv.gz` inputs are **not** used by this unchanged implementation; decompress them to `.csv` first.

## 4. Installation

Execute these commands from the **repository root**:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

Alternatively, using the included dependency file:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For all originally optional scientific and boosted-tree packages:

```bash
python -m pip install -e '.[all]'
# Or, after installing the base requirements:
python -m pip install -r requirements-optional.txt
```

For tests:

```bash
python -m pip install -e '.[dev]'
```

**PyTorch GPU builds:** install the wheel appropriate to your CUDA driver using the official PyTorch installation instructions before `pip install -e .`. The generic `torch` requirement does not guarantee CUDA availability. Optional **LightGBM** or **XGBoost** changes the learner selected by the original tabular fallback logic; for strict experiment reproduction, record the installed versions and whether they were present.

## 5. Input dataset layout

Place **your own authorized** raw files in these directories (case-sensitive on Linux):

```text
Dataset/Mimic4/
├── hosp/
│   ├── patients.csv                [required]
│   ├── admissions.csv              [required]
│   ├── labevents.csv               [required]
│   ├── d_labitems.csv              [required]
│   ├── diagnoses_icd.csv           [optional]
│   ├── d_icd_diagnoses.csv         [optional]
│   ├── prescriptions.csv          [optional]
│   ├── pharmacy.csv               [optional]
│   ├── emar.csv                   [optional]
│   └── emar_detail.csv            [optional]
└── icu/
    ├── icustays.csv                [required]
    ├── chartevents.csv             [required]
    ├── d_items.csv                 [required]
    ├── inputevents.csv             [required]
    └── procedureevents.csv         [optional]
```

The code reads the standard identifier/time/measurement columns by name (e.g., `subject_id`, `hadm_id`, `stay_id`, `itemid`, `charttime`, `valuenum`, `intime`). Item maps are curated from the raw dictionary tables during each run. Check filename presence without running any model:

```bash
python scripts/check_data.py
# or check a different location:
python scripts/check_data.py --root /path/to/Mimic4
```

Note that the checker verifies filenames only. The pipeline's own audit and subsequent CSV operations enforce the input requirements during execution. `Mimic4` is the source-script's exact folder convention and is not renamed to `MIMIC-IV` by this refactor.

## 6. How to run

From the repository root, after placing the raw CSVs:

```bash
python main.py
```

The installed-package alternatives run the **same** `main()`:

```bash
python -m hematrace
# or
hematrace-run
```

No old Step 2/3/4/4D outputs, predictions, NPZ arrays, or checkpoints are needed. Every full run starts with the raw MIMIC-IV CSVs and creates a timestamped result directory. The run is deliberately comprehensive: it trains tabular and temporal predictors, onset refinement, fusion, the ablation set, and DL comparator models and writes figures/reports. It is not a quick single-model inference command.

### Fast debug mode (optional)

The source includes a limited-data development flag. To use it, edit **`src/hematrace/config.py`**, keeping the remaining settings unchanged:

```python
FAST_DEBUG: bool = True
```

Default debug settings are `FAST_DEBUG_MAX_SUBJECTS=3000`, `FAST_DEBUG_MAX_CHUNKS=2`, and `FAST_DEBUG_BOOTSTRAP_N=50`. Because only the first two raw chunks are processed, debug outcomes and class rates may be unrepresentative, and small/debug splits may produce classes insufficient for all training or evaluation stages. **Fast debug mode is not a validated end-to-end miniature dataset.** Reset `FAST_DEBUG=False` for final research runs.

## 7. Configuration

Original configuration lives in **`src/hematrace/config.py`** as the unchanged `CFG` dataclass plus module-wide output paths. These are the **default values in the uploaded code**:

| Setting | Value | Description |
| --- | --- | --- |
| `DATA_ROOT` / `MIMIC_SUBDIR` | `Dataset` / `Mimic4` | Relative raw data root |
| `OUT_ROOT` / `PROJECT_NAME` | `HemaTrace_results` / `HemaTRACE_MANUSCRIPT` | Output location |
| `INPUT_START_HOURS`, `INPUT_END_HOURS` | `0.0`, `24.0` | Input event window |
| `TARGET_START_HOURS`, `TARGET_END_HOURS` | `24.0`, `72.0` | Outcome observation window |
| `BIN_SIZE_HOURS`, `N_BINS` | `4.0`, `6` | Temporal discretization |
| `TRAIN_FRAC`, `VAL_FRAC`, `TEST_FRAC` | `0.70`, `0.10`, `0.20` | Subject-level split |
| `RANDOM_STATE` | `42` | Random seed |
| `SEVERE_HB_THRESHOLD` | `8.0` | Severe-anemia cutoff |
| `MALE_ANEMIA_THRESHOLD`, `FEMALE_ANEMIA_THRESHOLD` | `13.0`, `12.0` | Source-script anemia cutoffs |
| `CHUNKSIZE_LABEVENTS`, `CHUNKSIZE_CHARTEVENTS` | `2,000,000` each | CSV chunking |
| `CHUNKSIZE_INPUTEVENTS` | `1,000,000` | ICU input chunking |
| `MAX_EPOCHS`, `PATIENCE` | `80`, `12` | Main temporal training |
| `BATCH_SIZE`, `NUM_WORKERS` | `512`, `12` | Main neural training dataloader |
| `LR`, `WEIGHT_DECAY` | `8e-4`, `1e-4` | Main temporal optimizer |
| `GRU_HIDDEN_DIM`, `TEMPORAL_EMBED_DIM` | `128`, `128` | Temporal architecture |
| `STATIC_HIDDEN_DIM`, `STATIC_EMBED_DIM` | `128`, `64` | Static branch |
| `SHARED_HIDDEN_DIM`, `DROPOUT` | `192`, `0.30` | Shared prediction layers |
| `ONSET_FINETUNE_EPOCHS` | `35` | Onset specialization |
| `BOOTSTRAP_N` | `1000` | Confidence intervals |
| `XAI_MAX_SAMPLES`, `DPI` | `2000`, `300` | XAI sample cap and figure dpi |
| `FAST_DEBUG` | `False` | Small development mode |

Additional loss weights, bootstrap seed, threshold grid, fine-tune optimizer, and comparator/ablation options remain in their original source locations. The main script calls `save_json(asdict(cfg), LOGS_DIR / "final_config.json")`; the comparator also records its settings.

**Caution:** `RUN_DIR` and sibling paths are initialized when `hematrace.config` is first imported. If you change any `CFG` path setting, do so in the dataclass **before running Python** (not by mutating `cfg` after import). Run from the project root so relative paths resolve correctly.

## 8. Model families

The project includes the source implementation of:

1. **Tabular models:** numerical/categorical preprocessing, strict feature selection, baseline regression/classification with original LightGBM/XGBoost/hist-gradient-boosting fallback.
2. **HemaTRACE-Temporal:** temporal value/mask/delta/count inputs, GRU, attention pooling, a static-data branch, and multitask regression/severe-anemia/onset heads.
3. **HemaTRACE-OnsetSpecialized:** fine-tunes the onset objective from the learned temporal model.
4. **HemaTRACE-Fusion:** temperature calibration and validation-trained combination of predictions.
5. **Deep-learning comparators:** static MLP, temporal-flatten MLP, 1D CNN, LSTM, GRU, BiGRU, attention-based BiLSTM, Transformer encoder, TCN.
6. **Ablation study:** retrained variants that remove or mask temporal/static branches, attention, temporal data channels, treatment features, or hemoglobin trend features as defined in the original ablation configuration.

Comparator extensions calculate timing/size/parameter estimates and manuscript-friendly visual comparisons. Refer to [Architecture](docs/ARCHITECTURE.md) for the source files of each stage.

## 9. Evaluation and generated outputs

The **actual directory naming from `CFG`** is:

```text
HemaTrace_results/
└── HemaTRACE_MANUSCRIPT/
    └── T_run_YYYY-MM-DD_at_HH-MM-SS/
        ├── data/                    # Cohort, splits, temporal NPZ, predictions
        ├── models/
        │   ├── tabular/
        │   ├── temporal/
        │   ├── onset_specialized/
        │   ├── fusion/
        │   ├── ablation/
        │   └── dl_comparators/
        ├── tables/                  # CSV tabulation of study outputs
        ├── figures/
        │   ├── png/
        │   └── pdf/
        ├── xai/                     # Explainability artifacts
        ├── reports/                 # JSON and combined PDFs
        ├── logs/                    # Configuration and logs
        ├── tmp_filtered_events/     # Filtered event CSV intermediates
        └── manuscript/             # Manuscript tables, figures, reports
```

Representative outputs (all use original names):

| Relative to run directory | Contents |
| --- | --- |
| `logs/final_config.json` | Config snapshot |
| `tables/final_raw_file_audit.csv` | Input file audit |
| `tables/final_cohort_flow.csv` | Cohort selection summary |
| `data/final_cohort_all.csv` | Final cohort and targets |
| `data/final_train.csv`, `final_validation.csv`, `final_test.csv` | Subject-disjoint sets |
| `data/temporal_train.npz`, `temporal_validation.npz`, `temporal_test.npz` | Neural tensors |
| `tables/final_temporal_feature_names.csv` | Ordered temporal feature names |
| `tables/final_all_model_metrics.csv` | Primary model-wise metrics |
| `tables/final_validation_tuned_thresholds.csv` | Chosen validation thresholds |
| `tables/final_bootstrap_ci.csv` | Bootstrap confidence intervals |
| `tables/final_paired_bootstrap_tests.csv` | Paired significance testing |
| `tables/final_subgroup_metrics.csv` | Stratified metrics |
| `tables/final_ablation_clean_table_for_manuscript.csv` | Ablation summary |
| `tables/final_dl_model_comparison_manuscript_table.csv` | DL comparator summary |
| `tables/final_model_complexity_analysis.csv` | Parameter/MAC/runtime complexity-related descriptors |
| `reports/final_summary.json` | End-of-run artifact pointers and counts |
| `reports/final_combined_all_figures.pdf` | Combined figure report |
| `manuscript/` | Manuscript tables/figures from late source extension |

Metrics include MAE, RMSE, R², Spearman (when SciPy is present), AUROC/AUPRC, accuracy, macro F1, sensitivity/specificity, Brier score, and ECE as appropriate to each task. Full output documentation is also exported **by the pipeline** to `tables/final_output_manifest.csv`.

## 10. Reproducibility and leakage control

The source pipeline was organized, not redesigned. Important existing guards include:

- Subject-level disjoint train/validation/test assignment so repeated stays from the same patient stay in one partition.
- Strict **0–24 h** predictor observation windows and **24–72 h** Hb outcomes.
- Exclusion of raw discharge/death/outtime, total ICU LOS and `future_*` fields from selected predictive features.
- Calibration and fusion trained/selected using validation predictions, with held-out test reporting.
- Original thresholds, seed, preprocessing, and model hyperparameters retained.

**Known limits inherited from the research source:** benchmark reproducibility depends on raw MIMIC-IV version, optional boosting libraries, Python/PyTorch environment, GPU kernels, class balance, and the dataset characteristics. Random seeding does not guarantee bitwise deterministic CUDA training (`cudnn.benchmark` remains enabled as in the original). A complete raw-data run and empirical equivalence check are **not included** because data is not bundled. For methodological and environment details, see [Reproducibility](docs/REPRODUCIBILITY.md).

## 11. Tests and code maintenance

After installing the project:

```bash
python -m pytest -q
python -m compileall -q src main.py
python scripts/check_data.py
```

The repository includes **no-data smoke tests** for metrics, model forward shapes, the subject split, and preservation of the late-comparison overrides. These tests are not a substitute for fully running the pipeline on authorized MIMIC-IV data.

To make research-code changes, use the stage-specific module, while keeping `src/hematrace/pipeline.py` as the single top-level orchestration. The numbered research pipeline stages are maintained as separate importable Python modules, rather than a single `~6000`-line file. The package can be installed with `pip install -e .` and called from standard Python imports, or via the `hematrace-run` console script.

## 12. Troubleshooting

| Symptom | Typical cause / action |
| --- | --- |
| `MIMIC root not found` | Run from repo root and place CSVs under `Dataset/Mimic4/`, or update `CFG.DATA_ROOT` before launch. |
| `Missing required raw MIMIC files` | Run `python scripts/check_data.py`; original reader requires actual `.csv` filenames. |
| `ModuleNotFoundError: hematrace` | Activate venv and run `python -m pip install -e .` or use the repository `python main.py`. |
| No CUDA / extremely slow training | Check `torch.cuda.is_available()`, install the appropriate CUDA-enabled PyTorch build, and provision enough RAM/disk. |
| DataLoader workers fail or exhaust resources | The unchanged default is `NUM_WORKERS=12`; change it in `CFG` to suit your environment. |
| Different tabular learner selected | Check installed `lightgbm`/`xgboost` versions; the original fallback chooses available libraries. |
| AUC undefined / small debug errors | Outcome class can be absent in small splits; use the full authorized cohort for research evaluation. |
| CSV schema errors | Confirm compatible MIMIC-IV tables, required columns, proper CSV decompression and encoding. |
| Memory issues | Original filtered-events and NPZ-building logic holds large arrays/frames in memory; reduce resource use only as a deliberate study code change. |

## 13. Data access, citation, and use

MIMIC-IV is a controlled clinical research dataset, not provided with this code. Researchers must complete the appropriate access and data-use process, cite the correct **MIMIC-IV dataset version** and its creators according to its official documentation, and protect derived patient-level data and checkpoints. Never upload source or intermediate patient records into public issues or commits.

The project title and research methods are from the uploaded script; no study publication citation or author license was supplied. Add your own manuscript/authorship, version-specific dataset citation, and project license **only when you have the rights and bibliographic details**. No license is presumed by this ZIP.


