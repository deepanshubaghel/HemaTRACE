"""Pipeline complexity, model complexity, and reproducibility artifacts."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..comparisons.deep_learning import _dl_build_model, _make_dl_cfg
from ..config import (
    DATA_DIR,
    FIGURES_DIR,
    FIGURE_PDF_DIR,
    FIGURE_PNG_DIR,
    FUSION_MODEL_DIR,
    LOGS_DIR,
    MODELS_DIR,
    ONSET_MODEL_DIR,
    REPORTS_DIR,
    RUN_DIR,
    TABLES_DIR,
    TABULAR_MODEL_DIR,
    TEMPORAL_MODEL_DIR,
    cfg,
)
from ..data.item_maps import TEMPORAL_FEATURES
from ..models.temporal import HemaTRACEModel
from ..utils import save_df, save_json

COMPLEXITY_SYMBOLS = {
    "N": "number of final ICU stays/cohort rows",
    "S": "number of unique subjects",
    "N_train": "number of training stays",
    "N_val": "number of validation stays",
    "N_test": "number of test stays",
    "E_lab_raw": "number of raw labevents rows scanned",
    "E_chart_raw": "number of raw chartevents rows scanned",
    "E_input_raw": "number of raw inputevents rows scanned",
    "E_filtered": "number of retained time-windowed events after item-id and cohort filtering",
    "F_tab": "number of tabular features after imputation and one-hot encoding",
    "F_raw_tab": "number of selected raw tabular columns before one-hot expansion",
    "F_temp": "number of temporal variables, len(TEMPORAL_FEATURES)",
    "F_static": "number of static numeric features used by temporal neural models",
    "T": "number of temporal bins, cfg.N_BINS",
    "B": "mini-batch size",
    "H": "hidden dimension",
    "D": "temporal embedding dimension",
    "L": "number of recurrent/Transformer/TCN layers",
    "R": "number of bootstrap resamples",
    "M": "number of evaluated model-task pairs",
    "P": "number of prediction columns used by fusion/calibration",
    "K": "number of files/items/categories, depending on row context",
}


def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return None


def count_trainable_parameters(model) -> int:
    """Return trainable parameter count for any torch model."""
    try:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception:
        return 0


def _read_npz_dimensions(npz_paths: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """Read tensor dimensions without loading unnecessary files into model code."""
    dims = {
        "N_train": None,
        "N_val": None,
        "N_test": None,
        "T": cfg.N_BINS,
        "F_temp": len(TEMPORAL_FEATURES),
        "F_static": None,
    }
    if not npz_paths:
        return dims
    for split_key, out_key in [("train", "N_train"), ("validation", "N_val"), ("test", "N_test")]:
        path = npz_paths.get(split_key)
        if path is None:
            continue
        try:
            z = np.load(path)
            dims[out_key] = int(z["x_value"].shape[0])
            dims["T"] = int(z["x_value"].shape[1])
            dims["F_temp"] = int(z["x_value"].shape[2])
            dims["F_static"] = int(z["static_value"].shape[1])
            z.close()
        except Exception:
            pass
    return dims


def pipeline_complexity_rows() -> List[Dict[str, object]]:
    """
    Stage-wise complexity table suitable for a journal supplement.

    The complexities are symbolic because exact runtime depends on MIMIC-IV table
    sizes, disk bandwidth, CPU/GPU, and whether LightGBM/XGBoost are installed.
    """
    return [
        {
            "stage_no": 1,
            "pipeline_stage": "Raw input audit",
            "main_functions": "raw_file_audit",
            "dominant_operation": "existence and file-size checks",
            "time_complexity_big_o": "O(K)",
            "space_complexity_big_o": "O(K)",
            "journal_note": "K is the number of required and optional raw MIMIC-IV files. No full CSV scan is performed.",
            "primary_outputs": "final_raw_file_audit.csv",
        },
        {
            "stage_no": 2,
            "pipeline_stage": "Adult ICU cohort construction",
            "main_functions": "load_static_raw; compute_age; base_icu_cohort",
            "dominant_operation": "CSV reads and joins over patients, admissions, icustays",
            "time_complexity_big_o": "O(N_patients + N_admissions + N_icustays)",
            "space_complexity_big_o": "O(N_patients + N_admissions + N_icustays)",
            "journal_note": "Adult stays with ICU length of stay at least 24 h are retained before outcome construction.",
            "primary_outputs": "final_base_adult_icu_cohort.csv; final_cohort_flow.csv",
        },
        {
            "stage_no": 3,
            "pipeline_stage": "Curated variable-to-itemid mapping",
            "main_functions": "build_lab_itemid_map; build_vital_itemid_map; build_input_itemid_map",
            "dominant_operation": "dictionary normalization and inclusion/exclusion rule matching",
            "time_complexity_big_o": "O(I_lab + I_icu)",
            "space_complexity_big_o": "O(I_lab + I_icu)",
            "journal_note": "Item IDs are selected only from MIMIC-IV dictionary tables using transparent curated rules.",
            "primary_outputs": "final_lab_itemid_map.csv; final_vital_itemid_map.csv; final_inputevents_rbc_itemid_map.csv",
        },
        {
            "stage_no": 4,
            "pipeline_stage": "Chunked raw event extraction",
            "main_functions": "extract_labevents; extract_chartevents; extract_inputevents",
            "dominant_operation": "single pass through large event tables",
            "time_complexity_big_o": "O(E_lab_raw + E_chart_raw + E_input_raw)",
            "space_complexity_big_o": "O(C + E_filtered_chunk)",
            "journal_note": "Large event tables are streamed in chunks; peak memory is controlled by configured chunk size C.",
            "primary_outputs": "filtered_labevents_0_72.csv; filtered_chartevents_0_24.csv; filtered_inputevents_0_24.csv",
        },
        {
            "stage_no": 5,
            "pipeline_stage": "Outcome construction and aggregate features",
            "main_functions": "build_targets_and_features; aggregate_events",
            "dominant_operation": "group-by stay and feature; event sorting for first/last/slope",
            "time_complexity_big_o": "O(E_filtered + G log G)",
            "space_complexity_big_o": "O(E_filtered + N * F_raw_tab)",
            "journal_note": "Only 0-24 h features are used for prediction; 24-72 h hemoglobin is used only for labels.",
            "primary_outputs": "final_cohort_all.csv",
        },
        {
            "stage_no": 6,
            "pipeline_stage": "Subject-level train/validation/test split",
            "main_functions": "split_subject_level",
            "dominant_operation": "subject sampling and row assignment",
            "time_complexity_big_o": "O(S + N)",
            "space_complexity_big_o": "O(S + N)",
            "journal_note": "Splitting by subject prevents the same patient from appearing in multiple partitions.",
            "primary_outputs": "final_train.csv; final_validation.csv; final_test.csv; final_split_summary.csv",
        },
        {
            "stage_no": 7,
            "pipeline_stage": "Temporal tensor construction",
            "main_functions": "build_temporal_npz",
            "dominant_operation": "event-to-bin aggregation and dense tensor assembly",
            "time_complexity_big_o": "O(E_filtered + N * T * F_temp)",
            "space_complexity_big_o": "O(N * T * F_temp + N * F_static)",
            "journal_note": "Four temporal channels are saved: normalized value, observation mask, time-since-last-observed, and count.",
            "primary_outputs": "temporal_train.npz; temporal_validation.npz; temporal_test.npz; final_temporal_feature_availability.csv",
        },
        {
            "stage_no": 8,
            "pipeline_stage": "Tabular baseline training",
            "main_functions": "train_tabular_models",
            "dominant_operation": "preprocessing plus gradient-boosted model fitting",
            "time_complexity_big_o": "O(3 * train_cost_tree_boosting(N_train, F_tab))",
            "space_complexity_big_o": "O(N_train * F_tab + model_size)",
            "journal_note": "Three tabular models are trained: regression, severe anemia classification, and new-onset anemia classification.",
            "primary_outputs": "tabular_regressor.pkl; tabular_severe_classifier.pkl; tabular_onset_classifier.pkl",
        },
        {
            "stage_no": 9,
            "pipeline_stage": "HemaTRACE temporal multitask neural training",
            "main_functions": "train_temporal_model; HemaTRACEModel",
            "dominant_operation": "GRU recurrent encoding with attention pooling and multitask heads",
            "time_complexity_big_o": "O(Epochs * ceil(N_train/B) * T * (D*H + H^2))",
            "space_complexity_big_o": "O(B * T * (F_temp + H) + parameters)",
            "journal_note": "The model jointly predicts future minimum Hb, severe anemia, and new-onset anemia.",
            "primary_outputs": "best_*_model.pt; temporal predictions; training_history.csv",
        },
        {
            "stage_no": 10,
            "pipeline_stage": "Onset-specialized fine-tuning, calibration, and fusion",
            "main_functions": "train_temporal_model; train_fusion; tune_thresholds_on_validation",
            "dominant_operation": "validation-only calibration and logistic fusion",
            "time_complexity_big_o": "O(N_val * P + train_cost_fusion(N_val, P))",
            "space_complexity_big_o": "O(N_val * P + fusion_model_size)",
            "journal_note": "Thresholds and fusion weights are learned only on validation data and then frozen for testing.",
            "primary_outputs": "final_master_predictions_validation.csv; final_master_predictions_test.csv; final_validation_tuned_thresholds.csv",
        },
        {
            "stage_no": 11,
            "pipeline_stage": "Deep-learning comparator training",
            "main_functions": "run_deep_learning_model_comparison",
            "dominant_operation": "sequential training of MLP, CNN, RNN, Transformer, and TCN comparators",
            "time_complexity_big_o": "O(sum_m Epochs_m * ceil(N_train/B) * model_cost_m)",
            "space_complexity_big_o": "O(max_m activation_memory_m + parameters_m)",
            "journal_note": "All comparator models use the same temporal/static tensors and multitask targets for fair comparison.",
            "primary_outputs": "final_dl_model_comparison_long.csv; final_dl_model_comparison_table.csv; final_dl_model_thresholds.csv",
        },
        {
            "stage_no": 12,
            "pipeline_stage": "Evaluation, uncertainty, and statistical testing",
            "main_functions": "evaluate_master; bootstrap_ci; paired_tests; subgroup_metrics",
            "dominant_operation": "metric calculation, bootstrap resampling, paired comparisons",
            "time_complexity_big_o": "O(R * N_test * M)",
            "space_complexity_big_o": "O(N_test * P + R * M)",
            "journal_note": "Bootstrap confidence intervals and paired tests are saved for manuscript reporting.",
            "primary_outputs": "final_all_model_metrics.csv; final_bootstrap_ci.csv; final_paired_bootstrap_tests.csv; final_subgroup_metrics.csv",
        },
        {
            "stage_no": 13,
            "pipeline_stage": "Journal figures and XAI",
            "main_functions": "plot_*; xai_occlusion; create_combined_pdf",
            "dominant_operation": "plotting and feature-time occlusion analysis",
            "time_complexity_big_o": "O(min(N_test, XAI_MAX_SAMPLES) * T * F_temp)",
            "space_complexity_big_o": "O(B * T * F_temp + figure_memory)",
            "journal_note": "Figures are saved in both PNG and PDF formats; XAI is bounded by the configured sample cap.",
            "primary_outputs": "figures/png; figures/pdf; final_combined_all_figures.pdf",
        },
    ]


def model_complexity_rows(
    npz_paths: Optional[Dict[str, str]] = None,
    static_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Return model-specific time/space complexity, including all trained models."""
    dims = _read_npz_dimensions(npz_paths)
    F_temp = int(dims.get("F_temp") or len(TEMPORAL_FEATURES))
    F_static = int(dims.get("F_static") or (_safe_len(static_cols) or 0))
    T = int(dims.get("T") or cfg.N_BINS)
    F_raw_tab = int((_safe_len(numeric_cols) or 0) + (_safe_len(cat_cols) or 0))
    H = int(cfg.GRU_HIDDEN_DIM)
    D = int(cfg.TEMPORAL_EMBED_DIM)

    rows: List[Dict[str, object]] = []

    def add(**kwargs):
        base = {
            "F_temp": F_temp,
            "F_static": F_static,
            "T": T,
            "F_raw_tab": F_raw_tab,
            "trainable_parameters": None,
            "parameter_count_status": "not_instantiated",
        }
        base.update(kwargs)
        rows.append(base)

    add(
        model_family="Tabular baseline",
        model_name="Tabular regressor",
        implementation="LightGBMRegressor, XGBRegressor, or HistGradientBoostingRegressor fallback",
        task="future minimum Hb regression",
        input_representation="0-24 h aggregate tabular features + static covariates",
        forward_time_per_sample="O(number_of_trees * tree_depth)",
        training_time_big_o="LightGBM/XGBoost approx. O(num_trees * N_train * F_tab * log N_train); HistGradientBoosting approx. O(iterations * N_train * F_tab)",
        inference_time_big_o="O(num_trees * tree_depth)",
        memory_big_o="O(N_train * F_tab + num_trees * leaves)",
        journal_use="Strict non-temporal baseline for regression comparison.",
    )
    add(
        model_family="Tabular baseline",
        model_name="Tabular severe-anemia classifier",
        implementation="LightGBMClassifier, XGBClassifier, or HistGradientBoostingClassifier fallback",
        task="future severe anemia classification",
        input_representation="0-24 h aggregate tabular features + static covariates",
        forward_time_per_sample="O(number_of_trees * tree_depth)",
        training_time_big_o="LightGBM/XGBoost approx. O(num_trees * N_train * F_tab * log N_train)",
        inference_time_big_o="O(num_trees * tree_depth)",
        memory_big_o="O(N_train * F_tab + num_trees * leaves)",
        journal_use="Strong tree-based baseline for severe anemia discrimination.",
    )
    add(
        model_family="Tabular baseline",
        model_name="Tabular new-onset anemia classifier",
        implementation="LightGBMClassifier, XGBClassifier, or HistGradientBoostingClassifier fallback",
        task="new-onset anemia classification among eligible stays",
        input_representation="0-24 h aggregate tabular features + static covariates",
        forward_time_per_sample="O(number_of_trees * tree_depth)",
        training_time_big_o="O(num_trees * N_onset_train * F_tab * log N_onset_train)",
        inference_time_big_o="O(num_trees * tree_depth)",
        memory_big_o="O(N_onset_train * F_tab + num_trees * leaves)",
        journal_use="Tabular comparator for the new-onset subgroup.",
    )
    add(
        model_family="Proposed model",
        model_name="HemaTRACE-Temporal GRU-attention multitask model",
        implementation="HemaTRACEModel",
        task="joint Hb regression, severe anemia classification, new-onset anemia classification",
        input_representation="B x T x 4F_temp temporal tensor + static value/mask vector",
        forward_time_per_sample="O(T * (4F_temp*D + D*H + H^2) + F_static*STATIC_HIDDEN_DIM + H*SHARED_HIDDEN_DIM)",
        training_time_big_o="O(Epochs * ceil(N_train/B) * B * forward_time_per_sample)",
        inference_time_big_o="O(T * (4F_temp*D + D*H + H^2))",
        memory_big_o="O(B*T*(4F_temp + H) + parameters)",
        journal_use="Primary proposed temporal model with attention-based time aggregation.",
    )
    add(
        model_family="Proposed model",
        model_name="HemaTRACE-OnsetSpecialized",
        implementation="fine-tuned HemaTRACEModel on onset-eligible stays",
        task="new-onset anemia classification",
        input_representation="same temporal/static tensors, onset-eligible subset only",
        forward_time_per_sample="same as HemaTRACE-Temporal",
        training_time_big_o="O(Onset_epochs * ceil(N_onset_train/B) * B * forward_time_per_sample)",
        inference_time_big_o="same as HemaTRACE-Temporal",
        memory_big_o="same as HemaTRACE-Temporal",
        journal_use="Specialized decision model for incident anemia among patients not anemic in the first 24 h.",
    )
    add(
        model_family="Fusion/calibration",
        model_name="HemaTRACE-Fusion",
        implementation="validation-trained logistic/ridge fusion models",
        task="fused calibrated predictions",
        input_representation="validation/test prediction columns from tabular and temporal models",
        forward_time_per_sample="O(P)",
        training_time_big_o="O(iterations * N_val * P)",
        inference_time_big_o="O(P)",
        memory_big_o="O(N_val * P + P)",
        journal_use="Validation-only fusion to combine tabular and temporal evidence without test leakage.",
    )

    comparator_specs = [
        ("DL_MLP_Static", "static MLP", "O(F_static*hidden + hidden^2)", "O(B*hidden + parameters)"),
        ("DL_MLP_TemporalFlatten", "flattened temporal MLP", "O(T*4F_temp*hidden + hidden^2)", "O(B*T*4F_temp + B*hidden + parameters)"),
        ("DL_CNN1D", "1D temporal convolution", "O(T*(4F_temp*H*k + H^2*k))", "O(B*T*H + parameters)"),
        ("DL_LSTM", "unidirectional LSTM", "O(T*(4F_temp*H + H^2))", "O(B*T*H + parameters)"),
        ("DL_GRU", "unidirectional GRU", "O(T*(4F_temp*H + H^2))", "O(B*T*H + parameters)"),
        ("DL_BiGRU", "bidirectional GRU", "O(2*T*(4F_temp*H + H^2))", "O(B*T*2H + parameters)"),
        ("DL_BiLSTM_Attention", "bidirectional LSTM with attention", "O(2*T*(4F_temp*H + H^2) + T*H)", "O(B*T*2H + parameters)"),
        ("DL_Transformer", "Transformer encoder", "O(L*(T^2*H + T*H^2))", "O(B*(T^2 + T*H) + parameters)"),
        ("DL_TCN", "temporal convolutional network", "O(L*T*H^2*k)", "O(B*T*H + parameters)"),
    ]
    for name, impl, fwd, mem in comparator_specs:
        add(
            model_family="Deep-learning comparator",
            model_name=name,
            implementation=impl,
            task="same multitask targets as HemaTRACE for fair architecture comparison",
            input_representation="same B x T x 4F_temp temporal tensor + static value/mask vector",
            forward_time_per_sample=fwd,
            training_time_big_o=f"O(Epochs * ceil(N_train/B) * B * {fwd})",
            inference_time_big_o=fwd,
            memory_big_o=mem,
            journal_use="Ablation/comparator architecture for journal model comparison table.",
        )

    # Fill exact trainable parameter counts when model classes are available.
    try:
        model = HemaTRACEModel(F_temp, F_static)
        p = count_trainable_parameters(model)
        for row in rows:
            if row["model_name"] in ["HemaTRACE-Temporal GRU-attention multitask model", "HemaTRACE-OnsetSpecialized"]:
                row["trainable_parameters"] = p
                row["parameter_count_status"] = "computed_from_current_config"
    except Exception as exc:
        for row in rows:
            if row["model_name"].startswith("HemaTRACE"):
                row["parameter_count_status"] = f"not_computed: {type(exc).__name__}"

    try:
        c = _make_dl_cfg()
        for name, _, _, _ in comparator_specs:
            try:
                mdl = _dl_build_model(name, F_temp, F_static, c)
                p = count_trainable_parameters(mdl)
                for row in rows:
                    if row["model_name"] == name:
                        row["trainable_parameters"] = p if p > 0 else None
                        row["parameter_count_status"] = "computed_from_current_config" if p > 0 else "lazy_layers_not_initialized"
            except Exception as exc:
                for row in rows:
                    if row["model_name"] == name:
                        row["parameter_count_status"] = f"not_computed: {type(exc).__name__}"
    except Exception:
        pass

    return rows


def journal_methods_rows() -> List[Dict[str, str]]:
    """Short paper-ready methods checklist for reproducibility."""
    return [
        {"section": "Study design", "item": "Prediction window", "description": "Features are restricted to ICU 0-24 h; outcomes are measured in ICU 24-72 h."},
        {"section": "Leakage control", "item": "Subject-level split", "description": "Train/validation/test partitions are assigned by subject_id, preventing patient overlap."},
        {"section": "Leakage control", "item": "Feature exclusion", "description": "Discharge time, death, length-of-stay, and future-window variables are excluded from model inputs."},
        {"section": "Feature engineering", "item": "Tabular features", "description": "For each variable, count, mean, standard deviation, minimum, maximum, first, last, slope, and missingness indicators are generated from 0-24 h events."},
        {"section": "Feature engineering", "item": "Temporal tensor", "description": "Each stay is represented as T bins with value, mask, delta-time, and count channels for every temporal feature."},
        {"section": "Primary model", "item": "HemaTRACE", "description": "A temporal encoder, GRU, attention pooling, static encoder, and multitask heads predict Hb regression and anemia risks."},
        {"section": "Baselines", "item": "Tabular and deep learning comparators", "description": "Tree-based tabular models and MLP/CNN/RNN/Transformer/TCN neural comparators are trained on the same split."},
        {"section": "Calibration", "item": "Validation-only", "description": "Temperature calibration, threshold selection, and fusion are trained on validation predictions only."},
        {"section": "Statistics", "item": "Uncertainty", "description": "Bootstrap confidence intervals, paired bootstrap tests, McNemar tests, subgroup metrics, and decision-curve analysis are saved."},
        {"section": "Reporting", "item": "Artifacts", "description": "All key tables are saved in CSV and JSON where appropriate for journal reproducibility."},
    ]


def pipeline_output_manifest_rows() -> List[Dict[str, str]]:
    """Return a comprehensive output manifest."""
    return [
        {"category": "configuration", "path": str(LOGS_DIR / "final_config.json"), "format": "JSON", "description": "Frozen configuration used for the run."},
        {"category": "documentation", "path": str(TABLES_DIR / "final_pipeline_complexity_analysis.csv"), "format": "CSV", "description": "Stage-wise time/space complexity table."},
        {"category": "documentation", "path": str(TABLES_DIR / "final_model_complexity_analysis.csv"), "format": "CSV", "description": "Model-wise complexity table covering tabular, HemaTRACE, fusion, and all DL comparators."},
        {"category": "documentation", "path": str(TABLES_DIR / "final_journal_methods_checklist.csv"), "format": "CSV", "description": "Paper-ready reproducibility and methods checklist."},
        {"category": "documentation", "path": str(REPORTS_DIR / "final_pipeline_complexity_analysis.json"), "format": "JSON", "description": "Machine-readable pipeline complexity table and notation."},
        {"category": "documentation", "path": str(REPORTS_DIR / "final_model_complexity_analysis.json"), "format": "JSON", "description": "Machine-readable model complexity table and notation."},
        {"category": "documentation", "path": str(TABLES_DIR / "final_output_manifest.csv"), "format": "CSV", "description": "Inventory of important output files."},
        {"category": "documentation", "path": str(REPORTS_DIR / "final_output_manifest.json"), "format": "JSON", "description": "Machine-readable output inventory."},
        {"category": "audit", "path": str(TABLES_DIR / "final_raw_file_audit.csv"), "format": "CSV", "description": "Required and optional raw MIMIC-IV file availability."},
        {"category": "cohort", "path": str(DATA_DIR / "final_base_adult_icu_cohort.csv"), "format": "CSV", "description": "Adult ICU cohort after first eligibility filtering."},
        {"category": "cohort", "path": str(DATA_DIR / "final_cohort_all.csv"), "format": "CSV", "description": "Final analytic cohort with outcomes and engineered features."},
        {"category": "split", "path": str(TABLES_DIR / "final_split_summary.csv"), "format": "CSV", "description": "Subject-level train/validation/test split summary."},
        {"category": "features", "path": str(TABLES_DIR / "final_lab_itemid_map.csv"), "format": "CSV", "description": "Curated laboratory item IDs."},
        {"category": "features", "path": str(TABLES_DIR / "final_vital_itemid_map.csv"), "format": "CSV", "description": "Curated vital-sign item IDs."},
        {"category": "features", "path": str(TABLES_DIR / "final_inputevents_rbc_itemid_map.csv"), "format": "CSV", "description": "Curated RBC transfusion item IDs."},
        {"category": "features", "path": str(TABLES_DIR / "final_tabular_numeric_features.csv"), "format": "CSV", "description": "Numeric tabular feature list after leakage filtering."},
        {"category": "features", "path": str(TABLES_DIR / "final_tabular_categorical_features.csv"), "format": "CSV", "description": "Categorical tabular feature list after leakage filtering."},
        {"category": "features", "path": str(TABLES_DIR / "final_temporal_feature_names.csv"), "format": "CSV", "description": "Temporal feature index mapping."},
        {"category": "features", "path": str(TABLES_DIR / "final_static_feature_names.csv"), "format": "CSV", "description": "Static feature list used by temporal neural models."},
        {"category": "features", "path": str(TABLES_DIR / "final_temporal_feature_availability.csv"), "format": "CSV", "description": "Temporal observation availability by variable."},
        {"category": "tensors", "path": str(DATA_DIR / "temporal_train.npz"), "format": "NPZ", "description": "Training temporal tensors and labels."},
        {"category": "tensors", "path": str(DATA_DIR / "temporal_validation.npz"), "format": "NPZ", "description": "Validation temporal tensors and labels."},
        {"category": "tensors", "path": str(DATA_DIR / "temporal_test.npz"), "format": "NPZ", "description": "Test temporal tensors and labels."},
        {"category": "models", "path": str(TABULAR_MODEL_DIR), "format": "PKL", "description": "Saved tabular sklearn pipelines."},
        {"category": "models", "path": str(TEMPORAL_MODEL_DIR), "format": "PT/PKL/CSV", "description": "Saved HemaTRACE temporal model artifacts."},
        {"category": "models", "path": str(ONSET_MODEL_DIR), "format": "PT", "description": "Onset-specialized temporal model weights."},
        {"category": "models", "path": str(FUSION_MODEL_DIR), "format": "PKL/JSON", "description": "Validation-trained fusion/calibration artifacts."},
        {"category": "predictions", "path": str(DATA_DIR / "final_master_predictions_validation.csv"), "format": "CSV", "description": "Validation truth, calibrated predictions, fusion predictions, and metadata."},
        {"category": "predictions", "path": str(DATA_DIR / "final_master_predictions_test.csv"), "format": "CSV", "description": "Test truth, calibrated predictions, fusion predictions, and metadata."},
        {"category": "metrics", "path": str(TABLES_DIR / "final_all_model_metrics.csv"), "format": "CSV", "description": "All validation/test task metrics."},
        {"category": "metrics", "path": str(TABLES_DIR / "final_clean_summary_for_paper.csv"), "format": "CSV", "description": "Compact paper-ready summary table."},
        {"category": "uncertainty", "path": str(TABLES_DIR / "final_bootstrap_ci.csv"), "format": "CSV", "description": "Bootstrap confidence intervals."},
        {"category": "statistical_tests", "path": str(TABLES_DIR / "final_paired_bootstrap_tests.csv"), "format": "CSV", "description": "Paired bootstrap comparisons."},
        {"category": "subgroups", "path": str(TABLES_DIR / "final_subgroup_metrics.csv"), "format": "CSV", "description": "Subgroup-specific performance metrics."},
        {"category": "decision_curve", "path": str(TABLES_DIR / "final_decision_curve_severe.csv"), "format": "CSV", "description": "Decision-curve analysis for severe anemia."},
        {"category": "decision_curve", "path": str(TABLES_DIR / "final_decision_curve_onset.csv"), "format": "CSV", "description": "Decision-curve analysis for new-onset anemia."},
        {"category": "deep_learning_comparison", "path": str(TABLES_DIR / "final_dl_model_comparison_long.csv"), "format": "CSV", "description": "Long-form metrics for all DL comparators."},
        {"category": "deep_learning_comparison", "path": str(TABLES_DIR / "final_dl_model_comparison_table.csv"), "format": "CSV", "description": "Wide comparison table for DL comparators."},
        {"category": "figures", "path": str(FIGURE_PNG_DIR), "format": "PNG", "description": "Journal-quality PNG figures."},
        {"category": "figures", "path": str(FIGURE_PDF_DIR), "format": "PDF", "description": "Journal-quality PDF figures."},
        {"category": "report", "path": str(REPORTS_DIR / "final_combined_all_figures.pdf"), "format": "PDF", "description": "Combined figure report."},
        {"category": "report", "path": str(REPORTS_DIR / "final_summary.json"), "format": "JSON", "description": "Final run summary and important output paths."},
    ]


def save_pipeline_documentation_artifacts() -> Dict[str, str]:
    """
    Save documentation artifacts that do not require trained model dimensions.

    This function is called at the start of main() so a run always contains
    pipeline-level complexity, output manifest, methods checklist, and
    reproducibility metadata even if a later model-training step fails.
    """
    complexity_df = pd.DataFrame(pipeline_complexity_rows())
    methods_df = pd.DataFrame(journal_methods_rows())
    output_df = pd.DataFrame(pipeline_output_manifest_rows())

    complexity_csv = TABLES_DIR / "final_pipeline_complexity_analysis.csv"
    complexity_json = REPORTS_DIR / "final_pipeline_complexity_analysis.json"
    methods_csv = TABLES_DIR / "final_journal_methods_checklist.csv"
    methods_json = REPORTS_DIR / "final_journal_methods_checklist.json"
    manifest_csv = TABLES_DIR / "final_output_manifest.csv"
    manifest_json = REPORTS_DIR / "final_output_manifest.json"
    reproducibility_json = REPORTS_DIR / "final_reproducibility_manifest.json"

    save_df(complexity_df, complexity_csv)
    save_json({"notation": COMPLEXITY_SYMBOLS, "complexity_table": complexity_df.to_dict(orient="records")}, complexity_json)
    save_df(methods_df, methods_csv)
    save_json({"journal_methods_checklist": methods_df.to_dict(orient="records")}, methods_json)
    save_df(output_df, manifest_csv)
    save_json({"outputs": output_df.to_dict(orient="records")}, manifest_json)

    import platform
    reproducibility = {
        "project": cfg.PROJECT_NAME,
        "random_state": cfg.RANDOM_STATE,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "input_window_hours": [cfg.INPUT_START_HOURS, cfg.INPUT_END_HOURS],
        "target_window_hours": [cfg.TARGET_START_HOURS, cfg.TARGET_END_HOURS],
        "split_fractions": {"train": cfg.TRAIN_FRAC, "validation": cfg.VAL_FRAC, "test": cfg.TEST_FRAC},
        "temporal_tensor": {"bin_size_hours": cfg.BIN_SIZE_HOURS, "n_bins": cfg.N_BINS, "n_temporal_features": len(TEMPORAL_FEATURES)},
        "clinical_thresholds": {
            "severe_hb_threshold": cfg.SEVERE_HB_THRESHOLD,
            "male_anemia_threshold": cfg.MALE_ANEMIA_THRESHOLD,
            "female_anemia_threshold": cfg.FEMALE_ANEMIA_THRESHOLD,
        },
        "important_directories": {
            "run_dir": str(RUN_DIR),
            "data_dir": str(DATA_DIR),
            "models_dir": str(MODELS_DIR),
            "tables_dir": str(TABLES_DIR),
            "figures_dir": str(FIGURES_DIR),
            "reports_dir": str(REPORTS_DIR),
        },
    }
    save_json(reproducibility, reproducibility_json)

    return {
        "pipeline_complexity_csv": str(complexity_csv),
        "pipeline_complexity_json": str(complexity_json),
        "journal_methods_csv": str(methods_csv),
        "journal_methods_json": str(methods_json),
        "output_manifest_csv": str(manifest_csv),
        "output_manifest_json": str(manifest_json),
        "reproducibility_json": str(reproducibility_json),
    }


def save_model_complexity_artifacts(
    npz_paths: Optional[Dict[str, str]] = None,
    static_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Save model-wise complexity after feature dimensions are known.

    This table is the main journal-ready complexity table for all models:
    tabular baselines, proposed HemaTRACE, onset-specialized HemaTRACE, fusion,
    and every deep-learning comparator.
    """
    model_df = pd.DataFrame(model_complexity_rows(npz_paths, static_cols, numeric_cols, cat_cols))
    model_csv = TABLES_DIR / "final_model_complexity_analysis.csv"
    model_json = REPORTS_DIR / "final_model_complexity_analysis.json"
    save_df(model_df, model_csv)
    save_json({"notation": COMPLEXITY_SYMBOLS, "model_complexity_table": model_df.to_dict(orient="records")}, model_json)
    return {"model_complexity_csv": str(model_csv), "model_complexity_json": str(model_json)}
