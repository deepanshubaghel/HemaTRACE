"""End-to-end execution from raw MIMIC-IV to paper-ready results."""

import json
from dataclasses import asdict

import pandas as pd

from .comparisons.deep_learning import run_deep_learning_model_comparison
from .config import (
    DATA_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    RUN_DIR,
    TABLES_DIR,
    cfg,
)
from .data.cohort import base_icu_cohort, raw_file_audit
from .data.events import extract_chartevents, extract_inputevents, extract_labevents
from .data.features import build_targets_and_features, build_temporal_npz, split_subject_level
from .data.item_maps import (
    TEMPORAL_FEATURES,
    build_input_itemid_map,
    build_lab_itemid_map,
    build_vital_itemid_map,
)
from .evaluation.explainability import (
    create_combined_pdf,
    plot_decision_curve,
    plot_model_overlays,
    plot_training_history,
    xai_occlusion,
)
from .evaluation.statistics import (
    bootstrap_ci,
    evaluate_master,
    paired_tests,
    subgroup_metrics,
    tune_thresholds_on_validation,
)
from .models.fusion import merge_predictions, train_fusion
from .models.tabular import train_tabular_models
from .models.temporal import train_temporal_model
from .reports.complexity import (
    save_model_complexity_artifacts,
    save_pipeline_documentation_artifacts,
)
from .studies.ablation import run_full_ablation_study
from .utils import json_converter, mimic_root, save_df, save_json

def main():
    print("=" * 100)
    print("HemaTRACE FINAL FULL PIPELINE FROM RAW MIMIC-IV")
    print("=" * 100)

    root = mimic_root()
    print("MIMIC root:", root)
    print("Run dir:", RUN_DIR)

    save_json(asdict(cfg), LOGS_DIR / "final_config.json")
    documentation_artifacts = save_pipeline_documentation_artifacts()

    audit = raw_file_audit(root)
    lab_map = build_lab_itemid_map(root)
    vital_map = build_vital_itemid_map(root)
    input_map = build_input_itemid_map(root)

    cohort = base_icu_cohort(root)
    lab_path = extract_labevents(root, cohort, lab_map)
    vital_path = extract_chartevents(root, cohort, vital_map)
    input_path = extract_inputevents(root, cohort, input_map)

    all_df = build_targets_and_features(cohort, lab_path, vital_path, input_path)
    all_df = split_subject_level(all_df)

    temporal_paths, static_cols = build_temporal_npz(all_df, lab_path, vital_path, input_path)
    tab_preds, numeric_cols, cat_cols = train_tabular_models(all_df)

    # Save the final model-wise complexity table after real feature dimensions are known.
    documentation_artifacts.update(
        save_model_complexity_artifacts(
            npz_paths=temporal_paths,
            static_cols=static_cols,
            numeric_cols=numeric_cols,
            cat_cols=cat_cols,
        )
    )

    temporal_result = train_temporal_model(temporal_paths)

    val_master, test_master = merge_predictions(
        tab_preds["validation"],
        tab_preds["test"],
        temporal_result["val_pred_df"],
        temporal_result["test_pred_df"],
        temporal_result["onset_val_df"],
        temporal_result["onset_test_df"],
    )

    val_master, test_master = train_fusion(val_master, test_master)

    threshold_df = tune_thresholds_on_validation(val_master)
    save_df(threshold_df, TABLES_DIR / "final_validation_tuned_thresholds.csv")

    met_val = evaluate_master(val_master, "validation", threshold_df)
    met_test = evaluate_master(test_master, "test", threshold_df)
    metrics = pd.concat([met_val, met_test], ignore_index=True)
    save_df(metrics, TABLES_DIR / "final_all_model_metrics.csv")

    # Full component-level ablation study. This re-trains controlled variants
    # of the HemaTRACE temporal model and saves manuscript-ready ablation tables,
    # deltas, predictions, histories, and figures.
    ablation_result = run_full_ablation_study(
        temporal_paths=temporal_paths,
        val_master=val_master,
        test_master=test_master,
        threshold_df=threshold_df,
    )

    dl_comparison_result = run_deep_learning_model_comparison(
        temporal_paths=temporal_paths,
        hema_metrics_path=TABLES_DIR / "final_all_model_metrics.csv",
    )

    clean = []
    for model in metrics.model.unique():
        t = metrics[(metrics.split == "test") & (metrics.model == model)]
        for _, r in t.iterrows():
            if r.task == "future_min_hb_regression":
                clean.append({"model": model, "task": "Future minimum Hb regression", "primary_metric": "MAE", "test_value": r.MAE, "n": r.N})
            elif r.task == "future_severe_anemia_classification":
                clean.append({"model": model, "task": "Future severe anemia classification", "primary_metric": "AUROC", "test_value": r.AUROC, "AUPRC": r.AUPRC, "ECE": r.ECE, "n": r.N})
            elif r.task == "new_onset_anemia_classification":
                clean.append({"model": model, "task": "New-onset anemia classification", "primary_metric": "AUROC", "test_value": r.AUROC, "AUPRC": r.AUPRC, "ECE": r.ECE, "n": r.N})
    save_df(pd.DataFrame(clean), TABLES_DIR / "final_clean_summary_for_paper.csv")

    boot = bootstrap_ci(test_master, cfg.BOOTSTRAP_N)
    pair = paired_tests(test_master)
    sub = subgroup_metrics(test_master)

    plot_training_history(temporal_result["history"])
    plot_model_overlays(test_master)

    dca1 = plot_decision_curve(
        test_master,
        "true_future_severe_anemia",
        {
            "Tabular": "tabular_pred_severe_prob_cal",
            "HemaTRACE-Temporal": "temporal_pred_severe_prob_cal",
            "HemaTRACE-Fusion": "fusion_pred_severe_prob",
        },
        "severe_anemia",
    )
    save_df(dca1, TABLES_DIR / "final_decision_curve_severe.csv")

    onset_mask = test_master.new_onset_mask.astype(bool)
    dca2 = plot_decision_curve(
        test_master.loc[onset_mask].copy(),
        "true_new_onset_anemia",
        {
            "Tabular": "tabular_pred_onset_prob_cal",
            "HemaTRACE-Temporal": "temporal_pred_onset_prob_cal",
            "HemaTRACE-OnsetSpecialized": "onset_specialized_pred_onset_prob_cal",
            "HemaTRACE-Fusion": "fusion_pred_onset_prob",
        },
        "new_onset_anemia",
    )
    save_df(dca2, TABLES_DIR / "final_decision_curve_onset.csv")

    xai_occlusion(None, temporal_result, test_master)
    combined_pdf = create_combined_pdf()

    summary = {
        "project": cfg.PROJECT_NAME,
        "run_dir": str(RUN_DIR),
        "mimic_root": str(root),
        "n_final_cohort": int(len(all_df)),
        "n_train": int((all_df.split == "train").sum()),
        "n_validation": int((all_df.split == "validation").sum()),
        "n_test": int((all_df.split == "test").sum()),
        "n_temporal_features": len(TEMPORAL_FEATURES),
        "n_static_features": len(static_cols),
        "n_tabular_numeric_features": len(numeric_cols),
        "n_tabular_categorical_features": len(cat_cols),
        "models": [
            "Tabular",
            "HemaTRACE-Temporal",
            "HemaTRACE-OnsetSpecialized",
            "HemaTRACE-Fusion",
            "DL_MLP_Static",
            "DL_MLP_TemporalFlatten",
            "DL_CNN1D",
            "DL_LSTM",
            "DL_GRU",
            "DL_BiGRU",
            "DL_BiLSTM_Attention",
            "DL_Transformer",
            "DL_TCN",
        ],
        "documentation_artifacts": documentation_artifacts,
        "ablation_artifacts": ablation_result.get("summary", {}) if isinstance(ablation_result, dict) else {},
        "deep_learning_comparison_artifacts": dl_comparison_result.get("outputs", {}) if isinstance(dl_comparison_result, dict) else {},
        "combined_pdf": str(combined_pdf),
        "important_outputs": {
            "master_validation": str(DATA_DIR / "final_master_predictions_validation.csv"),
            "master_test": str(DATA_DIR / "final_master_predictions_test.csv"),
            "metrics": str(TABLES_DIR / "final_all_model_metrics.csv"),
            "clean_summary": str(TABLES_DIR / "final_clean_summary_for_paper.csv"),
            "pipeline_complexity": str(TABLES_DIR / "final_pipeline_complexity_analysis.csv"),
            "model_complexity": str(TABLES_DIR / "final_model_complexity_analysis.csv"),
            "journal_methods_checklist": str(TABLES_DIR / "final_journal_methods_checklist.csv"),
            "output_manifest": str(TABLES_DIR / "final_output_manifest.csv"),
            "ablation_clean_table": str(TABLES_DIR / "final_ablation_clean_table_for_manuscript.csv"),
            "ablation_delta_vs_full": str(TABLES_DIR / "final_ablation_delta_vs_full.csv"),
            "ablation_summary": str(REPORTS_DIR / "final_ablation_summary.json"),
            "dl_comparison_table": str(TABLES_DIR / "final_dl_model_comparison_manuscript_table.csv"),
            "manuscript_dir": str(RUN_DIR / "manuscript"),
            "bootstrap_ci": str(TABLES_DIR / "final_bootstrap_ci.csv"),
            "paired_tests": str(TABLES_DIR / "final_paired_bootstrap_tests.csv"),
            "subgroup_metrics": str(TABLES_DIR / "final_subgroup_metrics.csv"),
            "combined_pdf": str(combined_pdf),
        },
    }
    save_json(summary, REPORTS_DIR / "final_summary.json")
    print(json.dumps(summary, indent=4, default=json_converter))
