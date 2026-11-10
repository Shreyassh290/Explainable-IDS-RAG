"""Comprehensive evaluation of ensemble intrusion detection on test dataset."""

import json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from ids_rag.config import get_project_root, load_config
from ids_rag.models.ensemble import (
    BinaryDNNEnsemble,
    assign_tiers,
    fit_and_save_temperatures,
)


def compute_expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width confidence bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    correctness = (predictions == targets).astype(np.float64)
    n_samples = len(confidences)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(correctness[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def format_table_vi(report_dict: dict[str, Any], class_names: list[str]) -> str:
    """Format classification report metrics matching Paper Table VI."""
    header = (
        "========================================================================\n"
        "                     TABLE VI: CLASSIFICATION REPORT\n"
        "========================================================================\n"
        f"{'Class / Metric':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}\n"
        "------------------------------------------------------------------------\n"
    )
    lines = [header]

    for name in class_names:
        c_metrics = report_dict.get(name, {})
        prec = f"{c_metrics.get('precision', 0.0):.4f}"
        rec = f"{c_metrics.get('recall', 0.0):.4f}"
        f1 = f"{c_metrics.get('f1-score', 0.0):.4f}"
        supp = f"{int(c_metrics.get('support', 0))}"
        lines.append(f"{name:<20} | {prec:<10} | {rec:<10} | {f1:<10} | {supp:<10}\n")

    lines.append("------------------------------------------------------------------------\n")
    acc = f"{report_dict.get('accuracy', 0.0):.4f}"
    total_supp = f"{int(report_dict.get('macro avg', {}).get('support', 0))}"
    lines.append(f"{'Accuracy':<20} | {'':<10} | {'':<10} | {acc:<10} | {total_supp:<10}\n")

    macro = report_dict.get("macro avg", {})
    lines.append(
        f"{'Macro Average':<20} | {macro.get('precision', 0.0):.4f}     | "
        f"{macro.get('recall', 0.0):.4f}     | {macro.get('f1-score', 0.0):.4f}     | "
        f"{int(macro.get('support', 0)):<10}\n"
    )

    weighted = report_dict.get("weighted avg", {})
    lines.append(
        f"{'Weighted Average':<20} | {weighted.get('precision', 0.0):.4f}     | "
        f"{weighted.get('recall', 0.0):.4f}     | {weighted.get('f1-score', 0.0):.4f}     | "
        f"{int(weighted.get('support', 0)):<10}\n"
    )
    lines.append("========================================================================\n")
    return "".join(lines)


def evaluate_test_set() -> dict[str, Any]:
    """Execute complete Stage 2b test evaluation pipeline."""
    config = load_config()
    root = get_project_root()
    paths_cfg = config.get("paths", {})

    processed_dir = root / paths_cfg.get("processed", "data/processed")
    models_dir = root / paths_cfg.get("models", "models")
    artifacts_dir = root / paths_cfg.get("artifacts", "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    test_path = processed_dir / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    # Load ensemble with raw confidence (paper default)
    ensemble = BinaryDNNEnsemble.load(models_dir=models_dir, use_temperature=False)
    class_names = ensemble.class_names
    label_map = ensemble.label_map

    print(f"Loading test set once from {test_path}...", flush=True)
    df_test = pd.read_parquet(test_path)
    y_test = df_test["label"].to_numpy(dtype=np.int64)
    x_test = df_test[ensemble.feature_names]

    # Predict raw probabilities, labels, confidences, and tiers
    print("Running ensemble inference on test set (raw confidence)...", flush=True)
    raw_probs = ensemble.predict_proba(x_test)
    y_pred, confidences, tiers = ensemble.predict(x_test)

    # 1. Classification Report & Table VI
    report_dict: dict[str, Any] = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )
    table_vi_text = format_table_vi(report_dict, class_names)
    print("\n" + table_vi_text, flush=True)

    # Save classification report text
    report_txt_path = artifacts_dir / "classification_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(table_vi_text)

    # 2. Confusion Matrices (counts and row-normalized)
    cm_counts = confusion_matrix(y_test, y_pred, labels=ensemble.class_ids)
    cm_norm = confusion_matrix(y_test, y_pred, labels=ensemble.class_ids, normalize="true")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_counts, display_labels=class_names)
    disp1.plot(ax=ax1, cmap="Blues", colorbar=True, values_format="d")
    ax1.set_title("Confusion Matrix (Counts)")

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp2.plot(ax=ax2, cmap="Blues", colorbar=True, values_format=".4f")
    ax2.set_title("Confusion Matrix (Row-Normalized)")

    plt.tight_layout()
    cm_path = artifacts_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}", flush=True)

    # 3. One-vs-Rest ROC Curves
    plt.figure(figsize=(8, 6))
    roc_aucs: dict[str, float] = {}
    for c_idx, class_id in enumerate(ensemble.class_ids):
        class_name = label_map[class_id]
        binary_y_true = (y_test == class_id).astype(int)
        c_probs = raw_probs[:, c_idx]
        fpr, tpr, _ = roc_curve(binary_y_true, c_probs)
        roc_val = float(auc(fpr, tpr))
        roc_aucs[class_name] = roc_val
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_val:.4f})", linewidth=1.8)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves (Raw Confidence)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = artifacts_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Saved ROC curves plot to {roc_path}", flush=True)

    # 4. Benign False-Positive Analysis
    benign_id = 0
    benign_mask = y_test == benign_id
    total_benign = int(np.sum(benign_mask))
    benign_fp_count = int(np.sum(benign_mask & (y_pred != benign_id)))
    benign_fp_rate = float(benign_fp_count / total_benign) if total_benign > 0 else 0.0

    print(
        f"Benign False-Positive Analysis:\n"
        f"  Total Benign samples: {total_benign}\n"
        f"  Benign False Positives (predicted as attack): {benign_fp_count}\n"
        f"  Benign False-Positive Rate: {benign_fp_rate:.6f} ({benign_fp_rate * 100:.4f}%)\n",
        flush=True,
    )

    # 5. Confidence Analysis by Tier
    total_test = len(y_test)
    correct_mask = y_test == y_pred
    mean_conf_correct = float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else 0.0
    mean_conf_incorrect = float(np.mean(confidences[~correct_mask])) if np.any(~correct_mask) else 0.0

    tier_analysis: dict[str, dict[str, Any]] = {}
    print("Confidence Tier Analysis:", flush=True)
    for tier_name in ["VERY_HIGH", "HIGH", "MEDIUM", "LOW"]:
        t_mask = tiers == tier_name
        t_count = int(np.sum(t_mask))
        t_acc = float(np.mean(y_test[t_mask] == y_pred[t_mask])) if t_count > 0 else 0.0
        t_pct = float(t_count / total_test * 100.0)
        tier_analysis[tier_name] = {
            "count": t_count,
            "percentage": round(t_pct, 2),
            "accuracy": round(t_acc, 4),
        }
        print(
            f"  {tier_name:<10}: count = {t_count:>6} ({t_pct:>5.1f}%) | "
            f"Accuracy = {t_acc:.4f}",
            flush=True,
        )

    print(
        f"\nMean Confidence:\n"
        f"  Correct predictions:   {mean_conf_correct:.4f}\n"
        f"  Incorrect predictions: {mean_conf_incorrect:.4f}\n",
        flush=True,
    )

    # 6. Expected Calibration Error (Raw Confidence vs Temperature Scaled)
    ece_raw = compute_expected_calibration_error(confidences, y_pred, y_test, n_bins=10)

    # Fit temperatures on validation set
    print("Fitting model temperatures on validation set...", flush=True)
    fitted_temps = fit_and_save_temperatures(models_dir=models_dir)

    # Evaluate test set with temperature scaling
    ensemble_temp = BinaryDNNEnsemble.load(models_dir=models_dir, use_temperature=True)
    temp_probs = ensemble_temp.predict_proba(x_test)
    y_pred_temp = np.argmax(temp_probs, axis=1)
    temp_confidences = np.max(temp_probs, axis=1)
    ece_temp = compute_expected_calibration_error(temp_confidences, y_pred_temp, y_test, n_bins=10)

    print(
        f"Calibration Analysis (10 bins):\n"
        f"  ECE (Raw Confidence - Main):       {ece_raw:.4f}\n"
        f"  ECE (Temperature-Scaled Optional): {ece_temp:.4f}\n",
        flush=True,
    )

    # 7. Save test_predictions.parquet
    predictions_df = pd.DataFrame(
        {
            "row_index": df_test.index.to_numpy(),
            "true_label": y_test,
            "true_class": [label_map[y] for y in y_test],
            "predicted_label": y_pred,
            "predicted_class": [label_map[y] for y in y_pred],
            "confidence": confidences,
            "tier": tiers,
        }
    )
    for c_idx, class_id in enumerate(ensemble.class_ids):
        class_name = label_map[class_id]
        predictions_df[f"prob_{class_name}"] = raw_probs[:, c_idx]

    test_preds_path = artifacts_dir / "test_predictions.parquet"
    predictions_df.to_parquet(test_preds_path, index=False)
    print(f"Saved test predictions to {test_preds_path}", flush=True)

    # 8. Save metrics.json
    metrics_data: dict[str, Any] = {
        "classification_report": report_dict,
        "roc_aucs": roc_aucs,
        "benign_fpr_analysis": {
            "total_benign": total_benign,
            "false_positives": benign_fp_count,
            "false_positive_rate": benign_fp_rate,
        },
        "confidence_analysis": {
            "tiers": tier_analysis,
            "mean_confidence_correct": mean_conf_correct,
            "mean_confidence_incorrect": mean_conf_incorrect,
        },
        "calibration": {
            "ece_raw_main": ece_raw,
            "ece_temperature_scaled": ece_temp,
            "fitted_temperatures": fitted_temps,
        },
    }
    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved full metrics dictionary to {metrics_path}", flush=True)

    return metrics_data


if __name__ == "__main__":
    evaluate_test_set()
