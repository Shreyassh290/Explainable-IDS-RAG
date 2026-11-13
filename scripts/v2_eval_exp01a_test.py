"""One-time held-out test evaluation of candidate EXP-01A (Standard Unweighted BCE).

Evaluates the exact checkpoint selected on validation data (model_EXP-01A.pt)
using the exact leak-free preprocessor (preprocessor_exp01.joblib)
and fixed decision threshold (tau = 0.4650) on UNSW_NB15_testing-set.csv (82,332 flows).

Writes strictly to artifacts/v2/unsw_eval/experiments/exp01/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root
from scripts.v2_exp01_loss_and_leakage import BoostedDNN, LeakFreePreprocessor, NOISE_COLUMNS


def main() -> None:
    root = get_project_root()
    data_dir = root / "data" / "unsw"
    exp_models_dir = root / "models_unsw" / "experiments" / "exp01"
    exp_artifacts_dir = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp01"

    model_path = exp_models_dir / "model_EXP-01A.pt"
    prep_path = exp_models_dir / "preprocessor_exp01.joblib"
    feat_path = exp_models_dir / "feature_names_exp01.json"
    pareto_path = exp_artifacts_dir / "EXP-01A_pareto.json"

    print("Verifying EXP-01A artifacts before execution...", flush=True)
    assert model_path.exists(), f"Model checkpoint not found: {model_path}"
    assert prep_path.exists(), f"Preprocessor artifact not found: {prep_path}"
    assert feat_path.exists(), f"Feature names artifact not found: {feat_path}"
    assert pareto_path.exists(), f"Validation Pareto artifact not found: {pareto_path}"

    with open(feat_path, encoding="utf-8") as f:
        feature_names = json.load(f)
    assert len(feature_names) == 66, f"Expected 66 features, found {len(feature_names)}"

    with open(pareto_path, encoding="utf-8") as f:
        pareto_data = json.load(f)
    tau = float(pareto_data["summary"]["calibrated_threshold"])
    assert np.isclose(tau, 0.4650, atol=1e-4), f"Expected validation threshold 0.4650, got {tau}"

    print(f"Artifacts verified. Using fixed validation-calibrated threshold: tau = {tau:.4f}", flush=True)

    # 1. Load Preprocessor and Test CSV
    preprocessor: LeakFreePreprocessor = joblib.load(prep_path)
    test_csv_path = data_dir / "UNSW_NB15_testing-set.csv"
    print(f"Loading official testing split from: {test_csv_path}...", flush=True)
    df_test = pd.read_csv(test_csv_path)

    for c in NOISE_COLUMNS:
        if c in df_test.columns:
            df_test = df_test.drop(columns=[c])

    y_test = df_test["label"].to_numpy(dtype=np.float32)
    test_attack_cats = df_test["attack_cat"].fillna("Normal").astype(str).to_numpy()

    print(f"Transforming test data through preprocessor fitted strictly on training data...", flush=True)
    X_test = preprocessor.transform(df_test)
    assert X_test.shape[1] == 66, f"Engineered test features count {X_test.shape[1]} != 66"

    # 2. Load Model and Predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoostedDNN(in_features=66, hidden=[256, 128, 64, 32], dropout=0.25).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_tensor = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        test_logits = model(test_tensor).squeeze(1).cpu().numpy()
        test_probs = expit(test_logits)

    test_preds = (test_probs >= tau).astype(int)

    # 3. Overall Test Metrics
    test_acc = float(accuracy_score(y_test, test_preds))
    macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    weighted_f1 = float(f1_score(y_test, test_preds, average="weighted"))
    precision = float(precision_score(y_test, test_preds))
    recall = float(recall_score(y_test, test_preds))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    pr_auc = float(average_precision_score(y_test, test_probs))

    cm = confusion_matrix(y_test, test_preds).tolist()
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    attack_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # 4. Per-Attack-Family Breakdown
    df_eval = pd.DataFrame({"attack_cat": test_attack_cats, "y_true": y_test, "y_pred": test_preds})
    per_cat: dict[str, dict[str, Any]] = {}

    for cat_name, group in df_eval.groupby("attack_cat"):
        cat_acc = float(accuracy_score(group["y_true"], group["y_pred"]))
        cat_support = int(len(group))
        cat_tp = int(np.sum((group["y_true"] == 1) & (group["y_pred"] == 1)))
        cat_fn = int(np.sum((group["y_true"] == 1) & (group["y_pred"] == 0)))
        cat_recall = float(cat_tp / (cat_tp + cat_fn)) if (cat_tp + cat_fn) > 0 else (1.0 - fpr if cat_name == "Normal" else 0.0)

        # Family precision and F1
        p_val, r_val, f1_val, _ = precision_recall_fscore_support(
            group["y_true"], group["y_pred"], average="binary", zero_division=0
        ) if cat_name != "Normal" else (float(tn / (tn + fn)), float(tn / (tn + fp)), 0.0, None)

        per_cat[str(cat_name)] = {
            "support": cat_support,
            "detection_rate": cat_recall if cat_name != "Normal" else cat_acc,
            "precision": float(p_val),
            "recall": float(r_val) if cat_name != "Normal" else cat_acc,
            "f1": float(f1_val),
        }

    # 5. Output Summary Table
    print("\n" + "=" * 75)
    print("EXP-01A HELD-OUT TEST EVALUATION SUMMARY (UNSW-NB15 TEST SPLIT)")
    print("=" * 75)
    print(f"Total Test Flows:          {len(y_test):,}")
    print(f"Calibrated Threshold:      tau = {tau:.4f}")
    print(f"Test Accuracy:             {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Macro F1:             {macro_f1:.4f}")
    print(f"Test Weighted F1:          {weighted_f1:.4f}")
    print(f"Test Precision:            {precision:.4f}")
    print(f"Test Recall (Attacks):     {recall:.4f} ({recall*100:.2f}%)")
    print(f"ROC-AUC:                   {roc_auc:.4f}")
    print(f"PR-AUC:                    {pr_auc:.4f}")
    print(f"Benign False Positive Rate:{fpr*100:.2f}% (FP: {fp:,} / {fp+tn:,})")
    print("-" * 75)
    print("Confusion Matrix:")
    print(f"  [TN: {tn:>6,},  FP: {fp:>6,}]")
    print(f"  [FN: {fn:>6,},  TP: {tp:>6,}]")
    print("=" * 75)

    print("\nPer-Attack-Family Classification Metrics:")
    print(f"{'Category':<16} | {'Support':<8} | {'Detection Rate':<15} | {'Precision':<10} | {'Recall':<10} | {'F1':<8}")
    print("-" * 75)
    for cat, data in sorted(per_cat.items()):
        print(
            f"{cat:<16} | {data['support']:<8,d} | {data['detection_rate']*100:13.2f}% | "
            f"{data['precision']:<10.4f} | {data['recall']:<10.4f} | {data['f1']:<8.4f}"
        )
    print("=" * 75)

    # 6. Benchmark Comparison
    print("\nBenchmark Progression Comparison:")
    print(f"{'Metric':<25} | {'Original Baseline':<18} | {'Boosted Baseline':<18} | {'EXP-01A (Selected)':<18}")
    print("-" * 85)
    print(f"{'Test Accuracy':<25} | {'86.09%':<18} | {'87.38%':<18} | {f'{test_acc*100:.2f}%':<18}")
    print(f"{'Benign FPR':<25} | {'25.84%':<18} | {'23.97%':<18} | {f'{fpr*100:.2f}%':<18}")
    print(f"{'Attack Recall':<25} | {'95.62%':<18} | {'96.64%':<18} | {f'{attack_recall*100:.2f}%':<18}")
    print(f"{'ROC-AUC':<25} | {'0.9757':<18} | {'0.9771':<18} | {f'{roc_auc:.4f}':<18}")
    print(f"{'Threshold (tau)':<25} | {'0.5200':<18} | {'0.3400':<18} | {f'{tau:.4f}':<18}")
    print("=" * 85)

    # 7. Save Results
    out_file = exp_artifacts_dir / "exp01a_test_evaluation.json"
    results_payload = {
        "model_id": "EXP-01A",
        "description": "Standard Unweighted BCE with LeakFreePreprocessor",
        "calibrated_threshold": tau,
        "test_flows_total": len(y_test),
        "test_accuracy": test_acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "test_precision": precision,
        "test_recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "benign_fpr": fpr,
        "attack_recall": attack_recall,
        "confusion_matrix": cm,
        "per_category_metrics": per_cat,
        "comparison_against_baselines": {
            "original_accuracy": 0.8609,
            "previous_boosted_accuracy": 0.8738,
            "accuracy_delta_vs_original": round(test_acc - 0.8609, 4),
            "accuracy_delta_vs_boosted": round(test_acc - 0.8738, 4),
            "original_fpr": 0.2584,
            "previous_boosted_fpr": 0.2397,
            "fpr_delta_vs_boosted": round(fpr - 0.2397, 4),
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nEvaluation payload saved to: {out_file}")


if __name__ == "__main__":
    main()

