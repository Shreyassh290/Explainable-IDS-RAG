"""One-time held-out test evaluation of candidate EXP-04B (HistGradientBoostingClassifier).

Evaluates the exact checkpoint selected on validation data (model_EXP-04B.joblib)
using the exact leak-free preprocessor (preprocessor_exp04.joblib)
and fixed validation threshold (tau = 0.4850) on UNSW_NB15_testing-set.csv (82,332 flows).

Writes strictly to artifacts/v2/unsw_eval/experiments/exp04/exp04b_test_evaluation.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root
from scripts.v2_exp01_loss_and_leakage import NOISE_COLUMNS
from scripts.v2_exp02_feature_engineering import LeakFreePreprocessorV2


class NpEncoder(json.JSONEncoder):
    """Robust JSON encoder for NumPy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main() -> None:
    root = get_project_root()
    data_dir = root / "data" / "unsw"
    exp04_models_dir = root / "models_unsw" / "experiments" / "exp04"
    exp04_artifacts_dir = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp04"
    exp01_artifacts_dir = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp01"
    exp02_artifacts_dir = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp02"

    model_path = exp04_models_dir / "model_EXP-04B.joblib"
    prep_path = exp04_models_dir / "preprocessor_exp04.joblib"
    feat_path = exp04_models_dir / "feature_names_exp04.json"
    val_summary_path = exp04_artifacts_dir / "exp04_val_summary.json"

    print("Verifying EXP-04B artifacts before execution...", flush=True)
    assert model_path.exists(), f"Model checkpoint not found: {model_path}"
    assert prep_path.exists(), f"Preprocessor artifact not found: {prep_path}"
    assert feat_path.exists(), f"Feature names artifact not found: {feat_path}"
    assert val_summary_path.exists(), f"Validation summary not found: {val_summary_path}"

    with open(feat_path, encoding="utf-8") as f:
        feature_names = json.load(f)
    assert len(feature_names) == 72, f"Expected 72 features, found {len(feature_names)}"

    with open(val_summary_path, encoding="utf-8") as f:
        val_summary = json.load(f)
    tau = float(val_summary["selected_threshold"])
    assert np.isclose(tau, 0.4850, atol=1e-4), f"Expected validation threshold 0.4850, got {tau}"

    print(f"Artifacts verified. Using fixed validation-calibrated threshold: tau = {tau:.4f}", flush=True)

    # 1. Load Preprocessor and Test CSV
    preprocessor: LeakFreePreprocessorV2 = joblib.load(prep_path)
    test_csv_path = data_dir / "UNSW_NB15_testing-set.csv"
    print(f"Loading official testing split from: {test_csv_path}...", flush=True)
    df_test = pd.read_csv(test_csv_path)

    for c in NOISE_COLUMNS:
        if c in df_test.columns:
            df_test = df_test.drop(columns=[c])

    y_test = df_test["label"].to_numpy(dtype=np.float32)
    test_attack_cats = df_test["attack_cat"].fillna("Normal").astype(str).to_numpy()

    print("Transforming test data through preprocessor fitted strictly on training data...", flush=True)
    X_test = preprocessor.transform(df_test)
    assert X_test.shape[1] == 72, f"Engineered test features count {X_test.shape[1]} != 72"
    assert np.all(np.isfinite(X_test)), "Non-finite values detected in preprocessed test features"

    # 2. Load Model and Predict
    clf: HistGradientBoostingClassifier = joblib.load(model_path)
    test_probs = clf.predict_proba(X_test)[:, 1]
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
    specificity = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0
    attack_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # 4. Per-Attack-Family Breakdown
    df_eval = pd.DataFrame({"attack_cat": test_attack_cats, "y_true": y_test, "y_pred": test_preds})
    per_cat: dict[str, dict[str, Any]] = {}
    fuzzers_recall = 0.0
    fuzzers_fn = 0
    analysis_recall = 0.0
    analysis_fn = 0

    for cat_name, group in df_eval.groupby("attack_cat"):
        cat_acc = float(accuracy_score(group["y_true"], group["y_pred"]))
        cat_support = int(len(group))
        cat_tp = int(np.sum((group["y_true"] == 1) & (group["y_pred"] == 1)))
        cat_fn = int(np.sum((group["y_true"] == 1) & (group["y_pred"] == 0)))
        cat_recall = float(cat_tp / (cat_tp + cat_fn)) if (cat_tp + cat_fn) > 0 else (1.0 - fpr if cat_name == "Normal" else 0.0)

        p_val, r_val, f1_val, _ = precision_recall_fscore_support(
            group["y_true"], group["y_pred"], average="binary", zero_division=0
        ) if cat_name != "Normal" else (float(tn / (tn + fn)), float(tn / (tn + fp)), 0.0, None)

        per_cat[str(cat_name)] = {
            "support": int(cat_support),
            "missed_fn": int(cat_fn) if cat_name != "Normal" else 0,
            "false_positives": int(fp) if cat_name == "Normal" else 0,
            "detection_rate": float(cat_recall if cat_name != "Normal" else cat_acc),
            "precision": float(p_val),
            "recall": float(r_val) if cat_name != "Normal" else float(cat_acc),
            "f1": float(f1_val),
        }

        if cat_name == "Fuzzers":
            fuzzers_recall = float(cat_recall)
            fuzzers_fn = int(cat_fn)
        elif cat_name == "Analysis":
            analysis_recall = float(cat_recall)
            analysis_fn = int(cat_fn)

    # 5. Output Summary Table
    print("\n" + "=" * 80)
    print("EXP-04B HELD-OUT TEST EVALUATION SUMMARY (UNSW-NB15 TEST SPLIT)")
    print("=" * 80)
    print(f"Model Architecture:        HistGradientBoostingClassifier (EXP-04B)")
    print(f"Total Test Flows:          {len(y_test):,}")
    print(f"Calibrated Threshold:      tau = {tau:.4f}")
    print(f"Test Accuracy:             {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Macro F1:             {macro_f1:.4f}")
    print(f"Test Weighted F1:          {weighted_f1:.4f}")
    print(f"Test Precision:            {precision:.4f}")
    print(f"Test Recall (Attacks):     {attack_recall:.4f} ({attack_recall*100:.2f}%)")
    print(f"Specificity (Benign):      {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"ROC-AUC:                   {roc_auc:.4f}")
    print(f"PR-AUC:                    {pr_auc:.4f}")
    print(f"Benign False Positive Rate:{fpr*100:.2f}% (FP: {fp:,} / {fp+tn:,})")
    print("-" * 80)
    print("Confusion Matrix:")
    print(f"  [TN: {tn:>6,},  FP: {fp:>6,}]")
    print(f"  [FN: {fn:>6,},  TP: {tp:>6,}]")
    print("=" * 80)

    print("\nPer-Attack-Family Classification Metrics:")
    print(f"{'Category':<16} | {'Support':<8} | {'Missed (FN)':<11} | {'Detection Rate':<15} | {'Precision':<10} | {'Recall':<10} | {'F1':<8}")
    print("-" * 85)
    for cat, data in sorted(per_cat.items()):
        missed_str = f"{data['missed_fn']:,}" if cat != "Normal" else f"FP: {fp:,}"
        print(
            f"{cat:<16} | {data['support']:<8,d} | {missed_str:<11} | {data['detection_rate']*100:13.2f}% | "
            f"{data['precision']:<10.4f} | {data['recall']:<10.4f} | {data['f1']:<8.4f}"
        )
    print("=" * 85)

    # 6. Load Historical Baselines for Benchmark Comparison
    exp01a_eval_path = exp01_artifacts_dir / "exp01a_test_evaluation.json"
    exp02_eval_path = exp02_artifacts_dir / "exp02_test_evaluation.json"

    exp01a_test = {}
    if exp01a_eval_path.exists():
        with open(exp01a_eval_path, encoding="utf-8") as f:
            exp01a_test = json.load(f)

    exp02_test = {}
    if exp02_eval_path.exists():
        with open(exp02_eval_path, encoding="utf-8") as f:
            exp02_test = json.load(f)

    e1_acc = f"{exp01a_test.get('test_accuracy', 0.8608)*100:.2f}%"
    e1_fpr = f"{exp01a_test.get('benign_fpr', 0.2891)*100:.2f}%"
    e1_rec = f"{exp01a_test.get('test_recall', 0.9832)*100:.2f}%"
    e1_fuzz = f"{exp01a_test.get('per_category_metrics', {}).get('Fuzzers', {}).get('detection_rate', 0.8981)*100:.2f}%"
    cm1 = exp01a_test.get("confusion_matrix")
    e1_fn_val = cm1[1][0] if isinstance(cm1, list) else (cm1.get("fn", 760) if isinstance(cm1, dict) else 760)
    e1_fn = f"{e1_fn_val:,}"
    e1_auc = f"{exp01a_test.get('roc_auc', 0.9782):.4f}"

    e2_acc = f"{exp02_test.get('test_accuracy', 0.8585)*100:.2f}%"
    e2_fpr = f"{exp02_test.get('benign_fpr', 0.3020)*100:.2f}%"
    e2_rec = f"{exp02_test.get('test_recall', 0.9895)*100:.2f}%"
    e2_fuzz = f"{exp02_test.get('per_category_metrics', {}).get('Fuzzers', {}).get('detection_rate', 0.9343)*100:.2f}%"
    cm2 = exp02_test.get("confusion_matrix")
    e2_fn_val = cm2[1][0] if isinstance(cm2, list) else (cm2.get("fn", 475) if isinstance(cm2, dict) else 475)
    e2_fn = f"{e2_fn_val:,}"
    e2_auc = f"{exp02_test.get('roc_auc', 0.9810):.4f}"

    e4_acc = f"{test_acc*100:.2f}%"
    e4_fpr = f"{fpr*100:.2f}%"
    e4_rec = f"{attack_recall*100:.2f}%"
    e4_fuzz = f"{fuzzers_recall*100:.2f}%"
    e4_fn = f"{fn:,}"
    e4_auc = f"{roc_auc:.4f}"

    print("\nBenchmark Progression Comparison (Held-Out Test Set):")
    print(f"{'Metric':<22} | {'Orig Base':<10} | {'Boosted DNN':<12} | {'EXP-01A (66)':<14} | {'EXP-02 (72)':<14} | {'EXP-04B (HistGBDT)':<18}")
    print("-" * 105)
    print(f"{'Test Accuracy':<22} | {'86.09%':<10} | {'87.38%':<12} | {e1_acc:<14} | {e2_acc:<14} | {e4_acc:<18}")
    print(f"{'Benign FPR':<22} | {'25.84%':<10} | {'23.97%':<12} | {e1_fpr:<14} | {e2_fpr:<14} | {e4_fpr:<18}")
    print(f"{'Attack Recall':<22} | {'95.62%':<10} | {'96.64%':<12} | {e1_rec:<14} | {e2_rec:<14} | {e4_rec:<18}")
    print(f"{'Fuzzers Recall':<22} | {'77.24%':<10} | {'77.83%':<12} | {e1_fuzz:<14} | {e2_fuzz:<14} | {e4_fuzz:<18}")
    print(f"{'Missed Attacks (FN)':<22} | {'1,985':<10} | {'1,523':<12} | {e1_fn:<14} | {e2_fn:<14} | {e4_fn:<18}")
    print(f"{'ROC-AUC':<22} | {'0.9757':<10} | {'0.9771':<12} | {e1_auc:<14} | {e2_auc:<14} | {e4_auc:<18}")
    print(f"{'Threshold (tau)':<22} | {'0.5200':<10} | {'0.3400':<12} | {'0.4650':<14} | {'0.4450':<14} | {f'{tau:.4f}':<18}")
    print("=" * 105)

    # 7. Save Complete Evaluation Payload
    out_file = exp04_artifacts_dir / "exp04b_test_evaluation.json"
    results_payload = {
        "model_id": "EXP-04B",
        "description": "Deep Regularized HistGradientBoostingClassifier (72 features)",
        "model_checkpoint_path": str(model_path),
        "preprocessor_path": str(prep_path),
        "feature_names_path": str(feat_path),
        "calibrated_threshold": float(tau),
        "test_flows_total": len(y_test),
        "test_accuracy": test_acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "test_precision": precision,
        "test_recall": attack_recall,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "benign_fpr": fpr,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "per_category_metrics": per_cat,
        "weak_categories_focus": {
            "fuzzers_recall": fuzzers_recall,
            "fuzzers_missed": fuzzers_fn,
            "analysis_recall": analysis_recall,
            "analysis_missed": analysis_fn,
        },
        "comparison_table": {
            "original_accuracy": 0.8609,
            "boosted_accuracy": 0.8738,
            "exp01a_accuracy": float(exp01a_test.get("test_accuracy", 0.8608)),
            "exp02_accuracy": float(exp02_test.get("test_accuracy", 0.8585)),
            "exp04b_accuracy": test_acc,
            "exp04b_fpr": fpr,
            "exp04b_attack_recall": attack_recall,
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, cls=NpEncoder)

    print(f"\nEvaluation payload saved to: {out_file}")


if __name__ == "__main__":
    main()

