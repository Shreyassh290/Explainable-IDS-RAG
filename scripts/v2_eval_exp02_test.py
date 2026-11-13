"""One-time held-out test evaluation of EXP-02 (72 Features, Standard Unweighted BCE).

Evaluates the exact checkpoint selected on validation data (model_EXP-02.pt)
using the exact leak-free preprocessor (preprocessor_exp02.joblib)
and fixed validation threshold (tau = 0.4450) on UNSW_NB15_testing-set.csv (82,332 flows).

Writes strictly to artifacts/v2/unsw_eval/experiments/exp02/.
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
from scripts.v2_exp01_loss_and_leakage import NOISE_COLUMNS
from scripts.v2_exp02_feature_engineering import (
    BoostedDNNV2,
    LeakFreePreprocessorV2,
    compute_domain_features,
)


class NpEncoder(json.JSONEncoder):
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
    exp02_models_dir = root / "models_unsw" / "experiments" / "exp02"
    exp02_artifacts_dir = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp02"
    exp01_artifacts_dir = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp01"

    model_path = exp02_models_dir / "model_EXP-02.pt"
    prep_path = exp02_models_dir / "preprocessor_exp02.joblib"
    feat_path = exp02_models_dir / "feature_names_exp02.json"
    val_summary_path = exp02_artifacts_dir / "exp02_val_summary.json"

    print("Verifying EXP-02 artifacts before execution...", flush=True)
    assert model_path.exists(), f"Model checkpoint not found: {model_path}"
    assert prep_path.exists(), f"Preprocessor artifact not found: {prep_path}"
    assert feat_path.exists(), f"Feature names artifact not found: {feat_path}"
    assert val_summary_path.exists(), f"Validation summary not found: {val_summary_path}"

    with open(feat_path, encoding="utf-8") as f:
        feature_names = json.load(f)
    assert len(feature_names) == 72, f"Expected 72 features, found {len(feature_names)}"

    with open(val_summary_path, encoding="utf-8") as f:
        val_summary = json.load(f)
    tau = float(val_summary["calibrated_threshold"])
    assert np.isclose(tau, 0.4450, atol=1e-4), f"Expected validation threshold 0.4450, got {tau}"

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoostedDNNV2(in_features=72, hidden=[256, 128, 64, 32], dropout=0.25).to(device)
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
    specificity = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0
    attack_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # 4. Per-Attack-Family Breakdown
    df_eval = pd.DataFrame({"attack_cat": test_attack_cats, "y_true": y_test, "y_pred": test_preds})
    per_cat: dict[str, dict[str, Any]] = {}
    fuzzers_recall = 0.0
    fuzzers_fn = 0

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
            "support": cat_support,
            "missed_fn": cat_fn if cat_name != "Normal" else 0,
            "false_positives": fp if cat_name == "Normal" else 0,
            "detection_rate": cat_recall if cat_name != "Normal" else cat_acc,
            "precision": float(p_val),
            "recall": float(r_val) if cat_name != "Normal" else cat_acc,
            "f1": float(f1_val),
        }

        if cat_name == "Fuzzers":
            fuzzers_recall = cat_recall
            fuzzers_fn = cat_fn

    # 5. Load EXP-01A test evaluation for side-by-side comparison
    exp01a_eval_path = exp01_artifacts_dir / "exp01a_test_evaluation.json"
    exp01a_test = {}
    if exp01a_eval_path.exists():
        with open(exp01a_eval_path, encoding="utf-8") as f:
            exp01a_test = json.load(f)

    exp01a_fp = exp01a_test.get("confusion_matrix", [[0, 0], [0, 0]])[0][1] if exp01a_test else 10698
    exp01a_fn = exp01a_test.get("confusion_matrix", [[0, 0], [0, 0]])[1][0] if exp01a_test else 760
    exp01a_fuzzers_rec = exp01a_test.get("per_category_metrics", {}).get("Fuzzers", {}).get("detection_rate", 0.8981) if exp01a_test else 0.8981

    fp_delta = fp - exp01a_fp
    fn_delta = fn - exp01a_fn
    fuzz_rec_delta = fuzzers_recall - exp01a_fuzzers_rec

    # 6. Output Summary Table
    print("\n" + "=" * 75)
    print("EXP-02 HELD-OUT TEST EVALUATION SUMMARY (UNSW-NB15 TEST SPLIT)")
    print("=" * 75)
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
    print("-" * 75)
    print("Confusion Matrix:")
    print(f"  [TN: {tn:>6,},  FP: {fp:>6,}]")
    print(f"  [FN: {fn:>6,},  TP: {tp:>6,}]")
    print("=" * 75)

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

    # 7. Comprehensive Benchmark Comparison
    print("\nBenchmark Progression Comparison (Held-Out Test Set):")
    print(f"{'Metric':<24} | {'Orig Baseline':<14} | {'Boosted Base':<14} | {'EXP-01A (66 Feat)':<18} | {'EXP-02 (72 Feat)':<18}")
    print("-" * 96)
    print(f"{'Test Accuracy':<24} | {'86.09%':<14} | {'87.38%':<14} | {'86.08%':<18} | {f'{test_acc*100:.2f}%':<18}")
    print(f"{'Benign FPR':<24} | {'25.84%':<14} | {'23.97%':<14} | {'28.91%':<18} | {f'{fpr*100:.2f}%':<18}")
    print(f"{'Attack Recall':<24} | {'95.62%':<14} | {'96.64%':<14} | {'98.32%':<18} | {f'{attack_recall*100:.2f}%':<18}")
    print(f"{'Fuzzers Recall':<24} | {'77.24%':<14} | {'77.83%':<14} | {'89.81%':<18} | {f'{fuzzers_recall*100:.2f}%':<18}")
    print(f"{'Missed Attacks (FN)':<24} | {'1,985':<14} | {'1,523':<14} | {'760':<18} | {f'{fn:,}':<18}")
    print(f"{'ROC-AUC':<24} | {'0.9757':<14} | {'0.9771':<14} | {'0.9782':<18} | {f'{roc_auc:.4f}':<18}")
    print(f"{'PR-AUC':<24} | {'—':<14} | {'—':<14} | {'0.9835':<18} | {f'{pr_auc:.4f}':<18}")
    print(f"{'Threshold (tau)':<24} | {'0.5200':<14} | {'0.3400':<14} | {'0.4650':<18} | {f'{tau:.4f}':<18}")
    print("=" * 96)

    print("\nDelta vs. EXP-01A:")
    print(f"  False Positives Delta: {fp_delta:+d} ({fp:,} vs {exp01a_fp:,})")
    print(f"  False Negatives Delta: {fn_delta:+d} ({fn:,} vs {exp01a_fn:,})")
    print(f"  Fuzzers Recall Delta:  {fuzz_rec_delta*100:+.2f}% ({fuzzers_recall*100:.2f}% vs {exp01a_fuzzers_rec*100:.2f}%)")

    # 8. Save Payload
    out_file = exp02_artifacts_dir / "exp02_test_evaluation.json"
    results_payload = {
        "model_id": "EXP-02",
        "description": "Domain Feature Engineering (72 features) with Standard Unweighted BCE",
        "model_checkpoint_path": str(model_path),
        "preprocessor_path": str(prep_path),
        "feature_names_path": str(feat_path),
        "calibrated_threshold": tau,
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
        "delta_vs_exp01a": {
            "accuracy_delta": round(test_acc - exp01a_test.get("test_accuracy", 0.8608), 4),
            "fpr_delta": round(fpr - exp01a_test.get("benign_fpr", 0.2891), 4),
            "attack_recall_delta": round(attack_recall - exp01a_test.get("attack_recall", 0.9832), 4),
            "fuzzers_recall_delta": round(float(fuzz_rec_delta), 4),
            "fp_count_delta": int(fp_delta),
            "fn_count_delta": int(fn_delta),
        },
        "comparison_table": {
            "original_accuracy": 0.8609,
            "boosted_accuracy": 0.8738,
            "exp01a_accuracy": 0.8608,
            "exp02_accuracy": test_acc,
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, cls=NpEncoder)

    print(f"\nEvaluation payload saved to: {out_file}")


if __name__ == "__main__":
    main()

