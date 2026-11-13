"""Single, One-Time Final Evaluation on Official UNSW-NB15 Testing Set.

STRICT PROTOCOL RULES:
1. Executed ONLY ONCE after user approval of the 5-fold cross-validation table.
2. Evaluates the single chosen model checkpoint from CV.
3. Preprocessor was fitted STRICTLY on deduplicated training data; NEVER touched test data.
4. Threshold is fixed from CV out-of-fold calibration (no tuning on test set).
5. Never iterate, retrain, or modify hyperparameters after seeing test metrics.
6. Reporting compliance:
   - Attack families show ONLY detection rate (recall).
   - Normal class reported as specificity and FPR (not F1 of 0).
   - Labeled: 'binary attack vs normal, official testing split'.
   - Notes paper benchmark differences (setup differences vs 95.30%).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ids_rag.unsw.preprocessor import UNSWStrictPreprocessor

TEST_CSV_PATH = REPO_ROOT / "data" / "unsw" / "UNSW_NB15_testing-set.csv"
CV_MODELS_DIR = REPO_ROOT / "models_unsw" / "cv"
MODEL_PATH = CV_MODELS_DIR / "hgb_tuned_deep_full_train.joblib"
PREP_PATH = CV_MODELS_DIR / "unsw_strict_preprocessor_full_train.joblib"
FINAL_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "v2" / "unsw_eval" / "final_test"


class NpEncoder(json.JSONEncoder):
    """Serialize NumPy primitives and arrays to clean JSON."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def evaluate_final_test_set(tau: float = 0.4900) -> dict[str, Any]:
    print("=" * 85)
    print("UNSW-NB15 OFFICIAL HELD-OUT TEST EVALUATION (SINGLE FINAL EXECUTION)")
    print("Label: binary attack vs normal, official testing split")
    print(f"Calibrated Threshold: tau = {tau:.4f}")
    print("=" * 85)

    assert TEST_CSV_PATH.exists(), f"Official testing CSV not found at {TEST_CSV_PATH}"
    assert MODEL_PATH.exists(), f"Model checkpoint not found at {MODEL_PATH}"
    assert PREP_PATH.exists(), f"Preprocessor artifact not found at {PREP_PATH}"

    # 1. Load artifacts
    print(f"Loading preprocessor fitted strictly on training data from: {PREP_PATH}...")
    preprocessor: UNSWStrictPreprocessor = joblib.load(PREP_PATH)
    print(f"Loading trained model from: {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # 2. Load official testing set
    print(f"Loading official testing set from: {TEST_CSV_PATH}...")
    df_test = pd.read_csv(TEST_CSV_PATH)
    n_test = len(df_test)
    print(f"Total official test flows: {n_test:,}")

    y_test = df_test["label"].to_numpy(dtype=np.float32)
    test_attack_cats = df_test["attack_cat"].fillna("Normal").astype(str).to_numpy()

    # 3. Transform strictly through training preprocessor
    print("Transforming test data through preprocessor fitted strictly on training data...")
    X_test = preprocessor.transform(df_test)
    assert X_test.shape[1] == 72, f"Expected 72 features, got {X_test.shape[1]}"
    assert np.all(np.isfinite(X_test)), "Non-finite values detected in preprocessed test features"

    # 4. Predict probabilities
    print("Performing inference on official testing set...")
    t0 = time.time()
    test_probs = model.predict_proba(X_test)[:, 1]
    infer_time = time.time() - t0
    print(f"Inference complete in {infer_time:.2f}s ({n_test / infer_time:,.0f} flows/sec)")

    # 5. Apply fixed calibrated threshold
    test_preds = (test_probs >= tau).astype(int)

    # 6. Overall Metrics
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    test_acc = float(accuracy_score(y_test, test_preds))
    test_rec = float(recall_score(y_test, test_preds, zero_division=0))
    test_prec = float(precision_score(y_test, test_preds, zero_division=0))
    macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    spec = float(tn / (tn + fp))
    fpr = float(fp / (tn + fp))

    print("\n" + "=" * 85)
    print("OFFICIAL TEST SPLIT EVALUATION METRICS")
    print("=" * 85)
    print(f"Total Test Flows:          {n_test:,}")
    print(f"Calibrated Threshold:      tau = {tau:.4f}")
    print(f"Test Accuracy:             {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Test Macro F1:             {macro_f1:.4f}")
    print(f"Test Weighted F1:          {weighted_f1:.4f}")
    print(f"Test Precision:            {test_prec:.4f}")
    print(f"Test Attack Recall:        {test_rec:.4f} ({test_rec * 100:.2f}%)")
    print(f"Specificity (Benign):      {spec:.4f} ({spec * 100:.2f}%)")
    print(f"False Positive Rate (FPR): {fpr:.4f} ({fpr * 100:.2f}%)")
    print(f"ROC-AUC:                   {roc_auc:.4f}")
    print(f"Confusion Matrix:          TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")

    # 7. Compliant Per-Family Table
    print("\n" + "=" * 85)
    print("PER-FAMILY DETECTION RATES ON OFFICIAL TEST SPLIT")
    print("Reporting Fix: Attack families show recall ONLY. Normal shows Specificity & FPR.")
    print("Label: binary attack vs normal, official testing split")
    print("=" * 85)
    print(f"{'Category':<20} | {'Total Flows':<12} | {'Detected (TP/TN)':<18} | {'Detection Rate / Specificity':<30}")
    print("-" * 85)

    per_family_results: dict[str, dict[str, Any]] = {}
    cats, counts = np.unique(test_attack_cats, return_counts=True)
    for cat, total in zip(cats, counts):
        mask = (test_attack_cats == cat)
        if cat == "Normal":
            cat_tn = int(np.sum((test_preds[mask] == 0)))
            cat_fp = int(np.sum((test_preds[mask] == 1)))
            cat_spec = float(cat_tn / total)
            cat_fpr = float(cat_fp / total)
            per_family_results[cat] = {
                "total_flows": int(total),
                "true_negative": cat_tn,
                "false_positive": cat_fp,
                "specificity": cat_spec,
                "false_positive_rate": cat_fpr,
            }
            print(f"{cat:<20} | {total:<12,d} | TN={cat_tn:,d}, FP={cat_fp:,d}      | Specificity: {cat_spec * 100:5.2f}% (FPR: {cat_fpr * 100:5.2f}%)")
        else:
            cat_tp = int(np.sum((test_preds[mask] == 1)))
            cat_fn = int(np.sum((test_preds[mask] == 0)))
            cat_rec = float(cat_tp / total)
            per_family_results[cat] = {
                "total_flows": int(total),
                "detected": cat_tp,
                "missed": cat_fn,
                "recall_detection_rate": cat_rec,
            }
            print(f"{cat:<20} | {total:<12,d} | TP={cat_tp:,d}, FN={cat_fn:,d}      | Recall: {cat_rec * 100:5.2f}%")

    print("\n" + "=" * 85)
    print("SCIENTIFIC BENCHMARK CONTEXT:")
    print("Note: Do NOT compare these results to the paper's 95.30% benchmark.")
    print("Reason: The original Moustafa & Slay (2015) paper evaluated a different setup/partition.")
    print("All results here are strictly evaluated under the official binary testing split without iteration.")
    print("=" * 85)

    # 8. Save artifact
    FINAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "metadata": {
            "protocol_label": "binary attack vs normal, official testing split",
            "model_type": "HistGradientBoostingClassifier (HGB_Tuned_Deep)",
            "checkpoint": str(MODEL_PATH),
            "calibrated_threshold": tau,
            "feature_count": 72,
            "test_flows_count": n_test,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "overall_metrics": {
            "accuracy": test_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "precision": test_prec,
            "recall": test_rec,
            "specificity": spec,
            "false_positive_rate": fpr,
            "roc_auc": roc_auc,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        },
        "per_family_detection": per_family_results,
    }

    out_file = FINAL_ARTIFACTS_DIR / "unsw_final_test_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, cls=NpEncoder)

    print(f"\nFinal test evaluation saved to: {out_file}")
    return summary_data


if __name__ == "__main__":
    evaluate_final_test_set()

