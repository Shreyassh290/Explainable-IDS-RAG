"""Threshold calibration for the Boosted UNSW-NB15 model.

Evaluates dnn_unsw_boosted.pt across a calibrated probability range (0.10 to 0.70)
to locate the global optimal decision threshold tau that maximizes overall test accuracy.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root
from ids_rag.v2.train_unsw_boosted import BoostedDNN, preprocess_boosted_unsw, set_seed


def main() -> None:
    set_seed(42)
    root = get_project_root()
    data_dir = root / "data" / "unsw"
    models_dir = root / "models_unsw"
    artifacts_dir = root / "artifacts" / "v2" / "unsw_eval"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading preprocessed UNSW data for boosted threshold calibration...", flush=True)

    X_train, y_train, X_test, y_test, feature_names, test_attack_cats, scaler = preprocess_boosted_unsw(data_dir)

    # 15% validation split
    n_train_full = len(X_train)
    val_size = int(n_train_full * 0.15)
    indices = np.random.permutation(n_train_full)
    val_idx = indices[:val_size]

    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # Load trained boosted model
    model_path = models_dir / "dnn_unsw_boosted.pt"
    if not model_path.exists():
        model_path = models_dir / "dnn_unsw_attack.pt"

    model = BoostedDNN(
        in_features=len(feature_names),
        hidden=[256, 128, 64, 32],
        dropout=0.25,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict validation probabilities stably
    val_tensor = torch.from_numpy(X_val).to(device)
    with torch.no_grad():
        val_logits = model(val_tensor).squeeze(1).cpu().numpy()
        val_probs = expit(val_logits)

    # Search over a wider range [0.10, 0.70] with 121 points
    thresholds = np.linspace(0.10, 0.70, 121)
    best_tau = 0.50
    best_val_score = 0.0

    for tau in thresholds:
        preds = (val_probs >= tau).astype(int)
        score = accuracy_score(y_val, preds)
        if score > best_val_score:
            best_val_score = score
            best_tau = tau

    print(f"Optimal decision threshold calibrated on validation split: tau = {best_tau:.4f}")
    print(f"Validation accuracy at optimal threshold: {best_val_score*100:.2f}%\n")

    # Evaluate on held-out test set with tuned threshold
    test_tensor = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        test_logits = model(test_tensor).squeeze(1).cpu().numpy()
        test_probs = expit(test_logits)
        tuned_preds = (test_probs >= best_tau).astype(int)

    tuned_acc = float(accuracy_score(y_test, tuned_preds))
    tuned_macro_f1 = float(f1_score(y_test, tuned_preds, average="macro"))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    cm = confusion_matrix(y_test, tuned_preds).tolist()
    tn, fp, fn, tp = confusion_matrix(y_test, tuned_preds).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Per-category accuracy breakdown
    df_eval = pd.DataFrame({"attack_cat": test_attack_cats, "y_true": y_test, "y_pred": tuned_preds})
    per_cat: dict[str, dict[str, Any]] = {}
    for cat_name, group in df_eval.groupby("attack_cat"):
        cat_acc = float(accuracy_score(group["y_true"], group["y_pred"]))
        cat_support = int(len(group))
        per_cat[str(cat_name)] = {"accuracy": cat_acc, "support": cat_support}

    print("=" * 60)
    print("CALIBRATED BOOSTED UNSW-NB15 TEST EVALUATION SUMMARY")
    print(f"Calibrated Threshold (tau): {best_tau:.4f}")
    print(f"Test Accuracy:              {tuned_acc:.4f} ({tuned_acc*100:.2f}%)")
    print(f"Test Macro F1-Score:        {tuned_macro_f1:.4f}")
    print(f"ROC-AUC:                    {roc_auc:.4f}")
    print(f"Benign False Positive Rate: {fpr*100:.2f}%")
    print("=" * 60)
    print("\nPer-Category Detection Rates at Calibrated Threshold:")
    for cat, data in sorted(per_cat.items()):
        print(f"  - {cat:16s}: {data['accuracy']*100:6.2f}% (Support: {data['support']:,})")

    # Save to metrics
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "unsw_metrics.json"
    summary = {
        "test_flows_total": len(y_test),
        "test_accuracy": tuned_acc,
        "validation_accuracy": best_val_score,
        "macro_f1": tuned_macro_f1,
        "roc_auc": roc_auc,
        "benign_fpr": fpr,
        "optimal_threshold": float(best_tau),
        "confusion_matrix": cm,
        "per_category_accuracy": per_cat,
        "paper_comparison": {
            "paper_accuracy": 0.9530,
            "our_accuracy": tuned_acc,
            "difference": round(tuned_acc - 0.9530, 4),
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nUpdated metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
