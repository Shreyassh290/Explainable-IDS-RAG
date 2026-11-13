"""Experiment 5 (EXP-05): Hybrid Ensemble — EXP-02 DNN + EXP-04B HistGradientBoosting.

Combines the continuous non-linear representations of EXP-02 (BoostedDNNV2)
with the discrete orthogonal partitioning of EXP-04B (HistGradientBoostingClassifier)
on the exact 72-feature leak-free representation.

Evaluates:
  1. EXP-02 alone (DNN baseline)
  2. EXP-04B alone (GBDT baseline)
  3. Equal-weight soft voting (0.50 / 0.50)
  4. Weighted soft voting grid (weights: 0.20/0.80, 0.35/0.65, 0.65/0.35, 0.80/0.20)
  5. Logistic Regression Stacking meta-learner

Strict Leak-Free Protocol:
- Zero retraining or overwriting of EXP-02 or EXP-04B checkpoints.
- Evaluated strictly on the 15% validation split (26,302 flows).
- Official test set remains completely held out (untouched).
- Preservation: All existing baseline and experimental artifacts remain untouched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root
from scripts.v2_exp01_loss_and_leakage import (
    NOISE_COLUMNS,
    analyze_threshold_pareto,
    load_and_split_unsw,
    set_seed,
)
from scripts.v2_exp02_feature_engineering import (
    BoostedDNNV2,
    LeakFreePreprocessorV2,
)


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


def evaluate_ensemble_metrics(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    val_attack_cats: np.ndarray,
    tau: float,
) -> dict[str, Any]:
    """Compute comprehensive validation metrics at a fixed decision threshold."""
    preds = (val_probs >= tau).astype(int)

    acc = float(accuracy_score(y_val, preds))
    macro_f1 = float(f1_score(y_val, preds, average="macro"))
    weighted_f1 = float(f1_score(y_val, preds, average="weighted"))
    prec = float(precision_score(y_val, preds, zero_division=0))
    rec = float(recall_score(y_val, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_val, val_probs))
    pr_auc = float(average_precision_score(y_val, val_probs))

    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    specificity = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Per-category validation breakdown
    df_cat = pd.DataFrame({"attack_cat": val_attack_cats, "y_true": y_val, "y_pred": preds})
    cat_recalls = {}
    fuzzers_rec = 0.0

    for cat_name, grp in df_cat.groupby("attack_cat"):
        c_tp = int(np.sum((grp["y_true"] == 1) & (grp["y_pred"] == 1)))
        c_fn = int(np.sum((grp["y_true"] == 1) & (grp["y_pred"] == 0)))
        rate = float(c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else float(accuracy_score(grp["y_true"], grp["y_pred"]))
        cat_recalls[str(cat_name)] = {
            "support": int(len(grp)),
            "recall": rate,
            "missed_fn": int(c_fn) if cat_name != "Normal" else 0,
        }
        if cat_name == "Fuzzers":
            fuzzers_rec = rate

    return {
        "threshold": float(tau),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "benign_fpr": fpr,
        "fuzzers_recall": fuzzers_rec,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_category_validation_recall": cat_recalls,
    }


def main() -> None:
    set_seed(42)
    root = get_project_root()
    data_dir = root / "data" / "unsw"
    exp02_models_dir = root / "models_unsw" / "experiments" / "exp02"
    exp04_models_dir = root / "models_unsw" / "experiments" / "exp04"
    exp05_models_dir = root / "models_unsw" / "experiments" / "exp05"
    exp05_artifacts_dir = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp05"

    exp05_models_dir.mkdir(parents=True, exist_ok=True)
    exp05_artifacts_dir.mkdir(parents=True, exist_ok=True)

    dnn_model_path = exp02_models_dir / "model_EXP-02.pt"
    gbdt_model_path = exp04_models_dir / "model_EXP-04B.joblib"
    prep_path = exp04_models_dir / "preprocessor_exp04.joblib"

    print("Verifying base models before ensemble construction...", flush=True)
    assert dnn_model_path.exists(), f"EXP-02 DNN checkpoint missing: {dnn_model_path}"
    assert gbdt_model_path.exists(), f"EXP-04B GBDT checkpoint missing: {gbdt_model_path}"
    assert prep_path.exists(), f"Preprocessor missing: {prep_path}"

    # 1. Load Data Splits
    df_tr, df_val, df_test = load_and_split_unsw(data_dir, val_ratio=0.15, seed=42)
    y_tr = df_tr["label"].to_numpy(dtype=np.float32)
    y_val = df_val["label"].to_numpy(dtype=np.float32)
    val_attack_cats = df_val["attack_cat"].fillna("Normal").astype(str).to_numpy()

    # 2. Transform Features with Saved Leak-Free Preprocessor (72 Features)
    preprocessor: LeakFreePreprocessorV2 = joblib.load(prep_path)
    print("Transforming train and validation features (72 features)...", flush=True)
    X_tr = preprocessor.transform(df_tr)
    X_val = preprocessor.transform(df_val)

    # 3. Load Trained Base Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dnn = BoostedDNNV2(in_features=72, hidden=[256, 128, 64, 32], dropout=0.25).to(device)
    model_dnn.load_state_dict(torch.load(dnn_model_path, map_location=device))
    model_dnn.eval()

    clf_gbdt = joblib.load(gbdt_model_path)

    # 4. Generate Predictions on Validation Set
    print("\nGenerating model probabilities on validation set (26,302 flows)...", flush=True)
    val_tensor = torch.from_numpy(X_val).to(device)
    with torch.no_grad():
        val_logits_dnn = model_dnn(val_tensor).squeeze(1).cpu().numpy()
        val_probs_dnn = expit(val_logits_dnn)

    val_probs_gbdt = clf_gbdt.predict_proba(X_val)[:, 1]

    # Positive-class verification
    assert clf_gbdt.classes_[1] == 1.0, "GBDT positive class index mismatch"
    assert len(val_probs_dnn) == len(val_probs_gbdt) == len(y_val)
    print("Probability alignment verified: both estimators output P(Y=1|X) in identical sample order.")

    # 5. Train Stacking Meta-Learner on Training Partition Predictions
    print("\nTraining Logistic Regression Stacking Meta-Learner on train predictions...", flush=True)
    t_tensor = torch.from_numpy(X_tr).to(device)
    with torch.no_grad():
        tr_logits_dnn = model_dnn(t_tensor).squeeze(1).cpu().numpy()
        tr_probs_dnn = expit(tr_logits_dnn)

    tr_probs_gbdt = clf_gbdt.predict_proba(X_tr)[:, 1]

    # Meta-features: [prob_dnn, prob_gbdt]
    X_meta_tr = np.column_stack([tr_probs_dnn, tr_probs_gbdt])
    X_meta_val = np.column_stack([val_probs_dnn, val_probs_gbdt])

    meta_learner = LogisticRegression(C=1.0, random_state=42, max_iter=200)
    meta_learner.fit(X_meta_tr, y_tr)
    val_probs_stacking = meta_learner.predict_proba(X_meta_val)[:, 1]

    # Save meta-learner
    stacking_path = exp05_models_dir / "stacking_meta_learner.joblib"
    joblib.dump(meta_learner, stacking_path)

    # 6. Ensemble Configurations to Evaluate
    ensemble_configs = [
        {
            "id": "EXP-05_DNN_Alone",
            "name": "EXP-02 DNN Alone (w=1.0)",
            "probs": val_probs_dnn,
            "weights": {"dnn": 1.0, "gbdt": 0.0},
        },
        {
            "id": "EXP-05_GBDT_Alone",
            "name": "EXP-04B GBDT Alone (w=1.0)",
            "probs": val_probs_gbdt,
            "weights": {"dnn": 0.0, "gbdt": 1.0},
        },
        {
            "id": "EXP-05_SoftVote_Equal",
            "name": "Equal Soft Voting (0.50 DNN + 0.50 GBDT)",
            "probs": 0.50 * val_probs_dnn + 0.50 * val_probs_gbdt,
            "weights": {"dnn": 0.50, "gbdt": 0.50},
        },
        {
            "id": "EXP-05_SoftVote_GBDT65",
            "name": "Weighted Soft Voting (0.35 DNN + 0.65 GBDT)",
            "probs": 0.35 * val_probs_dnn + 0.65 * val_probs_gbdt,
            "weights": {"dnn": 0.35, "gbdt": 0.65},
        },
        {
            "id": "EXP-05_SoftVote_GBDT80",
            "name": "Weighted Soft Voting (0.20 DNN + 0.80 GBDT)",
            "probs": 0.20 * val_probs_dnn + 0.80 * val_probs_gbdt,
            "weights": {"dnn": 0.20, "gbdt": 0.80},
        },
        {
            "id": "EXP-05_SoftVote_DNN65",
            "name": "Weighted Soft Voting (0.65 DNN + 0.35 GBDT)",
            "probs": 0.65 * val_probs_dnn + 0.35 * val_probs_gbdt,
            "weights": {"dnn": 0.65, "gbdt": 0.35},
        },
        {
            "id": "EXP-05_Stacking",
            "name": "Logistic Regression Stacking Meta-Learner",
            "probs": val_probs_stacking,
            "weights": {"meta_type": "LogisticRegression", "coefs": meta_learner.coef_.tolist()},
        },
    ]

    results_list = []
    best_config = None
    best_val_macro_f1 = -1.0

    print("\n" + "=" * 110)
    print("EXP-05 HYBRID ENSEMBLE VALIDATION PARETO SWEEP")
    print("=" * 110)
    print(f"{'Config ID':<24} | {'Ensemble Name':<32} | {'Val Acc':<8} | {'Val F1':<8} | {'Val AUC':<8} | {'Val FPR':<8} | {'Val Rec':<8} | {'Tau*'}")
    print("-" * 110)

    for cfg in ensemble_configs:
        probs = cfg["probs"]

        # Pareto threshold sweep (constraints: FPR <= 18%, Recall >= 96%)
        pareto = analyze_threshold_pareto(
            val_probs=probs,
            y_val=y_val,
            target_max_fpr=0.18,
            target_min_recall=0.96,
        )

        tau_star = pareto["best_constrained_tau"] if pareto["feasible_found"] else pareto["best_unconstrained_tau"]
        m = evaluate_ensemble_metrics(probs, y_val, val_attack_cats, tau=tau_star)

        entry = {
            "config_id": cfg["id"],
            "name": cfg["name"],
            "weights": cfg["weights"],
            "calibrated_threshold": round(tau_star, 4),
            "feasible_under_constraints": bool(pareto["feasible_found"]),
            "val_accuracy": m["accuracy"],
            "val_macro_f1": m["macro_f1"],
            "val_weighted_f1": m["weighted_f1"],
            "val_precision": m["precision"],
            "val_recall": m["recall"],
            "val_specificity": m["specificity"],
            "val_roc_auc": m["roc_auc"],
            "val_pr_auc": m["pr_auc"],
            "val_fpr": m["benign_fpr"],
            "val_fuzzers_recall": m["fuzzers_recall"],
            "confusion_matrix": m["confusion_matrix"],
            "per_category_validation_recall": m["per_category_validation_recall"],
            "target_eval": {
                "acc_ge_88": bool(m["accuracy"] >= 0.88),
                "recall_ge_97": bool(m["recall"] >= 0.97),
                "fpr_le_23": bool(m["benign_fpr"] <= 0.23),
                "macro_f1_ge_87": bool(m["macro_f1"] >= 0.87),
                "fuzzers_ge_90": bool(m["fuzzers_recall"] >= 0.90),
            },
        }

        results_list.append(entry)

        print(
            f"{entry['config_id']:<24} | {entry['name'][:32]:<32} | {entry['val_accuracy']*100:6.2f}% | "
            f"{entry['val_macro_f1']:<8.4f} | {entry['val_roc_auc']:<8.4f} | {entry['val_fpr']*100:6.2f}% | "
            f"{entry['val_recall']*100:6.2f}% | {entry['calibrated_threshold']:<6.4f}"
        )

        if m["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = m["macro_f1"]
            best_config = entry

    print("=" * 110)

    # 7. Identify Top Configurations by Objective
    best_acc_cfg = max(results_list, key=lambda x: x["val_accuracy"])
    best_rec_cfg = max(results_list, key=lambda x: x["val_recall"])
    best_auc_cfg = max(results_list, key=lambda x: x["val_roc_auc"])
    best_fuzz_cfg = max(results_list, key=lambda x: x["val_fuzzers_recall"])

    print("\nEnsemble Champions by Objective (Validation Split):")
    print(f"  Best Validation Accuracy: {best_acc_cfg['name']} ({best_acc_cfg['val_accuracy']*100:.2f}%)")
    print(f"  Best Attack Detection:    {best_rec_cfg['name']} (Recall: {best_rec_cfg['val_recall']*100:.2f}%, Fuzzers: {best_rec_cfg['val_fuzzers_recall']*100:.2f}%)")
    print(f"  Best Ranking Capacity:    {best_auc_cfg['name']} (ROC-AUC: {best_auc_cfg['val_roc_auc']:.4f}, PR-AUC: {best_auc_cfg['val_pr_auc']:.4f})")
    print(f"  Best Fuzzers Recall:      {best_fuzz_cfg['name']} ({best_fuzz_cfg['val_fuzzers_recall']*100:.2f}%)")
    print(f"  Best Balanced Champion:   {best_config['name']} (Macro F1: {best_config['val_macro_f1']:.4f}, Acc: {best_config['val_accuracy']*100:.2f}%, FPR: {best_config['val_fpr']*100:.2f}%)")

    # 8. Save Full Validation Payload
    full_payload = {
        "experiment_id": "EXP-05",
        "description": "Hybrid Ensemble combining EXP-02 (DNN) and EXP-04B (HistGBDT) on 72 features",
        "dnn_checkpoint_path": str(dnn_model_path),
        "gbdt_checkpoint_path": str(gbdt_model_path),
        "preprocessor_path": str(prep_path),
        "val_flows_total": len(y_val),
        "selected_best_balanced_config": best_config["config_id"],
        "selected_best_accuracy_config": best_acc_cfg["config_id"],
        "selected_best_attack_detection_config": best_rec_cfg["config_id"],
        "best_balanced_metrics": best_config,
        "all_ensemble_configurations": results_list,
    }

    out_file = exp05_artifacts_dir / "exp05_val_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2, cls=NpEncoder)

    print(f"\nAll EXP-05 validation artifacts saved to: {out_file}")


if __name__ == "__main__":
    main()

