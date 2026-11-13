"""Experiment 4 (EXP-04): HistGradientBoostingClassifier — Implementation & Validation.

Evaluates tree-based decision boundaries on the exact 72-feature representation
from EXP-02 to investigate whether discrete threshold rules can suppress benign false alarms
while retaining high attack and Fuzzers recall.

Strict Leak-Free Protocol:
- Partitioning: 85% train (149,039) / 15% val (26,302) with seed=42 (identical to EXP-01 & EXP-02).
- Preprocessing: LeakFreePreprocessorV2 fitted EXCLUSIVELY on df_tr.
- Model Selection & Threshold Calibration: Strictly on validation predictions (val_probs, y_val).
- Official test split: HELD OUT (untouched).
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root
from scripts.v2_exp01_loss_and_leakage import (
    analyze_threshold_pareto,
    load_and_split_unsw,
    set_seed,
)
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


def evaluate_validation_metrics(
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
    exp04_models_dir = root / "models_unsw" / "experiments" / "exp04"
    exp04_artifacts_dir = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp04"

    exp04_models_dir.mkdir(parents=True, exist_ok=True)
    exp04_artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXP-04: HIST-GRADIENT-BOOSTING VALIDATION EXPERIMENTS (72 FEATURES)")
    print("=" * 80)

    # 1. Leak-Free Split
    df_tr, df_val, df_test = load_and_split_unsw(data_dir, val_ratio=0.15, seed=42)
    y_tr = df_tr["label"].to_numpy(dtype=np.float32)
    y_val = df_val["label"].to_numpy(dtype=np.float32)
    val_attack_cats = df_val["attack_cat"].fillna("Normal").astype(str).to_numpy()

    # 2. Strict Preprocessor fit on df_tr ONLY
    print("\nFitting LeakFreePreprocessorV2 strictly on df_tr (72 features)...", flush=True)
    preprocessor = LeakFreePreprocessorV2()
    preprocessor.fit(df_tr)

    feature_names = preprocessor.feature_names_
    assert len(feature_names) == 72, f"Expected 72 features, got {len(feature_names)}"

    X_tr = preprocessor.transform(df_tr)
    X_val = preprocessor.transform(df_val)

    assert X_tr.shape[1] == 72 and X_val.shape[1] == 72
    assert np.all(np.isfinite(X_tr)) and np.all(np.isfinite(X_val))

    # Save Preprocessor Artifact
    joblib.dump(preprocessor, exp04_models_dir / "preprocessor_exp04.joblib")
    with open(exp04_models_dir / "feature_names_exp04.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # 3. Controlled Hyperparameter Configurations
    configurations = [
        {
            "config_id": "EXP-04A",
            "name": "Standard HistGBDT (lr=0.1, max_leaf=31)",
            "params": {
                "loss": "log_loss",
                "learning_rate": 0.1,
                "max_iter": 300,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.0,
                "early_stopping": True,
                "n_iter_no_change": 10,
                "random_state": 42,
            },
        },
        {
            "config_id": "EXP-04B",
            "name": "Deep Regularized HistGBDT (lr=0.05, max_leaf=63, l2=1.0)",
            "params": {
                "loss": "log_loss",
                "learning_rate": 0.05,
                "max_iter": 400,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 50,
                "l2_regularization": 1.0,
                "early_stopping": True,
                "n_iter_no_change": 10,
                "random_state": 42,
            },
        },
        {
            "config_id": "EXP-04C",
            "name": "Conservative HistGBDT (lr=0.05, max_leaf=31, l2=5.0)",
            "params": {
                "loss": "log_loss",
                "learning_rate": 0.05,
                "max_iter": 300,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 100,
                "l2_regularization": 5.0,
                "early_stopping": True,
                "n_iter_no_change": 10,
                "random_state": 42,
            },
        },
    ]

    exp_summaries = []
    best_config_model = None
    best_config_id = None
    best_config_val_f1 = -1.0
    best_config_metrics = None

    for cfg in configurations:
        cfg_id = cfg["config_id"]
        cfg_name = cfg["name"]
        params = cfg["params"]

        print(f"\n--- Training {cfg_id}: {cfg_name} ---", flush=True)
        clf = HistGradientBoostingClassifier(**params)

        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t0

        print(f"Fit completed in {fit_time:.2f}s (Iters run: {clf.n_iter_})", flush=True)

        # Validation Probabilities
        val_probs = clf.predict_proba(X_val)[:, 1]

        # Multi-Metric Pareto Sweep (target: FPR <= 18%, Recall >= 96%)
        pareto = analyze_threshold_pareto(
            val_probs=val_probs,
            y_val=y_val,
            target_max_fpr=0.18,
            target_min_recall=0.96,
        )

        tau_star = pareto["best_constrained_tau"] if pareto["feasible_found"] else pareto["best_unconstrained_tau"]
        val_metrics = evaluate_validation_metrics(val_probs, y_val, val_attack_cats, tau=tau_star)

        # Save Checkpoint
        model_file = exp04_models_dir / f"model_{cfg_id}.joblib"
        joblib.dump(clf, model_file)

        summary_entry = {
            "config_id": cfg_id,
            "name": cfg_name,
            "params": params,
            "n_iters": int(clf.n_iter_),
            "fit_time_s": round(fit_time, 2),
            "calibrated_threshold": round(tau_star, 4),
            "feasible_under_constraints": bool(pareto["feasible_found"]),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_fpr": val_metrics["benign_fpr"],
            "val_fuzzers_recall": val_metrics["fuzzers_recall"],
            "confusion_matrix": val_metrics["confusion_matrix"],
            "per_category_validation_recall": val_metrics["per_category_validation_recall"],
            "model_path": str(model_file),
        }

        exp_summaries.append(summary_entry)

        # Save individual Pareto curve
        with open(exp04_artifacts_dir / f"{cfg_id}_pareto.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary_entry, "curve": pareto["curve"]}, f, indent=2, cls=NpEncoder)

        if val_metrics["macro_f1"] > best_config_val_f1:
            best_config_val_f1 = val_metrics["macro_f1"]
            best_config_id = cfg_id
            best_config_model = clf
            best_config_metrics = summary_entry

    # 4. Comparative Validation Table
    # Load historical validation baselines
    exp01_summary_path = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp01" / "exp01_comparison_summary.json"
    exp02_summary_path = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp02" / "exp02_val_summary.json"

    val_exp01a = {}
    if exp01_summary_path.exists():
        with open(exp01_summary_path, encoding="utf-8") as f:
            all_e1 = json.load(f)
            val_exp01a = next((item for item in all_e1 if item["experiment_id"] == "EXP-01A"), {})

    val_exp02 = {}
    if exp02_summary_path.exists():
        with open(exp02_summary_path, encoding="utf-8") as f:
            val_exp02 = json.load(f)

    print("\n" + "=" * 105)
    print("EXP-04 HISTGRADIENTBOOSTING VALIDATION COMPARISON TABLE")
    print("=" * 105)
    print(f"{'Config ID':<10} | {'Model Description':<32} | {'Val Acc':<8} | {'Val F1':<8} | {'Val AUC':<8} | {'Val FPR':<8} | {'Val Rec':<8} | {'Fuzzers':<8} | {'Tau*'}")
    print("-" * 105)
    for s in exp_summaries:
        print(
            f"{s['config_id']:<10} | {s['name'][:32]:<32} | {s['val_accuracy']*100:6.2f}% | "
            f"{s['val_macro_f1']:<8.4f} | {s['val_roc_auc']:<8.4f} | {s['val_fpr']*100:6.2f}% | "
            f"{s['val_recall']*100:6.2f}% | {s['val_fuzzers_recall']*100:6.2f}% | {s['calibrated_threshold']:<6.4f}"
        )
    print("=" * 105)

    e1_acc_str = f"{val_exp01a.get('val_accuracy', 0.0) * 100:.2f}%"
    e1_f1_str = f"{val_exp01a.get('val_macro_f1', 0.0):.4f}"
    e1_auc_str = f"{val_exp01a.get('val_roc_auc', 0.0):.4f}"
    e1_fpr_str = f"{val_exp01a.get('val_fpr', 0.0) * 100:.2f}%"
    e1_rec_str = f"{val_exp01a.get('val_recall', 0.0) * 100:.2f}%"

    e2_acc_str = f"{val_exp02.get('val_accuracy', 0.0) * 100:.2f}%"
    e2_f1_str = f"{val_exp02.get('val_macro_f1', 0.0):.4f}"
    e2_auc_str = f"{val_exp02.get('val_roc_auc', 0.0):.4f}"
    e2_fpr_str = f"{val_exp02.get('val_fpr', 0.0) * 100:.2f}%"
    e2_rec_str = f"{val_exp02.get('val_recall', 0.0) * 100:.2f}%"

    e4_acc_str = f"{best_config_metrics['val_accuracy'] * 100:.2f}%"
    e4_f1_str = f"{best_config_metrics['val_macro_f1']:.4f}"
    e4_auc_str = f"{best_config_metrics['val_roc_auc']:.4f}"
    e4_fpr_str = f"{best_config_metrics['val_fpr'] * 100:.2f}%"
    e4_rec_str = f"{best_config_metrics['val_recall'] * 100:.2f}%"
    e4_name = f"EXP-04 Best ({best_config_id})"

    print("\nHistorical Validation Progression (Identical 15% Validation Split, Seed 42):")
    print(f"{'Model Experiment':<28} | {'Features':<10} | {'Val Accuracy':<14} | {'Val Macro F1':<14} | {'Val ROC-AUC':<14} | {'Val FPR':<12} | {'Val Recall'}")
    print("-" * 105)
    print(f"{'EXP-01A (Unweighted DNN)':<28} | {'66':<10} | {e1_acc_str:<14} | {e1_f1_str:<14} | {e1_auc_str:<14} | {e1_fpr_str:<12} | {e1_rec_str}")
    print(f"{'EXP-02 (Domain Eng DNN)':<28} | {'72':<10} | {e2_acc_str:<14} | {e2_f1_str:<14} | {e2_auc_str:<14} | {e2_fpr_str:<12} | {e2_rec_str}")
    print(f"{e4_name:<28} | {'72':<10} | {e4_acc_str:<14} | {e4_f1_str:<14} | {e4_auc_str:<14} | {e4_fpr_str:<12} | {e4_rec_str}")
    print("=" * 105)

    # 5. Save Complete Validation Payload
    full_payload = {
        "experiment_id": "EXP-04",
        "description": "HistGradientBoostingClassifier Validation on 72 features",
        "seed": 42,
        "n_features": 72,
        "train_rows": len(X_tr),
        "val_rows": len(X_val),
        "selected_best_config": best_config_id,
        "selected_threshold": best_config_metrics["calibrated_threshold"],
        "best_config_metrics": best_config_metrics,
        "all_configurations": exp_summaries,
        "historical_validation_comparison": {
            "exp01a_val": val_exp01a,
            "exp02_val": val_exp02,
        },
    }

    out_file = exp04_artifacts_dir / "exp04_val_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2, cls=NpEncoder)

    print(f"\nAll EXP-04 validation artifacts saved to: {out_file}")


if __name__ == "__main__":
    main()

