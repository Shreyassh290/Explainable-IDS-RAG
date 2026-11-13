"""Stratified 5-Fold Cross-Validation & Scientific Model Improvement on UNSW-NB15.

EVALUATION PROTOCOL:
1. Uses UNSW_NB15_training-set.csv ONLY.
   The official testing set is NEVER read or referenced here.
2. Checks for duplicate rows within the training set on feature columns and drops them.
3. Preprocessor (UNSWStrictPreprocessor) is fitted STRICTLY on training folds only.
4. Target and ID columns ('id', 'ID', 'label', 'attack_cat') are strictly excluded from features.
5. Evaluates:
   - HistGradientBoosting tuning (learning rate, max_leaf_nodes, l2, min_samples_leaf, class_weight).
   - Probability calibration (Sigmoid / Platt scaling and Isotonic regression) fitted inside CV.
   - Soft-voting ensemble combining best HGB with the existing EXP-02 DNN.
   - Out-of-fold threshold calibration.
6. Reports CV mean and std for: Accuracy, Attack Recall, Specificity, FPR, and ROC-AUC.
7. Reporting fixes:
   - Shows ONLY per-family recall (detection rate) for attack families.
   - Removes per-family precision and F1 for binary model.
   - Reports Normal as specificity and false positive rate.
   - Labeled as: "binary attack vs normal, stratified 5-fold cross-validation on official training set".
   - Contextual note: Do not compare to the paper's 95.30% (different data split and experimental setup).
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
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ids_rag.unsw.preprocessor import UNSWStrictPreprocessor

DATA_PATH = REPO_ROOT / "data" / "unsw" / "UNSW_NB15_training-set.csv"
CV_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "v2" / "unsw_eval" / "cv"
CV_MODELS_DIR = REPO_ROOT / "models_unsw" / "cv"
EXP02_DNN_PATH = REPO_ROOT / "models_unsw" / "experiments" / "exp02" / "model_EXP-02.pt"

EXCLUDED_COLS = ["id", "ID", "label", "attack_cat", "stcpb", "dtcpb"]


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


class BoostedDNNV2(nn.Module):
    """PyTorch architecture matching EXP-02 checkpoint for 72 domain features."""

    def __init__(self, in_features: int = 72, hidden: list[int] | None = None, dropout: float = 0.25):
        super().__init__()
        if hidden is None:
            hidden = [256, 128, 64, 32]
        layers: list[nn.Module] = []
        prev_dim = in_features
        for h_dim in hidden:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_and_deduplicate_training_data() -> tuple[pd.DataFrame, int, int]:
    """Load official training set and strictly drop internal duplicate feature rows."""
    print(f"Loading official training set strictly from: {DATA_PATH}...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    df_raw = pd.read_csv(DATA_PATH)
    n_raw = len(df_raw)

    feature_cols = [c for c in df_raw.columns if c not in ["id", "ID", "label", "attack_cat"]]
    dup_mask = df_raw.duplicated(subset=feature_cols, keep="first")
    n_dups = int(dup_mask.sum())

    df_cleaned = df_raw.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
    n_cleaned = len(df_cleaned)

    print(f"Total training flows loaded: {n_raw:,}")
    print(f"Exact duplicate feature rows identified in training: {n_dups:,} ({n_dups / n_raw * 100:.2f}%)")
    print(f"Deduplicated training flows retained for 5-fold CV:   {n_cleaned:,}")
    return df_cleaned, n_raw, n_dups


def calibrate_threshold_on_oof(
    y_true: np.ndarray,
    probs: np.ndarray,
    min_tau: float = 0.10,
    max_tau: float = 0.90,
    steps: int = 161,
) -> tuple[float, float]:
    """Find the threshold tau that maximizes Macro F1 on out-of-fold predictions."""
    best_tau = 0.50
    best_f1 = -1.0
    for tau in np.linspace(min_tau, max_tau, steps):
        preds = (probs >= tau).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)
    return best_tau, best_f1


def compute_binary_metrics(y_true: np.ndarray, probs: np.ndarray, tau: float) -> dict[str, float]:
    """Compute binary classification metrics with zero division safety."""
    preds = (probs >= tau).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, preds))
    rec = float(recall_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5

    return {
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "specificity": spec,
        "fpr": fpr,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_per_family_table(
    y_true: np.ndarray,
    preds: np.ndarray,
    attack_cats: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Compute reporting-compliant per-family table.
    
    Rules:
    - Attack families show ONLY per-family recall (detection rate).
    - No precision or F1 for binary model on attack families.
    - Normal class shows specificity and FPR.
    """
    results: dict[str, dict[str, Any]] = {}
    cats, counts = np.unique(attack_cats, return_counts=True)

    for cat, total in zip(cats, counts):
        mask = (attack_cats == cat)
        if cat == "Normal":
            # For Normal, positive prediction = false alarm (FP), negative prediction = true normal (TN)
            tn = int(np.sum((preds[mask] == 0)))
            fp = int(np.sum((preds[mask] == 1)))
            spec = float(tn / total) if total > 0 else 0.0
            fpr = float(fp / total) if total > 0 else 0.0
            results[cat] = {
                "total_flows": int(total),
                "true_negative": tn,
                "false_positive": fp,
                "specificity": spec,
                "false_positive_rate": fpr,
            }
        else:
            # For Attack families, positive prediction = detected attack (TP), negative = missed (FN)
            tp = int(np.sum((preds[mask] == 1)))
            fn = int(np.sum((preds[mask] == 0)))
            det_rate = float(tp / total) if total > 0 else 0.0
            results[cat] = {
                "total_flows": int(total),
                "detected": tp,
                "missed": fn,
                "recall_detection_rate": det_rate,
            }

    return results


def run_stratified_cv_experiments() -> dict[str, Any]:
    start_total_time = time.time()
    CV_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CV_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("UNSW-NB15 STRICT 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("Label: binary attack vs normal, stratified 5-fold cross-validation on official training set")
    print("=" * 90)

    # 1. Load and deduplicate training data
    df_train, n_raw_train, n_dups_train = load_and_deduplicate_training_data()
    n_samples = len(df_train)
    y_all = df_train["label"].to_numpy(dtype=np.float32)
    attack_cats_all = df_train["attack_cat"].fillna("Normal").astype(str).to_numpy()

    # 2. Define Candidate Models
    models_config = {
        "HGB_Default": {
            "class": HistGradientBoostingClassifier,
            "params": {
                "learning_rate": 0.10,
                "max_leaf_nodes": 31,
                "l2_regularization": 0.0,
                "min_samples_leaf": 20,
                "max_iter": 300,
                "early_stopping": True,
                "n_iter_no_change": 15,
                "random_state": 42,
            },
        },
        "HGB_Tuned_Deep": {
            "class": HistGradientBoostingClassifier,
            "params": {
                "learning_rate": 0.05,
                "max_leaf_nodes": 63,
                "max_depth": 12,
                "l2_regularization": 1.0,
                "min_samples_leaf": 30,
                "max_iter": 350,
                "early_stopping": True,
                "n_iter_no_change": 15,
                "random_state": 42,
            },
        },
        "HGB_Tuned_Regularized": {
            "class": HistGradientBoostingClassifier,
            "params": {
                "learning_rate": 0.05,
                "max_leaf_nodes": 31,
                "max_depth": 10,
                "l2_regularization": 5.0,
                "min_samples_leaf": 40,
                "max_iter": 300,
                "early_stopping": True,
                "n_iter_no_change": 15,
                "random_state": 42,
            },
        },
        "HGB_Tuned_Balanced": {
            "class": HistGradientBoostingClassifier,
            "params": {
                "learning_rate": 0.05,
                "max_leaf_nodes": 45,
                "max_depth": 10,
                "l2_regularization": 2.0,
                "min_samples_leaf": 25,
                "class_weight": "balanced",
                "max_iter": 300,
                "early_stopping": True,
                "n_iter_no_change": 15,
                "random_state": 42,
            },
        },
    }

    # Prepare out-of-fold probability containers
    all_model_names = list(models_config.keys())
    oof_probs: dict[str, np.ndarray] = {m: np.zeros(n_samples, dtype=np.float32) for m in all_model_names}
    oof_probs["HGB_Deep_Calibrated_Sigmoid"] = np.zeros(n_samples, dtype=np.float32)
    oof_probs["HGB_Deep_Calibrated_Isotonic"] = np.zeros(n_samples, dtype=np.float32)

    has_dnn = EXP02_DNN_PATH.exists()
    if has_dnn:
        oof_probs["DNN_EXP02"] = np.zeros(n_samples, dtype=np.float32)
        print(f"\n[INFO] Found pre-existing EXP-02 DNN checkpoint at {EXP02_DNN_PATH}. Will evaluate soft-voting ensemble.")
    else:
        print(f"\n[WARNING] EXP-02 DNN checkpoint not found at {EXP02_DNN_PATH}. Skipping DNN ensemble.")

    # Per-fold metrics containers
    fold_metrics: dict[str, list[dict[str, float]]] = {
        m: [] for m in list(oof_probs.keys())
    }

    # 3. Stratified 5-Fold Loop
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dnn_model: BoostedDNNV2 | None = None
    if has_dnn:
        dnn_model = BoostedDNNV2(in_features=72, hidden=[256, 128, 64, 32], dropout=0.25).to(device)
        dnn_model.load_state_dict(torch.load(EXP02_DNN_PATH, map_location=device))
        dnn_model.eval()

    print("\nStarting 5-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train["label"]), 1):
        fold_start = time.time()
        print(f"\n>>> Fold {fold} / 5 | Train size: {len(train_idx):,} | Val size: {len(val_idx):,}")

        df_tr = df_train.iloc[train_idx].copy()
        df_va = df_train.iloc[val_idx].copy()

        # STRICT PROTOCOL: Fit preprocessor strictly on training fold only
        preprocessor = UNSWStrictPreprocessor()
        preprocessor.fit(df_tr)

        # Assert no target/ID leakage
        assert len(preprocessor.feature_names_) == 72, f"Feature count != 72 in fold {fold}"
        for col in ["id", "ID", "label", "attack_cat"]:
            assert col not in preprocessor.feature_names_, f"Leakage: {col} in feature names"

        X_tr = preprocessor.transform(df_tr)
        X_va = preprocessor.transform(df_va)
        y_tr = df_tr["label"].to_numpy(dtype=np.float32)
        y_va = df_va["label"].to_numpy(dtype=np.float32)

        # 3A. Train and evaluate HGB candidates
        fitted_models: dict[str, Any] = {}
        for m_name, cfg in models_config.items():
            model = cfg["class"](**cfg["params"])
            model.fit(X_tr, y_tr)
            fitted_models[m_name] = model

            p_va = model.predict_proba(X_va)[:, 1]
            oof_probs[m_name][val_idx] = p_va

            # Default threshold metrics for this fold
            m_metrics = compute_binary_metrics(y_va, p_va, tau=0.50)
            fold_metrics[m_name].append(m_metrics)

        # 3B. Calibration on HGB_Tuned_Deep (fitted inside fold)
        base_deep = fitted_models["HGB_Tuned_Deep"]
        p_tr_deep = base_deep.predict_proba(X_tr)[:, 1]
        p_va_deep = oof_probs["HGB_Tuned_Deep"][val_idx]

        # Sigmoid calibration (Platt scaling) fitted via LogisticRegression on training fold probabilities
        cal_sig = LogisticRegression(solver="lbfgs")
        cal_sig.fit(p_tr_deep.reshape(-1, 1), y_tr)
        p_va_sig = cal_sig.predict_proba(p_va_deep.reshape(-1, 1))[:, 1]
        oof_probs["HGB_Deep_Calibrated_Sigmoid"][val_idx] = p_va_sig
        fold_metrics["HGB_Deep_Calibrated_Sigmoid"].append(compute_binary_metrics(y_va, p_va_sig, tau=0.50))

        # Isotonic regression calibrated inside fold
        iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_reg.fit(p_tr_deep, y_tr)
        p_va_iso = iso_reg.predict(p_va_deep)
        oof_probs["HGB_Deep_Calibrated_Isotonic"][val_idx] = p_va_iso
        fold_metrics["HGB_Deep_Calibrated_Isotonic"].append(compute_binary_metrics(y_va, p_va_iso, tau=0.50))

        # 3C. DNN Inference on fold validation split
        if has_dnn and dnn_model is not None:
            with torch.no_grad():
                va_tensor = torch.from_numpy(X_va).to(device)
                dnn_logits = dnn_model(va_tensor).squeeze(-1).cpu().numpy()
                dnn_p_va = expit(dnn_logits)
            oof_probs["DNN_EXP02"][val_idx] = dnn_p_va
            fold_metrics["DNN_EXP02"].append(compute_binary_metrics(y_va, dnn_p_va, tau=0.50))

        fold_elapsed = time.time() - fold_start
        print(f"    Fold {fold} finished in {fold_elapsed:.1f}s | HGB Deep AUC: {fold_metrics['HGB_Tuned_Deep'][-1]['roc_auc']:.4f}")

    # 4. Evaluate Soft-Voting Ensemble on Out-of-Fold Predictions
    ensemble_results: dict[str, Any] = {}
    if has_dnn:
        print("\nEvaluating Soft-Voting Grid between HGB_Tuned_Deep and DNN_EXP02 on OOF predictions...")
        best_w = 0.5
        best_ens_f1 = -1.0
        best_ens_tau = 0.5

        for w_hgb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            w_dnn = 1.0 - w_hgb
            p_ens = w_hgb * oof_probs["HGB_Tuned_Deep"] + w_dnn * oof_probs["DNN_EXP02"]
            tau, f1_macro = calibrate_threshold_on_oof(y_all, p_ens)
            if f1_macro > best_ens_f1:
                best_ens_f1 = f1_macro
                best_w = w_hgb
                best_ens_tau = tau

        ens_name = f"Ensemble_HGB_{int(best_w*100)}_DNN_{int((1-best_w)*100)}"
        p_best_ens = best_w * oof_probs["HGB_Tuned_Deep"] + (1.0 - best_w) * oof_probs["DNN_EXP02"]
        oof_probs[ens_name] = p_best_ens

        # Compute per-fold metrics for the chosen ensemble
        fold_metrics[ens_name] = []
        for fold, (_, val_idx) in enumerate(skf.split(df_train, df_train["label"]), 1):
            y_va = y_all[val_idx]
            p_va_ens = p_best_ens[val_idx]
            fold_metrics[ens_name].append(compute_binary_metrics(y_va, p_va_ens, tau=best_ens_tau))

        ensemble_results = {
            "ensemble_name": ens_name,
            "optimal_weight_hgb": best_w,
            "optimal_weight_dnn": 1.0 - best_w,
            "calibrated_threshold": best_ens_tau,
            "oof_macro_f1": best_ens_f1,
        }

    # 5. Compile and Summarize 5-Fold CV Table
    print("\n" + "=" * 90)
    print("5-FOLD CROSS-VALIDATION SUMMARY TABLE (MEAN +/- STD)")
    print("Label: binary attack vs normal, stratified 5-fold cross-validation on official training set")
    print("=" * 90)
    print(f"{'Model':<30} | {'Accuracy (%)':<15} | {'Recall (%)':<15} | {'FPR (%)':<15} | {'ROC-AUC':<15} | {'Macro F1':<10}")
    print("-" * 115)

    cv_summary_table: dict[str, dict[str, Any]] = {}
    for m_name in oof_probs.keys():
        if m_name not in fold_metrics or len(fold_metrics[m_name]) == 0:
            continue

        accs = [fm["accuracy"] * 100 for fm in fold_metrics[m_name]]
        recs = [fm["recall"] * 100 for fm in fold_metrics[m_name]]
        fprs = [fm["fpr"] * 100 for fm in fold_metrics[m_name]]
        specs = [fm["specificity"] * 100 for fm in fold_metrics[m_name]]
        aucs = [fm["roc_auc"] for fm in fold_metrics[m_name]]
        f1s = [fm["macro_f1"] for fm in fold_metrics[m_name]]

        # Calibrate optimal threshold on OOF predictions
        cal_tau, cal_macro_f1 = calibrate_threshold_on_oof(y_all, oof_probs[m_name])
        oof_cal_metrics = compute_binary_metrics(y_all, oof_probs[m_name], tau=cal_tau)

        cv_summary_table[m_name] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "recall_mean": float(np.mean(recs)),
            "recall_std": float(np.std(recs)),
            "fpr_mean": float(np.mean(fprs)),
            "fpr_std": float(np.std(fprs)),
            "specificity_mean": float(np.mean(specs)),
            "specificity_std": float(np.std(specs)),
            "roc_auc_mean": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "calibrated_threshold": cal_tau,
            "calibrated_oof_metrics": oof_cal_metrics,
        }

        print(
            f"{m_name:<30} | "
            f"{np.mean(accs):6.2f} +/- {np.std(accs):4.2f} | "
            f"{np.mean(recs):6.2f} +/- {np.std(recs):4.2f} | "
            f"{np.mean(fprs):6.2f} +/- {np.std(fprs):4.2f} | "
            f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f} | "
            f"{np.mean(f1s):.4f}"
        )

    # 6. Per-Family Recall Table for the Best Model
    # Determine best model by calibrated OOF Macro F1
    best_candidate_name = max(cv_summary_table.keys(), key=lambda k: cv_summary_table[k]["calibrated_oof_metrics"]["macro_f1"])
    best_tau = cv_summary_table[best_candidate_name]["calibrated_threshold"]
    best_oof_preds = (oof_probs[best_candidate_name] >= best_tau).astype(int)

    per_family_table = compute_per_family_table(y_all, best_oof_preds, attack_cats_all)

    print("\n" + "=" * 90)
    print(f"OUT-OF-FOLD PER-FAMILY DETECTION SUMMARY: {best_candidate_name} (tau = {best_tau:.4f})")
    print("Reporting Fix: Attack families show detection rate (recall) ONLY. Normal shows Specificity & FPR.")
    print("Label: binary attack vs normal, stratified 5-fold cross-validation on official training set")
    print("=" * 90)
    print(f"{'Category':<20} | {'Total Flows':<12} | {'Detected (TP/TN)':<18} | {'Detection Rate / Specificity':<30}")
    print("-" * 90)

    for cat, stats in sorted(per_family_table.items(), key=lambda x: -x[1]["total_flows"]):
        if cat == "Normal":
            tn = stats["true_negative"]
            fp = stats["false_positive"]
            spec_pct = stats["specificity"] * 100
            fpr_pct = stats["false_positive_rate"] * 100
            print(f"{cat:<20} | {stats['total_flows']:<12,d} | TN={tn:,d}, FP={fp:,d}      | Specificity: {spec_pct:5.2f}% (FPR: {fpr_pct:5.2f}%)")
        else:
            tp = stats["detected"]
            fn = stats["missed"]
            det_pct = stats["recall_detection_rate"] * 100
            print(f"{cat:<20} | {stats['total_flows']:<12,d} | TP={tp:,d}, FN={fn:,d}      | Recall: {det_pct:5.2f}%")

    print("\n" + "=" * 90)
    print("SCIENTIFIC BENCHMARK CONTEXT:")
    print("Note: Do NOT compare these results to the paper's 95.30% benchmark.")
    print("Reason: The original Moustafa & Slay (2015) paper evaluated a different split")
    print("(often 175,341 training / 82,332 testing, or alternative subsets with different benign/attack ratios).")
    print("All results here are strictly evaluated under Stratified 5-Fold CV on the training partition.")
    print("=" * 90)

    # 7. Save Final Model Artifacts and Summary
    print("\nRefitting top models on full deduplicated training set for candidate persistence...")
    best_hgb_model = models_config["HGB_Tuned_Deep"]["class"](**models_config["HGB_Tuned_Deep"]["params"])
    full_preprocessor = UNSWStrictPreprocessor()
    full_preprocessor.fit(df_train)
    X_full_train = full_preprocessor.transform(df_train)
    best_hgb_model.fit(X_full_train, y_all)

    # Save models
    joblib.dump(best_hgb_model, CV_MODELS_DIR / "hgb_tuned_deep_full_train.joblib")
    joblib.dump(full_preprocessor, CV_MODELS_DIR / "unsw_strict_preprocessor_full_train.joblib")

    # Serialize complete CV summary
    total_time = time.time() - start_total_time
    summary_data = {
        "metadata": {
            "protocol_label": "binary attack vs normal, stratified 5-fold cross-validation on official training set",
            "n_raw_training_flows": n_raw_train,
            "n_training_duplicates_dropped": n_dups_train,
            "n_deduplicated_training_flows": n_samples,
            "feature_count": 72,
            "cv_folds": 5,
            "seed": 42,
            "total_execution_seconds": total_time,
        },
        "cv_summary_table": cv_summary_table,
        "best_candidate": {
            "name": best_candidate_name,
            "calibrated_threshold": best_tau,
            "per_family_detection": per_family_table,
        },
        "ensemble_results": ensemble_results,
    }

    out_summary_file = CV_ARTIFACTS_DIR / "unsw_cv_summary.json"
    with open(out_summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, cls=NpEncoder)

    print(f"\nCV summary JSON saved to: {out_summary_file}")
    print(f"Fitted full-training model saved to: {CV_MODELS_DIR / 'hgb_tuned_deep_full_train.joblib'}")
    print(f"Total CV execution completed in {total_time:.1f}s.")
    print("=" * 90)

    return summary_data


if __name__ == "__main__":
    run_stratified_cv_experiments()

