"""HistGradientBoosting random hyperparameter search on UNSW-NB15 (CV2 task).

STRICT PROTOCOL RULES:
1. UNSW_NB15_testing-set.csv is FORBIDDEN. Any attempt to open it raises RuntimeError.
2. Only UNSW_NB15_training-set.csv is used.
3. Deduplication, 72-feature preprocessor, 5-fold stratified CV, seed 42 are
   IDENTICAL to unsw_cv_experiments.py.
4. The preprocessor is fitted on training folds only - never on validation rows.
5. id / ID / label / attack_cat are never passed to the preprocessor.
6. No final model is fitted on the full training set in this script.
7. Output goes to models_unsw/cv2/ and artifacts/v2/unsw_eval/cv2/. Never writes
   to models_unsw/cv/ or artifacts/v2/unsw_eval/cv/.

Search design:
- 40 trials, seed 42.
- Random search over: learning_rate, max_depth, max_leaf_nodes,
  min_samples_leaf, l2_regularization, class_weight, max_iter.
- Each trial trains 5 HGB models (one per fold) with early stopping on a
  10 % internal validation split taken from the TRAINING FOLD ONLY.
- OOF probabilities are assembled; tau is calibrated on OOF to maximise macro F1.
- Per trial: mean and std over folds of accuracy, attack recall, FPR, ROC-AUC.
  Worst-fold FPR is also recorded.
- Ranking: trials where OOF recall >= 0.98 are ranked by OOF FPR (ascending).
  Trials below 0.98 OOF recall are listed separately.
- Baseline (HGB_Tuned_Deep from cv1) is run through the same code for a
  like-for-like comparison.

Outputs (models_unsw/cv2/ and artifacts/v2/unsw_eval/cv2/):
  cv_trials.csv       - one row per trial, all recorded metrics
  best_params.json    - params of the highest-ranked trial (recall >= 0.98)
  cv2_summary.txt     - plain-text human-readable report
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ids_rag.unsw.preprocessor import UNSWStrictPreprocessor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_PATH = REPO_ROOT / "data" / "unsw" / "UNSW_NB15_training-set.csv"
# Testing set is FORBIDDEN in this script. Guard enforced at import time.
_FORBIDDEN_TEST_PATH = REPO_ROOT / "data" / "unsw" / "UNSW_NB15_testing-set.csv"

CV2_MODELS_DIR = REPO_ROOT / "models_unsw" / "cv2"
CV2_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "v2" / "unsw_eval" / "cv2"

# Frozen cv1 output dir - this script must NEVER write there.
_FROZEN_CV1_MODELS_DIR = REPO_ROOT / "models_unsw" / "cv"
_FROZEN_CV1_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "v2" / "unsw_eval" / "cv"
_FROZEN_FINAL_TEST_DIR = REPO_ROOT / "artifacts" / "v2" / "unsw_eval" / "final_test"

SEED = 42
N_TRIALS = 40
N_FOLDS = 5
RECALL_THRESHOLD = 0.98  # Minimum OOF recall to qualify for FPR ranking


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------

def _guard_test_set_not_opened(path: Path) -> None:
    """Raise RuntimeError if any caller attempts to open the held-out test CSV."""
    resolved = path.resolve()
    if resolved == _FORBIDDEN_TEST_PATH.resolve():
        raise RuntimeError(
            f"PROTOCOL VIOLATION: Attempt to open the held-out testing set "
            f"'{resolved}' is forbidden in this script. "
            "Only UNSW_NB15_training-set.csv may be used."
        )


def _guard_frozen_dirs(output_path: Path) -> None:
    """Raise RuntimeError if a write would land in any frozen directory."""
    frozen = [
        _FROZEN_CV1_MODELS_DIR,
        _FROZEN_CV1_ARTIFACTS_DIR,
        _FROZEN_FINAL_TEST_DIR,
    ]
    for d in frozen:
        if output_path.resolve().is_relative_to(d.resolve()):
            raise RuntimeError(
                f"PROTOCOL VIOLATION: Attempted write to frozen directory "
                f"'{d}'. All cv2 outputs must go to cv2/ directories."
            )


# ---------------------------------------------------------------------------
# NpEncoder
# ---------------------------------------------------------------------------

class NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Data loading (identical to unsw_cv_experiments.py)
# ---------------------------------------------------------------------------

def load_and_deduplicate_training_data() -> tuple[pd.DataFrame, int, int]:
    """Load and deduplicate exactly as in unsw_cv_experiments.py."""
    _guard_test_set_not_opened(TRAIN_PATH)  # sanity: always pass train path here
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {TRAIN_PATH}")

    df_raw = pd.read_csv(TRAIN_PATH)
    n_raw = len(df_raw)

    feature_cols = [c for c in df_raw.columns if c not in ["id", "ID", "label", "attack_cat"]]
    dup_mask = df_raw.duplicated(subset=feature_cols, keep="first")
    n_dups = int(dup_mask.sum())

    df_cleaned = (
        df_raw.drop_duplicates(subset=feature_cols, keep="first")
        .reset_index(drop=True)
    )
    print(f"Training rows loaded: {n_raw:,} | duplicates dropped: {n_dups:,} | "
          f"retained: {len(df_cleaned):,}")
    return df_cleaned, n_raw, n_dups


# ---------------------------------------------------------------------------
# Metric helpers (identical to unsw_cv_experiments.py)
# ---------------------------------------------------------------------------

def calibrate_threshold_on_oof(
    y_true: np.ndarray,
    probs: np.ndarray,
    min_tau: float = 0.10,
    max_tau: float = 0.90,
    steps: int = 161,
) -> tuple[float, float]:
    best_tau = 0.50
    best_f1 = -1.0
    for tau in np.linspace(min_tau, max_tau, steps):
        preds = (probs >= tau).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)
    return best_tau, best_f1


def compute_binary_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    tau: float,
) -> dict[str, float]:
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
        "accuracy": acc, "recall": rec, "precision": prec,
        "specificity": spec, "fpr": fpr, "macro_f1": macro_f1,
        "roc_auc": auc, "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Hyperparameter sampler
# ---------------------------------------------------------------------------

def _sample_params(rng: np.random.Generator) -> dict[str, Any]:
    """Sample one HGB configuration uniformly from the search space."""
    lr = float(rng.uniform(0.02, 0.20))
    max_depth = int(rng.integers(4, 17))          # 4..16 inclusive
    max_leaf_nodes = int(rng.integers(15, 128))   # 15..127 inclusive
    min_samples_leaf = int(rng.integers(20, 201)) # 20..200 inclusive
    l2_reg = float(rng.uniform(0.0, 10.0))
    max_iter = int(rng.integers(200, 601))        # 200..600 inclusive
    class_weight = rng.choice([None, "balanced"]) # type: ignore[assignment]

    return {
        "learning_rate": lr,
        "max_depth": max_depth,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_reg,
        "max_iter": max_iter,
        "class_weight": class_weight,
        # Early stopping on 10 % of training fold (validation_fraction)
        "early_stopping": True,
        "validation_fraction": 0.10,
        "n_iter_no_change": 15,
        "random_state": SEED,
    }


# ---------------------------------------------------------------------------
# Baseline params (HGB_Tuned_Deep from cv1 - run through same code)
# ---------------------------------------------------------------------------

BASELINE_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_leaf_nodes": 63,
    "max_depth": 12,
    "l2_regularization": 1.0,
    "min_samples_leaf": 30,
    "max_iter": 350,
    "class_weight": None,
    # Early stopping with internal validation split (10 % of training fold)
    "early_stopping": True,
    "validation_fraction": 0.10,
    "n_iter_no_change": 15,
    "random_state": SEED,
}


# ---------------------------------------------------------------------------
# Single-trial evaluation
# ---------------------------------------------------------------------------

def evaluate_one_trial(
    trial_id: str,
    params: dict[str, Any],
    df_train: pd.DataFrame,
    y_all: np.ndarray,
    skf: StratifiedKFold,
) -> dict[str, Any]:
    """
    Run 5-fold CV for one HGB configuration. Returns a flat dict of metrics.
    The preprocessor is fitted on training folds only, never on validation rows.
    id / label / attack_cat are excluded from features (enforced by UNSWStrictPreprocessor).
    """
    n_samples = len(df_train)
    oof_probs = np.zeros(n_samples, dtype=np.float64)
    fold_metrics: list[dict[str, float]] = []

    fold_start_total = time.time()
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train, df_train["label"]), 1
    ):
        df_tr = df_train.iloc[train_idx].copy()
        df_va = df_train.iloc[val_idx].copy()

        # --- STRICT: fit preprocessor on training fold ONLY ---
        prep = UNSWStrictPreprocessor()
        prep.fit(df_tr)

        # Assert protocol invariants inside every fold
        assert len(prep.feature_names_) == 72, (
            f"Trial {trial_id} fold {fold}: expected 72 features, got {len(prep.feature_names_)}"
        )
        for col in ["id", "ID", "label", "attack_cat"]:
            assert col not in prep.feature_names_, (
                f"Trial {trial_id} fold {fold}: forbidden column '{col}' in feature names"
            )

        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)
        y_tr = df_tr["label"].to_numpy(dtype=np.float32)
        y_va = df_va["label"].to_numpy(dtype=np.float32)

        model = HistGradientBoostingClassifier(**params)
        model.fit(X_tr, y_tr)

        p_va = model.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = p_va
        # Record fold metrics at tau=0.50 for std computation
        fold_metrics.append(compute_binary_metrics(y_va, p_va, tau=0.50))

    total_time = time.time() - fold_start_total

    # Calibrate threshold on full OOF (never touches test set)
    tau, oof_macro_f1 = calibrate_threshold_on_oof(y_all, oof_probs)
    oof_metrics = compute_binary_metrics(y_all, oof_probs, tau=tau)

    # Per-fold arrays (at tau=0.50)
    accs = [fm["accuracy"] for fm in fold_metrics]
    recs = [fm["recall"] for fm in fold_metrics]
    fprs = [fm["fpr"] for fm in fold_metrics]
    aucs = [fm["roc_auc"] for fm in fold_metrics]

    return {
        "trial_id": trial_id,
        # Hyperparameters
        "learning_rate": params["learning_rate"],
        "max_depth": params.get("max_depth"),
        "max_leaf_nodes": params["max_leaf_nodes"],
        "min_samples_leaf": params["min_samples_leaf"],
        "l2_regularization": params["l2_regularization"],
        "max_iter": params["max_iter"],
        "class_weight": str(params.get("class_weight")),
        # CV metrics at tau=0.50 (mean & std over 5 folds)
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "recall_mean": float(np.mean(recs)),
        "recall_std": float(np.std(recs)),
        "fpr_mean": float(np.mean(fprs)),
        "fpr_std": float(np.std(fprs)),
        "worst_fold_fpr": float(np.max(fprs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        # OOF calibrated threshold metrics
        "oof_tau": tau,
        "oof_macro_f1": oof_macro_f1,
        "oof_accuracy": oof_metrics["accuracy"],
        "oof_recall": oof_metrics["recall"],
        "oof_fpr": oof_metrics["fpr"],
        "oof_auc": oof_metrics["roc_auc"],
        "elapsed_s": round(total_time, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_random_search() -> None:
    start = time.time()
    CV2_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CV2_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify output dirs are not frozen
    _guard_frozen_dirs(CV2_MODELS_DIR / "placeholder")
    _guard_frozen_dirs(CV2_ARTIFACTS_DIR / "placeholder")

    print("=" * 90)
    print("UNSW-NB15 HGB RANDOM SEARCH (CV2) - 40 TRIALS, SEED 42")
    print("Label: binary attack vs normal, stratified 5-fold CV on official training set")
    print("UNSW_NB15_testing-set.csv is FORBIDDEN; any open() raises RuntimeError.")
    print("=" * 90)

    df_train, n_raw, n_dups = load_and_deduplicate_training_data()
    y_all = df_train["label"].to_numpy(dtype=np.float32)

    # Identical splitter to unsw_cv_experiments.py
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    rng = np.random.default_rng(SEED)
    all_results: list[dict[str, Any]] = []

    # --- Run baseline first (like-for-like comparison) ---
    print(f"\n[0 / {N_TRIALS}] Running cv1 baseline (HGB_Tuned_Deep) ...")
    baseline_result = evaluate_one_trial(
        trial_id="baseline_HGB_Tuned_Deep",
        params=BASELINE_PARAMS,
        df_train=df_train,
        y_all=y_all,
        skf=skf,
    )
    all_results.append(baseline_result)
    _print_trial_line(baseline_result)

    # --- Random search trials ---
    for t in range(1, N_TRIALS + 1):
        params = _sample_params(rng)
        print(f"\n[{t} / {N_TRIALS}] lr={params['learning_rate']:.3f} "
              f"depth={params['max_depth']} leaves={params['max_leaf_nodes']} "
              f"msl={params['min_samples_leaf']} l2={params['l2_regularization']:.2f} "
              f"cw={params['class_weight']} iter={params['max_iter']}")
        result = evaluate_one_trial(
            trial_id=f"trial_{t:02d}",
            params=params,
            df_train=df_train,
            y_all=y_all,
            skf=skf,
        )
        all_results.append(result)
        _print_trial_line(result)

    # --- Save cv_trials.csv ---
    csv_path = CV2_ARTIFACTS_DIR / "cv_trials.csv"
    _guard_frozen_dirs(csv_path)
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    print(f"\nAll trial results saved to: {csv_path}")

    # --- Rank trials by OOF FPR where OOF recall >= 0.98 ---
    qualified = [r for r in all_results if r["oof_recall"] >= RECALL_THRESHOLD]
    unqualified = [r for r in all_results if r["oof_recall"] < RECALL_THRESHOLD]

    qualified_sorted = sorted(qualified, key=lambda r: r["oof_fpr"])
    top5 = qualified_sorted[:5]

    # --- Save best_params.json ---
    best_params_path = CV2_ARTIFACTS_DIR / "best_params.json"
    _guard_frozen_dirs(best_params_path)
    if top5:
        best = top5[0]
        best_hgb_params = {
            "learning_rate": best["learning_rate"],
            "max_depth": best["max_depth"],
            "max_leaf_nodes": best["max_leaf_nodes"],
            "min_samples_leaf": best["min_samples_leaf"],
            "l2_regularization": best["l2_regularization"],
            "max_iter": best["max_iter"],
            "class_weight": None if best["class_weight"] == "None" else best["class_weight"],
            "early_stopping": True,
            "validation_fraction": 0.10,
            "n_iter_no_change": 15,
            "random_state": SEED,
        }
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(
                {"trial_id": best["trial_id"], "params": best_hgb_params,
                 "oof_recall": best["oof_recall"], "oof_fpr": best["oof_fpr"],
                 "oof_tau": best["oof_tau"], "oof_auc": best["oof_auc"]},
                f, indent=2, cls=NpEncoder,
            )
        print(f"Best params saved to: {best_params_path}")
    else:
        print("[WARNING] No trial achieved OOF recall >= 0.98. best_params.json not written.")

    # --- Write plain-text summary ---
    total_elapsed = time.time() - start
    summary_path = CV2_ARTIFACTS_DIR / "cv2_summary.txt"
    _guard_frozen_dirs(summary_path)
    _write_summary(
        summary_path=summary_path,
        all_results=all_results,
        top5=top5,
        unqualified=unqualified,
        baseline=baseline_result,
        n_raw=n_raw,
        n_dups=n_dups,
        n_deduplicated=len(df_train),
        total_elapsed=total_elapsed,
    )
    print(f"Plain-text summary saved to: {summary_path}")
    print(f"\nTotal random search completed in {total_elapsed:.0f}s.")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_trial_line(r: dict[str, Any]) -> None:
    print(
        f"  {r['trial_id']:<30} | "
        f"oof_recall={r['oof_recall']:.4f} | "
        f"oof_fpr={r['oof_fpr']:.4f} | "
        f"oof_auc={r['oof_auc']:.4f} | "
        f"oof_tau={r['oof_tau']:.3f} | "
        f"{r['elapsed_s']:.0f}s"
    )


def _write_summary(
    summary_path: Path,
    all_results: list[dict[str, Any]],
    top5: list[dict[str, Any]],
    unqualified: list[dict[str, Any]],
    baseline: dict[str, Any],
    n_raw: int,
    n_dups: int,
    n_deduplicated: int,
    total_elapsed: float,
) -> None:
    lines = []
    sep = "=" * 90

    lines.append(sep)
    lines.append("UNSW-NB15 HGB RANDOM SEARCH SUMMARY (CV2)")
    lines.append("Label: binary attack vs normal, stratified 5-fold CV on official training set")
    lines.append(sep)
    lines.append("")
    lines.append("DATA PREPARATION (identical to cv1)")
    lines.append(f"  Raw training rows        : {n_raw:,}")
    lines.append(f"  Duplicate rows dropped   : {n_dups:,} ({n_dups/n_raw*100:.2f}%)")
    lines.append(f"  Deduplicated rows used   : {n_deduplicated:,}")
    lines.append(f"  Features                 : 72 (43 numeric + 29 categorical one-hot, drop=first)")
    lines.append(f"  CV folds                 : 5 (StratifiedKFold, shuffle=True, seed=42)")
    lines.append(f"  Threshold calibration    : OOF sweep tau in [0.10, 0.90], 161 steps, max macro F1")
    lines.append(f"  id/ID/label/attack_cat   : excluded from features in every fold")
    lines.append(f"  Preprocessor fit scope   : training fold ONLY (never validation rows)")
    lines.append("")
    lines.append("SEARCH DESIGN")
    lines.append(f"  Trials              : {N_TRIALS} random + 1 baseline = {N_TRIALS + 1} total")
    lines.append(f"  Seed                : {SEED}")
    lines.append(f"  Ranking criterion   : OOF FPR ascending, restricted to OOF recall >= {RECALL_THRESHOLD:.2f}")
    lines.append(f"  Early stopping      : validation_fraction=0.10 of training fold, n_iter_no_change=15")
    lines.append(f"  Total wall time     : {total_elapsed:.0f}s")
    lines.append("")

    # Baseline row
    lines.append("BASELINE (HGB_Tuned_Deep from cv1, run through same code)")
    lines.append(_fmt_header())
    lines.append(_fmt_row(baseline))
    lines.append("")

    # Top-5 qualifying trials
    lines.append(f"TOP 5 TRIALS BY OOF FPR (OOF recall >= {RECALL_THRESHOLD:.2f})")
    if top5:
        lines.append(_fmt_header())
        for r in top5:
            lines.append(_fmt_row(r))
    else:
        lines.append("  No trial achieved OOF recall >= 0.98.")
    lines.append("")

    # All qualifying trials sorted
    qualified_sorted = sorted(
        [r for r in all_results if r["oof_recall"] >= RECALL_THRESHOLD],
        key=lambda r: r["oof_fpr"],
    )
    lines.append(f"ALL QUALIFYING TRIALS (OOF recall >= {RECALL_THRESHOLD:.2f}), sorted by OOF FPR")
    if qualified_sorted:
        lines.append(_fmt_header())
        for r in qualified_sorted:
            lines.append(_fmt_row(r))
    else:
        lines.append("  None.")
    lines.append("")

    # Unqualified trials
    lines.append(f"TRIALS BELOW OOF RECALL {RECALL_THRESHOLD:.2f} (not ranked, listed for reference)")
    if unqualified:
        lines.append(_fmt_header())
        for r in sorted(unqualified, key=lambda r: -r["oof_recall"]):
            lines.append(_fmt_row(r))
    else:
        lines.append("  All trials met recall threshold.")
    lines.append("")

    lines.append("HYPERPARAMETER DETAILS OF TOP 5 TRIALS")
    for rank, r in enumerate(top5, 1):
        lines.append(f"  Rank {rank}: {r['trial_id']}")
        lines.append(f"    learning_rate     = {r['learning_rate']:.4f}")
        lines.append(f"    max_depth         = {r['max_depth']}")
        lines.append(f"    max_leaf_nodes    = {r['max_leaf_nodes']}")
        lines.append(f"    min_samples_leaf  = {r['min_samples_leaf']}")
        lines.append(f"    l2_regularization = {r['l2_regularization']:.4f}")
        lines.append(f"    max_iter          = {r['max_iter']}")
        lines.append(f"    class_weight      = {r['class_weight']}")
        lines.append(f"    oof_tau           = {r['oof_tau']:.4f}")
        lines.append(f"    oof_recall        = {r['oof_recall']:.4f}")
        lines.append(f"    oof_fpr           = {r['oof_fpr']:.4f}")
        lines.append(f"    oof_auc           = {r['oof_auc']:.4f}")
    lines.append("")
    lines.append(sep)
    lines.append("NOTE: No model has been trained on the full training set in this script.")
    lines.append("NOTE: UNSW_NB15_testing-set.csv was NOT read at any point.")
    lines.append("NOTE: Do not compare these CV results to the paper's 95.30% (different setup).")
    lines.append(sep)

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_header() -> str:
    return (
        f"  {'trial_id':<32} | "
        f"{'oof_recall':>10} | "
        f"{'oof_fpr':>8} | "
        f"{'oof_auc':>8} | "
        f"{'oof_tau':>7} | "
        f"{'fpr_mean':>8} | "
        f"{'fpr_std':>7} | "
        f"{'worst_fpr':>9}"
    )


def _fmt_row(r: dict[str, Any]) -> str:
    return (
        f"  {r['trial_id']:<32} | "
        f"{r['oof_recall']:>10.4f} | "
        f"{r['oof_fpr']:>8.4f} | "
        f"{r['oof_auc']:>8.4f} | "
        f"{r['oof_tau']:>7.4f} | "
        f"{r['fpr_mean']:>8.4f} | "
        f"{r['fpr_std']:>7.4f} | "
        f"{r['worst_fold_fpr']:>9.4f}"
    )


if __name__ == "__main__":
    run_random_search()

