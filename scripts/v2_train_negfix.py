"""Train negfix v2 binary DNNs with baseline classes as negatives.

Root cause fix: the original v2 models (models_v2/) were trained with only the
three extra classes as the full dataset. Negatives for each one-vs-rest model
were only the other two extra classes, never Benign/DoS/DDoS. In the mixed
six-class argmax this causes baseline flows to leak into extra-class predictions.

Fix: for each extra class (Bot=3, SSH-Bruteforce=5, FTP-BruteForce=4) the
training negatives = (other extra classes) + (stratified sample from baseline
train). Validation negatives = (other extra val rows) + (stratified sample from
baseline val). The frozen baseline scaler is used via transform-only (never fit).

Output locations (new, never conflicting with frozen dirs):
  models_v2b/dnn_v2b_<ClassName>.pt
  models_v2b/label_map_v2b.json
  models_v2b/feature_names_v2b.json
  artifacts/v2b/training/training_history_v2b_<ClassName>.json
  artifacts/v2b/training/training_curves_v2b_<ClassName>.png
  artifacts/v2b/training/train_summary_v2b.json

Frozen dirs that must never be written to:
  models/, data/processed/, artifacts/eval/, artifacts/reports/,
  artifacts/fresh_eval/, artifacts/tuning_experiments/

Usage:
    uv run python scripts/v2_train_negfix.py --confirm-train

Add --baseline-ratio N to change how many baseline negative rows are drawn per
positive (default 3). The script stops and reports if any safety assertion fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ids_rag.config import get_project_root
from ids_rag.models.dnn import (
    BinaryDNN,
    check_monitor_improvement,
    compute_binary_class_weights,
    compute_binary_macro_f1,
)
from ids_rag.v2.prepare_extra import load_v2_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FROZEN_DIRS = [
    "models",
    "data/processed",
    "artifacts/eval",
    "artifacts/reports",
    "artifacts/fresh_eval",
    "artifacts/tuning_experiments",
]

INDICATIVE_ONLY_CLASSES = {"FTP-BruteForce"}
INDICATIVE_THRESHOLD = 500

# New output paths (never overlap with frozen dirs or models_v2/)
MODELS_DIR_REL = "models_v2b"
ARTIFACTS_DIR_REL = "artifacts/v2b/training"


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def check_negfix_output_safety(models_dir: Path, artifacts_dir: Path, root: Path) -> None:
    """Refuse to run if any output directory overlaps with frozen or original v2 directories."""
    forbidden_abs = [(root / fd).resolve() for fd in FROZEN_DIRS]
    # Also guard original v2 dirs from accidental overwrite
    extra_forbidden = [
        (root / "models_v2").resolve(),
        (root / "data" / "v2").resolve(),
    ]

    for out_dir in [models_dir, artifacts_dir]:
        out_abs = out_dir.resolve()
        for forbidden_abs_path in forbidden_abs + extra_forbidden:
            if (
                out_abs == forbidden_abs_path
                or forbidden_abs_path in out_abs.parents
                or out_abs in forbidden_abs_path.parents
            ):
                raise ValueError(
                    f"SAFETY REFUSAL: Output directory '{out_abs}' conflicts with "
                    f"forbidden directory '{forbidden_abs_path}'. "
                    f"negfix must write only to models_v2b/ and artifacts/v2b/."
                )

    # Confirm expected suffix
    if "v2b" not in str(models_dir) and "negfix" not in str(models_dir):
        raise ValueError(
            f"SAFETY REFUSAL: models_dir '{models_dir}' does not contain 'v2b' or 'negfix'. "
            f"This looks wrong. Aborting."
        )


def assert_scaler_fitted(scaler: Any) -> None:
    """Assert that the scaler has already been fit (mean_ attribute exists)."""
    if not hasattr(scaler, "mean_"):
        raise ValueError(
            "SAFETY REFUSAL: Scaler does not have 'mean_' attribute. "
            "Only a pre-fitted scaler may be used (transform-only, never fit)."
        )


def assert_no_test_data_loaded(path: Path) -> None:
    """Assert the given path is not a test parquet. Guards against accidental test leakage."""
    name = path.name.lower()
    if "test" in name:
        raise ValueError(
            f"SAFETY REFUSAL: Attempted to load test data '{path}' during training. "
            f"Only train and val parquets are allowed for model fitting."
        )


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> torch.Generator:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# ---------------------------------------------------------------------------
# Data construction: augment extra-class split with baseline negatives
# ---------------------------------------------------------------------------

def build_augmented_split(
    df_extra: pd.DataFrame,
    df_baseline: pd.DataFrame,
    class_id: int,
    feature_names: list[str],
    scaler: Any,
    baseline_ratio: int,
    seed: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build augmented feature/label arrays for one class in one split.

    Positives: rows from df_extra where label == class_id.
    Extra-class negatives: rows from df_extra where label != class_id (all of them).
    Baseline negatives: stratified sample from df_baseline (Benign=0, DoS=1, DDoS=2).

    Total negatives = max(extra_neg_count, baseline_ratio * n_pos).
    Baseline share: at least 50% of total negatives; drawn proportionally across
    baseline classes with seed for reproducibility.

    NOTE: All four parquet sources (data/processed/train.parquet,
    data/processed/val.parquet, data/v2/processed/train_extra.parquet,
    data/v2/processed/val_extra.parquet) are ALREADY scaled with the frozen
    baseline StandardScaler by their respective prepare scripts. The scaler
    argument is kept for documentation/safety checks but must NOT be applied
    here; applying it a second time would corrupt the feature values.

    Labels returned use class_id as the positive label (unchanged from extra data),
    so binary conversion later is: y_binary = (y_raw == class_id).

    Returns:
        x_out: float32 array of shape (N, 41), features as stored in parquet
        y_raw: int64 array of shape (N,) with original label IDs
    """
    rng = np.random.default_rng(seed)

    # Positives
    pos_mask = df_extra["label"] == class_id
    df_pos = df_extra[pos_mask]
    n_pos = len(df_pos)
    if n_pos == 0:
        raise ValueError(
            f"NO_POSITIVES: class_id={class_id} has 0 positive rows in {split_name} extra split."
        )

    # Extra-class negatives (all other extra rows)
    df_extra_neg = df_extra[~pos_mask]
    n_extra_neg = len(df_extra_neg)

    # Baseline negatives: at least 50% of total target negatives
    target_total_neg = max(n_extra_neg, baseline_ratio * n_pos)
    target_baseline_neg = max(target_total_neg - n_extra_neg, target_total_neg // 2)

    # Sample baseline negatives stratified by baseline label
    baseline_classes = sorted(df_baseline["label"].unique().tolist())
    per_class_baseline = target_baseline_neg // max(len(baseline_classes), 1)

    baseline_parts: list[pd.DataFrame] = []
    for bc in baseline_classes:
        bc_rows = df_baseline[df_baseline["label"] == bc]
        n_take = min(per_class_baseline, len(bc_rows))
        if n_take > 0:
            idx = rng.choice(len(bc_rows), size=n_take, replace=False)
            baseline_parts.append(bc_rows.iloc[idx])

    df_baseline_neg = pd.concat(baseline_parts, ignore_index=True) if baseline_parts else pd.DataFrame()

    # Combine
    parts = [df_pos, df_extra_neg]
    if len(df_baseline_neg) > 0:
        parts.append(df_baseline_neg)
    df_combined = pd.concat(parts, ignore_index=True)

    n_baseline_neg = len(df_baseline_neg)
    n_neg_total = n_extra_neg + n_baseline_neg
    print(
        f"  [{split_name}] class_id={class_id}: "
        f"pos={n_pos}, extra_neg={n_extra_neg}, baseline_neg={n_baseline_neg}, "
        f"total_neg={n_neg_total}, total_rows={len(df_combined)}",
        flush=True,
    )

    # Sanity assertion on first six feature columns:
    # All four parquet sources are already scaled with the frozen baseline scaler.
    for col in feature_names[:6]:
        # Baseline sources: abs(mean) < 0.5 and 0.5 < std < 2.0
        b_mean = float(df_baseline[col].mean())
        b_std = float(df_baseline[col].std())
        if not (abs(b_mean) < 0.5 and 0.5 < b_std < 2.0):
            raise AssertionError(
                f"Baseline source sanity check failed for column '{col}' in split '{split_name}': "
                f"abs(mean)={abs(b_mean):.4f} (expected < 0.5), std={b_std:.4f} (expected 0.5 < std < 2.0)"
            )

        # Extra sources: abs(mean) < 10 and std < 10
        # (These are scaled with the baseline scaler but hold only the extra classes, so std is not ~1)
        e_mean = float(df_extra[col].mean())
        e_std = float(df_extra[col].std())
        if not (abs(e_mean) < 10.0 and e_std < 10.0):
            raise AssertionError(
                f"Extra source sanity check failed for column '{col}' in split '{split_name}': "
                f"abs(mean)={abs(e_mean):.4f} (expected < 10), std={e_std:.4f} (expected < 10)"
            )

    x_raw = df_combined[feature_names].to_numpy(dtype=np.float32, copy=True)
    y_raw = df_combined["label"].to_numpy(dtype=np.int64, copy=True)

    # All four parquet sources are ALREADY scaled with the frozen baseline scaler,
    # and the eval script feeds them to the models unscaled.
    # The scaler argument is retained for signature compatibility and safety checks,
    # but not applied here (applying it again would corrupt already-scaled feature values).
    # Use the feature values as they are (float32).
    x_scaled = x_raw

    return x_scaled, y_raw


# ---------------------------------------------------------------------------
# Training loop (identical algorithm to train_extra.py::train_binary_model_v2)
# ---------------------------------------------------------------------------

def train_binary_model_negfix(
    class_id: int,
    class_name: str,
    x_train: torch.Tensor,
    y_train_raw: np.ndarray,
    x_val: torch.Tensor,
    y_val_raw: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    models_dir: Path,
    artifacts_dir: Path,
    data_generator: torch.Generator,
) -> dict[str, Any]:
    """Train one binary negfix classifier (class_id vs rest) with class-weighted BCE loss.

    Identical training algorithm to ids_rag.v2.train_extra.train_binary_model_v2.
    Saves weights to models_v2b/dnn_v2b_<class_name>.pt.
    """
    model_cfg = config["model"]
    batch_size: int = model_cfg.get("batch_size", 512)
    lr: float = model_cfg.get("lr", 0.001)
    max_epochs: int = model_cfg.get("max_epochs", 100)
    patience: int = model_cfg.get("early_stopping_patience", 5)
    monitor: str = model_cfg.get("monitor", "f1").lower()
    num_workers: int = model_cfg.get("num_workers", 0)
    hidden: list[int] = model_cfg.get("hidden", [128, 64, 32])
    dropout: float = model_cfg.get("dropout", 0.3)
    use_batchnorm: bool = model_cfg.get("use_batchnorm", False)

    if monitor not in ("f1", "loss"):
        raise ValueError(f"Invalid model.monitor '{monitor}'. Allowed values are 'f1' or 'loss'.")

    # Binary labels: 1 for this class, 0 for rest
    y_train_binary = (y_train_raw == class_id).astype(np.float32)
    y_val_binary = (y_val_raw == class_id).astype(np.float32)

    n_pos_train = int(np.sum(y_train_binary == 1.0))
    n_neg_train = int(np.sum(y_train_binary == 0.0))

    # Flag low-data classes (do not skip; document only)
    is_indicative = class_name in INDICATIVE_ONLY_CLASSES or n_pos_train < INDICATIVE_THRESHOLD
    if is_indicative:
        print(
            f"[WARNING] Class '{class_name}' has only {n_pos_train} positive training samples. "
            f"Results are INDICATIVE ONLY and must not be compared to other classes.",
            flush=True,
        )

    # Class weights w_c = N / (2 * N_c)
    w_0, w_1 = compute_binary_class_weights(y_train_binary)
    print(
        f"Class '{class_name}' (ID {class_id}): "
        f"Positive = {n_pos_train}, Negative = {n_neg_train}, "
        f"Weights: w_0 = {w_0:.4f}, w_1 = {w_1:.4f} | Monitor: {monitor.upper()}",
        flush=True,
    )

    # Per-sample weights
    train_sample_weights = np.where(y_train_binary == 1.0, w_1, w_0).astype(np.float32)
    val_sample_weights = np.where(y_val_binary == 1.0, w_1, w_0).astype(np.float32)

    y_train_target_t = torch.from_numpy(y_train_binary).unsqueeze(1)
    y_train_w_t = torch.from_numpy(train_sample_weights).unsqueeze(1)
    y_val_target_t = torch.from_numpy(y_val_binary).unsqueeze(1)
    w_val_t = torch.from_numpy(val_sample_weights).unsqueeze(1)

    train_dataset = TensorDataset(x_train, y_train_target_t, y_train_w_t)
    val_dataset = TensorDataset(x_val, y_val_target_t, w_val_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=data_generator,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )

    model = BinaryDNN(
        in_features=x_train.shape[1],
        hidden=hidden,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_macro_f1_history: list[float] = []

    best_val_loss = float("inf")
    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_weights: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    print(f"Starting training for '{class_name}' with monitor='{monitor}'...", flush=True)
    class_start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()

        # Training phase
        model.train()
        train_weighted_loss_sum = 0.0
        train_samples = 0

        for x_b, y_b, w_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            w_b = w_b.to(device)

            optimizer.zero_grad()
            logits = model(x_b)
            loss_unweighted = criterion(logits, y_b)
            loss = (w_b * loss_unweighted).mean()
            loss.backward()
            optimizer.step()

            train_weighted_loss_sum += loss.item() * len(x_b)
            train_samples += len(x_b)

        epoch_train_loss = train_weighted_loss_sum / train_samples
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_weighted_loss_sum = 0.0
        val_samples = 0
        epoch_val_preds_list: list[np.ndarray] = []
        epoch_val_targets_list: list[np.ndarray] = []

        with torch.no_grad():
            for x_b, y_b, w_b in val_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                w_b = w_b.to(device)

                logits = model(x_b)
                loss_unweighted = criterion(logits, y_b)
                loss = (w_b * loss_unweighted).mean()

                val_weighted_loss_sum += loss.item() * len(x_b)
                val_samples += len(x_b)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).int().cpu().numpy()
                epoch_val_preds_list.append(preds)
                epoch_val_targets_list.append(y_b.int().cpu().numpy())

        epoch_val_loss = val_weighted_loss_sum / val_samples
        val_losses.append(epoch_val_loss)

        epoch_val_targets_all = np.vstack(epoch_val_targets_list).ravel()
        epoch_val_preds_all = np.vstack(epoch_val_preds_list).ravel()
        epoch_val_macro_f1 = compute_binary_macro_f1(epoch_val_targets_all, epoch_val_preds_all)
        val_macro_f1_history.append(epoch_val_macro_f1)

        epoch_duration = time.perf_counter() - epoch_start

        improved = check_monitor_improvement(
            monitor=monitor,
            current_f1=epoch_val_macro_f1,
            current_loss=epoch_val_loss,
            best_f1=best_val_macro_f1,
            best_loss=best_val_loss,
        )

        if improved:
            best_val_loss = epoch_val_loss
            best_val_macro_f1 = epoch_val_macro_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improved_mark = "*"
        else:
            patience_counter += 1
            improved_mark = ""

        if epoch <= 5 or epoch % 5 == 0 or patience_counter >= patience:
            print(
                f"Epoch {epoch:03d}/{max_epochs:03d} | "
                f"Train Loss: {epoch_train_loss:.5f} | "
                f"Val Loss: {epoch_val_loss:.5f} | "
                f"Val Macro F1: {epoch_val_macro_f1:.4f} | "
                f"Time: {epoch_duration:.2f}s {improved_mark}",
                flush=True,
            )

        if patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).",
                flush=True,
            )
            break

    elapsed_class_time = time.perf_counter() - class_start_time

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save weights: models_v2b/dnn_v2b_<class_name>.pt
    model_save_path = models_dir / f"dnn_v2b_{class_name}.pt"
    torch.save(best_weights if best_weights is not None else model.state_dict(), model_save_path)
    print(f"Saved best model weights to {model_save_path}", flush=True)

    # Final validation metrics using best weights
    model.eval()
    val_probs_list: list[np.ndarray] = []
    with torch.no_grad():
        for x_b, _, _ in val_loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_probs_list.append(probs)

    val_probs = np.vstack(val_probs_list).ravel()
    val_preds = (val_probs >= 0.5).astype(np.int32)

    # AUC requires at least one positive sample; guard for tiny classes
    n_pos_val = int(np.sum(y_val_binary == 1.0))
    if n_pos_val >= 1 and n_pos_val < len(y_val_binary):
        val_auc = float(roc_auc_score(y_val_binary, val_probs))
    else:
        val_auc = float("nan")
        print(
            f"[WARNING] ROC-AUC undefined for '{class_name}' (val positive count = {n_pos_val}). "
            f"Recorded as NaN.",
            flush=True,
        )

    val_f1_pos = float(f1_score(y_val_binary, val_preds, pos_label=1, zero_division=0))
    val_macro_f1 = compute_binary_macro_f1(y_val_binary, val_preds)

    print(
        f"Class: {class_name} | Best Epoch: {best_epoch} | "
        f"Best Val Loss: {best_val_loss:.4f} | "
        f"Val ROC-AUC: {val_auc:.4f} | Val F1: {val_f1_pos:.4f} | "
        f"Val Macro F1: {val_macro_f1:.4f} | Elapsed: {elapsed_class_time:.2f}s",
        flush=True,
    )

    # Save training history JSON
    history = {
        "class_id": class_id,
        "class_name": class_name,
        "monitor": monitor,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_macro_f1": best_val_macro_f1,
        "val_roc_auc": val_auc if not (isinstance(val_auc, float) and val_auc != val_auc) else None,
        "val_f1_threshold_0_5": val_f1_pos,
        "val_macro_f1": val_macro_f1,
        "elapsed_seconds": elapsed_class_time,
        "n_pos_train": n_pos_train,
        "n_neg_train": n_neg_train,
        "is_indicative_only": is_indicative,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_macro_f1_history": val_macro_f1_history,
    }
    history_path = artifacts_dir / f"training_history_v2b_{class_name}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save training curves PNG
    curves_path = artifacts_dir / f"training_curves_v2b_{class_name}.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    epochs_range = list(range(1, len(train_losses) + 1))

    ax1.plot(epochs_range, train_losses, label="Train Loss", color="royalblue")
    ax1.plot(epochs_range, val_losses, label="Validation Loss", color="darkorange")
    if best_epoch > 0:
        ax1.axvline(best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax1.set_ylabel("Weighted Loss")
    indicative_note_str = " [INDICATIVE ONLY]" if is_indicative else ""
    ax1.set_title(
        f"Training Curves - v2b {class_name} negfix (Monitor: {monitor.upper()}){indicative_note_str}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, val_macro_f1_history, label="Val Macro F1", color="forestgreen")
    if best_epoch > 0:
        ax2.axvline(best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(curves_path, dpi=150)
    plt.close()

    return history


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def train_all_negfix(
    config_path: Path | str | None = None,
    baseline_ratio: int = 3,
) -> list[dict[str, Any]]:
    """Build augmented training data and train one binary DNN per extra class.

    Steps:
      B. Build negatives: extra positives + extra negatives + baseline negatives.
      C. Retrain with same architecture/hyperparameters as train_extra.py.
      D. Save weights, histories, curves, and summary to new paths.

    Output:
      models_v2b/dnn_v2b_<ClassName>.pt
      models_v2b/label_map_v2b.json
      models_v2b/feature_names_v2b.json
      artifacts/v2b/training/training_history_v2b_<ClassName>.json
      artifacts/v2b/training/training_curves_v2b_<ClassName>.png
      artifacts/v2b/training/train_summary_v2b.json
    """
    root = get_project_root()
    cfg = load_v2_config(config_path)

    seed: int = cfg.get("seed", 42)
    data_generator = set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Seed: {seed}", flush=True)
    print(f"Baseline ratio: {baseline_ratio} baseline negatives per positive", flush=True)

    models_dir = root / MODELS_DIR_REL
    artifacts_dir = root / ARTIFACTS_DIR_REL

    # Safety: never write to frozen dirs or original v2 dirs
    check_negfix_output_safety(models_dir, artifacts_dir, root)

    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Feature names from frozen baseline (41 features)
    feature_names_path = root / "models" / "feature_names.json"
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names not found at {feature_names_path}")
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names: list[str] = json.load(f)

    assert len(feature_names) == 41, (
        f"Expected 41 features from models/feature_names.json, got {len(feature_names)}."
    )
    print(f"Feature names loaded: {len(feature_names)} features", flush=True)

    # Frozen baseline scaler (transform-only)
    scaler_path = root / "models" / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    assert_scaler_fitted(scaler)
    print(f"Scaler loaded (pre-fitted, transform-only): {scaler_path}", flush=True)

    # Extra-class data (train + val splits only; test never touched here)
    v2_processed = root / "data" / "v2" / "processed"
    train_extra_path = v2_processed / "train_extra.parquet"
    val_extra_path = v2_processed / "val_extra.parquet"

    for p in [train_extra_path, val_extra_path]:
        assert_no_test_data_loaded(p)
        if not p.exists():
            raise FileNotFoundError(f"Required parquet not found: {p}")

    print(f"Loading extra train: {train_extra_path}", flush=True)
    df_train_extra = pd.read_parquet(train_extra_path)
    print(f"Loading extra val:   {val_extra_path}", flush=True)
    df_val_extra = pd.read_parquet(val_extra_path)

    # Baseline splits (train + val only; never test)
    baseline_processed = root / "data" / "processed"
    baseline_train_path = baseline_processed / "train.parquet"
    baseline_val_path = baseline_processed / "val.parquet"

    for p in [baseline_train_path, baseline_val_path]:
        assert_no_test_data_loaded(p)
        if not p.exists():
            raise FileNotFoundError(f"Required baseline parquet not found: {p}")

    print(f"Loading baseline train: {baseline_train_path}", flush=True)
    df_baseline_train = pd.read_parquet(baseline_train_path)
    print(f"Loading baseline val:   {baseline_val_path}", flush=True)
    df_baseline_val = pd.read_parquet(baseline_val_path)

    # Verify feature columns exist in all loaded frames
    for df, name in [
        (df_train_extra, "train_extra"),
        (df_val_extra, "val_extra"),
        (df_baseline_train, "baseline_train"),
        (df_baseline_val, "baseline_val"),
    ]:
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"NOT FOUND: features {missing} missing from {name}. "
                f"Available columns: {list(df.columns)}"
            )

    # Extra-class label map (IDs 3, 4, 5)
    label_map_path = v2_processed / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Extra class label map not found at {label_map_path}")
    with open(label_map_path, "r", encoding="utf-8") as f:
        raw_map: dict[str, str] = json.load(f)
    label_map: dict[int, str] = {int(k): v for k, v in raw_map.items()}
    print(f"Extra class label map: {label_map}", flush=True)

    # Print data shape summary
    print(f"\nData shape summary:", flush=True)
    print(f"  train_extra:    {df_train_extra.shape}", flush=True)
    print(f"  val_extra:      {df_val_extra.shape}", flush=True)
    print(f"  baseline_train: {df_baseline_train.shape}", flush=True)
    print(f"  baseline_val:   {df_baseline_val.shape}", flush=True)
    for cid, cname in sorted(label_map.items()):
        n_tr = int((df_train_extra["label"] == cid).sum())
        n_v = int((df_val_extra["label"] == cid).sum())
        print(f"  {cname} (ID {cid}): train_pos={n_tr}, val_pos={n_v}", flush=True)
    for cid in sorted(df_baseline_train["label"].unique()):
        n_tr = int((df_baseline_train["label"] == cid).sum())
        n_v = int((df_baseline_val["label"] == cid).sum())
        print(f"  Baseline label {cid}: train={n_tr}, val={n_v}", flush=True)

    # Save feature names and label map to models_v2b/
    fn_out = models_dir / "feature_names_v2b.json"
    with open(fn_out, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Saved {len(feature_names)} feature names to {fn_out}", flush=True)

    lm_out = models_dir / "label_map_v2b.json"
    with open(lm_out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    print(f"Saved label map to {lm_out}", flush=True)

    print("\n=================================================================", flush=True)
    print("STARTING V2b NEGFIX EXTRA CLASSES TRAINING", flush=True)
    print(f"Classes: {label_map}", flush=True)
    print(
        f"Each model: positives + extra-class negatives + baseline negatives "
        f"(ratio={baseline_ratio})",
        flush=True,
    )
    print("=================================================================\n", flush=True)

    all_histories: list[dict[str, Any]] = []
    training_start = time.perf_counter()

    for class_id in sorted(label_map.keys()):
        class_name = label_map[class_id]
        print(f"\n==========================================", flush=True)
        print(f"Building augmented data for '{class_name}' (ID {class_id}) vs Rest", flush=True)
        print(f"==========================================", flush=True)

        x_train_np, y_train_np = build_augmented_split(
            df_extra=df_train_extra,
            df_baseline=df_baseline_train,
            class_id=class_id,
            feature_names=feature_names,
            scaler=scaler,
            baseline_ratio=baseline_ratio,
            seed=seed,
            split_name="train",
        )

        x_val_np, y_val_np = build_augmented_split(
            df_extra=df_val_extra,
            df_baseline=df_baseline_val,
            class_id=class_id,
            feature_names=feature_names,
            scaler=scaler,
            baseline_ratio=baseline_ratio,
            seed=seed,
            split_name="val",
        )

        x_train_t = torch.from_numpy(x_train_np)
        x_val_t = torch.from_numpy(x_val_np)

        # Verify feature count
        assert x_train_t.shape[1] == 41, (
            f"Feature count mismatch: expected 41, got {x_train_t.shape[1]}"
        )

        print(f"\nTraining '{class_name}'...", flush=True)
        history = train_binary_model_negfix(
            class_id=class_id,
            class_name=class_name,
            x_train=x_train_t,
            y_train_raw=y_train_np,
            x_val=x_val_t,
            y_val_raw=y_val_np,
            config=cfg,
            device=device,
            models_dir=models_dir,
            artifacts_dir=artifacts_dir,
            data_generator=data_generator,
        )
        all_histories.append(history)

    total_elapsed = time.perf_counter() - training_start

    # Save aggregate summary
    summary = {
        "total_elapsed_seconds": total_elapsed,
        "device": str(device),
        "seed": seed,
        "baseline_ratio": baseline_ratio,
        "models_dir": str(models_dir),
        "negatives_strategy": (
            "extra-class negatives + stratified baseline negatives "
            f"(Benign/DoS/DDoS, ratio={baseline_ratio})"
        ),
        "classes_trained": [h["class_name"] for h in all_histories],
        "per_class": {
            h["class_name"]: {
                "best_epoch": h["best_epoch"],
                "val_roc_auc": h["val_roc_auc"],
                "val_f1_threshold_0_5": h["val_f1_threshold_0_5"],
                "val_macro_f1": h["val_macro_f1"],
                "n_pos_train": h["n_pos_train"],
                "n_neg_train": h["n_neg_train"],
                "is_indicative_only": h["is_indicative_only"],
            }
            for h in all_histories
        },
    }
    summary_path = artifacts_dir / "train_summary_v2b.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=================================================================", flush=True)
    print("V2b NEGFIX TRAINING COMPLETE", flush=True)
    print(f"Total elapsed: {total_elapsed:.1f}s", flush=True)
    for h in all_histories:
        indicative = " [INDICATIVE ONLY]" if h["is_indicative_only"] else ""
        auc_str = f"{h['val_roc_auc']:.4f}" if h["val_roc_auc"] is not None else "N/A"
        print(
            f"  {h['class_name']}: Best Epoch {h['best_epoch']} | "
            f"Val Macro F1 {h['val_macro_f1']:.4f} | "
            f"ROC-AUC {auc_str}{indicative}",
            flush=True,
        )
    print("=================================================================\n", flush=True)

    return all_histories


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/v2_train_negfix.py",
        description=(
            "Retrain v2 extra-class binary DNNs with baseline classes as negatives. "
            "Writes only to models_v2b/ and artifacts/v2b/training/. "
            "Refused if output paths conflict with frozen directories."
        ),
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config_v2.yaml.")
    parser.add_argument(
        "--confirm-train",
        action="store_true",
        help="Required. Confirms intentional training run.",
    )
    parser.add_argument(
        "--baseline-ratio",
        type=int,
        default=3,
        help=(
            "Number of baseline negative rows to draw per positive. "
            "Default 3. Increase to 5 if leakage criterion still fails."
        ),
    )
    args = parser.parse_args(argv)

    if not args.confirm_train:
        print(
            "Refusing to train: re-run with --confirm-train to confirm this is intentional.",
            file=sys.stderr,
        )
        return 2

    train_all_negfix(config_path=args.config, baseline_ratio=args.baseline_ratio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

