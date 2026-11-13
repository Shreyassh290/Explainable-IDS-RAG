"""Training module for extra attack class binary DNNs (Bot, FTP-BruteForce, SSH-Bruteforce).

Trains one binary DNN per extra class using the identical architecture, loss function,
optimizer, and early-stopping strategy as the existing three-class pipeline.

Key differences from the baseline training (ids_rag.models.train):
- Reads from data/v2/processed/ (never data/processed/).
- Saves model weights to models_v2/ (never models/).
- Saves artifacts to artifacts/v2/training/ (never artifacts/).
- Uses config_v2.yaml (never config.yaml).
- Refuses to run if any output path conflicts with frozen directories.

FTP-BruteForce note: Only 53 usable rows survive deduplication in CICIDS2018.
Training that class is documented as indicative only. Results for FTP-BruteForce
must NOT be compared to Bot or SSH-Bruteforce in evaluation summaries.
"""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ids_rag.config import get_project_root
from ids_rag.models.dnn import (
    BinaryDNN,
    check_monitor_improvement,
    compute_binary_class_weights,
    compute_binary_macro_f1,
)
from ids_rag.v2.prepare_extra import load_v2_config


# ---------------------------------------------------------------------------
# Safety
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
INDICATIVE_THRESHOLD = 500  # classes with fewer train rows are flagged


def check_v2_output_safety(
    models_dir: Path,
    artifacts_dir: Path,
    root: Path,
) -> None:
    """Refuse to run if any output directory overlaps with a frozen directory."""
    for frozen_rel in FROZEN_DIRS:
        frozen_abs = (root / frozen_rel).resolve()
        for out_dir in [models_dir, artifacts_dir]:
            out_abs = out_dir.resolve()
            if out_abs == frozen_abs or frozen_abs in out_abs.parents or out_abs in frozen_abs.parents:
                raise ValueError(
                    f"SAFETY REFUSAL: Output directory '{out_abs}' conflicts with "
                    f"frozen directory '{frozen_abs}'. Stage 2 must write only to "
                    f"models_v2/ and artifacts/v2/training/."
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
# Single binary model training (identical algorithm to baseline)
# ---------------------------------------------------------------------------

def train_binary_model_v2(
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
    """Train one binary v2 classifier (class_id vs rest) with class-weighted BCE loss.

    Identical training algorithm to ids_rag.models.train.train_binary_model.
    Saves weights to models_v2/dnn_v2_<class_name>.pt.
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

    # Save weights: models_v2/dnn_v2_<class_name>.pt
    model_save_path = models_dir / f"dnn_v2_{class_name}.pt"
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
        "val_roc_auc": val_auc,
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
    history_path = artifacts_dir / f"training_history_v2_{class_name}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save training curves PNG
    curves_path = artifacts_dir / f"training_curves_v2_{class_name}.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    epochs_range = list(range(1, len(train_losses) + 1))

    ax1.plot(epochs_range, train_losses, label="Train Loss", color="royalblue")
    ax1.plot(epochs_range, val_losses, label="Validation Loss", color="darkorange")
    if best_epoch > 0:
        ax1.axvline(best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax1.set_ylabel("Weighted Loss")
    indicative_note = " [INDICATIVE ONLY]" if is_indicative else ""
    ax1.set_title(f"Training Curves - v2 {class_name} (Monitor: {monitor.upper()}){indicative_note}")
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

def train_all_extra(config_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load v2 parquet data and train one binary DNN per extra class.

    Output:
      models_v2/dnn_v2_<class_name>.pt        -- best weights per class
      models_v2/feature_names_v2.json         -- ordered 41 feature names
      models_v2/label_map_v2.json             -- {id: class_name} for extra classes
      artifacts/v2/training/training_history_v2_<class_name>.json
      artifacts/v2/training/training_curves_v2_<class_name>.png
      artifacts/v2/training/train_summary_v2.json
    """
    root = get_project_root()
    cfg = load_v2_config(config_path)

    seed = cfg.get("seed", 42)
    data_generator = set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    paths = cfg["paths"]
    processed_dir = root / paths.get("processed", "data/v2/processed")
    models_dir = root / "models_v2"
    artifacts_dir = root / "artifacts" / "v2" / "training"

    # Enforce safety: never write to frozen directories
    check_v2_output_safety(models_dir, artifacts_dir, root)

    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Read label map (extra classes only: IDs 3, 4, 5)
    label_map_path = processed_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"Extra class label map not found at {label_map_path}. "
            f"Run `python -m ids_rag.v2.prepare_extra` first."
        )
    with open(label_map_path, "r", encoding="utf-8") as f:
        raw_map: dict[str, str] = json.load(f)
    # Keys from JSON are strings; convert to int
    label_map: dict[int, str] = {int(k): v for k, v in raw_map.items()}

    # Load train and val parquets (never load test.parquet during training)
    train_path = processed_dir / "train_extra.parquet"
    val_path = processed_dir / "val_extra.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"train_extra.parquet or val_extra.parquet not found in {processed_dir}. "
            f"Run `python -m ids_rag.v2.prepare_extra` first."
        )

    print(f"Loading {train_path} ...", flush=True)
    df_train = pd.read_parquet(train_path)
    print(f"Loading {val_path} ...", flush=True)
    df_val = pd.read_parquet(val_path)

    feature_cols = [c for c in df_train.columns if c != "label"]

    # Save feature names for the v2 ensemble loader
    feature_names_path = models_dir / "feature_names_v2.json"
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved {len(feature_cols)} feature names to {feature_names_path}", flush=True)

    # Save label map copy in models_v2/ for the ensemble loader
    label_map_out = models_dir / "label_map_v2.json"
    with open(label_map_out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    print(f"Saved label map to {label_map_out}", flush=True)

    x_train_np = df_train[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_train_np = df_train["label"].to_numpy(dtype=np.int64, copy=True)
    x_val_np = df_val[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_val_np = df_val["label"].to_numpy(dtype=np.int64, copy=True)

    x_train_t = torch.from_numpy(x_train_np)
    x_val_t = torch.from_numpy(x_val_np)

    print("\n=================================================================", flush=True)
    print("STARTING V2 EXTRA CLASSES TRAINING (STAGE 2)", flush=True)
    print(f"Classes: {label_map}", flush=True)
    print(f"Train rows: {len(df_train):,}  |  Val rows: {len(df_val):,}", flush=True)
    print("=================================================================\n", flush=True)

    all_histories: list[dict[str, Any]] = []
    training_start = time.perf_counter()

    for class_id in sorted(label_map.keys()):
        class_name = label_map[class_id]
        print(f"\n==========================================", flush=True)
        print(f"Training binary model for '{class_name}' (ID {class_id}) vs Rest", flush=True)
        print(f"==========================================", flush=True)
        history = train_binary_model_v2(
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
        "models_dir": str(models_dir),
        "processed_dir": str(processed_dir),
        "classes_trained": [h["class_name"] for h in all_histories],
        "per_class": {
            h["class_name"]: {
                "best_epoch": h["best_epoch"],
                "val_roc_auc": h["val_roc_auc"],
                "val_f1_threshold_0_5": h["val_f1_threshold_0_5"],
                "val_macro_f1": h["val_macro_f1"],
                "n_pos_train": h["n_pos_train"],
                "is_indicative_only": h["is_indicative_only"],
            }
            for h in all_histories
        },
    }
    summary_path = artifacts_dir / "train_summary_v2.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=================================================================", flush=True)
    print("V2 EXTRA CLASSES TRAINING COMPLETE", flush=True)
    print(f"Total elapsed: {total_elapsed:.1f}s", flush=True)
    for h in all_histories:
        indicative = " [INDICATIVE ONLY]" if h["is_indicative_only"] else ""
        print(
            f"  {h['class_name']}: Best Epoch {h['best_epoch']} | "
            f"Val Macro F1 {h['val_macro_f1']:.4f} | "
            f"ROC-AUC {h['val_roc_auc']:.4f}{indicative}",
            flush=True,
        )
    print("=================================================================\n", flush=True)

    return all_histories


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m ids_rag.v2.train_extra",
        description=(
            "Train binary DNNs for extra attack classes (Bot, FTP-BruteForce, SSH-Bruteforce). "
            "Writes only to models_v2/ and artifacts/v2/training/. "
            "Refused if output paths conflict with frozen directories."
        ),
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config_v2.yaml.")
    parser.add_argument(
        "--confirm-train",
        action="store_true",
        help="Required. Confirms intentional training run.",
    )
    args = parser.parse_args(argv)

    if not args.confirm_train:
        print(
            "Refusing to train: re-run with --confirm-train to confirm this is intentional.",
            file=sys.stderr,
        )
        return 2

    train_all_extra(config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

