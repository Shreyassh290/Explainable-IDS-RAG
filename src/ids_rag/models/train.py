"""Model training loop with early stopping, class-weighted loss, and artifact logging."""

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

from ids_rag.config import get_project_root, load_config
from ids_rag.models.dnn import (
    BinaryDNN,
    check_monitor_improvement,
    compute_binary_class_weights,
    compute_binary_macro_f1,
)


def set_seed(seed: int = 42) -> torch.Generator:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def train_binary_model(
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
    """Train one binary classifier for class_id vs rest using class-weighted BCE loss."""
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

    # Class weights w_c = N / (2 * N_c)
    w_0, w_1 = compute_binary_class_weights(y_train_binary)
    print(
        f"Class '{class_name}' (ID {class_id}): "
        f"Positive count = {int(np.sum(y_train_binary == 1.0))}, "
        f"Negative count = {int(np.sum(y_train_binary == 0.0))}, "
        f"Weights: w_0 = {w_0:.4f}, w_1 = {w_1:.4f} | Monitor: {monitor.upper()}",
        flush=True,
    )

    # Per-sample weights
    train_sample_weights = np.where(y_train_binary == 1.0, w_1, w_0).astype(np.float32)
    val_sample_weights = np.where(y_val_binary == 1.0, w_1, w_0).astype(np.float32)

    # Convert targets and sample weights to tensors
    y_train_t = torch.from_numpy(train_sample_weights).unsqueeze(1)
    # Target tensor
    y_train_target_t = torch.from_numpy(y_train_binary).unsqueeze(1)
    y_val_target_t = torch.from_numpy(y_val_binary).unsqueeze(1)
    w_val_t = torch.from_numpy(val_sample_weights).unsqueeze(1)

    # Build DataLoaders
    train_dataset = TensorDataset(x_train, y_train_target_t, y_train_t)
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

    # Initialize model, loss function, and optimizer
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

        # Validation phase: compute weighted loss and predictions at 0.5 threshold
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

        # Check early stopping according to configured monitor
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

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save model weights to models/dnn_<class>.pt
    model_save_path = models_dir / f"dnn_{class_name}.pt"
    torch.save(best_weights if best_weights is not None else model.state_dict(), model_save_path)
    print(f"Saved best model weights to {model_save_path}", flush=True)

    # Evaluate validation metrics at threshold 0.5 using best weights
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
    val_auc = float(roc_auc_score(y_val_binary, val_probs))
    val_f1_pos = float(f1_score(y_val_binary, val_preds, pos_label=1, zero_division=0))
    val_macro_f1 = compute_binary_macro_f1(y_val_binary, val_preds)

    print(
        f"Class: {class_name} | Best Epoch: {best_epoch} | "
        f"Best Val Loss: {best_val_loss:.4f} | "
        f"Val ROC-AUC: {val_auc:.4f} | Val F1: {val_f1_pos:.4f} | "
        f"Val Macro F1: {val_macro_f1:.4f} | Elapsed Time: {elapsed_class_time:.2f}s",
        flush=True,
    )

    # Log training history to artifacts/training_history_<class>.json
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
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_macro_f1_history": val_macro_f1_history,
    }
    history_path = artifacts_dir / f"training_history_{class_name}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot loss curves and macro F1 curves to artifacts/training_curves_<class>.png
    curves_path = artifacts_dir / f"training_curves_{class_name}.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    epochs_range = list(range(1, len(train_losses) + 1))

    # Subplot 1: Losses
    ax1.plot(epochs_range, train_losses, label="Train Loss", color="royalblue")
    ax1.plot(epochs_range, val_losses, label="Validation Loss", color="darkorange")
    if best_epoch > 0:
        ax1.axvline(best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax1.set_ylabel("Weighted Loss")
    ax1.set_title(f"Training Curves - {class_name} (Monitor: {monitor.upper()})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Validation Macro F1
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


def train_all() -> None:
    """Load preprocessed data, save feature names, and train one binary DNN per class."""
    config = load_config()
    root = get_project_root()

    seed = config.get("seed", 42)
    data_generator = set_seed(seed)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    paths_cfg = config.get("paths", {})
    processed_dir = root / paths_cfg.get("processed", "data/processed")
    models_dir = root / paths_cfg.get("models", "models")
    artifacts_dir = root / paths_cfg.get("artifacts", "artifacts")

    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Read label map
    label_map_path = processed_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found at {label_map_path}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map: dict[str, str] = json.load(f)

    # Load train and validation parquets (do NOT load test.parquet)
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("train.parquet or val.parquet not found in processed directory.")

    print(f"Loading {train_path} and {val_path}...", flush=True)
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    # Extract feature columns (every column except 'label')
    feature_cols = [c for c in df_train.columns if c != "label"]
    feature_names_path = models_dir / "feature_names.json"
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved {len(feature_cols)} feature column names to {feature_names_path}", flush=True)

    # Extract numpy arrays and convert features to tensors once
    x_train_np = df_train[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_train_np = df_train["label"].to_numpy(dtype=np.int64, copy=True)

    x_val_np = df_val[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_val_np = df_val["label"].to_numpy(dtype=np.int64, copy=True)

    x_train_t = torch.from_numpy(x_train_np)
    x_val_t = torch.from_numpy(x_val_np)

    # Train ONE binary model per class (one-vs-rest)
    for class_id_str, class_name in sorted(label_map.items(), key=lambda item: int(item[0])):
        class_id = int(class_id_str)
        print(f"\n==========================================", flush=True)
        print(f"Training binary model for class '{class_name}' (ID {class_id}) vs Rest", flush=True)
        print(f"==========================================", flush=True)
        train_binary_model(
            class_id=class_id,
            class_name=class_name,
            x_train=x_train_t,
            y_train_raw=y_train_np,
            x_val=x_val_t,
            y_val_raw=y_val_np,
            config=config,
            device=device,
            models_dir=models_dir,
            artifacts_dir=artifacts_dir,
            data_generator=data_generator,
        )


def main(argv=None) -> int:
    """Command-line entry point. Refuses to retrain unless --confirm-retrain is given."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m ids_rag.models.train",
        description=(
            "Retrain the three binary DNNs from scratch. This overwrites "
            "models/dnn_*.pt, models/feature_names.json and the training histories "
            "and curves in artifacts/. It is refused unless --confirm-retrain is passed."
        ),
    )
    parser.add_argument(
        "--confirm-retrain",
        action="store_true",
        help="Required. Confirms that overwriting the trained models is intended.",
    )
    args = parser.parse_args(argv)
    if not args.confirm_retrain:
        print(
            "Refusing to train: this would overwrite models/dnn_*.pt and the training "
            "histories and curves. Re-run with --confirm-retrain if that is intended.",
            file=sys.stderr,
        )
        return 2
    train_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
