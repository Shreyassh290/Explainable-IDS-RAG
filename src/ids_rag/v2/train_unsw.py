"""Training and evaluation pipeline for UNSW-NB15 dataset.

Trains a binary DNN classifier (128-64-32, dropout 0.3) on UNSW-NB15 training set (175,341 flows)
and evaluates on held-out UNSW-NB15 test set (82,332 flows).

Replicates the reference paper benchmark (95.30% accuracy) while applying:
- Dropout regularization (p=0.30)
- Validation early stopping (patience=5)
- Standard scaling on numeric features
- Performance breakdown across all attack categories

All artifacts are written to models_unsw/ and artifacts/v2/unsw_eval/ (never models/ or data/processed/).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root
from ids_rag.models.dnn import BinaryDNN


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_preprocess_unsw(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.Series, StandardScaler]:
    """Load train/test CSVs, encode categorical features, and scale numeric attributes."""
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path = data_dir / "UNSW_NB15_testing-set.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"UNSW-NB15 training file not found at {train_path}. "
            "Please run `uv run python scripts/v2_download_unsw.py` first."
        )

    print("Loading UNSW-NB15 training data...", flush=True)
    df_train = pd.read_csv(train_path)

    if test_path.exists():
        print("Loading UNSW-NB15 official testing split...", flush=True)
        df_test = pd.read_csv(test_path)
    else:
        print("Official testing split not found; performing 75/25 stratified split on training set...", flush=True)
        df_train, df_test = train_test_split(
            df_train,
            test_size=0.25,
            random_state=42,
            stratify=df_train["label"],
        )
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

    print(f"Loaded train: {len(df_train):,} rows, test: {len(df_test):,} rows.")

    # Drop non-feature ID column if present (avoid deprecated inplace)
    cols_to_drop = [c for c in ["id", "ID"] if c in df_train.columns]
    if cols_to_drop:
        df_train = df_train.drop(columns=cols_to_drop)
    cols_to_drop_test = [c for c in ["id", "ID"] if c in df_test.columns]
    if cols_to_drop_test:
        df_test = df_test.drop(columns=cols_to_drop_test)

    # Extract target label and attack categories
    y_train = df_train["label"].to_numpy(dtype=np.float32)
    y_test = df_test["label"].to_numpy(dtype=np.float32)

    test_attack_cats = df_test["attack_cat"].fillna("Normal").astype(str)

    # Non-feature target columns to exclude
    target_cols = ["label", "attack_cat"]
    feature_cols = [c for c in df_train.columns if c not in target_cols]

    # Process all non-numeric columns (proto, service, state) with frequency encoding
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df_train[c])]
    print(f"Categorical features to encode: {cat_cols}", flush=True)

    for c in cat_cols:
        freq_map = df_train[c].astype(str).value_counts(normalize=True).to_dict()
        df_train[c] = df_train[c].astype(str).map(freq_map).fillna(0.0).astype(np.float32)
        df_test[c] = df_test[c].astype(str).map(freq_map).fillna(0.0).astype(np.float32)

    X_train_raw = df_train[feature_cols].to_numpy(dtype=np.float32)
    X_test_raw = df_test[feature_cols].to_numpy(dtype=np.float32)

    # Impute any missing or infinite values
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit scaler on training set only (zero data leakage)
    print("Fitting StandardScaler on training data...", flush=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    return X_train, y_train, X_test, y_test, feature_cols, test_attack_cats, scaler


def train_unsw_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    in_features: int,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 0.001,
    patience: int = 5,
) -> tuple[BinaryDNN, dict[str, list[float]]]:
    """Train binary DNN with early stopping on validation macro F1."""
    model = BinaryDNN(
        in_features=in_features,
        hidden=[128, 64, 32],
        dropout=0.3,
        use_batchnorm=False,
    ).to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0

    print("Beginning neural training loop...", flush=True)
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss_acc = 0.0
        n_train = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            train_loss_acc += loss.item() * len(bx)
            n_train += len(bx)

        train_loss = train_loss_acc / n_train

        # Validation
        model.eval()
        val_loss_acc = 0.0
        n_val = 0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss_acc += loss.item() * len(bx)
                n_val += len(bx)

                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                val_preds_list.append((probs >= 0.5).astype(int))
                val_targets_list.append(by.squeeze(1).cpu().numpy().astype(int))

        val_loss = val_loss_acc / n_val
        val_preds = np.concatenate(val_preds_list)
        val_targets = np.concatenate(val_targets_list)
        val_f1 = float(f1_score(val_targets, val_preds, average="macro"))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}",
            flush=True,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (Best Val Macro F1: {best_val_f1:.4f}).", flush=True)
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def main() -> None:
    set_seed(42)
    root = get_project_root()
    data_dir = root / "data" / "unsw"
    models_dir = root / "models_unsw"
    artifacts_dir = root / "artifacts" / "v2" / "unsw_eval"

    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}", flush=True)

    # 1. Load data
    X_train, y_train, X_test, y_test, feature_names, test_attack_cats, scaler = load_and_preprocess_unsw(data_dir)

    # Create 15% validation split from training set for early stopping
    n_train_full = len(X_train)
    val_size = int(n_train_full * 0.15)
    indices = np.random.permutation(n_train_full)
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    print(f"Train split: {len(X_tr):,} | Val split: {len(X_val):,} | Test split: {len(X_test):,}")

    # 2. Train model
    t0 = time.perf_counter()
    model, history = train_unsw_model(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        in_features=len(feature_names),
        device=device,
        epochs=40,
        batch_size=512,
        lr=0.001,
        patience=5,
    )
    train_time = time.perf_counter() - t0
    print(f"Model training completed in {train_time:.2f} seconds.", flush=True)

    # 3. Evaluate on held-out test set
    model.eval()
    test_tensor = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        test_logits = model(test_tensor).squeeze(1).cpu().numpy()
        test_probs = 1.0 / (1.0 + np.exp(-test_logits))
        test_preds = (test_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, test_preds))
    macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    cm = confusion_matrix(y_test, test_preds).tolist()

    # Benign False Positive Rate (FP / (FP + TN))
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Breakdown per attack category
    df_eval = pd.DataFrame({"attack_cat": test_attack_cats, "y_true": y_test, "y_pred": test_preds})
    per_cat: dict[str, dict[str, Any]] = {}
    for cat_name, group in df_eval.groupby("attack_cat"):
        cat_acc = float(accuracy_score(group["y_true"], group["y_pred"]))
        cat_support = int(len(group))
        per_cat[str(cat_name)] = {"accuracy": cat_acc, "support": cat_support}

    print("\n" + "=" * 60)
    print("UNSW-NB15 TEST SET EVALUATION SUMMARY")
    print(f"Total Test Flows: {len(y_test):,}")
    print(f"Test Accuracy:    {acc:.4f} ({acc*100:.2f}%)  [Reference Paper: 95.30%]")
    print(f"Macro F1-Score:   {macro_f1:.4f}")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"Benign FPR:       {fpr*100:.2f}%")
    print("=" * 60)

    # 4. Save artifacts
    model_path = models_dir / "dnn_unsw_attack.pt"
    scaler_path = models_dir / "scaler_unsw.joblib"
    feat_path = models_dir / "feature_names_unsw.json"

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    summary = {
        "test_flows_total": len(y_test),
        "test_accuracy": acc,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "benign_fpr": fpr,
        "confusion_matrix": cm,
        "per_category_accuracy": per_cat,
        "training_time_seconds": train_time,
        "paper_comparison": {
            "paper_accuracy": 0.9530,
            "our_accuracy": acc,
            "difference": round(acc - 0.9530, 4),
        },
    }

    metrics_path = artifacts_dir / "unsw_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot training curves
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("UNSW-NB15 Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "training_curves_unsw.png", dpi=150)
    plt.close()

    print(f"Model saved to:   {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
