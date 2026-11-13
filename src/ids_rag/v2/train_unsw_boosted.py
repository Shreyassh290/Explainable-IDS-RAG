"""Boosted training and optimization pipeline for UNSW-NB15 dataset.

Applies modern machine learning optimizations to maximize detection accuracy:
1. Log1p Transformation on Heavy-Tailed Features (dur, sbytes, dbytes, sload, dload, sinpkt, etc.).
2. Removal of Random 32-bit TCP Sequence Numbers (stcpb, dtcpb) which add noise.
3. Target/One-Hot Categorical Encoding for Protocol, Service, and State.
4. Class-Weighted Loss (pos_weight) compensating for the 68/32 class imbalance.
5. Neural Architecture with Batch Normalization, LeakyReLU, and Dropout (p=0.25).
6. Learning Rate Scheduling (ReduceLROnPlateau).
7. Validation-calibrated decision threshold optimization.

Writes strictly to models_unsw/ and artifacts/v2/unsw_eval/ without touching frozen baseline files.
"""

from __future__ import annotations

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root


# Skewed features benefiting from log1p transform
SKEWED_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
    "sjit", "djit", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "response_body_len"
]

# Random 32-bit OS initial sequence numbers (noise in classification)
NOISE_COLUMNS = ["id", "ID", "stcpb", "dtcpb"]


class BoostedDNN(nn.Module):
    """Deep Neural Network with Batch Normalization, Dropout, and LeakyReLU."""

    def __init__(
        self,
        in_features: int,
        hidden: list[int] | None = None,
        dropout: float = 0.25,
    ) -> None:
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

        # Final single output logit
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_boosted_unsw(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.Series, StandardScaler]:
    """Load and process UNSW-NB15 with log1p transforms, one-hot encoding, and scaling."""
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path = data_dir / "UNSW_NB15_testing-set.csv"

    print("Loading UNSW-NB15 raw CSVs...", flush=True)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 1. Drop noise columns
    for c in NOISE_COLUMNS:
        if c in df_train.columns:
            df_train = df_train.drop(columns=[c])
        if c in df_test.columns:
            df_test = df_test.drop(columns=[c])

    y_train = df_train["label"].to_numpy(dtype=np.float32)
    y_test = df_test["label"].to_numpy(dtype=np.float32)
    test_attack_cats = df_test["attack_cat"].fillna("Normal").astype(str)

    target_cols = ["label", "attack_cat"]
    feature_cols = [c for c in df_train.columns if c not in target_cols]

    # 2. Log1p transformation on skewed numeric attributes
    print("Applying log1p transformation to heavy-tailed features...", flush=True)
    for col in SKEWED_FEATURES:
        if col in feature_cols:
            df_train[col] = np.log1p(np.maximum(df_train[col].to_numpy(dtype=np.float32), 0.0))
            df_test[col] = np.log1p(np.maximum(df_test[col].to_numpy(dtype=np.float32), 0.0))

    # 3. Categorical encoding (One-Hot for Service and State, Top-K for Proto)
    print("Encoding categorical protocol, service, and state features...", flush=True)
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df_train[c])]

    for col in cat_cols:
        # Keep top 10 categories, bucket the rest into 'other'
        top_cats = df_train[col].value_counts().nlargest(10).index.tolist()
        df_train[col] = df_train[col].apply(lambda x: x if x in top_cats else "other")
        df_test[col] = df_test[col].apply(lambda x: x if x in top_cats else "other")

    # One-hot encode categoricals alignment
    df_combined = pd.concat([df_train[feature_cols], df_test[feature_cols]], axis=0)
    df_combined_encoded = pd.get_dummies(df_combined, columns=cat_cols, drop_first=True, dtype=np.float32)

    n_train = len(df_train)
    df_train_enc = df_combined_encoded.iloc[:n_train].copy()
    df_test_enc = df_combined_encoded.iloc[n_train:].copy()

    final_feature_names = df_train_enc.columns.tolist()
    print(f"Engineered feature dimensionality: {len(final_feature_names)} features.", flush=True)

    X_train_raw = np.nan_to_num(df_train_enc.to_numpy(dtype=np.float32), nan=0.0)
    X_test_raw = np.nan_to_num(df_test_enc.to_numpy(dtype=np.float32), nan=0.0)

    # 4. Standard scale
    print("Fitting StandardScaler on engineered training features...", flush=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    return X_train, y_train, X_test, y_test, final_feature_names, test_attack_cats, scaler


def train_boosted_unsw(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    in_features: int,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 0.002,
    patience: int = 7,
) -> tuple[BoostedDNN, dict[str, list[float]]]:
    """Train BoostedDNN with class-weighted loss, scheduler, and early stopping."""
    model = BoostedDNN(
        in_features=in_features,
        hidden=[256, 128, 64, 32],
        dropout=0.25,
    ).to(device)

    # Class weighting to balance Normal vs Attack gradients
    n_pos = np.sum(y_train == 1.0)
    n_neg = np.sum(y_train == 0.0)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device, dtype=torch.float32)
    print(f"Computed loss positive class weight: {pos_weight.item():.4f} (neg={n_neg:,}, pos={n_pos:,})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    best_val_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0

    print("Beginning neural training loop with learning rate scheduling...", flush=True)
    for epoch in range(1, epochs + 1):
        # Training phase
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

        # Validation phase
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
        val_acc = float(accuracy_score(val_targets, val_preds))
        val_f1 = float(f1_score(val_targets, val_preds, average="macro"))

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}",
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

    # 1. Preprocess
    X_train, y_train, X_test, y_test, feature_names, test_attack_cats, scaler = preprocess_boosted_unsw(data_dir)

    # 15% validation split
    n_train_full = len(X_train)
    val_size = int(n_train_full * 0.15)
    indices = np.random.permutation(n_train_full)
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    print(f"Train split: {len(X_tr):,} | Val split: {len(X_val):,} | Test split: {len(X_test):,}")

    # 2. Train Boosted Model
    t0 = time.perf_counter()
    model, history = train_boosted_unsw(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        in_features=len(feature_names),
        device=device,
        epochs=35,
        batch_size=512,
        lr=0.002,
        patience=7,
    )
    train_time = time.perf_counter() - t0
    print(f"Boosted neural training completed in {train_time:.2f} seconds.", flush=True)

    # 3. Optimal Threshold Calibration on Validation Split
    model.eval()
    val_tensor = torch.from_numpy(X_val).to(device)
    with torch.no_grad():
        val_logits = model(val_tensor).squeeze(1).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    thresholds = np.linspace(0.35, 0.75, 81)
    best_tau = 0.50
    best_val_score = 0.0

    for tau in thresholds:
        preds = (val_probs >= tau).astype(int)
        score = accuracy_score(y_val, preds)
        if score > best_val_score:
            best_val_score = score
            best_tau = tau

    print(f"\nCalibrated optimal threshold tau = {best_tau:.4f} (Validation Acc: {best_val_score*100:.2f}%)")

    # 4. Final Evaluation on Official Test Split (82,332 flows)
    test_tensor = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        test_logits = model(test_tensor).squeeze(1).cpu().numpy()
        test_probs = 1.0 / (1.0 + np.exp(-test_logits))
        test_preds = (test_probs >= best_tau).astype(int)

    acc = float(accuracy_score(y_test, test_preds))
    macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    cm = confusion_matrix(y_test, test_preds).tolist()

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
    print("BOOSTED UNSW-NB15 TEST SET EVALUATION SUMMARY")
    print(f"Total Test Flows: {len(y_test):,}")
    print(f"Test Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1-Score:   {macro_f1:.4f}")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"Benign FPR:       {fpr*100:.2f}%")
    print("=" * 60)
    print("\nPer-Category Detection Rates:")
    for cat, data in sorted(per_cat.items()):
        print(f"  - {cat:16s}: {data['accuracy']*100:6.2f}% (Support: {data['support']:,})")

    # Save artifacts
    model_path = models_dir / "dnn_unsw_boosted.pt"
    scaler_path = models_dir / "scaler_unsw_boosted.joblib"
    feat_path = models_dir / "feature_names_unsw_boosted.json"

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    summary = {
        "test_flows_total": len(y_test),
        "test_accuracy": acc,
        "validation_accuracy": best_val_score,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "benign_fpr": fpr,
        "optimal_threshold": float(best_tau),
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
    plt.title("Boosted UNSW-NB15 Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "training_curves_unsw_boosted.png", dpi=150)
    plt.close()

    print(f"\nModel saved to:   {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

