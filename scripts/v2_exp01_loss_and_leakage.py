"""Experiment 1 (EXP-01): Leak-Free Data Partitioning & Loss Re-Weighting Comparison.

Evaluates three loss formulations under strict leak-free preprocessing:
  1. EXP-01A: Standard Unweighted BCE (pos_weight = 1.0)
  2. EXP-01B: Prior-Adjusted BCE (pos_weight dynamically computed from y_tr)
  3. EXP-01C: Focal Loss (alpha = 0.5, gamma = 2.0)

Design Rules:
- Splits training data into 85% train and 15% val BEFORE fitting scaler or encoder.
- Fits StandardScaler and categorical encodings strictly on X_tr.
- Evaluates strictly on validation data to select optimal thresholds and Pareto trade-offs.
- Holds the official test set untouched during model selection.
- Preserves all frozen baseline files and existing model checkpoints.
- All artifacts saved to models_unsw/experiments/exp01/ and artifacts/v2/unsw_eval/experiments/exp01/.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root


# Skewed features benefiting from log1p transform
SKEWED_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
    "sjit", "djit", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "response_body_len"
]

NOISE_COLUMNS = ["id", "ID", "stcpb", "dtcpb"]

# Explicit category vocabulary matching the 66-feature baseline schema
# When drop="first" is applied, drops the first element of each list:
# proto: drops '-' -> 10 features (arp..unas)
# service: drops '-' -> 10 features (dhcp..ssh)
# state: drops 'CON' -> 9 features (ECO..other)
PREDEFINED_CATEGORIES: dict[str, list[str]] = {
    "proto": ["-", "arp", "gre", "ospf", "other", "sctp", "sun-nd", "swipe", "tcp", "udp", "unas"],
    "service": ["-", "dhcp", "dns", "ftp", "ftp-data", "http", "other", "pop3", "smtp", "snmp", "ssh"],
    "state": ["CON", "ECO", "FIN", "INT", "PAR", "REQ", "RST", "URN", "no", "other"],
}


class FocalLoss(nn.Module):
    """Numerically stable binary focal loss using raw logits."""

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=0.0, max=1.0), self.gamma)
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BoostedDNN(nn.Module):
    """4-layer deep neural network with Batch Normalization, LeakyReLU, and Dropout."""

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


class LeakFreePreprocessor:
    """Strict leak-free preprocessor fitted exclusively on training split."""

    def __init__(self, predefined_categories: dict[str, list[str]] | None = None) -> None:
        self.cat_cols = ["proto", "service", "state"]
        self.categories_ = predefined_categories or PREDEFINED_CATEGORIES
        self.encoder_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.numeric_cols_: list[str] = []

    def fit(self, df_tr: pd.DataFrame) -> LeakFreePreprocessor:
        df = df_tr.copy()

        # Identify feature columns
        target_cols = ["label", "attack_cat"] + NOISE_COLUMNS
        feature_cols = [c for c in df.columns if c not in target_cols]

        self.numeric_cols_ = [c for c in feature_cols if c not in self.cat_cols]

        # 1. Transform categoricals on training data to predefined categories + other
        cat_df_tr = pd.DataFrame(index=df.index)
        for col in self.cat_cols:
            cats = set(self.categories_[col])
            cat_df_tr[col] = df[col].astype(str).apply(lambda x: x if x in cats else "other")

        # 2. Fit OneHotEncoder strictly on df_tr using explicit categories
        cat_lists = [self.categories_[c] for c in self.cat_cols]
        self.encoder_ = OneHotEncoder(
            categories=cat_lists,
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
        )
        self.encoder_.fit(cat_df_tr)

        # 3. Fit StandardScaler on numeric features strictly on df_tr (with log1p)
        num_vals = df[self.numeric_cols_].copy()
        for col in SKEWED_FEATURES:
            if col in self.numeric_cols_:
                num_vals[col] = np.log1p(np.maximum(num_vals[col].to_numpy(dtype=np.float32), 0.0))

        self.scaler_ = StandardScaler()
        self.scaler_.fit(num_vals.to_numpy(dtype=np.float32))

        # 4. Build final feature names list (37 numeric + 29 categorical = 66)
        encoded_cat_names = self.encoder_.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names_ = self.numeric_cols_ + encoded_cat_names

        return self

    def transform(self, df_in: pd.DataFrame) -> np.ndarray:
        if self.scaler_ is None or self.encoder_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform is called.")

        df = df_in.copy()

        # 1. Numeric transform
        num_vals = df[self.numeric_cols_].copy()
        for col in SKEWED_FEATURES:
            if col in self.numeric_cols_:
                num_vals[col] = np.log1p(np.maximum(num_vals[col].to_numpy(dtype=np.float32), 0.0))

        num_scaled = self.scaler_.transform(num_vals.to_numpy(dtype=np.float32))

        # 2. Categorical transform
        cat_df = pd.DataFrame(index=df.index)
        for col in self.cat_cols:
            cats = set(self.categories_[col])
            cat_df[col] = df[col].astype(str).apply(lambda x: x if x in cats else "other")

        cat_encoded = self.encoder_.transform(cat_df)

        # 3. Concatenate
        combined = np.hstack([num_scaled, cat_encoded]).astype(np.float32)
        return np.nan_to_num(combined, nan=0.0)


def load_and_split_unsw(
    data_dir: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSVs and partition training set into train/val splits strictly BEFORE preprocessing."""
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path = data_dir / "UNSW_NB15_testing-set.csv"

    print(f"Loading raw UNSW-NB15 files from {data_dir}...", flush=True)
    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Clean noise columns
    for c in NOISE_COLUMNS:
        if c in df_train_full.columns:
            df_train_full = df_train_full.drop(columns=[c])
        if c in df_test.columns:
            df_test = df_test.drop(columns=[c])

    # Stratified split: 85% train, 15% validation
    df_tr, df_val = train_test_split(
        df_train_full,
        test_size=val_ratio,
        random_state=seed,
        stratify=df_train_full["label"],
    )

    df_tr = df_tr.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(f"Stratified partition: Train={len(df_tr):,} | Val={len(df_val):,} | Test={len(df_test):,}", flush=True)
    return df_tr, df_val, df_test


def train_single_model(
    name: str,
    criterion: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    in_features: int,
    device: torch.device,
    epochs: int = 35,
    batch_size: int = 512,
    lr: float = 0.002,
    patience: int = 7,
) -> tuple[BoostedDNN, dict[str, list[float]], float]:
    """Train BoostedDNN with early stopping on validation macro F1."""
    set_seed(42)
    model = BoostedDNN(in_features=in_features, hidden=[256, 128, 64, 32], dropout=0.25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).unsqueeze(1))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0

    t0 = time.perf_counter()
    print(f"\n--- Training {name} ---", flush=True)

    for epoch in range(1, epochs + 1):
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
        val_logits_list = []
        val_targets_list = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss_acc += loss.item() * len(bx)
                n_val += len(bx)

                val_logits_list.append(logits.squeeze(1).cpu().numpy())
                val_targets_list.append(by.squeeze(1).cpu().numpy().astype(int))

        val_loss = val_loss_acc / n_val
        val_logits = np.concatenate(val_logits_list)
        val_targets = np.concatenate(val_targets_list)
        val_probs = expit(val_logits)
        val_preds = (val_probs >= 0.5).astype(int)

        val_acc = float(accuracy_score(val_targets, val_preds))
        val_f1 = float(f1_score(val_targets, val_preds, average="macro"))

        scheduler.step(val_f1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train: {train_loss:.4f} | "
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
                print(f"Early stopping at epoch {epoch} (Best Val Macro F1: {best_val_f1:.4f}).", flush=True)
                break

    train_time = time.perf_counter() - t0
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history, train_time


def analyze_threshold_pareto(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    target_max_fpr: float = 0.18,
    target_min_recall: float = 0.96,
) -> dict[str, Any]:
    """Sweep decision thresholds on validation data and compute Pareto trade-offs."""
    thresholds = np.linspace(0.10, 0.90, 161)
    results = []
    best_constrained_tau = None
    best_constrained_f1 = -1.0
    best_unconstrained_tau = 0.50
    best_unconstrained_f1 = -1.0

    feasible_found = False

    for tau in thresholds:
        preds = (val_probs >= tau).astype(int)
        acc = float(accuracy_score(y_val, preds))
        macro_f1 = float(f1_score(y_val, preds, average="macro"))

        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        is_feasible = (fpr <= target_max_fpr) and (recall >= target_min_recall)
        if is_feasible:
            feasible_found = True
            if macro_f1 > best_constrained_f1:
                best_constrained_f1 = macro_f1
                best_constrained_tau = float(tau)

        if macro_f1 > best_unconstrained_f1:
            best_unconstrained_f1 = macro_f1
            best_unconstrained_tau = float(tau)

        results.append({
            "threshold": float(tau),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "benign_fpr": fpr,
            "attack_recall": recall,
            "feasible": is_feasible,
        })

    return {
        "best_unconstrained_tau": best_unconstrained_tau,
        "best_unconstrained_f1": best_unconstrained_f1,
        "best_constrained_tau": best_constrained_tau,
        "best_constrained_f1": best_constrained_f1,
        "feasible_found": feasible_found,
        "curve": results,
    }


def main() -> None:
    set_seed(42)
    root = get_project_root()
    data_dir = root / "data" / "unsw"
    exp_models_dir = root / "models_unsw" / "experiments" / "exp01"
    exp_artifacts_dir = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp01"

    exp_models_dir.mkdir(parents=True, exist_ok=True)
    exp_artifacts_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}", flush=True)

    # 1. Leak-Free Partitioning
    df_tr, df_val, df_test = load_and_split_unsw(data_dir, val_ratio=0.15, seed=42)

    y_tr = df_tr["label"].to_numpy(dtype=np.float32)
    y_val = df_val["label"].to_numpy(dtype=np.float32)
    y_test = df_test["label"].to_numpy(dtype=np.float32)

    n_pos_tr = int(np.sum(y_tr == 1.0))
    n_neg_tr = int(np.sum(y_tr == 0.0))
    dynamic_pos_weight = float(n_neg_tr / max(n_pos_tr, 1))
    print(f"Train split labels: Normal={n_neg_tr:,}, Attack={n_pos_tr:,} (Attack Ratio: {n_pos_tr/len(y_tr):.2%})")
    print(f"Dynamically computed pos_weight: {dynamic_pos_weight:.4f}")

    # 2. Strict Preprocessing Fit on df_tr ONLY
    print("\nFitting LeakFreePreprocessor strictly on df_tr...", flush=True)
    preprocessor = LeakFreePreprocessor()
    preprocessor.fit(df_tr)

    print(f"Engineered feature dimensionality: {len(preprocessor.feature_names_)} features.")
    assert len(preprocessor.feature_names_) == 66, f"Expected 66 features, got {len(preprocessor.feature_names_)}"

    X_tr = preprocessor.transform(df_tr)
    X_val = preprocessor.transform(df_val)
    # df_test is transformed here only to save the preprocessed artifact, NOT evaluated during selection
    X_test = preprocessor.transform(df_test)

    # Save preprocessing artifacts
    joblib.dump(preprocessor, exp_models_dir / "preprocessor_exp01.joblib")
    with open(exp_models_dir / "feature_names_exp01.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor.feature_names_, f, indent=2)

    # 3. Configure the Three Controlled Experiments
    experiments = [
        {
            "id": "EXP-01A",
            "name": "Standard Unweighted BCE",
            "criterion": nn.BCEWithLogitsLoss(),
            "loss_type": "BCE_unweighted",
        },
        {
            "id": "EXP-01B",
            "name": "Prior-Adjusted BCE",
            "criterion": nn.BCEWithLogitsLoss(pos_weight=torch.tensor([dynamic_pos_weight], device=device)),
            "loss_type": f"BCE_weighted(pos_weight={dynamic_pos_weight:.4f})",
        },
        {
            "id": "EXP-01C",
            "name": "Focal Loss",
            "criterion": FocalLoss(alpha=0.5, gamma=2.0),
            "loss_type": "FocalLoss(alpha=0.5,gamma=2.0)",
        },
    ]

    summary_results = []

    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        criterion = exp["criterion"]

        model, history, train_time = train_single_model(
            name=f"{exp_id}: {exp_name}",
            criterion=criterion,
            X_tr=X_tr,
            y_tr=y_tr,
            X_val=X_val,
            y_val=y_val,
            in_features=66,
            device=device,
            epochs=35,
            batch_size=512,
            lr=0.002,
            patience=7,
        )

        # Validation Predictions
        model.eval()
        val_tensor = torch.from_numpy(X_val).to(device)
        with torch.no_grad():
            val_logits = model(val_tensor).squeeze(1).cpu().numpy()
            val_probs = expit(val_logits)

        val_roc_auc = float(roc_auc_score(y_val, val_probs))

        # Evaluate Pareto Trade-offs
        pareto = analyze_threshold_pareto(
            val_probs=val_probs,
            y_val=y_val,
            target_max_fpr=0.18,
            target_min_recall=0.96,
        )

        tau_star = pareto["best_constrained_tau"] if pareto["feasible_found"] else pareto["best_unconstrained_tau"]
        preds_star = (val_probs >= tau_star).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds_star).ravel()

        val_acc = float(accuracy_score(y_val, preds_star))
        val_f1 = float(f1_score(y_val, preds_star, average="macro"))
        val_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        val_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # Save model checkpoint
        torch.save(model.state_dict(), exp_models_dir / f"model_{exp_id}.pt")

        exp_result = {
            "experiment_id": exp_id,
            "name": exp_name,
            "loss_type": exp["loss_type"],
            "seed": 42,
            "n_features": 66,
            "train_rows": len(X_tr),
            "val_rows": len(X_val),
            "training_time_s": round(train_time, 2),
            "val_roc_auc": round(val_roc_auc, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_fpr": round(val_fpr, 4),
            "val_recall": round(val_recall, 4),
            "calibrated_threshold": round(tau_star, 4),
            "feasible_under_constraints": pareto["feasible_found"],
            "pareto_summary": {
                "unconstrained_tau": pareto["best_unconstrained_tau"],
                "unconstrained_f1": pareto["best_unconstrained_f1"],
                "constrained_tau": pareto["best_constrained_tau"],
                "constrained_f1": pareto["best_constrained_f1"],
            },
        }

        summary_results.append(exp_result)

        # Save individual experiment details
        with open(exp_artifacts_dir / f"{exp_id}_pareto.json", "w", encoding="utf-8") as f:
            json.dump({"summary": exp_result, "curve": pareto["curve"]}, f, indent=2)

    # 4. Print Comparison Table
    print("\n" + "=" * 90)
    print("EXPERIMENT 1 (EXP-01): VALIDATION RESULTS COMPARISON")
    print("=" * 90)
    print(f"{'Exp ID':<8} | {'Loss Type':<22} | {'Val Acc':<9} | {'Val F1':<8} | {'Val AUC':<8} | {'Val FPR':<8} | {'Val Rec':<8} | {'Tau*':<6} | {'Status'}")
    print("-" * 90)
    for r in summary_results:
        status = "Feasible" if r["feasible_under_constraints"] else "Trade-off"
        print(
            f"{r['experiment_id']:<8} | {r['name']:<22} | {r['val_accuracy']*100:6.2f}%  | "
            f"{r['val_macro_f1']:<8.4f} | {r['val_roc_auc']:<8.4f} | {r['val_fpr']*100:6.2f}%  | "
            f"{r['val_recall']*100:6.2f}%  | {r['calibrated_threshold']:<6.4f} | {status}"
        )
    print("=" * 90)

    # Save complete summary
    with open(exp_artifacts_dir / "exp01_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nAll artifacts saved to: {exp_artifacts_dir}")


if __name__ == "__main__":
    main()

