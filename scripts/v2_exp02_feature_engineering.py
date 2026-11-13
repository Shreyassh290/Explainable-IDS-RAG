"""Experiment 2 (EXP-02): Domain Feature Engineering for Fuzzers & Benign Discrimination.

Builds upon EXP-01A (Standard Unweighted BCE) by adding 6 engineered domain features:
  1. bytes_per_pkt_src = sbytes / (spkts + 1)
  2. bytes_per_pkt_dst = dbytes / (dpkts + 1)
  3. packet_asymmetry = (spkts - dpkts) / (spkts + dpkts + 1)
  4. byte_asymmetry = (sbytes - dbytes) / (sbytes + dbytes + 1)
  5. ttl_ratio = sttl / (dttl + 1)
  6. load_ratio = sload / (dload + 1)

Dimensionality:
  37 baseline numeric + 6 engineered numeric = 43 numeric features (with log1p on heavy tails).
  29 one-hot categorical dummy features (10 proto, 10 service, 9 state).
  Total dimensionality: Exactly 72 engineered features.

Strict Leak-Free Protocol:
- Partitioning: 85% train (149,039) / 15% val (26,302) BEFORE preprocessing.
- StandardScaler and OneHotEncoder fitted EXCLUSIVELY on df_tr.
- Model selection and threshold calibration strictly on validation predictions.
- Official test split held out until validation selection is reviewed.
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root
from scripts.v2_exp01_loss_and_leakage import (
    NOISE_COLUMNS,
    PREDEFINED_CATEGORIES,
    SKEWED_FEATURES,
    analyze_threshold_pareto,
    load_and_split_unsw,
    set_seed,
)

# New engineered features skewed and benefiting from log1p scaling
ENGINEERED_SKEWED_FEATURES = [
    "bytes_per_pkt_src",
    "bytes_per_pkt_dst",
    "ttl_ratio",
    "load_ratio",
]


def compute_domain_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Compute 6 domain ratio and asymmetry features with strict numerical safety."""
    df = df_in.copy()

    spkts = np.maximum(df["spkts"].to_numpy(dtype=np.float32), 0.0)
    dpkts = np.maximum(df["dpkts"].to_numpy(dtype=np.float32), 0.0)
    sbytes = np.maximum(df["sbytes"].to_numpy(dtype=np.float32), 0.0)
    dbytes = np.maximum(df["dbytes"].to_numpy(dtype=np.float32), 0.0)
    sttl = np.maximum(df["sttl"].to_numpy(dtype=np.float32), 0.0)
    dttl = np.maximum(df["dttl"].to_numpy(dtype=np.float32), 0.0)
    sload = np.maximum(df["sload"].to_numpy(dtype=np.float32), 0.0)
    dload = np.maximum(df["dload"].to_numpy(dtype=np.float32), 0.0)

    # 1. Bytes per packet ratios
    df["bytes_per_pkt_src"] = sbytes / (spkts + 1.0)
    df["bytes_per_pkt_dst"] = dbytes / (dpkts + 1.0)

    # 2. Normalized packet & byte asymmetry [-1.0, 1.0]
    df["packet_asymmetry"] = (spkts - dpkts) / (spkts + dpkts + 1.0)
    df["byte_asymmetry"] = (sbytes - dbytes) / (sbytes + dbytes + 1.0)

    # 3. Protocol & load ratios
    df["ttl_ratio"] = sttl / (dttl + 1.0)
    df["load_ratio"] = sload / (dload + 1.0)

    return df


class LeakFreePreprocessorV2:
    """Strict leak-free preprocessor for 72-feature representation fitted exclusively on df_tr."""

    def __init__(self, predefined_categories: dict[str, list[str]] | None = None) -> None:
        self.cat_cols = ["proto", "service", "state"]
        self.categories_ = predefined_categories or PREDEFINED_CATEGORIES
        self.encoder_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.numeric_cols_: list[str] = []

    def fit(self, df_tr: pd.DataFrame) -> LeakFreePreprocessorV2:
        df = compute_domain_features(df_tr)

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
        all_skewed = SKEWED_FEATURES + ENGINEERED_SKEWED_FEATURES
        for col in all_skewed:
            if col in self.numeric_cols_:
                num_vals[col] = np.log1p(np.maximum(num_vals[col].to_numpy(dtype=np.float32), 0.0))

        self.scaler_ = StandardScaler()
        self.scaler_.fit(num_vals.to_numpy(dtype=np.float32))

        # 4. Build final feature names list (43 numeric + 29 categorical = 72)
        encoded_cat_names = self.encoder_.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names_ = self.numeric_cols_ + encoded_cat_names

        return self

    def transform(self, df_in: pd.DataFrame) -> np.ndarray:
        if self.scaler_ is None or self.encoder_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform is called.")

        df = compute_domain_features(df_in)

        # 1. Numeric transform
        num_vals = df[self.numeric_cols_].copy()
        all_skewed = SKEWED_FEATURES + ENGINEERED_SKEWED_FEATURES
        for col in all_skewed:
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


class BoostedDNNV2(nn.Module):
    """Deep Neural Network configured for 72 engineered features."""

    def __init__(
        self,
        in_features: int = 72,
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


def train_exp02_model(
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
) -> tuple[BoostedDNNV2, dict[str, list[float]], float]:
    """Train BoostedDNNV2 on 72 features using Standard Unweighted BCE."""
    set_seed(42)
    model = BoostedDNNV2(in_features=in_features, hidden=[256, 128, 64, 32], dropout=0.25).to(device)
    criterion = nn.BCEWithLogitsLoss()
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
    print("\n--- Training EXP-02 (72 Features, Unweighted BCE) ---", flush=True)

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


def main() -> None:
    set_seed(42)
    root = get_project_root()
    data_dir = root / "data" / "unsw"
    exp02_models_dir = root / "models_unsw" / "experiments" / "exp02"
    exp02_artifacts_dir = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp02"

    exp02_models_dir.mkdir(parents=True, exist_ok=True)
    exp02_artifacts_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}", flush=True)

    # 1. Leak-Free Partitioning (Identical seed=42 split to EXP-01)
    df_tr, df_val, df_test = load_and_split_unsw(data_dir, val_ratio=0.15, seed=42)

    y_tr = df_tr["label"].to_numpy(dtype=np.float32)
    y_val = df_val["label"].to_numpy(dtype=np.float32)
    val_attack_cats = df_val["attack_cat"].fillna("Normal").astype(str).to_numpy()

    # 2. Strict Preprocessing Fit on df_tr ONLY
    print("\nFitting LeakFreePreprocessorV2 on df_tr with 6 engineered domain features...", flush=True)
    preprocessor = LeakFreePreprocessorV2()
    preprocessor.fit(df_tr)

    print(f"Engineered feature dimensionality: {len(preprocessor.feature_names_)} features.")
    assert len(preprocessor.feature_names_) == 72, f"Expected 72 features, got {len(preprocessor.feature_names_)}"

    X_tr = preprocessor.transform(df_tr)
    X_val = preprocessor.transform(df_val)

    # Save preprocessing artifacts
    joblib.dump(preprocessor, exp02_models_dir / "preprocessor_exp02.joblib")
    with open(exp02_models_dir / "feature_names_exp02.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor.feature_names_, f, indent=2)

    # 3. Train EXP-02 Model
    model, history, train_time = train_exp02_model(
        X_tr=X_tr,
        y_tr=y_tr,
        X_val=X_val,
        y_val=y_val,
        in_features=72,
        device=device,
        epochs=35,
        batch_size=512,
        lr=0.002,
        patience=7,
    )

    # 4. Validation Predictions & Multi-Metric Pareto Search
    model.eval()
    val_tensor = torch.from_numpy(X_val).to(device)
    with torch.no_grad():
        val_logits = model(val_tensor).squeeze(1).cpu().numpy()
        val_probs = expit(val_logits)

    val_roc_auc = float(roc_auc_score(y_val, val_probs))

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

    # Validation Fuzzers Recall
    df_val_eval = pd.DataFrame({"attack_cat": val_attack_cats, "y_true": y_val, "y_pred": preds_star})
    fuzzers_val = df_val_eval[df_val_eval["attack_cat"] == "Fuzzers"]
    fuzz_recall_val = float(accuracy_score(fuzzers_val["y_true"], fuzzers_val["y_pred"]))

    # Save model checkpoint
    model_save_path = exp02_models_dir / "model_EXP-02.pt"
    torch.save(model.state_dict(), model_save_path)

    # 5. Load EXP-01A validation summary for comparison
    exp01_summary_path = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp01" / "exp01_comparison_summary.json"
    exp01a_val = {}
    if exp01_summary_path.exists():
        with open(exp01_summary_path, encoding="utf-8") as f:
            all_exp01 = json.load(f)
            exp01a_val = next((item for item in all_exp01 if item["experiment_id"] == "EXP-01A"), {})

    exp01_acc_str = f"{exp01a_val.get('val_accuracy', 0.0) * 100:.2f}%"
    exp01_f1_str = f"{exp01a_val.get('val_macro_f1', 0.0):.4f}"
    exp01_auc_str = f"{exp01a_val.get('val_roc_auc', 0.0):.4f}"
    exp01_fpr_str = f"{exp01a_val.get('val_fpr', 0.0) * 100:.2f}%"
    exp01_rec_str = f"{exp01a_val.get('val_recall', 0.0) * 100:.2f}%"
    exp01_tau_str = f"{exp01a_val.get('calibrated_threshold', 0.4650):.4f}"

    exp02_acc_str = f"{val_acc * 100:.2f}%"
    exp02_f1_str = f"{val_f1:.4f}"
    exp02_auc_str = f"{val_roc_auc:.4f}"
    exp02_fpr_str = f"{val_fpr * 100:.2f}%"
    exp02_rec_str = f"{val_recall * 100:.2f}%"
    exp02_fuzz_str = f"{fuzz_recall_val * 100:.2f}%"
    exp02_tau_str = f"{tau_star:.4f}"

    print("\n" + "=" * 85)
    print("VALIDATION PERFORMANCE COMPARISON: EXP-01A (66 Feats) vs EXP-02 (72 Feats)")
    print("=" * 85)
    print(f"{'Metric':<28} | {'EXP-01A (66 Features)':<22} | {'EXP-02 (72 Features)':<22}")
    print("-" * 85)
    print(f"{'Engineered Features':<28} | {'66':<22} | {'72':<22}")
    print(f"{'Validation Accuracy':<28} | {exp01_acc_str:<22} | {exp02_acc_str:<22}")
    print(f"{'Validation Macro F1':<28} | {exp01_f1_str:<22} | {exp02_f1_str:<22}")
    print(f"{'Validation ROC-AUC':<28} | {exp01_auc_str:<22} | {exp02_auc_str:<22}")
    print(f"{'Validation Benign FPR':<28} | {exp01_fpr_str:<22} | {exp02_fpr_str:<22}")
    print(f"{'Validation Attack Recall':<28} | {exp01_rec_str:<22} | {exp02_rec_str:<22}")
    print(f"{'Validation Fuzzers Recall':<28} | {'~90% (Val)':<22} | {exp02_fuzz_str:<22}")
    print(f"{'Calibrated Threshold':<28} | {exp01_tau_str:<22} | {exp02_tau_str:<22}")
    print("=" * 85)

    exp02_summary = {
        "experiment_id": "EXP-02",
        "description": "Domain Feature Engineering (72 features) with Standard Unweighted BCE",
        "seed": 42,
        "n_features": 72,
        "train_rows": len(X_tr),
        "val_rows": len(X_val),
        "training_time_s": round(train_time, 2),
        "val_roc_auc": round(val_roc_auc, 4),
        "val_accuracy": round(val_acc, 4),
        "val_macro_f1": round(val_f1, 4),
        "val_fpr": round(val_fpr, 4),
        "val_recall": round(val_recall, 4),
        "val_fuzzers_recall": round(fuzz_recall_val, 4),
        "calibrated_threshold": round(tau_star, 4),
        "feasible_under_constraints": pareto["feasible_found"],
        "comparison_vs_exp01a": {
            "val_acc_delta": round(val_acc - exp01a_val.get("val_accuracy", 0.9507), 4),
            "val_f1_delta": round(val_f1 - exp01a_val.get("val_macro_f1", 0.9424), 4),
            "val_fpr_delta": round(val_fpr - exp01a_val.get("val_fpr", 0.1038), 4),
        },
    }

    out_summary_path = exp02_artifacts_dir / "exp02_val_summary.json"
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(exp02_summary, f, indent=2)

    with open(exp02_artifacts_dir / "exp02_pareto.json", "w", encoding="utf-8") as f:
        json.dump({"summary": exp02_summary, "curve": pareto["curve"]}, f, indent=2)

    print(f"\nEXP-02 validation results saved to: {out_summary_path}")


if __name__ == "__main__":
    main()

