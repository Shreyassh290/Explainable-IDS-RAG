"""Evaluation script for Stage 2 v2 extra-class binary DNNs.

Loads the trained v2 models from models_v2/, runs inference on the held-out
test_extra.parquet, and writes per-class and overall metrics to
artifacts/v2/eval/.

Frozen directories are never read as output targets. The script is read-only
with respect to models/, data/processed/, and all original artifact paths.

Usage:
    uv run python scripts/v2_eval_extra.py

Output:
    artifacts/v2/eval/eval_summary_v2.json
    artifacts/v2/eval/eval_report_v2.txt
    artifacts/v2/eval/confusion_matrix_v2_<class_name>.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import torch

from ids_rag.config import get_project_root
from ids_rag.models.dnn import BinaryDNN, compute_binary_macro_f1
from ids_rag.v2.prepare_extra import load_v2_config
from ids_rag.v2.train_extra import INDICATIVE_ONLY_CLASSES, INDICATIVE_THRESHOLD


def load_v2_models(
    models_dir: Path,
    label_map: dict[int, str],
    feature_names: list[str],
    config: dict,
    device: torch.device,
) -> dict[int, BinaryDNN]:
    """Load all trained v2 binary DNN weights into eval mode."""
    model_cfg = config["model"]
    hidden = model_cfg.get("hidden", [128, 64, 32])
    dropout = model_cfg.get("dropout", 0.3)
    use_batchnorm = model_cfg.get("use_batchnorm", False)

    models: dict[int, BinaryDNN] = {}
    for class_id, class_name in label_map.items():
        weights_path = models_dir / f"dnn_v2_{class_name}.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Trained weights not found at {weights_path}. "
                f"Run `python -m ids_rag.v2.train_extra --confirm-train` first."
            )
        model = BinaryDNN(
            in_features=len(feature_names),
            hidden=hidden,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models[class_id] = model
        print(f"Loaded model for '{class_name}' from {weights_path}", flush=True)

    return models


def evaluate_binary_model(
    class_id: int,
    class_name: str,
    model: BinaryDNN,
    x_test: torch.Tensor,
    y_test_raw: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> dict:
    """Run inference and compute metrics for one binary v2 model on the test set."""
    y_binary = (y_test_raw == class_id).astype(np.float32)
    model.eval()

    probs_list: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            batch = x_test[start : start + batch_size].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)

    probs_all = np.vstack(probs_list).ravel()
    preds_all = (probs_all >= 0.5).astype(np.int32)

    n_pos = int(np.sum(y_binary == 1.0))
    n_neg = int(np.sum(y_binary == 0.0))
    is_indicative = class_name in INDICATIVE_ONLY_CLASSES or n_pos < INDICATIVE_THRESHOLD

    # ROC-AUC requires both classes present
    if n_pos >= 1 and n_pos < len(y_binary):
        auc = float(roc_auc_score(y_binary, probs_all))
    else:
        auc = float("nan")

    f1_pos = float(f1_score(y_binary, preds_all, pos_label=1, zero_division=0))
    f1_neg = float(f1_score(y_binary, preds_all, pos_label=0, zero_division=0))
    macro_f1 = compute_binary_macro_f1(y_binary, preds_all)
    acc = float(np.mean(preds_all == y_binary.astype(np.int32)))

    cm = confusion_matrix(y_binary.astype(np.int32), preds_all)

    report = {
        "class_id": class_id,
        "class_name": class_name,
        "n_pos_test": n_pos,
        "n_neg_test": n_neg,
        "is_indicative_only": is_indicative,
        "accuracy": round(acc, 6),
        "val_roc_auc": round(auc, 6) if not np.isnan(auc) else None,
        "f1_positive_class": round(f1_pos, 6),
        "f1_negative_class": round(f1_neg, 6),
        "macro_f1": round(macro_f1, 6),
        "confusion_matrix": cm.tolist(),
    }

    indicative_note = " [INDICATIVE ONLY]" if is_indicative else ""
    print(
        f"  {class_name}: Acc={acc:.4f} | Macro F1={macro_f1:.4f} | "
        f"ROC-AUC={auc:.4f} | Pos F1={f1_pos:.4f} | "
        f"Pos={n_pos} Neg={n_neg}{indicative_note}",
        flush=True,
    )

    return report


def plot_confusion_matrix(
    cm: list[list[int]],
    class_name: str,
    output_path: Path,
    is_indicative: bool,
) -> None:
    """Save a binary confusion-matrix plot."""
    cm_arr = np.array(cm)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_arr,
        display_labels=[f"Not {class_name}", class_name],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    note = " [INDICATIVE ONLY]" if is_indicative else ""
    ax.set_title(f"Confusion Matrix - v2 {class_name} vs Rest{note}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_v2_eval(config_path: Path | str | None = None) -> None:
    """Main evaluation entry point."""
    root = get_project_root()
    cfg = load_v2_config(config_path)

    paths = cfg["paths"]
    processed_dir = root / paths.get("processed", "data/v2/processed")
    models_dir = root / "models_v2"
    eval_dir = root / "artifacts" / "v2" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load label map from models_v2/ (produced by train_extra.py)
    label_map_path = models_dir / "label_map_v2.json"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"Label map not found at {label_map_path}. "
            f"Run training first."
        )
    with open(label_map_path, "r", encoding="utf-8") as f:
        raw_map: dict[str, str] = json.load(f)
    label_map: dict[int, str] = {int(k): v for k, v in raw_map.items()}

    # Load feature names
    feature_names_path = models_dir / "feature_names_v2.json"
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names: list[str] = json.load(f)

    # Load test parquet (held-out, never used during training)
    test_path = processed_dir / "test_extra.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"test_extra.parquet not found at {test_path}.")

    print(f"Loading test set from {test_path} ...", flush=True)
    df_test = pd.read_parquet(test_path)
    x_test_np = df_test[feature_names].to_numpy(dtype=np.float32, copy=True)
    y_test_np = df_test["label"].to_numpy(dtype=np.int64, copy=True)
    x_test_t = torch.from_numpy(x_test_np)

    print(f"Test rows: {len(df_test):,}", flush=True)
    for cid, cname in sorted(label_map.items()):
        n = int(np.sum(y_test_np == cid))
        print(f"  Class '{cname}' (ID {cid}): {n:,} test rows", flush=True)

    # Load trained models
    models = load_v2_models(models_dir, label_map, feature_names, cfg, device)

    print("\n=================================================================", flush=True)
    print("V2 EXTRA CLASSES EVALUATION (TEST SET)", flush=True)
    print("=================================================================\n", flush=True)

    per_class_results: list[dict] = []
    for class_id in sorted(label_map.keys()):
        class_name = label_map[class_id]
        result = evaluate_binary_model(
            class_id=class_id,
            class_name=class_name,
            model=models[class_id],
            x_test=x_test_t,
            y_test_raw=y_test_np,
            device=device,
        )
        per_class_results.append(result)

        # Confusion matrix plot
        cm_path = eval_dir / f"confusion_matrix_v2_{class_name}.png"
        plot_confusion_matrix(
            cm=result["confusion_matrix"],
            class_name=class_name,
            output_path=cm_path,
            is_indicative=result["is_indicative_only"],
        )

    # Aggregate summary (excluding indicative-only classes from macro averages)
    reliable = [r for r in per_class_results if not r["is_indicative_only"]]
    indicative = [r for r in per_class_results if r["is_indicative_only"]]

    reliable_macro_f1 = float(np.mean([r["macro_f1"] for r in reliable])) if reliable else float("nan")
    reliable_auc = [r["val_roc_auc"] for r in reliable if r["val_roc_auc"] is not None]
    reliable_mean_auc = float(np.mean(reliable_auc)) if reliable_auc else float("nan")

    summary = {
        "test_rows_total": len(df_test),
        "device": str(device),
        "reliable_classes": [r["class_name"] for r in reliable],
        "indicative_only_classes": [r["class_name"] for r in indicative],
        "reliable_mean_macro_f1": round(reliable_macro_f1, 6),
        "reliable_mean_roc_auc": round(reliable_mean_auc, 6),
        "per_class": {r["class_name"]: r for r in per_class_results},
    }

    summary_path = eval_dir / "eval_summary_v2.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Human-readable text report
    lines = [
        "V2 EXTRA CLASSES EVALUATION REPORT",
        "=" * 60,
        f"Test rows: {len(df_test):,}",
        f"Device: {device}",
        "",
        "Per-Class Binary Results (one-vs-rest, threshold 0.5):",
        "-" * 60,
    ]
    for r in per_class_results:
        note = " [INDICATIVE ONLY]" if r["is_indicative_only"] else ""
        auc_str = f"{r['val_roc_auc']:.4f}" if r["val_roc_auc"] is not None else "N/A"
        lines += [
            f"  {r['class_name']}{note}",
            f"    Test positives:   {r['n_pos_test']}",
            f"    Accuracy:         {r['accuracy']:.4f}",
            f"    Macro F1:         {r['macro_f1']:.4f}",
            f"    Positive F1:      {r['f1_positive_class']:.4f}",
            f"    ROC-AUC:          {auc_str}",
            f"    Confusion matrix: {r['confusion_matrix']}",
            "",
        ]

    lines += [
        "-" * 60,
        "Aggregated over reliable classes only (FTP-BruteForce excluded):",
        f"  Mean Macro F1:  {reliable_macro_f1:.4f}",
        f"  Mean ROC-AUC:   {reliable_mean_auc:.4f}",
        "",
        "NOTE: FTP-BruteForce results are indicative only (53 total samples",
        "in CICIDS2018 after deduplication). Do not compare to Bot or SSH-Bruteforce.",
    ]

    report_text = "\n".join(lines)
    report_path = eval_dir / "eval_report_v2.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n{report_text}", flush=True)
    print(f"\nEvaluation artifacts saved to {eval_dir}", flush=True)


if __name__ == "__main__":
    run_v2_eval()

