"""Mixed 6-class neural ensemble evaluation check.

Samples up to 3,000 flows per class across both baseline and extra test sets:
- data/processed/test.parquet (Benign, DoS, DDoS)
- data/v2/processed/test_extra.parquet (Bot, FTP-BruteForce, SSH-Bruteforce)

Evaluates the unified 6-class neural ensemble (argmax over calibrated/uncalibrated probabilities).
Computes:
- Full 6x6 confusion matrix.
- Per-class recall.
- Leakage / misclassification from original baseline classes into extra classes.

Prominently highlights data limitations:
FTP-BruteForce has limited test flows (N=8) and is indicative only.

Results are written to artifacts/v2/mixed_check/mixed_confusion_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.models.ensemble import BinaryDNNEnsemble
from ids_rag.v2.pipeline_v2 import PipelineV2Context
from ids_rag.v2.prepare_extra import load_v2_config

CLASSES = ["Benign", "DoS", "DDoS", "Bot", "FTP-BruteForce", "SSH-Bruteforce"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


def build_six_class_ensemble(ctx: PipelineV2Context) -> BinaryDNNEnsemble:
    """Build unified 6-class ensemble combining baseline and extra binary DNNs."""
    if len(ctx.ensemble.models) >= 6:
        return ctx.ensemble

    models_dir = repo_root / "models"
    base_ensemble = BinaryDNNEnsemble.load(models_dir=models_dir, use_temperature=False)

    combined_models: dict[int, Any] = {}
    combined_label_map: dict[int, str] = {}

    for cid, m in base_ensemble.models.items():
        combined_models[cid] = m
        combined_label_map[cid] = base_ensemble.label_map[cid]

    for cid, m in ctx.ensemble.models.items():
        combined_models[cid] = m
        combined_label_map[cid] = ctx.ensemble.label_map[cid]

    return BinaryDNNEnsemble(
        models=combined_models,
        label_map=combined_label_map,
        feature_names=ctx.feature_names,
        tiers_config=ctx.ensemble.tiers_config,
        temperatures={name: 1.0 for name in combined_label_map.values()},
        use_temperature=False,
        device=ctx.ensemble.device,
    )


def main() -> None:
    print("=" * 78)
    print("V2 MIXED 6-CLASS NEURAL ENSEMBLE EVALUATION CHECK")
    print("Target Classes: Benign, DoS, DDoS, Bot, FTP-BruteForce, SSH-Bruteforce")
    print("NOTE: FTP-BruteForce has limited test flows (N=8) and is indicative only.")
    print("=" * 78)

    cfg = load_v2_config()
    ctx = PipelineV2Context.load(cfg)
    ensemble = build_six_class_ensemble(ctx)

    base_test_path = repo_root / "data" / "processed" / "test.parquet"
    extra_test_path = repo_root / "data" / "v2" / "processed" / "test_extra.parquet"

    if not base_test_path.exists():
        raise FileNotFoundError(f"Baseline test parquet not found at {base_test_path}")
    if not extra_test_path.exists():
        raise FileNotFoundError(f"Extra classes test parquet not found at {extra_test_path}")

    print(f"\nLoading baseline test set: {base_test_path}")
    df_base = pd.read_parquet(base_test_path)
    print(f"Loading extra test set:    {extra_test_path}")
    df_extra = pd.read_parquet(extra_test_path)

    # Sample up to 3,000 rows per class with seed 42
    sampled_dfs: list[pd.DataFrame] = []
    sample_counts: dict[str, int] = {}

    # Baseline classes (0: Benign, 1: DoS, 2: DDoS)
    for cid in [0, 1, 2]:
        cname = ID_TO_CLASS[cid]
        subset = df_base[df_base["label"] == cid]
        n_take = min(3000, len(subset))
        sampled = subset.sample(n=n_take, random_state=42)
        sampled_dfs.append(sampled)
        sample_counts[cname] = len(sampled)

    # Extra classes (3: Bot, 4: FTP-BruteForce, 5: SSH-Bruteforce)
    for cid in [3, 4, 5]:
        cname = ID_TO_CLASS[cid]
        subset = df_extra[df_extra["label"] == cid]
        n_take = min(3000, len(subset))
        sampled = subset.sample(n=n_take, random_state=42)
        sampled_dfs.append(sampled)
        sample_counts[cname] = len(sampled)

    df_eval = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\nSampled {len(df_eval):,} total flows across 6 classes:")
    for cname, cnt in sample_counts.items():
        flag = " [INDICATIVE ONLY - N=8]" if cname == "FTP-BruteForce" else ""
        print(f"  - {cname:<16}: {cnt:>5} flows{flag}")

    # Prepare feature tensor and true labels
    feature_names = ctx.feature_names
    x_features = df_eval[feature_names].to_numpy(dtype=np.float32)
    y_true = df_eval["label"].to_numpy(dtype=np.int64)

    # Run batched ensemble prediction
    print("\nRunning inference across all flows with 6-class ensemble...")
    pred_labels, confidences, tiers = ensemble.predict(x_features)
    y_pred = pred_labels.astype(np.int64)

    # Compute 6x6 confusion matrix
    cm = np.zeros((6, 6), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Format text table for console
    col_width = 15
    col_headers = [f"{c[:13]:^{col_width}}" for c in CLASSES]
    header_label = "True/Pred"
    table_header = f"{header_label:<16} | " + " | ".join(col_headers) + " | Total"
    sep = "-" * len(table_header)

    print("\n" + sep)
    print("6x6 CONFUSION MATRIX (Sampled Test Flows)")
    print(sep)
    print(table_header)
    print(sep)

    per_class_metrics: dict[str, Any] = {}
    for r_idx, true_name in enumerate(CLASSES):
        row_vals = cm[r_idx]
        total_row = row_vals.sum()
        row_str = " | ".join(f"{val:^{col_width},}" for val in row_vals)
        print(f"{true_name:<16} | {row_str} | {total_row:>5,}")

        rec = float(row_vals[r_idx] / total_row) if total_row > 0 else 0.0
        per_class_metrics[true_name] = {
            "true_count": int(total_row),
            "correct": int(row_vals[r_idx]),
            "recall": round(rec, 6),
            "is_indicative_only": bool(true_name == "FTP-BruteForce"),
        }

    print(sep)

    # Original to Extra leakage analysis
    # Rows 0, 1, 2 (Benign, DoS, DDoS) predicted as columns 3, 4, 5 (Bot, FTP, SSH)
    leakage_per_class = {}
    total_baseline_flows = 0
    total_leaked = 0

    for orig_idx, orig_name in enumerate(["Benign", "DoS", "DDoS"]):
        leaked_counts = {CLASSES[col_idx]: int(cm[orig_idx, col_idx]) for col_idx in [3, 4, 5]}
        sum_leaked = sum(leaked_counts.values())
        tot_class = int(cm[orig_idx].sum())
        leak_pct = (sum_leaked / tot_class * 100) if tot_class > 0 else 0.0

        leakage_per_class[orig_name] = {
            "total_tested": tot_class,
            "misclassified_as_extra": sum_leaked,
            "leakage_percent": round(leak_pct, 4),
            "breakdown": leaked_counts,
        }
        total_baseline_flows += tot_class
        total_leaked += sum_leaked

    print("\n================== PER-CLASS RECALL ==================")
    for cname, m in per_class_metrics.items():
        flag = " [INDICATIVE ONLY - N=8]" if m["is_indicative_only"] else ""
        print(f"{cname:<16}: Recall = {m['recall']*100:6.2f}% ({m['correct']}/{m['true_count']}){flag}")

    print("\n========= MISCLASSIFICATION INTO EXTRA CLASSES =========")
    print("Baseline flows (Benign, DoS, DDoS) misclassified into Bot, FTP-BruteForce, or SSH-Bruteforce:")
    for orig_name, info in leakage_per_class.items():
        print(
            f"  - {orig_name:<8}: {info['misclassified_as_extra']:>4}/{info['total_tested']} "
            f"({info['leakage_percent']:.2f}%) -> {info['breakdown']}"
        )
    print(f"Overall baseline-to-extra misclassification: {total_leaked}/{total_baseline_flows} ({total_leaked/total_baseline_flows*100:.2f}%)")
    print("========================================================\n")

    # Save artifact
    out_dir = repo_root / "artifacts" / "v2" / "mixed_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "mixed_confusion_summary.json"

    summary_payload = {
        "classes": CLASSES,
        "sample_counts": sample_counts,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "baseline_leakage_to_extra": {
            "total_baseline_tested": total_baseline_flows,
            "total_misclassified_into_extra": total_leaked,
            "leakage_rate": round(total_leaked / total_baseline_flows, 6) if total_baseline_flows > 0 else 0.0,
            "per_class": leakage_per_class,
        },
        "data_notice": "FTP-BruteForce has limited test flows (N=8) and is indicative only.",
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"Saved evaluation results to: {summary_path}")


if __name__ == "__main__":
    main()

