"""Evaluate negfix v2 models (models_v2b/) against acceptance criteria.

Runs four evaluation checks:

D1. Isolated test on extra classes only (3-class argmax over extra classes).
    Test set: data/v2/processed/test_extra.parquet.
    Models:   models_v2b/dnn_v2b_<ClassName>.pt.

D2. Mixed 6-class confusion check (replicate v2_mixed_confusion.py logic with
    negfix models). Samples up to 3,000 flows per class from baseline test +
    extra test, runs 6-class argmax, computes 6x6 confusion matrix, per-class
    recall, and baseline-to-extra leakage.

D3. Full baseline test through 6-class argmax. Loads all flows from
    data/processed/test.parquet, runs all 6 models (3 frozen baseline +
    3 new negfix), computes accuracy and per-class F1 for Benign/DoS/DDoS.

D4. Demo samples check. Loads app/raw_samples_v2.csv and runs 6-class ensemble
    prediction with negfix models for each sample. Reports match/mismatch.

Acceptance criteria (all must hold before wiring into the app):
  - Baseline 6-class test accuracy >= 0.995 AND no baseline class F1 < 0.995
  - Baseline leakage (baseline flows predicted as extra class) < 0.5% in D2
  - Bot recall >= 0.95 and SSH-Bruteforce recall >= 0.95 in D2
  - FTP-BruteForce reported separately (indicative only)

Output:
  artifacts/v2b/eval/eval_negfix_summary.json  -- machine-readable results
  artifacts/v2b/eval/eval_negfix_report.txt    -- human-readable report
  artifacts/v2b/eval/mixed_confusion_negfix.json
  Console: PASS / FAIL per criterion

Usage:
    uv run python scripts/v2_eval_negfix.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ids_rag.config import get_project_root
from ids_rag.models.dnn import BinaryDNN
from ids_rag.models.ensemble import BinaryDNNEnsemble
from ids_rag.pipeline import validate_raw_features
from ids_rag.v2.prepare_extra import load_v2_config

# Negfix output locations
MODELS_V2B_DIR = "models_v2b"
EVAL_DIR_REL = "artifacts/v2b/eval"

# Classes
CLASSES_ALL = ["Benign", "DoS", "DDoS", "Bot", "FTP-BruteForce", "SSH-Bruteforce"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES_ALL)}
ID_TO_CLASS = {i: c for i, c in enumerate(CLASSES_ALL)}

# Acceptance thresholds
BASELINE_ACC_THRESHOLD = 0.995
BASELINE_F1_THRESHOLD = 0.995
LEAKAGE_THRESHOLD = 0.005        # < 0.5%
BOT_RECALL_THRESHOLD = 0.95
SSH_RECALL_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_negfix_models(
    models_dir: Path,
    label_map: dict[int, str],
    feature_names: list[str],
    config: dict,
    device: torch.device,
) -> dict[int, BinaryDNN]:
    """Load all trained negfix binary DNN weights into eval mode."""
    model_cfg = config.get("model", {})
    hidden: list[int] = model_cfg.get("hidden", [128, 64, 32])
    dropout: float = model_cfg.get("dropout", 0.3)
    use_batchnorm: bool = model_cfg.get("use_batchnorm", False)

    models: dict[int, BinaryDNN] = {}
    for class_id, class_name in label_map.items():
        weights_path = models_dir / f"dnn_v2b_{class_name}.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Negfix weights not found at {weights_path}. "
                f"Run v2_train_negfix.py --confirm-train first."
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
        print(f"Loaded negfix model for '{class_name}' from {weights_path}", flush=True)

    return models


def load_frozen_baseline_models(
    models_dir: Path,
    feature_names: list[str],
    config: dict,
    device: torch.device,
) -> dict[int, BinaryDNN]:
    """Load frozen baseline models (Benign=0, DoS=1, DDoS=2) for 6-class argmax."""
    baseline_label_map = {0: "Benign", 1: "DoS", 2: "DDoS"}
    model_cfg = config.get("model", {})
    hidden: list[int] = model_cfg.get("hidden", [128, 64, 32])
    dropout: float = model_cfg.get("dropout", 0.3)
    use_batchnorm: bool = model_cfg.get("use_batchnorm", False)

    models: dict[int, BinaryDNN] = {}
    for class_id, class_name in baseline_label_map.items():
        weights_path = models_dir / f"dnn_{class_name}.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Frozen baseline weights not found at {weights_path}. "
                f"Expected dnn_<ClassName>.pt files in models/."
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
        print(f"Loaded frozen baseline model for '{class_name}' from {weights_path}", flush=True)

    return models


def run_six_class_argmax(
    all_models: dict[int, BinaryDNN],
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run all 6 models on x, return argmax class predictions.

    Args:
        all_models: dict mapping class_id -> BinaryDNN (IDs 0..5).
        x: float32 array of shape (N, 41), already scaled.
        device: torch device.
        batch_size: inference batch size.

    Returns:
        y_pred: int64 array of shape (N,), predicted class IDs.
    """
    n = len(x)
    n_classes = 6
    # Collect sigmoid probabilities per class
    all_probs = np.zeros((n, n_classes), dtype=np.float32)

    for class_id, model in sorted(all_models.items()):
        model.eval()
        probs_list: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = torch.from_numpy(x[start: start + batch_size]).to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(probs.ravel())
        all_probs[:, class_id] = np.concatenate(probs_list)

    y_pred = np.argmax(all_probs, axis=1).astype(np.int64)
    return y_pred


# ---------------------------------------------------------------------------
# D1: Isolated extra-class evaluation (3-class argmax over extra classes only)
# ---------------------------------------------------------------------------

def eval_d1_isolated(
    negfix_models: dict[int, BinaryDNN],
    extra_label_map: dict[int, str],
    feature_names: list[str],
    device: torch.device,
    root: Path,
) -> dict[str, Any]:
    """D1: Isolated test on extra classes (3-class argmax)."""
    test_path = root / "data" / "v2" / "processed" / "test_extra.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"test_extra.parquet not found at {test_path}")

    print(f"\nD1: Loading extra test set from {test_path}", flush=True)
    df_test = pd.read_parquet(test_path)
    x_test = df_test[feature_names].to_numpy(dtype=np.float32)
    y_true = df_test["label"].to_numpy(dtype=np.int64)

    print(f"D1: test rows = {len(df_test):,}", flush=True)
    for cid, cname in sorted(extra_label_map.items()):
        n = int((y_true == cid).sum())
        print(f"  Class '{cname}' (ID {cid}): {n:,} test rows", flush=True)

    # 3-class argmax over extra models only
    n = len(x_test)
    extra_ids = sorted(extra_label_map.keys())
    all_probs = np.zeros((n, max(extra_ids) + 1), dtype=np.float32)

    for cid, model in negfix_models.items():
        model.eval()
        probs_list: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, 1024):
                batch = torch.from_numpy(x_test[start: start + 1024]).to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                probs_list.append(probs)
        all_probs[:, cid] = np.concatenate(probs_list)

    # Argmax over extra class columns only
    extra_probs = all_probs[:, extra_ids]
    pred_extra_idx = np.argmax(extra_probs, axis=1)
    y_pred = np.array([extra_ids[i] for i in pred_extra_idx], dtype=np.int64)

    per_class: dict[str, Any] = {}
    for cid, cname in sorted(extra_label_map.items()):
        mask = y_true == cid
        n_true = int(mask.sum())
        n_correct = int((y_pred[mask] == cid).sum())
        recall = n_correct / n_true if n_true > 0 else 0.0
        is_ind = cname == "FTP-BruteForce" or n_true < 500
        per_class[cname] = {
            "n_true": n_true,
            "n_correct": n_correct,
            "recall": round(recall, 6),
            "is_indicative_only": is_ind,
        }
        flag = " [INDICATIVE ONLY]" if is_ind else ""
        print(
            f"  D1 {cname}: Recall={recall*100:.2f}% ({n_correct}/{n_true}){flag}",
            flush=True,
        )

    # Accuracy over all extra-class test flows
    acc = float((y_pred == y_true).mean())
    print(f"D1 isolated extra-class accuracy: {acc:.4f}", flush=True)

    return {"accuracy": round(acc, 6), "per_class": per_class}


# ---------------------------------------------------------------------------
# D2: Mixed 6-class confusion (replicate v2_mixed_confusion.py with negfix)
# ---------------------------------------------------------------------------

def eval_d2_mixed(
    all_models: dict[int, BinaryDNN],
    feature_names: list[str],
    device: torch.device,
    root: Path,
) -> dict[str, Any]:
    """D2: Mixed 6-class confusion check with negfix models."""
    base_test_path = root / "data" / "processed" / "test.parquet"
    extra_test_path = root / "data" / "v2" / "processed" / "test_extra.parquet"

    if not base_test_path.exists():
        raise FileNotFoundError(f"Baseline test parquet not found: {base_test_path}")
    if not extra_test_path.exists():
        raise FileNotFoundError(f"Extra test parquet not found: {extra_test_path}")

    print(f"\nD2: Loading baseline test: {base_test_path}", flush=True)
    df_base = pd.read_parquet(base_test_path)
    print(f"D2: Loading extra test:    {extra_test_path}", flush=True)
    df_extra = pd.read_parquet(extra_test_path)

    # Sample up to 3,000 rows per class with seed 42
    sampled_dfs: list[pd.DataFrame] = []
    sample_counts: dict[str, int] = {}

    for cid in [0, 1, 2]:
        cname = ID_TO_CLASS[cid]
        subset = df_base[df_base["label"] == cid]
        n_take = min(3000, len(subset))
        sampled = subset.sample(n=n_take, random_state=42)
        sampled_dfs.append(sampled)
        sample_counts[cname] = len(sampled)

    for cid in [3, 4, 5]:
        cname = ID_TO_CLASS[cid]
        subset = df_extra[df_extra["label"] == cid]
        n_take = min(3000, len(subset))
        sampled = subset.sample(n=n_take, random_state=42)
        sampled_dfs.append(sampled)
        sample_counts[cname] = len(sampled)

    df_eval = pd.concat(sampled_dfs, ignore_index=True)
    print(f"D2: Sampled {len(df_eval):,} total flows:", flush=True)
    for cname, cnt in sample_counts.items():
        flag = " [INDICATIVE ONLY - N limited]" if cname == "FTP-BruteForce" else ""
        print(f"  - {cname:<16}: {cnt:>5} flows{flag}", flush=True)

    x_features = df_eval[feature_names].to_numpy(dtype=np.float32)
    y_true = df_eval["label"].to_numpy(dtype=np.int64)

    print("D2: Running 6-class inference...", flush=True)
    y_pred = run_six_class_argmax(all_models, x_features, device)

    # 6x6 confusion matrix
    cm = np.zeros((6, 6), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Print confusion matrix
    col_width = 15
    col_headers = [f"{c[:13]:^{col_width}}" for c in CLASSES_ALL]
    header_label = "True/Pred"
    table_header = f"{header_label:<16} | " + " | ".join(col_headers) + " | Total"
    sep = "-" * len(table_header)
    print(f"\n{sep}", flush=True)
    print("D2: 6x6 CONFUSION MATRIX (Negfix Models, Sampled Test Flows)", flush=True)
    print(sep, flush=True)
    print(table_header, flush=True)
    print(sep, flush=True)

    per_class_metrics: dict[str, Any] = {}
    for r_idx, true_name in enumerate(CLASSES_ALL):
        row_vals = cm[r_idx]
        total_row = row_vals.sum()
        row_str = " | ".join(f"{val:^{col_width},}" for val in row_vals)
        print(f"{true_name:<16} | {row_str} | {total_row:>5,}", flush=True)
        rec = float(row_vals[r_idx] / total_row) if total_row > 0 else 0.0
        per_class_metrics[true_name] = {
            "true_count": int(total_row),
            "correct": int(row_vals[r_idx]),
            "recall": round(rec, 6),
            "is_indicative_only": bool(true_name == "FTP-BruteForce"),
        }

    print(sep, flush=True)

    # Leakage analysis
    leakage_per_class: dict[str, Any] = {}
    total_baseline_flows = 0
    total_leaked = 0

    for orig_idx, orig_name in enumerate(["Benign", "DoS", "DDoS"]):
        leaked_counts = {CLASSES_ALL[col_idx]: int(cm[orig_idx, col_idx]) for col_idx in [3, 4, 5]}
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

    print("\nD2: PER-CLASS RECALL", flush=True)
    for cname, m in per_class_metrics.items():
        flag = " [INDICATIVE ONLY]" if m["is_indicative_only"] else ""
        print(
            f"  {cname:<16}: Recall = {m['recall']*100:6.2f}% ({m['correct']}/{m['true_count']}){flag}",
            flush=True,
        )

    leakage_rate = total_leaked / total_baseline_flows if total_baseline_flows > 0 else 0.0
    print(
        f"\nD2: Baseline-to-extra leakage: {total_leaked}/{total_baseline_flows} "
        f"({leakage_rate*100:.2f}%)",
        flush=True,
    )
    for orig_name, info in leakage_per_class.items():
        print(
            f"  - {orig_name:<8}: {info['misclassified_as_extra']:>4}/{info['total_tested']} "
            f"({info['leakage_percent']:.2f}%) -> {info['breakdown']}",
            flush=True,
        )

    return {
        "classes": CLASSES_ALL,
        "sample_counts": sample_counts,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "baseline_leakage_to_extra": {
            "total_baseline_tested": total_baseline_flows,
            "total_misclassified_into_extra": total_leaked,
            "leakage_rate": round(leakage_rate, 6),
            "per_class": leakage_per_class,
        },
        "data_notice": "FTP-BruteForce has limited test flows and is indicative only.",
    }


# ---------------------------------------------------------------------------
# D3: Full baseline test through 6-class argmax
# ---------------------------------------------------------------------------

def eval_d3_full_baseline(
    all_models: dict[int, BinaryDNN],
    feature_names: list[str],
    device: torch.device,
    root: Path,
) -> dict[str, Any]:
    """D3: Run all baseline test flows through 6-class argmax with negfix models."""
    test_path = root / "data" / "processed" / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Baseline test.parquet not found at {test_path}")

    print(f"\nD3: Loading full baseline test: {test_path}", flush=True)
    df_test = pd.read_parquet(test_path)
    print(f"D3: test rows = {len(df_test):,}", flush=True)

    baseline_label_map = {0: "Benign", 1: "DoS", 2: "DDoS"}
    for cid, cname in baseline_label_map.items():
        n = int((df_test["label"] == cid).sum())
        print(f"  Class '{cname}' (ID {cid}): {n:,} test rows", flush=True)

    x_test = df_test[feature_names].to_numpy(dtype=np.float32)
    y_true = df_test["label"].to_numpy(dtype=np.int64)

    print("D3: Running 6-class inference on full baseline test set...", flush=True)
    y_pred = run_six_class_argmax(all_models, x_test, device)

    overall_acc = float((y_pred == y_true).mean())
    print(f"D3: Overall accuracy (baseline flows): {overall_acc:.6f}", flush=True)

    per_class: dict[str, Any] = {}
    for cid, cname in baseline_label_map.items():
        mask = y_true == cid
        n_true = int(mask.sum())
        y_true_binary = (y_true == cid).astype(np.int32)
        y_pred_binary = (y_pred == cid).astype(np.int32)
        f1 = float(f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0))
        n_correct = int((y_pred[mask] == cid).sum())
        recall = n_correct / n_true if n_true > 0 else 0.0
        per_class[cname] = {
            "n_true": n_true,
            "n_correct": n_correct,
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        print(
            f"  D3 {cname}: Recall={recall*100:.2f}% F1={f1:.4f} ({n_correct}/{n_true})",
            flush=True,
        )

    return {"accuracy": round(overall_acc, 6), "per_class": per_class}


# ---------------------------------------------------------------------------
# D4: Demo samples check
# ---------------------------------------------------------------------------

def normalize_source_label(lbl: str) -> str:
    """Map raw dataset label strings to canonical class names."""
    low = lbl.strip().lower()
    if low == "benign":
        return "Benign"
    # Check DDoS before DoS because "ddos" contains the substring "dos"
    if "ddos" in low or "loic" in low or "hoic" in low:
        return "DDoS"
    if "dos" in low or "slowhttp" in low or "hulk" in low or "goldeneye" in low:
        return "DoS"
    if "bot" in low:
        return "Bot"
    if "ftp" in low:
        return "FTP-BruteForce"
    if "ssh" in low:
        return "SSH-Bruteforce"
    return lbl


def eval_d4_demo_samples(
    all_models: dict[int, BinaryDNN],
    feature_names: list[str],
    device: torch.device,
    root: Path,
) -> dict[str, Any]:
    """D4: Run negfix 6-class ensemble on app/raw_samples_v2.csv demo samples."""
    csv_path = root / "app" / "raw_samples_v2.csv"
    if not csv_path.exists():
        print(f"D4: SKIPPED - {csv_path} does not exist.", flush=True)
        return {"skipped": True, "reason": str(csv_path) + " not found"}

    # Load frozen baseline scaler
    scaler_path = root / "models" / "scaler.joblib"
    import joblib
    scaler = joblib.load(scaler_path)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\nD4: Loaded {len(rows)} demo samples from {csv_path}", flush=True)

    header = f"{'#':<3} | {'Source Label':<26} | {'Predicted':<18} | {'Match?':<6} | Provenance"
    sep = "-" * len(header)
    print(sep, flush=True)
    print(header, flush=True)
    print(sep, flush=True)

    matches = 0
    results: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        raw_label = row.get("Label", "")
        norm_source = normalize_source_label(raw_label)
        provenance = row.get("Provenance", "")

        feature_dict = {k: v for k, v in row.items() if k.lower() not in ("label", "provenance")}

        # Scale with frozen baseline scaler
        x_scaled = validate_raw_features(
            raw_features=feature_dict,
            feature_names=feature_names,
            scaler=scaler,
        )

        y_pred_arr = run_six_class_argmax(all_models, x_scaled.reshape(1, -1), device)
        pred_id = int(y_pred_arr[0])
        pred_class = ID_TO_CLASS.get(pred_id, str(pred_id))

        is_match = norm_source == pred_class
        if is_match:
            matches += 1

        match_str = "YES" if is_match else "NO"
        print(f"{idx:<3} | {raw_label:<26} | {pred_class:<18} | {match_str:<6} | {provenance}", flush=True)
        results.append({
            "idx": idx,
            "raw_label": raw_label,
            "normalized_source": norm_source,
            "predicted_class": pred_class,
            "match": is_match,
        })

    print(sep, flush=True)
    total = len(rows)
    pct = matches / total * 100 if total > 0 else 0.0
    print(f"\nD4 SUMMARY: {matches}/{total} matches ({pct:.1f}%)", flush=True)

    return {
        "total": total,
        "matches": matches,
        "match_rate": round(pct / 100, 6),
        "per_sample": results,
    }


# ---------------------------------------------------------------------------
# Acceptance criteria check
# ---------------------------------------------------------------------------

def check_acceptance_criteria(
    d2: dict[str, Any],
    d3: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate all acceptance criteria and print PASS/FAIL."""
    criteria: dict[str, Any] = {}

    # Criterion 1: Baseline 6-class test accuracy >= 0.995
    baseline_acc = d3["accuracy"]
    crit1_pass = baseline_acc >= BASELINE_ACC_THRESHOLD
    criteria["baseline_accuracy"] = {
        "value": baseline_acc,
        "threshold": BASELINE_ACC_THRESHOLD,
        "pass": crit1_pass,
    }

    # Criterion 2: No baseline class F1 < 0.995
    baseline_f1_values = {cname: d3["per_class"][cname]["f1"] for cname in ["Benign", "DoS", "DDoS"]}
    min_f1_class = min(baseline_f1_values, key=baseline_f1_values.get)
    min_f1_val = baseline_f1_values[min_f1_class]
    crit2_pass = min_f1_val >= BASELINE_F1_THRESHOLD
    criteria["baseline_min_f1"] = {
        "values": baseline_f1_values,
        "min_class": min_f1_class,
        "min_value": min_f1_val,
        "threshold": BASELINE_F1_THRESHOLD,
        "pass": crit2_pass,
    }

    # Criterion 3: Baseline leakage < 0.5%
    leakage_rate = d2["baseline_leakage_to_extra"]["leakage_rate"]
    crit3_pass = leakage_rate < LEAKAGE_THRESHOLD
    criteria["baseline_leakage"] = {
        "value": leakage_rate,
        "threshold": LEAKAGE_THRESHOLD,
        "pass": crit3_pass,
    }

    # Criterion 4: Bot recall >= 0.95
    bot_recall = d2["per_class_metrics"]["Bot"]["recall"]
    crit4_pass = bot_recall >= BOT_RECALL_THRESHOLD
    criteria["bot_recall"] = {
        "value": bot_recall,
        "threshold": BOT_RECALL_THRESHOLD,
        "pass": crit4_pass,
    }

    # Criterion 5: SSH-Bruteforce recall >= 0.95
    ssh_recall = d2["per_class_metrics"]["SSH-Bruteforce"]["recall"]
    crit5_pass = ssh_recall >= SSH_RECALL_THRESHOLD
    criteria["ssh_recall"] = {
        "value": ssh_recall,
        "threshold": SSH_RECALL_THRESHOLD,
        "pass": crit5_pass,
    }

    # FTP: reported separately, no pass/fail
    ftp_recall = d2["per_class_metrics"].get("FTP-BruteForce", {}).get("recall", None)
    criteria["ftp_recall_indicative"] = {
        "value": ftp_recall,
        "note": "FTP-BruteForce is indicative only - not a pass/fail criterion",
    }

    all_pass = all(
        v["pass"] for k, v in criteria.items()
        if "pass" in v and k != "ftp_recall_indicative"
    )
    criteria["overall_pass"] = all_pass

    print("\n==================== ACCEPTANCE CRITERIA ====================", flush=True)
    print(f"  Criterion 1 - Baseline accuracy >= {BASELINE_ACC_THRESHOLD}: "
          f"{'PASS' if crit1_pass else 'FAIL'} ({baseline_acc:.4f})", flush=True)
    print(f"  Criterion 2 - Min baseline F1 >= {BASELINE_F1_THRESHOLD}: "
          f"{'PASS' if crit2_pass else 'FAIL'} "
          f"(min={min_f1_val:.4f} for {min_f1_class})", flush=True)
    print(f"  Criterion 3 - Baseline leakage < {LEAKAGE_THRESHOLD:.1%}: "
          f"{'PASS' if crit3_pass else 'FAIL'} ({leakage_rate:.4%})", flush=True)
    print(f"  Criterion 4 - Bot recall >= {BOT_RECALL_THRESHOLD}: "
          f"{'PASS' if crit4_pass else 'FAIL'} ({bot_recall:.4f})", flush=True)
    print(f"  Criterion 5 - SSH recall >= {SSH_RECALL_THRESHOLD}: "
          f"{'PASS' if crit5_pass else 'FAIL'} ({ssh_recall:.4f})", flush=True)
    print(f"  FTP recall (indicative only): {ftp_recall}", flush=True)
    print("-------------------------------------------------------------", flush=True)
    if all_pass:
        print("  OVERALL: ALL CRITERIA PASS - negfix models ready for app wiring.", flush=True)
    else:
        print("  OVERALL: ONE OR MORE CRITERIA FAIL - do NOT wire into app.", flush=True)
        print(
            "  Suggested fix if leakage/recall fails: re-run train with --baseline-ratio 5.",
            flush=True,
        )
    print("=============================================================", flush=True)

    return criteria


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = get_project_root()
    cfg = load_v2_config()

    eval_dir = root / EVAL_DIR_REL
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load negfix label map and feature names
    models_v2b_dir = root / MODELS_V2B_DIR
    label_map_path = models_v2b_dir / "label_map_v2b.json"
    feature_names_path = models_v2b_dir / "feature_names_v2b.json"

    if not label_map_path.exists():
        raise FileNotFoundError(
            f"Negfix label map not found at {label_map_path}. "
            f"Run v2_train_negfix.py --confirm-train first."
        )
    with open(label_map_path, "r", encoding="utf-8") as f:
        raw_map: dict[str, str] = json.load(f)
    extra_label_map: dict[int, str] = {int(k): v for k, v in raw_map.items()}

    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names: list[str] = json.load(f)

    assert len(feature_names) == 41, (
        f"Expected 41 features, got {len(feature_names)} from {feature_names_path}"
    )

    # Load negfix extra-class models
    negfix_models = load_negfix_models(
        models_dir=models_v2b_dir,
        label_map=extra_label_map,
        feature_names=feature_names,
        config=cfg,
        device=device,
    )

    # Load frozen baseline models (from models/ directory)
    baseline_models_dir = root / "models"
    baseline_models = load_frozen_baseline_models(
        models_dir=baseline_models_dir,
        feature_names=feature_names,
        config=cfg,
        device=device,
    )

    # Combine into 6-class model dict
    all_models: dict[int, BinaryDNN] = {}
    all_models.update(baseline_models)   # IDs 0, 1, 2
    all_models.update(negfix_models)     # IDs 3, 4, 5

    print("\n=================================================================", flush=True)
    print("NEGFIX V2 EVALUATION", flush=True)
    print(f"Models: {sorted(all_models.keys())} -> {[ID_TO_CLASS[i] for i in sorted(all_models.keys())]}", flush=True)
    print("=================================================================", flush=True)

    # D1: Isolated extra-class evaluation
    print("\n--- D1: ISOLATED EXTRA-CLASS EVALUATION ---", flush=True)
    d1_results = eval_d1_isolated(
        negfix_models=negfix_models,
        extra_label_map=extra_label_map,
        feature_names=feature_names,
        device=device,
        root=root,
    )

    # D2: Mixed 6-class confusion
    print("\n--- D2: MIXED 6-CLASS CONFUSION CHECK ---", flush=True)
    d2_results = eval_d2_mixed(
        all_models=all_models,
        feature_names=feature_names,
        device=device,
        root=root,
    )

    # D3: Full baseline test
    print("\n--- D3: FULL BASELINE TEST (6-CLASS ARGMAX) ---", flush=True)
    d3_results = eval_d3_full_baseline(
        all_models=all_models,
        feature_names=feature_names,
        device=device,
        root=root,
    )

    # D4: Demo samples
    print("\n--- D4: DEMO SAMPLES CHECK ---", flush=True)
    d4_results = eval_d4_demo_samples(
        all_models=all_models,
        feature_names=feature_names,
        device=device,
        root=root,
    )

    # Check acceptance criteria
    criteria = check_acceptance_criteria(d2_results, d3_results)

    # Save mixed confusion artifact (same format as v2_mixed_confusion.py)
    mixed_path = eval_dir / "mixed_confusion_negfix.json"
    with open(mixed_path, "w", encoding="utf-8") as f:
        json.dump(d2_results, f, indent=2)
    print(f"\nSaved mixed confusion to {mixed_path}", flush=True)

    # Save full summary
    full_summary = {
        "d1_isolated_extra": d1_results,
        "d2_mixed_6class": d2_results,
        "d3_full_baseline": d3_results,
        "d4_demo_samples": d4_results,
        "acceptance_criteria": criteria,
    }
    summary_path = eval_dir / "eval_negfix_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Saved full eval summary to {summary_path}", flush=True)

    # Human-readable report
    lines = [
        "NEGFIX V2 EVALUATION REPORT",
        "=" * 60,
        "",
        "--- D1: ISOLATED EXTRA-CLASS ---",
        f"  Accuracy: {d1_results['accuracy']:.4f}",
    ]
    for cname, m in d1_results["per_class"].items():
        flag = " [INDICATIVE ONLY]" if m["is_indicative_only"] else ""
        lines.append(
            f"  {cname}: Recall={m['recall']*100:.2f}% ({m['n_correct']}/{m['n_true']}){flag}"
        )
    lines += [
        "",
        "--- D2: MIXED 6-CLASS CONFUSION ---",
    ]
    for cname, m in d2_results["per_class_metrics"].items():
        flag = " [INDICATIVE ONLY]" if m["is_indicative_only"] else ""
        lines.append(
            f"  {cname}: Recall={m['recall']*100:.2f}% ({m['correct']}/{m['true_count']}){flag}"
        )
    leakage_info = d2_results["baseline_leakage_to_extra"]
    lines.append(
        f"  Baseline leakage: {leakage_info['total_misclassified_into_extra']}/"
        f"{leakage_info['total_baseline_tested']} "
        f"({leakage_info['leakage_rate']*100:.4f}%)"
    )
    lines += [
        "",
        "--- D3: FULL BASELINE TEST ---",
        f"  Accuracy: {d3_results['accuracy']:.4f}",
    ]
    for cname, m in d3_results["per_class"].items():
        lines.append(
            f"  {cname}: Recall={m['recall']*100:.2f}% F1={m['f1']:.4f} ({m['n_correct']}/{m['n_true']})"
        )
    lines += [
        "",
        "--- ACCEPTANCE CRITERIA ---",
    ]
    for key, val in criteria.items():
        if key == "overall_pass":
            continue
        if key == "ftp_recall_indicative":
            lines.append(f"  FTP recall (indicative): {val.get('value')}")
        elif "pass" in val:
            status = "PASS" if val["pass"] else "FAIL"
            lines.append(f"  {key}: {status} (value={val.get('value', val.get('min_value'))}, threshold={val['threshold']})")
    lines.append(f"  OVERALL: {'PASS' if criteria['overall_pass'] else 'FAIL'}")

    report_text = "\n".join(lines)
    report_path = eval_dir / "eval_negfix_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved eval report to {report_path}", flush=True)
    print(f"\n{report_text}", flush=True)

    # -----------------------------------------------------------------------
    # D5: Gated decision rule evaluation
    # -----------------------------------------------------------------------
    d5_results = eval_d5_gated(
        all_models=all_models,
        feature_names=feature_names,
        device=device,
        root=root,
        eval_dir=eval_dir,
    )


# ---------------------------------------------------------------------------
# D5 implementation
# ---------------------------------------------------------------------------

def compute_all_probabilities(
    all_models: dict[int, BinaryDNN],
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Compute sigmoid probabilities for all 6 models on feature matrix x."""
    n = len(x)
    all_probs = np.zeros((n, 6), dtype=np.float32)
    for class_id, model in sorted(all_models.items()):
        model.eval()
        probs_list: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(x[start : start + batch_size])).to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                probs_list.append(probs)
        all_probs[:, class_id] = np.concatenate(probs_list)
    return all_probs


def predict_gated(
    all_probs: np.ndarray,
    include_ftp: bool = True,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply gated decision rule on precomputed 6-class probabilities.

    If max over extra classes >= threshold (fixed at 0.5), predict argmax
    over extra classes; otherwise predict argmax over baseline classes.
    """
    extra_cids = [3, 4, 5] if include_ftp else [3, 5]
    baseline_cids = [0, 1, 2]

    extra_probs = all_probs[:, extra_cids]
    max_extra = np.max(extra_probs, axis=1)
    best_extra_idx = np.argmax(extra_probs, axis=1)
    best_extra_cids = np.array([extra_cids[i] for i in best_extra_idx], dtype=np.int64)

    baseline_probs = all_probs[:, baseline_cids]
    best_baseline_cids = np.argmax(baseline_probs, axis=1).astype(np.int64)

    y_pred = np.where(max_extra >= threshold, best_extra_cids, best_baseline_cids)
    return y_pred


def eval_gated_validation_sets(
    all_models: dict[int, BinaryDNN],
    feature_names: list[str],
    device: torch.device,
    root: Path,
) -> dict[str, Any]:
    """Evaluate gated decision rule on validation sets ONLY (no test data)."""
    val_base_path = root / "data" / "processed" / "val.parquet"
    val_extra_path = root / "data" / "v2" / "processed" / "val_extra.parquet"

    print("\n" + "=" * 70, flush=True)
    print("D5: VALIDATION SET BEHAVIOR UNDER GATED DECISION RULE (NO TEST DATA)", flush=True)
    print("=" * 70, flush=True)
    print(f"Loading baseline val: {val_base_path}", flush=True)
    df_val_base = pd.read_parquet(val_base_path)
    print(f"Loading extra val:    {val_extra_path}", flush=True)
    df_val_extra = pd.read_parquet(val_extra_path)

    x_val_base = df_val_base[feature_names].to_numpy(dtype=np.float32)
    x_val_extra = df_val_extra[feature_names].to_numpy(dtype=np.float32)
    y_val_extra = df_val_extra["label"].to_numpy(dtype=np.int64)

    print("Computing validation probabilities...", flush=True)
    probs_val_base = compute_all_probabilities(all_models, x_val_base, device)
    probs_val_extra = compute_all_probabilities(all_models, x_val_extra, device)

    val_results: dict[str, Any] = {}

    for include_ftp, vname in [(True, "gated_with_ftp"), (False, "gated_without_ftp")]:
        tag = "WITH FTP" if include_ftp else "WITHOUT FTP"
        extra_cids = [3, 4, 5] if include_ftp else [3, 5]

        # Baseline val leakage
        y_pred_base = predict_gated(probs_val_base, include_ftp=include_ftp, threshold=0.5)
        leaked_mask = np.isin(y_pred_base, extra_cids)
        leaked_cnt = int(leaked_mask.sum())
        leakage_rate = float(leaked_cnt / len(df_val_base))

        # Extra val recall
        y_pred_extra = predict_gated(probs_val_extra, include_ftp=include_ftp, threshold=0.5)
        bot_mask = y_val_extra == 3
        bot_recall = float((y_pred_extra[bot_mask] == 3).sum() / bot_mask.sum()) if bot_mask.sum() > 0 else 0.0

        ssh_mask = y_val_extra == 5
        ssh_recall = float((y_pred_extra[ssh_mask] == 5).sum() / ssh_mask.sum()) if ssh_mask.sum() > 0 else 0.0

        ftp_mask = y_val_extra == 4
        ftp_recall = float((y_pred_extra[ftp_mask] == 4).sum() / ftp_mask.sum()) if (ftp_mask.sum() > 0 and include_ftp) else None

        val_results[vname] = {
            "include_ftp": include_ftp,
            "baseline_val_rows": len(df_val_base),
            "baseline_leaked_count": leaked_cnt,
            "baseline_leakage_rate": round(leakage_rate, 6),
            "bot_val_rows": int(bot_mask.sum()),
            "bot_val_recall": round(bot_recall, 6),
            "ssh_val_rows": int(ssh_mask.sum()),
            "ssh_val_recall": round(ssh_recall, 6),
            "ftp_val_rows": int(ftp_mask.sum()),
            "ftp_val_recall": round(ftp_recall, 6) if ftp_recall is not None else None,
        }

        print(f"\n--- Validation Set Results [{tag}] ---", flush=True)
        print(f"  Baseline val flows tested:    {len(df_val_base):,}", flush=True)
        print(f"  Baseline leakage into extra:  {leaked_cnt}/{len(df_val_base)} ({leakage_rate*100:.4f}%)", flush=True)
        print(f"  Bot val recall:               {bot_recall*100:.2f}% ({(y_pred_extra[bot_mask] == 3).sum()}/{bot_mask.sum()})", flush=True)
        print(f"  SSH-Bruteforce val recall:    {ssh_recall*100:.2f}% ({(y_pred_extra[ssh_mask] == 5).sum()}/{ssh_mask.sum()})", flush=True)
        if include_ftp:
            print(f"  FTP-BruteForce val recall:    {ftp_recall*100:.2f}% ({(y_pred_extra[ftp_mask] == 4).sum()}/{ftp_mask.sum()}) [INDICATIVE]", flush=True)

    return val_results


def eval_gated_mixed_sample(
    probs_mixed: np.ndarray,
    y_true_mixed: np.ndarray,
    sample_counts: dict[str, int],
    include_ftp: bool,
    title: str,
) -> dict[str, Any]:
    """Evaluate gated decision rule on the sampled mixed test set (15,008 flows)."""
    y_pred = predict_gated(probs_mixed, include_ftp=include_ftp, threshold=0.5)

    # 6x6 confusion matrix
    cm = np.zeros((6, 6), dtype=np.int64)
    for t, p in zip(y_true_mixed, y_pred):
        cm[t, p] += 1

    col_width = 15
    col_headers = [f"{c[:13]:^{col_width}}" for c in CLASSES_ALL]
    header_label = "True/Pred"
    table_header = f"{header_label:<16} | " + " | ".join(col_headers) + " | Total"
    sep = "-" * len(table_header)

    print(f"\n{sep}", flush=True)
    print(f"D5 MIXED 6x6 CONFUSION MATRIX: {title}", flush=True)
    print(sep, flush=True)
    print(table_header, flush=True)
    print(sep, flush=True)

    per_class_metrics: dict[str, Any] = {}
    for r_idx, true_name in enumerate(CLASSES_ALL):
        row_vals = cm[r_idx]
        total_row = row_vals.sum()
        row_str = " | ".join(f"{val:^{col_width},}" for val in row_vals)
        print(f"{true_name:<16} | {row_str} | {total_row:>5,}", flush=True)
        rec = float(row_vals[r_idx] / total_row) if total_row > 0 else 0.0
        per_class_metrics[true_name] = {
            "true_count": int(total_row),
            "correct": int(row_vals[r_idx]),
            "recall": round(rec, 6),
            "is_indicative_only": bool(true_name == "FTP-BruteForce"),
        }
    print(sep, flush=True)

    # Leakage
    leakage_per_class: dict[str, Any] = {}
    total_baseline_flows = 0
    total_leaked = 0
    extra_cols = [3, 4, 5] if include_ftp else [3, 5]

    for orig_idx, orig_name in enumerate(["Benign", "DoS", "DDoS"]):
        leaked_counts = {CLASSES_ALL[col_idx]: int(cm[orig_idx, col_idx]) for col_idx in extra_cols}
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

    leakage_rate = total_leaked / total_baseline_flows if total_baseline_flows > 0 else 0.0
    print(f"\n{title} - PER-CLASS RECALL:", flush=True)
    for cname, m in per_class_metrics.items():
        flag = " [INDICATIVE ONLY]" if m["is_indicative_only"] else ""
        print(f"  {cname:<16}: Recall = {m['recall']*100:6.2f}% ({m['correct']}/{m['true_count']}){flag}", flush=True)

    print(f"\n{title} - Baseline leakage into extra classes: {total_leaked}/{total_baseline_flows} ({leakage_rate*100:.4f}%)", flush=True)
    for orig_name, info in leakage_per_class.items():
        print(f"  - {orig_name:<8}: {info['misclassified_as_extra']:>4}/{info['total_tested']} ({info['leakage_percent']:.2f}%) -> {info['breakdown']}", flush=True)

    return {
        "title": title,
        "classes": CLASSES_ALL,
        "sample_counts": sample_counts,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "baseline_leakage_to_extra": {
            "total_baseline_tested": total_baseline_flows,
            "total_misclassified_into_extra": total_leaked,
            "leakage_rate": round(leakage_rate, 6),
            "per_class": leakage_per_class,
        },
    }


def eval_gated_full_baseline(
    probs_baseline: np.ndarray,
    y_true_baseline: np.ndarray,
    include_ftp: bool,
    title: str,
) -> dict[str, Any]:
    """Evaluate gated decision rule on full baseline test set (119,454 flows)."""
    y_pred = predict_gated(probs_baseline, include_ftp=include_ftp, threshold=0.5)

    overall_acc = float((y_pred == y_true_baseline).mean())
    baseline_label_map = {0: "Benign", 1: "DoS", 2: "DDoS"}

    per_class: dict[str, Any] = {}
    for cid, cname in baseline_label_map.items():
        mask = y_true_baseline == cid
        n_true = int(mask.sum())
        y_true_bin = (y_true_baseline == cid).astype(np.int32)
        y_pred_bin = (y_pred == cid).astype(np.int32)
        f1 = float(f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0))
        n_correct = int((y_pred[mask] == cid).sum())
        recall = n_correct / n_true if n_true > 0 else 0.0
        per_class[cname] = {
            "n_true": n_true,
            "n_correct": n_correct,
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    # Breakdown of baseline flows predicted as extra classes
    extra_cids_list = [(3, "Bot"), (4, "FTP-BruteForce"), (5, "SSH-Bruteforce")] if include_ftp else [(3, "Bot"), (5, "SSH-Bruteforce")]
    baseline_as_extra = {cname: int((y_pred == cid).sum()) for cid, cname in extra_cids_list}
    total_baseline_as_extra = sum(baseline_as_extra.values())

    print(f"\n{title} - Full Baseline Test Set (119,454 flows):", flush=True)
    print(f"  Overall accuracy: {overall_acc:.6f}", flush=True)
    for cname, m in per_class.items():
        print(f"  {cname:<8}: Recall={m['recall']*100:.2f}% F1={m['f1']:.4f} ({m['n_correct']}/{m['n_true']})", flush=True)
    print(f"  Baseline flows predicted as extra classes: {total_baseline_as_extra}/{len(y_true_baseline)} ({total_baseline_as_extra/len(y_true_baseline)*100:.4f}%) -> {baseline_as_extra}", flush=True)

    return {
        "title": title,
        "accuracy": round(overall_acc, 6),
        "per_class": per_class,
        "baseline_flows_predicted_as_extra": {
            "total": total_baseline_as_extra,
            "rate": round(total_baseline_as_extra / len(y_true_baseline), 6),
            "breakdown": baseline_as_extra,
        },
    }


def eval_gated_demo_samples(
    all_models: dict[int, BinaryDNN],
    feature_names: list[str],
    device: torch.device,
    root: Path,
    include_ftp: bool,
    title: str,
) -> dict[str, Any]:
    """Evaluate gated decision rule on 18 demo samples in app/raw_samples_v2.csv."""
    csv_path = root / "app" / "raw_samples_v2.csv"
    if not csv_path.exists():
        print(f"Demo samples check skipped: {csv_path} not found.", flush=True)
        return {"skipped": True}

    import joblib
    scaler_path = root / "models" / "scaler.joblib"
    scaler = joblib.load(scaler_path)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Scale features
    x_scaled_list: list[np.ndarray] = []
    for row in rows:
        fdict = {k: v for k, v in row.items() if k.lower() not in ("label", "provenance")}
        xs = validate_raw_features(raw_features=fdict, feature_names=feature_names, scaler=scaler)
        x_scaled_list.append(xs)
    x_demo = np.vstack(x_scaled_list)

    probs_demo = compute_all_probabilities(all_models, x_demo, device)
    y_pred_demo = predict_gated(probs_demo, include_ftp=include_ftp, threshold=0.5)

    header = f"{'#':<3} | {'Source Label':<26} | {'Predicted':<18} | {'Match?':<6} | Provenance"
    sep = "-" * len(header)
    print(f"\n{sep}", flush=True)
    print(f"{title} - DEMO SAMPLES CHECK (18 flows)", flush=True)
    print(sep, flush=True)
    print(header, flush=True)
    print(sep, flush=True)

    matches = 0
    results: list[dict[str, Any]] = []

    for idx, (row, pred_id) in enumerate(zip(rows, y_pred_demo), start=1):
        raw_label = row.get("Label", "")
        norm_source = normalize_source_label(raw_label)
        pred_class = ID_TO_CLASS.get(int(pred_id), str(pred_id))
        provenance = row.get("Provenance", "")

        is_match = norm_source == pred_class
        if is_match:
            matches += 1

        match_str = "YES" if is_match else "NO"
        print(f"{idx:<3} | {raw_label:<26} | {pred_class:<18} | {match_str:<6} | {provenance}", flush=True)
        results.append({
            "idx": idx,
            "raw_label": raw_label,
            "normalized_source": norm_source,
            "predicted_class": pred_class,
            "match": is_match,
        })

    print(sep, flush=True)
    pct = matches / len(rows) * 100 if len(rows) > 0 else 0.0
    print(f"{title} - Demo Matches: {matches}/{len(rows)} ({pct:.1f}%)\n", flush=True)

    return {
        "title": title,
        "total": len(rows),
        "matches": matches,
        "match_rate": round(pct / 100, 6),
        "per_sample": results,
    }


def eval_d5_gated(
    all_models: dict[int, BinaryDNN],
    feature_names: list[str],
    device: torch.device,
    root: Path,
    eval_dir: Path,
) -> dict[str, Any]:
    """Execute Section D5: Gated decision rule evaluation."""
    print("\n" + "=" * 70, flush=True)
    print("SECTION D5: GATED DECISION RULE EVALUATION", flush=True)
    print("Rule: if max(p_extra) >= 0.5 -> argmax(extra); else -> argmax(baseline)", flush=True)
    print("=" * 70, flush=True)

    # 1. Validation set check (no test data)
    val_behavior = eval_gated_validation_sets(all_models, feature_names, device, root)

    # 2. Prepare test data for (a) and (b)
    base_test_path = root / "data" / "processed" / "test.parquet"
    extra_test_path = root / "data" / "v2" / "processed" / "test_extra.parquet"

    df_base = pd.read_parquet(base_test_path)
    df_extra = pd.read_parquet(extra_test_path)

    # Mixed 15,008 sample (same seed 42 as D2)
    sampled_dfs: list[pd.DataFrame] = []
    sample_counts: dict[str, int] = {}
    for cid in [0, 1, 2]:
        cname = ID_TO_CLASS[cid]
        subset = df_base[df_base["label"] == cid]
        n_take = min(3000, len(subset))
        sampled = subset.sample(n=n_take, random_state=42)
        sampled_dfs.append(sampled)
        sample_counts[cname] = len(sampled)

    for cid in [3, 4, 5]:
        cname = ID_TO_CLASS[cid]
        subset = df_extra[df_extra["label"] == cid]
        n_take = min(3000, len(subset))
        sampled = subset.sample(n=n_take, random_state=42)
        sampled_dfs.append(sampled)
        sample_counts[cname] = len(sampled)

    df_mixed = pd.concat(sampled_dfs, ignore_index=True)
    x_mixed = df_mixed[feature_names].to_numpy(dtype=np.float32)
    y_true_mixed = df_mixed["label"].to_numpy(dtype=np.int64)

    print("\nComputing probabilities for 15,008-flow mixed sample...", flush=True)
    probs_mixed = compute_all_probabilities(all_models, x_mixed, device)

    # Full baseline test set
    x_full_base = df_base[feature_names].to_numpy(dtype=np.float32)
    y_true_full_base = df_base["label"].to_numpy(dtype=np.int64)

    print("Computing probabilities for 119,454-flow baseline test set...", flush=True)
    probs_full_base = compute_all_probabilities(all_models, x_full_base, device)

    # 3. Run Variant 1: Gated WITH FTP
    title_v1 = "GATED DECISION RULE (WITH FTP)"
    mixed_v1 = eval_gated_mixed_sample(probs_mixed, y_true_mixed, sample_counts, include_ftp=True, title=title_v1)
    base_v1 = eval_gated_full_baseline(probs_full_base, y_true_full_base, include_ftp=True, title=title_v1)
    demo_v1 = eval_gated_demo_samples(all_models, feature_names, device, root, include_ftp=True, title=title_v1)
    crit_v1 = check_acceptance_criteria(mixed_v1, base_v1)

    # 4. Run Variant 2: Gated WITHOUT FTP
    title_v2 = "GATED DECISION RULE (WITHOUT FTP)"
    mixed_v2 = eval_gated_mixed_sample(probs_mixed, y_true_mixed, sample_counts, include_ftp=False, title=title_v2)
    base_v2 = eval_gated_full_baseline(probs_full_base, y_true_full_base, include_ftp=False, title=title_v2)
    demo_v2 = eval_gated_demo_samples(all_models, feature_names, device, root, include_ftp=False, title=title_v2)
    crit_v2 = check_acceptance_criteria(mixed_v2, base_v2)

    # Save to artifacts/v2b/eval/eval_negfix_gated.json
    gated_summary = {
        "threshold": 0.5,
        "validation_behavior": val_behavior,
        "gated_with_ftp": {
            "mixed_sample_15008": mixed_v1,
            "full_baseline_test_119454": base_v1,
            "demo_samples": demo_v1,
            "acceptance_criteria": crit_v1,
        },
        "gated_without_ftp": {
            "mixed_sample_15008": mixed_v2,
            "full_baseline_test_119454": base_v2,
            "demo_samples": demo_v2,
            "acceptance_criteria": crit_v2,
        },
    }

    out_json = eval_dir / "eval_negfix_gated.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(gated_summary, f, indent=2)
    print(f"\nSaved gated evaluation results to: {out_json}", flush=True)

    # Print summary comparison table
    print("\n" + "=" * 78, flush=True)
    print("D5 SUMMARY COMPARISON: ACCEPTANCE CRITERIA UNDER GATED DECISION RULES", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Metric / Criterion':<32} | {'Threshold':<12} | {'With FTP':<15} | {'Without FTP':<15}", flush=True)
    print("-" * 78, flush=True)
    print(f"{'1. Baseline Accuracy':<32} | {'>= 0.9950':<12} | {base_v1['accuracy']:<15.6f} | {base_v2['accuracy']:<15.6f}", flush=True)
    print(f"{'2. Min Baseline F1 (Benign)':<32} | {'>= 0.9950':<12} | {base_v1['per_class']['Benign']['f1']:<15.4f} | {base_v2['per_class']['Benign']['f1']:<15.4f}", flush=True)
    print(f"{'3. Baseline Leakage Rate':<32} | {'< 0.50%':<12} | {mixed_v1['baseline_leakage_to_extra']['leakage_rate']*100:<14.4f}% | {mixed_v2['baseline_leakage_to_extra']['leakage_rate']*100:<14.4f}%", flush=True)
    print(f"{'4. Bot Recall (Mixed)':<32} | {'>= 0.9500':<12} | {mixed_v1['per_class_metrics']['Bot']['recall']*100:<14.2f}% | {mixed_v2['per_class_metrics']['Bot']['recall']*100:<14.2f}%", flush=True)
    print(f"{'5. SSH Recall (Mixed)':<32} | {'>= 0.9500':<12} | {mixed_v1['per_class_metrics']['SSH-Bruteforce']['recall']*100:<14.2f}% | {mixed_v2['per_class_metrics']['SSH-Bruteforce']['recall']*100:<14.2f}%", flush=True)
    ftp_rec_str = f"{mixed_v1['per_class_metrics']['FTP-BruteForce']['recall']*100:.2f}%"
    print(f"{'FTP Recall (Indicative)':<32} | {'N/A':<12} | {ftp_rec_str:<15} | {'N/A (excluded)':<15}", flush=True)
    demo_str_v1 = f"{demo_v1['matches']}/18 ({demo_v1['match_rate']*100:.1f}%)"
    demo_str_v2 = f"{demo_v2['matches']}/18 ({demo_v2['match_rate']*100:.1f}%)"
    print(f"{'Demo Matches (out of 18)':<32} | {'N/A':<12} | {demo_str_v1:<15} | {demo_str_v2:<15}", flush=True)
    print("-" * 78, flush=True)
    status_v1 = "PASS" if crit_v1["overall_pass"] else "FAIL"
    status_v2 = "PASS" if crit_v2["overall_pass"] else "FAIL"
    print(f"{'OVERALL ACCEPTANCE':<32} | {'ALL PASS':<12} | {status_v1:<15} | {status_v2:<15}", flush=True)
    print("=" * 78 + "\n", flush=True)

    return gated_summary


if __name__ == "__main__":
    main()

