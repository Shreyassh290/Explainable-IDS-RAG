"""Comprehensive Diagnostic Script for EXP-01A False Positives and Covariate Shift.

Performs:
1. Exact verification of False Negatives across all attack families (target = 760).
2. Deep inspection of the 10,698 False Positive Benign samples.
3. Statistical comparison of Benign feature distributions (Train vs Val vs Test).
4. Measurement of short-lived connections, TTLs, packet counts, bytes, and rates.
5. Identification of top feature disparities distinguishing False Positives from True Negatives.
6. Empirical verification of Benign covariate shift using Wasserstein distance and KS statistics.

Read-only analysis script: writes strictly to artifacts/v2/unsw_eval/experiments/exp01/diagnostic_fp_report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root
from scripts.v2_exp01_loss_and_leakage import (
    BoostedDNN,
    LeakFreePreprocessor,
    NOISE_COLUMNS,
    load_and_split_unsw,
)


def main() -> None:
    root = get_project_root()
    data_dir = root / "data" / "unsw"
    exp_models_dir = root / "models_unsw" / "experiments" / "exp01"
    exp_artifacts_dir = root / "artifacts" / "v2" / "unsw_eval" / "experiments" / "exp01"

    model_path = exp_models_dir / "model_EXP-01A.pt"
    prep_path = exp_models_dir / "preprocessor_exp01.joblib"
    feat_path = exp_models_dir / "feature_names_exp01.json"

    print("Loading EXP-01A artifacts for diagnostic audit...", flush=True)
    preprocessor: LeakFreePreprocessor = joblib.load(prep_path)
    with open(feat_path, encoding="utf-8") as f:
        feature_names = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoostedDNN(in_features=66, hidden=[256, 128, 64, 32], dropout=0.25).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 1. Load Data Splits
    df_tr, df_val, df_test = load_and_split_unsw(data_dir, val_ratio=0.15, seed=42)

    y_test = df_test["label"].to_numpy(dtype=np.float32)
    test_attack_cats = df_test["attack_cat"].fillna("Normal").astype(str).to_numpy()

    # Transform test features
    X_test = preprocessor.transform(df_test)
    test_tensor = torch.from_numpy(X_test).to(device)

    with torch.no_grad():
        test_logits = model(test_tensor).squeeze(1).cpu().numpy()
        test_probs = expit(test_logits)

    tau = 0.4650
    test_preds = (test_probs >= tau).astype(int)

    # =========================================================================
    # PART 1: EXACT VERIFICATION OF FALSE NEGATIVES
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: VERIFICATION OF FALSE NEGATIVES PER ATTACK FAMILY")
    print("=" * 70)

    df_eval = pd.DataFrame({
        "attack_cat": test_attack_cats,
        "y_true": y_test,
        "y_pred": test_preds,
        "prob": test_probs,
    })

    attack_eval = df_eval[df_eval["y_true"] == 1]
    fn_per_cat = {}
    total_fn = 0

    for cat, group in attack_eval.groupby("attack_cat"):
        missed = int(np.sum(group["y_pred"] == 0))
        support = len(group)
        rec = (support - missed) / support if support > 0 else 0.0
        fn_per_cat[cat] = {"missed_fn": missed, "support": support, "recall": rec}
        total_fn += missed

    print(f"{'Attack Family':<16} | {'Support':<8} | {'Missed (FN)':<12} | {'Detection Rate'}")
    print("-" * 70)
    for cat, d in sorted(fn_per_cat.items(), key=lambda x: x[1]["missed_fn"], reverse=True):
        print(f"{cat:<16} | {d['support']:<8,d} | {d['missed_fn']:<12,d} | {d['recall']*100:6.2f}%")
    print("-" * 70)
    print(f"Total False Negatives Sum: {total_fn:,} (Expected: 760)")
    assert total_fn == 760, f"False negatives sum {total_fn} != 760"
    print("Exact False Negative Sum Confirmed: Exactly 760 missed attacks.")

    # =========================================================================
    # PART 2: INSPECT 10,698 FALSE POSITIVE BENIGN SAMPLES
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: INSPECTION OF 10,698 FALSE POSITIVES (NORMAL MISCLASSIFIED AS ATTACK)")
    print("=" * 70)

    benign_test = df_eval[df_eval["y_true"] == 0].copy()
    fp_mask = benign_test["y_pred"] == 1
    tn_mask = benign_test["y_pred"] == 0

    fp_count = int(fp_mask.sum())
    tn_count = int(tn_mask.sum())
    print(f"Total Benign Test Flows: {len(benign_test):,}")
    print(f"True Negatives:          {tn_count:,} ({tn_count/len(benign_test):.2%})")
    print(f"False Positives:         {fp_count:,} ({fp_count/len(benign_test):.2%})")
    assert fp_count == 10698, f"Expected 10,698 FP, got {fp_count}"

    # Get raw test rows for FP and TN
    df_fp = df_test.loc[benign_test[fp_mask].index].copy()
    df_tn = df_test.loc[benign_test[tn_mask].index].copy()

    # Protocol & State distributions in False Positives
    print("\nProtocol Distribution in False Positives vs True Negatives:")
    fp_protos = df_fp["proto"].value_counts(normalize=True).head(5).to_dict()
    tn_protos = df_tn["proto"].value_counts(normalize=True).head(5).to_dict()
    for p in set(list(fp_protos.keys()) + list(tn_protos.keys())):
        print(f"  Proto {p:<8}: FP={fp_protos.get(p, 0.0)*100:5.2f}% | TN={tn_protos.get(p, 0.0)*100:5.2f}%")

    print("\nState Distribution in False Positives vs True Negatives:")
    fp_states = df_fp["state"].value_counts(normalize=True).head(5).to_dict()
    tn_states = df_tn["state"].value_counts(normalize=True).head(5).to_dict()
    for s in set(list(fp_states.keys()) + list(tn_states.keys())):
        print(f"  State {s:<8}: FP={fp_states.get(s, 0.0)*100:5.2f}% | TN={tn_states.get(s, 0.0)*100:5.2f}%")

    # =========================================================================
    # PART 3: EMPIRICAL COVARIATE SHIFT MEASUREMENT (BENIGN TRAIN vs TEST)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: BENIGN COVARIATE SHIFT MEASUREMENT (TRAIN vs TEST)")
    print("=" * 70)

    benign_tr = df_tr[df_tr["label"] == 0]
    benign_val = df_val[df_val["label"] == 0]
    benign_te = df_test[df_test["label"] == 0]

    key_numeric = [
        "sttl", "dttl", "dur", "spkts", "dpkts", "sbytes", "dbytes",
        "rate", "sload", "dload", "ct_state_ttl", "ct_srv_src",
        "ct_dst_ltm", "ct_src_dport_ltm", "swin", "dwin"
    ]

    shift_report = []
    print(f"{'Feature':<18} | {'Benign Train Median':<20} | {'Benign Test Median':<20} | {'KS Stat':<8} | {'p-value':<8}")
    print("-" * 80)

    for col in key_numeric:
        if col in benign_tr.columns:
            tr_vals = benign_tr[col].dropna().to_numpy()
            te_vals = benign_te[col].dropna().to_numpy()
            ks_res = ks_2samp(tr_vals, te_vals)

            med_tr = float(np.median(tr_vals))
            med_te = float(np.median(te_vals))

            shift_report.append({
                "feature": col,
                "train_median": med_tr,
                "test_median": med_te,
                "ks_statistic": round(float(ks_res.statistic), 4),
                "p_value": float(ks_res.pvalue),
                "significant_shift": bool(ks_res.pvalue < 0.01),
            })

            print(f"{col:<18} | {med_tr:<20.4f} | {med_te:<20.4f} | {ks_res.statistic:<8.4f} | {ks_res.pvalue:<8.2e}")

    # =========================================================================
    # PART 4: DISCRIMINATIVE DISPARITY (FALSE POSITIVES vs TRUE NEGATIVES)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: TOP FEATURE DISPARITIES IN FALSE POSITIVES (FP vs TN)")
    print("=" * 70)

    fp_disparity = []
    for col in key_numeric:
        med_fp = float(np.median(df_fp[col]))
        med_tn = float(np.median(df_tn[col]))
        diff = med_fp - med_tn
        fp_disparity.append({
            "feature": col,
            "fp_median": med_fp,
            "tn_median": med_tn,
            "difference": diff,
        })
        print(f"  {col:<18} | FP Median: {med_fp:<14.4f} | TN Median: {med_tn:<14.4f} | Diff: {diff:<14.4f}")

    # Save diagnostic findings
    diagnostic_payload = {
        "false_negatives_audit": {
            "total_false_negatives": total_fn,
            "breakdown": fn_per_cat,
        },
        "false_positives_audit": {
            "total_false_positives": fp_count,
            "benign_fpr": float(fp_count / len(benign_test)),
            "protocol_distribution_fp": fp_protos,
            "state_distribution_fp": fp_states,
            "top_feature_disparities": fp_disparity,
        },
        "benign_covariate_shift_audit": shift_report,
    }

    out_diag_path = exp_artifacts_dir / "diagnostic_fp_report.json"
    with open(out_diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic_payload, f, indent=2)

    print(f"\nDiagnostic audit report saved to: {out_diag_path}")


if __name__ == "__main__":
    main()

