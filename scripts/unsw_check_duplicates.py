"""Check duplicate rows within UNSW-NB15 training set and between train/test.

Protocol Requirements:
1. Check for exact duplicate rows within UNSW_NB15_training-set.csv.
2. Check for exact duplicate rows between training and testing sets.
3. Exclude 'id', 'ID', 'label', 'attack_cat' when comparing feature rows.
4. Report counts clearly.
5. In training only: drop duplicates. Never modify the testing set.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "unsw"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "v2" / "unsw_eval" / "audit"

EXCLUDED_COLS = ["id", "ID", "label", "attack_cat"]


def audit_duplicates() -> dict[str, int | float]:
    train_path = DATA_DIR / "UNSW_NB15_training-set.csv"
    test_path = DATA_DIR / "UNSW_NB15_testing-set.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training set not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing set not found at {test_path}")

    print("=" * 80)
    print("UNSW-NB15 DUPLICATE AUDIT (EXCLUDING ID, LABEL, ATTACK_CAT)")
    print("=" * 80)
    print(f"Loading training set from: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"Loading testing set from:  {test_path}")
    df_test = pd.read_csv(test_path)

    n_train_raw = len(df_train)
    n_test_raw = len(df_test)
    print(f"Raw rows: Train = {n_train_raw:,} | Test = {n_test_raw:,}")

    feature_cols = [c for c in df_train.columns if c not in EXCLUDED_COLS]
    print(f"Comparing on {len(feature_cols)} feature columns (excluding id, ID, label, attack_cat)...")

    # 1. Duplicates within training set
    train_dup_mask = df_train.duplicated(subset=feature_cols, keep="first")
    n_train_dups = int(train_dup_mask.sum())
    n_train_unique = n_train_raw - n_train_dups
    print("\n--- 1. Within Training Set ---")
    print(f"Total rows:             {n_train_raw:,}")
    print(f"Exact duplicate rows:   {n_train_dups:,} ({n_train_dups / n_train_raw * 100:.2f}%)")
    print(f"Unique feature rows:    {n_train_unique:,} ({n_train_unique / n_train_raw * 100:.2f}%)")

    # Inspect class breakdown of duplicates in training
    dup_train_labels = df_train.loc[train_dup_mask, "label"].value_counts().to_dict()
    print(f"Training duplicates by label: {dup_train_labels}")

    # 2. Duplicates within testing set (reporting only, test set is never modified)
    test_dup_mask = df_test.duplicated(subset=feature_cols, keep="first")
    n_test_dups = int(test_dup_mask.sum())
    n_test_unique = n_test_raw - n_test_dups
    print("\n--- 2. Within Testing Set (Observation Only, Never Modified) ---")
    print(f"Total rows:             {n_test_raw:,}")
    print(f"Exact duplicate rows:   {n_test_dups:,} ({n_test_dups / n_test_raw * 100:.2f}%)")
    print(f"Unique feature rows:    {n_test_unique:,}")

    # 3. Overlap between training and testing sets
    print("\n--- 3. Overlap Between Training and Testing Sets ---")
    # Identify test rows whose feature representation exactly matches any training row
    df_train_dedup = df_train.drop_duplicates(subset=feature_cols, keep="first")
    train_features_set = set(tuple(x) for x in df_train_dedup[feature_cols].itertuples(index=False, name=None))
    
    test_features_tuples = [tuple(x) for x in df_test[feature_cols].itertuples(index=False, name=None)]
    test_in_train = [t in train_features_set for t in test_features_tuples]
    n_test_in_train = sum(test_in_train)
    pct_test_overlap = n_test_in_train / n_test_raw * 100
    print(f"Test flows exactly identical to a training flow: {n_test_in_train:,} / {n_test_raw:,} ({pct_test_overlap:.2f}%)")

    # 4. Strict Protocol Decision
    print("\n--- 4. Protocol Action ---")
    print("Rule: Drop duplicate rows strictly in the training set.")
    print("Rule: Never drop or modify rows in the official test set.")
    print(f"Deduplicated training set will contain: {n_train_unique:,} flows.")

    summary = {
        "n_train_raw": n_train_raw,
        "n_train_dups": n_train_dups,
        "n_train_unique": n_train_unique,
        "n_test_raw": n_test_raw,
        "n_test_dups": n_test_dups,
        "n_test_unique": n_test_unique,
        "n_test_flows_in_train": n_test_in_train,
        "pct_test_overlap": pct_test_overlap,
        "action": "Drop duplicates strictly in training data. Test set left untouched.",
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = ARTIFACTS_DIR / "unsw_duplicate_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAudit saved to: {out_file}")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    audit_duplicates()

