"""List candidate rows per class from test_predictions.parquet.

Filters for true_class == predicted_class and tier == 'VERY_HIGH'.
Prints the first 10 candidate row indices per class with confidence scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
preds_path = repo_root / "artifacts" / "test_predictions.parquet"

if not preds_path.exists():
    print(f"ERROR: {preds_path} not found", file=sys.stderr)
    sys.exit(1)

df_preds = pd.read_parquet(preds_path)

# Verify required columns exist
cols = list(df_preds.columns)
true_col = "true_class" if "true_class" in cols else "true_label"
pred_col = "predicted_class" if "predicted_class" in cols else "predicted_label"
tier_col = "tier"
conf_col = "confidence"

classes = sorted(df_preds[true_col].unique())

print(f"Total test rows: {len(df_preds)}")
print(f"Filter: {true_col} == {pred_col} and {tier_col} == 'VERY_HIGH'\n")

for cls_name in classes:
    mask = (
        (df_preds[true_col] == cls_name)
        & (df_preds[pred_col] == cls_name)
        & (df_preds[tier_col] == "VERY_HIGH")
    )
    matching_indices = df_preds.index[mask].tolist()
    print(f"--- Class: {cls_name} ({len(matching_indices)} matching rows) ---")
    sample_indices = matching_indices[:10]
    for idx in sample_indices:
        row = df_preds.iloc[idx]
        conf = float(row[conf_col])
        print(f"  Index {idx:>6}: conf={conf:.4f}")
    print()

