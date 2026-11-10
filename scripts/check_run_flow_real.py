"""Verification script: test run_flow against real models and test_predictions.parquet.

Loads the real PipelineContext, takes 20 test rows (the first 20 by index, plus 5
rows where artifacts/test_predictions.parquet has a misclassification), feeds
scaler.inverse_transform(row) into run_flow with generate=False and a temp out_dir,
and verifies predicted class and confidence (within 1e-4) against saved predictions.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import load_config
from ids_rag.pipeline import PipelineContext, run_flow


def main() -> None:
    print("Loading real PipelineContext...", flush=True)
    cfg = load_config()
    ctx = PipelineContext.load(cfg)

    df_preds = ctx.df_preds
    if df_preds is None:
        preds_path = repo_root / "artifacts" / "test_predictions.parquet"
        if preds_path.exists():
            df_preds = pd.read_parquet(preds_path)
        else:
            print(f"ERROR: test_predictions.parquet not found at {preds_path}", file=sys.stderr)
            sys.exit(1)

    # Read column names from code dynamically - do not guess
    cols = list(df_preds.columns)
    print(f"test_predictions.parquet columns ({len(cols)}): {cols}\n", flush=True)

    # Identify predicted class column
    if "predicted_class" in cols:
        pred_class_col = "predicted_class"
    elif "predicted_label" in cols:
        pred_class_col = "predicted_label"
    else:
        print(f"ERROR: Unable to find predicted class column in {cols}", file=sys.stderr)
        sys.exit(1)

    # Identify confidence column
    if "confidence" in cols:
        conf_col = "confidence"
    else:
        print(f"ERROR: Unable to find confidence column in {cols}", file=sys.stderr)
        sys.exit(1)

    # Identify ground truth / true class column if available
    true_class_col: str | None = None
    if "true_class" in cols:
        true_class_col = "true_class"
    elif "true_label" in cols:
        true_class_col = "true_label"

    print(
        f"Using columns: predicted_class='{pred_class_col}', confidence='{conf_col}', "
        f"true_class='{true_class_col}'",
        flush=True,
    )

    # Select test rows:
    # 1. First 20 rows by index
    selected_indices: list[int] = list(range(min(20, len(ctx.df_test))))

    # 2. Plus up to 5 rows where predictions had a misclassification
    misclass_indices: list[int] = []
    if true_class_col is not None:
        misclass_mask = df_preds[true_class_col] != df_preds[pred_class_col]
        pos_indices = np.where(misclass_mask)[0]
        for pos in pos_indices:
            idx = int(pos)
            if idx not in selected_indices:
                misclass_indices.append(idx)
            if len(misclass_indices) >= 5:
                break
        print(
            f"Found {len(pos_indices)} misclassified rows in test_predictions.parquet; "
            f"selected 5 outside first 20: {misclass_indices}\n",
            flush=True,
        )
    else:
        print("Note: True class column not available to identify misclassifications.\n", flush=True)

    target_indices = selected_indices + misclass_indices
    print(f"Evaluating {len(target_indices)} rows total with run_flow (generate=False)...\n", flush=True)

    matches = 0
    total = len(target_indices)

    # Use a system temporary directory to guarantee artifacts/ is untouched
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_out = Path(tmp_dir)

        for idx in target_indices:
            # 1. Extract scaled test row features
            row_series = ctx.df_test.iloc[idx]
            x_scaled = row_series[ctx.feature_names].to_numpy(dtype=np.float32)

            # 2. Invert scaling to obtain raw unscaled features
            x_raw = ctx.scaler.inverse_transform(x_scaled.reshape(1, -1))[0]
            raw_dict = dict(zip(ctx.feature_names, x_raw))

            # 3. Run flow through public run_flow entry point
            summary = run_flow(
                raw_features=raw_dict,
                ctx=ctx,
                cfg=cfg,
                flow_id=idx,
                modes="rag",
                generate=False,
                out_dir=temp_out,
            )

            # 4. Compare with saved predictions in test_predictions.parquet
            saved_pred_class = str(df_preds.iloc[idx][pred_class_col])
            saved_conf = float(df_preds.iloc[idx][conf_col])

            flow_pred_class = str(summary["predicted_class"])
            flow_conf = float(summary["confidence"])

            class_matched = flow_pred_class == saved_pred_class
            conf_diff = abs(flow_conf - saved_conf)
            conf_matched = conf_diff < 1e-4

            row_matched = class_matched and conf_matched
            if row_matched:
                matches += 1

            status = "MATCH" if row_matched else "MISMATCH"
            print(
                f"Row {idx:>6} [{status}]: "
                f"saved=({saved_pred_class}, {saved_conf:.4f}) | "
                f"run_flow=({flow_pred_class}, {flow_conf:.4f}) | "
                f"diff={conf_diff:.2e}",
                flush=True,
            )

    print(f"\n==========================================", flush=True)
    print(f"Final Results: {matches}/{total} matched (class equal and confidence within 1e-4)", flush=True)
    print(f"==========================================", flush=True)

    if matches != total:
        sys.exit(1)


if __name__ == "__main__":
    main()

