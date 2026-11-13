"""Audit script for v2 extra classes (Bot, FTP-BruteForce, SSH-Bruteforce).

Reads data/v2/processed/extra_preprocess_report.json and the generated parquet datasets,
printing per new class:
- Raw rows seen
- Rows after numeric/header cleaning
- Rows after deduplication
- Final capped count
- Stratified split counts (Train 70%, Val 15%, Test 15%)

Read-only / printing only. Does not alter any files or models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ids_rag.config import get_project_root


def audit_extra_classes(processed_dir: Path | str | None = None) -> None:
    root = get_project_root()
    pdir = Path(processed_dir) if processed_dir else root / "data" / "v2" / "processed"

    report_file = pdir / "extra_preprocess_report.json"
    if not report_file.exists():
        print(f"STOP: report file not found at {report_file}", file=sys.stderr)
        print("Please run `python -m ids_rag.v2.prepare_extra` first.", file=sys.stderr)
        sys.exit(1)

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_class = data.get("per_class_summary", {})
    splits = data.get("class_counts_per_split", {})
    rem_steps = data.get("rows_removed_per_step", {})

    print("\n" + "=" * 94)
    print("V2 EXTRA CLASSES AUDIT REPORT (Bot, FTP-BruteForce, SSH-Bruteforce)")
    print("=" * 94)
    header = (
        f"{'Class':<18} {'Raw Rows':<12} {'After Clean':<13} {'After DUP':<12} "
        f"{'Final (Cap)':<13} {'Train (70%)':<13} {'Val (15%)':<11} {'Test (15%)':<11}"
    )
    print(header)
    print("-" * 94)

    totals = {
        "raw": 0,
        "clean": 0,
        "dedup": 0,
        "final": 0,
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for cname, stats in per_class.items():
        raw_val = stats.get("raw_seen", 0)
        clean_val = stats.get("after_clean", 0)
        dedup_val = stats.get("after_dedup", 0)
        final_val = stats.get("final_count", 0)
        train_val = splits.get("train", {}).get(cname, 0)
        val_val = splits.get("val", {}).get(cname, 0)
        test_val = splits.get("test", {}).get(cname, 0)

        totals["raw"] += raw_val
        totals["clean"] += clean_val
        totals["dedup"] += dedup_val
        totals["final"] += final_val
        totals["train"] += train_val
        totals["val"] += val_val
        totals["test"] += test_val

        print(
            f"{cname:<18} {raw_val:<12,d} {clean_val:<13,d} {dedup_val:<12,d} "
            f"{final_val:<13,d} {train_val:<13,d} {val_val:<11,d} {test_val:<11,d}"
        )

    print("-" * 94)
    print(
        f"{'TOTAL':<18} {totals['raw']:<12,d} {totals['clean']:<13,d} {totals['dedup']:<12,d} "
        f"{totals['final']:<13,d} {totals['train']:<13,d} {totals['val']:<11,d} {totals['test']:<11,d}"
    )
    print("=" * 94)

    print("\nRemoval & Filtering Step Summary:")
    print(f"  - Initial raw rows processed:    {rem_steps.get('total_initial_raw_rows', 0):,d}")
    print(f"  - Repeated header rows dropped:  {rem_steps.get('repeated_headers_removed', 0):,d}")
    print(f"  - Unmapped / non-target rows:    {rem_steps.get('unmapped_rows_filtered', 0):,d}")
    print(f"  - Rows with NaN or +/-Inf:       {rem_steps.get('nan_or_inf_rows_removed', 0):,d}")
    print(f"  - Duplicate rows removed:        {rem_steps.get('duplicate_rows_removed', 0):,d}")
    capping_pruned = rem_steps.get("capping_pruned_per_class", {})
    if any(v > 0 for v in capping_pruned.values()):
        print(f"  - Capping pruned per class:      {capping_pruned}")
    else:
        print("  - Capping pruned:                0 (all extra classes within attack_cap)")

    print("\nParquet Dataset Verification:")
    for split_name in ("train", "val", "test"):
        pq_path = pdir / f"{split_name}_extra.parquet"
        pq_meta = pdir / f"{split_name}_extra_meta.parquet"
        if pq_path.exists() and pq_meta.exists():
            import pyarrow.parquet as pq
            table = pq.read_table(pq_path)
            meta_table = pq.read_table(pq_meta)
            print(f"  - {split_name}_extra: {table.num_rows:,d} rows, {table.num_columns} cols | meta: {meta_table.num_rows:,d} rows")
        else:
            print(f"  - {split_name}_extra: MISSING file(s)")

    print(f"\nScaler Used: {data.get('scaler_used', 'N/A')}")
    print(f"Feature count: {data.get('feature_count', 0)}")
    print("Audit completed successfully (read-only).\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit v2 extra classes datasets.")
    parser.add_argument("--dir", type=str, default=None, help="Path to data/v2/processed directory.")
    args = parser.parse_args()
    audit_extra_classes(processed_dir=args.dir)

