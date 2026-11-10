"""Dataset inspection and schema profiling module."""

from collections import Counter
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from ids_rag.config import get_project_root, load_config


def identify_identifier_columns(columns: list[str]) -> list[str]:
    """Identify candidate identifier columns based on naming patterns."""
    id_keywords = ["id", "ip", "port", "timestamp", "time", "date", "mac", "flow"]
    candidates: list[str] = []
    for col in columns:
        cleaned = col.strip().lower()
        if any(keyword in cleaned for keyword in id_keywords):
            candidates.append(col)
    return candidates


def inspect_dataset() -> dict[str, Any]:
    """Inspect all raw CSV datasets in chunks and compile diagnostic metrics."""
    # Seed for deterministic execution
    np.random.seed(42)

    config = load_config()
    project_root = get_project_root()

    raw_path_str = config["paths"]["raw"]
    raw_dir = Path(raw_path_str)
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    artifacts_path_str = config["paths"]["artifacts"]
    artifacts_dir = Path(artifacts_path_str)
    if not artifacts_dir.is_absolute():
        artifacts_dir = project_root / artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    chunksize = int(config.get("inspect", {}).get("chunksize", 200000))

    csv_files = sorted(list(raw_dir.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    first_file_cols: list[str] = []
    overall_label_counts: Counter[str] = Counter()
    total_dataset_rows = 0
    total_dataset_repeated_headers = 0
    total_dataset_nan = 0
    total_dataset_pos_inf = 0
    total_dataset_neg_inf = 0

    inspected_files: list[dict[str, Any]] = []

    print("=================================================================")
    print("STARTING DATASET INSPECTION")
    print(f"Directory: {raw_dir}")
    print(f"Chunksize: {chunksize}")
    print(f"Files found: {len(csv_files)}")
    print("=================================================================\n")

    for file_idx, file_path in enumerate(csv_files):
        file_name = file_path.name
        print(f"--- Processing [{file_idx + 1}/{len(csv_files)}]: {file_name} ---")

        file_row_count = 0
        file_repeated_headers = 0
        file_nan_count = 0
        file_pos_inf_count = 0
        file_neg_inf_count = 0
        file_label_counts: Counter[str] = Counter()
        file_dtypes: dict[str, str] = {}
        file_cols: list[str] = []

        # Read CSV chunk by chunk
        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
            if file_row_count == 0:
                file_cols = list(chunk.columns)
                file_dtypes = {col: str(dtype) for col, dtype in chunk.dtypes.items()}
                if file_idx == 0:
                    first_file_cols = list(file_cols)

            chunk_len = len(chunk)
            file_row_count += chunk_len

            # Check if Label column exists
            if "Label" in chunk.columns:
                # Count repeated header rows where Label matches column names
                repeated_header_mask = chunk["Label"].isin(chunk.columns)
                repeated_headers_in_chunk = int(repeated_header_mask.sum())
                file_repeated_headers += repeated_headers_in_chunk

                # Accumulate label counts
                for lbl, count in chunk["Label"].value_counts(dropna=False).items():
                    lbl_str = str(lbl)
                    file_label_counts[lbl_str] += count
                    overall_label_counts[lbl_str] += count

            # Count NaN and +/-Inf across numeric columns only
            numeric_df = chunk.select_dtypes(include=[np.number])
            file_nan_count += int(numeric_df.isna().sum().sum())
            file_pos_inf_count += int((numeric_df == np.inf).sum().sum())
            file_neg_inf_count += int((numeric_df == -np.inf).sum().sum())

        # Column comparison against first file
        extra_cols = [c for c in file_cols if c not in first_file_cols]
        missing_cols = [c for c in first_file_cols if c not in file_cols]
        identical_to_first = (file_cols == first_file_cols)

        total_dataset_rows += file_row_count
        total_dataset_repeated_headers += file_repeated_headers
        total_dataset_nan += file_nan_count
        total_dataset_pos_inf += file_pos_inf_count
        total_dataset_neg_inf += file_neg_inf_count

        file_summary = {
            "file_name": file_name,
            "row_count": file_row_count,
            "column_count": len(file_cols),
            "columns": file_cols,
            "dtypes": file_dtypes,
            "repeated_header_rows": file_repeated_headers,
            "nan_numeric_count": file_nan_count,
            "pos_inf_numeric_count": file_pos_inf_count,
            "neg_inf_numeric_count": file_neg_inf_count,
            "label_counts": dict(file_label_counts),
            "column_diff_vs_first": {
                "identical": identical_to_first,
                "extra_columns": extra_cols,
                "missing_columns": missing_cols,
            },
        }
        inspected_files.append(file_summary)

        print(f"  Rows: {file_row_count:,}")
        print(f"  Columns: {len(file_cols)}")
        print(f"  Repeated header rows: {file_repeated_headers}")
        print(f"  Numeric NaN count: {file_nan_count:,}")
        print(f"  Numeric +Inf count: {file_pos_inf_count:,}")
        print(f"  Numeric -Inf count: {file_neg_inf_count:,}")
        if not identical_to_first:
            print(f"  Column differences vs first file:")
            if extra_cols:
                print(f"    Extra columns: {extra_cols}")
            if missing_cols:
                print(f"    Missing columns: {missing_cols}")
        print(f"  Label distribution: {dict(file_label_counts)}\n")

    # Determine candidate identifier columns across all unique columns
    all_unique_columns: list[str] = []
    for f in inspected_files:
        for c in f["columns"]:
            if c not in all_unique_columns:
                all_unique_columns.append(c)
    candidate_id_cols = identify_identifier_columns(all_unique_columns)

    overall_summary = {
        "total_files": len(csv_files),
        "total_rows": total_dataset_rows,
        "total_repeated_headers": total_dataset_repeated_headers,
        "total_nan_numeric": total_dataset_nan,
        "total_pos_inf_numeric": total_dataset_pos_inf,
        "total_neg_inf_numeric": total_dataset_neg_inf,
        "overall_label_counts": dict(overall_label_counts),
        "candidate_id_columns": candidate_id_cols,
    }

    full_inspection_data = {
        "files": inspected_files,
        "overall": overall_summary,
    }

    output_file = artifacts_dir / "data_inspection.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_inspection_data, f, indent=2)

    print("=================================================================")
    print("INSPECTION SUMMARY")
    print("=================================================================")
    print(f"Total files: {len(csv_files)}")
    print(f"Total rows: {total_dataset_rows:,}")
    print(f"Total repeated header rows: {total_dataset_repeated_headers}")
    print(f"Total numeric NaNs: {total_dataset_nan:,}")
    print(f"Total numeric +Infs: {total_dataset_pos_inf:,}")
    print(f"Total numeric -Infs: {total_dataset_neg_inf:,}")
    print("\nOverall Label Value Counts:")
    for lbl, count in overall_label_counts.most_common():
        print(f"  {lbl}: {count:,}")
    print(f"\nCandidate Identifier Columns: {candidate_id_cols}")
    print(f"Diagnostics saved to: {output_file}")
    print("=================================================================")

    return full_inspection_data


if __name__ == "__main__":
    inspect_dataset()
