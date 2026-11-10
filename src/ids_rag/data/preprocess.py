"""Data cleaning, normalization, capping, and splitting module."""

from collections import Counter
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ids_rag.config import get_project_root, load_config


def build_label_mapping(
    label_map_cfg: dict[str, list[str]]
) -> tuple[dict[str, str], dict[str, int], dict[int, str]]:
    """Build case-insensitive raw label to target class and integer id mappings."""
    raw_to_class: dict[str, str] = {}
    class_to_id: dict[str, int] = {}
    id_to_class: dict[int, str] = {}

    for idx, (target_class, raw_labels) in enumerate(label_map_cfg.items()):
        class_to_id[target_class] = idx
        id_to_class[idx] = target_class
        for raw in raw_labels:
            cleaned_raw = str(raw).strip().lower()
            raw_to_class[cleaned_raw] = target_class

    return raw_to_class, class_to_id, id_to_class


def filter_and_map_chunk(
    chunk: pd.DataFrame,
    raw_to_class: dict[str, str],
    class_to_id: dict[str, int],
) -> tuple[pd.DataFrame, int, int]:
    """Filter repeated header rows, map labels case-insensitively, and drop unmapped rows."""
    if "Label" not in chunk.columns:
        return pd.DataFrame(), 0, len(chunk)

    # 1. Detect and drop repeated header rows where Label == 'Label'
    label_str_series = chunk["Label"].astype(str).str.strip()
    is_header = label_str_series.str.lower() == "label"
    repeated_headers_count = int(is_header.sum())
    valid_chunk = chunk[~is_header].copy()

    if valid_chunk.empty:
        return valid_chunk, repeated_headers_count, 0

    # 2. Case-insensitive label mapping
    cleaned_labels = valid_chunk["Label"].astype(str).str.strip().str.lower()
    mapped_target = cleaned_labels.map(raw_to_class)

    # Drop unmapped classes
    is_mapped = mapped_target.notna()
    unmapped_count = int((~is_mapped).sum())
    filtered_chunk = valid_chunk[is_mapped].copy()

    if filtered_chunk.empty:
        return filtered_chunk, repeated_headers_count, unmapped_count

    # Assign standardized class name and integer label id
    filtered_chunk["mapped_class"] = mapped_target[is_mapped].values
    filtered_chunk["label"] = filtered_chunk["mapped_class"].map(class_to_id).astype(np.int32).values

    return filtered_chunk, repeated_headers_count, unmapped_count


def clean_numeric_features(
    chunk: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, int]:
    """Convert feature columns to numeric, replace +/-Inf with NaN, and drop rows with NaN."""
    if chunk.empty:
        return chunk, 0

    initial_len = len(chunk)
    working_df = chunk.copy()

    # Convert features to numeric (coercing non-convertible to NaN)
    for col in feature_cols:
        if col in working_df.columns:
            working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    # Replace +/-Inf with NaN across feature columns
    working_df[feature_cols] = working_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN in any feature column
    valid_mask = working_df[feature_cols].notna().all(axis=1)
    dropped_nan_count = initial_len - int(valid_mask.sum())
    clean_df = working_df[valid_mask].copy()

    return clean_df, dropped_nan_count


def standardize_metadata_chunk(
    chunk: pd.DataFrame,
    source_file: str,
) -> pd.DataFrame:
    """Standardize metadata schema and dtypes for consistent parquet serialization."""
    meta = pd.DataFrame(index=chunk.index)

    # 1. Port columns -> Int64
    for port_col in ["Dst Port", "Src Port"]:
        if port_col in chunk.columns:
            meta[port_col] = pd.to_numeric(chunk[port_col], errors="coerce").astype("Int64")
        else:
            meta[port_col] = pd.Series(pd.NA, index=chunk.index, dtype="Int64")

    # 2. Text columns -> pandas string dtype
    for text_col in ["Timestamp", "Flow ID", "Src IP", "Dst IP"]:
        if text_col in chunk.columns:
            meta[text_col] = chunk[text_col].astype("string")
        else:
            meta[text_col] = pd.Series(pd.NA, index=chunk.index, dtype="string")

    meta["source_file"] = pd.Series(source_file, index=chunk.index, dtype="string")

    ordered_cols = ["Timestamp", "Dst Port", "Flow ID", "Src IP", "Src Port", "Dst IP", "source_file"]
    return meta[ordered_cols]


def assert_metadata_schema(meta_df: pd.DataFrame) -> None:
    """Assert standardized dtypes on metadata dataframe before parquet export."""
    expected_ports = ["Dst Port", "Src Port"]
    expected_strings = ["Timestamp", "Flow ID", "Src IP", "Dst IP", "source_file"]
    for col in expected_ports:
        assert meta_df[col].dtype == "Int64", f"Expected Int64 for {col}, got {meta_df[col].dtype}"
    for col in expected_strings:
        assert meta_df[col].dtype == "string", f"Expected string for {col}, got {meta_df[col].dtype}"


def select_features_on_train(
    train_features: pd.DataFrame,
    corr_threshold: float,
) -> tuple[list[str], list[dict[str, str]]]:
    """Select non-constant and non-redundant features strictly on train split to prevent leakage."""
    dropped_records: list[dict[str, str]] = []

    # 1. Identify constant columns (zero variance) on TRAIN split only
    constant_cols = [
        col for col in train_features.columns
        if train_features[col].std() == 0.0 or train_features[col].nunique() <= 1
    ]
    for col in constant_cols:
        dropped_records.append({
            "column": col,
            "reason": "constant_zero_variance_on_train"
        })

    remaining_cols = [col for col in train_features.columns if col not in constant_cols]

    # 2. Compute correlation matrix on TRAIN split only to prevent validation/test leakage
    corr_matrix = train_features[remaining_cols].corr().abs()

    # Drop one column from each pair with correlation above threshold
    to_drop_corr: set[str] = set()
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        col_i = cols[i]
        if col_i in to_drop_corr:
            continue
        for j in range(i + 1, len(cols)):
            col_j = cols[j]
            if col_j in to_drop_corr:
                continue
            if corr_matrix.at[col_i, col_j] > corr_threshold:
                to_drop_corr.add(col_j)
                dropped_records.append({
                    "column": col_j,
                    "reason": f"high_correlation_above_{corr_threshold}_with_{col_i}"
                })

    final_feature_cols = [col for col in remaining_cols if col not in to_drop_corr]
    return final_feature_cols, dropped_records


def fit_and_transform_scaler(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    test_features: pd.DataFrame,
    scaler_path: Path,
) -> tuple[StandardScaler, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler strictly on train split, save artifact, and transform all splits."""
    scaler = StandardScaler()
    # Scaler is fitted on train only to avoid data leakage
    scaler.fit(train_features)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    train_scaled = pd.DataFrame(
        scaler.transform(train_features),
        columns=train_features.columns,
        dtype=np.float32,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_features),
        columns=val_features.columns,
        dtype=np.float32,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_features),
        columns=test_features.columns,
        dtype=np.float32,
    )

    return scaler, train_scaled, val_scaled, test_scaled


def preprocess_pipeline() -> dict[str, Any]:
    """Execute end-to-end preprocessing, stratified splitting, scaling, and artifact saving."""
    # Seed for reproducibility
    np.random.seed(42)
    rng = np.random.default_rng(42)

    config = load_config()
    project_root = get_project_root()

    paths = config["paths"]
    raw_dir = project_root / paths["raw"]
    processed_dir = project_root / paths["processed"]
    models_dir = project_root / paths["models"]
    artifacts_dir = project_root / paths["artifacts"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    preprocess_cfg = config["preprocess"]
    id_columns: list[str] = preprocess_cfg["id_columns"]
    label_map_cfg: dict[str, list[str]] = preprocess_cfg["label_map"]
    benign_cap = int(preprocess_cfg["benign_cap"])
    attack_cap = int(preprocess_cfg["attack_cap"])
    corr_threshold = float(preprocess_cfg["corr_threshold"])
    split_cfg = preprocess_cfg["split"]
    chunksize = int(config.get("inspect", {}).get("chunksize", 200000))

    raw_to_class, class_to_id, id_to_class = build_label_mapping(label_map_cfg)

    # Save id-to-name mapping in data\processed\label_map.json
    label_map_path = processed_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(id_to_class, f, indent=2)

    # Compute Benign sampling fraction to keep total streamed Benign near 2x benign_cap
    inspection_file = artifacts_dir / "data_inspection.json"
    total_benign_inspected = 13484708
    if inspection_file.exists():
        try:
            with open(inspection_file, "r", encoding="utf-8") as f:
                insp_data = json.load(f)
                total_benign_inspected = insp_data.get("overall", {}).get(
                    "overall_label_counts", {}
                ).get("Benign", total_benign_inspected)
        except Exception:
            pass

    target_stream_benign = 2 * benign_cap
    benign_sampling_fraction = min(1.0, target_stream_benign / total_benign_inspected)

    csv_files = sorted(list(raw_dir.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")

    total_initial_raw_rows = 0
    total_repeated_headers = 0
    total_unmapped_rows = 0
    total_nan_inf_rows = 0
    total_benign_subsampled_out = 0

    collected_features_list: list[pd.DataFrame] = []
    collected_labels_list: list[pd.Series] = []
    collected_meta_list: list[pd.DataFrame] = []

    print("=================================================================")
    print("STARTING DATA PREPROCESSING (STAGE 1b)")
    print(f"Classes from config: {list(class_to_id.keys())}")
    print(f"Benign cap: {benign_cap:,} | Attack cap: {attack_cap:,}")
    print(f"Benign sampling fraction: {benign_sampling_fraction:.6f}")
    print("=================================================================\n")

    for file_idx, file_path in enumerate(csv_files):
        print(f"--- Streaming [{file_idx + 1}/{len(csv_files)}]: {file_path.name} ---")

        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
            total_initial_raw_rows += len(chunk)

            # 1. Drop repeated headers and map labels
            filtered_chunk, n_headers, n_unmapped = filter_and_map_chunk(
                chunk, raw_to_class, class_to_id
            )
            total_repeated_headers += n_headers
            total_unmapped_rows += n_unmapped

            if filtered_chunk.empty:
                continue

            # Identify feature columns (all columns except id_columns, Label, and added label helpers)
            non_feature_names = set(id_columns) | {"Label", "mapped_class", "label"}
            feature_cols = [c for c in filtered_chunk.columns if c not in non_feature_names]

            # 2. Clean numeric features and drop NaN/Inf rows
            cleaned_chunk, n_nan_inf = clean_numeric_features(filtered_chunk, feature_cols)
            total_nan_inf_rows += n_nan_inf

            if cleaned_chunk.empty:
                continue

            # 3. Keep all attack rows; subsample Benign rows with seeded random generator
            is_benign = cleaned_chunk["mapped_class"] == "Benign"
            benign_indices = cleaned_chunk[is_benign].index
            attack_indices = cleaned_chunk[~is_benign].index

            if len(benign_indices) > 0:
                random_draws = rng.random(len(benign_indices))
                kept_benign_mask = random_draws < benign_sampling_fraction
                kept_benign_indices = benign_indices[kept_benign_mask]
                total_benign_subsampled_out += (len(benign_indices) - len(kept_benign_indices))
            else:
                kept_benign_indices = pd.Index([])

            surviving_indices = attack_indices.union(kept_benign_indices).sort_values()
            surviving_chunk = cleaned_chunk.loc[surviving_indices]

            if surviving_chunk.empty:
                continue

            # 4. Standardize metadata schema and dtypes per chunk before collecting
            meta_chunk = standardize_metadata_chunk(surviving_chunk, file_path.name)

            # 5. Extract feature matrix and label series
            feat_chunk = surviving_chunk[feature_cols].astype(np.float32)
            lbl_chunk = surviving_chunk["label"].astype(np.int32)

            collected_features_list.append(feat_chunk)
            collected_labels_list.append(lbl_chunk)
            collected_meta_list.append(meta_chunk)

    print("\nConcatenating collected chunks...")
    features_df = pd.concat(collected_features_list, axis=0, ignore_index=True)
    labels_s = pd.concat(collected_labels_list, axis=0, ignore_index=True)
    meta_df = pd.concat(collected_meta_list, axis=0, ignore_index=True)

    print(f"Rows before deduplication: {len(features_df):,}")

    # 4. Drop duplicate rows (features plus label) while keeping metadata aligned
    combined_for_dup = features_df.copy()
    combined_for_dup["__target_label__"] = labels_s
    duplicate_mask = combined_for_dup.duplicated()
    total_duplicates = int(duplicate_mask.sum())

    features_df = features_df[~duplicate_mask].reset_index(drop=True)
    labels_s = labels_s[~duplicate_mask].reset_index(drop=True)
    meta_df = meta_df[~duplicate_mask].reset_index(drop=True)

    print(f"Dropped duplicate rows: {total_duplicates:,}")
    print(f"Rows after deduplication: {len(features_df):,}")

    # Capping each class at its cap with seed 42
    capped_indices: list[int] = []
    capping_pruned_records: dict[str, int] = {}
    capping_rng = np.random.default_rng(42)

    for class_name, class_id in class_to_id.items():
        matching_indices = np.where(labels_s == class_id)[0]
        class_count = len(matching_indices)
        cap = benign_cap if class_name == "Benign" else attack_cap

        if class_count > cap:
            selected_idx = capping_rng.choice(matching_indices, size=cap, replace=False)
            capped_indices.extend(selected_idx)
            capping_pruned_records[class_name] = class_count - cap
            print(f"Class '{class_name}' capped from {class_count:,} to {cap:,}")
        else:
            capped_indices.extend(matching_indices)
            capping_pruned_records[class_name] = 0
            print(f"Class '{class_name}' retained full count: {class_count:,}")

    capped_indices.sort()
    features_df = features_df.iloc[capped_indices].reset_index(drop=True)
    labels_s = labels_s.iloc[capped_indices].reset_index(drop=True)
    meta_df = meta_df.iloc[capped_indices].reset_index(drop=True)

    print(f"Rows after capping: {len(features_df):,}")

    # 5. Stratified 70/15/15 split on final label with seed 42
    indices = np.arange(len(features_df))
    train_frac = float(split_cfg["train"])
    val_frac = float(split_cfg["val"])
    test_frac = float(split_cfg["test"])

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_frac,
        random_state=42,
        stratify=labels_s.values,
    )

    val_relative_frac = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_relative_frac,
        random_state=42,
        stratify=labels_s.iloc[temp_idx].values,
    )

    train_feat = features_df.iloc[train_idx].reset_index(drop=True)
    val_feat = features_df.iloc[val_idx].reset_index(drop=True)
    test_feat = features_df.iloc[test_idx].reset_index(drop=True)

    train_lbl = labels_s.iloc[train_idx].reset_index(drop=True)
    val_lbl = labels_s.iloc[val_idx].reset_index(drop=True)
    test_lbl = labels_s.iloc[test_idx].reset_index(drop=True)

    train_meta = meta_df.iloc[train_idx].reset_index(drop=True)
    val_meta = meta_df.iloc[val_idx].reset_index(drop=True)
    test_meta = meta_df.iloc[test_idx].reset_index(drop=True)

    # 6. Feature selection strictly on TRAIN split (constant and correlation)
    final_features, dropped_feature_records = select_features_on_train(
        train_feat, corr_threshold
    )
    print(f"\nFeatures selected on train: {len(final_features)} / {len(features_df.columns)}")
    print(f"Dropped {len(dropped_feature_records)} features due to zero variance or high correlation.")

    # 7. Fit StandardScaler on train only, save artifact, and transform all splits
    scaler_path = models_dir / "scaler.joblib"
    _, train_scaled, val_scaled, test_scaled = fit_and_transform_scaler(
        train_feat[final_features],
        val_feat[final_features],
        test_feat[final_features],
        scaler_path,
    )

    # 8. Assert metadata schemas before Parquet serialization
    assert_metadata_schema(train_meta)
    assert_metadata_schema(val_meta)
    assert_metadata_schema(test_meta)

    # 9. Add integer label column and save Parquet files
    train_scaled["label"] = train_lbl.astype(np.int32).values
    val_scaled["label"] = val_lbl.astype(np.int32).values
    test_scaled["label"] = test_lbl.astype(np.int32).values

    train_scaled.to_parquet(processed_dir / "train.parquet", index=False)
    val_scaled.to_parquet(processed_dir / "val.parquet", index=False)
    test_scaled.to_parquet(processed_dir / "test.parquet", index=False)

    train_meta.to_parquet(processed_dir / "train_meta.parquet", index=False)
    val_meta.to_parquet(processed_dir / "val_meta.parquet", index=False)
    test_meta.to_parquet(processed_dir / "test_meta.parquet", index=False)

    # 10. Compile and save preprocess report
    def get_split_class_counts(s: pd.Series) -> dict[str, int]:
        counts = Counter(s.values)
        return {id_to_class[cid]: counts.get(cid, 0) for cid in sorted(id_to_class.keys())}

    class_distribution = {
        "train": get_split_class_counts(train_lbl),
        "val": get_split_class_counts(val_lbl),
        "test": get_split_class_counts(test_lbl),
    }

    all_dropped_cols = [
        {"column": col, "reason": "identifier_column_excluded_from_features"}
        for col in id_columns
    ] + dropped_feature_records

    preprocess_report = {
        "label_map": id_to_class,
        "class_counts_per_split": class_distribution,
        "final_feature_count": len(final_features),
        "final_features": final_features,
        "dropped_columns": all_dropped_cols,
        "rows_removed_per_step": {
            "total_initial_raw_rows": total_initial_raw_rows,
            "repeated_headers_removed": total_repeated_headers,
            "unmapped_label_rows_removed": total_unmapped_rows,
            "nan_or_inf_rows_removed": total_nan_inf_rows,
            "benign_subsampling_pruned": total_benign_subsampled_out,
            "duplicate_rows_removed": total_duplicates,
            "capping_pruned_per_class": capping_pruned_records,
            "final_total_rows": len(features_df),
        },
    }

    report_path = processed_dir / "preprocess_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(preprocess_report, f, indent=2)

    print("\n=================================================================")
    print("PREPROCESSING COMPLETE")
    print("=================================================================")
    print(f"Final feature count: {len(final_features)}")
    print("Class distribution per split:")
    for split_name, dist in class_distribution.items():
        print(f"  {split_name.upper()} ({sum(dist.values()):,} total rows):")
        for cname, count in dist.items():
            print(f"    {cname}: {count:,}")
    print(f"\nArtifacts saved to: {processed_dir}")
    print(f"Scaler saved to: {scaler_path}")
    print("=================================================================")

    return preprocess_report


if __name__ == "__main__":
    preprocess_pipeline()
