"""Data preparation module for extra attack classes: Bot, FTP-BruteForce, SSH-Bruteforce.

Reads raw CICIDS2018 CSVs, extracts only the three new classes, applies identical cleaning,
metadata standardization, deduplication, and capping rules, transforms features with the
existing frozen StandardScaler (transform only, never fit), and serializes parquet datasets
to data/v2/processed/.

Refuses to execute if the output directory points to or conflicts with data/processed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

from ids_rag.config import get_project_root


EXTRA_CLASS_NAMES = ["Bot", "FTP-BruteForce", "SSH-Bruteforce"]
EXTRA_CLASS_IDS = {"Bot": 3, "FTP-BruteForce": 4, "SSH-Bruteforce": 5}


def check_safety_paths(processed_dir: Path | str, project_root: Path | str) -> None:
    """Refuse to run if the output directory is or resides in data/processed."""
    root = Path(project_root).resolve()
    target = Path(processed_dir).resolve()
    forbidden = (root / "data" / "processed").resolve()

    if target == forbidden or forbidden in target.parents:
        raise ValueError(
            f"SAFETY REFUSAL: Target output directory '{target}' conflicts with protected "
            f"baseline directory '{forbidden}'. v2 extra class processing must never write into data/processed."
        )


def load_v2_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load config_v2.yaml from root or custom path."""
    root = get_project_root()
    path = Path(config_path) if config_path else root / "config_v2.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_extra_label_mapping(
    extra_classes_cfg: dict[str, list[str]],
    class_id_offset: int = 3,
) -> tuple[dict[str, str], dict[str, int], dict[int, str]]:
    """Build case-insensitive raw label to target class and integer ID mappings for extra classes."""
    raw_to_class: dict[str, str] = {}
    class_to_id: dict[str, int] = {}
    id_to_class: dict[int, str] = {}

    for idx, (target_class, raw_labels) in enumerate(extra_classes_cfg.items(), start=class_id_offset):
        class_to_id[target_class] = idx
        id_to_class[idx] = target_class
        for raw in raw_labels:
            cleaned_raw = str(raw).strip().lower()
            raw_to_class[cleaned_raw] = target_class

    return raw_to_class, class_to_id, id_to_class


def filter_and_map_extra_chunk(
    chunk: pd.DataFrame,
    raw_to_class: dict[str, str],
    class_to_id: dict[str, int],
) -> tuple[pd.DataFrame, int, int]:
    """Drop repeated header rows and filter chunk to include only extra classes."""
    if "Label" not in chunk.columns:
        return pd.DataFrame(), 0, len(chunk)

    # 1. Detect and drop repeated header rows
    label_str_series = chunk["Label"].astype(str).str.strip()
    is_header = label_str_series.str.lower() == "label"
    repeated_headers_count = int(is_header.sum())
    valid_chunk = chunk[~is_header].copy()

    if valid_chunk.empty:
        return valid_chunk, repeated_headers_count, 0

    # 2. Case-insensitive label mapping
    cleaned_labels = valid_chunk["Label"].astype(str).str.strip().str.lower()
    mapped_target = cleaned_labels.map(raw_to_class)

    # Keep only rows belonging to the extra classes
    is_mapped = mapped_target.notna()
    unmapped_count = int((~is_mapped).sum())
    filtered_chunk = valid_chunk[is_mapped].copy()

    if filtered_chunk.empty:
        return filtered_chunk, repeated_headers_count, unmapped_count

    filtered_chunk["mapped_class"] = mapped_target[is_mapped].values
    filtered_chunk["label"] = filtered_chunk["mapped_class"].map(class_to_id).astype(np.int32).values

    return filtered_chunk, repeated_headers_count, unmapped_count


def clean_numeric_features(
    chunk: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, int]:
    """Convert features to numeric, replace +/-Inf with NaN, and drop rows with NaN."""
    if chunk.empty:
        return chunk, 0

    initial_len = len(chunk)
    working_df = chunk.copy()

    for col in feature_cols:
        if col in working_df.columns:
            working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    working_df[feature_cols] = working_df[feature_cols].replace([np.inf, -np.inf], np.nan)
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

    for port_col in ["Dst Port", "Src Port"]:
        if port_col in chunk.columns:
            meta[port_col] = pd.to_numeric(chunk[port_col], errors="coerce").astype("Int64")
        else:
            meta[port_col] = pd.Series(pd.NA, index=chunk.index, dtype="Int64")

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


def apply_existing_scaler_transform(
    scaler: StandardScaler,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Transform splits using the existing frozen scaler without calling fit or fit_transform."""
    # Strict verification: scaler must never be fit on extra classes
    if not hasattr(scaler, "mean_") or scaler.mean_ is None:
        raise ValueError("Scaler must be pre-fitted before transform. Fitting on extra classes is strictly prohibited.")
    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(feature_names):
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features, but {len(feature_names)} were provided."
        )

    train_scaled = pd.DataFrame(
        scaler.transform(train_feat[feature_names]),
        columns=feature_names,
        dtype=np.float32,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_feat[feature_names]),
        columns=feature_names,
        dtype=np.float32,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_feat[feature_names]),
        columns=feature_names,
        dtype=np.float32,
    )
    return train_scaled, val_scaled, test_scaled


def prepare_extra_pipeline(config_path: Path | str | None = None) -> dict[str, Any]:
    """Execute streaming, cleaning, deduplication, capping, scaling, and saving for extra classes."""
    np.random.seed(42)
    root = get_project_root()
    cfg = load_v2_config(config_path)

    paths = cfg["paths"]
    raw_dir = root / paths.get("raw", "data/raw")
    processed_dir = root / paths.get("processed", "data/v2/processed")
    models_dir = root / paths.get("models", "models")
    scaler_path = root / paths.get("scaler", "models/scaler.joblib")

    # 1. Enforce safety check against overwriting data/processed
    check_safety_paths(processed_dir, root)

    processed_dir.mkdir(parents=True, exist_ok=True)

    preprocess_cfg = cfg["preprocess"]
    id_columns: list[str] = preprocess_cfg["id_columns"]
    extra_classes_cfg: dict[str, list[str]] = preprocess_cfg.get(
        "extra_classes",
        {
            "Bot": ["Bot"],
            "FTP-BruteForce": ["FTP-BruteForce"],
            "SSH-Bruteforce": ["SSH-Bruteforce"],
        },
    )
    attack_cap = int(preprocess_cfg.get("attack_cap", 300000))
    split_cfg = preprocess_cfg.get("split", {"train": 0.70, "val": 0.15, "test": 0.15})
    chunksize = int(cfg.get("inspect", {}).get("chunksize", 200000))

    # Features: exact 41 features from baseline
    feature_names = preprocess_cfg.get("features")
    if not feature_names:
        feature_names_path = root / paths.get("feature_names", "models/feature_names.json")
        if feature_names_path.exists():
            with open(feature_names_path, "r", encoding="utf-8") as f:
                feature_names = json.load(f)
        else:
            raise FileNotFoundError(f"Feature names list not found in config or at {feature_names_path}")

    raw_to_class, class_to_id, id_to_class = build_extra_label_mapping(
        extra_classes_cfg, class_id_offset=3
    )

    # Save extra class label map
    label_map_path = processed_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(id_to_class, f, indent=2)

    csv_files = sorted(list(raw_dir.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")

    total_initial_raw_rows = 0
    total_repeated_headers = 0
    total_unmapped_rows = 0
    total_nan_inf_rows = 0
    raw_class_counter: Counter[str] = Counter()

    collected_features: list[pd.DataFrame] = []
    collected_labels: list[pd.Series] = []
    collected_meta: list[pd.DataFrame] = []

    print("=================================================================")
    print("STARTING V2 EXTRA CLASSES DATA PREPARATION (STAGE 1)")
    print(f"Target extra classes: {list(class_to_id.keys())} -> IDs: {class_to_id}")
    print(f"Output directory: {processed_dir}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Attack cap per class: {attack_cap:,}")
    print("=================================================================\n")

    for file_idx, file_path in enumerate(csv_files):
        print(f"--- Streaming [{file_idx + 1}/{len(csv_files)}]: {file_path.name} ---")

        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
            total_initial_raw_rows += len(chunk)

            # 1. Drop repeated headers and map extra classes
            filtered_chunk, n_headers, n_unmapped = filter_and_map_extra_chunk(
                chunk, raw_to_class, class_to_id
            )
            total_repeated_headers += n_headers
            total_unmapped_rows += n_unmapped

            if filtered_chunk.empty:
                continue

            for cname in filtered_chunk["mapped_class"]:
                raw_class_counter[cname] += 1

            non_feature_names = set(id_columns) | {"Label", "mapped_class", "label"}
            chunk_feature_cols = [c for c in filtered_chunk.columns if c not in non_feature_names]

            # 2. Clean numeric features and drop NaN/Inf rows
            cleaned_chunk, n_nan_inf = clean_numeric_features(filtered_chunk, chunk_feature_cols)
            total_nan_inf_rows += n_nan_inf

            if cleaned_chunk.empty:
                continue

            # 3. Standardize metadata schema
            meta_chunk = standardize_metadata_chunk(cleaned_chunk, file_path.name)

            # 4. Extract required features aligned with 41 baseline features
            missing_cols = [c for c in feature_names if c not in cleaned_chunk.columns]
            if missing_cols:
                raise KeyError(f"Missing required baseline feature columns in {file_path.name}: {missing_cols}")

            feat_chunk = cleaned_chunk[feature_names].astype(np.float32)
            lbl_chunk = cleaned_chunk["label"].astype(np.int32)

            collected_features.append(feat_chunk)
            collected_labels.append(lbl_chunk)
            collected_meta.append(meta_chunk)

    if not collected_features:
        raise ValueError("No extra class rows were found across the raw CSV files.")

    print("\nConcatenating collected extra class chunks...")
    features_df = pd.concat(collected_features, axis=0, ignore_index=True)
    labels_s = pd.concat(collected_labels, axis=0, ignore_index=True)
    meta_df = pd.concat(collected_meta, axis=0, ignore_index=True)

    print(f"Rows before deduplication: {len(features_df):,}")
    clean_class_counter: Counter[str] = Counter(labels_s.map(id_to_class))

    # 4. Deduplicate across features and label
    combined_for_dup = features_df.copy()
    combined_for_dup["__target_label__"] = labels_s
    duplicate_mask = combined_for_dup.duplicated()
    total_duplicates = int(duplicate_mask.sum())

    features_df = features_df[~duplicate_mask].reset_index(drop=True)
    labels_s = labels_s[~duplicate_mask].reset_index(drop=True)
    meta_df = meta_df[~duplicate_mask].reset_index(drop=True)

    print(f"Dropped duplicate rows: {total_duplicates:,}")
    print(f"Rows after deduplication: {len(features_df):,}")
    dedup_class_counter: Counter[str] = Counter(labels_s.map(id_to_class))

    # 5. Capping each class at attack_cap with seed 42
    capped_indices: list[int] = []
    capping_pruned_records: dict[str, int] = {}
    capping_rng = np.random.default_rng(42)

    for class_name, class_id in class_to_id.items():
        matching_indices = np.where(labels_s == class_id)[0]
        class_count = len(matching_indices)

        if class_count > attack_cap:
            selected_idx = capping_rng.choice(matching_indices, size=attack_cap, replace=False)
            capped_indices.extend(selected_idx)
            capping_pruned_records[class_name] = class_count - attack_cap
            print(f"Class '{class_name}' capped from {class_count:,} to {attack_cap:,}")
        else:
            capped_indices.extend(matching_indices)
            capping_pruned_records[class_name] = 0
            print(f"Class '{class_name}' retained full count: {class_count:,}")

    capped_indices.sort()
    features_df = features_df.iloc[capped_indices].reset_index(drop=True)
    labels_s = labels_s.iloc[capped_indices].reset_index(drop=True)
    meta_df = meta_df.iloc[capped_indices].reset_index(drop=True)

    print(f"Rows after capping: {len(features_df):,}")
    final_class_counter: Counter[str] = Counter(labels_s.map(id_to_class))

    # 6. Stratified 70/15/15 split on label with seed 42
    indices = np.arange(len(features_df))
    train_frac = float(split_cfg.get("train", 0.70))
    val_frac = float(split_cfg.get("val", 0.15))
    test_frac = float(split_cfg.get("test", 0.15))

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

    # 7. Apply existing frozen scaler (transform only, never fit)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Existing scaler artifact not found at {scaler_path}")
    scaler: StandardScaler = joblib.load(scaler_path)

    train_scaled, val_scaled, test_scaled = apply_existing_scaler_transform(
        scaler, train_feat, val_feat, test_feat, feature_names
    )

    # 8. Assert metadata schemas
    assert_metadata_schema(train_meta)
    assert_metadata_schema(val_meta)
    assert_metadata_schema(test_meta)

    # 9. Add integer label column and save Parquet datasets
    train_scaled["label"] = train_lbl.astype(np.int32).values
    val_scaled["label"] = val_lbl.astype(np.int32).values
    test_scaled["label"] = test_lbl.astype(np.int32).values

    train_scaled.to_parquet(processed_dir / "train_extra.parquet", index=False)
    val_scaled.to_parquet(processed_dir / "val_extra.parquet", index=False)
    test_scaled.to_parquet(processed_dir / "test_extra.parquet", index=False)

    train_meta.to_parquet(processed_dir / "train_extra_meta.parquet", index=False)
    val_meta.to_parquet(processed_dir / "val_extra_meta.parquet", index=False)
    test_meta.to_parquet(processed_dir / "test_extra_meta.parquet", index=False)

    # 10. Compile and save extra preprocess report
    def get_split_counts(s: pd.Series) -> dict[str, int]:
        counts = Counter(s.values)
        return {id_to_class[cid]: counts.get(cid, 0) for cid in sorted(id_to_class.keys())}

    class_distribution = {
        "train": get_split_counts(train_lbl),
        "val": get_split_counts(val_lbl),
        "test": get_split_counts(test_lbl),
    }

    report = {
        "label_map": id_to_class,
        "feature_count": len(feature_names),
        "features": feature_names,
        "scaler_used": str(scaler_path),
        "class_counts_per_split": class_distribution,
        "rows_removed_per_step": {
            "total_initial_raw_rows": total_initial_raw_rows,
            "repeated_headers_removed": total_repeated_headers,
            "unmapped_rows_filtered": total_unmapped_rows,
            "nan_or_inf_rows_removed": total_nan_inf_rows,
            "duplicate_rows_removed": total_duplicates,
            "capping_pruned_per_class": capping_pruned_records,
            "final_total_rows": len(features_df),
        },
        "per_class_summary": {
            cname: {
                "raw_seen": raw_class_counter[cname],
                "after_clean": clean_class_counter[cname],
                "after_dedup": dedup_class_counter[cname],
                "final_count": final_class_counter[cname],
                "train_count": class_distribution["train"][cname],
                "val_count": class_distribution["val"][cname],
                "test_count": class_distribution["test"][cname],
            }
            for cname in id_to_class.values()
        },
    }

    report_path = processed_dir / "extra_preprocess_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=================================================================")
    print("V2 EXTRA CLASSES PREPARATION COMPLETE")
    print(f"Artifacts saved to: {processed_dir}")
    print(f"Scaler used (transform only): {scaler_path}")
    print("=================================================================\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare extra classes dataset for v2.")
    parser.add_argument("--config", type=str, default=None, help="Path to config_v2.yaml.")
    args = parser.parse_args()
    prepare_extra_pipeline(config_path=args.config)

