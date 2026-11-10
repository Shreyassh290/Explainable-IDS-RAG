"""Unit tests for data preprocessing module using synthetic data."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ids_rag.data.preprocess import (
    assert_metadata_schema,
    build_label_mapping,
    filter_and_map_chunk,
    fit_and_transform_scaler,
    standardize_metadata_chunk,
)


def test_label_mapping_case_insensitive_and_drop_unmapped() -> None:
    """Verify label mapping matches case-insensitively and drops unmapped classes."""
    label_map_cfg = {
        "Benign": ["Benign"],
        "DoS": ["DoS attacks-Hulk", "DoS attacks-Slowloris"],
        "DDoS": ["DDOS attack-HOIC"],
    }
    raw_to_class, class_to_id, _ = build_label_mapping(label_map_cfg)

    synthetic_df = pd.DataFrame({
        "Feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Label": [
            "benign",               # Lowercase match -> Benign (id 0)
            "DOS ATTACKS-HULK",     # Uppercase match -> DoS (id 1)
            "ddos attack-hoic",     # Mixed/lower match -> DDoS (id 2)
            "UnknownAttack",        # Unmapped -> should be dropped
            "Infilteration",        # Unmapped -> should be dropped
        ],
    })

    filtered_chunk, _, unmapped_count = filter_and_map_chunk(
        synthetic_df, raw_to_class, class_to_id
    )

    assert unmapped_count == 2
    assert len(filtered_chunk) == 3
    assert list(filtered_chunk["mapped_class"]) == ["Benign", "DoS", "DDoS"]
    assert list(filtered_chunk["label"]) == [0, 1, 2]


def test_header_rows_removed() -> None:
    """Verify repeated header rows where Label == 'Label' are removed."""
    label_map_cfg = {
        "Benign": ["Benign"],
    }
    raw_to_class, class_to_id, _ = build_label_mapping(label_map_cfg)

    synthetic_df = pd.DataFrame({
        "Feature1": [10.0, 20.0, 30.0, 40.0],
        "Label": ["Benign", "Label", "LABEL", "Benign"],
    })

    filtered_chunk, headers_count, _ = filter_and_map_chunk(
        synthetic_df, raw_to_class, class_to_id
    )

    assert headers_count == 2
    assert len(filtered_chunk) == 2
    assert (filtered_chunk["mapped_class"] == "Benign").all()


def test_scaler_fitted_on_train_only(tmp_path: Path) -> None:
    """Verify StandardScaler is strictly fitted on train split without leakage."""
    # Train has mean 10.0
    train_feat = pd.DataFrame({
        "feat_a": [8.0, 10.0, 12.0],
        "feat_b": [6.0, 10.0, 14.0],
    })
    # Val and Test have drastically different distribution (mean 100.0)
    val_feat = pd.DataFrame({
        "feat_a": [100.0, 100.0],
        "feat_b": [100.0, 100.0],
    })
    test_feat = pd.DataFrame({
        "feat_a": [200.0],
        "feat_b": [200.0],
    })

    scaler_file = tmp_path / "scaler.joblib"
    scaler, train_scaled, val_scaled, _ = fit_and_transform_scaler(
        train_feat, val_feat, test_feat, scaler_file
    )

    # Scaler mean must match train mean (10.0), not pooled data
    np.testing.assert_allclose(scaler.mean_, np.array([10.0, 10.0]))

    # Train scaled mean must be 0
    np.testing.assert_allclose(train_scaled.mean().values, np.zeros(2), atol=1e-6)

    # Val scaled values must reflect normalization using train parameters
    assert (val_scaled > 10.0).all().all()
    assert scaler_file.exists()


def test_metadata_mixed_types_and_missing_columns(tmp_path: Path) -> None:
    """Verify metadata standardization handles mixed port types and missing columns for Parquet."""
    # Chunk 1: Dst Port is int, has Flow ID, missing Src IP
    chunk_1 = pd.DataFrame({
        "Dst Port": [80, 443],
        "Src Port": [50000, 50001],
        "Timestamp": ["01/01/2018 08:00:00", "01/01/2018 08:00:01"],
        "Flow ID": ["flow_1", "flow_2"],
    })
    meta_1 = standardize_metadata_chunk(chunk_1, "file_1.csv")

    # Chunk 2: Dst Port is string '80', missing Flow ID, has Src IP
    chunk_2 = pd.DataFrame({
        "Dst Port": ["80", "8080"],
        "Src Port": ["60000", "60001"],
        "Timestamp": ["02/01/2018 09:00:00", "02/01/2018 09:00:01"],
        "Src IP": ["192.168.1.10", "192.168.1.11"],
    })
    meta_2 = standardize_metadata_chunk(chunk_2, "file_2.csv")

    concatenated_meta = pd.concat([meta_1, meta_2], axis=0, ignore_index=True)

    # Assert schema validation
    assert_metadata_schema(concatenated_meta)
    assert len(concatenated_meta) == 4
    assert concatenated_meta["Dst Port"].tolist() == [80, 443, 80, 8080]

    # Verify Parquet serialization and reload
    parquet_out = tmp_path / "meta_test.parquet"
    concatenated_meta.to_parquet(parquet_out, index=False)
    reloaded_meta = pd.read_parquet(parquet_out)

    assert len(reloaded_meta) == 4
    assert pd.isna(reloaded_meta["Flow ID"].iloc[2])  # Missing value in chunk 2 preserved as null
    assert pd.isna(reloaded_meta["Src IP"].iloc[0])    # Missing value in chunk 1 preserved as null
    assert reloaded_meta["source_file"].tolist() == ["file_1.csv", "file_1.csv", "file_2.csv", "file_2.csv"]
