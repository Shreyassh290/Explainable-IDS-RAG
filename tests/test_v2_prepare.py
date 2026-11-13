"""Unit tests for src/ids_rag/v2/prepare_extra.py (Stage 1 extra-class data preparation).

All tests use synthetic in-memory data only. No raw CSVs, no model files,
no data/processed, and no data/v2/processed are read or written.

Run:
    uv run pytest tests/test_v2_prepare.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ids_rag.v2.prepare_extra import (
    EXTRA_CLASS_IDS,
    EXTRA_CLASS_NAMES,
    apply_existing_scaler_transform,
    assert_metadata_schema,
    build_extra_label_mapping,
    check_safety_paths,
    clean_numeric_features,
    filter_and_map_extra_chunk,
    standardize_metadata_chunk,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURES_41 = [
    "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Fwd IAT Std", "Bwd IAT Tot", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
    "Fwd URG Flags", "Bwd Pkts/s", "Pkt Len Min", "Pkt Len Mean",
    "Pkt Len Var", "FIN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "Down/Up Ratio", "Init Fwd Win Byts",
    "Init Bwd Win Byts", "Fwd Seg Size Min", "Active Mean", "Active Std",
    "Active Min", "Idle Mean", "Idle Std",
]

EXTRA_CLASSES_CFG = {
    "Bot": ["Bot"],
    "FTP-BruteForce": ["FTP-BruteForce"],
    "SSH-Bruteforce": ["SSH-Bruteforce"],
}


def _make_feature_df(n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Return a synthetic DataFrame with the 41 baseline features."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.random((n_rows, len(FEATURES_41))).astype(np.float32),
        columns=FEATURES_41,
    )


def _make_fitted_scaler() -> StandardScaler:
    """Return a StandardScaler fitted on 41 synthetic columns."""
    rng = np.random.default_rng(99)
    data = rng.random((200, len(FEATURES_41))).astype(np.float32)
    sc = StandardScaler()
    sc.fit(data)
    return sc


# ---------------------------------------------------------------------------
# 1. Extra label mapping
# ---------------------------------------------------------------------------

class TestExtraLabelMapping:
    """build_extra_label_mapping maps raw labels to class names and IDs 3-5."""

    def test_class_ids_are_3_4_5(self):
        raw_to_class, class_to_id, id_to_class = build_extra_label_mapping(EXTRA_CLASSES_CFG)
        assert class_to_id["Bot"] == 3
        assert class_to_id["FTP-BruteForce"] == 4
        assert class_to_id["SSH-Bruteforce"] == 5

    def test_id_to_class_is_inverse(self):
        _, class_to_id, id_to_class = build_extra_label_mapping(EXTRA_CLASSES_CFG)
        for name, cid in class_to_id.items():
            assert id_to_class[cid] == name

    def test_raw_to_class_is_case_insensitive(self):
        """Keys in raw_to_class must be lowercase so label matching is case-insensitive."""
        raw_to_class, _, _ = build_extra_label_mapping(EXTRA_CLASSES_CFG)
        # All keys must be lowercase
        for key in raw_to_class:
            assert key == key.lower(), f"raw_to_class key not lowercase: '{key}'"
        # Canonical lowered forms must be present
        assert "bot" in raw_to_class
        assert "ftp-bruteforce" in raw_to_class
        assert "ssh-bruteforce" in raw_to_class

    def test_unmapped_labels_absent(self):
        """Benign, DoS, DDoS raw labels must NOT appear in raw_to_class."""
        raw_to_class, _, _ = build_extra_label_mapping(EXTRA_CLASSES_CFG)
        for benign_or_dos in ["benign", "dos attacks-hulk", "ddos attack-hoic"]:
            assert benign_or_dos not in raw_to_class

    def test_custom_offset(self):
        raw_to_class, class_to_id, id_to_class = build_extra_label_mapping(
            EXTRA_CLASSES_CFG, class_id_offset=10
        )
        assert class_to_id["Bot"] == 10
        assert class_to_id["FTP-BruteForce"] == 11
        assert class_to_id["SSH-Bruteforce"] == 12

    def test_module_level_constants_match_mapping(self):
        """EXTRA_CLASS_IDS constant must equal the mapping produced at offset 3."""
        _, class_to_id, _ = build_extra_label_mapping(EXTRA_CLASSES_CFG)
        assert class_to_id == EXTRA_CLASS_IDS

    def test_extra_class_names_constant(self):
        assert set(EXTRA_CLASS_NAMES) == {"Bot", "FTP-BruteForce", "SSH-Bruteforce"}


# ---------------------------------------------------------------------------
# 2. Safety path check
# ---------------------------------------------------------------------------

class TestCheckSafetyPaths:
    """check_safety_paths raises ValueError when target is data/processed."""

    def test_exact_forbidden_path_raises(self, tmp_path):
        project_root = tmp_path / "project"
        forbidden = project_root / "data" / "processed"
        with pytest.raises(ValueError, match="SAFETY REFUSAL"):
            check_safety_paths(forbidden, project_root)

    def test_subdirectory_of_forbidden_raises(self, tmp_path):
        project_root = tmp_path / "project"
        target = project_root / "data" / "processed" / "extra"
        with pytest.raises(ValueError, match="SAFETY REFUSAL"):
            check_safety_paths(target, project_root)

    def test_allowed_v2_path_does_not_raise(self, tmp_path):
        project_root = tmp_path / "project"
        allowed = project_root / "data" / "v2" / "processed"
        # Must not raise
        check_safety_paths(allowed, project_root)

    def test_sibling_directory_does_not_raise(self, tmp_path):
        project_root = tmp_path / "project"
        sibling = project_root / "data" / "processed_v2"
        # Must not raise (processed_v2 is not data/processed)
        check_safety_paths(sibling, project_root)

    def test_error_message_contains_both_paths(self, tmp_path):
        project_root = tmp_path / "project"
        forbidden = project_root / "data" / "processed"
        with pytest.raises(ValueError) as exc_info:
            check_safety_paths(forbidden, project_root)
        msg = str(exc_info.value)
        assert "data" in msg.lower() and "processed" in msg.lower()


# ---------------------------------------------------------------------------
# 3. Scaler is never fitted
# ---------------------------------------------------------------------------

class TestScalerNeverFitted:
    """apply_existing_scaler_transform must only call transform(), never fit/fit_transform."""

    def test_transform_is_called_on_each_split(self):
        scaler = _make_fitted_scaler()
        train_df = _make_feature_df(30, seed=1)
        val_df = _make_feature_df(10, seed=2)
        test_df = _make_feature_df(10, seed=3)

        with patch.object(scaler, "transform", wraps=scaler.transform) as mock_transform, \
             patch.object(scaler, "fit", side_effect=AssertionError("fit must NOT be called")) as mock_fit, \
             patch.object(scaler, "fit_transform", side_effect=AssertionError("fit_transform must NOT be called")) as mock_fit_transform:

            train_s, val_s, test_s = apply_existing_scaler_transform(
                scaler, train_df, val_df, test_df, FEATURES_41
            )
            assert mock_transform.call_count == 3, "transform must be called exactly 3 times (train, val, test)"

    def test_fit_is_never_invoked(self):
        """Directly verify that calling the function does not invoke fit."""
        scaler = _make_fitted_scaler()
        train_df = _make_feature_df(20, seed=10)
        val_df = _make_feature_df(5, seed=11)
        test_df = _make_feature_df(5, seed=12)

        fit_called = []
        original_fit = scaler.fit

        def spy_fit(*args, **kwargs):
            fit_called.append(True)
            return original_fit(*args, **kwargs)

        scaler.fit = spy_fit  # type: ignore[assignment]

        apply_existing_scaler_transform(scaler, train_df, val_df, test_df, FEATURES_41)
        assert len(fit_called) == 0, "fit() was called when it must not be"

    def test_fit_transform_is_never_invoked(self):
        scaler = _make_fitted_scaler()
        train_df = _make_feature_df(20, seed=20)
        val_df = _make_feature_df(5, seed=21)
        test_df = _make_feature_df(5, seed=22)

        ft_called = []
        original_ft = scaler.fit_transform

        def spy_ft(*args, **kwargs):
            ft_called.append(True)
            return original_ft(*args, **kwargs)

        scaler.fit_transform = spy_ft  # type: ignore[assignment]

        apply_existing_scaler_transform(scaler, train_df, val_df, test_df, FEATURES_41)
        assert len(ft_called) == 0, "fit_transform() was called when it must not be"

    def test_unfitted_scaler_raises(self):
        """Calling with an unfitted scaler must raise ValueError, not silently proceed."""
        scaler = StandardScaler()  # not fitted: no mean_ attribute
        train_df = _make_feature_df(10, seed=30)
        val_df = _make_feature_df(5, seed=31)
        test_df = _make_feature_df(5, seed=32)
        with pytest.raises(ValueError, match="pre-fitted"):
            apply_existing_scaler_transform(scaler, train_df, val_df, test_df, FEATURES_41)

    def test_wrong_feature_count_raises(self):
        scaler = _make_fitted_scaler()
        too_few_features = FEATURES_41[:10]
        train_df = _make_feature_df(20, seed=40)
        val_df = _make_feature_df(5, seed=41)
        test_df = _make_feature_df(5, seed=42)
        # Provide only 10 features but scaler expects 41
        with pytest.raises(ValueError, match="features"):
            apply_existing_scaler_transform(scaler, train_df, val_df, test_df, too_few_features)

    def test_output_shapes_match_input(self):
        scaler = _make_fitted_scaler()
        train_df = _make_feature_df(50, seed=50)
        val_df = _make_feature_df(15, seed=51)
        test_df = _make_feature_df(15, seed=52)
        train_s, val_s, test_s = apply_existing_scaler_transform(
            scaler, train_df, val_df, test_df, FEATURES_41
        )
        assert train_s.shape == (50, 41)
        assert val_s.shape == (15, 41)
        assert test_s.shape == (15, 41)
        assert list(train_s.columns) == FEATURES_41


# ---------------------------------------------------------------------------
# 4. Cleaning and deduplication
# ---------------------------------------------------------------------------

class TestCleanAndDeduplicate:
    """clean_numeric_features removes NaN/Inf rows; filter_and_map_extra_chunk drops headers and unmapped."""

    def test_repeated_header_rows_are_dropped(self):
        """Rows where Label == 'label' (case-insensitive) must be removed."""
        raw_to_class, class_to_id, _ = build_extra_label_mapping(EXTRA_CLASSES_CFG)

        df = pd.DataFrame({
            "Label": ["Bot", "label", "LABEL", "FTP-BruteForce", "label"],
            **{feat: [1.0] * 5 for feat in FEATURES_41},
        })
        filtered, n_headers, n_unmapped = filter_and_map_extra_chunk(df, raw_to_class, class_to_id)
        assert n_headers == 3, f"Expected 3 header rows, got {n_headers}"
        assert len(filtered) == 2

    def test_benign_and_dos_rows_are_unmapped(self):
        raw_to_class, class_to_id, _ = build_extra_label_mapping(EXTRA_CLASSES_CFG)
        df = pd.DataFrame({
            "Label": ["Bot", "Benign", "DoS attacks-Hulk", "SSH-Bruteforce"],
            **{feat: [1.0] * 4 for feat in FEATURES_41},
        })
        filtered, n_headers, n_unmapped = filter_and_map_extra_chunk(df, raw_to_class, class_to_id)
        assert n_unmapped == 2, f"Expected 2 unmapped, got {n_unmapped}"
        assert len(filtered) == 2

    def test_case_insensitive_label_matching(self):
        raw_to_class, class_to_id, _ = build_extra_label_mapping(EXTRA_CLASSES_CFG)
        df = pd.DataFrame({
            "Label": ["BOT", "bot", "Bot", "FTP-BruteForce", "ftp-bruteforce"],
            **{feat: [1.0] * 5 for feat in FEATURES_41},
        })
        filtered, n_headers, n_unmapped = filter_and_map_extra_chunk(df, raw_to_class, class_to_id)
        assert len(filtered) == 5
        assert n_unmapped == 0

    def test_nan_rows_are_dropped(self):
        df = _make_feature_df(10)
        df.loc[2, FEATURES_41[0]] = float("nan")
        df.loc[7, FEATURES_41[5]] = float("nan")
        clean_df, dropped = clean_numeric_features(df, FEATURES_41)
        assert dropped == 2
        assert len(clean_df) == 8

    def test_inf_rows_are_dropped(self):
        df = _make_feature_df(10)
        df.loc[0, FEATURES_41[3]] = float("inf")
        df.loc[9, FEATURES_41[10]] = float("-inf")
        clean_df, dropped = clean_numeric_features(df, FEATURES_41)
        assert dropped == 2
        assert len(clean_df) == 8

    def test_non_numeric_strings_become_nan_and_are_dropped(self):
        df = _make_feature_df(5)
        # Insert a string into a numeric column
        df = df.astype(object)
        df.loc[1, "Protocol"] = "not-a-number"
        clean_df, dropped = clean_numeric_features(df, FEATURES_41)
        assert dropped == 1
        assert len(clean_df) == 4

    def test_empty_chunk_returns_empty(self):
        df = pd.DataFrame(columns=["Label"] + FEATURES_41)
        raw_to_class, class_to_id, _ = build_extra_label_mapping(EXTRA_CLASSES_CFG)
        filtered, n_headers, n_unmapped = filter_and_map_extra_chunk(df, raw_to_class, class_to_id)
        assert filtered.empty
        assert n_headers == 0
        assert n_unmapped == 0

    def test_chunk_without_label_column_returns_empty(self):
        df = _make_feature_df(5)  # no "Label" column
        raw_to_class, class_to_id, _ = build_extra_label_mapping(EXTRA_CLASSES_CFG)
        filtered, n_headers, n_unmapped = filter_and_map_extra_chunk(df, raw_to_class, class_to_id)
        assert filtered.empty

    def test_mapped_class_and_integer_label_added(self):
        raw_to_class, class_to_id, _ = build_extra_label_mapping(EXTRA_CLASSES_CFG)
        df = pd.DataFrame({
            "Label": ["Bot", "FTP-BruteForce", "SSH-Bruteforce"],
            **{feat: [2.0] * 3 for feat in FEATURES_41},
        })
        filtered, _, _ = filter_and_map_extra_chunk(df, raw_to_class, class_to_id)
        assert "mapped_class" in filtered.columns
        assert "label" in filtered.columns
        assert list(filtered["label"].values) == [3, 4, 5]

    def test_clean_numeric_on_empty_input_returns_empty(self):
        df = pd.DataFrame(columns=FEATURES_41)
        clean_df, dropped = clean_numeric_features(df, FEATURES_41)
        assert clean_df.empty
        assert dropped == 0


# ---------------------------------------------------------------------------
# 5. Metadata standardization
# ---------------------------------------------------------------------------

class TestMetadataStandardization:
    """standardize_metadata_chunk produces the correct dtypes."""

    def _make_chunk(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "Dst Port": [443] * n,
            "Src Port": [12345] * n,
            "Timestamp": ["2018-02-14 10:00:00"] * n,
            "Flow ID": [f"flow-{i}" for i in range(n)],
            "Src IP": ["192.168.1.1"] * n,
            "Dst IP": ["10.0.0.1"] * n,
            **{feat: [1.0] * n for feat in FEATURES_41},
        })

    def test_port_columns_are_int64(self):
        chunk = self._make_chunk(5)
        meta = standardize_metadata_chunk(chunk, "test.csv")
        assert meta["Dst Port"].dtype == "Int64", f"Got {meta['Dst Port'].dtype}"
        assert meta["Src Port"].dtype == "Int64", f"Got {meta['Src Port'].dtype}"

    def test_string_columns_are_string_dtype(self):
        chunk = self._make_chunk(5)
        meta = standardize_metadata_chunk(chunk, "test.csv")
        for col in ["Timestamp", "Flow ID", "Src IP", "Dst IP", "source_file"]:
            assert meta[col].dtype == "string", f"Column {col}: got {meta[col].dtype}"

    def test_source_file_is_populated(self):
        chunk = self._make_chunk(3)
        meta = standardize_metadata_chunk(chunk, "friday.csv")
        assert (meta["source_file"] == "friday.csv").all()

    def test_missing_optional_columns_filled_with_na(self):
        """If Src IP / Dst IP are absent, they must be NA strings, not raise."""
        chunk = pd.DataFrame({
            "Dst Port": [80, 443],
            "Timestamp": ["2018-02-14", "2018-02-14"],
        })
        meta = standardize_metadata_chunk(chunk, "partial.csv")
        # Should not raise; Src IP, Dst IP, Flow ID, Src Port fill as NA
        assert meta["Src IP"].dtype == "string"
        assert meta["Dst IP"].dtype == "string"

    def test_assert_metadata_schema_passes_on_valid(self):
        chunk = self._make_chunk(4)
        meta = standardize_metadata_chunk(chunk, "valid.csv")
        # Must not raise
        assert_metadata_schema(meta)

    def test_assert_metadata_schema_fails_on_wrong_port_dtype(self):
        chunk = self._make_chunk(4)
        meta = standardize_metadata_chunk(chunk, "valid.csv")
        # Corrupt port dtype
        meta["Dst Port"] = meta["Dst Port"].astype(float)
        with pytest.raises(AssertionError, match="Int64"):
            assert_metadata_schema(meta)

    def test_assert_metadata_schema_fails_on_wrong_string_dtype(self):
        chunk = self._make_chunk(4)
        meta = standardize_metadata_chunk(chunk, "valid.csv")
        # Corrupt string dtype
        meta["Timestamp"] = meta["Timestamp"].astype(object)
        with pytest.raises(AssertionError, match="string"):
            assert_metadata_schema(meta)


# ---------------------------------------------------------------------------
# 6. Constants integrity
# ---------------------------------------------------------------------------

class TestConstants:
    """Module-level constants must reflect the design spec."""

    def test_exactly_three_extra_class_names(self):
        assert len(EXTRA_CLASS_NAMES) == 3

    def test_extra_class_ids_start_at_3(self):
        assert min(EXTRA_CLASS_IDS.values()) == 3

    def test_extra_class_ids_are_contiguous(self):
        ids = sorted(EXTRA_CLASS_IDS.values())
        assert ids == list(range(3, 3 + len(EXTRA_CLASS_IDS)))

    def test_all_extra_class_names_have_ids(self):
        for name in EXTRA_CLASS_NAMES:
            assert name in EXTRA_CLASS_IDS, f"'{name}' missing from EXTRA_CLASS_IDS"

