"""Synthetic data tests for V2 demo sample extraction and fingerprint caching.

Verifies:
- Safety assertions refuse to write to app/raw_samples.csv.
- Row numeric validation rejects rows with NaN, Inf, -Inf, missing, or non-numeric values.
- get_features_fingerprint in app/streamlit_app.py produces identical digests regardless
  of whether 'Label' or 'Provenance' are present, absent, uppercase, or lowercase.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest


def _load_streamlit_app_module():
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app", app_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_extract_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "v2_extract_demo_samples.py"
    spec = importlib.util.spec_from_file_location("v2_extract_demo_samples", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_safety_assertion_refuses_overwrite_raw_samples():
    """Ensure extract script contains safety assertion protecting app/raw_samples.csv."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "v2_extract_demo_samples.py"
    code = script_path.read_text(encoding="utf-8")
    assert "assert out_csv.resolve() != frozen_csv.resolve()" in code
    assert "assert out_csv.name == \"raw_samples_v2.csv\"" in code


def test_is_valid_numeric_row_skips_inf_nan_and_bad_values():
    """Ensure is_valid_numeric_row accepts finite floats and rejects Inf, -Inf, NaN, and text."""
    mod = _load_extract_script_module()
    feature_cols = ["feat_a", "feat_b", "feat_c"]

    valid_row = {"feat_a": "12.5", "feat_b": "0", "feat_c": "-3.1415"}
    assert mod.is_valid_numeric_row(valid_row, feature_cols) is True

    # NaN variations
    nan_row = {"feat_a": "12.5", "feat_b": "NaN", "feat_c": "-3.1415"}
    assert mod.is_valid_numeric_row(nan_row, feature_cols) is False

    nan_lower = {"feat_a": "12.5", "feat_b": "nan", "feat_c": "-3.1415"}
    assert mod.is_valid_numeric_row(nan_lower, feature_cols) is False

    # Infinity variations
    inf_row = {"feat_a": "Infinity", "feat_b": "0", "feat_c": "-3.1415"}
    assert mod.is_valid_numeric_row(inf_row, feature_cols) is False

    neg_inf_row = {"feat_a": "12.5", "feat_b": "-Infinity", "feat_c": "-3.1415"}
    assert mod.is_valid_numeric_row(neg_inf_row, feature_cols) is False

    # Non-numeric text / missing
    bad_text_row = {"feat_a": "12.5", "feat_b": "bad_value", "feat_c": "-3.1415"}
    assert mod.is_valid_numeric_row(bad_text_row, feature_cols) is False

    missing_col_row = {"feat_a": "12.5", "feat_b": "0"}
    assert mod.is_valid_numeric_row(missing_col_row, feature_cols) is False


def test_get_features_fingerprint_ignores_label_and_provenance():
    """Ensure get_features_fingerprint yields identical hash regardless of Label/Provenance."""
    mod = _load_streamlit_app_module()

    base_features = {
        "Protocol": 6.0,
        "Flow Duration": 150000.0,
        "Tot Fwd Pkts": 10.0,
        "Tot Bwd Pkts": 8.0,
    }

    fp_base = mod.get_features_fingerprint(base_features)

    with_label = dict(base_features, Label="Benign")
    assert mod.get_features_fingerprint(with_label) == fp_base

    with_lower_label = dict(base_features, label="DoS")
    assert mod.get_features_fingerprint(with_lower_label) == fp_base

    with_provenance = dict(base_features, Provenance="Friday.csv:12345")
    assert mod.get_features_fingerprint(with_provenance) == fp_base

    with_both = dict(base_features, Label="Bot", Provenance="Friday.csv:99999")
    assert mod.get_features_fingerprint(with_both) == fp_base

    with_lowercase_both = dict(base_features, label="Bot", provenance="Friday.csv:99999")
    assert mod.get_features_fingerprint(with_lowercase_both) == fp_base

    # Changing an actual feature alters the fingerprint
    different_feature = dict(base_features, Flow_Duration=999999.0)
    assert mod.get_features_fingerprint(different_feature) != fp_base

