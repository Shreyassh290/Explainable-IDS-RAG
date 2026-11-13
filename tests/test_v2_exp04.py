"""Unit tests for EXP-04 HistGradientBoosting validation and evaluation logic."""

from __future__ import annotations

import json
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.v2_exp04_hgb_validation import (
    NpEncoder,
    evaluate_validation_metrics,
)


class TestHistGradientBoostingValidation:
    """Test GBDT fitting, prediction, and validation evaluation."""

    @pytest.fixture
    def dummy_data(self):
        np.random.seed(42)
        n_tr = 100
        n_val = 40
        X_tr = np.random.randn(n_tr, 72)
        y_tr = np.random.choice([0, 1], size=n_tr)
        X_val = np.random.randn(n_val, 72)
        y_val = np.random.choice([0, 1], size=n_val)
        val_cats = np.random.choice(["Normal", "Generic", "Fuzzers"], size=n_val)
        return X_tr, y_tr, X_val, y_val, val_cats

    def test_hgb_fit_and_predict_proba_bounds(self, dummy_data):
        X_tr, y_tr, X_val, y_val, _ = dummy_data
        clf = HistGradientBoostingClassifier(max_iter=10, random_state=42)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_val)[:, 1]

        assert probs.shape == (len(X_val),)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert np.all(np.isfinite(probs))

    def test_evaluate_validation_metrics_schema(self, dummy_data):
        X_tr, y_tr, X_val, y_val, val_cats = dummy_data
        clf = HistGradientBoostingClassifier(max_iter=10, random_state=42)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_val)[:, 1]

        metrics = evaluate_validation_metrics(probs, y_val, val_cats, tau=0.50)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "roc_auc" in metrics
        assert "benign_fpr" in metrics
        assert "recall" in metrics
        assert "confusion_matrix" in metrics
        assert "per_category_validation_recall" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["benign_fpr"] <= 1.0

    def test_np_encoder_serializes_cleanly(self):
        payload = {
            "int64_val": np.int64(42),
            "float32_val": np.float32(0.95),
            "arr_val": np.array([1, 2, 3]),
        }
        serialized = json.dumps(payload, cls=NpEncoder)
        deserialized = json.loads(serialized)
        assert deserialized["int64_val"] == 42
        assert np.isclose(deserialized["float32_val"], 0.95, atol=1e-4)
        assert deserialized["arr_val"] == [1, 2, 3]

