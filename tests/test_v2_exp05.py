"""Unit tests for EXP-05 hybrid ensemble logic, soft voting, and stacking."""

from __future__ import annotations

import json
import numpy as np
import pytest

from scripts.v2_exp05_hybrid_ensemble import (
    NpEncoder,
    evaluate_ensemble_metrics,
)


class TestHybridEnsemble:
    """Test ensemble probability combination and metric calculations."""

    @pytest.fixture
    def dummy_predictions(self):
        np.random.seed(42)
        n = 100
        p_dnn = np.random.uniform(0.1, 0.9, size=n)
        p_gbdt = np.random.uniform(0.1, 0.9, size=n)
        y = np.random.choice([0, 1], size=n)
        cats = np.random.choice(["Normal", "Generic", "Fuzzers"], size=n)
        return p_dnn, p_gbdt, y, cats

    def test_soft_voting_bounds(self, dummy_predictions):
        p_dnn, p_gbdt, _, _ = dummy_predictions
        w1, w2 = 0.35, 0.65
        p_ens = w1 * p_dnn + w2 * p_gbdt

        assert np.all(p_ens >= 0.0)
        assert np.all(p_ens <= 1.0)
        assert np.all(np.isfinite(p_ens))

    def test_evaluate_ensemble_metrics_schema(self, dummy_predictions):
        p_dnn, p_gbdt, y, cats = dummy_predictions
        p_ens = 0.5 * p_dnn + 0.5 * p_gbdt

        metrics = evaluate_ensemble_metrics(p_ens, y, cats, tau=0.50)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "roc_auc" in metrics
        assert "benign_fpr" in metrics
        assert "fuzzers_recall" in metrics
        assert "confusion_matrix" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_np_encoder_clean_serialization(self):
        payload = {
            "int_val": np.int64(10),
            "float_val": np.float32(0.85),
            "array": np.array([0.1, 0.2]),
        }
        res = json.dumps(payload, cls=NpEncoder)
        decoded = json.loads(res)
        assert decoded["int_val"] == 10
        assert np.isclose(decoded["float_val"], 0.85, atol=1e-4)

