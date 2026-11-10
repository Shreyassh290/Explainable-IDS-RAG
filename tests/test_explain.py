"""Unit tests for gradient-based feature attribution, scaler inversion, and explanation formatting."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from ids_rag.explain.shap_topk import (
    compute_gradient_importance,
    explain_flow,
    extract_metadata_dict,
)
from ids_rag.models.dnn import BinaryDNN
from ids_rag.models.ensemble import BinaryDNNEnsemble


class SyntheticLinearModel(nn.Module):
    """Linear model y = w . x for testing analytical gradient attribution."""

    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.w = nn.Parameter(weights.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w)


def test_gradient_linear_model_matches_weights() -> None:
    """Verify that on y = w . x, the gradient equals w, top-k picks largest |w|, and direction matches sign."""
    w = torch.tensor([0.5, -3.2, 1.8, -0.1, 0.0], dtype=torch.float32)
    model = SyntheticLinearModel(w)

    x = np.array([1.0, 2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    importance, grad = compute_gradient_importance(x, model)

    # 1. Gradient equals weights
    np.testing.assert_allclose(grad, w.numpy(), atol=1e-6)

    # 2. Importance equals |w|
    np.testing.assert_allclose(importance, np.abs(w.numpy()), atol=1e-6)

    # 3. Top-3 picks largest absolute weights: index 1 (3.2), index 2 (1.8), index 0 (0.5)
    top_3_indices = np.argsort(-importance)[:3]
    np.testing.assert_array_equal(top_3_indices, np.array([1, 2, 0]))

    # 4. Direction matches sign
    assert grad[1] < 0  # "lowers"
    assert grad[2] > 0  # "raises"
    assert grad[0] > 0  # "raises"


def test_grad_x_input_linear_model() -> None:
    """Verify that on y = w . x, grad_x_input importance equals |x * w|."""
    w = torch.tensor([0.5, -3.2, 1.8, -0.1, 0.0], dtype=torch.float32)
    model = SyntheticLinearModel(w)

    x = np.array([1.0, 2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    importance, grad = compute_gradient_importance(x, model, method="grad_x_input")

    # Gradient still equals w
    np.testing.assert_allclose(grad, w.numpy(), atol=1e-6)

    # Importance equals |x * w| = [0.5, 6.4, 1.8, 0.05, 0.0]
    expected_importance = np.abs(x * w.numpy())
    np.testing.assert_allclose(importance, expected_importance, atol=1e-6)

    # Top-3 indices: index 1 (6.4), index 2 (1.8), index 0 (0.5)
    top_3_indices = np.argsort(-importance)[:3]
    np.testing.assert_array_equal(top_3_indices, np.array([1, 2, 0]))


def test_unsupported_attribution_method_raises() -> None:
    """Verify that unsupported method string raises ValueError."""
    w = torch.tensor([1.0, 2.0], dtype=torch.float32)
    model = SyntheticLinearModel(w)
    x = np.array([1.0, 2.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported attribution method"):
        compute_gradient_importance(x, model, method="invalid_method")


def test_scaler_inversion_synthetic() -> None:
    """Verify that original-unit transformation inverts a known StandardScaler accurately."""
    data = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]], dtype=np.float32)
    scaler = StandardScaler().fit(data)

    test_point = np.array([25.0, 250.0], dtype=np.float32)
    scaled_point = scaler.transform(test_point.reshape(1, -1))[0]

    # Invert back
    inverted_point = scaler.inverse_transform(scaled_point.reshape(1, -1))[0]
    np.testing.assert_allclose(inverted_point, test_point, rtol=1e-5, atol=1e-5)


def test_explain_flow_structure_and_sorting() -> None:
    """Verify that explain_flow returns expected schema and exactly k features sorted by descending importance."""
    feature_names = ["Protocol", "Flow Duration", "Tot Fwd Pkts", "Flow Byts/s", "Flow Pkts/s"]
    in_features = len(feature_names)

    # Create dummy BinaryDNN models
    model_0 = BinaryDNN(in_features=in_features, hidden=[8], dropout=0.0, use_batchnorm=False)
    model_1 = BinaryDNN(in_features=in_features, hidden=[8], dropout=0.0, use_batchnorm=False)
    models = {0: model_0, 1: model_1}
    label_map = {0: "Benign", 1: "DoS"}

    ensemble = BinaryDNNEnsemble(
        models=models,
        label_map=label_map,
        feature_names=feature_names,
    )

    x_dummy = np.array([0.1, -0.5, 1.2, 0.8, -1.0], dtype=np.float32)
    k = 3

    result = explain_flow(
        x_scaled=x_dummy,
        ensemble=ensemble,
        k=k,
        scaler=None,
        meta_row=None,
        feature_names=feature_names,
    )

    # Expected top-level keys
    expected_keys = {"predicted_class", "predicted_label", "confidence", "tier", "features", "metadata"}
    assert set(result.keys()) == expected_keys

    # Exactly k features
    assert len(result["features"]) == k

    # Features sorted by descending importance
    importances = [f["importance"] for f in result["features"]]
    assert importances == sorted(importances, reverse=True)

    # Each feature entry has required fields
    required_feat_keys = {
        "name",
        "scaled_value",
        "original_value",
        "importance",
        "direction",
        "description",
        "implication",
    }
    for f in result["features"]:
        assert set(f.keys()) == required_feat_keys
        assert f["direction"] in ("raises", "lowers")
        assert len(f["description"]) > 0
        assert len(f["implication"]) > 0


def test_metadata_missing_values_become_none() -> None:
    """Verify that NaN and missing metadata values become None (JSON null)."""
    meta_with_nans = {
        "Timestamp": np.nan,
        "Dst Port": float("nan"),
        "Flow ID": None,
        "Src IP": "10.0.0.1",
        "Src Port": 443,
    }

    result = extract_metadata_dict(meta_with_nans, protocol_value=6.0)

    assert result["Timestamp"] is None
    assert result["Dst Port"] is None
    assert result["Flow ID"] is None
    assert result["Src IP"] == "10.0.0.1"
    assert result["Src Port"] == 443
    assert result["Dst IP"] is None
    assert result["source_file"] is None
    assert result["protocol"] == 6
    assert result["dataset"] == "CICIDS2018"

