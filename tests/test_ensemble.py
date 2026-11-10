"""Unit tests for ensemble prediction, tier assignment, and temperature scaling."""

import numpy as np
import pytest
from scipy.special import expit

from ids_rag.models.ensemble import (
    apply_temperature_to_logits,
    assign_tier,
    assign_tiers,
)


def test_tier_boundaries() -> None:
    """Verify tier assignment threshold boundary conditions."""
    tiers_cfg = {"very_high": 0.95, "high": 0.70, "medium": 0.50}

    # Exactly at thresholds (boundary values go to higher tier)
    assert assign_tier(0.95, tiers_cfg) == "VERY_HIGH"
    assert assign_tier(0.70, tiers_cfg) == "HIGH"
    assert assign_tier(0.50, tiers_cfg) == "MEDIUM"

    # Strictly below medium threshold
    assert assign_tier(0.4999, tiers_cfg) == "LOW"
    assert assign_tier(0.0, tiers_cfg) == "LOW"

    # Well inside brackets
    assert assign_tier(0.99, tiers_cfg) == "VERY_HIGH"
    assert assign_tier(0.85, tiers_cfg) == "HIGH"
    assert assign_tier(0.60, tiers_cfg) == "MEDIUM"

    # Vectorized check
    confs = np.array([0.95, 0.70, 0.50, 0.4999, 0.82])
    expected = np.array(["VERY_HIGH", "HIGH", "MEDIUM", "LOW", "HIGH"], dtype=object)
    np.testing.assert_array_equal(assign_tiers(confs, tiers_cfg), expected)


def test_argmax_and_confidence_hand_made_matrix() -> None:
    """Verify argmax and confidence selection on a hand-crafted probability matrix."""
    # Row 0: max at index 1 with conf 0.82
    # Row 1: max at index 0 with conf 0.96
    # Row 2: max at index 2 with conf 0.65
    probs = np.array(
        [
            [0.15, 0.82, 0.30],
            [0.96, 0.20, 0.05],
            [0.40, 0.45, 0.65],
        ]
    )

    pred_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    tiers = assign_tiers(confidences)

    np.testing.assert_array_equal(pred_labels, np.array([1, 0, 2]))
    np.testing.assert_allclose(confidences, np.array([0.82, 0.96, 0.65]))
    np.testing.assert_array_equal(tiers, np.array(["HIGH", "VERY_HIGH", "MEDIUM"], dtype=object))


def test_temperature_one_leaves_probabilities_unchanged() -> None:
    """Verify that scaling logits by T=1.0 yields standard sigmoid probabilities."""
    logits = np.array([-3.5, -1.0, 0.0, 1.5, 4.2], dtype=np.float64)
    expected_probs = expit(logits)
    scaled_probs = apply_temperature_to_logits(logits, temperature=1.0)

    np.testing.assert_allclose(scaled_probs, expected_probs, rtol=1e-6, atol=1e-6)

