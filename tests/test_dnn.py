"""Unit tests for BinaryDNN architecture, class weights, and macro-F1 monitoring."""

import numpy as np
import pytest
import torch
from torch import nn

from ids_rag.models.dnn import (
    BinaryDNN,
    check_monitor_improvement,
    compute_binary_class_weights,
    compute_binary_macro_f1,
)


def test_binary_dnn_output_shape() -> None:
    """Ensure BinaryDNN outputs tensor of shape (batch, 1) without sigmoid."""
    batch_size = 16
    in_features = 20
    model = BinaryDNN(in_features=in_features, hidden=[32, 16], dropout=0.3, use_batchnorm=False)
    model.eval()

    x = torch.randn(batch_size, in_features)
    out = model(x)

    assert out.shape == (batch_size, 1)
    assert out.dtype == torch.float32


def test_binary_dnn_no_batchnorm_when_disabled() -> None:
    """Verify that when use_batchnorm=False, no BatchNorm layers exist in the model."""
    model_without_bn = BinaryDNN(in_features=10, hidden=[32, 16, 8], use_batchnorm=False)
    bn_modules = [m for m in model_without_bn.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert len(bn_modules) == 0

    # Also check that when use_batchnorm=True, BatchNorm layers are present
    model_with_bn = BinaryDNN(in_features=10, hidden=[32, 16, 8], use_batchnorm=True)
    bn_modules_present = [m for m in model_with_bn.modules() if isinstance(m, nn.BatchNorm1d)]
    assert len(bn_modules_present) == 3


def test_class_weight_formula_small_example() -> None:
    """Verify weight formula w_c = N / (2 * N_c) on small synthetic examples."""
    # Example: 3 negative samples, 1 positive sample -> N = 4
    # w_0 = 4 / (2 * 3) = 2/3 ≈ 0.666667
    # w_1 = 4 / (2 * 1) = 2.0
    y_imbalanced = np.array([0, 0, 0, 1])
    w_0, w_1 = compute_binary_class_weights(y_imbalanced)

    assert pytest.approx(w_0, rel=1e-5) == 4.0 / (2.0 * 3.0)
    assert pytest.approx(w_1, rel=1e-5) == 4.0 / (2.0 * 1.0)

    # Example: Balanced case -> N = 10, N_0 = 5, N_1 = 5 -> w_0 = 1.0, w_1 = 1.0
    y_balanced = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    w_0_bal, w_1_bal = compute_binary_class_weights(y_balanced)

    assert pytest.approx(w_0_bal, rel=1e-5) == 1.0
    assert pytest.approx(w_1_bal, rel=1e-5) == 1.0


def test_compute_binary_macro_f1_small_example() -> None:
    """Verify macro-averaged F1 calculation on tiny synthetic examples."""
    # Perfect match: Macro F1 = 1.0
    y_true_perfect = [0, 1, 0, 1]
    y_pred_perfect = [0, 1, 0, 1]
    assert pytest.approx(compute_binary_macro_f1(y_true_perfect, y_pred_perfect), rel=1e-5) == 1.0

    # Hand-calculated case: y_true = [0, 0, 1, 1], y_pred = [1, 1, 1, 1]
    # For class 1: TP=2, FP=2, FN=0 -> P=0.5, R=1.0 -> F1 = 2/3
    # For class 0: TP=0, FP=0, FN=2 -> P=0, R=0 -> F1 = 0
    # Macro F1 = (2/3 + 0) / 2 = 1/3 ≈ 0.333333
    y_true_all_pos = [0, 0, 1, 1]
    y_pred_all_pos = [1, 1, 1, 1]
    expected_f1 = (2.0 / 3.0 + 0.0) / 2.0
    assert pytest.approx(compute_binary_macro_f1(y_true_all_pos, y_pred_all_pos), rel=1e-5) == expected_f1


def test_macro_f1_monitor_improvement_rules() -> None:
    """Verify monitor logic: higher macro F1 improves; ties broken by lower loss."""
    # 1. Strictly higher macro F1 improves (even if loss increased)
    assert check_monitor_improvement("f1", current_f1=0.85, current_loss=0.20, best_f1=0.80, best_loss=0.05) is True

    # 2. Lower macro F1 rejected (even if loss is much lower)
    assert check_monitor_improvement("f1", current_f1=0.79, current_loss=0.01, best_f1=0.80, best_loss=0.05) is False

    # 3. Macro F1 tie broken by strictly lower loss
    assert check_monitor_improvement("f1", current_f1=0.80, current_loss=0.04, best_f1=0.80, best_loss=0.05) is True

    # 4. Macro F1 tie with equal or higher loss is rejected
    assert check_monitor_improvement("f1", current_f1=0.80, current_loss=0.05, best_f1=0.80, best_loss=0.05) is False
    assert check_monitor_improvement("f1", current_f1=0.80, current_loss=0.06, best_f1=0.80, best_loss=0.05) is False

    # 5. Monitor 'loss' mode continues to function properly
    assert check_monitor_improvement("loss", current_f1=0.50, current_loss=0.04, best_f1=0.90, best_loss=0.05) is True
    assert check_monitor_improvement("loss", current_f1=0.99, current_loss=0.06, best_f1=0.90, best_loss=0.05) is False
