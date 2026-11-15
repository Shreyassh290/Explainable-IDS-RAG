"""Unit tests for the gated decision rule in v2 six-class mode.

Tests:
1. Extra-class wins when its probability >= 0.5 even if Benign (or any baseline) is higher.
2. Baseline argmax is used when all extra-class probabilities < 0.5.
3. FTP-BruteForce is excluded from the decision gate when include_ftp=False.
4. Batch inputs of shape (N, 6) produce matching vectorized predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from ids_rag.v2.pipeline_v2 import predict_gated_decision


def test_extra_class_wins_when_ge_threshold_even_if_benign_higher() -> None:
    """Extra-class wins when its probability >= 0.5 even if Benign is higher."""
    # Probabilities: [Benign=0, DoS=1, DDoS=2, Bot=3, FTP=4, SSH=5]
    # Case A: Bot (0.60) >= 0.5 even though Benign (0.95) is much higher
    probs_bot = np.array([0.95, 0.01, 0.01, 0.60, 0.05, 0.10], dtype=np.float32)
    pred_id, conf = predict_gated_decision(probs_bot, include_ftp=True, threshold=0.5)
    assert pred_id == 3, f"Expected Bot (3), got {pred_id}"
    assert conf == pytest.approx(0.60, rel=1e-4)

    # Case B: SSH-Bruteforce (0.85) >= 0.5 even though Benign (0.99) is higher
    probs_ssh = np.array([0.99, 0.01, 0.01, 0.10, 0.05, 0.85], dtype=np.float32)
    pred_id, conf = predict_gated_decision(probs_ssh, include_ftp=True, threshold=0.5)
    assert pred_id == 5, f"Expected SSH-Bruteforce (5), got {pred_id}"
    assert conf == pytest.approx(0.85, rel=1e-4)

    # Case C: Between two extra classes >= 0.5, the higher extra class wins
    probs_multi_extra = np.array([0.95, 0.01, 0.01, 0.70, 0.20, 0.90], dtype=np.float32)
    pred_id, conf = predict_gated_decision(probs_multi_extra, include_ftp=True, threshold=0.5)
    assert pred_id == 5, f"Expected highest extra class SSH (5), got {pred_id}"
    assert conf == pytest.approx(0.90, rel=1e-4)


def test_baseline_argmax_used_when_all_extras_below_threshold() -> None:
    """Baseline argmax is used when all extra-class probabilities < 0.5."""
    # Case A: DoS is highest baseline (0.88), all extras < 0.5
    probs_dos = np.array([0.20, 0.88, 0.05, 0.45, 0.30, 0.40], dtype=np.float32)
    pred_id, conf = predict_gated_decision(probs_dos, include_ftp=True, threshold=0.5)
    assert pred_id == 1, f"Expected DoS (1), got {pred_id}"
    assert conf == pytest.approx(0.88, rel=1e-4)

    # Case B: Benign is highest baseline (0.92), all extras < 0.5
    probs_benign = np.array([0.92, 0.05, 0.03, 0.49, 0.20, 0.10], dtype=np.float32)
    pred_id, conf = predict_gated_decision(probs_benign, include_ftp=True, threshold=0.5)
    assert pred_id == 0, f"Expected Benign (0), got {pred_id}"
    assert conf == pytest.approx(0.92, rel=1e-4)

    # Case C: DDoS is highest baseline (0.99), all extras near zero
    probs_ddos = np.array([0.01, 0.02, 0.99, 0.05, 0.02, 0.01], dtype=np.float32)
    pred_id, conf = predict_gated_decision(probs_ddos, include_ftp=True, threshold=0.5)
    assert pred_id == 2, f"Expected DDoS (2), got {pred_id}"
    assert conf == pytest.approx(0.99, rel=1e-4)


def test_ftp_excluded_when_flag_false() -> None:
    """FTP-BruteForce is excluded from the decision gate when include_ftp=False."""
    # Flow has strong FTP probability (0.95), other extras low (0.10), Benign high (0.90)
    probs = np.array([0.90, 0.05, 0.05, 0.10, 0.95, 0.10], dtype=np.float32)

    # With FTP included: FTP (4) wins because 0.95 >= 0.5
    pred_with_ftp, conf_with_ftp = predict_gated_decision(probs, include_ftp=True, threshold=0.5)
    assert pred_with_ftp == 4, f"Expected FTP (4) when include_ftp=True, got {pred_with_ftp}"
    assert conf_with_ftp == pytest.approx(0.95, rel=1e-4)

    # Without FTP: FTP is ignored, remaining extras (0.10, 0.10) < 0.5, so Benign (0) wins
    pred_no_ftp, conf_no_ftp = predict_gated_decision(probs, include_ftp=False, threshold=0.5)
    assert pred_no_ftp == 0, f"Expected Benign (0) when include_ftp=False, got {pred_no_ftp}"
    assert conf_no_ftp == pytest.approx(0.90, rel=1e-4)


def test_batched_gated_decision() -> None:
    """2D batched probabilities input produces matching vectorized outputs."""
    probs_batch = np.array(
        [
            [0.95, 0.01, 0.01, 0.60, 0.05, 0.10],  # Bot wins (0.60 >= 0.5)
            [0.20, 0.88, 0.05, 0.45, 0.30, 0.40],  # DoS wins (extras < 0.5)
            [0.90, 0.05, 0.05, 0.10, 0.95, 0.10],  # FTP wins if True, Benign if False
        ],
        dtype=np.float32,
    )

    # Batch with FTP
    preds, confs = predict_gated_decision(probs_batch, include_ftp=True, threshold=0.5)
    assert list(preds) == [3, 1, 4]
    assert confs[0] == pytest.approx(0.60, rel=1e-4)
    assert confs[1] == pytest.approx(0.88, rel=1e-4)
    assert confs[2] == pytest.approx(0.95, rel=1e-4)

    # Batch without FTP
    preds_no_ftp, confs_no_ftp = predict_gated_decision(probs_batch, include_ftp=False, threshold=0.5)
    assert list(preds_no_ftp) == [3, 1, 0]
    assert confs_no_ftp[2] == pytest.approx(0.90, rel=1e-4)

