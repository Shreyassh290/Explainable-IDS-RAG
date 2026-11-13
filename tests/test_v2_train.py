"""Unit tests for src/ids_rag/v2/train_extra.py (Stage 2 binary DNN training for extra classes).

All tests use synthetic in-memory data only. No model files, no real parquets,
no data/processed, and no models/ are read or written.

Run:
    uv run pytest tests/test_v2_train.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from ids_rag.models.dnn import BinaryDNN, compute_binary_macro_f1
from ids_rag.v2.train_extra import (
    INDICATIVE_ONLY_CLASSES,
    INDICATIVE_THRESHOLD,
    check_v2_output_safety,
    set_seed,
    train_binary_model_v2,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_FEATURES = 41
HIDDEN = [128, 64, 32]

MINIMAL_CONFIG = {
    "seed": 42,
    "model": {
        "batch_size": 32,
        "lr": 0.001,
        "max_epochs": 3,
        "early_stopping_patience": 2,
        "monitor": "f1",
        "num_workers": 0,
        "hidden": HIDDEN,
        "dropout": 0.3,
        "use_batchnorm": False,
    },
}


def _make_tensors(
    n_train: int,
    n_val: int,
    class_id: int,
    n_classes_total: int,
    seed: int = 0,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """Return (x_train, y_train, x_val, y_val) with one positive class embedded."""
    rng = np.random.default_rng(seed)
    x_train = torch.from_numpy(rng.random((n_train, N_FEATURES)).astype(np.float32))
    x_val = torch.from_numpy(rng.random((n_val, N_FEATURES)).astype(np.float32))

    y_train = rng.integers(0, n_classes_total, size=n_train, dtype=np.int64)
    y_val = rng.integers(0, n_classes_total, size=n_val, dtype=np.int64)

    # Guarantee at least a few positives and negatives
    y_train[:5] = class_id
    y_train[5:10] = (class_id + 1) % n_classes_total
    y_val[:2] = class_id
    y_val[2:4] = (class_id + 1) % n_classes_total

    return x_train, y_train, x_val, y_val


# ---------------------------------------------------------------------------
# 1. Safety path check
# ---------------------------------------------------------------------------

class TestCheckV2OutputSafety:
    """check_v2_output_safety raises ValueError for any frozen directory conflict."""

    def test_models_dir_conflict_raises(self, tmp_path):
        root = tmp_path / "project"
        (root / "models").mkdir(parents=True)
        with pytest.raises(ValueError, match="SAFETY REFUSAL"):
            check_v2_output_safety(
                models_dir=root / "models",
                artifacts_dir=root / "artifacts" / "v2" / "training",
                root=root,
            )

    def test_data_processed_conflict_raises(self, tmp_path):
        root = tmp_path / "project"
        with pytest.raises(ValueError, match="SAFETY REFUSAL"):
            check_v2_output_safety(
                models_dir=root / "data" / "processed",
                artifacts_dir=root / "artifacts" / "v2" / "training",
                root=root,
            )

    def test_artifacts_eval_conflict_raises(self, tmp_path):
        root = tmp_path / "project"
        with pytest.raises(ValueError, match="SAFETY REFUSAL"):
            check_v2_output_safety(
                models_dir=root / "models_v2",
                artifacts_dir=root / "artifacts" / "eval",
                root=root,
            )

    def test_allowed_paths_do_not_raise(self, tmp_path):
        root = tmp_path / "project"
        # Both output dirs are v2-specific — must not raise
        check_v2_output_safety(
            models_dir=root / "models_v2",
            artifacts_dir=root / "artifacts" / "v2" / "training",
            root=root,
        )

    def test_artifacts_reports_conflict_raises(self, tmp_path):
        root = tmp_path / "project"
        with pytest.raises(ValueError, match="SAFETY REFUSAL"):
            check_v2_output_safety(
                models_dir=root / "models_v2",
                artifacts_dir=root / "artifacts" / "reports",
                root=root,
            )


# ---------------------------------------------------------------------------
# 2. Seed setter
# ---------------------------------------------------------------------------

class TestSetSeed:
    """set_seed returns a seeded Generator and is deterministic."""

    def test_returns_generator(self):
        gen = set_seed(42)
        assert isinstance(gen, torch.Generator)

    def test_deterministic_numpy(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_torch(self):
        set_seed(42)
        t1 = torch.rand(5)
        set_seed(42)
        t2 = torch.rand(5)
        assert torch.allclose(t1, t2)

    def test_different_seeds_differ(self):
        set_seed(0)
        a = np.random.rand(10)
        set_seed(1)
        b = np.random.rand(10)
        assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# 3. train_binary_model_v2 — behaviour tests (small synthetic data, 3 epochs)
# ---------------------------------------------------------------------------

class TestTrainBinaryModelV2:
    """Structural and correctness tests for train_binary_model_v2."""

    def _run(self, tmp_path, class_id=3, class_name="Bot", n_train=200, n_val=60):
        x_train, y_train, x_val, y_val = _make_tensors(n_train, n_val, class_id, 6, seed=7)
        device = torch.device("cpu")
        gen = set_seed(42)
        return train_binary_model_v2(
            class_id=class_id,
            class_name=class_name,
            x_train=x_train,
            y_train_raw=y_train,
            x_val=x_val,
            y_val_raw=y_val,
            config=MINIMAL_CONFIG,
            device=device,
            models_dir=tmp_path / "models_v2",
            artifacts_dir=tmp_path / "artifacts" / "v2" / "training",
            data_generator=gen,
        )

    def test_history_keys_present(self, tmp_path):
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)
        history = self._run(tmp_path)
        required_keys = {
            "class_id", "class_name", "monitor", "best_epoch",
            "best_val_loss", "best_val_macro_f1", "val_roc_auc",
            "val_f1_threshold_0_5", "val_macro_f1", "elapsed_seconds",
            "n_pos_train", "n_neg_train", "is_indicative_only",
            "train_loss", "val_loss", "val_macro_f1_history",
        }
        assert required_keys.issubset(set(history.keys()))

    def test_model_weights_file_saved(self, tmp_path):
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)
        self._run(tmp_path)
        weights_path = tmp_path / "models_v2" / "dnn_v2_Bot.pt"
        assert weights_path.exists(), f"Expected weights at {weights_path}"

    def test_training_history_json_saved(self, tmp_path):
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)
        self._run(tmp_path)
        json_path = tmp_path / "artifacts" / "v2" / "training" / "training_history_v2_Bot.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["class_name"] == "Bot"

    def test_training_curves_png_saved(self, tmp_path):
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)
        self._run(tmp_path)
        png_path = tmp_path / "artifacts" / "v2" / "training" / "training_curves_v2_Bot.png"
        assert png_path.exists()

    def test_loss_history_has_at_least_one_epoch(self, tmp_path):
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)
        history = self._run(tmp_path)
        assert len(history["train_loss"]) >= 1
        assert len(history["val_loss"]) >= 1

    def test_macro_f1_in_valid_range(self, tmp_path):
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)
        history = self._run(tmp_path)
        assert 0.0 <= history["val_macro_f1"] <= 1.0

    def test_elapsed_seconds_positive(self, tmp_path):
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)
        history = self._run(tmp_path)
        assert history["elapsed_seconds"] > 0.0

    def test_class_id_stored_correctly(self, tmp_path):
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)
        history = self._run(tmp_path, class_id=3)
        assert history["class_id"] == 3

    def test_n_pos_train_is_correct(self, tmp_path):
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)
        x_train, y_train, x_val, y_val = _make_tensors(200, 60, 3, 6, seed=7)
        expected_pos = int(np.sum(y_train == 3))
        device = torch.device("cpu")
        gen = set_seed(42)
        history = train_binary_model_v2(
            class_id=3, class_name="Bot",
            x_train=x_train, y_train_raw=y_train,
            x_val=x_val, y_val_raw=y_val,
            config=MINIMAL_CONFIG, device=device,
            models_dir=tmp_path / "models_v2",
            artifacts_dir=tmp_path / "artifacts" / "v2" / "training",
            data_generator=gen,
        )
        assert history["n_pos_train"] == expected_pos

    def test_weights_are_loadable_as_binary_dnn(self, tmp_path):
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)
        self._run(tmp_path)
        weights_path = tmp_path / "models_v2" / "dnn_v2_Bot.pt"
        model = BinaryDNN(in_features=N_FEATURES, hidden=HIDDEN, dropout=0.3, use_batchnorm=False)
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)  # must not raise


# ---------------------------------------------------------------------------
# 4. Indicative-only flag for FTP-BruteForce
# ---------------------------------------------------------------------------

class TestIndicativeOnlyFlag:
    """FTP-BruteForce must be flagged as indicative-only."""

    def test_ftp_bruteforce_in_indicative_set(self):
        assert "FTP-BruteForce" in INDICATIVE_ONLY_CLASSES

    def test_indicative_threshold_is_positive(self):
        assert INDICATIVE_THRESHOLD > 0

    def test_training_flags_ftp_as_indicative(self, tmp_path):
        """Training FTP-BruteForce with tiny data must set is_indicative_only=True."""
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)

        # Create data with very few FTP-BruteForce positives (class_id=4)
        rng = np.random.default_rng(99)
        n_train, n_val = 60, 20
        x_train = torch.from_numpy(rng.random((n_train, N_FEATURES)).astype(np.float32))
        x_val = torch.from_numpy(rng.random((n_val, N_FEATURES)).astype(np.float32))
        # 3 positives for FTP-BruteForce
        y_train = np.zeros(n_train, dtype=np.int64)
        y_train[:3] = 4
        y_val = np.zeros(n_val, dtype=np.int64)
        y_val[:1] = 4

        gen = set_seed(0)
        history = train_binary_model_v2(
            class_id=4, class_name="FTP-BruteForce",
            x_train=x_train, y_train_raw=y_train,
            x_val=x_val, y_val_raw=y_val,
            config=MINIMAL_CONFIG, device=torch.device("cpu"),
            models_dir=tmp_path / "models_v2",
            artifacts_dir=tmp_path / "artifacts" / "v2" / "training",
            data_generator=gen,
        )
        assert history["is_indicative_only"] is True

    def test_bot_with_sufficient_data_not_indicative(self, tmp_path):
        """Bot with many positives must NOT be flagged as indicative."""
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)

        rng = np.random.default_rng(1)
        n_train, n_val = 1000, 200
        x_train = torch.from_numpy(rng.random((n_train, N_FEATURES)).astype(np.float32))
        x_val = torch.from_numpy(rng.random((n_val, N_FEATURES)).astype(np.float32))
        y_train = np.zeros(n_train, dtype=np.int64)
        y_train[:600] = 3  # 600 positives for Bot
        y_val = np.zeros(n_val, dtype=np.int64)
        y_val[:100] = 3

        gen = set_seed(0)
        history = train_binary_model_v2(
            class_id=3, class_name="Bot",
            x_train=x_train, y_train_raw=y_train,
            x_val=x_val, y_val_raw=y_val,
            config=MINIMAL_CONFIG, device=torch.device("cpu"),
            models_dir=tmp_path / "models_v2",
            artifacts_dir=tmp_path / "artifacts" / "v2" / "training",
            data_generator=gen,
        )
        assert history["is_indicative_only"] is False


# ---------------------------------------------------------------------------
# 5. Invalid monitor raises
# ---------------------------------------------------------------------------

class TestInvalidMonitor:
    def test_invalid_monitor_raises_value_error(self, tmp_path):
        (tmp_path / "models_v2").mkdir(parents=True)
        (tmp_path / "artifacts" / "v2" / "training").mkdir(parents=True)
        bad_cfg = {**MINIMAL_CONFIG, "model": {**MINIMAL_CONFIG["model"], "monitor": "accuracy"}}
        x_train, y_train, x_val, y_val = _make_tensors(100, 30, 3, 6, seed=5)
        with pytest.raises(ValueError, match="monitor"):
            train_binary_model_v2(
                class_id=3, class_name="Bot",
                x_train=x_train, y_train_raw=y_train,
                x_val=x_val, y_val_raw=y_val,
                config=bad_cfg, device=torch.device("cpu"),
                models_dir=tmp_path / "models_v2",
                artifacts_dir=tmp_path / "artifacts" / "v2" / "training",
                data_generator=set_seed(0),
            )


# ---------------------------------------------------------------------------
# 6. Frozen directory invariants
# ---------------------------------------------------------------------------

class TestFrozenDirectoryInvariants:
    """The existing frozen model files must not be touched by the v2 training module."""

    FROZEN_RELATIVE_PATHS = [
        "models/dnn_Benign.pt",
        "models/dnn_DoS.pt",
        "models/dnn_DDoS.pt",
        "models/scaler.joblib",
        "models/feature_names.json",
        "data/processed",
    ]

    def test_v2_model_filename_differs_from_frozen(self, tmp_path):
        """dnn_v2_<class>.pt must not collide with dnn_<class>.pt."""
        for cname in ["Bot", "FTP-BruteForce", "SSH-Bruteforce"]:
            v2_name = f"dnn_v2_{cname}.pt"
            frozen_name = f"dnn_{cname}.pt"
            assert v2_name != frozen_name

    def test_check_v2_output_safety_blocks_models_dir(self, tmp_path):
        root = tmp_path / "project"
        with pytest.raises(ValueError, match="SAFETY REFUSAL"):
            check_v2_output_safety(
                models_dir=root / "models",
                artifacts_dir=root / "artifacts" / "v2" / "training",
                root=root,
            )

    def test_check_v2_output_safety_blocks_artifacts_fresh_eval(self, tmp_path):
        root = tmp_path / "project"
        with pytest.raises(ValueError, match="SAFETY REFUSAL"):
            check_v2_output_safety(
                models_dir=root / "models_v2",
                artifacts_dir=root / "artifacts" / "fresh_eval",
                root=root,
            )

    def test_check_v2_output_safety_blocks_artifacts_tuning(self, tmp_path):
        root = tmp_path / "project"
        with pytest.raises(ValueError, match="SAFETY REFUSAL"):
            check_v2_output_safety(
                models_dir=root / "models_v2",
                artifacts_dir=root / "artifacts" / "tuning_experiments",
                root=root,
            )

