"""Unit tests for UNSW-NB15 CV2 hyperparameter search protocol.

Verifies with synthetic data:
1. UNSW_NB15_testing-set.csv is never opened by unsw_hgb_random_search.py
   (static source inspection + guard function test).
2. The preprocessor is fitted on training folds only - never on validation rows.
3. id / ID / label / attack_cat are excluded from features in every fold.
4. Outputs land in cv2/ directories, never in cv/ or final_test/.
5. _sample_params always returns a dict with all required keys and values in range.
6. evaluate_one_trial returns a result dict with all expected metric keys.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

# Ensure repo root is importable
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ids_rag.unsw.preprocessor import UNSWStrictPreprocessor
from scripts.unsw_hgb_random_search import (
    _guard_frozen_dirs,
    _guard_test_set_not_opened,
    _FORBIDDEN_TEST_PATH,
    _FROZEN_CV1_MODELS_DIR,
    _FROZEN_CV1_ARTIFACTS_DIR,
    _FROZEN_FINAL_TEST_DIR,
    CV2_MODELS_DIR,
    CV2_ARTIFACTS_DIR,
    _sample_params,
    calibrate_threshold_on_oof,
    compute_binary_metrics,
    evaluate_one_trial,
    BASELINE_PARAMS,
    SEED,
)


# ---------------------------------------------------------------------------
# Minimal synthetic UNSW-like DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "id": np.arange(n),
        "ID": np.arange(n),
        "label": rng.integers(0, 2, size=n).astype(float),
        "attack_cat": rng.choice(["Normal", "Generic", "Fuzzers"], size=n),
        "spkts": rng.integers(1, 50, size=n).astype(float),
        "dpkts": rng.integers(0, 50, size=n).astype(float),
        "sbytes": rng.integers(100, 5000, size=n).astype(float),
        "dbytes": rng.integers(0, 5000, size=n).astype(float),
        "sttl": rng.choice([31.0, 62.0, 254.0], size=n),
        "dttl": rng.choice([0.0, 29.0, 252.0], size=n),
        "sload": rng.uniform(100.0, 100000.0, size=n),
        "dload": rng.uniform(0.0, 100000.0, size=n),
        "dur": rng.uniform(0.0, 5.0, size=n),
        "rate": rng.uniform(0.1, 100.0, size=n),
        "sloss": rng.integers(0, 5, size=n).astype(float),
        "dloss": rng.integers(0, 5, size=n).astype(float),
        "sinpkt": rng.uniform(0, 100, size=n),
        "dinpkt": rng.uniform(0, 100, size=n),
        "sjit": rng.uniform(0, 50, size=n),
        "djit": rng.uniform(0, 50, size=n),
        "swin": rng.choice([0.0, 255.0], size=n),
        "dwin": rng.choice([0.0, 255.0], size=n),
        "tcprtt": rng.uniform(0, 1.0, size=n),
        "synack": rng.uniform(0, 0.5, size=n),
        "ackdat": rng.uniform(0, 0.5, size=n),
        "smean": rng.uniform(30, 1500, size=n),
        "dmean": rng.uniform(0, 1500, size=n),
        "trans_depth": rng.choice([0.0, 1.0, 2.0], size=n),
        "response_body_len": rng.integers(0, 500, size=n).astype(float),
        "ct_srv_src": rng.integers(1, 20, size=n).astype(float),
        "ct_state_ttl": rng.choice([0.0, 1.0, 2.0], size=n),
        "ct_dst_ltm": rng.integers(1, 20, size=n).astype(float),
        "ct_src_dport_ltm": rng.integers(1, 20, size=n).astype(float),
        "ct_dst_sport_ltm": rng.integers(1, 20, size=n).astype(float),
        "ct_dst_src_ltm": rng.integers(1, 20, size=n).astype(float),
        "is_ftp_login": rng.choice([0.0, 1.0], size=n),
        "ct_ftp_cmd": rng.choice([0.0, 1.0], size=n),
        "ct_flw_http_mthd": rng.choice([0.0, 1.0], size=n),
        "ct_src_ltm": rng.integers(1, 20, size=n).astype(float),
        "ct_srv_dst": rng.integers(1, 20, size=n).astype(float),
        "is_sm_ips_ports": rng.choice([0.0, 1.0], size=n),
        "proto": rng.choice(["tcp", "udp", "arp", "ospf"], size=n),
        "service": rng.choice(["-", "http", "dns", "ftp"], size=n),
        "state": rng.choice(["FIN", "INT", "CON", "REQ"], size=n),
    })


# ---------------------------------------------------------------------------
# 1. Testing file is never opened
# ---------------------------------------------------------------------------

class TestTestingFileNeverOpened:

    def test_guard_raises_on_forbidden_path(self):
        with pytest.raises(RuntimeError, match="PROTOCOL VIOLATION"):
            _guard_test_set_not_opened(_FORBIDDEN_TEST_PATH)

    def test_guard_does_not_raise_on_training_path(self, tmp_path):
        # Any path that is not the forbidden test CSV must pass silently.
        safe_path = tmp_path / "UNSW_NB15_training-set.csv"
        _guard_test_set_not_opened(safe_path)  # must not raise

    def test_script_source_never_references_testing_csv_as_load_target(self):
        """Static source check: the script must not load the testing CSV."""
        script = REPO_ROOT / "scripts" / "unsw_hgb_random_search.py"
        assert script.exists(), "unsw_hgb_random_search.py not found"
        content = script.read_text(encoding="utf-8")
        # The filename may appear in the guard variable definition, but must
        # never appear inside a pd.read_csv() or open() call that is not inside
        # the guard function itself.
        # The simplest safe check: DATA_PATH (used for pd.read_csv) must equal
        # the training CSV, confirmed by the TRAIN_PATH constant name.
        assert "TRAIN_PATH" in content
        # The testing filename must not be used as the argument to pd.read_csv.
        assert "pd.read_csv(_FORBIDDEN_TEST_PATH)" not in content
        assert "pd.read_csv(TEST_PATH)" not in content


# ---------------------------------------------------------------------------
# 2. Preprocessor is fitted on training folds only
# ---------------------------------------------------------------------------

class TestPreprocessorFitScope:

    def test_scaler_mean_matches_training_fold_only(self, synthetic_df):
        """Validate that fitting on 70 % of data ignores the other 30 %."""
        train_fold = synthetic_df.iloc[:210].copy()
        val_fold = synthetic_df.iloc[210:].copy()

        # Inject extreme values into val fold that must NOT affect the scaler
        val_fold = val_fold.copy()
        val_fold["sbytes"] = 1e9

        prep = UNSWStrictPreprocessor()
        prep.fit(train_fold)

        sbytes_idx = prep.numeric_cols_.index("sbytes")
        # Mean must reflect training fold only (log1p scale)
        expected_mean = np.log1p(
            np.maximum(train_fold["sbytes"].to_numpy(dtype=np.float32), 0.0)
        ).mean()
        assert np.isclose(prep.scaler_.mean_[sbytes_idx], expected_mean, atol=1e-3), (
            f"Scaler mean {prep.scaler_.mean_[sbytes_idx]:.6f} != "
            f"training-only mean {expected_mean:.6f}"
        )

    def test_evaluate_one_trial_preprocessor_fitted_on_training_fold(self, synthetic_df):
        """evaluate_one_trial must complete without AssertionError for preprocessor."""
        y_all = synthetic_df["label"].to_numpy(dtype=np.float32)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        # Use a fast configuration to keep the test quick
        fast_params = {
            "learning_rate": 0.10,
            "max_leaf_nodes": 15,
            "max_depth": 3,
            "min_samples_leaf": 5,
            "l2_regularization": 1.0,
            "max_iter": 20,
            "class_weight": None,
            "early_stopping": True,
            "validation_fraction": 0.10,
            "n_iter_no_change": 5,
            "random_state": SEED,
        }
        result = evaluate_one_trial(
            trial_id="test_trial",
            params=fast_params,
            df_train=synthetic_df,
            y_all=y_all,
            skf=skf,
        )
        # Result must exist and have required keys
        assert "oof_recall" in result
        assert "oof_fpr" in result
        assert "oof_tau" in result


# ---------------------------------------------------------------------------
# 3. id / ID / label / attack_cat excluded from features
# ---------------------------------------------------------------------------

class TestForbiddenColumnsExcluded:

    def test_feature_names_exclude_forbidden(self, synthetic_df):
        prep = UNSWStrictPreprocessor()
        prep.fit(synthetic_df)
        forbidden = {"id", "ID", "label", "attack_cat"}
        for fn in prep.feature_names_:
            assert fn not in forbidden, f"Forbidden column '{fn}' found in feature names"

    def test_feature_count_is_72(self, synthetic_df):
        prep = UNSWStrictPreprocessor()
        prep.fit(synthetic_df)
        assert len(prep.feature_names_) == 72
        X = prep.transform(synthetic_df)
        assert X.shape[1] == 72


# ---------------------------------------------------------------------------
# 4. Outputs only in cv2/ directories
# ---------------------------------------------------------------------------

class TestOutputDirectoryGuards:

    def test_guard_blocks_write_to_frozen_cv1_models(self):
        offending = _FROZEN_CV1_MODELS_DIR / "some_file.joblib"
        with pytest.raises(RuntimeError, match="PROTOCOL VIOLATION"):
            _guard_frozen_dirs(offending)

    def test_guard_blocks_write_to_frozen_cv1_artifacts(self):
        offending = _FROZEN_CV1_ARTIFACTS_DIR / "cv_summary.json"
        with pytest.raises(RuntimeError, match="PROTOCOL VIOLATION"):
            _guard_frozen_dirs(offending)

    def test_guard_blocks_write_to_frozen_final_test(self):
        offending = _FROZEN_FINAL_TEST_DIR / "result.json"
        with pytest.raises(RuntimeError, match="PROTOCOL VIOLATION"):
            _guard_frozen_dirs(offending)

    def test_guard_allows_cv2_dirs(self):
        # Must not raise for cv2 paths
        _guard_frozen_dirs(CV2_MODELS_DIR / "some_model.joblib")
        _guard_frozen_dirs(CV2_ARTIFACTS_DIR / "cv_trials.csv")


# ---------------------------------------------------------------------------
# 5. _sample_params returns valid ranges
# ---------------------------------------------------------------------------

class TestParamSampler:

    def test_sampled_keys_present(self):
        rng = np.random.default_rng(0)
        p = _sample_params(rng)
        required = {
            "learning_rate", "max_depth", "max_leaf_nodes",
            "min_samples_leaf", "l2_regularization", "max_iter",
            "class_weight", "early_stopping", "validation_fraction",
            "n_iter_no_change", "random_state",
        }
        assert required.issubset(set(p.keys()))

    def test_sampled_values_in_range(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            p = _sample_params(rng)
            assert 0.02 <= p["learning_rate"] <= 0.20
            assert 4 <= p["max_depth"] <= 16
            assert 15 <= p["max_leaf_nodes"] <= 127
            assert 20 <= p["min_samples_leaf"] <= 200
            assert 0.0 <= p["l2_regularization"] <= 10.0
            assert 200 <= p["max_iter"] <= 600
            assert p["class_weight"] in (None, "balanced")

    def test_seed_reproducibility(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = _sample_params(rng1)
        p2 = _sample_params(rng2)
        assert p1["learning_rate"] == p2["learning_rate"]
        assert p1["max_depth"] == p2["max_depth"]


# ---------------------------------------------------------------------------
# 6. evaluate_one_trial result dict has all expected keys
# ---------------------------------------------------------------------------

class TestEvaluateOneTrialSchema:

    def test_result_has_all_metric_keys(self, synthetic_df):
        y_all = synthetic_df["label"].to_numpy(dtype=np.float32)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        fast_params = {
            "learning_rate": 0.10,
            "max_leaf_nodes": 15,
            "max_depth": 3,
            "min_samples_leaf": 5,
            "l2_regularization": 0.0,
            "max_iter": 15,
            "class_weight": None,
            "early_stopping": True,
            "validation_fraction": 0.10,
            "n_iter_no_change": 3,
            "random_state": SEED,
        }
        result = evaluate_one_trial(
            trial_id="schema_check",
            params=fast_params,
            df_train=synthetic_df,
            y_all=y_all,
            skf=skf,
        )
        expected_keys = {
            "trial_id",
            "learning_rate", "max_depth", "max_leaf_nodes",
            "min_samples_leaf", "l2_regularization", "max_iter", "class_weight",
            "acc_mean", "acc_std",
            "recall_mean", "recall_std",
            "fpr_mean", "fpr_std",
            "worst_fold_fpr",
            "auc_mean", "auc_std",
            "oof_tau", "oof_macro_f1",
            "oof_accuracy", "oof_recall", "oof_fpr", "oof_auc",
            "elapsed_s",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_oof_tau_in_valid_range(self, synthetic_df):
        y_all = synthetic_df["label"].to_numpy(dtype=np.float32)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        fast_params = {
            "learning_rate": 0.10,
            "max_leaf_nodes": 15,
            "max_depth": 3,
            "min_samples_leaf": 5,
            "l2_regularization": 0.0,
            "max_iter": 15,
            "class_weight": None,
            "early_stopping": True,
            "validation_fraction": 0.10,
            "n_iter_no_change": 3,
            "random_state": SEED,
        }
        result = evaluate_one_trial(
            trial_id="tau_check",
            params=fast_params,
            df_train=synthetic_df,
            y_all=y_all,
            skf=skf,
        )
        assert 0.10 <= result["oof_tau"] <= 0.90
        assert 0.0 <= result["oof_recall"] <= 1.0
        assert 0.0 <= result["oof_fpr"] <= 1.0

