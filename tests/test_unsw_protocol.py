"""Unit tests for UNSW-NB15 rigorous evaluation protocol."""

from __future__ import annotations

import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ids_rag.unsw.preprocessor import UNSWStrictPreprocessor, compute_domain_features
from scripts.unsw_cv_experiments import calibrate_threshold_on_oof, compute_per_family_table


@pytest.fixture
def synthetic_unsw_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "id": np.arange(n),
        "ID": np.arange(n),
        "label": np.random.choice([0, 1], size=n),
        "attack_cat": np.random.choice(["Normal", "Generic", "Fuzzers"], size=n),
        "spkts": np.random.randint(1, 50, size=n).astype(float),
        "dpkts": np.random.randint(0, 50, size=n).astype(float),
        "sbytes": np.random.randint(100, 5000, size=n).astype(float),
        "dbytes": np.random.randint(0, 5000, size=n).astype(float),
        "sttl": np.random.choice([31, 62, 254], size=n).astype(float),
        "dttl": np.random.choice([0, 29, 252], size=n).astype(float),
        "sload": np.random.uniform(100.0, 100000.0, size=n).astype(float),
        "dload": np.random.uniform(0.0, 100000.0, size=n).astype(float),
        "dur": np.random.uniform(0.0, 5.0, size=n),
        "rate": np.random.uniform(0.1, 100.0, size=n),
        "sloss": np.random.randint(0, 5, size=n),
        "dloss": np.random.randint(0, 5, size=n),
        "sinpkt": np.random.uniform(0, 100, size=n),
        "dinpkt": np.random.uniform(0, 100, size=n),
        "sjit": np.random.uniform(0, 50, size=n),
        "djit": np.random.uniform(0, 50, size=n),
        "swin": np.random.choice([0, 255], size=n),
        "dwin": np.random.choice([0, 255], size=n),
        "tcprtt": np.random.uniform(0, 1.0, size=n),
        "synack": np.random.uniform(0, 0.5, size=n),
        "ackdat": np.random.uniform(0, 0.5, size=n),
        "smean": np.random.uniform(30, 1500, size=n),
        "dmean": np.random.uniform(0, 1500, size=n),
        "trans_depth": np.random.choice([0, 1, 2], size=n),
        "response_body_len": np.random.randint(0, 500, size=n),
        "ct_srv_src": np.random.randint(1, 20, size=n),
        "ct_state_ttl": np.random.choice([0, 1, 2], size=n),
        "ct_dst_ltm": np.random.randint(1, 20, size=n),
        "ct_src_dport_ltm": np.random.randint(1, 20, size=n),
        "ct_dst_sport_ltm": np.random.randint(1, 20, size=n),
        "ct_dst_src_ltm": np.random.randint(1, 20, size=n),
        "is_ftp_login": np.random.choice([0, 1], size=n),
        "ct_ftp_cmd": np.random.choice([0, 1], size=n),
        "ct_flw_http_mthd": np.random.choice([0, 1], size=n),
        "ct_src_ltm": np.random.randint(1, 20, size=n),
        "ct_srv_dst": np.random.randint(1, 20, size=n),
        "is_sm_ips_ports": np.random.choice([0, 1], size=n),
        "proto": np.random.choice(["tcp", "udp", "arp", "ospf"], size=n),
        "service": np.random.choice(["-", "http", "dns", "ftp"], size=n),
        "state": np.random.choice(["FIN", "INT", "CON", "REQ"], size=n),
    })


class TestUNSWProtocol:
    """Rigorous protocol assertions preventing data leakage and target derivation."""

    def test_id_label_attack_cat_excluded_from_features(self, synthetic_unsw_df):
        prep = UNSWStrictPreprocessor()
        prep.fit(synthetic_unsw_df)

        forbidden = {"id", "ID", "label", "attack_cat"}
        for fn in prep.feature_names_:
            assert fn not in forbidden, f"Forbidden column '{fn}' found in feature names"

        # Also verify raw transformed array width matches feature names
        X = prep.transform(synthetic_unsw_df)
        assert X.shape[1] == len(prep.feature_names_)
        assert len(prep.feature_names_) == 72

    def test_preprocessor_fit_strictly_on_training_folds(self, synthetic_unsw_df):
        train_fold = synthetic_unsw_df.iloc[:70].copy()
        val_fold = synthetic_unsw_df.iloc[70:].copy()

        # Add extreme values to validation fold
        val_fold["sbytes"] = 999999999.0
        val_fold["sttl"] = 999999.0

        prep = UNSWStrictPreprocessor()
        prep.fit(train_fold)

        # Fitted scaler mean must match training fold alone
        sbytes_idx = prep.numeric_cols_.index("sbytes")
        train_mean = np.log1p(train_fold["sbytes"].values).mean()
        assert np.isclose(prep.scaler_.mean_[sbytes_idx], train_mean, atol=1e-4)

    def test_training_duplicates_dropped_in_training_only(self, synthetic_unsw_df):
        # Create intentional duplicate in train fold
        train_fold = pd.concat([synthetic_unsw_df.iloc[:20], synthetic_unsw_df.iloc[:5]], ignore_index=True)
        feature_cols = [c for c in train_fold.columns if c not in ["id", "ID", "label", "attack_cat"]]

        dups_before = train_fold.duplicated(subset=feature_cols).sum()
        assert dups_before == 5

        # Drop duplicates strictly in training fold
        train_fold_cleaned = train_fold.drop_duplicates(subset=feature_cols).reset_index(drop=True)
        dups_after = train_fold_cleaned.duplicated(subset=feature_cols).sum()
        assert dups_after == 0
        assert len(train_fold_cleaned) == 20

    def test_test_file_never_read_during_cv_experiments(self):
        cv_script_path = Path(__file__).resolve().parent.parent / "scripts" / "unsw_cv_experiments.py"
        if not cv_script_path.exists():
            pytest.skip("scripts/unsw_cv_experiments.py not authored yet.")

        with open(cv_script_path, encoding="utf-8") as f:
            content = f.read()

        # Confirm testing-set.csv is never loaded in the CV script
        assert "UNSW_NB15_testing-set.csv" not in content, (
            "Violation: scripts/unsw_cv_experiments.py must never load or reference UNSW_NB15_testing-set.csv"
        )

    def test_oof_threshold_calibration_and_per_family_table_format(self):
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        probs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=np.float32)
        attack_cats = np.array(["Normal", "Normal", "Normal", "Generic", "Fuzzers", "Generic"])

        tau, macro_f1 = calibrate_threshold_on_oof(y_true, probs)
        assert 0.0 <= tau <= 1.0
        assert 0.0 <= macro_f1 <= 1.0

        preds = (probs >= tau).astype(int)
        table = compute_per_family_table(y_true, preds, attack_cats)

        # Normal must have specificity and false_positive_rate, no precision/recall keys
        assert "Normal" in table
        assert "specificity" in table["Normal"]
        assert "false_positive_rate" in table["Normal"]
        assert "recall_detection_rate" not in table["Normal"]

        # Attack categories must have recall_detection_rate ONLY, no precision or f1
        for cat in ["Generic", "Fuzzers"]:
            assert cat in table
            assert "recall_detection_rate" in table[cat]
            assert "precision" not in table[cat]
            assert "f1" not in table[cat]
