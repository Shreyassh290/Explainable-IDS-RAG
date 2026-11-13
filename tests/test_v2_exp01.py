"""Unit tests for EXP-01 leak-free preprocessor and Focal Loss implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from scripts.v2_exp01_loss_and_leakage import (
    BoostedDNN,
    FocalLoss,
    LeakFreePreprocessor,
    analyze_threshold_pareto,
)


class TestFocalLoss:
    """Test mathematical and numerical properties of FocalLoss."""

    def test_focal_loss_zero_when_perfect(self):
        loss_fn = FocalLoss(alpha=0.5, gamma=2.0)
        # Extremely large positive logit for y=1 -> probability ~ 1.0
        logits = torch.tensor([[50.0], [-50.0]])
        targets = torch.tensor([[1.0], [0.0]])
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        assert loss.item() < 1e-4

    def test_focal_loss_penalty_on_mistake(self):
        loss_fn = FocalLoss(alpha=0.5, gamma=2.0)
        # Confidently wrong
        logits = torch.tensor([[-20.0], [20.0]])
        targets = torch.tensor([[1.0], [0.0]])
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        assert loss.item() > 5.0

    def test_focal_loss_no_nan_on_extreme_values(self):
        loss_fn = FocalLoss(alpha=0.5, gamma=2.0)
        logits = torch.tensor([[-1000.0], [1000.0], [0.0], [1e-6]])
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_focal_loss_reductions(self):
        logits = torch.tensor([[1.0], [-1.0]])
        targets = torch.tensor([[1.0], [0.0]])
        loss_mean = FocalLoss(reduction="mean")(logits, targets)
        loss_sum = FocalLoss(reduction="sum")(logits, targets)
        loss_none = FocalLoss(reduction="none")(logits, targets)

        assert torch.isclose(loss_mean * 2, loss_sum)
        assert loss_none.shape == (2, 1)


class TestLeakFreePreprocessor:
    """Verify leak-free preprocessing, feature count, and unknown category handling."""

    @pytest.fixture
    def dummy_data(self):
        np.random.seed(42)
        n = 100
        data = {
            "dur": np.random.exponential(1.0, size=n),
            "spkts": np.random.randint(1, 50, size=n),
            "dpkts": np.random.randint(0, 50, size=n),
            "sbytes": np.random.randint(100, 5000, size=n),
            "dbytes": np.random.randint(0, 5000, size=n),
            "rate": np.random.uniform(0.1, 100.0, size=n),
            "sttl": np.random.choice([31, 62, 254], size=n),
            "dttl": np.random.choice([0, 29, 252], size=n),
            "sload": np.random.uniform(100, 100000, size=n),
            "dload": np.random.uniform(0, 100000, size=n),
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
            "proto": np.random.choice(["tcp", "udp", "arp", "ospf", "sctp", "gre"], size=n),
            "service": np.random.choice(["-", "http", "dns", "ftp", "smtp", "ssh"], size=n),
            "state": np.random.choice(["FIN", "INT", "CON", "REQ", "RST"], size=n),
            "label": np.random.choice([0, 1], size=n),
            "attack_cat": np.random.choice(["Normal", "Generic", "Exploits"], size=n),
            "id": np.arange(n),
        }
        return pd.DataFrame(data)

    def test_preprocessor_does_not_fit_on_val_or_test(self, dummy_data):
        df_tr = dummy_data.iloc[:60].copy()
        df_val = dummy_data.iloc[60:80].copy()
        df_test = dummy_data.iloc[80:].copy()

        # Artificially alter a feature in val/test to an extreme value
        df_val["dur"] = 9999999.0
        df_test["dur"] = 8888888.0

        prep = LeakFreePreprocessor()
        prep.fit(df_tr)

        # Scaler mean must match df_tr, unaffected by df_val or df_test
        expected_dur_log = np.log1p(df_tr["dur"].values)
        dur_col_idx = prep.numeric_cols_.index("dur")
        fitted_mean = prep.scaler_.mean_[dur_col_idx]

        assert np.isclose(fitted_mean, expected_dur_log.mean(), atol=1e-4)

    def test_preprocessor_handles_unseen_categories(self, dummy_data):
        df_tr = dummy_data.iloc[:70].copy()
        df_test = dummy_data.iloc[70:].copy()

        # Introduce never-before-seen protocol and service in test
        df_test.loc[df_test.index[0], "proto"] = "quantum_proto"
        df_test.loc[df_test.index[0], "service"] = "alien_service"
        df_test.loc[df_test.index[0], "state"] = "SUPERPOSITION"

        prep = LeakFreePreprocessor()
        prep.fit(df_tr)

        # Transforming test should not raise and produce finite array of identical width
        X_test = prep.transform(df_test)
        assert X_test.shape[1] == len(prep.feature_names_)
        assert len(prep.feature_names_) == 66
        assert np.all(np.isfinite(X_test))

    def test_preprocessor_feature_count_on_full_unsw(self):
        from ids_rag.config import get_project_root
        data_dir = get_project_root() / "data" / "unsw"
        train_path = data_dir / "UNSW_NB15_training-set.csv"
        if not train_path.exists():
            pytest.skip("Full UNSW training data not found.")

        # Read small sample to verify 66 feature schema
        df = pd.read_csv(train_path, nrows=5000)
        prep = LeakFreePreprocessor()
        prep.fit(df)
        assert len(prep.feature_names_) == 66


class TestParetoAnalysis:
    """Test threshold Pareto analysis logic."""

    def test_pareto_finds_feasible_when_possible(self):
        # Perfect probabilities: normal=0.05, attack=0.95
        y_val = np.array([0] * 50 + [1] * 50)
        val_probs = np.array([0.05] * 50 + [0.95] * 50)

        pareto = analyze_threshold_pareto(val_probs, y_val, target_max_fpr=0.18, target_min_recall=0.96)
        assert pareto["feasible_found"] is True
        assert pareto["best_constrained_tau"] is not None
        assert 0.10 <= pareto["best_constrained_tau"] <= 0.90

    def test_pareto_reports_infeasible_when_unattainable(self):
        # Inverted or noisy probabilities where FPR<=0.18 and Recall>=0.96 is impossible
        y_val = np.array([0] * 50 + [1] * 50)
        val_probs = np.array([0.95] * 50 + [0.05] * 50)

        pareto = analyze_threshold_pareto(val_probs, y_val, target_max_fpr=0.10, target_min_recall=0.99)
        assert pareto["feasible_found"] is False
        assert pareto["best_constrained_tau"] is None

