"""Unit tests for EXP-02 feature engineering and LeakFreePreprocessorV2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.v2_exp02_feature_engineering import (
    BoostedDNNV2,
    LeakFreePreprocessorV2,
    compute_domain_features,
)


@pytest.fixture
def dummy_traffic_df():
    np.random.seed(42)
    n = 50
    data = {
        "spkts": np.random.randint(0, 50, size=n).astype(float),
        "dpkts": np.random.randint(0, 50, size=n).astype(float),
        "sbytes": np.random.randint(0, 5000, size=n).astype(float),
        "dbytes": np.random.randint(0, 5000, size=n).astype(float),
        "sttl": np.random.choice([0, 31, 62, 254], size=n).astype(float),
        "dttl": np.random.choice([0, 29, 252], size=n).astype(float),
        "sload": np.random.uniform(0.0, 100000.0, size=n).astype(float),
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
        "label": np.random.choice([0, 1], size=n),
        "attack_cat": np.random.choice(["Normal", "Generic", "Fuzzers"], size=n),
    }
    return pd.DataFrame(data)


class TestDomainFeatureComputation:
    """Test safety and mathematical validity of engineered features."""

    def test_feature_computation_zeros_no_division_error(self):
        # Force exact zeros in denominators
        df = pd.DataFrame({
            "spkts": [0.0, 0.0],
            "dpkts": [0.0, 0.0],
            "sbytes": [0.0, 0.0],
            "dbytes": [0.0, 0.0],
            "sttl": [0.0, 0.0],
            "dttl": [0.0, 0.0],
            "sload": [0.0, 0.0],
            "dload": [0.0, 0.0],
        })
        out = compute_domain_features(df)
        assert np.all(np.isfinite(out["bytes_per_pkt_src"]))
        assert np.all(np.isfinite(out["bytes_per_pkt_dst"]))
        assert np.all(np.isfinite(out["packet_asymmetry"]))
        assert np.all(np.isfinite(out["byte_asymmetry"]))
        assert np.all(np.isfinite(out["ttl_ratio"]))
        assert np.all(np.isfinite(out["load_ratio"]))

    def test_asymmetry_bounded_range(self, dummy_traffic_df):
        out = compute_domain_features(dummy_traffic_df)
        assert np.all(out["packet_asymmetry"] >= -1.0)
        assert np.all(out["packet_asymmetry"] <= 1.0)
        assert np.all(out["byte_asymmetry"] >= -1.0)
        assert np.all(out["byte_asymmetry"] <= 1.0)


class TestLeakFreePreprocessorV2:
    """Test preprocessor V2 dimensionality and leak-free fitting."""

    def test_preprocessor_v2_exact_72_features(self, dummy_traffic_df):
        prep = LeakFreePreprocessorV2()
        prep.fit(dummy_traffic_df)
        assert len(prep.feature_names_) == 72
        X = prep.transform(dummy_traffic_df)
        assert X.shape == (len(dummy_traffic_df), 72)
        assert np.all(np.isfinite(X))

    def test_preprocessor_v2_zero_leakage_from_val(self, dummy_traffic_df):
        df_tr = dummy_traffic_df.iloc[:30].copy()
        df_val = dummy_traffic_df.iloc[30:].copy()

        # Extreme values in val must not affect scaler
        df_val["sbytes"] = 999999999.0
        prep = LeakFreePreprocessorV2()
        prep.fit(df_tr)

        sbytes_idx = prep.numeric_cols_.index("sbytes")
        fitted_mean = prep.scaler_.mean_[sbytes_idx]
        expected_mean = np.log1p(df_tr["sbytes"].values).mean()
        assert np.isclose(fitted_mean, expected_mean, atol=1e-4)


class TestBoostedDNNV2:
    """Test neural network forward pass with 72 inputs."""

    def test_forward_pass_72_features(self):
        model = BoostedDNNV2(in_features=72)
        x = torch.randn(8, 72)
        out = model(x)
        assert out.shape == (8, 1)
        assert torch.all(torch.isfinite(out))

