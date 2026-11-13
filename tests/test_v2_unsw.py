"""Unit tests for Stage 5 UNSW-NB15 cross-dataset adapter and pipeline evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from ids_rag.config import get_project_root
from ids_rag.v2.pipeline_v2 import PipelineV2Context, run_flow_v2
from ids_rag.v2.prepare_extra import load_v2_config
from ids_rag.v2.unsw_adapter import (
    get_unsw_benchmark_samples,
    map_unsw_to_cicids41,
)


@pytest.fixture(scope="module")
def shared_ctx():
    cfg = load_v2_config()
    return PipelineV2Context.load(cfg)


class TestUNSWAdapter:
    """Test schema mapping from UNSW-NB15 to CICIDS2018 41 features."""

    def test_map_unsw_to_cicids41_keys(self, shared_ctx):
        sample = {
            "proto": "tcp",
            "dur": 1.0,
            "spkts": 10,
            "dpkts": 8,
            "sbytes": 1000,
            "dbytes": 800,
            "rate": 18.0,
            "swin": 65535,
            "dwin": 65535,
            "state": "FIN",
        }
        mapped = map_unsw_to_cicids41(sample, shared_ctx.feature_names)
        assert len(mapped) == 41
        for fn in shared_ctx.feature_names:
            assert fn in mapped
            assert np.isfinite(mapped[fn])

    def test_duration_scaling(self, shared_ctx):
        # UNSW dur in seconds (1.5s) -> CICIDS microseconds (1,500,000 us)
        sample = {"dur": 1.5, "proto": "tcp"}
        mapped = map_unsw_to_cicids41(sample, shared_ctx.feature_names)
        assert mapped["Flow Duration"] == 1_500_000.0

    def test_udp_protocol_mapping(self, shared_ctx):
        sample = {"dur": 0.1, "proto": "udp"}
        mapped = map_unsw_to_cicids41(sample, shared_ctx.feature_names)
        assert mapped["Protocol"] == 17.0

    def test_benchmark_samples_mapping_and_inference(self, shared_ctx, tmp_path):
        samples = get_unsw_benchmark_samples()
        assert len(samples) >= 4

        for s in samples:
            mapped_features = map_unsw_to_cicids41(s["unsw_features"], shared_ctx.feature_names)
            assert len(mapped_features) == 41

            # Run through V2 pipeline with generate=False
            summary = run_flow_v2(
                raw_features=mapped_features,
                ctx=shared_ctx,
                flow_id=f"unsw_test_{s['attack_cat']}",
                modes="rag",
                generate=False,
                out_dir=tmp_path / "artifacts" / "v2" / "reports" / "flows",
            )

            assert summary["predicted_class"] in ("Bot", "FTP-BruteForce", "SSH-Bruteforce")
            assert 0.0 <= summary["confidence"] <= 1.0
            assert summary["tier"] in ("VERY_HIGH", "HIGH", "MEDIUM", "LOW")
            assert len(summary["chunks"]) >= 5

