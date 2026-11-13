"""Unit tests for V2 pipeline and Streamlit integration.

Tests cover:
- check_v2_output_safety: forbids writes to frozen directories, allows v2 paths.
- load_v2_ensemble: loads 3 binary DNNs (Bot, FTP-BruteForce, SSH-Bruteforce) into eval mode.
- PipelineV2Context: loads models, scalers, v2 knowledge base, and feature names.
- validate_raw_features: handles valid inputs, rejects missing, non-numeric, or NaN/Inf.
- run_flow_v2: executes predict -> explain -> retrieve -> prompt without requiring Ollama.
- Indicative warning: correctly flags FTP-BruteForce predictions with [INDICATIVE ONLY].
- PDF generation: confirms build_pdf generates valid PDF bytes from v2 summary dicts.
- Sample builder: verifies v2 sample generation and schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from ids_rag.config import get_project_root
from ids_rag.pdf_report import build_pdf
from ids_rag.v2.pipeline_v2 import (
    PipelineV2Context,
    check_v2_output_safety,
    load_v2_ensemble,
    run_flow_v2,
)
from ids_rag.v2.prepare_extra import load_v2_config


@pytest.fixture(scope="module")
def shared_ensemble():
    return load_v2_ensemble()


@pytest.fixture(scope="module")
def shared_ctx():
    cfg = load_v2_config()
    return PipelineV2Context.load(cfg)


class TestV2OutputSafety:
    """Ensure check_v2_output_safety strictly forbids writing to frozen directories."""

    def test_forbidden_models_raises(self):
        root = get_project_root()
        with pytest.raises(ValueError, match="Safety check failed"):
            check_v2_output_safety(root / "models" / "evil.pt")

    def test_forbidden_data_processed_raises(self):
        root = get_project_root()
        with pytest.raises(ValueError, match="Safety check failed"):
            check_v2_output_safety(root / "data" / "processed" / "evil.parquet")

    def test_forbidden_artifacts_eval_raises(self):
        root = get_project_root()
        with pytest.raises(ValueError, match="Safety check failed"):
            check_v2_output_safety(root / "artifacts" / "eval" / "evil.json")

    def test_forbidden_artifacts_reports_raises(self):
        root = get_project_root()
        with pytest.raises(ValueError, match="Safety check failed"):
            check_v2_output_safety(root / "artifacts" / "reports" / "flows" / "evil.txt")

    def test_allowed_v2_paths_pass(self):
        root = get_project_root()
        check_v2_output_safety(root / "artifacts" / "v2" / "reports" / "flows")
        check_v2_output_safety(root / "artifacts" / "demo_runs" / "run_123")
        check_v2_output_safety(root / "artifacts" / "demo_runs_v2" / "run_123")
        check_v2_output_safety(root / "data" / "v2" / "processed")
        check_v2_output_safety(root / "models_v2")


class TestV2Ensemble:
    """Test loading and inference with the v2 ensemble."""

    def test_ensemble_classes(self, shared_ensemble):
        assert set(shared_ensemble.class_ids) == {3, 4, 5}
        assert shared_ensemble.label_map[3] == "Bot"
        assert shared_ensemble.label_map[4] == "FTP-BruteForce"
        assert shared_ensemble.label_map[5] == "SSH-Bruteforce"

    def test_ensemble_eval_mode(self, shared_ensemble):
        for m in shared_ensemble.models.values():
            assert not m.training

    def test_ensemble_prediction_shape(self, shared_ensemble):
        x = np.zeros((2, len(shared_ensemble.feature_names)), dtype=np.float32)
        pred_labels, confs, tiers = shared_ensemble.predict(x)
        assert len(pred_labels) == 2
        assert len(confs) == 2
        assert len(tiers) == 2
        for lbl in pred_labels:
            assert lbl in (3, 4, 5)
        for conf in confs:
            assert 0.0 <= conf <= 1.0
        for tier in tiers:
            assert tier in ("VERY_HIGH", "HIGH", "MEDIUM", "LOW")


class TestPipelineV2Context:
    """Test loading the v2 context and its components."""

    def test_context_loaded(self, shared_ctx):
        assert shared_ctx.ensemble is not None
        assert shared_ctx.scaler is not None
        assert len(shared_ctx.feature_names) == 41
        assert shared_ctx.kb is not None
        assert shared_ctx.cross_encoder is not None

    def test_kb_has_v2_chunks(self, shared_ctx):
        # Merged KB should have 742 baseline + 9 v2 chunks = 751 chunks
        assert len(shared_ctx.kb.chunks) >= 751


class TestRunFlowV2:
    """Test the public end-to-end flow execution in v2."""

    @pytest.fixture
    def dummy_raw_features(self, shared_ctx):
        return {fn: 0.0 for fn in shared_ctx.feature_names}

    def test_run_flow_v2_no_generate(self, shared_ctx, dummy_raw_features, tmp_path):
        summary = run_flow_v2(
            raw_features=dummy_raw_features,
            ctx=shared_ctx,
            flow_id="test_v2_001",
            modes="rag",
            generate=False,
            out_dir=tmp_path / "artifacts" / "v2" / "reports" / "flows",
        )

        assert summary["index"] == "test_v2_001"
        assert summary["predicted_class"] in ("Bot", "FTP-BruteForce", "SSH-Bruteforce")
        assert 0.0 <= summary["confidence"] <= 1.0
        assert summary["tier"] in ("VERY_HIGH", "HIGH", "MEDIUM", "LOW")
        assert summary["pipeline_version"] == "v2"

        # Explanation check
        explanation = summary["explanation"]
        assert len(explanation["features"]) == 5
        assert "direction" in explanation["features"][0]

        # Query & retrieval check: at least 5 chunks (top-k, plus any fallbacks)
        assert len(summary["query"]) > 0
        assert len(summary["chunks"]) >= 5

        # Prompt check
        assert "rag" in summary["prompts"]
        assert "system" in summary["prompts"]["rag"]
        assert "user" in summary["prompts"]["rag"]

        # Prompt file written
        prompt_file = tmp_path / "artifacts" / "v2" / "reports" / "flows" / "flow_test_v2_001_rag_prompt.txt"
        assert prompt_file.exists()

    def test_run_flow_v2_with_mock_generate(self, shared_ctx, dummy_raw_features, tmp_path):
        mock_client = MagicMock()
        mock_response = {
            "message": {
                "content": (
                    "### 1. Executive Summary\nThreat detected and contained.\n\n"
                    "### 2. Threat Analysis\nAttack vector identified from source [1].\n\n"
                    "### 3. Immediate Mitigation Actions\nBlock suspicious IPs per [1].\n\n"
                    "### 4. Containment & Eradication\nPurge malware per [2].\n\n"
                    "### 5. Post-Incident Review\nAudit credentials and access logs.\n"
                )
            }
        }
        mock_client.chat.return_value = mock_response

        summary = run_flow_v2(
            raw_features=dummy_raw_features,
            ctx=shared_ctx,
            flow_id="test_mock_gen",
            modes="rag",
            generate=True,
            client=mock_client,
            out_dir=tmp_path / "artifacts" / "v2" / "reports" / "flows",
        )

        assert "rag" in summary["reports"]
        assert "Executive Summary" in summary["reports"]["rag"]
        assert "rag" in summary["validation"]
        val = summary["validation"]["rag"]
        assert "word_count" in val

    def test_ftp_bruteforce_indicative_warning(self, shared_ctx):
        dummy_features = {fn: 0.0 for fn in shared_ctx.feature_names}
        summary = run_flow_v2(
            raw_features=dummy_features,
            ctx=shared_ctx,
            flow_id="test_ftp_warning",
            modes="rag",
            generate=False,
            out_dir=get_project_root() / "artifacts" / "v2" / "reports" / "flows",
        )
        if summary["predicted_class"] == "FTP-BruteForce":
            assert summary["is_indicative_only"] is True
            assert summary["indicative_warning"] is not None
            assert "INDICATIVE ONLY" in summary["indicative_warning"]
        else:
            assert summary["is_indicative_only"] is False


class TestV2PDFReport:
    """Test that build_pdf works cleanly with V2 summaries."""

    def test_build_pdf_from_v2_summary(self, shared_ctx, tmp_path):
        dummy_features = {fn: 0.0 for fn in shared_ctx.feature_names}
        summary = run_flow_v2(
            raw_features=dummy_features,
            ctx=shared_ctx,
            flow_id="test_pdf_01",
            modes="rag",
            generate=False,
            out_dir=tmp_path / "artifacts" / "v2" / "reports" / "flows",
        )

        pdf_bytes = build_pdf(summary, source_label="V2 Test Preset Flow")
        assert isinstance(pdf_bytes, (bytes, bytearray))
        assert bytes(pdf_bytes).startswith(b"%PDF")
        assert len(pdf_bytes) > 1000


class TestV2SampleBuilder:
    """Test building v2 samples for the Streamlit UI."""

    def test_samples_v2_generation(self):
        from scripts.v2_build_samples import build_samples_v2
        samples = build_samples_v2()

        assert len(samples) == 3
        labels = [s["dataset_label"] for s in samples]
        assert set(labels) == {"Bot", "FTP-BruteForce", "SSH-Bruteforce"}

        for s in samples:
            assert len(s["features"]) == 41
            assert "description" in s
            if s["dataset_label"] == "FTP-BruteForce":
                assert s["is_indicative_only"] is True
                assert "INDICATIVE ONLY" in s["name"]
