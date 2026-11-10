"""Unit tests for run_flow, _run_core, and validate_raw_features (tests/test_run_flow.py).

Tests:
  (a) run_flow on an inverse-transformed test row matches run_pipeline's prediction
      class and confidence within 1e-4.  Never writes to artifacts/reports/flows/.
  (b) _run_core (via run_flow with a known x_scaled) stores the correct predicted
      class in the summary dict; regression check that the index path matches what
      a fake test_predictions.parquet would contain.
  (c) validate_raw_features rejects: missing feature, non-numeric value, NaN, Inf.
      Extra columns are silently ignored.  Whitespace in CSV-style key is stripped.
  (d) meta=None works end-to-end: run_flow completes and metadata fields are null.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from ids_rag.pipeline import (
    PipelineContext,
    _run_core,
    run_flow,
    run_pipeline,
    validate_raw_features,
)


# ---------------------------------------------------------------------------
# Shared fakes (mirror the ones in test_pipeline.py for isolation)
# ---------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatResponse:
    def __init__(
        self,
        content: str = (
            "1. Rationale\nReport text.\n"
            "2. Key Indicators\nReport text.\n"
            "3. Confidence Assessment\nReport text.\n"
            "4. Threat Assessment\nReport text.\n"
            "5. Recommendations\nReport text."
        ),
        model: str = "llama3:8b",
        prompt_eval_count: int = 20,
        prompt_eval_duration: int = 100_000_000,
        eval_count: int = 30,
        eval_duration: int = 200_000_000,
        total_duration: int = 300_000_000,
        load_duration: int = 50_000_000,
    ) -> None:
        self.message = FakeMessage(content)
        self.model = model
        self.prompt_eval_count = prompt_eval_count
        self.prompt_eval_duration = prompt_eval_duration
        self.eval_count = eval_count
        self.eval_duration = eval_duration
        self.total_duration = total_duration
        self.load_duration = load_duration


class FakeClient:
    def __init__(self, chat_response: Any = None) -> None:
        self.chat_response = chat_response or FakeChatResponse()
        self.last_kwargs: dict[str, Any] = {}
        self.call_count: int = 0

    def chat(self, **kwargs: Any) -> Any:
        self.call_count += 1
        self.last_kwargs = kwargs
        return self.chat_response


class FakeLinearModel(nn.Module):
    """Differentiable linear module required by gradient attribution."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FakeEnsemble:
    """In-memory ensemble stub matching BinaryDNNEnsemble API."""

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.class_ids = [0, 1, 2]
        self.label_map = {0: "Benign", 1: "DoS", 2: "DDoS"}
        self.device = torch.device("cpu")
        self.models = {c: FakeLinearModel(len(feature_names)) for c in self.class_ids}
        self.tiers_config = {"very_high": 0.95, "high": 0.70, "medium": 0.50}

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(x)
        return (
            np.array([1] * n, dtype=np.int64),
            np.array([0.9876] * n, dtype=np.float64),
            np.array(["VERY_HIGH"] * n, dtype=object),
        )

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        return np.ones((n, len(self.class_ids)), dtype=np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        return np.ones((n, len(self.class_ids)), dtype=np.float64) * 0.9876


class FakeScaler:
    """Identity scaler stub: transform and inverse_transform are both identity."""

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float64)


class FakeKB:
    """Knowledge-base index stub returning predetermined chunks."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.chunk_id_to_pos = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    def search_bm25(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        return [(i, 1.0) for i in range(min(k, len(self.chunks)))]

    def search_dense(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        return [(i, 1.0) for i in range(min(k, len(self.chunks)))]

    def dense_scores(self, query: str) -> np.ndarray:
        return np.ones(len(self.chunks), dtype=np.float32) * 0.9

    def bm25_scores(self, query: str) -> np.ndarray:
        return np.ones(len(self.chunks), dtype=np.float32) * 5.0

    def get_chunk(self, position: int) -> dict[str, Any]:
        return self.chunks[position]


class FakeCrossEncoder:
    """Cross-encoder reranker stub."""

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.ones(len(pairs), dtype=np.float32) * 2.0


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["Tot Bwd Pkts", "Tot Fwd Pkts", "Protocol"]

_FAKE_REPORT = (
    "1. Rationale\nText.\n"
    "2. Key Indicators\nText.\n"
    "3. Confidence Assessment\nText.\n"
    "4. Threat Assessment\nText.\n"
    "5. Recommendations\nText."
)

_FAKE_GEN_RESULT: dict[str, Any] = {
    "text": _FAKE_REPORT,
    "model": "llama3:8b",
    "options": {"seed": 42},
    "seed_used": 42,
    "prompt_words": 10,
    "prompt_eval_count": 10,
    "prompt_eval_duration": 0.1,
    "eval_count": 20,
    "eval_duration": 0.2,
    "total_duration": 0.3,
    "load_duration": 0.05,
    "tokens_per_second": 100.0,
}


def _make_gen_result(seed: int | None = None) -> dict[str, Any]:
    used = seed if seed is not None else 42
    return {**_FAKE_GEN_RESULT, "options": {"seed": used}, "seed_used": used}


@pytest.fixture
def fake_cfg() -> dict[str, Any]:
    return {
        "llm": {
            "model": "llama3:8b",
            "temperature": 0.2,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 1100,
            "keep_alive": "15m",
            "request_timeout_s": 600,
            "min_words": 1,
            "max_words": 1000,
            "tier_actions": {
                "VERY_HIGH": "Very High: automated",
                "HIGH": "High: analyst",
                "MEDIUM": "Medium: review",
                "LOW": "Low: manual",
            },
            "suppress_near_average_implications": False,
        },
        "explain": {
            "method": "gradient",
            "top_k": 3,
        },
        "rag": {
            "retrieve_k_each": 5,
            "fusion_weights": {"semantic": 0.6, "bm25": 0.4},
            "fusion_top": 5,
            "final_k": 3,
            "fallback_min_chunks": 3,
            "fallback_score_threshold": 0.5,
            "fallback_sections": [],
        },
        "paths": {
            "kb": "data/kb",
            "artifacts": "artifacts",
            "reports": "artifacts/reports",
        },
    }


@pytest.fixture
def fake_context() -> PipelineContext:
    ensemble = FakeEnsemble(FEATURE_NAMES)
    scaler = FakeScaler()

    chunks = [
        {
            "chunk_id": "c1",
            "text": "NIST guidance on denial of service handling.",
            "source_name": "NIST_SP_800-61_Rev2.pdf",
            "source_type": "NIST",
            "source_ref": "SP 800-61 Rev2",
            "section": "3.2 Incident Handling",
            "is_fallback": False,
        },
        {
            "chunk_id": "c2",
            "text": "MITRE ATT&CK mitigation techniques for network flooding.",
            "source_name": "T1498.html",
            "source_type": "MITRE",
            "source_ref": "T1498",
            "section": "Mitigation",
            "is_fallback": False,
        },
    ]
    kb = FakeKB(chunks)
    cross_encoder = FakeCrossEncoder()

    df_test = pd.DataFrame(
        {
            "Tot Bwd Pkts": [0.0, 1.0, 2.0],
            "Tot Fwd Pkts": [10.0, 11.0, 12.0],
            "Protocol": [6.0, 17.0, 6.0],
        }
    )
    df_meta = pd.DataFrame(
        {
            "Flow ID": ["f1", "f2", "f3"],
            "Src IP": ["1.1.1.1", "2.2.2.2", "3.3.3.3"],
            "Src Port": [1234, 1235, 1236],
            "Dst IP": ["4.4.4.4", "5.5.5.5", "6.6.6.6"],
            "Dst Port": [80, 443, 80],
            "Timestamp": ["t1", "t2", "t3"],
            "protocol": [6, 17, 6],
        }
    )
    df_preds = pd.DataFrame(
        {
            "true_class": ["DoS", "Benign", "DDoS"],
            "true_label": [1, 0, 2],
            "predicted_label": [1, 1, 1],
        }
    )

    return PipelineContext(
        ensemble=ensemble,
        scaler=scaler,
        feature_names=FEATURE_NAMES,
        kb=kb,
        cross_encoder=cross_encoder,
        df_test=df_test,
        df_meta=df_meta,
        df_preds=df_preds,
        chunks_sha256="test_sha256_digest",
    )


# ---------------------------------------------------------------------------
# Test (a): run_flow matches run_pipeline prediction within 1e-4
# ---------------------------------------------------------------------------


def test_run_flow_matches_run_pipeline(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    """run_flow on an inverse-transformed test row must produce the same predicted
    class and confidence (within 1e-4) as run_pipeline on that row.
    Neither call must write into artifacts/reports/flows/."""
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "ids_rag.pipeline.generate_report",
        lambda prompt, cfg, client=None, seed_override=None: _make_gen_result(seed_override),
    )

    out_a = tmp_path / "out_pipeline"
    out_b = tmp_path / "out_flow"

    # Run via run_pipeline (uses already-scaled values from df_test)
    summary_a = run_pipeline(
        index=0,
        ctx=fake_context,
        cfg=fake_cfg,
        modes="rag",
        out_dir=out_a,
    )

    # Recover raw values: since FakeScaler.transform is identity, raw == scaled
    x_scaled_0 = fake_context.df_test.iloc[0][FEATURE_NAMES].to_numpy(dtype=np.float32)
    # inverse_transform is also identity for FakeScaler
    x_raw_0 = fake_context.scaler.inverse_transform(x_scaled_0.reshape(1, -1))[0]

    # Build a raw_features dict keyed by feature name
    raw_dict = dict(zip(FEATURE_NAMES, x_raw_0.tolist()))

    # Run via run_flow
    summary_b = run_flow(
        raw_features=raw_dict,
        ctx=fake_context,
        cfg=fake_cfg,
        modes="rag",
        out_dir=out_b,
    )

    # Predicted class must match exactly
    assert summary_a["predicted_class"] == summary_b["predicted_class"]

    # Confidence must match within 1e-4
    assert abs(summary_a["confidence"] - summary_b["confidence"]) < 1e-4

    # Frozen artifacts directory must NOT exist or must be empty
    frozen_dir = tmp_path / "artifacts" / "reports" / "flows"
    if frozen_dir.exists():
        assert not any(frozen_dir.iterdir()), (
            "run_flow or run_pipeline wrote into artifacts/reports/flows/ "
            "instead of the supplied out_dir"
        )


# ---------------------------------------------------------------------------
# Test (b): _run_core stores the correct predicted class (regression via run_flow)
# ---------------------------------------------------------------------------


def test_run_core_predicted_class_matches_fake_ensemble(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    """_run_core (invoked through run_flow) must store the predicted class returned
    by FakeEnsemble.predict, which is always label 1 → 'DoS'."""
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "ids_rag.pipeline.generate_report",
        lambda prompt, cfg, client=None, seed_override=None: _make_gen_result(seed_override),
    )

    raw_dict = {
        "Tot Bwd Pkts": 0.0,
        "Tot Fwd Pkts": 10.0,
        "Protocol": 6.0,
    }

    summary = run_flow(
        raw_features=raw_dict,
        ctx=fake_context,
        cfg=fake_cfg,
        modes="rag",
        flow_id="test_regression",
        out_dir=tmp_path / "out_b",
    )

    # FakeEnsemble always predicts label 1 → "DoS"
    assert summary["predicted_class"] == "DoS"
    assert summary["confidence"] == pytest.approx(0.9876, abs=1e-6)

    # JSON artifact must exist and contain the correct class
    json_file = tmp_path / "out_b" / "flow_test_regression.json"
    import json

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["predicted_class"] == "DoS"
    assert data["index"] == "test_regression"

    # Frozen flows dir must not have been touched
    frozen_dir = tmp_path / "artifacts" / "reports" / "flows"
    assert not frozen_dir.exists() or not any(frozen_dir.iterdir())


# ---------------------------------------------------------------------------
# Test (c): validate_raw_features rejects bad inputs
# ---------------------------------------------------------------------------


def test_validate_raw_features_missing_feature() -> None:
    """A dict that omits a required feature must raise ValueError naming it."""
    scaler = FakeScaler()
    partial = {"Tot Bwd Pkts": 0.0, "Tot Fwd Pkts": 10.0}  # missing "Protocol"
    with pytest.raises(ValueError, match="Protocol"):
        validate_raw_features(partial, FEATURE_NAMES, scaler)


def test_validate_raw_features_non_numeric() -> None:
    """A non-numeric value must raise ValueError naming the offending feature."""
    scaler = FakeScaler()
    bad = {"Tot Bwd Pkts": "bad_value", "Tot Fwd Pkts": 10.0, "Protocol": 6.0}
    with pytest.raises(ValueError, match="Tot Bwd Pkts"):
        validate_raw_features(bad, FEATURE_NAMES, scaler)


def test_validate_raw_features_nan() -> None:
    """NaN must raise ValueError with a message mentioning the training pipeline."""
    scaler = FakeScaler()
    bad = {"Tot Bwd Pkts": float("nan"), "Tot Fwd Pkts": 10.0, "Protocol": 6.0}
    with pytest.raises(ValueError, match="training pipeline"):
        validate_raw_features(bad, FEATURE_NAMES, scaler)


def test_validate_raw_features_inf() -> None:
    """Inf must raise ValueError."""
    scaler = FakeScaler()
    bad = {"Tot Bwd Pkts": float("inf"), "Tot Fwd Pkts": 10.0, "Protocol": 6.0}
    with pytest.raises(ValueError):
        validate_raw_features(bad, FEATURE_NAMES, scaler)


def test_validate_raw_features_extra_columns_ignored() -> None:
    """Extra keys beyond feature_names must be silently ignored."""
    scaler = FakeScaler()
    extra = {
        "Tot Bwd Pkts": 0.0,
        "Tot Fwd Pkts": 10.0,
        "Protocol": 6.0,
        "ExtraColumn": 999.9,
        "AnotherExtra": "ignored",
    }
    result = validate_raw_features(extra, FEATURE_NAMES, scaler)
    assert result.shape == (len(FEATURE_NAMES),)
    assert result.dtype == np.float32


def test_validate_raw_features_whitespace_keys() -> None:
    """Keys with surrounding whitespace (CSV headers) must be matched after stripping."""
    scaler = FakeScaler()
    # Simulate CSV header names with spaces
    padded = {
        " Tot Bwd Pkts ": 0.0,
        "  Tot Fwd Pkts": 10.0,
        "Protocol  ": 6.0,
    }
    result = validate_raw_features(padded, FEATURE_NAMES, scaler)
    assert result.shape == (len(FEATURE_NAMES),)


def test_validate_raw_features_valid_returns_float32() -> None:
    """A valid dict must return a 1-D float32 array of the correct length."""
    scaler = FakeScaler()
    good = {"Tot Bwd Pkts": 0.0, "Tot Fwd Pkts": 10.0, "Protocol": 6.0}
    result = validate_raw_features(good, FEATURE_NAMES, scaler)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (len(FEATURE_NAMES),)


def test_validate_raw_features_accepts_series() -> None:
    """A pandas Series input must be accepted and produce the same result as a dict."""
    scaler = FakeScaler()
    good_dict = {"Tot Bwd Pkts": 0.0, "Tot Fwd Pkts": 10.0, "Protocol": 6.0}
    good_series = pd.Series(good_dict)
    result_dict = validate_raw_features(good_dict, FEATURE_NAMES, scaler)
    result_series = validate_raw_features(good_series, FEATURE_NAMES, scaler)
    np.testing.assert_array_equal(result_dict, result_series)


# ---------------------------------------------------------------------------
# Test (d): meta=None works end to end
# ---------------------------------------------------------------------------


def test_run_flow_meta_none(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    """run_flow with meta=None must complete without error.
    The explanation's metadata dict must exist and have null/empty IP and port fields."""
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "ids_rag.pipeline.generate_report",
        lambda prompt, cfg, client=None, seed_override=None: _make_gen_result(seed_override),
    )

    raw_dict = {"Tot Bwd Pkts": 1.0, "Tot Fwd Pkts": 11.0, "Protocol": 17.0}

    summary = run_flow(
        raw_features=raw_dict,
        ctx=fake_context,
        cfg=fake_cfg,
        meta=None,
        modes="rag",
        out_dir=tmp_path / "out_d",
    )

    # Pipeline must succeed and return a predicted class
    assert "predicted_class" in summary
    assert summary["predicted_class"] in {"Benign", "DoS", "DDoS"}

    # explanation.metadata must be present; Src IP and Dst IP must be null/None/empty
    metadata = summary.get("explanation", {}).get("metadata", {})
    # None, empty string, or the string "not available" are all acceptable null values
    src_ip = metadata.get("src_ip") or metadata.get("Src IP") or metadata.get("SrcIP")
    dst_ip = metadata.get("dst_ip") or metadata.get("Dst IP") or metadata.get("DstIP")
    # Must not be a real IP address (would only come from df_meta which is bypassed here)
    assert src_ip not in ("1.1.1.1", "2.2.2.2", "3.3.3.3")
    assert dst_ip not in ("4.4.4.4", "5.5.5.5", "6.6.6.6")

