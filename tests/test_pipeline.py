"""Unit tests for end-to-end incident response inference pipeline (tests/test_pipeline.py).

Tests run entirely in-memory with injected fake contexts and fake clients:
1. run_pipeline with a fake context and fake client writes flow_<index>_rag.txt containing
   exactly the fake model text and a JSON with all required keys.
2. Changing the ground-truth label in the fake test frame does not change the prompt text,
   the retrieval query, or the explanation.
3. mode 'both' writes both report files; --no-generate (generate=False) writes no report file
   and does not call the client.
4. An out-of-range index raises a clear error that names the valid range.
5. With several indices, one failing flow does not stop the others and the exit code is non-zero.
6. --seed overrides llm.seed and the used seed appears in the JSON.
7. pipeline --indices-file writes into custom out_dir and not default.
8. --skip-existing leaves existing JSON intact/byte-identical, avoids client call, runs remaining flows.
9. --summary-csv contains all 20 columns, 1 row per flow/mode, values match JSON.
10. ollama_version stored in flow JSON (subprocess patched).
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from ids_rag.pipeline import PipelineContext, main, run_pipeline


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChatResponse:
    def __init__(
        self,
        content: str = "1. Rationale\nReport text.\n2. Key Indicators\nReport text.\n3. Confidence Assessment\nReport text.\n4. Threat Assessment\nReport text.\n5. Recommendations\nReport text.",
        model: str = "llama3:8b",
        prompt_eval_count: int = 20,
        prompt_eval_duration: int = 100_000_000,
        eval_count: int = 30,
        eval_duration: int = 200_000_000,
        total_duration: int = 300_000_000,
        load_duration: int = 50_000_000,
    ):
        self.message = FakeMessage(content)
        self.model = model
        self.prompt_eval_count = prompt_eval_count
        self.prompt_eval_duration = prompt_eval_duration
        self.eval_count = eval_count
        self.eval_duration = eval_duration
        self.total_duration = total_duration
        self.load_duration = load_duration


class FakeClient:
    def __init__(self, chat_response: Any = None):
        self.chat_response = chat_response or FakeChatResponse()
        self.last_kwargs: dict[str, Any] = {}
        self.call_count: int = 0

    def chat(self, **kwargs: Any) -> Any:
        self.call_count += 1
        self.last_kwargs = kwargs
        return self.chat_response


class FakeLinearModel(nn.Module):
    """Simple linear module with differentiable parameters for gradient attribution."""

    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FakeEnsemble:
    """In-memory ensemble stub matching BinaryDNNEnsemble API."""

    def __init__(self, feature_names: list[str]):
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
    """Identity scaler stub."""

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x.copy()


class FakeKB:
    """Knowledge base index stub returning predetermined chunks."""

    def __init__(self, chunks: list[dict[str, Any]]):
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
        """Return chunk metadata dictionary for the given index position."""
        return self.chunks[position]


class FakeCrossEncoder:
    """Cross-encoder reranker stub."""

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.ones(len(pairs), dtype=np.float32) * 2.0


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
    feature_names = ["Tot Bwd Pkts", "Tot Fwd Pkts", "Protocol"]
    ensemble = FakeEnsemble(feature_names)
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
        feature_names=feature_names,
        kb=kb,
        cross_encoder=cross_encoder,
        df_test=df_test,
        df_meta=df_meta,
        df_preds=df_preds,
        chunks_sha256="test_sha256_digest",
    )


# ---------------------------------------------------------------------------
# Test 1: writes flow_<index>_rag.txt and JSON with all required keys
# ---------------------------------------------------------------------------


def test_run_pipeline_writes_report_and_complete_json(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)

    expected_text = (
        "1. Rationale\nObserved DoS flooding.\n"
        "2. Key Indicators\nHigh forward packet count.\n"
        "3. Confidence Assessment\nVery high confidence.\n"
        "4. Threat Assessment\nDisruption risk.\n"
        "5. Recommendations\nBlock traffic."
    )
    fake_client = FakeClient(chat_response=FakeChatResponse(content=expected_text))

    summary = run_pipeline(
        index=0,
        ctx=fake_context,
        cfg=fake_cfg,
        modes="rag",
        client=fake_client,
    )

    flows_dir = tmp_path / "artifacts" / "reports" / "flows"
    report_file = flows_dir / "flow_0_rag.txt"
    json_file = flows_dir / "flow_0.json"

    # 1. Report file contains exactly the fake model text
    assert report_file.exists()
    assert report_file.read_text(encoding="utf-8") == expected_text

    # 2. JSON file exists and contains all keys from item 8
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))

    required_keys = [
        "index",
        "ground_truth",
        "predicted_class",
        "confidence",
        "tier",
        "explanation",
        "query",
        "chunks",
        "prompts",
        "reports",
        "validation",
        "model",
        "options",
        "timings",
        "config",
        "chunks_sha256",
    ]
    for key in required_keys:
        assert key in data, f"Missing required key '{key}' in JSON artifact"

    # Verify chunk structure contains required fields
    assert len(data["chunks"]) > 0
    c0 = data["chunks"][0]
    chunk_keys = ["chunk_id", "source", "section", "bm25_rank", "dense_rank", "ce_score", "is_fallback"]
    for ck in chunk_keys:
        assert ck in c0, f"Missing chunk key '{ck}' in retrieved chunk record"


# ---------------------------------------------------------------------------
# Test 2: changing ground-truth label does not affect prompts, query, or explanation
# ---------------------------------------------------------------------------


def test_ground_truth_label_does_not_affect_pipeline(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)
    fake_client = FakeClient()

    # Run A with original ground truth ("DoS")
    fake_context.df_preds.loc[0, "true_class"] = "DoS"
    res_a = run_pipeline(0, fake_context, fake_cfg, modes="rag", client=fake_client)

    # Run B with altered ground truth ("Benign")
    fake_context.df_preds.loc[0, "true_class"] = "Benign"
    res_b = run_pipeline(0, fake_context, fake_cfg, modes="rag", client=fake_client)

    # Ground truth changed
    assert res_a["ground_truth"] == "DoS"
    assert res_b["ground_truth"] == "Benign"
    assert res_a["ground_truth"] != res_b["ground_truth"]

    # Explanations, queries, and prompts remain identical
    assert res_a["explanation"] == res_b["explanation"]
    assert res_a["query"] == res_b["query"]
    assert res_a["prompts"] == res_b["prompts"]


# ---------------------------------------------------------------------------
# Test 3: mode 'both' writes both reports; --no-generate writes no report files
# ---------------------------------------------------------------------------


def test_mode_both_and_no_generate(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)
    fake_client = FakeClient()
    flows_dir = tmp_path / "artifacts" / "reports" / "flows"

    # Part A: mode 'both' writes both report files
    run_pipeline(index=1, ctx=fake_context, cfg=fake_cfg, modes="both", client=fake_client)
    assert (flows_dir / "flow_1_rag.txt").exists()
    assert (flows_dir / "flow_1_vanilla.txt").exists()

    # Part B: --no-generate (generate=False) writes no report files and does not call client
    call_counting_client = FakeClient()
    run_pipeline(
        index=2,
        ctx=fake_context,
        cfg=fake_cfg,
        modes="rag",
        generate=False,
        client=call_counting_client,
    )
    assert not (flows_dir / "flow_2_rag.txt").exists()
    assert not (flows_dir / "flow_2_vanilla.txt").exists()
    assert call_counting_client.call_count == 0


# ---------------------------------------------------------------------------
# Test 4: out-of-range index raises clear error naming valid range
# ---------------------------------------------------------------------------


def test_out_of_range_index_raises_error(
    fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    # fake_context has 3 rows (indices 0, 1, 2)
    with pytest.raises(IndexError, match=r"Valid range is 0 to 2"):
        run_pipeline(index=99, ctx=fake_context, cfg=fake_cfg)

    with pytest.raises(IndexError, match=r"Valid range is 0 to 2"):
        run_pipeline(index=-1, ctx=fake_context, cfg=fake_cfg)


# ---------------------------------------------------------------------------
# Test 5: several indices continue after a failure and exit non-zero
# ---------------------------------------------------------------------------


def test_multiple_indices_failure_handling(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("ids_rag.pipeline.load_config", lambda: fake_cfg)
    monkeypatch.setattr("ids_rag.pipeline.check_ollama", lambda cfg: ["llama3:8b"])
    monkeypatch.setattr("ids_rag.pipeline.PipelineContext.load", lambda cfg: fake_context)

    # Monkeypatch generate_report to avoid real Ollama
    fake_client = FakeClient()
    monkeypatch.setattr(
        "ids_rag.pipeline.generate_report",
        lambda prompt, cfg, client=None, seed_override=None: {
            "text": "1. Rationale\nText.\n2. Key Indicators\nText.\n3. Confidence Assessment\nText.\n4. Threat Assessment\nText.\n5. Recommendations\nText.",
            "model": "llama3:8b",
            "options": {"seed": seed_override or 42},
            "seed_used": seed_override or 42,
            "prompt_words": 10,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 0.1,
            "eval_count": 20,
            "eval_duration": 0.2,
            "total_duration": 0.3,
            "load_duration": 0.05,
            "tokens_per_second": 100.0,
        },
    )

    # Pass indices 0, 99 (invalid), 1
    monkeypatch.setattr(sys, "argv", ["pipeline", "--indices", "0,99,1"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    # Exit code must be non-zero
    assert exc_info.value.code != 0

    # Index 0 and 1 were generated despite 99 failing
    flows_dir = tmp_path / "artifacts" / "reports" / "flows"
    assert (flows_dir / "flow_0_rag.txt").exists()
    assert (flows_dir / "flow_1_rag.txt").exists()


# ---------------------------------------------------------------------------
# Test 6: --seed overrides llm.seed and appears in JSON
# ---------------------------------------------------------------------------


def test_seed_override_appears_in_json(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)
    fake_client = FakeClient()

    summary = run_pipeline(
        index=0,
        ctx=fake_context,
        cfg=fake_cfg,
        modes="rag",
        seed=1337,
        client=fake_client,
    )

    # In summary dict
    assert summary["seed_used"] == 1337
    assert summary["options"]["seed"] == 1337

    # In options passed to client
    assert fake_client.last_kwargs["options"]["seed"] == 1337

    # In saved JSON artifact
    flows_dir = tmp_path / "artifacts" / "reports" / "flows"
    json_data = json.loads((flows_dir / "flow_0.json").read_text(encoding="utf-8"))
    assert json_data["seed_used"] == 1337
    assert json_data["options"]["seed"] == 1337


# ---------------------------------------------------------------------------
# Test 7: pipeline --indices-file writes into custom out_dir and not default
# ---------------------------------------------------------------------------


def test_pipeline_indices_file_custom_out_dir(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("ids_rag.pipeline.load_config", lambda: fake_cfg)
    monkeypatch.setattr("ids_rag.pipeline.check_ollama", lambda cfg: ["llama3:8b"])
    monkeypatch.setattr("ids_rag.pipeline.PipelineContext.load", lambda cfg: fake_context)

    monkeypatch.setattr(
        "ids_rag.pipeline.generate_report",
        lambda prompt, cfg, client=None, seed_override=None: {
            "text": "1. Rationale\nText.\n2. Key Indicators\nText.\n3. Confidence Assessment\nText.\n4. Threat Assessment\nText.\n5. Recommendations\nText.",
            "model": "llama3:8b",
            "options": {"seed": seed_override or 42},
            "seed_used": seed_override or 42,
            "prompt_words": 10,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 0.1,
            "eval_count": 20,
            "eval_duration": 0.2,
            "total_duration": 0.3,
            "load_duration": 0.05,
            "tokens_per_second": 100.0,
        },
    )

    indices_file = tmp_path / "test_indices.json"
    indices_file.write_text(json.dumps({"indices": [0, 1]}), encoding="utf-8")

    custom_out_dir = tmp_path / "custom_eval_reports"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            "--indices-file",
            str(indices_file),
            "--out-dir",
            str(custom_out_dir),
            "--mode",
            "rag",
        ],
    )

    main()

    # Files must exist in custom_out_dir
    assert (custom_out_dir / "flow_0_rag.txt").exists()
    assert (custom_out_dir / "flow_0.json").exists()
    assert (custom_out_dir / "flow_1_rag.txt").exists()
    assert (custom_out_dir / "flow_1.json").exists()

    # Default output dir must NOT have been used
    default_dir = tmp_path / "artifacts" / "reports" / "flows"
    assert not default_dir.exists() or not (default_dir / "flow_0_rag.txt").exists()


# ---------------------------------------------------------------------------
# Test 8: --skip-existing leaves existing JSON intact/byte-identical, avoids client call
# ---------------------------------------------------------------------------


def test_pipeline_skip_existing(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("ids_rag.pipeline.load_config", lambda: fake_cfg)
    monkeypatch.setattr("ids_rag.pipeline.check_ollama", lambda cfg: ["llama3:8b"])
    monkeypatch.setattr("ids_rag.pipeline.PipelineContext.load", lambda cfg: fake_context)

    custom_out_dir = tmp_path / "eval_out"
    custom_out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create flow_0.json with distinct content
    pre_existing_data = {
        "index": 0,
        "is_correct": True,
        "predicted_class": "DoS",
        "true_class": "DoS",
        "custom_marker": "DO_NOT_OVERWRITE",
        "validation": {
            "rag": {
                "word_count": 350,
                "words_in_range": True,
                "headings_found": {"Rationale": True},
                "headings_any_count": 5,
                "cited": [1, 2],
                "citation_ranges": [],
                "copied_source_headers": 0,
                "attack_ids_not_in_kb": [],
                "kb_attack_ids_named": ["T1498"],
                "sp_not_in_kb": [],
            }
        },
    }
    flow_0_json = custom_out_dir / "flow_0.json"
    pre_bytes = json.dumps(pre_existing_data, indent=2).encode("utf-8")
    flow_0_json.write_bytes(pre_bytes)

    calls: list[Any] = []

    def fake_generate(prompt: Any, cfg: Any, client: Any = None, seed_override: Any = None) -> dict[str, Any]:
        calls.append(prompt)
        return {
            "text": "1. Rationale\nText.\n2. Key Indicators\nText.\n3. Confidence Assessment\nText.\n4. Threat Assessment\nText.\n5. Recommendations\nText.",
            "model": "llama3:8b",
            "options": {"seed": seed_override or 42},
            "seed_used": seed_override or 42,
            "prompt_words": 10,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 0.1,
            "eval_count": 20,
            "eval_duration": 0.2,
            "total_duration": 0.3,
            "load_duration": 0.05,
            "tokens_per_second": 100.0,
        }

    monkeypatch.setattr("ids_rag.pipeline.generate_report", fake_generate)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            "--indices",
            "0,1",
            "--out-dir",
            str(custom_out_dir),
            "--skip-existing",
            "--mode",
            "rag",
        ],
    )

    main()

    # flow_0.json must be byte-identical to pre_bytes
    assert flow_0_json.read_bytes() == pre_bytes

    # flow 1 was executed and written
    assert (custom_out_dir / "flow_1.json").exists()
    assert (custom_out_dir / "flow_1_rag.txt").exists()

    # fake_generate should only have been called for flow 1 (1 call, not 2)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Test 9: --summary-csv contains all 20 columns, 1 row per flow/mode, values match JSON
# ---------------------------------------------------------------------------


def test_pipeline_summary_csv(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("ids_rag.pipeline.load_config", lambda: fake_cfg)
    monkeypatch.setattr("ids_rag.pipeline.check_ollama", lambda cfg: ["llama3:8b"])
    monkeypatch.setattr("ids_rag.pipeline.PipelineContext.load", lambda cfg: fake_context)

    monkeypatch.setattr(
        "ids_rag.pipeline.generate_report",
        lambda prompt, cfg, client=None, seed_override=None: {
            "text": "1. Rationale\nText.\n2. Key Indicators\nText.\n3. Confidence Assessment\nText.\n4. Threat Assessment\nText.\n5. Recommendations\nText.",
            "model": "llama3:8b",
            "options": {"seed": seed_override or 42},
            "seed_used": seed_override or 42,
            "prompt_words": 10,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 0.1,
            "eval_count": 20,
            "eval_duration": 0.2,
            "total_duration": 0.3,
            "load_duration": 0.05,
            "tokens_per_second": 100.0,
        },
    )

    csv_path = tmp_path / "summary.csv"
    out_dir = tmp_path / "flows_csv_test"

    # Evaluate indices 0 and 1 with mode "both" -> 2 flows * 2 modes = 4 rows
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            "--indices",
            "0,1",
            "--mode",
            "both",
            "--out-dir",
            str(out_dir),
            "--summary-csv",
            str(csv_path),
        ],
    )

    main()

    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    expected_cols = [
        "index",
        "true_class",
        "predicted_class",
        "is_correct",
        "confidence",
        "tier",
        "mode",
        "words",
        "words_in_range",
        "headings_numbered_count",
        "headings_any_count",
        "cited",
        "citation_ranges_count",
        "copied_source_headers",
        "attack_ids_not_in_kb",
        "kb_attack_ids_named",
        "sp_not_in_kb",
        "generate_seconds",
        "seed_used",
        "report_file",
    ]

    assert len(expected_cols) == 20
    assert list(reader[0].keys()) == expected_cols
    assert len(reader) == 4

    # Verify rows correspond to flow 0 rag, flow 0 vanilla, flow 1 rag, flow 1 vanilla
    row0 = reader[0]
    assert int(row0["index"]) == 0
    assert row0["mode"] == "rag"
    assert row0["predicted_class"] == "DoS"
    assert row0["tier"] == "VERY_HIGH"
    assert row0["report_file"] == str(out_dir / "flow_0_rag.txt")


# ---------------------------------------------------------------------------
# Test 10: ollama_version stored in flow JSON (subprocess patched)
# ---------------------------------------------------------------------------


def test_pipeline_ollama_version_recorded(
    tmp_path: Path, monkeypatch: Any, fake_context: PipelineContext, fake_cfg: dict[str, Any]
) -> None:
    monkeypatch.setattr("ids_rag.pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("ids_rag.pipeline.load_config", lambda: fake_cfg)
    monkeypatch.setattr("ids_rag.pipeline.check_ollama", lambda cfg: ["llama3:8b"])
    monkeypatch.setattr("ids_rag.pipeline.PipelineContext.load", lambda cfg: fake_context)

    # Patch get_ollama_version to return a known string
    mock_version = "ollama version 0.1.32"
    monkeypatch.setattr("ids_rag.pipeline.get_ollama_version", lambda: mock_version)

    monkeypatch.setattr(
        "ids_rag.pipeline.generate_report",
        lambda prompt, cfg, client=None, seed_override=None: {
            "text": "1. Rationale\nText.\n2. Key Indicators\nText.\n3. Confidence Assessment\nText.\n4. Threat Assessment\nText.\n5. Recommendations\nText.",
            "model": "llama3:8b",
            "options": {"seed": seed_override or 42},
            "seed_used": seed_override or 42,
            "prompt_words": 10,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 0.1,
            "eval_count": 20,
            "eval_duration": 0.2,
            "total_duration": 0.3,
            "load_duration": 0.05,
            "tokens_per_second": 100.0,
        },
    )

    out_dir = tmp_path / "reports_ver"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            "--index",
            "0",
            "--out-dir",
            str(out_dir),
            "--mode",
            "rag",
        ],
    )

    main()

    flow_json = out_dir / "flow_0.json"
    assert flow_json.exists()
    data = json.loads(flow_json.read_text(encoding="utf-8"))
    assert data["ollama_version"] == mock_version


