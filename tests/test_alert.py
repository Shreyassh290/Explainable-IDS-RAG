"""Unit tests for analyst alert builder (tests/test_alert.py).

Tests run entirely in-memory with small hand-made dictionaries.
No real files, no Ollama, no model loading.

Covers:
1. Output key order and excluded keys are absent.
2. Exact original_value is preserved from the flow dict.
3. High-overlap citation is not flagged; low-overlap citation is flagged.
4. Citation number out of range gives out_of_range True.
5. The three fixed warning strings are always present.
6. "Report contains no inline citations." warning appears when there are no [n] tags.
7. features_named_in_report vs features_not_named_in_report.
8. ValueError raised when flow["reports"]["rag"] is missing.
9. ValueError raised when flow["validation"]["rag"] is missing.
10. write_alert refuses to overwrite when force=False and exits with code 1.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from ids_rag.alert import build_alert, write_alert

# Also import extract_flow so we can verify citation overlap logic via the same code path.
from ids_rag.eval.citation_check import extract_flow


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

CHUNKS_BY_ID: dict[str, dict[str, Any]] = {
    # High overlap with "Intrusion detection systems monitor network traffic anomalies."
    "chunk_high": {
        "chunk_id": "chunk_high",
        "text": "Intrusion detection systems monitor network traffic anomalies for threats.",
    },
    # Low overlap with "Apply mandatory firewall filter immediately."
    "chunk_low": {
        "chunk_id": "chunk_low",
        "text": "Completely different zebra savanna content without shared words.",
    },
}

RETRIEVED_CHUNKS: list[dict[str, Any]] = [
    {
        "chunk_id": "chunk_high",
        "source": "NIST_SP_800-61",
        "section": "3.1",
        "ce_score": 0.95,
    },
    {
        "chunk_id": "chunk_low",
        "source": "MITRE_T1498",
        "section": "Description",
        "ce_score": 0.40,
    },
]

# Report sentence [1]: high overlap with chunk_high -> should NOT be flagged at thr=0.5
# Report sentence [2]: no shared tokens with chunk_low -> flagged
RAG_REPORT_TEXT = (
    "1. Rationale\n"
    "Intrusion detection systems monitor network traffic anomalies [1].\n"
    "2. Key Indicators\n"
    "Apply mandatory firewall filter immediately [2].\n"
    "3. Confidence Assessment\n"
    "High confidence based on evidence.\n"
    "4. Threat Assessment\n"
    "Traffic anomaly indicates network intrusion.\n"
    "5. Recommendations\n"
    "Deploy monitoring systems network-wide.\n"
)

VALIDATION_FLAGS: dict[str, Any] = {
    "word_count": 45,
    "words_in_range": True,
    "headings_found": {"Rationale": True, "Key Indicators": True},
    "headings_any_count": 5,
    "cited": [1, 2],
    "invalid_citations": [],
    "has_reference_list": False,
    "unsupported_ips": [],
    "attack_ids_not_in_kb": [],
    "sp_not_in_kb": [],
    "placeholder_citations": 0,
}

# Top features with exact original_value values (not rounded differently).
TOP_FEATURES_RAW: list[dict[str, Any]] = [
    {
        "name": "Tot Fwd Pkts",
        "scaled_value": 1.2345,
        "original_value": 987.654321,  # exact float that must be preserved
        "importance": 0.123456,
        "direction": "raises",
        "description": "Total forward packets in the flow.",
        "implication": "High packet count implies high throughput.",
    },
    {
        "name": "Protocol",
        "scaled_value": 0.5,
        "original_value": 6.0,
        "importance": 0.054321,
        "direction": "lowers",
        "description": "Transport protocol identifier.",
        "implication": "TCP protocol detected.",
    },
]

FLOW: dict[str, Any] = {
    "index": 7,
    "predicted_class": "DoS",
    "confidence": 0.9876,
    "tier": "VERY_HIGH",
    "explanation": {
        "predicted_class": "DoS",
        "confidence": 0.9876,
        "tier": "VERY_HIGH",
        "features": TOP_FEATURES_RAW,
        "metadata": {
            "Timestamp": "2018-02-14 10:00:00",
            "Dst Port": 80,
            "Flow ID": "1.1.1.1-2.2.2.2-80-1234-6",
            "Src IP": "1.1.1.1",
            "Src Port": 1234,
            "Dst IP": "2.2.2.2",
            "source_file": "Wednesday.pcap",
            "protocol": 6,
            "dataset": "CICIDS2018",
        },
    },
    "reports": {
        "rag": RAG_REPORT_TEXT,
        "vanilla": "Vanilla report text ignored.",
    },
    "validation": {
        "rag": VALIDATION_FLAGS,
    },
    "retrieved_chunks": RETRIEVED_CHUNKS,
    "model": "llama3:8b",
    "seed_used": 42,
    "ollama_version": "ollama version 0.1.32",
    "chunks_sha256": "abcdef1234567890",
    # Fields that must NOT appear in the alert output:
    "ground_truth": "DoS",
    "true_class": "DoS",
    "is_correct": True,
    "prompts": {"rag": "SYSTEM\n..."},
    "config": {"llm": {}, "rag": {}},
    "llm_config": {},
    "rag_config": {},
    "options": {"seed": 42, "temperature": 0.2},
    "timings": {"explain": 0.1, "retrieve": 0.2},
}

EXCLUDED_KEYS = {
    "ground_truth",
    "true_class",
    "is_correct",
    "prompts",
    "config",
    "llm_config",
    "rag_config",
    "options",
    "timings",
}

EXPECTED_KEY_ORDER = [
    "alert_version",
    "flow_index",
    "detected",
    "flow_metadata",
    "top_features",
    "mitigation_report",
    "sources",
    "citations",
    "checks",
    "provenance",
    "warnings",
]

THR = 0.5


# ---------------------------------------------------------------------------
# Test 1: Output key order and excluded keys are absent
# ---------------------------------------------------------------------------


def test_output_key_order_and_excluded_keys_absent() -> None:
    """Alert dict has exactly the required keys in order; excluded keys are not present."""
    alert = build_alert(FLOW, CHUNKS_BY_ID, THR)
    keys = list(alert.keys())
    assert keys == EXPECTED_KEY_ORDER, f"Key order mismatch: {keys}"
    for bad_key in EXCLUDED_KEYS:
        assert bad_key not in alert, f"Excluded key present: {bad_key}"


# ---------------------------------------------------------------------------
# Test 2: Exact original_value preserved
# ---------------------------------------------------------------------------


def test_original_value_preserved() -> None:
    """original_value in top_features must equal the source float exactly, not rounded."""
    alert = build_alert(FLOW, CHUNKS_BY_ID, THR)
    assert alert["top_features"][0]["original_value"] == 987.654321
    assert alert["top_features"][1]["original_value"] == 6.0
    # rank must be 1-based
    assert alert["top_features"][0]["rank"] == 1
    assert alert["top_features"][1]["rank"] == 2


# ---------------------------------------------------------------------------
# Test 3: Citation overlap - high overlap not flagged, low overlap flagged
# ---------------------------------------------------------------------------


def test_citation_overlap_flagging() -> None:
    """High-overlap citation is not flagged; low-overlap citation is flagged at thr=0.5."""
    alert = build_alert(FLOW, CHUNKS_BY_ID, THR)
    cites = alert["citations"]
    # At least two citations (n=1 and n=2)
    by_n = {c["n"]: c for c in cites}
    assert 1 in by_n, "Citation n=1 expected"
    assert 2 in by_n, "Citation n=2 expected"

    # Citation 1: sentence shares "intrusion", "detection", "systems", "monitor", "network",
    # "traffic", "anomalies" with chunk_high -> high overlap -> NOT flagged
    assert by_n[1]["flagged_unverified"] is False
    assert by_n[1]["overlap"] is not None
    assert by_n[1]["overlap"] > 0.0

    # Citation 2: "apply", "mandatory", "firewall", "filter", "immediately" vs
    # "completely", "different", "zebra", "savanna", "content", "without", "shared", "words"
    # -> zero shared tokens -> flagged
    assert by_n[2]["flagged_unverified"] is True


# ---------------------------------------------------------------------------
# Test 4: Out-of-range citation gives out_of_range True
# ---------------------------------------------------------------------------


def test_citation_out_of_range() -> None:
    """Citation with number exceeding length of retrieved_chunks has out_of_range=True."""
    # Sentence references [3] but only 2 chunks retrieved
    oor_report = "Intrusion detection response procedure [3].\n"
    flow_oor = {**FLOW, "reports": {"rag": oor_report}}
    alert = build_alert(flow_oor, CHUNKS_BY_ID, THR)
    by_n = {c["n"]: c for c in alert["citations"]}
    assert 3 in by_n, "Citation n=3 expected"
    assert by_n[3]["out_of_range"] is True
    assert by_n[3]["overlap"] is None


# ---------------------------------------------------------------------------
# Test 5: Three fixed warning strings are always present
# ---------------------------------------------------------------------------

FIXED_WARNINGS = [
    "Draft generated by a local language model. Review before acting.",
    "Generated from a stored evaluation flow, not live traffic.",
    "Citation flags are a word-overlap heuristic, not a verdict.",
]


def test_fixed_warnings_always_present() -> None:
    """The three mandatory warning strings appear in every alert."""
    alert = build_alert(FLOW, CHUNKS_BY_ID, THR)
    for w in FIXED_WARNINGS:
        assert w in alert["warnings"], f"Fixed warning missing: {w}"


# ---------------------------------------------------------------------------
# Test 6: "Report contains no inline citations." warning when no [n] tags
# ---------------------------------------------------------------------------


def test_no_citations_warning() -> None:
    """'Report contains no inline citations.' appears when report has no [n] tags."""
    no_cite_report = (
        "1. Rationale\nSuspicious traffic observed.\n"
        "2. Key Indicators\nHigh packet rate detected.\n"
        "3. Confidence Assessment\nHigh confidence.\n"
        "4. Threat Assessment\nDoS attack likely.\n"
        "5. Recommendations\nBlock source IP.\n"
    )
    flow_no_cite = {**FLOW, "reports": {"rag": no_cite_report}}
    alert = build_alert(flow_no_cite, CHUNKS_BY_ID, THR)
    assert "Report contains no inline citations." in alert["warnings"]
    # The flagged-count warning must NOT appear
    for w in alert["warnings"]:
        assert "citation rows flagged" not in w


# ---------------------------------------------------------------------------
# Test 7: features_named_in_report vs features_not_named_in_report
# ---------------------------------------------------------------------------


def test_features_named_in_report() -> None:
    """Features whose names appear case-insensitively in report text are separated correctly."""
    # RAG_REPORT_TEXT does not contain 'Tot Fwd Pkts' or 'Protocol' literally,
    # so both features should be in features_not_named_in_report.
    alert = build_alert(FLOW, CHUNKS_BY_ID, THR)
    checks = alert["checks"]
    named = checks["features_named_in_report"]
    not_named = checks["features_not_named_in_report"]
    all_names = [f["name"] for f in TOP_FEATURES_RAW]
    assert set(named) | set(not_named) == set(all_names)
    assert set(named) & set(not_named) == set()

    # Build a flow with a report that explicitly mentions one feature name
    report_with_feature = RAG_REPORT_TEXT + "\nTot Fwd Pkts was extremely high.\n"
    flow_named = {**FLOW, "reports": {"rag": report_with_feature}}
    alert2 = build_alert(flow_named, CHUNKS_BY_ID, THR)
    assert "Tot Fwd Pkts" in alert2["checks"]["features_named_in_report"]
    assert "Tot Fwd Pkts" not in alert2["checks"]["features_not_named_in_report"]


# ---------------------------------------------------------------------------
# Test 8 & 9: ValueError when required fields are missing
# ---------------------------------------------------------------------------


def test_missing_reports_rag_raises() -> None:
    """ValueError raised when flow['reports']['rag'] is absent."""
    bad_flow = {**FLOW, "reports": {"vanilla": "vanilla only"}}
    with pytest.raises(ValueError, match="reports.*rag"):
        build_alert(bad_flow, CHUNKS_BY_ID, THR)


def test_missing_reports_key_raises() -> None:
    """ValueError raised when flow['reports'] is entirely absent."""
    bad_flow = {k: v for k, v in FLOW.items() if k != "reports"}
    with pytest.raises(ValueError, match="reports.*rag"):
        build_alert(bad_flow, CHUNKS_BY_ID, THR)


def test_missing_validation_rag_raises() -> None:
    """ValueError raised when flow['validation']['rag'] is absent."""
    bad_flow = {**FLOW, "validation": {"vanilla": {}}}
    with pytest.raises(ValueError, match="validation.*rag"):
        build_alert(bad_flow, CHUNKS_BY_ID, THR)


# ---------------------------------------------------------------------------
# Test 10: write_alert refuses to overwrite when force=False
# ---------------------------------------------------------------------------


def test_write_alert_refuses_overwrite(tmp_path: Path) -> None:
    """write_alert exits with code 1 and leaves the file unchanged when force=False."""
    out_file = tmp_path / "alert_flow_7.json"
    out_file.write_text("existing content", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        write_alert({"alert_version": 1}, out_file, force=False)

    assert exc_info.value.code == 1
    assert out_file.read_text(encoding="utf-8") == "existing content"


def test_write_alert_overwrites_with_force(tmp_path: Path) -> None:
    """write_alert overwrites the file when force=True."""
    out_file = tmp_path / "alert_flow_7.json"
    out_file.write_text("old content", encoding="utf-8")

    alert = build_alert(FLOW, CHUNKS_BY_ID, THR)
    write_alert(alert, out_file, force=True)

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["alert_version"] == 1
    assert data["flow_index"] == 7


def test_write_alert_creates_parent_dirs(tmp_path: Path) -> None:
    """write_alert creates parent directories if they do not exist."""
    out_file = tmp_path / "nested" / "dir" / "alert_flow_7.json"
    alert = build_alert(FLOW, CHUNKS_BY_ID, THR)
    write_alert(alert, out_file, force=False)
    assert out_file.exists()


# ---------------------------------------------------------------------------
# Test: provenance block contains expected fields
# ---------------------------------------------------------------------------


def test_provenance_fields() -> None:
    """Provenance block contains model, seed_used, ollama_version, chunks_sha256, overlap_threshold."""
    alert = build_alert(FLOW, CHUNKS_BY_ID, THR)
    prov = alert["provenance"]
    assert prov["model"] == "llama3:8b"
    assert prov["seed_used"] == 42
    assert prov["ollama_version"] == "ollama version 0.1.32"
    assert prov["chunks_sha256"] == "abcdef1234567890"
    assert prov["overlap_threshold"] == THR


# ---------------------------------------------------------------------------
# Test: sources block is correctly numbered and uses retrieved_chunks fields
# ---------------------------------------------------------------------------


def test_sources_block() -> None:
    """Sources are 1-based numbered and preserve chunk_id, source, section, ce_score."""
    alert = build_alert(FLOW, CHUNKS_BY_ID, THR)
    sources = alert["sources"]
    assert sources[0]["number"] == 1
    assert sources[0]["chunk_id"] == "chunk_high"
    assert sources[0]["source"] == "NIST_SP_800-61"
    assert sources[0]["section"] == "3.1"
    assert sources[0]["ce_score"] == 0.95
    assert sources[1]["number"] == 2
    assert sources[1]["chunk_id"] == "chunk_low"


# ---------------------------------------------------------------------------
# Test: mitigation_report block
# ---------------------------------------------------------------------------


def test_mitigation_report() -> None:
    """mitigation_report has mode 'rag', correct word_count, and exact text."""
    alert = build_alert(FLOW, CHUNKS_BY_ID, THR)
    mr = alert["mitigation_report"]
    assert mr["mode"] == "rag"
    assert mr["word_count"] == len(RAG_REPORT_TEXT.split())
    assert mr["text"] == RAG_REPORT_TEXT


# ---------------------------------------------------------------------------
# Test: vanilla report text is NOT present anywhere in the alert
# ---------------------------------------------------------------------------


def test_vanilla_report_absent() -> None:
    """Vanilla report text must not appear anywhere in the serialised alert."""
    alert = build_alert(FLOW, CHUNKS_BY_ID, THR)
    serialised = json.dumps(alert)
    assert "Vanilla report text ignored." not in serialised

