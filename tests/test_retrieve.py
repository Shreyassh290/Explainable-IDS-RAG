"""Unit tests for ids_rag.rag.retrieve using mock KBIndex and fake cross-encoder."""

from typing import Any

import numpy as np
import pytest

from ids_rag.rag.retrieve import (
    SYNONYMS,
    build_query,
    compute_fused_score,
    expand_query,
    min_max_normalize_bm25,
    retrieve,
)

# ---------------------------------------------------------------------------
# Test 1: Query expansion adds synonyms for DoS and DDoS
# ---------------------------------------------------------------------------

def test_query_expansion_adds_synonyms() -> None:
    """Query expansion must add domain synonyms for DoS and DDoS (and Benign)."""
    dos_res = expand_query("Anomalous DoS activity detected on host.")
    for s in SYNONYMS["DoS"]:
        assert s in dos_res, f"Expected synonym {s!r} in expanded query: {dos_res!r}"

    ddos_res = expand_query("Volumetric DDoS flood targeting web server.")
    for s in SYNONYMS["DDoS"]:
        assert s in ddos_res, f"Expected synonym {s!r} in expanded query: {ddos_res!r}"

    benign_res = expand_query("Routine Benign baseline flow.")
    for s in SYNONYMS["Benign"]:
        assert s in benign_res, f"Expected synonym {s!r} in expanded query: {benign_res!r}"


# ---------------------------------------------------------------------------
# Test 2: BM25 normalization maps min to 0 and max to 1, handles constant list
# ---------------------------------------------------------------------------

def test_bm25_normalization() -> None:
    """Min-max normalization maps minimum to 0, maximum to 1, and handles constant lists."""
    scores = [10.0, 20.0, 50.0]
    norm = min_max_normalize_bm25(scores)
    assert np.isclose(norm[0], 0.0), f"Expected 0.0 for min, got {norm[0]}"
    assert np.isclose(norm[1], 0.25), f"Expected 0.25 for middle, got {norm[1]}"
    assert np.isclose(norm[2], 1.0), f"Expected 1.0 for max, got {norm[2]}"

    # Constant list guard against zero-division
    const_scores = [5.0, 5.0, 5.0]
    const_norm = min_max_normalize_bm25(const_scores)
    assert len(const_norm) == 3
    for v in const_norm:
        assert np.isclose(v, 1.0)


# ---------------------------------------------------------------------------
# Test 3: Fused score on hand-made example
# ---------------------------------------------------------------------------

def test_fused_score_formula_handmade() -> None:
    """Fused score must equal 0.6 * dense + 0.4 * bm25_norm."""
    dense_score = 0.80
    bm25_norm = 0.50
    expected = 0.60 * 0.80 + 0.40 * 0.50  # 0.48 + 0.20 = 0.68
    actual = compute_fused_score(dense_score, bm25_norm, w_dense=0.60, w_bm25=0.40)
    assert np.isclose(actual, expected, atol=1e-6), f"Expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# Fake KBIndex and Fake CrossEncoder for integration tests
# ---------------------------------------------------------------------------

class FakeKB:
    """Mock KBIndex supporting BM25 and dense retrieval over N synthetic chunks."""

    def __init__(self, n_chunks: int = 40) -> None:
        self.chunks: list[dict[str, Any]] = []
        for i in range(n_chunks):
            if i >= 30:
                sec = "Executive Summary" if i % 2 == 0 else "3.6 Recommendations"
            else:
                sec = "3.2.1 Attack Vectors"
            self.chunks.append(
                {
                    "chunk_id": f"chunk_{i:03d}",
                    "text": f"Synthetic chunk text number {i} discussing security mitigation and attack detection.",
                    "source_name": "NIST_SP_800-61_Rev2.pdf",
                    "source_type": "NIST",
                    "source_ref": "SP 800-61",
                    "section": sec,
                    "char_start": i * 100,
                    "char_end": (i + 1) * 100,
                }
            )

    def search_bm25(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        # Return chunks 0 to k-1 with descending scores
        return [(i, float(100.0 - i)) for i in range(min(k, len(self.chunks)))]

    def search_dense(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        # Return chunks in offset order (e.g. chunks 10 to 10+k-1) to create a union > k
        res: list[tuple[int, float]] = []
        for rank, i in enumerate(range(10, min(10 + k, len(self.chunks)))):
            res.append((i, float(0.95 - 0.01 * rank)))
        return res

    def bm25_scores(self, query: str) -> np.ndarray:
        return np.array([float(100.0 - i) for i in range(len(self.chunks))], dtype=np.float32)

    def dense_scores(self, query: str) -> np.ndarray:
        return np.array([float(0.95 - 0.005 * i) for i in range(len(self.chunks))], dtype=np.float32)

    def get_chunk(self, position: int) -> dict[str, Any]:
        return self.chunks[position]


class FakeCrossEncoder:
    """Mock CrossEncoder returning predetermined or linear logit scores."""

    def __init__(self, score_pattern: str = "high") -> None:
        self.score_pattern = score_pattern

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        # Return raw logits that map through sigmoid
        if self.score_pattern == "high":
            # Logits >= 1.0 map to sigmoid >= 0.73 (> 0.5)
            return [2.0 - 0.1 * i for i in range(len(pairs))]
        elif self.score_pattern == "low":
            # Only 1 chunk has logit >= 0 (sigmoid >= 0.5), rest are negative (sigmoid < 0.5)
            return [1.0] + [-2.0 - 0.1 * i for i in range(len(pairs) - 1)]
        elif self.score_pattern == "descending":
            return [5.0 - 0.2 * i for i in range(len(pairs))]
        return [0.0] * len(pairs)


# ---------------------------------------------------------------------------
# Test 4: Fusion keeps top 30 of a larger union
# ---------------------------------------------------------------------------

def test_fusion_keeps_top_30_of_larger_union() -> None:
    """When BM25 (top 30) and dense (top 30) union produces > 30 items, fusion keeps top 30."""
    fake_kb = FakeKB(n_chunks=45)
    fake_ce = FakeCrossEncoder(score_pattern="high")

    # In FakeKB: BM25 returns 0..29 (30 chunks), Dense returns 10..39 (30 chunks)
    # Total unique union is 40 chunks (0..39).
    cfg = {
        "rag": {
            "retrieve_k_each": 30,
            "fusion_top": 30,
            "final_k": 5,
            "fusion_weights": {"semantic": 0.60, "bm25": 0.40},
            "fallback_min_chunks": 3,
            "fallback_score_threshold": 0.5,
            "fallback_sections": [
                {"source": "NIST_SP_800-61_Rev2.pdf", "section": "Executive Summary"},
                {"source": "NIST_SP_800-61_Rev2.pdf", "section": "3.6 Recommendations"},
            ],
        }
    }

    results = retrieve("denial of service flood", cfg=cfg, kb=fake_kb, cross_encoder=fake_ce)
    # Reranking is performed on exactly top 30 fused candidates, and top 5 are returned
    assert len(results) == 5
    for r in results:
        assert r["rank"] in range(1, 6)
        assert r["is_fallback"] is False


# ---------------------------------------------------------------------------
# Test 5: Final list has exactly 5 entries sorted by ce_score
# ---------------------------------------------------------------------------

def test_final_list_has_exactly_5_sorted_by_ce_score() -> None:
    """The returned top-5 list must be strictly sorted by cross-encoder score descending."""
    fake_kb = FakeKB(n_chunks=35)
    fake_ce = FakeCrossEncoder(score_pattern="descending")

    cfg = {
        "rag": {
            "retrieve_k_each": 30,
            "fusion_top": 30,
            "final_k": 5,
            "fusion_weights": {"semantic": 0.60, "bm25": 0.40},
            "fallback_min_chunks": 3,
            "fallback_score_threshold": 0.5,
            "fallback_sections": [
                {"source": "NIST_SP_800-61_Rev2.pdf", "section": "Executive Summary"},
                {"source": "NIST_SP_800-61_Rev2.pdf", "section": "3.6 Recommendations"},
            ],
        }
    }

    results = retrieve("syn flood mitigation", cfg=cfg, kb=fake_kb, cross_encoder=fake_ce)
    assert len(results) == 5
    ce_scores = [r["ce_score"] for r in results]
    assert ce_scores == sorted(ce_scores, reverse=True), f"Results not sorted by CE score: {ce_scores}"


# ---------------------------------------------------------------------------
# Test 6: Fallback logic fires when < 3 scores exceed 0.5, and does not fire otherwise
# ---------------------------------------------------------------------------

def test_fallback_fires_when_few_high_scores_and_not_otherwise() -> None:
    """Fallback must inject guidance when < 3 chunks exceed threshold, and not fire otherwise."""
    fake_kb = FakeKB(n_chunks=35)

    cfg = {
        "rag": {
            "retrieve_k_each": 30,
            "fusion_top": 30,
            "final_k": 5,
            "fusion_weights": {"semantic": 0.60, "bm25": 0.40},
            "fallback_min_chunks": 3,
            "fallback_score_threshold": 0.5,
            "fallback_sections": [
                {"source": "NIST_SP_800-61_Rev2.pdf", "section": "Executive Summary"},
                {"source": "NIST_SP_800-61_Rev2.pdf", "section": "3.6 Recommendations"},
            ],
        }
    }

    # Case A: Low scores (< 3 above 0.5) -> Fallback MUST fire
    fake_ce_low = FakeCrossEncoder(score_pattern="low")
    res_fallback = retrieve("obscure anomaly query", cfg=cfg, kb=fake_kb, cross_encoder=fake_ce_low)

    # Must contain 5 normal chunks + at most 2 fallback chunks
    assert len(res_fallback) > 5
    fallback_entries = [r for r in res_fallback if r["is_fallback"]]
    assert len(fallback_entries) in (1, 2)
    for fb in fallback_entries:
        assert fb["section"] in ("Executive Summary", "3.6 Recommendations")

    # Case B: High scores (>= 3 above 0.5) -> Fallback MUST NOT fire
    fake_ce_high = FakeCrossEncoder(score_pattern="high")
    res_no_fallback = retrieve("denial of service flood", cfg=cfg, kb=fake_kb, cross_encoder=fake_ce_high)

    assert len(res_no_fallback) == 5
    assert not any(r["is_fallback"] for r in res_no_fallback)


# ---------------------------------------------------------------------------
# Test 7: build_query contains class, synonyms, features, and no IP address
# ---------------------------------------------------------------------------

def test_build_query_contains_features_and_no_ips() -> None:
    """build_query must include predicted class, synonyms, features, protocol, and port, but NO IP addresses."""
    explanation = {
        "predicted_class": "DoS",
        "predicted_label": 1,
        "confidence": 0.998,
        "tier": "VERY_HIGH",
        "features": [
            {"name": "Tot Fwd Pkts", "direction": "raises"},
            {"name": "Init Bwd Win Byts", "direction": "lowers"},
        ],
        "metadata": {
            "Src IP": "192.168.1.100",
            "Dst IP": "10.0.0.5",
            "Src Port": 54321,
            "Dst Port": 80,
            "protocol": 6,
            "Timestamp": "20/02/2018 10:11:12",
            "Flow ID": "192.168.1.100-10.0.0.5-54321-80-6",
        },
    }

    q = build_query(explanation)

    # Must contain class and domain synonyms
    assert "DoS" in q
    assert "denial of service" in q

    # Must contain features and directions
    assert "Tot Fwd Pkts (raises)" in q
    assert "Init Bwd Win Byts (lowers)" in q

    # Must contain protocol name and destination port
    assert "Protocol: TCP" in q
    assert "Destination Port: 80" in q

    # Must strictly NOT contain raw IP addresses, timestamps, or flow IDs
    assert "192.168.1.100" not in q
    assert "10.0.0.5" not in q
    assert "20/02/2018" not in q
    assert "192.168.1.100-10.0.0.5" not in q


# ---------------------------------------------------------------------------
# Test 8: Fallback filters by both source and section
# ---------------------------------------------------------------------------

def test_fallback_filters_by_source_and_section() -> None:
    """When two chunks have the same section label 'Executive Summary',
    one from NIST_SP_800-94.pdf and one from NIST_SP_800-61_Rev2.pdf,
    only the SP 800-61 chunk can be injected as fallback."""
    kb = FakeKB(n_chunks=20)
    # Add two chunks both labelled "Executive Summary" from different documents
    chunk_94 = {
        "chunk_id": "chunk_94_exec",
        "text": "SP 800-94 Executive Summary on intrusion detection and prevention systems.",
        "source_name": "NIST_SP_800-94.pdf",
        "source_type": "NIST",
        "source_ref": "SP 800-94",
        "section": "Executive Summary",
        "char_start": 0,
        "char_end": 100,
    }
    chunk_61 = {
        "chunk_id": "chunk_61_exec",
        "text": "SP 800-61 Rev 2 Executive Summary on incident response lifecycle guidance.",
        "source_name": "NIST_SP_800-61_Rev2.pdf",
        "source_type": "NIST",
        "source_ref": "SP 800-61 Rev2",
        "section": "Executive Summary",
        "char_start": 0,
        "char_end": 100,
    }
    # Put SP 800-94 chunk earlier in the list to ensure order doesn't accidentally favor SP 800-61
    kb.chunks.append(chunk_94)
    kb.chunks.append(chunk_61)

    cfg = {
        "rag": {
            "retrieve_k_each": 10,
            "fusion_top": 10,
            "final_k": 5,
            "fusion_weights": {"semantic": 0.60, "bm25": 0.40},
            "fallback_min_chunks": 3,
            "fallback_score_threshold": 0.5,
            "fallback_sections": [
                {"source": "NIST_SP_800-61_Rev2.pdf", "section": "Executive Summary"}
            ],
        }
    }

    fake_ce_low = FakeCrossEncoder(score_pattern="low")
    res = retrieve("low scoring query", cfg=cfg, kb=kb, cross_encoder=fake_ce_low)

    fallback_chunks = [r for r in res if r["is_fallback"]]
    assert len(fallback_chunks) >= 1
    # Ensure SP 800-61 was injected, and SP 800-94 was NOT injected
    fb_ids = [r["chunk_id"] for r in fallback_chunks]
    assert "chunk_61_exec" in fb_ids
    assert "chunk_94_exec" not in fb_ids
    for fb in fallback_chunks:
        assert fb["source_name"] == "NIST_SP_800-61_Rev2.pdf"

