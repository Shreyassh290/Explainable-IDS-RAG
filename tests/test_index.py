"""Unit tests for ids_rag.rag.index using synthetic data and fake encoder (no model download)."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ids_rag.rag.index import (
    build_index,
    compute_file_sha256,
    load_index,
    tokenize_bm25,
)

# ---------------------------------------------------------------------------
# Deterministic fake encoder for testing without Hugging Face models
# ---------------------------------------------------------------------------

def fake_encoder(texts: list[str]) -> np.ndarray:
    """Deterministic hashed bag-of-words L2-normalized vector encoder."""
    dim = 64
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, txt in enumerate(texts):
        tokens = tokenize_bm25(txt)
        for t in tokens:
            idx = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % dim
            vecs[i, idx] += 1.0
        norm = float(np.linalg.norm(vecs[i]))
        if norm > 1e-9:
            vecs[i] /= norm
        else:
            vecs[i, 0] = 1.0
    return np.ascontiguousarray(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Test 1: Tokenizer preserves identifiers and lowercases
# ---------------------------------------------------------------------------

def test_tokenizer_preserves_identifiers() -> None:
    """The tokenizer must keep 't1498.001' and '800-61' whole and lowercase."""
    text = "T1498.001 Direct Network Flood in NIST SP 800-61 Rev 2 guidelines."
    tokens = tokenize_bm25(text)

    assert "t1498.001" in tokens, f"Expected 't1498.001' in tokens, got: {tokens}"
    assert "800-61" in tokens, f"Expected '800-61' in tokens, got: {tokens}"
    assert "direct" in tokens
    assert "flood" in tokens
    assert "nist" in tokens
    assert "sp" in tokens

    # Check that upper case characters are strictly lowercased
    for tok in tokens:
        assert tok == tok.lower(), f"Token {tok!r} is not lowercased"

    # Trailing dots/punctuation outside tokens must be stripped
    punctuated = "Attack ID: T1498.001... Refer to SP 800-61."
    p_tokens = tokenize_bm25(punctuated)
    assert "t1498.001" in p_tokens
    assert "800-61" in p_tokens


# ---------------------------------------------------------------------------
# Helper: create synthetic chunks file
# ---------------------------------------------------------------------------

def _create_sample_chunks(filepath: Path) -> list[dict[str, Any]]:
    chunks = [
        {
            "chunk_id": "chunk_0",
            "source_name": "MITRE_T1498.html",
            "source_type": "MITRE",
            "source_ref": "T1498",
            "section": "Description",
            "sections": ["Description"],
            "text": "Adversaries conduct network denial of service flooding targeting bandwidth.",
            "token_count": 10,
            "char_start": 0,
            "char_end": 75,
        },
        {
            "chunk_id": "chunk_1",
            "source_name": "MITRE_T1498_001.html",
            "source_type": "MITRE",
            "source_ref": "T1498.001",
            "section": "Description",
            "sections": ["Description"],
            "text": "Direct network floods like UDP reflection and SYN floods saturate connections.",
            "token_count": 12,
            "char_start": 76,
            "char_end": 155,
        },
        {
            "chunk_id": "chunk_2",
            "source_name": "NIST_SP_800-61_Rev2.pdf",
            "source_type": "NIST",
            "source_ref": "SP 800-61",
            "section": "3.2.1 Attack Vectors",
            "sections": ["3.2.1 Attack Vectors"],
            "text": "Attack vectors include denial of service, web compromises, and unauthorized access.",
            "token_count": 12,
            "char_start": 156,
            "char_end": 240,
        },
        {
            "chunk_id": "chunk_3",
            "source_name": "NIST_SP_800-61_Rev2.pdf",
            "source_type": "NIST",
            "source_ref": "SP 800-61",
            "section": "3.2.6 Incident Prioritization",
            "sections": ["3.2.6 Incident Prioritization"],
            "text": "Incident prioritization guidelines emphasize business impact and technical criticality.",
            "token_count": 10,
            "char_start": 241,
            "char_end": 330,
        },
        {
            "chunk_id": "chunk_4",
            "source_name": "NIST_SP_800-61_Rev2.pdf",
            "source_type": "NIST",
            "source_ref": "SP 800-61",
            "section": "3.3.1 Choosing a Containment Strategy",
            "sections": ["3.3.1 Choosing a Containment Strategy"],
            "text": "Containment strategies must isolate affected systems while preserving forensic evidence.",
            "token_count": 10,
            "char_start": 331,
            "char_end": 420,
        },
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return chunks


# ---------------------------------------------------------------------------
# Test 2: Each chunk's own embedding retrieves that chunk first from FAISS
# ---------------------------------------------------------------------------

def test_chunk_own_embedding_retrieves_itself_first(tmp_path: Path) -> None:
    """Each chunk's own text embedding must retrieve that chunk at rank 1 with score ~ 1.0."""
    chunks_file = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "index"
    chunks = _create_sample_chunks(chunks_file)

    cfg = {
        "seed": 42,
        "rag": {
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "embed_model": "fake",
            "embed_batch_size": 16,
            "embed_device": "cpu",
        },
    }

    build_index(cfg=cfg, chunks_path=chunks_file, index_dir=index_dir, encoder_fn=fake_encoder)
    kb = load_index(cfg=cfg, index_dir=index_dir, chunks_path=chunks_file, encoder_fn=fake_encoder)

    for i, c in enumerate(chunks):
        dense_results = kb.search_dense(c["text"], k=3)
        top_pos, top_score = dense_results[0]

        assert top_pos == i, (
            f"Chunk {i} ('{c['chunk_id']}') did not retrieve itself at rank 1. "
            f"Got pos {top_pos} ('{chunks[top_pos]['chunk_id']}') with score {top_score:.4f}"
        )
        assert np.isclose(top_score, 1.0, atol=1e-5), (
            f"Expected self-retrieval score close to 1.0, got {top_score:.5f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Save then load gives identical search results
# ---------------------------------------------------------------------------

def test_save_then_load_gives_identical_search_results(tmp_path: Path) -> None:
    """Loading saved index yields identical search positions and scores for both BM25 and dense."""
    chunks_file = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "index"
    _create_sample_chunks(chunks_file)

    cfg = {
        "seed": 42,
        "rag": {
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "embed_model": "fake",
            "embed_batch_size": 16,
            "embed_device": "cpu",
        },
    }

    build_index(cfg=cfg, chunks_path=chunks_file, index_dir=index_dir, encoder_fn=fake_encoder)

    # Load two independent instances from disk
    kb1 = load_index(cfg=cfg, index_dir=index_dir, chunks_path=chunks_file, encoder_fn=fake_encoder)
    kb2 = load_index(cfg=cfg, index_dir=index_dir, chunks_path=chunks_file, encoder_fn=fake_encoder)

    test_queries = [
        "denial of service flood",
        "incident prioritization guidelines",
        "containment forensic evidence",
    ]

    for q in test_queries:
        # Check BM25
        res_bm25_1 = kb1.search_bm25(q, k=4)
        res_bm25_2 = kb2.search_bm25(q, k=4)
        assert res_bm25_1 == res_bm25_2, f"BM25 results mismatch for query: {q!r}"

        # Check Dense
        res_dense_1 = kb1.search_dense(q, k=4)
        res_dense_2 = kb2.search_dense(q, k=4)
        for (p1, s1), (p2, s2) in zip(res_dense_1, res_dense_2):
            assert p1 == p2, f"Dense position mismatch for query {q!r}: {p1} vs {p2}"
            assert np.isclose(s1, s2, atol=1e-6), f"Dense score mismatch for query {q!r}: {s1} vs {s2}"


# ---------------------------------------------------------------------------
# Test 4: Stale check fires when chunks.jsonl changes
# ---------------------------------------------------------------------------

def test_stale_check_fires_when_chunks_change(tmp_path: Path) -> None:
    """load_index must raise RuntimeError when chunks.jsonl SHA-256 no longer matches saved index."""
    chunks_file = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "index"
    _create_sample_chunks(chunks_file)

    cfg = {
        "seed": 42,
        "rag": {
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "embed_model": "fake",
            "embed_batch_size": 16,
            "embed_device": "cpu",
        },
    }

    build_index(cfg=cfg, chunks_path=chunks_file, index_dir=index_dir, encoder_fn=fake_encoder)

    # Initial load succeeds
    kb = load_index(cfg=cfg, index_dir=index_dir, chunks_path=chunks_file, encoder_fn=fake_encoder)
    assert len(kb.chunks) == 5

    # Modify chunks file by appending a modified chunk
    with open(chunks_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": "chunk_modified", "text": "New modified content."}) + "\n")

    # Now load_index must fail with stale error
    with pytest.raises(RuntimeError) as exc_info:
        load_index(cfg=cfg, index_dir=index_dir, chunks_path=chunks_file, encoder_fn=fake_encoder)

    err_msg = str(exc_info.value).lower()
    assert "stale" in err_msg or "sha-256" in err_msg, (
        f"Expected stale error message, got: {exc_info.value}"
    )

