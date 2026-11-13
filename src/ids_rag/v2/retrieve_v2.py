"""V2 retrieval module for extra attack classes (Bot, FTP-BruteForce, SSH-Bruteforce).

Extends the existing retrieval pipeline (ids_rag.rag.retrieve) with:
- Extended SYNONYMS dict covering Bot, FTP-BruteForce, and SSH-Bruteforce.
- Extended build_query() covering the three new predicted class names.
- A merged KBIndex combining the frozen 742-chunk KB with new v2 chunks.
- A merged FAISS index stored in data/v2/kb/index/ (never data/kb/index/).
- Identical retrieval algorithm (BM25 + dense + CrossEncoder + fallback).

The existing retrieve() function is imported and used unchanged for all
retrieval steps. This module only extends the query expansion and index loading.

Frozen directories: data/kb/, data/kb/index/, and all original artifacts are
never read as output targets. The v2 index is always separate.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch

from ids_rag.config import get_project_root
from ids_rag.rag.index import KBIndex, tokenize_bm25
from ids_rag.rag.retrieve import (
    PROTOCOL_MAP,
    compute_fused_score,
    get_cached_cross_encoder,
    min_max_normalize_bm25,
)
from ids_rag.v2.prepare_extra import load_v2_config

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise ImportError("rank_bm25 is required. Install with: pip install rank-bm25") from e


# ---------------------------------------------------------------------------
# Extended synonym dictionary (adds Bot, FTP-BruteForce, SSH-Bruteforce)
# ---------------------------------------------------------------------------

SYNONYMS_V2: dict[str, list[str]] = {
    # Existing classes (preserved from retrieve.py)
    "DoS": ["denial of service", "resource exhaustion", "flooding"],
    "DDoS": ["distributed denial of service", "botnet flood", "amplification attack"],
    "Benign": ["normal traffic", "baseline profiling", "false positive"],
    # New extra classes
    "Bot": [
        "botnet", "command and control", "C2", "malware infection",
        "beaconing", "bot traffic", "T1583.002", "T1071",
    ],
    "FTP-BruteForce": [
        "brute force", "password guessing", "credential attack",
        "FTP authentication", "port 21", "T1110", "T1110.001",
    ],
    "SSH-Bruteforce": [
        "SSH brute force", "password spraying", "credential attack",
        "SSH authentication", "port 22", "T1110", "T1110.003",
    ],
}


def expand_query_v2(query: str) -> str:
    """Expand query with v2 synonyms (superset of original expand_query)."""
    expanded_parts = [query]
    q_lower = query.lower()

    # Order matters: check most-specific first
    if "ssh" in q_lower and ("brute" in q_lower or "ssh-bruteforce" in q_lower):
        expanded_parts.extend(SYNONYMS_V2["SSH-Bruteforce"])
    elif "ftp" in q_lower and ("brute" in q_lower or "ftp-bruteforce" in q_lower):
        expanded_parts.extend(SYNONYMS_V2["FTP-BruteForce"])
    elif "bot" in q_lower and "botnet" not in q_lower:
        # "bot" match — also capture "botnet" explicitly
        expanded_parts.extend(SYNONYMS_V2["Bot"])
    elif "botnet" in q_lower:
        expanded_parts.extend(SYNONYMS_V2["Bot"])
    elif "ddos" in q_lower:
        expanded_parts.extend(SYNONYMS_V2["DDoS"])
    elif "dos" in q_lower:
        expanded_parts.extend(SYNONYMS_V2["DoS"])

    if "benign" in q_lower:
        expanded_parts.extend(SYNONYMS_V2["Benign"])

    return " ".join(dict.fromkeys(expanded_parts))


def build_query_v2(explanation: dict[str, Any]) -> str:
    """Build a retrieval query from a v2 flow explanation dictionary.

    Identical to the original build_query() but uses v2 synonyms for all 6 classes.
    Raw identifiers (IP, flow ID) are excluded; protocol and port are retained.
    """
    pred_class = explanation.get("predicted_class", "")
    syns = SYNONYMS_V2.get(pred_class, [])
    if syns:
        class_str = f"{pred_class} ({', '.join(syns[:3])})"
    else:
        class_str = pred_class

    features = explanation.get("features", [])
    feat_strs: list[str] = []
    for f in features:
        name = f.get("name", "")
        direction = f.get("direction", "raises")
        if name:
            feat_strs.append(f"{name} ({direction})")

    feat_text = f"Key indicators: {', '.join(feat_strs)}" if feat_strs else ""

    meta = explanation.get("metadata", {})
    proto_val = meta.get("protocol")
    proto_str = f"Protocol: {PROTOCOL_MAP.get(proto_val, str(proto_val))}" if proto_val is not None else ""

    dst_port = meta.get("Dst Port")
    port_str = f"Destination Port: {dst_port}" if dst_port is not None else ""

    parts = [p for p in [class_str, proto_str, port_str, feat_text] if p]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# V2 merged index builder
# ---------------------------------------------------------------------------

def _compute_combined_sha256(paths: list[Path]) -> str:
    """Compute a combined SHA-256 from multiple files for stale detection."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
    return h.hexdigest()


def build_v2_merged_index(
    base_chunks_path: Path,
    v2_chunks_path: Path,
    index_dir: Path,
    embed_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    embed_device: str = "cpu",
    embed_batch_size: int = 32,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    force_rebuild: bool = False,
) -> KBIndex:
    """Build or load a merged KBIndex from the frozen KB + v2 extra chunks.

    Index stored in index_dir (must NOT be data/kb/index/).
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / "index_meta_v2.json"
    faiss_path = index_dir / "faiss_v2.index"
    corpus_path = index_dir / "bm25_corpus_v2.json"

    # Stale detection: compare SHA-256 of both input files
    combined_sha = _compute_combined_sha256([base_chunks_path, v2_chunks_path])
    is_fresh = False
    if not force_rebuild and meta_path.exists() and faiss_path.exists() and corpus_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("combined_sha256") == combined_sha:
                is_fresh = True
        except Exception:
            is_fresh = False

    # Load all chunks from both files
    def _load_jsonl(path: Path) -> list[dict]:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in lines if l.strip()]

    base_chunks = _load_jsonl(base_chunks_path)
    v2_chunks = _load_jsonl(v2_chunks_path)

    # Merge: base chunks first (preserve positions), then v2 chunks
    all_chunks = base_chunks + v2_chunks
    print(f"Merged KB: {len(base_chunks)} base + {len(v2_chunks)} v2 = {len(all_chunks)} total chunks")

    if is_fresh:
        print(f"Loading existing v2 index from {index_dir} (up to date).")
        # Load BM25 from persisted tokenized corpus
        with open(corpus_path, encoding="utf-8") as f:
            tokenized_corpus = json.load(f)
        bm25 = BM25Okapi(tokenized_corpus, k1=bm25_k1, b=bm25_b)

        # Load FAISS index
        faiss_index = faiss.read_index(str(faiss_path))

        # Load embeddings from FAISS (reconstruct for dense_scores)
        n = faiss_index.ntotal
        dim = faiss_index.d
        embeddings = np.zeros((n, dim), dtype=np.float32)
        faiss_index.reconstruct_n(0, n, embeddings)

        with open(meta_path, encoding="utf-8") as f:
            meta_data = json.load(f)

        # Load sentence transformer for query encoding
        from sentence_transformers import SentenceTransformer
        dense_model = SentenceTransformer(embed_model_name, device=embed_device)

        return KBIndex(
            chunks=all_chunks,
            bm25=bm25,
            faiss_index=faiss_index,
            embeddings=embeddings,
            meta=meta_data,
            dense_model=dense_model,
            device=embed_device,
        )

    # Build from scratch
    print(f"Building v2 merged index ({len(all_chunks)} chunks)...")
    t0 = time.perf_counter()

    # BM25
    texts = [c["text"] for c in all_chunks]
    tokenized_corpus = [tokenize_bm25(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus, k1=bm25_k1, b=bm25_b)

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(tokenized_corpus, f)
    print(f"  BM25 corpus built and saved ({len(tokenized_corpus)} docs)")

    # Dense embeddings
    from sentence_transformers import SentenceTransformer
    dense_model = SentenceTransformer(embed_model_name, device=embed_device)
    embeddings = dense_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=embed_batch_size,
        device=embed_device,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # FAISS flat inner product index
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, str(faiss_path))
    print(f"  FAISS index built ({faiss_index.ntotal} vectors, dim={dim}) and saved")

    elapsed = time.perf_counter() - t0
    meta_data = {
        "combined_sha256": combined_sha,
        "chunk_count": len(all_chunks),
        "base_chunks": len(base_chunks),
        "v2_chunks": len(v2_chunks),
        "embed_model": embed_model_name,
        "embed_dim": dim,
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    print(f"  Index built in {elapsed:.1f}s. Meta saved to {meta_path}")

    return KBIndex(
        chunks=all_chunks,
        bm25=bm25,
        faiss_index=faiss_index,
        embeddings=embeddings,
        meta=meta_data,
        dense_model=dense_model,
        device=embed_device,
    )


# ---------------------------------------------------------------------------
# Cached v2 KB singleton
# ---------------------------------------------------------------------------

_CACHED_V2_KB: KBIndex | None = None


def get_cached_v2_kb(cfg: dict[str, Any] | None = None) -> KBIndex:
    """Load and cache the v2 merged KBIndex singleton."""
    global _CACHED_V2_KB
    if _CACHED_V2_KB is None:
        root = get_project_root()
        if cfg is None:
            cfg = load_v2_config()

        rag_cfg = cfg.get("rag", {})
        embed_model = rag_cfg.get("embed_model", "sentence-transformers/all-mpnet-base-v2")
        embed_device_cfg = rag_cfg.get("embed_device", "auto")
        embed_device = "cuda" if (embed_device_cfg == "auto" and torch.cuda.is_available()) else "cpu"
        embed_batch_size = int(rag_cfg.get("embed_batch_size", 32))
        bm25_k1 = float(rag_cfg.get("bm25_k1", 1.5))
        bm25_b = float(rag_cfg.get("bm25_b", 0.75))

        base_chunks_path = root / "data" / "kb" / "chunks.jsonl"
        v2_chunks_path = root / "data" / "v2" / "kb" / "chunks_v2.jsonl"
        index_dir = root / "data" / "v2" / "kb" / "index"

        if not base_chunks_path.exists():
            raise FileNotFoundError(f"Frozen base KB not found at {base_chunks_path}.")
        if not v2_chunks_path.exists():
            raise FileNotFoundError(
                f"V2 KB chunks not found at {v2_chunks_path}. "
                f"Run `python scripts/v2_build_kb_extra.py` first."
            )

        _CACHED_V2_KB = build_v2_merged_index(
            base_chunks_path=base_chunks_path,
            v2_chunks_path=v2_chunks_path,
            index_dir=index_dir,
            embed_model_name=embed_model,
            embed_device=embed_device,
            embed_batch_size=embed_batch_size,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
    return _CACHED_V2_KB


# ---------------------------------------------------------------------------
# V2 retrieve (identical algorithm to retrieve.py, uses v2 KB and query expansion)
# ---------------------------------------------------------------------------

def retrieve_v2(
    query: str,
    cfg: dict[str, Any] | None = None,
    kb: KBIndex | None = None,
    cross_encoder: Any | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k relevant chunks using the merged v2 KB and extended query expansion.

    Identical algorithm to ids_rag.rag.retrieve.retrieve() with:
    - expand_query_v2 instead of expand_query (covers Bot/BruteForce classes)
    - Merged KB (742 base + v2 extra chunks)
    - V2 fallback sections for new attack classes
    """
    if cfg is None:
        cfg = load_v2_config()

    rag_cfg = cfg.get("rag", {})
    retrieve_k_each = int(rag_cfg.get("retrieve_k_each", 30))
    fusion_top = int(rag_cfg.get("fusion_top", 30))
    final_k = int(rag_cfg.get("final_k", 5))

    weights = rag_cfg.get("fusion_weights", {})
    w_dense = float(weights.get("semantic", 0.60))
    w_bm25 = float(weights.get("bm25", 0.40))

    fallback_min = int(rag_cfg.get("fallback_min_chunks", 3))
    fallback_thresh = float(rag_cfg.get("fallback_score_threshold", 0.5))

    # V2 fallback sections include NIST 800-61 (existing) + v2 brute-force and botnet chunks
    raw_fb_sections = rag_cfg.get(
        "v2_fallback_sections",
        [
            {"source": "NIST_SP_800-61_Rev2.pdf", "section": "Executive Summary"},
            {"source": "NIST_SP_800-61_Rev2.pdf", "section": "3.6 Recommendations"},
            {"source": "MITRE_T1110.txt", "section": "Mitigations"},
            {"source": "MITRE_T1583_002.txt", "section": "Mitigations"},
        ],
    )
    fallback_rules: list[dict[str, str]] = [
        {"source": r.get("source", ""), "section": r.get("section", "")}
        for r in raw_fb_sections
        if isinstance(r, dict)
    ]

    if kb is None:
        kb = get_cached_v2_kb(cfg)
    if cross_encoder is None:
        cross_encoder = get_cached_cross_encoder(cfg)

    # 1. Expand query with v2 synonyms
    exp_query = expand_query_v2(query)

    # 2. Candidate pooling
    bm25_top = kb.search_bm25(exp_query, k=retrieve_k_each)
    dense_top = kb.search_dense(exp_query, k=retrieve_k_each)

    bm25_rank_map = {pos: rank for rank, (pos, _) in enumerate(bm25_top, 1)}
    dense_rank_map = {pos: rank for rank, (pos, _) in enumerate(dense_top, 1)}

    cand_positions = list(dict.fromkeys([p for p, _ in bm25_top] + [p for p, _ in dense_top]))

    if not cand_positions:
        return []

    # 3. Fused score
    all_dense_scores = kb.dense_scores(exp_query)
    all_bm25_scores = kb.bm25_scores(exp_query)

    cand_dense_scores = [float(all_dense_scores[p]) for p in cand_positions]
    cand_bm25_raw = [float(all_bm25_scores[p]) for p in cand_positions]
    cand_bm25_norm = min_max_normalize_bm25(cand_bm25_raw)

    fused_candidates: list[dict[str, Any]] = []
    for pos, d_score, b_norm in zip(cand_positions, cand_dense_scores, cand_bm25_norm):
        f_score = compute_fused_score(d_score, b_norm, w_dense=w_dense, w_bm25=w_bm25)
        fused_candidates.append({"pos": pos, "dense_score": d_score, "bm25_norm": b_norm, "fused_score": f_score})

    fused_candidates.sort(key=lambda x: x["fused_score"], reverse=True)
    top_fused = fused_candidates[:fusion_top]

    # 4. CrossEncoder reranking
    pairs = [(query, kb.get_chunk(item["pos"])["text"]) for item in top_fused]
    raw_ce_scores = cross_encoder.predict(pairs)
    ce_probs = 1.0 / (1.0 + np.exp(-np.array(raw_ce_scores, dtype=np.float32)))

    for item, prob in zip(top_fused, ce_probs):
        item["ce_score"] = float(prob)

    top_fused.sort(key=lambda x: x["ce_score"], reverse=True)
    selected_top = top_fused[:final_k]

    results: list[dict[str, Any]] = []
    for rank, item in enumerate(selected_top, 1):
        pos = item["pos"]
        chunk = kb.get_chunk(pos)
        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source_name": chunk["source_name"],
            "source_type": chunk["source_type"],
            "source_ref": chunk["source_ref"],
            "section": chunk["section"],
            "bm25_rank": bm25_rank_map.get(pos),
            "dense_rank": dense_rank_map.get(pos),
            "bm25_norm": item["bm25_norm"],
            "dense_score": item["dense_score"],
            "fused_score": item["fused_score"],
            "ce_score": item["ce_score"],
            "is_fallback": False,
            "rank": rank,
            "char_start": chunk.get("char_start", 0),
            "char_end": chunk.get("char_end", 0),
        })

    # 5. Fallback injection
    high_score_count = sum(1 for r in results if r["ce_score"] >= fallback_thresh)
    if high_score_count < fallback_min:
        existing_ids = {r["chunk_id"] for r in results}
        fallback_chunks: list[dict[str, Any]] = []

        for rule in fallback_rules:
            req_src = rule["source"]
            req_sec = rule["section"]
            for c in kb.chunks:
                if c.get("source_name") == req_src and c.get("section") == req_sec:
                    if c["chunk_id"] not in existing_ids:
                        fallback_chunks.append(c)
                        existing_ids.add(c["chunk_id"])
                        break
            if len(fallback_chunks) >= 2:
                break

        current_rank = len(results) + 1
        for fb_c in fallback_chunks[:2]:
            results.append({
                "chunk_id": fb_c["chunk_id"],
                "text": fb_c["text"],
                "source_name": fb_c["source_name"],
                "source_type": fb_c["source_type"],
                "source_ref": fb_c["source_ref"],
                "section": fb_c["section"],
                "bm25_rank": None,
                "dense_rank": None,
                "bm25_norm": 0.0,
                "dense_score": 0.0,
                "fused_score": 0.0,
                "ce_score": 0.0,
                "is_fallback": True,
                "rank": current_rank,
                "char_start": fb_c.get("char_start", 0),
                "char_end": fb_c.get("char_end", 0),
            })
            current_rank += 1

    return results

