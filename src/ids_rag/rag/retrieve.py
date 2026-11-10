"""Hybrid retrieval with Reciprocal Rank Fusion, BM25 normalization, and cross-encoder reranking.

Implements the paper's Algorithm 2 (arXiv 2605.17960):
1. Query expansion with domain synonyms (DoS, DDoS, Benign).
2. Candidate pooling: retrieve top-k (30) from BM25 and top-k (30) from dense FAISS; union.
3. Candidate score fusion: min-max normalized BM25 over the candidate set, combined with
   dense cosine similarity (0.60 * dense + 0.40 * norm_bm25). Keeps top fusion_top (30).
4. Cross-encoder reranking: CrossEncoder(cross-encoder/ms-marco-MiniLM-L-6-v2) on (query, chunk text).
   Applies explicit logistic sigmoid transformation to map raw logits into [0, 1]. Top final_k (5).
5. Fallback injection: If fewer than fallback_min_chunks (3) have cross-encoder score >= 0.5,
   injects general NIST SP 800-61 guidance chunks from configured fallback sections.
6. Query builder: builds search queries from explanation dictionaries without raw identifiers.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ids_rag.config import get_project_root, load_config
from ids_rag.rag.index import KBIndex, load_index

# ---------------------------------------------------------------------------
# Domain synonym dictionary for query expansion
# ---------------------------------------------------------------------------

SYNONYMS: dict[str, list[str]] = {
    "DoS": ["denial of service", "resource exhaustion", "flooding"],
    "DDoS": ["distributed denial of service", "botnet flood", "amplification attack"],
    # The Benign entry is our own addition because the paper gives examples only for DoS and DDoS.
    "Benign": ["normal traffic", "baseline profiling", "false positive"],
}

PROTOCOL_MAP: dict[int, str] = {
    6: "TCP",
    17: "UDP",
    1: "ICMP",
}


def expand_query(query: str) -> str:
    """Expand query with domain synonyms for DoS, DDoS, or Benign."""
    expanded_parts = [query]
    q_lower = query.lower()

    # Match DDoS before DoS to prevent substring collisions
    if re.search(r"\bddos\b", q_lower):
        expanded_parts.extend(SYNONYMS["DDoS"])
    elif re.search(r"\bdos\b", q_lower):
        expanded_parts.extend(SYNONYMS["DoS"])

    if re.search(r"\bbenign\b", q_lower):
        expanded_parts.extend(SYNONYMS["Benign"])

    # Preserve uniqueness while maintaining order
    return " ".join(dict.fromkeys(expanded_parts))


# ---------------------------------------------------------------------------
# Query builder from Stage 3 explanation dictionary
# ---------------------------------------------------------------------------

def build_query(explanation: dict[str, Any]) -> str:
    """Build a search query from a Stage 3 flow explanation dictionary.

    Extracts:
    - Predicted class (expanded with domain synonyms)
    - Key feature names and their directions (raises / lowers)
    - Protocol name (TCP for 6, UDP for 17) and destination port when present.

    Note: The paper's query description mentions metadata, but raw network identifiers
    (IP addresses, timestamps, flow IDs) add only noise to text retrieval and are
    strictly excluded; only protocol and destination port are retained.
    """
    pred_class = explanation.get("predicted_class", "")
    syns = SYNONYMS.get(pred_class, [])
    if syns:
        class_str = f"{pred_class} ({', '.join(syns)})"
    else:
        class_str = pred_class

    # Top-k feature names and directions
    features = explanation.get("features", [])
    feat_strs: list[str] = []
    for f in features:
        name = f.get("name", "")
        direction = f.get("direction", "raises")
        if name:
            feat_strs.append(f"{name} ({direction})")

    feat_text = f"Key indicators: {', '.join(feat_strs)}" if feat_strs else ""

    # Protocol and port metadata
    meta = explanation.get("metadata", {})
    proto_val = meta.get("protocol")
    if proto_val in PROTOCOL_MAP:
        proto_str = f"Protocol: {PROTOCOL_MAP[proto_val]}"
    elif proto_val is not None:
        proto_str = f"Protocol: {proto_val}"
    else:
        proto_str = ""

    dst_port = meta.get("Dst Port")
    port_str = f"Destination Port: {dst_port}" if dst_port is not None else ""

    parts = [p for p in [class_str, proto_str, port_str, feat_text] if p]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Min-Max Normalization & Score Fusion helpers
# ---------------------------------------------------------------------------

def min_max_normalize_bm25(scores: list[float]) -> list[float]:
    """Min-max normalize BM25 scores over the candidate pool.

    Note: As specified by the paper's Algorithm 2, BM25 scores are normalized
    over the candidate set so they are commensurable with cosine similarity.
    Guards against division-by-zero when max == min.
    """
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    diff = max_s - min_s
    if diff <= 1e-9:
        return [1.0] * len(scores)
    return [(s - min_s) / diff for s in scores]


def compute_fused_score(dense_score: float, bm25_norm: float, w_dense: float = 0.60, w_bm25: float = 0.40) -> float:
    """Compute weighted hybrid fusion score: 0.60 * dense + 0.40 * norm_bm25."""
    return float(w_dense * dense_score + w_bm25 * bm25_norm)


# ---------------------------------------------------------------------------
# Cached Singletons
# ---------------------------------------------------------------------------

_CACHED_KB: KBIndex | None = None
_CACHED_CE: Any | None = None


def get_cached_kb(cfg: dict[str, Any] | None = None) -> KBIndex:
    """Load and cache the KBIndex singleton."""
    global _CACHED_KB
    if _CACHED_KB is None:
        _CACHED_KB = load_index(cfg)
    return _CACHED_KB


def get_cached_cross_encoder(cfg: dict[str, Any] | None = None) -> Any:
    """Load and cache the CrossEncoder singleton."""
    global _CACHED_CE
    if _CACHED_CE is None:
        if cfg is None:
            cfg = load_config()
        rag_cfg = cfg.get("rag", {})
        reranker_name = rag_cfg.get("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        embed_device = rag_cfg.get("embed_device", "auto")
        if embed_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = embed_device

        from sentence_transformers import CrossEncoder

        _CACHED_CE = CrossEncoder(reranker_name, device=device)
    return _CACHED_CE


# ---------------------------------------------------------------------------
# Core Retrieval Function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    cfg: dict[str, Any] | None = None,
    kb: KBIndex | None = None,
    cross_encoder: Any | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k relevant chunks for a query using hybrid search and reranking.

    Steps:
    1. Expand query with synonyms.
    2. Retrieve top-k (30) from BM25 and top-k (30) from dense FAISS; take union.
    3. Compute dense and normalized BM25 scores; fuse with 0.60 / 0.40 weights; keep top 30.
    4. Rerank with CrossEncoder; apply logistic sigmoid; take top final_k (5).
    5. Fallback: If fewer than fallback_min_chunks have score >= threshold, append NIST general guidance.
    """
    if cfg is None:
        cfg = load_config()

    rag_cfg = cfg.get("rag", {})
    retrieve_k_each = int(rag_cfg.get("retrieve_k_each", 30))
    fusion_top = int(rag_cfg.get("fusion_top", 30))
    final_k = int(rag_cfg.get("final_k", 5))

    weights = rag_cfg.get("fusion_weights", {})
    w_dense = float(weights.get("semantic", 0.60))
    w_bm25 = float(weights.get("bm25", 0.40))

    fallback_min = int(rag_cfg.get("fallback_min_chunks", 3))
    fallback_thresh = float(rag_cfg.get("fallback_score_threshold", 0.5))
    raw_fb_sections = rag_cfg.get(
        "fallback_sections",
        [
            {"source": "NIST_SP_800-61_Rev2.pdf", "section": "Executive Summary"},
            {"source": "NIST_SP_800-61_Rev2.pdf", "section": "3.6 Recommendations"},
        ],
    )
    fallback_rules: list[dict[str, str]] = []
    for item in raw_fb_sections:
        if isinstance(item, dict):
            fallback_rules.append(
                {
                    "source": item.get("source", "NIST_SP_800-61_Rev2.pdf"),
                    "section": item.get("section", ""),
                }
            )
        else:
            fallback_rules.append(
                {
                    "source": "NIST_SP_800-61_Rev2.pdf",
                    "section": str(item),
                }
            )

    if kb is None:
        kb = get_cached_kb(cfg)
    if cross_encoder is None:
        cross_encoder = get_cached_cross_encoder(cfg)

    # 1. Expand query
    exp_query = expand_query(query)

    # 2. Candidate pooling: retrieve top-k from BM25 and dense
    bm25_top = kb.search_bm25(exp_query, k=retrieve_k_each)
    dense_top = kb.search_dense(exp_query, k=retrieve_k_each)

    bm25_rank_map = {pos: rank for rank, (pos, _) in enumerate(bm25_top, 1)}
    dense_rank_map = {pos: rank for rank, (pos, _) in enumerate(dense_top, 1)}

    # Union of candidates preserving discovery order
    cand_positions = list(dict.fromkeys([p for p, _ in bm25_top] + [p for p, _ in dense_top]))

    if not cand_positions:
        return []

    # 3. Compute dense and BM25 scores for all candidates
    all_dense_scores = kb.dense_scores(exp_query)
    all_bm25_scores = kb.bm25_scores(exp_query)

    cand_dense_scores = [float(all_dense_scores[p]) for p in cand_positions]
    cand_bm25_raw = [float(all_bm25_scores[p]) for p in cand_positions]
    cand_bm25_norm = min_max_normalize_bm25(cand_bm25_raw)

    # Compute fused score for each candidate
    fused_candidates: list[dict[str, Any]] = []
    for pos, d_score, b_norm in zip(cand_positions, cand_dense_scores, cand_bm25_norm):
        f_score = compute_fused_score(d_score, b_norm, w_dense=w_dense, w_bm25=w_bm25)
        fused_candidates.append(
            {
                "pos": pos,
                "dense_score": d_score,
                "bm25_norm": b_norm,
                "fused_score": f_score,
            }
        )

    # Keep top fusion_top by fused score descending
    fused_candidates.sort(key=lambda x: x["fused_score"], reverse=True)
    top_fused = fused_candidates[:fusion_top]

    # 4. Cross-encoder reranking
    pairs = [(query, kb.get_chunk(item["pos"])["text"]) for item in top_fused]
    raw_ce_scores = cross_encoder.predict(pairs)

    # Note: CrossEncoder.predict() returns raw unbounded logits.
    # We explicitly apply the logistic sigmoid function to bound relevance scores in [0, 1].
    ce_probs = 1.0 / (1.0 + np.exp(-np.array(raw_ce_scores, dtype=np.float32)))

    for item, prob in zip(top_fused, ce_probs):
        item["ce_score"] = float(prob)

    # Sort candidates by cross-encoder score descending and take top final_k
    top_fused.sort(key=lambda x: x["ce_score"], reverse=True)
    selected_top = top_fused[:final_k]

    # Build initial result list
    results: list[dict[str, Any]] = []
    for rank, item in enumerate(selected_top, 1):
        pos = item["pos"]
        chunk = kb.get_chunk(pos)
        results.append(
            {
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
            }
        )

    # 5. Fallback injection (paper rule interpretation)
    # If fewer than fallback_min_chunks have ce_score >= fallback_score_threshold,
    # inject general NIST SP 800-61 guidance chunks from configured fallback sections.
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
                        break  # Take at most 1 chunk per fallback entry

            if len(fallback_chunks) >= 2:
                break

        current_rank = len(results) + 1
        for fb_c in fallback_chunks[:2]:
            results.append(
                {
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
                }
            )
            current_rank += 1

    return results


# ---------------------------------------------------------------------------
# CLI Check & Diagnostic runner
# ---------------------------------------------------------------------------

_INDEX_QUERIES: tuple[str, ...] = (
    "denial of service flood attack",
    "T1498 network denial of service",
    "SYN flood mitigation",
    "how to prioritize an incident",
    "distributed attack from many sources",
)


def _cli_check(cfg: dict[str, Any]) -> None:
    """Run retrieval diagnostics across 5 standard queries and 4 Stage 3 explanations."""
    print("Loading KB index and Cross-Encoder...")
    kb = get_cached_kb(cfg)
    cross_encoder = get_cached_cross_encoder(cfg)
    print(f"Loaded index ({len(kb.chunks)} chunks) and cross-encoder.")

    root = get_project_root()
    expl_file = root / "artifacts" / "sample_explanations.json"
    explanations: list[dict[str, Any]] = []
    if expl_file.is_file():
        with open(expl_file, "r", encoding="utf-8") as f:
            explanations = json.load(f)

    # Collect 9 queries: 5 index check queries + 4 built queries
    all_runs: list[tuple[str, str]] = []
    for q in _INDEX_QUERIES:
        all_runs.append(("Direct Query", q))

    for i, expl in enumerate(explanations):
        built_q = build_query(expl)
        case_label = expl.get("case", f"Explanation {i+1}")
        all_runs.append((f"Stage 3 Expl ({case_label})", built_q))

    print(f"\nRunning {len(all_runs)} benchmark retrieval queries...\n")

    sep_wide = "=" * 90
    sep_sub = "-" * 90

    total_queries = len(all_runs)
    appendix_a_total = 0
    fallback_fired_total = 0
    chunks_above_half_total = 0
    overlap_shares: list[float] = []

    for idx, (source_tag, query_text) in enumerate(all_runs, 1):
        results = retrieve(query_text, cfg=cfg, kb=kb, cross_encoder=cross_encoder)

        top_5 = [r for r in results if not r["is_fallback"]]
        fallback_fired = any(r["is_fallback"] for r in results)
        if fallback_fired:
            fallback_fired_total += 1

        chunks_above = sum(1 for r in top_5 if r["ce_score"] >= 0.5)
        chunks_above_half_total += chunks_above

        # Check Appendix A slots
        app_a_count = sum(1 for r in top_5 if "Appendix A" in r["section"])
        appendix_a_total += app_a_count

        # Compute pairwise character overlap within the same document among top-5
        pairs_count = 0
        overlapping_pairs = 0
        n_top = len(top_5)
        for i in range(n_top):
            for j in range(i + 1, n_top):
                pairs_count += 1
                c1 = top_5[i]
                c2 = top_5[j]
                if c1["source_name"] == c2["source_name"]:
                    s1, e1 = c1.get("char_start", 0), c1.get("char_end", 0)
                    s2, e2 = c2.get("char_start", 0), c2.get("char_end", 0)
                    if max(s1, s2) < min(e1, e2):
                        overlapping_pairs += 1

        pair_share = (overlapping_pairs / pairs_count) if pairs_count > 0 else 0.0
        overlap_shares.append(pair_share)

        print(sep_wide)
        print(f"Query [{idx}/{total_queries}] ({source_tag})")
        print(f"Full Query: {query_text!r}")
        print(f"Fallback Fired: {fallback_fired} | Chunks with CE >= 0.5: {chunks_above}/5 | Pair Overlap Share: {pair_share:.1%}")
        print(sep_sub)

        for r in results:
            tag = "[FALLBACK]" if r["is_fallback"] else f"Rank {r['rank']}"
            b_str = f"{r['bm25_rank']:>2}" if r["bm25_rank"] is not None else " -"
            d_str = f"{r['dense_rank']:>2}" if r["dense_rank"] is not None else " -"
            preview = r["text"][:100].replace("\n", " ").strip()
            print(f"  {tag:<10} | CE: {r['ce_score']:.4f} | BM25_rk: {b_str} | Dense_rk: {d_str} | {r['source_name']:<22} | {r['section']}")
            print(f"             Snippet: {preview}...")

    # Print summary statistics across all 9 queries
    mean_above_half = chunks_above_half_total / total_queries
    mean_overlap_share = sum(overlap_shares) / len(overlap_shares) if overlap_shares else 0.0

    print(f"\n{sep_wide}")
    print("SUMMARY RETRIEVAL DIAGNOSTICS (9 Queries Total)")
    print(sep_sub)
    print(f"Total top-5 slots evaluated           : {total_queries * 5}")
    print(f"Appendix A chunks in top-5 slots      : {appendix_a_total} / {total_queries * 5} ({appendix_a_total / (total_queries * 5):.1%})")
    print(f"Queries where fallback fired          : {fallback_fired_total} / {total_queries} ({fallback_fired_total / total_queries:.1%})")
    print(f"Mean number of chunks with CE >= 0.5  : {mean_above_half:.2f} of 5")
    print(f"Mean intra-doc pairwise overlap share : {mean_overlap_share:.1%}")
    print(sep_wide)


def main() -> None:
    """CLI entry point for retrieval diagnostics."""
    parser = argparse.ArgumentParser(description="Hybrid retrieval and cross-encoder reranking.")
    parser.add_argument("--check", action="store_true", help="Run 9 benchmark diagnostic retrieval queries")
    parser.add_argument("--query", type=str, default="", help="Custom query string to retrieve")
    args = parser.parse_args()

    cfg = load_config()
    if args.check:
        _cli_check(cfg)
    elif args.query:
        results = retrieve(args.query, cfg)
        for r in results:
            print(f"[{r['rank']}] CE={r['ce_score']:.4f} | {r['source_name']} | {r['section']}")
            print(f"    {r['text'][:120]}...\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
