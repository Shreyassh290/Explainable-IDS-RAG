"""Dense vector and sparse BM25 indexing for IDS knowledge base.

Indexes chunks in data/kb/chunks.jsonl:
- Sparse BM25: rank_bm25.BM25Okapi over custom alphanumeric tokens (internal dots/hyphens preserved).
  Persists tokenized corpus as JSON; rebuilds BM25 on load.
- Dense FAISS: SentenceTransformer embeddings (all-mpnet-base-v2), L2-normalized float32 vectors.
  Exact IndexFlatIP (inner product on normalized vectors equals cosine similarity).
  Exact flat search is exhaustive and avoids recall loss for small knowledge bases.
- Stale verification: SHA-256 fingerprint of chunks.jsonl stored in index_meta.json.
"""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi

from ids_rag.config import get_project_root, load_config

# ---------------------------------------------------------------------------
# Regex tokenizer for BM25
# ---------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[.\-][a-zA-Z0-9]+)*")


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, keep alphanumeric words and internal hyphens/dots.

    Keeps terms like 't1498.001' and '800-61' whole. No stemming, no stopword removal.
    """
    return _TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

def compute_file_sha256(filepath: Path | str) -> str:
    """Compute hex SHA-256 digest of file contents."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Knowledge Base Index Wrapper
# ---------------------------------------------------------------------------

class KBIndex:
    """Unified container for sparse BM25 and dense FAISS knowledge base indexes."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        bm25: BM25Okapi,
        faiss_index: faiss.Index,
        embeddings: np.ndarray,
        meta: dict[str, Any],
        dense_model: Any = None,
        encoder_fn: Callable[[list[str]], np.ndarray] | None = None,
        device: str = "cpu",
    ) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.embeddings = embeddings
        self.meta = meta
        self.dense_model = dense_model
        self.encoder_fn = encoder_fn
        self.device = device
        self.chunk_id_to_pos = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    def search_bm25(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return top-k (position, score) pairs from sparse BM25 search."""
        tokens = tokenize_bm25(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def search_dense(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return top-k (position, score) pairs from dense FAISS search."""
        if self.encoder_fn is not None:
            q_emb = self.encoder_fn([query])
        elif self.dense_model is not None:
            q_emb = self.dense_model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self.device,
            )
        else:
            raise RuntimeError("KBIndex has neither dense_model nor encoder_fn loaded.")

        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
        scores, indices = self.faiss_index.search(q_emb, k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

    def bm25_scores(self, query: str) -> np.ndarray:
        """Return raw BM25 score array for every chunk in the index."""
        tokens = tokenize_bm25(query)
        return np.array(self.bm25.get_scores(tokens), dtype=np.float32)

    def dense_scores(self, query: str) -> np.ndarray:
        """Return cosine similarity score array for every chunk in the index."""
        if self.encoder_fn is not None:
            q_emb = self.encoder_fn([query])
        elif self.dense_model is not None:
            q_emb = self.dense_model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self.device,
            )
        else:
            raise RuntimeError("KBIndex has neither dense_model nor encoder_fn loaded.")

        q_vec = np.ascontiguousarray(q_emb, dtype=np.float32).reshape(-1)
        return np.dot(self.embeddings, q_vec).astype(np.float32)

    def get_chunk(self, position: int) -> dict[str, Any]:
        """Return chunk metadata dictionary for the given index position."""
        return self.chunks[position]


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(
    cfg: dict[str, Any] | None = None,
    chunks_path: Path | str | None = None,
    index_dir: Path | str | None = None,
    encoder_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> Path:
    """Build sparse BM25 and dense FAISS indexes from chunks.jsonl and persist to index_dir."""
    if cfg is None:
        cfg = load_config()

    root = get_project_root()
    rag_cfg = cfg.get("rag", {})

    if chunks_path is None:
        chunks_path = root / cfg.get("paths", {}).get("kb", "data/kb") / "chunks.jsonl"
    else:
        chunks_path = Path(chunks_path)

    if index_dir is None:
        rel_idx = rag_cfg.get("index_dir", "data/kb/index")
        index_dir = root / rel_idx
    else:
        index_dir = Path(index_dir)

    index_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")

    # Set random seeds for determinism
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load chunks preserving order and chunk_id
    chunks: list[dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                chunks.append(json.loads(line_str))

    if not chunks:
        raise ValueError(f"No chunks loaded from {chunks_path}")

    chunks_sha256 = compute_file_sha256(chunks_path)
    chunk_texts = [c["text"] for c in chunks]

    # 2. Build and persist BM25 index
    bm25_k1 = float(rag_cfg.get("bm25_k1", 1.5))
    bm25_b = float(rag_cfg.get("bm25_b", 0.75))

    tokenized_corpus = [tokenize_bm25(t) for t in chunk_texts]
    with open(index_dir / "tokenized_corpus.json", "w", encoding="utf-8") as f:
        json.dump(tokenized_corpus, f, ensure_ascii=False)

    # 3. Dense FAISS index
    embed_model_name = rag_cfg.get("embed_model", "sentence-transformers/all-mpnet-base-v2")
    embed_batch_size = int(rag_cfg.get("embed_batch_size", 32))
    embed_device_cfg = rag_cfg.get("embed_device", "auto")

    if embed_device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = embed_device_cfg

    t0 = time.perf_counter()
    if encoder_fn is not None:
        embeddings = encoder_fn(chunk_texts)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(embed_model_name, device=device)
        embeddings = model.encode(
            chunk_texts,
            batch_size=embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=device,
        )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    encode_time = time.perf_counter() - t0

    # Exact search via IndexFlatIP (inner product on normalized embeddings equals cosine similarity).
    # The reference paper mentions approximate search, but for a knowledge base of a few hundred
    # vectors (e.g. 217 chunks), exact flat inner-product search is completely exhaustive,
    # ultra-fast (< 1 ms), and avoids recall loss.
    d = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))
    np.save(str(index_dir / "embeddings.npy"), embeddings)

    # 4. Save metadata
    meta = {
        "embed_model": embed_model_name if encoder_fn is None else "custom_encoder",
        "embedding_dim": int(d),
        "chunk_count": len(chunks),
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "chunks_sha256": chunks_sha256,
        "bm25_params": {"k1": bm25_k1, "b": bm25_b},
        "build_time": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "encode_time_sec": round(encode_time, 3),
        "seed": seed,
    }

    with open(index_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return index_dir


# ---------------------------------------------------------------------------
# Index loader with stale check
# ---------------------------------------------------------------------------

def load_index(
    cfg: dict[str, Any] | None = None,
    index_dir: Path | str | None = None,
    chunks_path: Path | str | None = None,
    encoder_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> KBIndex:
    """Load persisted BM25 and FAISS indexes and verify against chunks.jsonl SHA-256."""
    if cfg is None:
        cfg = load_config()

    root = get_project_root()
    rag_cfg = cfg.get("rag", {})

    if index_dir is None:
        rel_idx = rag_cfg.get("index_dir", "data/kb/index")
        index_dir = root / rel_idx
    else:
        index_dir = Path(index_dir)

    if chunks_path is None:
        chunks_path = root / cfg.get("paths", {}).get("kb", "data/kb") / "chunks.jsonl"
    else:
        chunks_path = Path(chunks_path)

    meta_file = index_dir / "index_meta.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Index metadata file not found at: {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Stale verification: current SHA-256 of chunks.jsonl must match saved SHA-256
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file missing for verification at: {chunks_path}")

    current_sha256 = compute_file_sha256(chunks_path)
    saved_sha256 = meta.get("chunks_sha256")
    if current_sha256 != saved_sha256:
        raise RuntimeError(
            f"Knowledge base index is stale: SHA-256 of '{chunks_path}' ({current_sha256}) "
            f"does not match saved index SHA-256 ({saved_sha256}). "
            f"Rebuild index with `python -m ids_rag.rag.index --build`."
        )

    # 1. Load chunks
    chunks: list[dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                chunks.append(json.loads(line_str))

    # 2. Rebuild BM25 from saved tokenized corpus
    tok_corpus_file = index_dir / "tokenized_corpus.json"
    if not tok_corpus_file.is_file():
        raise FileNotFoundError(f"Tokenized corpus not found at: {tok_corpus_file}")

    with open(tok_corpus_file, "r", encoding="utf-8") as f:
        tokenized_corpus = json.load(f)

    bm25_params = meta.get("bm25_params", {"k1": 1.5, "b": 0.75})
    bm25 = BM25Okapi(
        tokenized_corpus,
        k1=float(bm25_params["k1"]),
        b=float(bm25_params["b"]),
    )

    # 3. Load FAISS index and embeddings
    faiss_file = index_dir / "faiss.index"
    emb_file = index_dir / "embeddings.npy"

    if not faiss_file.is_file() or not emb_file.is_file():
        raise FileNotFoundError(f"FAISS files missing in: {index_dir}")

    faiss_index = faiss.read_index(str(faiss_file))
    embeddings = np.load(str(emb_file))

    # 4. Dense model
    embed_device_cfg = rag_cfg.get("embed_device", "auto")
    if embed_device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = embed_device_cfg

    dense_model = None
    if encoder_fn is None:
        from sentence_transformers import SentenceTransformer

        embed_model_name = meta.get(
            "embed_model",
            rag_cfg.get("embed_model", "sentence-transformers/all-mpnet-base-v2"),
        )
        dense_model = SentenceTransformer(embed_model_name, device=device)

    return KBIndex(
        chunks=chunks,
        bm25=bm25,
        faiss_index=faiss_index,
        embeddings=embeddings,
        meta=meta,
        dense_model=dense_model,
        encoder_fn=encoder_fn,
        device=device,
    )


# ---------------------------------------------------------------------------
# CLI Command handler
# ---------------------------------------------------------------------------

_CHECK_QUERIES: tuple[str, ...] = (
    "denial of service flood attack",
    "T1498 network denial of service",
    "SYN flood mitigation",
    "how to prioritize an incident",
    "distributed attack from many sources",
)


def _cli_build(cfg: dict[str, Any]) -> None:
    """Execute --build CLI action."""
    print("Building knowledge base indexes...")
    index_dir = build_index(cfg)
    meta_path = index_dir / "index_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    faiss_sz = (index_dir / "faiss.index").stat().st_size
    emb_sz = (index_dir / "embeddings.npy").stat().st_size
    tok_sz = (index_dir / "tokenized_corpus.json").stat().st_size
    meta_sz = meta_path.stat().st_size

    print(f"Chunks indexed      : {meta['chunk_count']}")
    print(f"Embedding dimension : {meta['embedding_dim']}")
    print(f"Device              : {meta.get('device', 'cpu')}")
    print(f"Encode time         : {meta.get('encode_time_sec', 0.0):.2f} s")
    print("Index file sizes    :")
    print(f"  faiss.index           : {faiss_sz:>8} bytes ({faiss_sz / 1024:.1f} KB)")
    print(f"  embeddings.npy        : {emb_sz:>8} bytes ({emb_sz / 1024:.1f} KB)")
    print(f"  tokenized_corpus.json : {tok_sz:>8} bytes ({tok_sz / 1024:.1f} KB)")
    print(f"  index_meta.json       : {meta_sz:>8} bytes ({meta_sz / 1024:.1f} KB)")
    print(f"Index successfully written to: {index_dir}")


def _cli_check(cfg: dict[str, Any]) -> None:
    """Execute --check CLI action on the 5 required benchmark queries."""
    print("Loading index for verification...")
    kb = load_index(cfg)
    print(f"Index loaded successfully: {len(kb.chunks)} chunks, model={kb.meta.get('embed_model')}")

    sep_wide = "=" * 90
    sep_sub = "-" * 90

    for q_idx, query in enumerate(_CHECK_QUERIES, 1):
        print(f"\n{sep_wide}")
        print(f"Query [{q_idx}/5]: {query!r}")
        print(sep_sub)

        bm25_res = kb.search_bm25(query, k=3)
        dense_res = kb.search_dense(query, k=3)

        for rank in range(3):
            b_pos, b_score = bm25_res[rank]
            d_pos, d_score = dense_res[rank]
            b_chunk = kb.get_chunk(b_pos)
            d_chunk = kb.get_chunk(d_pos)

            b_snippet = b_chunk["text"][:120].replace("\n", " ").strip()
            d_snippet = d_chunk["text"][:120].replace("\n", " ").strip()

            print(f"Rank {rank + 1}:")
            print(f"  BM25  | Score: {b_score:7.4f} | Src: {b_chunk['source_name']:<24} | Sec: {b_chunk['section']}")
            print(f"        | Text: {b_snippet}...")
            print(f"  Dense | Score: {d_score:7.4f} | Src: {d_chunk['source_name']:<24} | Sec: {d_chunk['section']}")
            print(f"        | Text: {d_snippet}...")


def main() -> None:
    """Entry point for CLI execution."""
    parser = argparse.ArgumentParser(description="Build and check dense and sparse indexes.")
    parser.add_argument("--build", action="store_true", help="Build and persist indexes from chunks.jsonl")
    parser.add_argument("--check", action="store_true", help="Run 5 diagnostic queries comparing BM25 vs Dense")
    args = parser.parse_args()

    cfg = load_config()
    if args.build:
        _cli_build(cfg)
    elif args.check:
        _cli_check(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
