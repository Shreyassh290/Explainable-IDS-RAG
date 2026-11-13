"""Build the v2 extra-class knowledge base chunks from raw text files.

Reads plain-text documents from data/v2/kb/raw/ and produces
data/v2/kb/chunks_v2.jsonl using the same chunk schema as the
frozen data/kb/chunks.jsonl.

Schema per chunk:
  chunk_id      -- UUID4 string (deterministic via seed)
  source_name   -- filename (e.g. MITRE_T1110.txt)
  source_type   -- "MITRE" or "NIST"
  source_ref    -- technique ID (e.g. "T1110")
  section       -- heading label for this chunk's start
  sections      -- list of all heading labels in this chunk
  token_count   -- approximate count using whitespace tokenization
  char_start    -- character offset in source text
  char_end      -- character offset in source text
  text          -- chunk text

Chunking rules (simplified from chunking.py for plain-text input):
  - Split document on blank lines into paragraphs.
  - Lines that match heading patterns (all-caps or Title with ID/Mitre keyword)
    start a new section.
  - Accumulate paragraphs until token count exceeds chunk_max_tokens (350).
  - Minimum chunk size: chunk_min_tokens (30 tokens). Discard smaller fragments.
  - Overlap: carry the last overlap_tokens (200) tokens of a chunk into the next.

Output: data/v2/kb/chunks_v2.jsonl
Never modifies data/kb/chunks.jsonl or any file in data/kb/.

Usage:
    uv run python scripts/v2_build_kb_extra.py
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path

from ids_rag.config import get_project_root


# ---------------------------------------------------------------------------
# Configuration (matches config.yaml rag section)
# ---------------------------------------------------------------------------

CHUNK_MAX_TOKENS = 350
CHUNK_MIN_TOKENS = 30
OVERLAP_TOKENS = 200

# Source metadata derived from filename
_SOURCE_META: dict[str, dict[str, str]] = {
    "MITRE_T1110.txt":     {"source_type": "MITRE", "source_ref": "T1110"},
    "MITRE_T1110_001.txt": {"source_type": "MITRE", "source_ref": "T1110.001"},
    "MITRE_T1110_003.txt": {"source_type": "MITRE", "source_ref": "T1110.003"},
    "MITRE_T1583_002.txt": {"source_type": "MITRE", "source_ref": "T1583.002"},
    "MITRE_T1071.txt":     {"source_type": "MITRE", "source_ref": "T1071"},
}

# Heading patterns (lines that begin a new section)
_HEADING_RE = re.compile(
    r"^("
    r"Description|Mitigations|Detection Strategy|Procedure Examples"
    r"|ID:\s*M\d+|ID:\s*T\d+"
    r"|MITRE ATT&CK"
    r")[\s:—]?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Approximate token count (whitespace tokenization)
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Count whitespace-separated words as a proxy for tokens."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Deterministic UUID from content hash
# ---------------------------------------------------------------------------

def content_uuid(source_name: str, section: str, char_start: int) -> str:
    """Generate a deterministic UUID based on source, section, and position."""
    seed = f"{source_name}|{section}|{char_start}"
    h = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return str(uuid.UUID(h))


# ---------------------------------------------------------------------------
# Parse a plain-text KB file into (section, text) paragraph blocks
# ---------------------------------------------------------------------------

def parse_txt_document(raw_text: str) -> list[dict]:
    """Parse plain-text document into labelled paragraph blocks.

    Returns list of dicts: {section: str, text: str, char_start: int, char_end: int}
    """
    lines = raw_text.splitlines()
    current_section = "Description"
    blocks: list[dict] = []
    para_lines: list[str] = []
    pos = 0
    block_start = 0

    def flush_para(end_pos: int) -> None:
        nonlocal para_lines, block_start
        joined = " ".join(para_lines).strip()
        joined = re.sub(r"\s+", " ", joined)
        if joined:
            blocks.append({
                "section": current_section,
                "text": joined,
                "char_start": block_start,
                "char_end": end_pos,
            })
        para_lines = []
        block_start = end_pos

    for line in lines:
        line_end = pos + len(line) + 1  # +1 for newline

        stripped = line.strip()

        if not stripped:
            # Blank line: flush current paragraph
            flush_para(pos)
            pos = line_end
            block_start = pos
            continue

        # Check if this line starts a new section
        if _HEADING_RE.match(stripped):
            flush_para(pos)
            current_section = stripped.rstrip(":").strip()
            pos = line_end
            block_start = pos
            continue

        para_lines.append(stripped)
        pos = line_end

    flush_para(pos)
    return blocks


# ---------------------------------------------------------------------------
# Chunk a list of paragraph blocks with overlap
# ---------------------------------------------------------------------------

def chunk_blocks(
    blocks: list[dict],
    source_name: str,
    source_type: str,
    source_ref: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict]:
    """Merge paragraph blocks into chunks, respecting token limits."""
    chunks: list[dict] = []
    current_texts: list[str] = []
    current_sections: list[str] = []
    current_token_count = 0
    current_char_start = blocks[0]["char_start"] if blocks else 0

    def flush_chunk(char_end: int) -> None:
        nonlocal current_texts, current_sections, current_token_count, current_char_start
        full_text = " ".join(current_texts).strip()
        full_text = re.sub(r"\s+", " ", full_text)
        if approx_tokens(full_text) < min_tokens:
            return
        section_label = current_sections[0] if current_sections else "Description"
        chunk = {
            "chunk_id": content_uuid(source_name, section_label, current_char_start),
            "source_name": source_name,
            "source_type": source_type,
            "source_ref": source_ref,
            "section": section_label,
            "sections": list(dict.fromkeys(current_sections)),
            "token_count": approx_tokens(full_text),
            "char_start": current_char_start,
            "char_end": char_end,
            "text": full_text,
        }
        chunks.append(chunk)

    for block in blocks:
        block_tokens = approx_tokens(block["text"])
        section = block["section"]

        if current_token_count + block_tokens > max_tokens and current_texts:
            flush_chunk(block["char_start"])

            # Carry overlap: keep last overlap_tokens tokens from previous chunk
            overlap_words: list[str] = []
            for prev_text in reversed(current_texts):
                words = prev_text.split()
                needed = overlap_tokens - len(overlap_words)
                if needed <= 0:
                    break
                overlap_words = words[-needed:] + overlap_words

            current_texts = [" ".join(overlap_words)] if overlap_words else []
            current_sections = [section]
            current_token_count = len(overlap_words)
            current_char_start = block["char_start"]

        current_texts.append(block["text"])
        current_sections.append(section)
        current_token_count += block_tokens

    if current_texts:
        last_char_end = blocks[-1]["char_end"] if blocks else 0
        flush_chunk(last_char_end)

    return chunks


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_v2_kb(
    raw_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> int:
    """Read raw text files and write chunks_v2.jsonl. Returns chunk count."""
    root = get_project_root()

    if raw_dir is None:
        raw_dir = root / "data" / "v2" / "kb" / "raw"
    else:
        raw_dir = Path(raw_dir)

    if output_path is None:
        output_path = root / "data" / "v2" / "kb" / "chunks_v2.jsonl"
    else:
        output_path = Path(output_path)

    # Safety: never write to data/kb/
    frozen_kb = (root / "data" / "kb").resolve()
    if output_path.resolve() == frozen_kb or frozen_kb in output_path.resolve().parents:
        raise ValueError(
            f"SAFETY REFUSAL: Output path '{output_path}' is inside frozen data/kb/. "
            f"v2 KB must be written to data/v2/kb/."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}.")

    print(f"Building v2 KB from {len(txt_files)} source files in {raw_dir}")
    print(f"Output: {output_path}\n")

    for txt_path in txt_files:
        filename = txt_path.name
        meta = _SOURCE_META.get(filename)
        if meta is None:
            print(f"  [SKIP] Unknown file '{filename}' — not in _SOURCE_META. Add it first.")
            continue

        raw_text = txt_path.read_text(encoding="utf-8")
        blocks = parse_txt_document(raw_text)
        file_chunks = chunk_blocks(
            blocks,
            source_name=filename,
            source_type=meta["source_type"],
            source_ref=meta["source_ref"],
        )
        print(f"  {filename}: {len(blocks)} blocks -> {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    # Write chunks_v2.jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nTotal v2 chunks written: {len(all_chunks)}")
    print(f"Output: {output_path}")
    return len(all_chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_v2_kb()

