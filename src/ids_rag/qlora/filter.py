"""Rejection sampling filters and dataset formatting for QLoRA fine-tuning.

Filter Criteria:
1. Exactly five required numbered section headings:
   - 1. Rationale
   - 2. Key Indicators
   - 3. Confidence Assessment
   - 4. Threat Assessment
   - 5. Recommendations
2. Word count strictly between 300 and 600 words.
3. Citation numbers valid: in range [1, N_sources], no placeholder '[n]', no ranges '[1-3]'.
4. No copied source-header lines (e.g. '[1] NIST SP...').
5. No unsupported claims about botnets, multiple hosts, flooding or saturation.
6. Inline citation word overlap not below citation_check.overlap_threshold in config.yaml.

Outputs:
- Chat-format JSONL with system, user, assistant messages.
- 90% train split (dataset_train.jsonl) and 10% val split (dataset_val.jsonl).
- Token length statistics and flagging of examples > 1,800 tokens.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from ids_rag.config import get_project_root, load_config
from ids_rag.eval.citation_check import drop_header_lines, extract_flow
from ids_rag.llm.generate import validate_report

# Regex for copied source headers
COPIED_HDR_RE = re.compile(r"^[#*_>\-\s]*\[(\d+)\]\s+(?:NIST|MITRE)\b", re.IGNORECASE)

# Required headings and exact numbers
REQUIRED_HEADINGS = [
    (1, "Rationale"),
    (2, "Key Indicators"),
    (3, "Confidence Assessment"),
    (4, "Threat Assessment"),
    (5, "Recommendations"),
]

# Patterns for claims that must not be asserted without direct evidentiary basis
CLAIM_PATTERNS = {
    "botnet": re.compile(r"\b(?:botnets?|bot-net|zombie hosts?|command-and-control bot)\b", re.IGNORECASE),
    "multiple_hosts": re.compile(r"\b(?:multiple (?:attacking |compromised |distinct )?hosts?|distributed (?:agents?|attackers?|sources?|nodes?))\b", re.IGNORECASE),
    "flooding": re.compile(r"\b(?:volumetric flood(?:ing)?|flood(?:ing)? the (?:network|target|server|link)|flood of (?:packets|traffic|requests))\b", re.IGNORECASE),
    "saturation": re.compile(r"\b(?:(?:network|link|bandwidth|buffer) saturation|saturat(?:ing|ed) the (?:network|bandwidth|link))\b", re.IGNORECASE),
}

# Negation prefixes that indicate legitimate grounding (stating that the flow DOES NOT exhibit these)
NEGATION_RE = re.compile(
    r"\b(?:no|not|does not|do not|never|without|lacks?|neither|unlikely|cannot|no evidence of|does not exhibit|no indication of)\b.{0,60}",
    re.IGNORECASE,
)


def has_unsupported_claims(report_text: str, explanation: dict[str, Any] | None = None) -> list[str]:
    """Detect ungrounded assertions of botnets, multiple hosts, flooding, or saturation.

    Returns list of violated claim categories, empty if report is compliant.
    """
    violations: list[str] = []

    # Check sentences individually to discern positive assertions from negated qualifiers
    sentences = re.split(r"(?<=[.!?])\s+", report_text)
    for sent in sentences:
        s_clean = sent.strip()
        if not s_clean:
            continue

        for claim_type, pattern in CLAIM_PATTERNS.items():
            if pattern.search(s_clean):
                # Check if this sentence contains an explicit negation
                # E.g. "does not exhibit volumetric flooding", "no botnet involvement is observed"
                is_negated = False
                for match in pattern.finditer(s_clean):
                    start_idx = match.start()
                    preceding = s_clean[max(0, start_idx - 60):start_idx]
                    if NEGATION_RE.search(preceding):
                        is_negated = True
                        break

                if not is_negated:
                    if claim_type not in violations:
                        violations.append(claim_type)

    return violations


def evaluate_report_filters(
    report_text: str,
    retrieved_chunks: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    overlap_threshold: float = 0.5,
    min_words: int = 300,
    max_words: int = 600,
    explanation: dict[str, Any] | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Evaluate all filter criteria on a single generated report.

    Returns:
        (is_accepted, list_of_rejection_reasons, details_dict)
    """
    reasons: list[str] = []
    details: dict[str, Any] = {}

    words = report_text.split()
    word_count = len(words)
    details["word_count"] = word_count

    # 1. Check all five numbered headings
    missing_headings: list[str] = []
    for num, name in REQUIRED_HEADINGS:
        pattern = rf"^\W*{num}\.\s*{re.escape(name)}"
        if not re.search(pattern, report_text, re.IGNORECASE | re.MULTILINE):
            missing_headings.append(f"{num}. {name}")

    if missing_headings:
        reasons.append(f"missing_headings: {', '.join(missing_headings)}")
    details["missing_headings"] = missing_headings

    # 2. Check word count limits [300, 600]
    if word_count < min_words:
        reasons.append(f"word_count_too_low ({word_count} < {min_words})")
    elif word_count > max_words:
        reasons.append(f"word_count_too_high ({word_count} > {max_words})")

    # 3. Check copied source headers
    copied_headers: int = 0
    for line in report_text.splitlines():
        if COPIED_HDR_RE.match(line):
            copied_headers += 1
    if copied_headers > 0:
        reasons.append(f"copied_source_headers ({copied_headers} lines)")
    details["copied_source_headers"] = copied_headers

    # 4. Check for citation ranges and placeholder citations
    if re.search(r"\[\s*\d+\s*[-\u2013]\s*\d+\s*\]", report_text):
        reasons.append("citation_ranges_present")
    if "[n]" in report_text.lower():
        reasons.append("placeholder_citations_present")

    # 5. Extract and validate inline citation numbers
    n_sources = len(retrieved_chunks)
    cleaned_text, _ = drop_header_lines(report_text)
    citation_rows = extract_flow(cleaned_text, retrieved_chunks, chunks_by_id, overlap_threshold)
    details["citation_count"] = len(citation_rows)

    if len(citation_rows) == 0:
        reasons.append("no_citations_present")
    else:
        invalid_nums: list[int] = []
        low_overlap_cites: list[dict[str, Any]] = []

        for row in citation_rows:
            n = row.get("n", 0)
            if n < 1 or n > n_sources or row.get("out_of_range", False):
                invalid_nums.append(n)
            elif row.get("overlap") is None or row.get("overlap", 0.0) < overlap_threshold:
                low_overlap_cites.append({
                    "n": n,
                    "overlap": row.get("overlap"),
                    "sentence": row.get("sentence", "")[:80],
                })

        if invalid_nums:
            reasons.append(f"invalid_citation_numbers: {sorted(set(invalid_nums))}")
        if low_overlap_cites:
            reasons.append(
                f"citation_overlap_below_threshold: {len(low_overlap_cites)} citations < {overlap_threshold}"
            )
        details["invalid_citation_numbers"] = invalid_nums
        details["low_overlap_citations"] = low_overlap_cites

    # 6. Check unsupported claims (botnets, multiple hosts, flooding, saturation)
    claim_violations = has_unsupported_claims(report_text, explanation)
    if claim_violations:
        reasons.append(f"unsupported_claims: {', '.join(claim_violations)}")
    details["unsupported_claims"] = claim_violations

    is_accepted = (len(reasons) == 0)
    return is_accepted, reasons, details


def estimate_token_length(text: str, tokenizer: Any = None) -> int:
    """Return token count using Llama 3 tokenizer if available, otherwise character estimate."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    # Standard conservative estimate: ~3.7 characters per token for English technical prose
    return int(math.ceil(len(text) / 3.7))


def load_tokenizer_if_available() -> Any:
    """Attempt to load Llama 3 tokenizer if available locally or in cache."""
    try:
        from transformers import AutoTokenizer  # type: ignore
        for name in [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B",
            "NousResearch/Meta-Llama-3-8B-Instruct",
        ]:
            try:
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
                return tok
            except Exception:
                continue
    except Exception:
        pass
    return None


def split_and_save_dataset(
    kept_records: list[dict[str, Any]],
    out_dir: Path,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Split kept records into train (90%) and val (10%) JSONL files in chat format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Stratify by class if available
    class_buckets: dict[str, list[dict[str, Any]]] = {}
    for rec in kept_records:
        cls_name = rec.get("class", rec.get("true_class", "UNKNOWN"))
        class_buckets.setdefault(cls_name, []).append(rec)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []

    for cls_name, items in sorted(class_buckets.items()):
        indices = np.arange(len(items))
        rng.shuffle(indices)

        n_val = max(1, int(round(len(items) * val_fraction))) if len(items) >= 5 else 0
        val_idx_set = set(indices[:n_val])

        for i, item in enumerate(items):
            chat_item = {
                "messages": [
                    {"role": "system", "content": item["system_text"]},
                    {"role": "user", "content": item["user_text"]},
                    {"role": "assistant", "content": item["report_text"]},
                ],
                "flow_index": item.get("flow_index"),
                "class": cls_name,
            }
            if i in val_idx_set:
                val_records.append(chat_item)
            else:
                train_records.append(chat_item)

    train_file = out_dir / "dataset_train.jsonl"
    val_file = out_dir / "dataset_val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return train_file, val_file

