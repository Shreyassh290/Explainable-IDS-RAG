"""Unit tests for ids_rag.rag.chunking – synthetic text only, no real KB files."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from ids_rag.rag.chunking import (
    _FOOTNOTE_RE,
    _build_chunks,
    _derive_source_ref,
    _derive_source_type,
    _extract_toc_sp800_30,
    _extract_toc_sp800_94,
    _load_text,
    _parse_html,
    clean_mitre_segments,
    count_tokens,
    detect_heading,
    is_dot_leader_line,
    is_parent_and_first_child,
    is_sp800_30_header,
    is_sp800_30_page_label,
    is_sp800_94_header,
    is_sp800_94_page_label,
    normalize_heading_key,
    postprocess_short_chunks,
    remove_footnotes,
    repair_drop_caps,
    title_case_heading,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_DOC_ARGS: dict[str, Any] = {
    "source_name": "TEST_DOC.txt",
    "source_type": "OTHER",
    "source_ref": "TEST_DOC",
    "min_tokens": 1,          # allow tiny chunks in tests
    "max_tokens": 50,
    "overlap_tokens": 10,
}


def _make_text_segments(paragraphs: list[str]) -> list[dict[str, Any]]:
    """Build a list of non-heading segments from plain strings."""
    return [{"is_heading": False, "heading_level": 0, "text": p} for p in paragraphs]


def _make_heading_segment(heading: str, level: int = 2) -> dict[str, Any]:
    return {"is_heading": True, "heading_level": level, "text": heading}


# ---------------------------------------------------------------------------
# Test: no chunk exceeds chunk_max_tokens
# ---------------------------------------------------------------------------

def test_no_chunk_exceeds_max_tokens() -> None:
    """Every emitted chunk must have token_count <= chunk_max_tokens."""
    sentences = [
        "Network packet analysis reveals high-volume flooding consistent with DDoS attacks.",
        "Flow inter-arrival time statistics are key discriminators of benign from attack traffic.",
        "NIST SP 800-61 recommends a structured incident response lifecycle.",
        "Mitigation strategies include rate-limiting and filtering upstream traffic.",
        "Detection thresholds must be calibrated against known benign baselines.",
        "Attack signatures extracted from MITRE ATT&CK inform real-time alerting rules.",
        "Packet length variance separates UDP flooding from slow-rate HTTP attacks.",
        "Window size initialization differs between OS stacks and can fingerprint attackers.",
    ] * 5

    segments = _make_text_segments(sentences)
    max_toks = 50
    chunks = _build_chunks(
        segments=segments,
        source_name="TEST.txt",
        source_type="OTHER",
        source_ref="TEST",
        min_tokens=1,
        max_tokens=max_toks,
        overlap_tokens=10,
    )

    assert len(chunks) > 0, "Expected at least one chunk"
    for c in chunks:
        assert c["token_count"] <= max_toks, (
            f"Chunk token_count={c['token_count']} exceeds max_tokens={max_toks}: "
            f"{c['text'][:80]}"
        )


# ---------------------------------------------------------------------------
# Test: consecutive chunks share the configured overlap
# ---------------------------------------------------------------------------

def test_consecutive_chunks_share_overlap() -> None:
    """When a section produces multiple chunks, consecutive chunks share overlap tokens."""
    sent = "The intrusion detection system flagged multiple suspicious flows."
    n_sent = 20
    segments = _make_text_segments([sent] * n_sent)

    max_toks = 30
    overlap_toks = 10

    chunks = _build_chunks(
        segments=segments,
        source_name="OVERLAP_TEST.txt",
        source_type="OTHER",
        source_ref="OVERLAP_TEST",
        min_tokens=1,
        max_tokens=max_toks,
        overlap_tokens=overlap_toks,
    )

    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

    for i in range(len(chunks) - 1):
        c_prev = chunks[i]
        c_next = chunks[i + 1]
        if c_prev["section"] != c_next["section"]:
            continue
        prev_words = c_prev["text"].split()
        next_words_set = set(c_next["text"].split())
        overlap_found = any(w in next_words_set for w in prev_words[-5:])
        assert overlap_found, (
            f"No overlap detected between chunk {i} and chunk {i+1}.\n"
            f"  chunk {i} end: {prev_words[-5:]}\n"
            f"  chunk {i+1} start: {c_next['text'][:80]}"
        )


# ---------------------------------------------------------------------------
# Test: headings are captured as the section field
# ---------------------------------------------------------------------------

def test_headings_captured_as_section() -> None:
    """Chunks following a heading segment must have section == that heading text."""
    segments = [
        _make_heading_segment("Detection Strategy", level=2),
        _make_text_segments(
            ["Detecting denial-of-service requires packet rate thresholds."]
        )[0],
        _make_text_segments(
            ["Flow duration and packet count features are most informative."]
        )[0],
        _make_heading_segment("Mitigations", level=2),
        _make_text_segments(
            ["Rate limiting upstream is an effective mitigation measure."]
        )[0],
    ]

    chunks = _build_chunks(
        segments=segments,
        source_name="SECTION_TEST.html",
        source_type="MITRE",
        source_ref="T1498",
        min_tokens=1,
        max_tokens=200,
        overlap_tokens=0,
    )

    sections = {c["section"] for c in chunks}
    assert "Detection Strategy" in sections, (
        f"Expected 'Detection Strategy' in sections, got {sections}"
    )
    assert "Mitigations" in sections, (
        f"Expected 'Mitigations' in sections, got {sections}"
    )

    for c in chunks:
        if "rate limiting" in c["text"].lower():
            assert c["section"] == "Mitigations", (
                f"Expected section 'Mitigations', got {c['section']!r}"
            )


# ---------------------------------------------------------------------------
# Test: HTML script and nav text is excluded
# ---------------------------------------------------------------------------

def test_html_script_and_nav_excluded() -> None:
    """Text inside <script>, <nav>, and <style> must not appear in parsed segments."""
    html = """<!DOCTYPE html>
<html>
<head>
  <script>var secret = 'do not extract';</script>
  <style>body { color: red; } /* css only */</style>
</head>
<body>
  <nav>
    <a href="/home">Home</a>
    <a href="/about">About</a>
  </nav>
  <main>
    <h2>Detection Strategy</h2>
    <p>High packet rates indicate potential flooding attacks.</p>
    <p>Monitor flow byte rates for anomalous spikes.</p>
  </main>
  <footer>Copyright 2024</footer>
</body>
</html>"""

    segments = _parse_html(html)
    all_text = " ".join(s["text"] for s in segments)

    assert "do not extract" not in all_text, "Script content leaked into segments"
    assert "css only" not in all_text, "Style content leaked into segments"
    assert "Copyright 2024" not in all_text, "Footer content leaked into segments"
    assert "High packet rates" in all_text, "Main content missing from segments"

    heading_texts = [s["text"] for s in segments if s["is_heading"]]
    assert any("Detection Strategy" in h for h in heading_texts), (
        f"Expected heading 'Detection Strategy', got {heading_texts}"
    )


# ---------------------------------------------------------------------------
# Test: nav text excluded specifically
# ---------------------------------------------------------------------------

def test_html_nav_text_excluded() -> None:
    """Text inside <nav> elements must not appear in any segment."""
    html = """<html><body>
    <nav class="main-nav">
      <ul><li>Matrices</li><li>Techniques</li><li>Groups</li></ul>
    </nav>
    <article>
      <h1>Network Denial of Service</h1>
      <p>Adversaries may perform Network Denial of Service to degrade availability.</p>
    </article>
    </body></html>"""

    segments = _parse_html(html)
    all_text = " ".join(s["text"] for s in segments)

    assert "Matrices" not in all_text, "Nav item 'Matrices' leaked into segments"
    assert "Techniques" not in all_text, "Nav item 'Techniques' leaked into segments"
    assert "Network Denial of Service" in " ".join(
        s["text"] for s in segments if s["is_heading"]
    ), "Article heading missing"
    assert "degrade availability" in all_text, "Article body content missing"


# ---------------------------------------------------------------------------
# Test: source type and source_ref derivation
# ---------------------------------------------------------------------------

def test_derive_source_type() -> None:
    """File name prefix determines source type."""
    assert _derive_source_type("MITRE_T1498.html") == "MITRE"
    assert _derive_source_type("NIST_SP_800-61_Rev2.pdf") == "NIST"
    assert _derive_source_type("other_doc.txt") == "OTHER"


def test_derive_source_ref_mitre() -> None:
    """MITRE source ref extracts the technique ID."""
    assert _derive_source_ref("MITRE_T1498.html", "MITRE") == "T1498"
    assert _derive_source_ref("MITRE_T1498_001.html", "MITRE") == "T1498.001"
    assert _derive_source_ref("MITRE_T1499.html", "MITRE") == "T1499"


def test_derive_source_ref_nist() -> None:
    """NIST source ref extracts the SP number."""
    ref = _derive_source_ref("NIST_SP_800-61_Rev2.pdf", "NIST")
    assert "800" in ref, f"Expected SP number in ref, got {ref!r}"


# ---------------------------------------------------------------------------
# Test: chunk schema completeness
# ---------------------------------------------------------------------------

def test_chunk_schema_has_required_fields() -> None:
    """Every chunk must carry all required metadata fields."""
    required_fields = {
        "chunk_id",
        "source_name",
        "source_type",
        "source_ref",
        "section",
        "sections",
        "text",
        "token_count",
        "char_start",
        "char_end",
    }
    segments = _make_text_segments(
        ["Flood attack generates extremely high packet rates per second."]
    )
    chunks = _build_chunks(
        segments=segments,
        source_name="SCHEMA_TEST.txt",
        source_type="OTHER",
        source_ref="SCHEMA_TEST",
        min_tokens=1,
        max_tokens=100,
        overlap_tokens=0,
    )
    assert len(chunks) >= 1
    for c in chunks:
        missing = required_fields - set(c.keys())
        assert not missing, f"Chunk missing fields: {missing}"
        assert isinstance(c["chunk_id"], str) and len(c["chunk_id"]) > 0
        assert isinstance(c["token_count"], int) and c["token_count"] > 0
        assert isinstance(c["text"], str) and len(c["text"]) > 0
        assert isinstance(c["sections"], list) and len(c["sections"]) > 0


# ---------------------------------------------------------------------------
# Test: markdown heading detection in _load_text
# ---------------------------------------------------------------------------

def test_load_text_markdown_headings(tmp_path: Path) -> None:
    """Markdown headings are parsed as heading segments."""
    md_file = tmp_path / "NIST_test.md"
    md_file.write_text(
        "# Incident Response\n\nPrepare a response plan before incidents occur.\n\n"
        "## Containment\n\nContain the attack by isolating affected hosts.\n",
        encoding="utf-8",
    )
    segments = _load_text(md_file)
    heading_texts = [s["text"] for s in segments if s["is_heading"]]
    assert "Incident Response" in heading_texts
    assert "Containment" in heading_texts

    body_texts = " ".join(s["text"] for s in segments if not s["is_heading"])
    assert "response plan" in body_texts
    assert "isolating" in body_texts


# ---------------------------------------------------------------------------
# Test: NIST-style numbered heading detection
# ---------------------------------------------------------------------------

def test_nist_heading_detection() -> None:
    """Detect numbered headings, appendices, and Executive Summary from lines."""
    # Pattern: 3.2.4 Incident Analysis
    is_h, lvl, label = detect_heading("3.2.4 Incident Analysis")
    assert is_h is True
    assert lvl == 3
    assert label == "3.2.4 Incident Analysis"

    # Pattern: 3.2 Detection and Analysis
    is_h, lvl, label = detect_heading("3.2 Detection and Analysis")
    assert is_h is True
    assert lvl == 2
    assert label == "3.2 Detection and Analysis"

    # Pattern: 3. Handling an Incident
    is_h, lvl, label = detect_heading("3. Handling an Incident")
    assert is_h is True
    assert lvl == 1
    assert label == "3. Handling an Incident"

    # Pattern: 3 Handling an Incident (without dot)
    is_h, lvl, label = detect_heading("3 Handling an Incident")
    assert is_h is True
    assert lvl == 1
    assert label == "3 Handling an Incident"

    # Appendix
    is_h, lvl, label = detect_heading("Appendix A Incident Handling Scenarios")
    assert is_h is True
    assert lvl == 1
    assert "Appendix A" in label

    # Executive Summary
    is_h, lvl, label = detect_heading("Executive Summary")
    assert is_h is True
    assert lvl == 1
    assert label == "Executive Summary"

    # Non-heading normal sentences must NOT be detected as headings
    is_h, _, _ = detect_heading("This is a standard paragraph discussing incident mitigation.")
    assert is_h is False

    is_h, _, _ = detect_heading("1. Determine whether an incident has occurred.")
    assert is_h is False


# ---------------------------------------------------------------------------
# Test: Dropping dot-leader lines
# ---------------------------------------------------------------------------

def test_drop_dot_leader_lines() -> None:
    """Lines with dot leaders (e.g. from TOCs) must be recognized and dropped."""
    toc_lines = [
        "Executive Summary ................................ 1",
        "1. Introduction ................................ 4",
        "3.2.4 Incident Analysis . . . . . . . . . . . . 28",
        "Appendix A—Incident Handling Scenarios ............ 52",
    ]
    for line in toc_lines:
        assert is_dot_leader_line(line) is True, f"Failed to detect dot leader in: {line!r}"

    content_lines = [
        "Network flows provide session-level traffic metadata.",
        "Figure 3-2 illustrates the incident response lifecycle.",
        "An ellipsis ... within normal prose should not be dropped.",
    ]
    for line in content_lines:
        assert is_dot_leader_line(line) is False, f"False positive dot leader in: {line!r}"


# ---------------------------------------------------------------------------
# Test: Short chunk merging rules and drop rules
# ---------------------------------------------------------------------------

def test_short_chunk_merge_rules() -> None:
    """Validate strict merge rules:
    - Same section chunks merge if under 100 tokens and sum <= max_tokens.
    - Parent intro merges into first child (e.g. 2.3 into 2.3.1); child label is kept.
    - No cross-section merges otherwise (1.2 never merges with 1.3; 4.1.2 never with 4.2).
    - MITRE chunks never merge across sections.
    - Chunk under 30 tokens dropped only if no complete sentence.
    """
    # 1. Parent intro into first child
    c_parent = {
        "chunk_id": "c_p",
        "source_name": "NIST_SP_800-61_Rev2.pdf",
        "source_type": "NIST",
        "source_ref": "SP 800-61",
        "section": "2.3 Incident Handling Policy",
        "sections": ["2.3 Incident Handling Policy"],
        "text": "Organizations must establish clear incident handling policies.",
        "token_count": 10,
        "char_start": 0,
        "char_end": 60,
    }
    c_child = {
        "chunk_id": "c_c",
        "source_name": "NIST_SP_800-61_Rev2.pdf",
        "source_type": "NIST",
        "source_ref": "SP 800-61",
        "section": "2.3.1 Policy Elements",
        "sections": ["2.3.1 Policy Elements"],
        "text": "Policy elements include statement of management commitment and scope.",
        "token_count": 120,
        "char_start": 61,
        "char_end": 180,
    }

    # 2. Distinct non parent-child sections: must NEVER merge
    c_12 = {
        "chunk_id": "c_12",
        "source_name": "NIST_SP_800-61_Rev2.pdf",
        "source_type": "NIST",
        "source_ref": "SP 800-61",
        "section": "1.2 Purpose and Scope",
        "sections": ["1.2 Purpose and Scope"],
        "text": "This publication provides guidelines for incident handling.",
        "token_count": 12,
        "char_start": 181,
        "char_end": 240,
    }
    c_13 = {
        "chunk_id": "c_13",
        "source_name": "NIST_SP_800-61_Rev2.pdf",
        "source_type": "NIST",
        "source_ref": "SP 800-61",
        "section": "1.3 Audience",
        "sections": ["1.3 Audience"],
        "text": "The primary audience is system and network administrators responsible for security.",
        "token_count": 130,
        "char_start": 241,
        "char_end": 380,
    }

    # 3. MITRE cross-section: must NEVER merge
    c_m1 = {
        "chunk_id": "c_m1",
        "source_name": "MITRE_T1498.html",
        "source_type": "MITRE",
        "source_ref": "T1498",
        "section": "Description",
        "sections": ["Description"],
        "text": "Adversaries may conduct denial of service attacks.",
        "token_count": 10,
        "char_start": 381,
        "char_end": 430,
    }
    c_m2 = {
        "chunk_id": "c_m2",
        "source_name": "MITRE_T1498.html",
        "source_type": "MITRE",
        "source_ref": "T1498",
        "section": "Mitigations",
        "sections": ["Mitigations"],
        "text": "Filter incoming traffic using upstream firewall rules and packet inspection.",
        "token_count": 120,
        "char_start": 431,
        "char_end": 560,
    }

    # 4. Under 30 tokens: one incomplete sentence (drop) and one complete sentence (keep)
    c_frag = {
        "chunk_id": "c_frag",
        "source_name": "NIST_SP_800-61_Rev2.pdf",
        "source_type": "NIST",
        "source_ref": "SP 800-61",
        "section": "3.3 Incomplete",
        "sections": ["3.3 Incomplete"],
        "text": "incomplete fragment without terminal punctuation",
        "token_count": 8,
        "char_start": 561,
        "char_end": 610,
    }
    c_complete = {
        "chunk_id": "c_complete",
        "source_name": "NIST_SP_800-61_Rev2.pdf",
        "source_type": "NIST",
        "source_ref": "SP 800-61",
        "section": "3.4 Complete",
        "sections": ["3.4 Complete"],
        "text": "This is complete.",
        "token_count": 5,
        "char_start": 611,
        "char_end": 630,
    }

    # 5. Same section short chunks: DO merge
    c_same1 = {
        "chunk_id": "c_s1",
        "source_name": "NIST_SP_800-61_Rev2.pdf",
        "source_type": "NIST",
        "source_ref": "SP 800-61",
        "section": "3.5 Analysis",
        "sections": ["3.5 Analysis"],
        "text": "First part of analysis discussing network flow captures.",
        "token_count": 15,
        "char_start": 631,
        "char_end": 690,
    }
    c_same2 = {
        "chunk_id": "c_s2",
        "source_name": "NIST_SP_800-61_Rev2.pdf",
        "source_type": "NIST",
        "source_ref": "SP 800-61",
        "section": "3.5 Analysis",
        "sections": ["3.5 Analysis"],
        "text": "Second part of analysis detailing protocol anomalies.",
        "token_count": 15,
        "char_start": 691,
        "char_end": 750,
    }

    chunks = [c_parent, c_child, c_12, c_13, c_m1, c_m2, c_frag, c_complete, c_same1, c_same2]
    merged, dropped, cannot, merges_done = postprocess_short_chunks(chunks, max_tokens=350)

    # c_frag should be dropped
    assert len(dropped) == 1
    assert dropped[0]["chunk_id"] == "c_frag"

    # c_parent should be merged into c_child
    p_c_matches = [c for c in merged if c["section"] == "2.3.1 Policy Elements"]
    assert len(p_c_matches) == 1
    assert "clear incident handling policies" in p_c_matches[0]["text"]
    assert "Policy elements include" in p_c_matches[0]["text"]
    assert "2.3 Incident Handling Policy" in p_c_matches[0]["sections"]
    assert "2.3.1 Policy Elements" in p_c_matches[0]["sections"]

    # c_12 and c_13 must NOT be merged
    assert any(c["section"] == "1.2 Purpose and Scope" for c in merged)
    assert any(c["section"] == "1.3 Audience" for c in merged)

    # MITRE c_m1 and c_m2 must NOT be merged
    assert any(c["section"] == "Description" for c in merged)
    assert any(c["section"] == "Mitigations" for c in merged)

    # c_complete should be kept
    assert any(c["chunk_id"] == "c_complete" for c in merged)

    # c_same1 and c_same2 should be merged
    same_matches = [c for c in merged if c["section"] == "3.5 Analysis"]
    assert len(same_matches) == 1
    assert "First part" in same_matches[0]["text"]
    assert "Second part" in same_matches[0]["text"]


# ---------------------------------------------------------------------------
# Test: MITRE HTML cleaning
# ---------------------------------------------------------------------------

def test_mitre_html_cleaning() -> None:
    """Test that MITRE navigation text, References section, 'Version Permalink Live Version',
    and info symbol never appear in any chunk, that text before first real heading
    is labeled 'Description', and that metadata fields are preserved in Description.
    """
    html = """<!DOCTYPE html>
<html>
<body>
  <h1>Network Denial of Service</h1>
  <div class="card">
    <span class="info-icon">ⓘ</span>
    <div>ID: T1498</div>
    <div>Tactic: Impact</div>
    <div>Platforms: Linux, Windows, macOS</div>
    <div>Impact Type: Availability</div>
    <div>Version Permalink Live Version</div>
    <div>Contributors: MITRE Engenuity</div>
  </div>
  <h2>Sub-techniques</h2>
  <p>Adversaries may conduct network denial of service by exhausting bandwidth or system resources.</p>
  <h2>Mitigations</h2>
  <p>Employ upstream filtering and firewall rate limiting to restrict attack traffic.</p>
  <h2>References</h2>
  <p>1. Author A. DDoS Defense Guide. 2021.</p>
  <p>2. Author B. Volumetric Flooding Analysis. 2022.</p>
  <div>
    <span>×</span>
    <p>Core ATT&CK Objects Matrices Tactics Techniques Groups Software</p>
  </div>
</body>
</html>"""

    segments = _parse_html(html)
    cleaned = clean_mitre_segments(segments)
    chunks = _build_chunks(
        segments=cleaned,
        source_name="MITRE_T1498.html",
        source_type="MITRE",
        source_ref="T1498",
        min_tokens=1,
        max_tokens=350,
        overlap_tokens=0,
    )

    all_chunk_text = " ".join(c["text"] for c in chunks)
    all_sections = {c["section"] for c in chunks}

    # Verify unwanted boilerplate/navigation/references are completely removed
    assert "Core ATT&CK Objects" not in all_chunk_text
    assert "Matrices Tactics Techniques" not in all_chunk_text
    assert "Version Permalink" not in all_chunk_text
    assert "Live Version" not in all_chunk_text
    assert "Contributors:" not in all_chunk_text
    assert "ⓘ" not in all_chunk_text
    assert "\u24d8" not in all_chunk_text
    assert "DDoS Defense Guide" not in all_chunk_text
    assert "Volumetric Flooding Analysis" not in all_chunk_text
    assert "References" not in all_sections

    # Verify section labeling
    assert "Sub-techniques" not in all_sections
    assert "Description" in all_sections
    assert "Mitigations" in all_sections

    # Verify metadata fields are preserved in Description
    desc_chunks = [c for c in chunks if c["section"] == "Description"]
    assert len(desc_chunks) >= 1
    desc_text = " ".join(c["text"] for c in desc_chunks)
    assert "ID: T1498" in desc_text
    assert "Tactic: Impact" in desc_text
    assert "Platforms: Linux, Windows, macOS" in desc_text
    assert "Impact Type: Availability" in desc_text


# ---------------------------------------------------------------------------
# Test: Header and page-label stripping for SP 800-94 and SP 800-30 profiles
# ---------------------------------------------------------------------------

def test_header_and_page_label_stripping() -> None:
    """Validate header and page-label recognition rules for both PDF profiles."""
    # SP 800-94 running headers
    assert is_sp800_94_header("GUIDE TO INTRUSION DETECTION AND PREVENTION SYSTEMS (IDPS)")
    assert is_sp800_94_header("GUIDE TO INTRUSION DETECTION AND PREVENTION SYSTEM")
    assert is_sp800_94_header("guide to intrusion detection and prevention systems (idps)")
    assert not is_sp800_94_header("1. Introduction to Intrusion Detection")

    # SP 800-94 page labels
    for lbl in ["ES-4", "1-3", "2-12", "A-1", "B-10", "D-2"]:
        assert is_sp800_94_page_label(lbl), f"Expected valid page label: {lbl}"
    assert not is_sp800_94_page_label("Section 1-3 Overview")
    assert not is_sp800_94_page_label("Table 2-12 Detection Rates")

    # SP 800-30 three-header-lines
    assert is_sp800_30_header("Special Publication 800-30 Guide for Conducting Risk Assessments")
    assert is_sp800_30_header("Special Publication 800-30")
    assert not is_sp800_30_header("NIST Special Publication 800-30")

    # SP 800-30 page labels (Roman numerals for front matter, CHAPTER/APPENDIX for body)
    for pl in ["PAGE iv", "PAGE ix", "CHAPTER 1 PAGE 1", "CHAPTER 2   PAGE 15", "APPENDIX A PAGE A-1"]:
        assert is_sp800_30_page_label(pl), f"Expected valid SP 800-30 page label: {pl}"
    assert not is_sp800_30_page_label("Risk Assessment Process Page 1")
    assert not is_sp800_30_page_label("PAGE 7")


# ---------------------------------------------------------------------------
# Test: Sequential TOC matching ignores early cross-references
# ---------------------------------------------------------------------------

def test_sequential_toc_matching() -> None:
    """Sequential matching in TOC order ensures earlier cross-references
    matching a future heading string are not recognized prematurely as headings."""
    toc = ["1. Introduction", "1.1 Authority", "2.3.1 Signature-Based Detection"]
    lines = [
        "1. Introduction",
        "This section introduces IDPS technologies.",
        "As discussed in 2.3.1 Signature-Based Detection, signatures match known patterns.",
        "2.3.1 Signature-Based Detection",
        "1.1 Authority",
        "Statutory authority statement text.",
        "2.3.1 Signature-Based Detection",
        "Detailed discussion of signature algorithms.",
    ]

    toc_idx = 0
    matched_sections: list[str] = []
    for line in lines:
        if toc_idx < len(toc):
            target = toc[toc_idx]
            if line.strip() == target:
                matched_sections.append(target)
                toc_idx += 1

    assert matched_sections == [
        "1. Introduction",
        "1.1 Authority",
        "2.3.1 Signature-Based Detection",
    ]


# ---------------------------------------------------------------------------
# Test: Title-casing helper (SP 800-30)
# ---------------------------------------------------------------------------

def test_title_case_heading() -> None:
    """Validate title-casing of uppercase headings:
    - Lowercase for and, of, the, for, to, in, on, a, an (except as first word)
    - Preserve uppercase for NIST, IT, SP
    """
    assert title_case_heading("THE FUNDAMENTALS") == "The Fundamentals"
    assert title_case_heading("RISK ASSESSMENT AND NIST") == "Risk Assessment and NIST"
    assert title_case_heading("PURPOSE AND APPLICABILITY") == "Purpose and Applicability"
    assert title_case_heading("AN OVERVIEW OF IT SYSTEMS") == "An Overview of IT Systems"


# ---------------------------------------------------------------------------
# Test: Drop-cap repair (SP 800-30)
# ---------------------------------------------------------------------------

def test_drop_cap_repair() -> None:
    """Validate drop-cap prepending when initial letter is on a separate line."""
    lines = [
        "CHAPTER ONE",
        "INTRODUCTION",
        "rganizations in the public and private sectors rely on IT.",
        "O",
        "This is subsequent paragraph text.",
    ]
    repaired, done, not_done = repair_drop_caps(lines)
    assert done == 1
    assert not_done == 0
    assert "Organizations in the public and private sectors rely on IT." in repaired
    assert "O" not in repaired
    assert "rganizations in the public and private sectors rely on IT." not in repaired

    # Unconfident case: no single uppercase line to restore
    no_cap_lines = [
        "his chapter describes the risk assessment process.",
        "Next sentence without stray cap.",
    ]
    repaired2, done2, not_done2 = repair_drop_caps(no_cap_lines)
    assert done2 == 0
    assert not_done2 == 1


# ---------------------------------------------------------------------------
# Test: Footnote regex and removal
# ---------------------------------------------------------------------------

def test_footnote_removal() -> None:
    """Footnote numbers glued to words are removed; technical terms remain unchanged."""
    # Positive examples from item D
    cleaned1, s1 = remove_footnotes("Analyzing security incidents.5 promptly is critical.")
    assert "incidents." in cleaned1
    assert "incidents.5" not in cleaned1
    assert len(s1) == 1

    cleaned2, s2 = remove_footnotes("Organizations5 must establish policies.")
    assert "Organizations" in cleaned2
    assert "Organizations5" not in cleaned2
    assert len(s2) == 1

    cleaned3, s3 = remove_footnotes("Coordinate with management them.7")
    assert "management them." in cleaned3
    assert "them.7" not in cleaned3
    assert len(s3) == 1

    # Negative examples from item D that must remain untouched
    negatives = [
        "IPv6",
        "sha256",
        "SP 800-61",
        "802.11",
        "Section 3.2",
        "T1498.001",
    ]
    for term in negatives:
        cleaned_neg, samples = remove_footnotes(f"Testing term {term} in network flow.")
        assert term in cleaned_neg, f"Term {term!r} was modified unexpectedly"
        assert len(samples) == 0, f"Unexpected footnote extraction for {term!r}: {samples}"


# ---------------------------------------------------------------------------
# Test: TOC parser removes page fragments like '...2-4'
# ---------------------------------------------------------------------------

def test_toc_fragment_removal() -> None:
    """Trailing dot leaders and page fragments (e.g. '...2-4', '...2-') are removed."""
    synthetic_toc_page = """
1. Introduction ............................................................................................1-1
1.1 Authority ................................................................................................1-2
2.3.1 Signature-Based Detection ..............................................................2-4
2.3.2 Anomaly-Based Detection ..............................................................2-
Appendix A— Glossary ..................................................................................A-1
Appendix B— Acronyms ..................................................................................B-1
"""
    entries = _extract_toc_sp800_94([synthetic_toc_page])
    assert "1. Introduction" in entries
    assert "1.1 Authority" in entries
    assert "2.3.1 Signature-Based Detection" in entries
    assert "2.3.2 Anomaly-Based Detection" in entries
    assert any("Appendix A" in e for e in entries)
    # Ensure no trailing fragments leaked into entry titles
    for e in entries:
        assert not re.search(r"\d+-\d*$", e), f"Page fragment leaked into entry: {e!r}"


# ---------------------------------------------------------------------------
# Test: normalize_heading_key
# ---------------------------------------------------------------------------

def test_normalize_heading_key() -> None:
    """Validate normalize_heading_key:
    - '1. Introduction' == '1 Introduction'
    - 'Signature\\u2011Based Detection' == 'Signature\\u2013Based Detection' == 'Signature-Based Detection'
    - 'Appendix A\\u2014 Glossary' == 'Appendix A\\u2014Glossary'
    """
    assert normalize_heading_key("1. Introduction") == normalize_heading_key("1 Introduction")
    h1 = normalize_heading_key("Signature\u2011Based Detection")
    h2 = normalize_heading_key("Signature\u2013Based Detection")
    h3 = normalize_heading_key("Signature-Based Detection")
    assert h1 == h2 == h3 == "signature-based detection"
    assert normalize_heading_key("Appendix A\u2014 Glossary") == normalize_heading_key("Appendix A\u2014Glossary")


# ---------------------------------------------------------------------------
# Test: Sequential TOC matching with wrapping & cross-reference rejection
# ---------------------------------------------------------------------------

def test_sequential_toc_matching_with_wrapping() -> None:
    """Sequential matching in TOC order ensures earlier cross-references
    matching a future heading string are not recognized prematurely as headings,
    and long headings that wrap over two lines are joined and matched."""
    toc_entries = [
        "1. Introduction",
        "1.1 Authority",
        "2.3.1 Signature-Based Detection",
        "2.3.2 Anomaly-Based Detection Recommendations and Best Practices",
    ]
    toc_keys = [normalize_heading_key(e) for e in toc_entries]

    lines = [
        "1. Introduction",
        "This section introduces IDPS technologies.",
        "As discussed in 2.3.1 Signature-Based Detection, signatures match known patterns.",
        "2.3.1 Signature-Based Detection",  # Premature cross-reference line
        "1.1 Authority",
        "Statutory authority statement text.",
        "2.3.1 Signature\u2011Based Detection",  # Non-breaking hyphen heading
        "Detailed discussion of signature algorithms.",
        "2.3.2 Anomaly-Based Detection",  # Wrapped line 1
        "Recommendations and Best Practices",  # Wrapped line 2
        "Details on anomaly detection methods.",
    ]

    matched_sections: list[str] = []
    toc_idx = 0
    line_i = 0
    while line_i < len(lines):
        line_s = lines[line_i].strip()
        if toc_idx < len(toc_keys):
            target_key = toc_keys[toc_idx]
            cand_key = normalize_heading_key(line_s)
            if cand_key == target_key:
                matched_sections.append(toc_entries[toc_idx])
                toc_idx += 1
                line_i += 1
                continue
            elif target_key.startswith(cand_key) and line_i + 1 < len(lines):
                next_s = lines[line_i + 1].strip()
                joined_key = normalize_heading_key(line_s + " " + next_s)
                if joined_key == target_key:
                    matched_sections.append(toc_entries[toc_idx])
                    toc_idx += 1
                    line_i += 2
                    continue
        line_i += 1

    assert matched_sections == [
        "1. Introduction",
        "1.1 Authority",
        "2.3.1 Signature-Based Detection",
        "2.3.2 Anomaly-Based Detection Recommendations and Best Practices",
    ]


# ---------------------------------------------------------------------------
# Test: SP 800-94 Appendix B, C, D dropped by label
# ---------------------------------------------------------------------------

def test_sp800_94_appendix_dropping_by_label() -> None:
    """SP 800-94 profile keeps Executive Summary, sections 1 to 9, and Appendix A,
    and drops Appendix B, C, D completely by section label."""
    toc_entries = [
        "Executive Summary",
        "1. Introduction",
        "Appendix A—Glossary",
        "Appendix B—Acronyms",
        "Appendix C—Tools and Resources",
        "Appendix D—Index",
    ]
    toc_keys = [normalize_heading_key(e) for e in toc_entries]

    body_lines = [
        "Executive Summary",
        "Executive summary content for intrusion prevention.",
        "1. Introduction",
        "Introduction content covering basics.",
        "Appendix A—Glossary",
        "Glossary definitions content.",
        "Appendix B—Acronyms",
        "IDPS: Intrusion Detection and Prevention System.",
        "Appendix C—Tools and Resources",
        "List of security tools and online portals.",
        "Appendix D—Index",
        "Alphabetical index entries.",
    ]

    segments: list[dict[str, Any]] = []
    toc_idx = 0
    in_dropped = False
    for line in body_lines:
        cand_key = normalize_heading_key(line)
        if toc_idx < len(toc_keys) and cand_key == toc_keys[toc_idx]:
            label = toc_entries[toc_idx]
            toc_idx += 1
            if label.startswith("Appendix B") or label.startswith("Appendix C") or label.startswith("Appendix D"):
                in_dropped = True
            else:
                in_dropped = False
                segments.append({"is_heading": True, "heading_level": 1, "text": label})
        else:
            if not in_dropped:
                segments.append({"is_heading": False, "heading_level": 0, "text": line})

    kept_headings = [s["text"] for s in segments if s["is_heading"]]
    assert kept_headings == ["Executive Summary", "1. Introduction", "Appendix A—Glossary"]
    all_kept_text = " ".join(s["text"] for s in segments)
    assert "Executive summary content" in all_kept_text
    assert "Introduction content" in all_kept_text
    assert "Glossary definitions" in all_kept_text
    assert "IDPS: Intrusion Detection" not in all_kept_text
    assert "List of security tools" not in all_kept_text
    assert "Alphabetical index" not in all_kept_text


# ---------------------------------------------------------------------------
# Test: SP 800-94 Section 1.1 Authority dropped by label
# ---------------------------------------------------------------------------

def test_sp800_94_authority_dropping_by_label() -> None:
    """SP 800-94 profile drops section 1.1 Authority (statutory boilerplate)
    while keeping surrounding sections 1. Introduction and 1.2 Purpose and Scope."""
    toc_entries = [
        "1. Introduction",
        "1.1 Authority",
        "1.2 Purpose and Scope",
    ]
    toc_keys = [normalize_heading_key(e) for e in toc_entries]

    body_lines = [
        "1. Introduction",
        "Introductory guidance on IDPS technologies.",
        "1.1 Authority",
        "Statutory boilerplate text under NIST authority requirements.",
        "1.2 Purpose and Scope",
        "Scope of this publication covers intrusion detection and prevention.",
    ]

    segments: list[dict[str, Any]] = []
    toc_idx = 0
    in_dropped = False
    for line in body_lines:
        cand_key = normalize_heading_key(line)
        if toc_idx < len(toc_keys) and cand_key == toc_keys[toc_idx]:
            label = toc_entries[toc_idx]
            toc_idx += 1
            if normalize_heading_key(label) == "1.1 authority" or label.startswith("1.1 Authority"):
                in_dropped = True
            else:
                in_dropped = False
                segments.append({"is_heading": True, "heading_level": 2, "text": label})
        else:
            if not in_dropped:
                segments.append({"is_heading": False, "heading_level": 0, "text": line})

    kept_headings = [s["text"] for s in segments if s["is_heading"]]
    assert kept_headings == ["1. Introduction", "1.2 Purpose and Scope"]
    all_kept_text = " ".join(s["text"] for s in segments)
    assert "Introductory guidance" in all_kept_text
    assert "Scope of this publication" in all_kept_text
    assert "Statutory boilerplate" not in all_kept_text



