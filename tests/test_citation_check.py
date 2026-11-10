"""Unit tests for citation support checker (tests/test_citation_check.py).

Tests run entirely in-memory with small synthetic fixtures:
1. expand and form_of handle single IDs, comma/semicolon lists, and ranges (hyphen and en dash).
2. split_units and extract_flow operate on a synthetic report with single/list citations and dropped header lines.
3. overlap_score produces 1.0 for identical text, 0.0 for disjoint text, and an exact hand-computed partial score.
4. drop_header_lines drops copied NIST/MITRE header lines and retains normal sentences.
5. extract_flow flagging uses the threshold passed in (same pair flagged at 0.5 and kept at lower threshold).
6. main refuses to overwrite existing files and exits with code 1, leaving files intact.
"""

from pathlib import Path
import pytest

from ids_rag.eval.citation_check import (
    clean_sentence,
    drop_header_lines,
    expand,
    extract_flow,
    form_of,
    main,
    overlap_score,
    split_units,
    tokens,
)


def test_expand_and_form_of() -> None:
    """Test citation expansion and form classification for singles, lists, and ranges."""
    # Single ID
    assert expand("1") == [1]
    assert form_of("1") == "single"
    assert expand("12") == [12]
    assert form_of("12") == "single"

    # Comma or semicolon list
    assert expand("1, 2, 4") == [1, 2, 4]
    assert form_of("1, 2, 4") == "list"
    assert expand("2; 3") == [2, 3]
    assert form_of("2; 3") == "list"

    # Range with hyphen
    assert expand("1-3") == [1, 2, 3]
    assert form_of("1-3") == "range"
    assert expand("5-4") == [5, 4]
    assert form_of("5-4") == "range"

    # Range with en dash (normalized as done in extract_flow)
    en_dash_body = "2\u20134".replace("\u2013", "-")
    assert expand(en_dash_body) == [2, 3, 4]
    assert form_of(en_dash_body) == "range"


def test_split_units_and_extract_flow() -> None:
    """Test sentence splitting and citation extraction on a short synthetic report."""
    report_text = (
        "[1] NIST SP 800-94, section 3.2.3 Guide to Intrusion Detection\n"
        "Suspicious network traffic was observed on target port 80 [1].\n"
        "Mitigation actions should be applied to prevent recurring incidents [2, 3].\n"
    )

    cleaned_text, dropped = drop_header_lines(report_text)
    assert dropped == 1
    assert "[1] NIST SP 800-94" not in cleaned_text

    units = split_units(cleaned_text)
    assert len(units) == 2

    retrieved = [
        {"chunk_id": "c1", "source": "NIST_SP_800-94"},
        {"chunk_id": "c2", "source": "MITRE_T1498"},
        {"chunk_id": "c3", "source": "MITRE_T1499"},
    ]
    chunks = {
        "c1": {"chunk_id": "c1", "text": "Intrusion detection systems monitor suspicious network traffic on port 80."},
        "c2": {"chunk_id": "c2", "text": "Mitigation steps against denial of service incidents."},
        "c3": {"chunk_id": "c3", "text": "Prevent recurring incidents and apply firewall filters."},
    }

    rows = extract_flow(cleaned_text, retrieved, chunks, thr=0.5)
    assert len(rows) == 3

    # Citation 1: single form, NIST family
    assert rows[0]["n"] == 1
    assert rows[0]["form"] == "single"
    assert rows[0]["family"] == "NIST"
    assert rows[0]["out_of_range"] is False
    assert rows[0]["overlap"] is not None
    assert rows[0]["overlap"] > 0.0

    # Citations 2 and 3: list form, MITRE family
    assert rows[1]["n"] == 2
    assert rows[1]["form"] == "list"
    assert rows[1]["family"] == "MITRE"
    assert rows[1]["out_of_range"] is False

    assert rows[2]["n"] == 3
    assert rows[2]["form"] == "list"
    assert rows[2]["family"] == "MITRE"
    assert rows[2]["out_of_range"] is False


def test_overlap_score() -> None:
    """Test word overlap scoring on identical, disjoint, and partial text."""
    # Identical text gives 1.0
    text1 = "malicious network traffic detected"
    assert overlap_score(text1, text1) == 1.0

    # Disjoint text gives 0.0
    text2 = "completely unrelated document content"
    assert overlap_score(text1, text2) == 0.0

    # Hand-computed partial case:
    # Sentence tokens: "network packet inspection protocol" -> 4 tokens {"network", "packet", "inspection", "protocol"}
    # Chunk tokens: "network firewall protocol guidelines" -> 4 tokens {"network", "firewall", "protocol", "guidelines"}
    # Intersection: {"network", "protocol"} (2 tokens) -> 2 / 4 = 0.5
    sent = "network packet inspection protocol"
    chunk = "network firewall protocol guidelines"
    assert overlap_score(sent, chunk) == 0.5

    # Empty tokens give 0.0
    assert overlap_score("", "") == 0.0
    assert overlap_score("this that with from", "chunk") == 0.0


def test_drop_header_lines() -> None:
    """Test that copied header lines are removed and normal text is preserved."""
    text = (
        "[1] NIST SP 800-94, section 3.2.3 Guide to Intrusion Detection\n"
        "This is a normal sentence describing incident response.\n"
        "[2] MITRE ATT&CK T1498, section Description\n"
        "Another normal sentence."
    )
    cleaned, dropped = drop_header_lines(text)
    assert dropped == 2
    assert "[1] NIST" not in cleaned
    assert "[2] MITRE" not in cleaned
    assert "This is a normal sentence describing incident response." in cleaned
    assert "Another normal sentence." in cleaned


def test_flagging_threshold() -> None:
    """Test that the same citation pair is flagged at 0.5 and kept at a lower threshold."""
    # Sentence has 5 tokens: {"network", "packet", "inspection", "protocol", "anomaly"}
    # Chunk has 2 matching tokens: {"network", "protocol"} out of 4
    # Overlap = 2 / 5 = 0.4
    sent = "network packet inspection protocol anomaly [1]"
    chunk_text = "network firewall protocol guidelines"
    retrieved = [{"chunk_id": "c1", "source": "NIST_SP_800-61"}]
    chunks = {"c1": {"chunk_id": "c1", "text": chunk_text}}

    # At threshold 0.5: overlap (0.4) < 0.5 -> flagged = True
    rows_high = extract_flow(sent, retrieved, chunks, thr=0.5)
    assert len(rows_high) == 1
    assert rows_high[0]["overlap"] == pytest.approx(0.4)
    assert rows_high[0]["flagged"] is True

    # At lower threshold 0.3: overlap (0.4) >= 0.3 -> flagged = False
    rows_low = extract_flow(sent, retrieved, chunks, thr=0.3)
    assert len(rows_low) == 1
    assert rows_low[0]["overlap"] == pytest.approx(0.4)
    assert rows_low[0]["flagged"] is False


def test_output_refusal(tmp_path: Path) -> None:
    """Test that existing output files cause main to exit with code 1 without modifying them."""
    summary_file = tmp_path / "citation_check_summary.txt"
    summary_file.write_text("pre-existing content", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(["--out-dir", str(tmp_path)])

    assert exc_info.value.code == 1
    assert summary_file.read_text(encoding="utf-8") == "pre-existing content"
    assert not (tmp_path / "citation_check_rows.csv").exists()
    assert not (tmp_path / "citation_check_per_report.csv").exists()

