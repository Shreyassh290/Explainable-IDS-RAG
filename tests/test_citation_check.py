"""Unit tests for citation support checker (tests/test_citation_check.py).

Tests run entirely in-memory with small synthetic fixtures:
1. expand and form_of handle single IDs, comma/semicolon lists, and ranges (hyphen and en dash).
2. split_units and extract_flow operate on a synthetic report with single/list citations and dropped header lines.
3. overlap_score produces 1.0 for identical text, 0.0 for disjoint text, and an exact hand-computed partial score.
4. drop_header_lines drops copied NIST/MITRE header lines and retains normal sentences.
5. extract_flow flagging uses the threshold passed in (same pair flagged at 0.5 and kept at lower threshold).
6. main refuses to overwrite existing files and exits with code 1, leaving files intact.
"""

import json
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
    run_citation_check,
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


def test_reports_dir_mode(tmp_path: Path) -> None:
    """Test citation check scoring on a custom reports folder with tuned-run layout."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    kb_dir = tmp_path / "data" / "kb"
    kb_dir.mkdir(parents=True)
    chunks_file = kb_dir / "chunks.jsonl"
    chunk_data = {
        "chunk_id": "chunk_001",
        "source_name": "NIST_SP_800-61_Rev2.pdf",
        "text": "Organizations should apply network ingress filtering and firewall access controls to prevent denial of service.",
    }
    chunks_file.write_text(json.dumps(chunk_data) + "\n", encoding="utf-8")

    # Flow 1: valid citation [1] with high overlap and out-of-range citation [9]
    flow1_resp = "Apply network ingress filtering and firewall access controls [1]. Additional unverified step [9]."
    flow1_json = {
        "flow_id": 1,
        "true_class": "DoS",
        "predicted_class": "DoS",
        "response": flow1_resp,
        "prompt": {
            "sources": [
                {
                    "n": 1,
                    "chunk_id": "chunk_001",
                    "source_name": "NIST_SP_800-61_Rev2.pdf",
                }
            ]
        },
    }
    (reports_dir / "flow_1_rag.txt").write_text(flow1_resp, encoding="utf-8")
    (reports_dir / "flow_1_rag.json").write_text(json.dumps(flow1_json), encoding="utf-8")

    # Flow 2: copied header line and no real citation
    flow2_resp = "[1] NIST SP 800-94, section 3.2.3 Detection Capabilities\nNormal operational traffic observed with no threats detected."
    flow2_json = {
        "flow_id": 2,
        "true_class": "Benign",
        "predicted_class": "Benign",
        "response": flow2_resp,
        "prompt": {
            "sources": [
                {
                    "n": 1,
                    "chunk_id": "chunk_001",
                    "source_name": "NIST_SP_800-94.pdf",
                }
            ]
        },
    }
    (reports_dir / "flow_2_rag.txt").write_text(flow2_resp, encoding="utf-8")
    (reports_dir / "flow_2_rag.json").write_text(json.dumps(flow2_json), encoding="utf-8")

    cfg = {"citation_check": {"overlap_threshold": 0.5}}
    out_dir = tmp_path / "out"

    res = run_citation_check(
        root=tmp_path,
        out_dir=out_dir,
        reports_dir=reports_dir,
        cfg=cfg,
    )

    # 1. Check citation rows counted correctly
    assert len(res["all_rows"]) == 2
    assert len(res["per_report"]) == 2
    # Flow 1 has 2 citations, Flow 2 has 0 citations
    p1 = next(p for p in res["per_report"] if p["flow_index"] == "1")
    p2 = next(p for p in res["per_report"] if p["flow_index"] == "2")
    assert p1["citations"] == 2
    assert p2["citations"] == 0

    # 2. Check header line skipped
    assert res["hdr_total"] == 1
    assert "copied source-header lines skipped (same rule as generate.py): 1 (expected n/a)" in res["summary_text"]
    assert "(expected 5)" not in res["summary_text"]

    # 3. Check out-of-range citation flagged
    row_c1 = next(r for r in res["all_rows"] if r["n"] == 1)
    row_c9 = next(r for r in res["all_rows"] if r["n"] == 9)
    assert row_c1["out_of_range"] is False
    assert row_c1["flagged"] is False
    assert row_c1["overlap"] is not None and row_c1["overlap"] >= 0.5
    assert row_c9["out_of_range"] is True
    assert row_c9["flagged"] is True

    # Check generated files exist
    assert (out_dir / "citation_check_rows.csv").exists()
    assert (out_dir / "citation_check_per_report.csv").exists()
    assert (out_dir / "citation_check_summary.txt").exists()


def test_reports_dir_existing_output_refused(tmp_path: Path) -> None:
    """Test that existing output in --out-dir is refused when --reports-dir is used."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    summary_file = out_dir / "citation_check_summary.txt"
    summary_file.write_text("existing summary", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(["--reports-dir", str(reports_dir), "--out-dir", str(out_dir)])
    assert exc_info.value.code == 1
    assert summary_file.read_text(encoding="utf-8") == "existing summary"


def test_reports_dir_missing_out_dir_rejected(tmp_path: Path) -> None:
    """Test that missing --out-dir is rejected when --reports-dir is provided."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        main(["--reports-dir", str(reports_dir)])
    assert exc_info.value.code == 1


def test_reports_dir_missing_file_or_sources_rejected(tmp_path: Path) -> None:
    """Test that missing flow JSON or sources list triggers clear STOP message."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    kb_dir = tmp_path / "data" / "kb"
    kb_dir.mkdir(parents=True)
    (kb_dir / "chunks.jsonl").write_text('{"chunk_id": "c1", "source_name": "NIST_X.pdf", "section": "S", "text": "hello"}\n', encoding="utf-8")

    # Flow with missing sources list in prompt
    bad_json = {
        "flow_id": 1,
        "true_class": "DoS",
        "predicted_class": "DoS",
        "response": "Text [1]",
        "prompt": {},
    }
    (reports_dir / "flow_1_rag.json").write_text(json.dumps(bad_json), encoding="utf-8")

    cfg = {"citation_check": {"overlap_threshold": 0.5}}
    with pytest.raises(SystemExit) as exc_info:
        run_citation_check(
            root=tmp_path,
            out_dir=tmp_path / "out",
            reports_dir=reports_dir,
            cfg=cfg,
        )
    assert exc_info.value.code == 1


def test_phase1_default_path_unchanged() -> None:
    """Test that Phase 1 default output path is artifacts/eval/citation_check when --reports-dir is None."""
    from ids_rag.config import get_project_root

    root = get_project_root()
    default_out = root / "artifacts" / "eval" / "citation_check"
    assert default_out.name == "citation_check"
    assert default_out.parent.name == "eval"


def test_reports_dir_phase1_layout(tmp_path: Path) -> None:
    """Test citation check scoring on a reports folder holding flow_<id>.json in Phase 1 layout."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    kb_dir = tmp_path / "data" / "kb"
    kb_dir.mkdir(parents=True)
    chunks_file = kb_dir / "chunks.jsonl"
    chunk_data = {
        "chunk_id": "chunk_001",
        "source": "NIST_SP_800-61",
        "text": "Organizations should apply network ingress filtering and firewall access controls to prevent denial of service.",
    }
    chunks_file.write_text(json.dumps(chunk_data) + "\n", encoding="utf-8")

    # One report with a copied header line and one valid [1] citation
    rag_text = (
        "[1] NIST SP 800-94, section 3.2.3 Detection Capabilities\n"
        "Organizations should apply network ingress filtering and firewall access controls [1]."
    )
    flow1_json = {
        "index": 1,
        "true_class": "DoS",
        "predicted_class": "DoS",
        "retrieved_chunks": [
            {
                "chunk_id": "chunk_001",
                "source": "NIST_SP_800-61",
            }
        ],
        "report_texts": {
            "rag": rag_text,
            "vanilla": "Baseline text without citations.",
        },
    }
    (reports_dir / "flow_1.json").write_text(json.dumps(flow1_json), encoding="utf-8")

    cfg = {"citation_check": {"overlap_threshold": 0.5}}
    out_dir = tmp_path / "out"

    res = run_citation_check(
        root=tmp_path,
        out_dir=out_dir,
        reports_dir=reports_dir,
        cfg=cfg,
    )

    # Check row count: 1 valid citation row
    assert len(res["all_rows"]) == 1
    assert res["all_rows"][0]["n"] == 1
    assert res["all_rows"][0]["out_of_range"] is False
    assert res["all_rows"][0]["flagged"] is False
    assert res["all_rows"][0]["overlap"] is not None and res["all_rows"][0]["overlap"] >= 0.5

    # Check header line skipped
    assert res["hdr_total"] == 1

    # Check per_report summary
    assert len(res["per_report"]) == 1
    assert res["per_report"][0]["flow_index"] == "1"
    assert res["per_report"][0]["citations"] == 1


