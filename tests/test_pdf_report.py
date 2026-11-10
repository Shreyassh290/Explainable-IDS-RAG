"""Unit tests for offline PDF incident report generation (tests/test_pdf_report.py).

Tests run entirely in-memory with hand-written summary dictionaries:
- Validates PDF bytes output starting with b"%PDF" across edge cases
- Tests non-Latin-1 character sanitization (curly quotes, em-dashes, unicode)
- Tests step parsing from recommendations section, ensuring markdown and citations are stripped
"""

from __future__ import annotations

from typing import Any

import pytest

from ids_rag.pdf_report import build_pdf, extract_recommendation_steps, sanitize_text


@pytest.fixture
def ddos_summary() -> dict[str, Any]:
    return {
        "predicted_class": "DDoS",
        "confidence": 0.9985,
        "tier": "VERY_HIGH",
        "explanation": {
            "features": [
                {
                    "name": "Flow Pkts/s",
                    "original_value": 220.12,
                    "direction": "lowers",
                    "importance": 10.72,
                },
                {
                    "name": "Init Fwd Win Byts",
                    "original_value": 32738.0,
                    "direction": "raises",
                    "importance": 6.45,
                },
            ]
        },
        "reports": {
            "rag": (
                "**1. Rationale**\nObserved high packet rate.\n\n"
                "**5. Recommendations**\n"
                "- **Isolate** the target subnet [1].\n"
                "- Apply rate-limiting filter rules [2].\n"
                "- Inspect upstream firewall logs [3]."
            )
        },
        "chunks": [
            {
                "source": "MITRE_T1498_001.html",
                "section": "Description",
                "ce_score": 0.9809,
            },
            {
                "source": "MITRE_T1498.html",
                "section": "Mitigation",
                "ce_score": 0.9312,
            },
        ],
    }


@pytest.fixture
def benign_summary_7_chunks() -> dict[str, Any]:
    return {
        "predicted_class": "Benign",
        "confidence": 1.0000,
        "tier": "VERY_HIGH",
        "explanation": {
            "features": [
                {
                    "name": "Tot Bwd Pkts",
                    "original_value": 7.0,
                    "direction": "raises",
                    "importance": 170.52,
                }
            ]
        },
        "reports": {
            "rag": "Normal administrative session observed. No malicious activity detected."
        },
        "chunks": [
            {"source": "NIST_SP_800-94.pdf", "section": "6.2.3 Detection", "ce_score": 0.2642},
            {"source": "NIST_SP_800-94.pdf", "section": "2.5 Summary", "ce_score": 0.2335},
            {"source": "NIST_SP_800-94.pdf", "section": "3.2.3 Detection", "ce_score": 0.2073},
            {"source": "NIST_SP_800-94.pdf", "section": "2.5 Summary", "ce_score": 0.1546},
            {"source": "NIST_SP_800-94.pdf", "section": "2.5 Summary", "ce_score": 0.0997},
            {"source": "NIST_SP_800-61_Rev2.pdf", "section": "Executive Summary", "ce_score": 0.0000},
            {"source": "NIST_SP_800-61_Rev2.pdf", "section": "3.6 Recommendations", "ce_score": 0.0000},
        ],
    }


def test_build_pdf_ddos_summary(ddos_summary: dict[str, Any]) -> None:
    pdf_bytes = build_pdf(ddos_summary, source_label="DDoS Benchmark (Index 0)")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500


def test_build_pdf_benign_7_chunks(benign_summary_7_chunks: dict[str, Any]) -> None:
    pdf_bytes = build_pdf(benign_summary_7_chunks, source_label="uploaded row 0")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500


def test_build_pdf_no_recommendations_heading() -> None:
    summary = {
        "predicted_class": "DoS",
        "confidence": 0.95,
        "tier": "HIGH",
        "reports": {"rag": "Single paragraph report without explicit section headings."},
        "chunks": [],
    }
    pdf_bytes = build_pdf(summary)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_build_pdf_non_latin1_and_quotes() -> None:
    summary = {
        "predicted_class": "DDoS",
        "confidence": 0.99,
        "tier": "VERY_HIGH",
        "reports": {
            "rag": (
                "Here are \u201ccurly quotes\u201d, an em\u2014dash, a bullet \u2022, "
                "an arrow \u2192, and cyrillic \u0416\u0435\u043b\u0430\u043d\u0438\u0435 "
                "with emoji \U0001F6E1."
            )
        },
        "chunks": [{"source": "Test\u2014Source.pdf", "section": "Sec \u2022 1", "ce_score": 0.5}],
    }
    pdf_bytes = build_pdf(summary, source_label="Test \u2013 Flow")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_build_pdf_empty_reports_dict() -> None:
    summary = {
        "predicted_class": "Benign",
        "confidence": 0.88,
        "tier": "HIGH",
        "reports": {},
        "chunks": [],
    }
    pdf_bytes = build_pdf(summary)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_extract_recommendation_steps_strips_formatting() -> None:
    text = (
        "## 4. Threat Assessment\n"
        "Threat is high.\n\n"
        "## 5. Recommendations\n"
        "- **Block** source IPs at edge router [1].\n"
        "* Configure **SYN proxy** defense [2, 3].\n"
        "1. Alert incident response team [4].\n"
        "2. Review firewall metrics [5].\n"
    )
    steps = extract_recommendation_steps(text)
    assert len(steps) == 4

    for step in steps:
        assert "**" not in step
        assert not step.startswith("-")
        assert not step.startswith("*")
        assert not step[0].isdigit() if step else True
        assert "[" not in step
        assert "]" not in step

    assert "Block source IPs at edge router." in steps[0]
    assert "Configure SYN proxy defense." in steps[1]
    assert "Alert incident response team." in steps[2]
    assert "Review firewall metrics." in steps[3]

