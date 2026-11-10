"""Unit tests for incident response prompt construction (tests/test_prompt.py).

Verifies the prompt builder requirements from arXiv 2605.17960:
1. Both modes contain the five numbered headings and the system sentence unchanged.
2. Vanilla user text contains no '[1]' and the retrieved heading is followed directly by the next heading;
   RAG text contains '[1]' to '[n]' in rank order, the grounding rule, and the '(general guidance)' suffix on fallbacks.
3. Missing metadata (None, NaN, empty string) prints 'not available' for each field.
4. position_word returns the right word at z = 0, 0.49, 0.5, 1.49, 1.5, -1.5, 3.0.
5. format_value never returns scientific notation for 4.7e6 and keeps at most 4 significant digits.
6. Private-use characters are replaced and no U+E000 to U+F8FF character remains in the prompt.
7. A Benign explanation adds the Benign paragraph; a DoS explanation does not.
8. Developer notes of a feature never appear in the prompt.
9. Invalid mode raises ValueError.
10. MITRE bracketed citation removal while keeping source headers.
11. Grounding rule wording: contains "never a range" and "do not add a reference list"; no literal "[n]".
12. Near-average implication suppression flag behaviour.
"""

import re
from typing import Any
import pytest

from ids_rag.llm.prompt import build_prompt, clean_chunk_text, format_value, position_word


@pytest.fixture
def sample_explanation() -> dict[str, Any]:
    """Synthetic explanation dictionary for testing."""
    return {
        "predicted_class": "DoS",
        "predicted_label": 1,
        "confidence": 0.9876,
        "tier": "VERY_HIGH",
        "features": [
            {
                "name": "Tot Bwd Pkts",
                "scaled_value": -0.0354,
                "original_value": 0.0,
                "importance": 2.74,
                "direction": "lowers",
                "description": "Total backward packets transmitted.",
                "implication": "Near-zero backward packet count is consistent with flooding.",
            },
            {
                "name": "ACK Flag Cnt",
                "scaled_value": 1.1667,
                "original_value": 1.0,
                "importance": 2.78,
                "direction": "raises",
                "description": "Total count of ACK packets.",
                "implication": "Elevated ACK count under connection interaction.",
            },
        ],
        "metadata": {
            "Timestamp": "20/02/2018 10:46:44",
            "Dst Port": 80,
            "Flow ID": "172.31.64.122-199.247.10.78-80-50260-6",
            "Src IP": "199.247.10.78",
            "Src Port": 50260,
            "Dst IP": "172.31.64.122",
            "protocol": 6,
            "dataset": "CICIDS2018",
        },
    }


@pytest.fixture
def sample_chunks() -> list[dict[str, Any]]:
    """Synthetic retrieved chunks including standard and fallback entries."""
    return [
        {
            "chunk_id": "c1",
            "text": "NIST guidance on denial of service handling.",
            "source_name": "NIST_SP_800-61_Rev2.pdf",
            "source_type": "NIST",
            "source_ref": "SP 800-61 Rev2",
            "section": "3.2 Incident Handling",
            "is_fallback": False,
        },
        {
            "chunk_id": "c2",
            "text": "MITRE ATT&CK mitigation techniques for network flooding.",
            "source_name": "T1498.html",
            "source_type": "MITRE",
            "source_ref": "T1498",
            "section": "Mitigation",
            "is_fallback": False,
        },
        {
            "chunk_id": "c3",
            "text": "General recommendations for incident response teams.",
            "source_name": "NIST_SP_800-61_Rev2.pdf",
            "source_type": "NIST",
            "source_ref": "SP 800-61 Rev2",
            "section": "3.6 Recommendations",
            "is_fallback": True,
        },
    ]


@pytest.fixture
def test_cfg() -> dict[str, Any]:
    """Test configuration dictionary with thresholds and tier mappings."""
    return {
        "llm": {
            "min_words": 400,
            "max_words": 500,
            "tier_actions": {
                "VERY_HIGH": "Very High: fully automated response",
                "HIGH": "High: analyst review within 15 minutes",
                "MEDIUM": "Medium: analyst review within 1 hour",
                "LOW": "Low: manual deep inspection",
            },
            "z_near": 0.5,
            "z_far": 1.5,
            "suppress_near_average_implications": False,
        }
    }


# ---------------------------------------------------------------------------
# Test 1: Both modes contain five numbered headings and system sentence
# ---------------------------------------------------------------------------


def test_both_modes_contain_five_headings_and_system_sentence(
    sample_explanation: dict[str, Any],
    sample_chunks: list[dict[str, Any]],
    test_cfg: dict[str, Any],
) -> None:
    expected_system = (
        "You are a senior cybersecurity analyst. Generate a structured incident response report based on the evidence provided. "
        "Cite NIST Special Publications and MITRE ATT&CK techniques explicitly where applicable."
    )
    expected_headings = [
        "1. Rationale",
        "2. Key Indicators",
        "3. Confidence Assessment",
        "4. Threat Assessment",
        "5. Recommendations",
    ]

    for mode in ("rag", "vanilla"):
        res = build_prompt(sample_explanation, sample_chunks, mode=mode, cfg=test_cfg)
        assert res["system"] == expected_system
        for heading in expected_headings:
            assert heading in res["user"]


# ---------------------------------------------------------------------------
# Test 2: Vanilla empty field vs RAG numbered sources with fallback suffix
# ---------------------------------------------------------------------------


def test_vanilla_empty_and_rag_numbered_sources_with_fallback(
    sample_explanation: dict[str, Any],
    sample_chunks: list[dict[str, Any]],
    test_cfg: dict[str, Any],
) -> None:
    # Vanilla mode
    vanilla_res = build_prompt(sample_explanation, sample_chunks, mode="vanilla", cfg=test_cfg)
    assert "[1]" not in vanilla_res["user"]
    assert len(vanilla_res["sources"]) == 0
    match = re.search(r"\[RETRIEVED KNOWLEDGE CONTEXT\]\s*\[REPORT REQUEST\]", vanilla_res["user"])
    assert match is not None

    # RAG mode
    rag_res = build_prompt(sample_explanation, sample_chunks, mode="rag", cfg=test_cfg)
    assert "[1]" in rag_res["user"]
    assert "[2]" in rag_res["user"]
    assert "[3]" in rag_res["user"]
    # Updated grounding rule assertion — no longer checks old wording
    assert "never a range" in rag_res["user"]
    assert "do not add a reference list" in rag_res["user"]
    assert "NIST SP 800-61 Rev2 (general guidance)" in rag_res["user"]
    assert len(rag_res["sources"]) == 3
    assert rag_res["sources"][0]["n"] == 1
    assert rag_res["sources"][2]["is_fallback"] is True


# ---------------------------------------------------------------------------
# Test 3: Missing metadata prints 'not available'
# ---------------------------------------------------------------------------


def test_missing_metadata_prints_not_available(test_cfg: dict[str, Any]) -> None:
    exp = {
        "predicted_class": "Benign",
        "confidence": 0.5,
        "tier": "LOW",
        "features": [],
        "metadata": {
            "Flow ID": None,
            "Timestamp": float("nan"),
            "Src IP": "",
            "Src Port": None,
            "Dst IP": None,
            "Dst Port": float("nan"),
            "protocol": None,
        },
    }
    res = build_prompt(exp, None, mode="vanilla", cfg=test_cfg)
    user_text = res["user"]
    assert "Flow ID: not available" in user_text
    assert "Timestamp: not available" in user_text
    assert "Source: not available" in user_text
    assert "Destination: not available" in user_text
    assert "Protocol: not available" in user_text


# ---------------------------------------------------------------------------
# Test 4: position_word returns right word across thresholds
# ---------------------------------------------------------------------------


def test_position_word_thresholds(test_cfg: dict[str, Any]) -> None:
    assert position_word(0.0, test_cfg) == "near the training average"
    assert position_word(0.49, test_cfg) == "near the training average"
    assert position_word(0.5, test_cfg) == "moderately above the training average"
    assert position_word(1.49, test_cfg) == "moderately above the training average"
    assert position_word(1.5, test_cfg) == "far above the training average"
    assert position_word(-1.5, test_cfg) == "far below the training average"
    assert position_word(3.0, test_cfg) == "far above the training average"


# ---------------------------------------------------------------------------
# Test 5: format_value no scientific notation for 4.7e6 and <= 4 sig figs
# ---------------------------------------------------------------------------


def test_format_value_no_scientific_and_max_four_sig_figs() -> None:
    formatted_large = format_value(4.7e6)
    assert "e" not in formatted_large.lower()
    assert formatted_large == "4700000"

    assert format_value(123456.0) == "123500"
    assert format_value(0.0186) == "0.0186"
    assert format_value(123.456) == "123.5"
    assert format_value(0.0) == "0"


# ---------------------------------------------------------------------------
# Test 6: Private-use characters replaced
# ---------------------------------------------------------------------------


def test_private_use_characters_replacement(
    sample_explanation: dict[str, Any],
    test_cfg: dict[str, Any],
) -> None:
    chunks = [
        {
            "chunk_id": "c_priv",
            "text": "Heading\n\uE001 First item\n\uF000 Second item\n\uF8FF Third item",
            "source_name": "NIST_SP_800-94.pdf",
            "source_type": "NIST",
            "source_ref": "SP 800-94",
            "section": "Executive Summary",
            "is_fallback": False,
        }
    ]
    res = build_prompt(sample_explanation, chunks, mode="rag", cfg=test_cfg)
    user_text = res["user"]
    assert not any("\uE000" <= ch <= "\uF8FF" for ch in user_text)
    assert "- First item" in user_text


# ---------------------------------------------------------------------------
# Test 7: Benign explanation adds Benign paragraph; DoS does not
# ---------------------------------------------------------------------------


def test_benign_branch_handling(
    sample_explanation: dict[str, Any],
    test_cfg: dict[str, Any],
) -> None:
    benign_text = "No attack was detected. Say so in the Rationale"

    benign_exp = dict(sample_explanation)
    benign_exp["predicted_class"] = "Benign"
    benign_res = build_prompt(benign_exp, None, mode="vanilla", cfg=test_cfg)
    assert benign_text in benign_res["user"]

    dos_exp = dict(sample_explanation)
    dos_exp["predicted_class"] = "DoS"
    dos_res = build_prompt(dos_exp, None, mode="vanilla", cfg=test_cfg)
    assert benign_text not in dos_res["user"]


# ---------------------------------------------------------------------------
# Test 8: Developer notes never appear in prompt
# ---------------------------------------------------------------------------


def test_developer_notes_never_appear_in_prompt(
    sample_explanation: dict[str, Any],
    test_cfg: dict[str, Any],
) -> None:
    secret_note = "CONFIDENTIAL_DEV_NOTE_NEVER_LEAK_IN_PROMPT"
    exp = dict(sample_explanation)
    exp["features"] = [
        {
            "name": "Custom Feat",
            "scaled_value": 2.0,
            "original_value": 100.0,
            "importance": 5.0,
            "direction": "raises",
            "description": "Public feature description.",
            "implication": "Public security implication.",
            "note": secret_note,
        }
    ]
    for mode in ("rag", "vanilla"):
        res = build_prompt(exp, None, mode=mode, cfg=test_cfg)
        assert secret_note not in res["user"]
        assert secret_note not in res["system"]


# ---------------------------------------------------------------------------
# Test 9: Invalid mode raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_mode_raises_value_error(
    sample_explanation: dict[str, Any],
    test_cfg: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="Invalid mode"):
        build_prompt(sample_explanation, None, mode="invalid_mode", cfg=test_cfg)


# ---------------------------------------------------------------------------
# Test 10: MITRE bracketed numbers removal while keeping source headers
# ---------------------------------------------------------------------------


def test_mitre_bracketed_citation_removal(
    sample_explanation: dict[str, Any],
    test_cfg: dict[str, Any],
) -> None:
    input_text = "attack [4] , hacktivism, and extortion. [3] More"
    cleaned = clean_chunk_text(input_text)
    assert cleaned == "attack , hacktivism, and extortion. More"

    chunks = [
        {
            "chunk_id": "c_mitre",
            "text": input_text,
            "source_name": "T1498.html",
            "source_type": "MITRE",
            "source_ref": "T1498",
            "section": "Description",
            "is_fallback": False,
        }
    ]
    res = build_prompt(sample_explanation, chunks, mode="rag", cfg=test_cfg)
    user_text = res["user"]
    # Verify the source header is untouched
    assert "[1] MITRE ATT&CK T1498, section Description" in user_text
    # Verify the body text inside the chunk is cleaned
    assert "attack , hacktivism, and extortion. More" in user_text
    assert "[4]" not in user_text
    assert "[3]" not in user_text


# ---------------------------------------------------------------------------
# Test 11: Grounding rule wording
# ---------------------------------------------------------------------------


def test_grounding_rule_wording(
    sample_explanation: dict[str, Any],
    sample_chunks: list[dict[str, Any]],
    test_cfg: dict[str, Any],
) -> None:
    """Grounding rule must contain the two required phrases and must not contain literal '[n]'."""
    rag_res = build_prompt(sample_explanation, sample_chunks, mode="rag", cfg=test_cfg)
    user_text = rag_res["user"]
    assert "never a range" in user_text
    assert "do not add a reference list" in user_text
    # The literal placeholder "[n]" must not appear anywhere in the user prompt
    assert "[n]" not in user_text


# ---------------------------------------------------------------------------
# Test 12: Near-average implication suppression flag
# ---------------------------------------------------------------------------


def _make_flag_cfg(suppress: bool) -> dict[str, Any]:
    """Return a test config with the suppression flag set."""
    return {
        "llm": {
            "min_words": 10,
            "max_words": 2000,
            "tier_actions": {"VERY_HIGH": "Very High: fully automated response"},
            "z_near": 0.5,
            "z_far": 1.5,
            "suppress_near_average_implications": suppress,
        }
    }


def _make_two_feature_exp(z_near_val: float = 0.2, z_far_val: float = 2.0) -> dict[str, Any]:
    """Explanation with one near-average feature (|z| < 0.5) and one far feature (|z| >= 0.5)."""
    return {
        "predicted_class": "DoS",
        "confidence": 0.99,
        "tier": "VERY_HIGH",
        "features": [
            {
                "name": "Near Feature",
                "scaled_value": z_near_val,
                "original_value": 1.0,
                "importance": 1.0,
                "direction": "raises",
                "implication": "UNIQUE_NEAR_IMPLICATION_TEXT",
            },
            {
                "name": "Far Feature",
                "scaled_value": z_far_val,
                "original_value": 99.0,
                "importance": 3.0,
                "direction": "raises",
                "implication": "UNIQUE_FAR_IMPLICATION_TEXT",
            },
        ],
        "metadata": {},
    }


def test_near_average_suppression_flag_on(test_cfg: dict[str, Any]) -> None:
    """With flag on: near-average feature shows baseline sentence; far feature keeps its implication."""
    cfg_on = _make_flag_cfg(suppress=True)
    exp = _make_two_feature_exp()
    res = build_prompt(exp, None, mode="vanilla", cfg=cfg_on)
    user_text = res["user"]

    near_baseline = "This value is near the training average, so it does not by itself indicate an attack."
    assert near_baseline in user_text
    # The original near-feature implication must NOT appear
    assert "UNIQUE_NEAR_IMPLICATION_TEXT" not in user_text
    # The far-feature implication MUST still appear
    assert "UNIQUE_FAR_IMPLICATION_TEXT" in user_text


def test_near_average_suppression_flag_off(test_cfg: dict[str, Any]) -> None:
    """With flag off: both implications appear as-is; baseline sentence does NOT replace them."""
    cfg_off = _make_flag_cfg(suppress=False)
    exp = _make_two_feature_exp()
    res = build_prompt(exp, None, mode="vanilla", cfg=cfg_off)
    user_text = res["user"]

    # Both feature implications must appear unchanged
    assert "UNIQUE_NEAR_IMPLICATION_TEXT" in user_text
    assert "UNIQUE_FAR_IMPLICATION_TEXT" in user_text
    # The baseline replacement sentence must NOT appear for the near-average feature
    near_baseline = "This value is near the training average, so it does not by itself indicate an attack."
    assert near_baseline not in user_text


def test_near_average_suppression_applies_in_rag_mode(
    sample_chunks: list[dict[str, Any]],
) -> None:
    """Suppression also applies in rag mode (same behaviour in both modes)."""
    cfg_on = _make_flag_cfg(suppress=True)
    exp = _make_two_feature_exp()
    res = build_prompt(exp, sample_chunks, mode="rag", cfg=cfg_on)
    user_text = res["user"]

    near_baseline = "This value is near the training average, so it does not by itself indicate an attack."
    assert near_baseline in user_text
    assert "UNIQUE_NEAR_IMPLICATION_TEXT" not in user_text
    assert "UNIQUE_FAR_IMPLICATION_TEXT" in user_text
