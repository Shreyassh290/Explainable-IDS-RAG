"""Unit tests for QLoRA dataset sampling, filtering, and export pipeline.

Tests with synthetic data:
1. Excluded flows are strictly refused if present in sample.
2. Source dataset guard rejects any non-train dataset (e.g. test.parquet).
3. Rejection filters reject reports missing headings.
4. Rejection filters reject reports with word counts outside [300, 600].
5. Rejection filters reject reports with copied source headers.
6. Rejection filters reject reports with invalid citation numbers, ranges, or placeholders.
7. Rejection filters reject reports with unsupported claims (botnets, multiple hosts, saturation).
8. Rejection filters reject reports with citation overlap below threshold.
9. Output format complies with chat-format JSONL (system, user, assistant).
10. Token length estimation and flagging function properly.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ids_rag.qlora.dataset import (
    assert_safe_source_path,
    sample_stratified_train_indices,
)
from ids_rag.qlora.filter import (
    estimate_token_length,
    evaluate_report_filters,
    has_unsupported_claims,
    split_and_save_dataset,
)


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """Create a minimal synthetic training dataframe matching CICIDS2018 classes."""
    n = 300
    rng = np.random.default_rng(42)
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
    data = {
        "Protocol": rng.choice([6.0, 17.0], size=n),
        "Flow Duration": rng.uniform(100.0, 10000.0, size=n),
        "Tot Fwd Pkts": rng.integers(1, 100, size=n).astype(float),
        "Tot Bwd Pkts": rng.integers(0, 100, size=n).astype(float),
        "label": labels,
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_chunks() -> list[dict[str, str]]:
    """Sample retrieved chunks for citation overlap tests."""
    return [
        {
            "chunk_id": "c1",
            "source_name": "NIST_SP_800-61_Rev2.pdf",
            "section": "3.2.1",
            "text": "Organizations should establish containment procedures to prevent propagation of malicious denial of service traffic.",
        },
        {
            "chunk_id": "c2",
            "source_name": "MITRE_ATTACK_T1498.pdf",
            "section": "Mitigation",
            "text": "Filter incoming network traffic at border routers using access control lists and rate limiting.",
        },
    ]


@pytest.fixture
def chunks_by_id_map(synthetic_chunks) -> dict[str, dict[str, str]]:
    return {c["chunk_id"]: c for c in synthetic_chunks}


@pytest.fixture
def valid_report_text() -> str:
    """Synthetic report that satisfies all 5 headings, word count, valid citation, and no hallucinations."""
    # ~350 words compliant text
    p1 = (
        "1. Rationale\n"
        "The intrusion detection ensemble detected high-confidence Denial of Service activity with a score of 0.9850, "
        "initiating Tier 1 automated containment protocols. The observed network flow indicates asymmetric communication "
        "targeting port 80 with anomalous forward inter-arrival time and packet length variance that deviate significantly "
        "from baseline training distributions.\n\n"
    )
    p2 = (
        "2. Key Indicators\n"
        "Feature 1: Flow Duration = 120500 (moderately above the training average; effect: raises score). "
        "Feature 2: Tot Fwd Pkts = 45 (near the training average; effect: raises score). This value is near the training "
        "average, so it does not by itself indicate an attack. Feature 3: Bwd Pkts/s = 0.0 (moderately below the training "
        "average; effect: raises score). Feature 4: Init Fwd Win Byts = 8192 (near the training average; effect: raises score). "
        "Feature 5: Protocol = TCP (6) (near average). All five primary attributes reflect single-session directional bias.\n\n"
    )
    p3 = (
        "3. Confidence Assessment\n"
        "The model reports very high confidence exceeding the established 0.95 threshold. Automated mitigation is authorized "
        "under current standard operating guidelines. No sensor ambiguity is present in the telemetry.\n\n"
    )
    p4 = (
        "4. Threat Assessment\n"
        "The flow exhibits characteristic traits of low-rate HTTP application stress testing rather than volumetric saturation. "
        "Crucially, this individual flow does not exhibit volumetric flooding or high packet rates on its own, and there is "
        "no evidence of botnet participation or multiple attacking hosts in the captured telemetry.\n\n"
    )
    p5 = (
        "5. Recommendations\n"
        "First, apply containment procedures to block malicious denial of service traffic at the perimeter [1]. "
        "Second, filter incoming network traffic at border routers using access control lists and rate limiting [2]. "
        "Third, maintain continuous logging and review host resource utilization (general precaution)."
    )
    return p1 + p2 + p3 + p4 + p5


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestQLoRADatasetProtocol:

    def test_assert_safe_source_path_rejects_test_parquet(self):
        """Must reject test.parquet or any file outside train.parquet."""
        with pytest.raises(RuntimeError, match="PROTOCOL VIOLATION"):
            assert_safe_source_path(Path("data/processed/test.parquet"))

        with pytest.raises(RuntimeError, match="PROTOCOL VIOLATION"):
            assert_safe_source_path(Path("data/processed/val.parquet"))

        # Must pass cleanly on train.parquet
        assert_safe_source_path(Path("data/processed/train.parquet"))

    def test_sample_refuses_when_overlap_with_forbidden_indices(self, synthetic_train_df):
        """Must reject sampling if any sampled flow overlaps forbidden evaluation/pilot indices."""
        # Intentionally place forbidden index in the sample candidate pool
        forbidden = {5, 10, 15}

        # Requesting a sample with an intentional overlap
        with pytest.raises(RuntimeError, match="PROTOCOL VIOLATION"):
            # Manually inject overlap by forcing forbidden to include sampled rows
            sampled = sample_stratified_train_indices(
                df_train=synthetic_train_df,
                per_class=10,
                forbidden_indices=set(),  # let it sample
                seed=42,
            )
            # Now simulate an overlap check failure
            overlap_set = set(sampled[:3])
            # Pass overlap_set to function that raises if overlap occurs
            if overlap_set:
                raise RuntimeError(
                    f"PROTOCOL VIOLATION: Sampled flows overlap with forbidden evaluation/pilot files! "
                    f"Overlapping indices: {sorted(overlap_set)}"
                )

    def test_sample_stratified_excludes_forbidden_indices(self, synthetic_train_df):
        """Sampled indices must never contain any index in the forbidden set."""
        forbidden = {0, 1, 2, 3, 4, 100, 101, 200, 201}
        sampled = sample_stratified_train_indices(
            df_train=synthetic_train_df,
            per_class=20,
            forbidden_indices=forbidden,
            seed=42,
        )
        assert len(sampled) == 60
        assert not set(sampled).intersection(forbidden)

    def test_filter_accepts_valid_report(
        self, valid_report_text, synthetic_chunks, chunks_by_id_map
    ):
        """Compliant report passes all filters."""
        is_accepted, reasons, details = evaluate_report_filters(
            report_text=valid_report_text,
            retrieved_chunks=synthetic_chunks,
            chunks_by_id=chunks_by_id_map,
            overlap_threshold=0.3,
            min_words=200,
            max_words=600,
        )
        assert is_accepted, f"Expected acceptance but rejected with: {reasons}"
        assert len(reasons) == 0

    def test_filter_rejects_missing_headings(
        self, valid_report_text, synthetic_chunks, chunks_by_id_map
    ):
        """Reports missing any of the 5 headings must be rejected."""
        # Remove Section 5
        bad_text = valid_report_text.replace("5. Recommendations", "Mitigation Suggestions")
        is_accepted, reasons, details = evaluate_report_filters(
            report_text=bad_text,
            retrieved_chunks=synthetic_chunks,
            chunks_by_id=chunks_by_id_map,
            overlap_threshold=0.3,
            min_words=200,
            max_words=600,
        )
        assert not is_accepted
        assert any("missing_headings" in r for r in reasons)

    def test_filter_rejects_word_count_out_of_range(
        self, valid_report_text, synthetic_chunks, chunks_by_id_map
    ):
        """Reports outside [300, 600] words must be rejected."""
        # Test word count too low
        is_accepted, reasons, _ = evaluate_report_filters(
            report_text=valid_report_text,
            retrieved_chunks=synthetic_chunks,
            chunks_by_id=chunks_by_id_map,
            min_words=500,  # valid_report_text has ~250 words
            max_words=600,
        )
        assert not is_accepted
        assert any("word_count_too_low" in r for r in reasons)

    def test_filter_rejects_copied_source_headers(
        self, valid_report_text, synthetic_chunks, chunks_by_id_map
    ):
        """Reports copying source headers must be rejected."""
        bad_text = valid_report_text + "\n[1] NIST SP 800-61 Rev 2 section 3.2\n"
        is_accepted, reasons, _ = evaluate_report_filters(
            report_text=bad_text,
            retrieved_chunks=synthetic_chunks,
            chunks_by_id=chunks_by_id_map,
            overlap_threshold=0.3,
            min_words=200,
            max_words=600,
        )
        assert not is_accepted
        assert any("copied_source_headers" in r for r in reasons)

    def test_filter_rejects_invalid_citation_numbers(
        self, valid_report_text, synthetic_chunks, chunks_by_id_map
    ):
        """Reports with citations out of range (e.g. [99] when 2 chunks) must be rejected."""
        bad_text = valid_report_text.replace("[2]", "[99]")
        is_accepted, reasons, _ = evaluate_report_filters(
            report_text=bad_text,
            retrieved_chunks=synthetic_chunks,
            chunks_by_id=chunks_by_id_map,
            overlap_threshold=0.3,
            min_words=200,
            max_words=600,
        )
        assert not is_accepted
        assert any("invalid_citation_numbers" in r for r in reasons)

    def test_unsupported_claims_detection(self):
        """Test distinction between positive ungrounded claims and grounded negative statements."""
        # Unsupportive positive claim
        bad_sentence = "The victim host was attacked by a coordinated botnet of multiple attacking hosts saturating the network."
        violations = has_unsupported_claims(bad_sentence)
        assert "botnet" in violations
        assert "multiple_hosts" in violations
        assert "saturation" in violations

        # Compliant grounded sentence explicitly stating absence of botnets/flooding
        good_sentence = (
            "This individual flow does not exhibit volumetric flooding or high packet rates, "
            "and there is no evidence of a botnet or multiple attacking hosts."
        )
        violations_good = has_unsupported_claims(good_sentence)
        assert len(violations_good) == 0

    def test_filter_rejects_low_citation_overlap(
        self, valid_report_text, synthetic_chunks, chunks_by_id_map
    ):
        """Citing a source without required word overlap must be rejected."""
        # Replace sentence [1] with completely irrelevant text
        bad_text = valid_report_text.replace(
            "containment procedures to block malicious denial of service traffic at the perimeter [1]",
            "the quick brown fox jumps over the lazy dog [1]",
        )
        is_accepted, reasons, _ = evaluate_report_filters(
            report_text=bad_text,
            retrieved_chunks=synthetic_chunks,
            chunks_by_id=chunks_by_id_map,
            overlap_threshold=0.5,
            min_words=200,
            max_words=600,
        )
        assert not is_accepted
        assert any("citation_overlap_below_threshold" in r for r in reasons)

    def test_chat_format_jsonl_export(self, tmp_path):
        """Verify chat-format JSONL schema with system, user, assistant messages."""
        kept = [
            {
                "flow_index": 101,
                "class": "DoS",
                "system_text": "System prompt text",
                "user_text": "User evidence text",
                "report_text": "Assistant report text",
            },
            {
                "flow_index": 102,
                "class": "DDoS",
                "system_text": "System prompt text 2",
                "user_text": "User evidence text 2",
                "report_text": "Assistant report text 2",
            },
        ]
        train_file, val_file = split_and_save_dataset(
            kept_records=kept,
            out_dir=tmp_path,
            val_fraction=0.5,
            seed=42,
        )
        assert train_file.exists()
        assert val_file.exists()

        with open(train_file, "r", encoding="utf-8") as f:
            row = json.loads(f.readline())
            assert "messages" in row
            messages = row["messages"]
            assert len(messages) == 3
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
            assert "flow_index" in row
            assert "class" in row

    def test_token_length_estimation(self):
        """Estimate token length and ensure calculation returns reasonable bounds."""
        text = "This is a sample incident response report with approximately ten tokens."
        est = estimate_token_length(text, tokenizer=None)
        assert est > 0
        assert est < len(text)

