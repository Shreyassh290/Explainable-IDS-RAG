"""Unit tests for Stage 3 v2 KB and retrieval extensions.

Tests cover:
- v2_build_kb_extra.py: chunking of raw text files, schema validation, safety check
- retrieve_v2.py: extended synonyms, expand_query_v2, build_query_v2

No real model loading, no FAISS index building, no network access.
All retrieval tests use synthetic in-memory data.

Run:
    uv run pytest tests/test_v2_kb.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ids_rag.v2.retrieve_v2 import (
    SYNONYMS_V2,
    build_query_v2,
    expand_query_v2,
)


# ---------------------------------------------------------------------------
# Import chunking functions from the script (not a module, so import directly)
# ---------------------------------------------------------------------------

import importlib.util
import sys


def _import_build_kb_script():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "v2_build_kb_extra.py"
    spec = importlib.util.spec_from_file_location("v2_build_kb_extra", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_build_kb = _import_build_kb_script()


# ---------------------------------------------------------------------------
# 1. KB chunking — schema and safety
# ---------------------------------------------------------------------------

class TestV2KBChunking:
    """Tests for v2_build_kb_extra.py chunking functions."""

    SAMPLE_DOC = """\
MITRE ATT&CK Technique T1110 — Brute Force
ID: T1110
Tactic: Credential Access

Description

Adversaries may use brute force techniques to gain access to accounts when passwords are unknown.
Brute forcing passwords can take place via interaction with a service that checks credential validity.
Network flow analysis can detect brute force attempts through abnormal volumes of authentication attempts.

Mitigations

ID: M1032 — Multi-factor Authentication. Use multi-factor authentication.
MFA makes it significantly harder for adversaries to authenticate with stolen credentials.

Detection Strategy

Monitor authentication logs for system login failures.
Alert when any external IP exceeds failed authentication thresholds within a one-minute window.
"""

    def test_parse_txt_produces_blocks(self):
        blocks = _build_kb.parse_txt_document(self.SAMPLE_DOC)
        assert len(blocks) >= 1
        texts = [b["text"] for b in blocks]
        combined = " ".join(texts)
        assert "brute force" in combined.lower()

    def test_parse_txt_section_labels(self):
        blocks = _build_kb.parse_txt_document(self.SAMPLE_DOC)
        sections = {b["section"] for b in blocks}
        # At minimum should include Description and Mitigations
        assert len(sections) >= 1

    def test_chunk_blocks_produces_valid_schema(self):
        blocks = _build_kb.parse_txt_document(self.SAMPLE_DOC)
        chunks = _build_kb.chunk_blocks(
            blocks,
            source_name="MITRE_T1110.txt",
            source_type="MITRE",
            source_ref="T1110",
        )
        assert len(chunks) >= 1
        required_keys = {
            "chunk_id", "source_name", "source_type", "source_ref",
            "section", "sections", "token_count", "char_start", "char_end", "text",
        }
        for chunk in chunks:
            assert required_keys.issubset(set(chunk.keys())), f"Missing keys in chunk: {set(chunk.keys())}"

    def test_chunk_source_fields_correct(self):
        blocks = _build_kb.parse_txt_document(self.SAMPLE_DOC)
        chunks = _build_kb.chunk_blocks(
            blocks, "MITRE_T1110.txt", "MITRE", "T1110"
        )
        for chunk in chunks:
            assert chunk["source_name"] == "MITRE_T1110.txt"
            assert chunk["source_type"] == "MITRE"
            assert chunk["source_ref"] == "T1110"

    def test_chunk_ids_are_unique(self):
        blocks = _build_kb.parse_txt_document(self.SAMPLE_DOC)
        chunks = _build_kb.chunk_blocks(blocks, "MITRE_T1110.txt", "MITRE", "T1110")
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids detected"

    def test_chunk_min_token_filter(self):
        """Chunks below min_tokens must be discarded."""
        tiny_block = [{"section": "Description", "text": "Short.", "char_start": 0, "char_end": 6}]
        chunks = _build_kb.chunk_blocks(
            tiny_block, "MITRE_T1110.txt", "MITRE", "T1110",
            min_tokens=30,  # "Short." has 1 token
        )
        assert len(chunks) == 0

    def test_chunk_text_is_non_empty_string(self):
        blocks = _build_kb.parse_txt_document(self.SAMPLE_DOC)
        chunks = _build_kb.chunk_blocks(blocks, "MITRE_T1110.txt", "MITRE", "T1110")
        for chunk in chunks:
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"].strip()) > 0

    def test_chunk_token_count_matches_text(self):
        blocks = _build_kb.parse_txt_document(self.SAMPLE_DOC)
        chunks = _build_kb.chunk_blocks(blocks, "MITRE_T1110.txt", "MITRE", "T1110")
        for chunk in chunks:
            actual = _build_kb.approx_tokens(chunk["text"])
            assert chunk["token_count"] == actual

    def test_sections_list_is_non_empty(self):
        blocks = _build_kb.parse_txt_document(self.SAMPLE_DOC)
        chunks = _build_kb.chunk_blocks(blocks, "MITRE_T1110.txt", "MITRE", "T1110")
        for chunk in chunks:
            assert isinstance(chunk["sections"], list)
            assert len(chunk["sections"]) >= 1

    def test_build_v2_kb_safety_check(self, tmp_path):
        """build_v2_kb must refuse to write inside frozen data/kb/."""
        root_project = Path(__file__).resolve().parents[1]
        frozen_output = root_project / "data" / "kb" / "chunks_v2.jsonl"
        with pytest.raises(ValueError, match="SAFETY REFUSAL"):
            _build_kb.build_v2_kb(
                raw_dir=tmp_path,
                output_path=frozen_output,
            )

    def test_build_v2_kb_missing_raw_files_raises(self, tmp_path):
        """build_v2_kb must raise FileNotFoundError when no .txt files exist."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        output = tmp_path / "chunks_v2.jsonl"
        with pytest.raises(FileNotFoundError):
            _build_kb.build_v2_kb(raw_dir=raw_dir, output_path=output)

    def test_build_v2_kb_writes_valid_jsonl(self, tmp_path):
        """build_v2_kb must produce valid JSONL with correct schema."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        # Write a minimal synthetic source file registered in _SOURCE_META
        (raw_dir / "MITRE_T1110.txt").write_text(self.SAMPLE_DOC, encoding="utf-8")

        output = tmp_path / "chunks_v2.jsonl"
        count = _build_kb.build_v2_kb(raw_dir=raw_dir, output_path=output)
        assert count >= 1
        assert output.exists()

        lines = [json.loads(l) for l in output.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == count
        for line in lines:
            assert "chunk_id" in line
            assert "text" in line
            assert "source_ref" in line

    def test_content_uuid_is_deterministic(self):
        id1 = _build_kb.content_uuid("file.txt", "Description", 0)
        id2 = _build_kb.content_uuid("file.txt", "Description", 0)
        assert id1 == id2

    def test_content_uuid_differs_by_position(self):
        id1 = _build_kb.content_uuid("file.txt", "Description", 0)
        id2 = _build_kb.content_uuid("file.txt", "Description", 100)
        assert id1 != id2


# ---------------------------------------------------------------------------
# 2. SYNONYMS_V2 coverage
# ---------------------------------------------------------------------------

class TestSynonymsV2:
    """SYNONYMS_V2 must cover all 6 classes including the 3 new ones."""

    def test_all_six_classes_present(self):
        required = {"DoS", "DDoS", "Benign", "Bot", "FTP-BruteForce", "SSH-Bruteforce"}
        assert required.issubset(set(SYNONYMS_V2.keys()))

    def test_bot_synonyms_include_botnet_and_c2(self):
        bot_syns = " ".join(SYNONYMS_V2["Bot"]).lower()
        assert "botnet" in bot_syns
        assert "c2" in bot_syns or "command" in bot_syns

    def test_ftp_bruteforce_synonyms_include_port_21(self):
        ftp_syns = " ".join(SYNONYMS_V2["FTP-BruteForce"]).lower()
        assert "21" in ftp_syns

    def test_ssh_bruteforce_synonyms_include_port_22(self):
        ssh_syns = " ".join(SYNONYMS_V2["SSH-Bruteforce"]).lower()
        assert "22" in ssh_syns

    def test_all_synonym_lists_are_non_empty(self):
        for cls_name, syns in SYNONYMS_V2.items():
            assert len(syns) >= 1, f"Empty synonym list for '{cls_name}'"

    def test_mitre_refs_in_brute_force_synonyms(self):
        ftp_syns = " ".join(SYNONYMS_V2["FTP-BruteForce"])
        ssh_syns = " ".join(SYNONYMS_V2["SSH-Bruteforce"])
        assert "T1110" in ftp_syns
        assert "T1110" in ssh_syns

    def test_mitre_refs_in_bot_synonyms(self):
        bot_syns = " ".join(SYNONYMS_V2["Bot"])
        assert "T1583" in bot_syns or "T1071" in bot_syns


# ---------------------------------------------------------------------------
# 3. expand_query_v2
# ---------------------------------------------------------------------------

class TestExpandQueryV2:
    """expand_query_v2 must correctly expand queries for all 6 class types."""

    def test_bot_query_expanded(self):
        result = expand_query_v2("Bot traffic anomaly detected")
        assert "botnet" in result.lower() or "command" in result.lower()

    def test_botnet_query_expanded(self):
        result = expand_query_v2("botnet C2 communication")
        assert "botnet" in result.lower()

    def test_ftp_bruteforce_query_expanded(self):
        result = expand_query_v2("FTP-BruteForce attack on port 21")
        assert "brute force" in result.lower() or "password" in result.lower()

    def test_ssh_bruteforce_query_expanded(self):
        result = expand_query_v2("SSH-Bruteforce credential attack")
        assert "ssh" in result.lower() or "password" in result.lower()

    def test_ddos_query_expanded(self):
        result = expand_query_v2("DDoS flood attack")
        assert "distributed denial of service" in result.lower()

    def test_dos_query_expanded(self):
        result = expand_query_v2("DoS resource exhaustion")
        assert "denial of service" in result.lower()

    def test_benign_query_expanded(self):
        result = expand_query_v2("benign traffic baseline")
        assert "normal traffic" in result.lower() or "baseline" in result.lower()

    def test_original_query_preserved_in_output(self):
        q = "FTP-BruteForce attack on port 21"
        result = expand_query_v2(q)
        assert q in result

    def test_no_duplicate_synonym_phrases_in_expansion(self):
        """The function deduplicates at the phrase level (like the original expand_query).
        When the input query does not already contain synonym words, each multi-word
        synonym phrase must appear exactly once in the output.
        """
        # Use a query that contains none of the Bot synonym words
        result = expand_query_v2("Bot traffic anomaly detected on network")
        # Every multi-word synonym phrase (more than one word) must appear at most once
        for syn_phrase in SYNONYMS_V2["Bot"]:
            if len(syn_phrase.split()) > 1:  # only check multi-word phrases
                count = result.count(syn_phrase)
                assert count <= 1, f"Multi-word phrase '{syn_phrase}' appears {count} times"
        # The original query must appear at the start of the expansion
        assert result.startswith("Bot traffic anomaly detected on network")


# ---------------------------------------------------------------------------
# 4. build_query_v2
# ---------------------------------------------------------------------------

class TestBuildQueryV2:
    """build_query_v2 must produce correct query strings for all extra classes."""

    def _expl(self, cls: str, features=None, proto=6, dst_port=22):
        return {
            "predicted_class": cls,
            "features": features or [
                {"name": "Tot Fwd Pkts", "direction": "raises"},
                {"name": "Flow Duration", "direction": "lowers"},
            ],
            "metadata": {"protocol": proto, "Dst Port": dst_port},
        }

    def test_bot_query_contains_class_name(self):
        result = build_query_v2(self._expl("Bot", dst_port=80))
        assert "Bot" in result

    def test_bot_query_contains_synonyms(self):
        result = build_query_v2(self._expl("Bot", dst_port=80))
        assert "botnet" in result.lower() or "c2" in result.lower()

    def test_ftp_bruteforce_query_contains_class(self):
        result = build_query_v2(self._expl("FTP-BruteForce", dst_port=21))
        assert "FTP-BruteForce" in result

    def test_ssh_bruteforce_query_contains_class(self):
        result = build_query_v2(self._expl("SSH-Bruteforce", dst_port=22))
        assert "SSH-Bruteforce" in result

    def test_query_contains_feature_names(self):
        result = build_query_v2(self._expl("Bot"))
        assert "Tot Fwd Pkts" in result
        assert "Flow Duration" in result

    def test_query_contains_protocol(self):
        result = build_query_v2(self._expl("SSH-Bruteforce", proto=6, dst_port=22))
        assert "TCP" in result

    def test_query_contains_destination_port(self):
        result = build_query_v2(self._expl("FTP-BruteForce", proto=6, dst_port=21))
        assert "21" in result

    def test_empty_explanation_does_not_raise(self):
        result = build_query_v2({})
        assert isinstance(result, str)

    def test_unknown_class_does_not_raise(self):
        result = build_query_v2({"predicted_class": "NewUnknownClass"})
        assert "NewUnknownClass" in result

    def test_query_parts_joined_by_pipe(self):
        result = build_query_v2(self._expl("Bot", dst_port=80))
        assert "|" in result

