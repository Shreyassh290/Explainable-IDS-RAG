"""Unit tests for FEATURE_INFO registry coverage, structural integrity, and prompt safety."""

import json
import re
from pathlib import Path

import pytest

from ids_rag.config import get_project_root
from ids_rag.explain.features import (
    FEATURE_INFO,
    check_coverage,
    describe,
    describe_for_prompt,
)


def _load_feature_names() -> list[str]:
    """Load canonical feature name list from models/feature_names.json."""
    path = get_project_root() / "models" / "feature_names.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


FEATURE_NAMES = _load_feature_names()


def test_every_feature_name_has_an_entry() -> None:
    """Every name in feature_names.json must exist in FEATURE_INFO."""
    missing = [name for name in FEATURE_NAMES if name not in FEATURE_INFO]
    assert missing == [], f"Features missing from FEATURE_INFO: {missing}"


def test_no_extra_entries_in_registry() -> None:
    """FEATURE_INFO must not contain names absent from feature_names.json."""
    names_set = set(FEATURE_NAMES)
    extra = [key for key in FEATURE_INFO if key not in names_set]
    assert extra == [], f"Extra entries in FEATURE_INFO not in feature_names.json: {extra}"


def test_every_entry_has_non_empty_description_and_implication() -> None:
    """Every FEATURE_INFO entry must have non-empty description and implication strings."""
    for name in FEATURE_NAMES:
        entry = FEATURE_INFO[name]
        assert "description" in entry, f"'description' missing for '{name}'"
        assert "implication" in entry, f"'implication' missing for '{name}'"
        assert isinstance(entry["description"], str) and entry["description"].strip(), (
            f"Empty description for '{name}'"
        )
        assert isinstance(entry["implication"], str) and entry["implication"].strip(), (
            f"Empty implication for '{name}'"
        )


def test_describe_returns_correct_entry() -> None:
    """describe() should return the same dict as direct FEATURE_INFO lookup."""
    for name in FEATURE_NAMES:
        result = describe(name)
        assert result == FEATURE_INFO[name], f"describe('{name}') returned wrong entry"


def test_describe_for_prompt_returns_only_prompt_fields() -> None:
    """describe_for_prompt() must return only 'description' and 'implication'."""
    for name in FEATURE_NAMES:
        prompt_dict = describe_for_prompt(name)
        assert set(prompt_dict.keys()) == {"description", "implication"}, (
            f"describe_for_prompt('{name}') contains keys other than description/implication: {prompt_dict.keys()}"
        )
        assert prompt_dict["description"] == FEATURE_INFO[name]["description"]
        assert prompt_dict["implication"] == FEATURE_INFO[name]["implication"]


def test_describe_raises_key_error_for_unknown_name() -> None:
    """describe() must raise KeyError for a name not in the registry."""
    with pytest.raises(KeyError, match="not in FEATURE_INFO"):
        describe("NonExistentFeature_XYZ")


def test_describe_for_prompt_raises_key_error_for_unknown_name() -> None:
    """describe_for_prompt() must raise KeyError for an unknown feature."""
    with pytest.raises(KeyError, match="not in FEATURE_INFO"):
        describe_for_prompt("NonExistentFeature_XYZ")


def test_check_coverage_returns_empty_for_exact_match() -> None:
    """check_coverage should return ([], []) when input exactly matches FEATURE_INFO keys."""
    missing, extra = check_coverage(FEATURE_NAMES)
    assert missing == [], f"Unexpected missing: {missing}"
    assert extra == [], f"Unexpected extra: {extra}"


def test_check_coverage_detects_missing_and_extra() -> None:
    """check_coverage should correctly report missing and extra names."""
    partial_names = FEATURE_NAMES[:5] + ["FakeFeature"]
    missing, extra = check_coverage(partial_names)
    assert "FakeFeature" in missing
    assert len(extra) == len(FEATURE_NAMES) - 5


def test_no_banned_phrases_in_descriptions_and_implications() -> None:
    """Descriptions and implications must not contain analyst/statistical phrases or UNCERTAIN."""
    banned_patterns = [
        re.compile(r"\bUNCERTAIN\b", re.IGNORECASE),
        re.compile(r"\bmedian\b", re.IGNORECASE),
        re.compile(r"\bin this dataset\b", re.IGNORECASE),
    ]

    for name in FEATURE_NAMES:
        desc = FEATURE_INFO[name]["description"]
        imp = FEATURE_INFO[name]["implication"]
        for pat in banned_patterns:
            assert not pat.search(desc), f"Feature '{name}' description contains banned pattern '{pat.pattern}': {desc}"
            assert not pat.search(imp), f"Feature '{name}' implication contains banned pattern '{pat.pattern}': {imp}"


def test_ascii_only_in_features_file() -> None:
    """The features.py source file must contain only ASCII characters."""
    features_py_path = (
        get_project_root() / "src" / "ids_rag" / "explain" / "features.py"
    )
    with open(features_py_path, "rb") as f:
        content = f.read()
    # Attempt decoding strictly as ASCII
    try:
        content.decode("ascii")
    except UnicodeDecodeError as exc:
        pytest.fail(f"Non-ASCII character found in features.py: {exc}")


def test_implications_under_20_words() -> None:
    """Every feature implication must be under 20 words, with sentences under 15 words."""
    for name in FEATURE_NAMES:
        imp = FEATURE_INFO[name]["implication"]
        word_count = len(imp.split())
        assert word_count <= 20, f"Feature '{name}' implication has {word_count} words (limit is <=20): '{imp}'"
        # Check per-sentence length (split by semicolon or period)
        sentences = [s.strip() for s in re.split(r"[;.]", imp) if s.strip()]
        for s in sentences:
            s_words = len(s.split())
            assert s_words <= 15, f"Feature '{name}' sentence has {s_words} words (limit is <=15): '{s}'"
