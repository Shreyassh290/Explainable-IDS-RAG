"""Knowledge base text splitting and token-window chunking.

Loaders:  PDF (pypdf), HTML (stdlib html.parser), .txt, .md.
Chunking: heading/paragraph-aware; overlap between consecutive chunks of the
          same section; token counts via the embedding model's tokenizer.
Output:   data/kb/chunks.jsonl  (one JSON object per line).

NIST PDF Profiles:
------------------
1. NIST SP 800-61 Rev 2:
   - TOC parsing for canonical section titles.
   - Front-matter before Executive Summary dropped.
   - Section 1.1 Authority dropped.
   - Non-content pages (TOC, LoF, LoT) dropped.
   - Appendix D (acronyms) and Appendix H (change log) dropped on purpose.
   - Footnote numbers stripped; unique pairs logged to artifacts/logs.

2. NIST SP 800-94:
   - Front matter before Executive Summary dropped (first page after TOC/LoF
     whose first content line is 'Executive Summary').
   - Running header stripped ('GUIDE TO INTRUSION DETECTION AND PREVENTION SYSTEM...').
   - Page labels stripped ('ES-4', '1-3', '2-12', 'A-1').
   - Blank page notice stripped ('This page has been left blank intentionally.').
   - TOC parsed with dot leaders and page fragments ('...2-4') stripped.
   - Robust normalization via normalize_heading_key() on both TOC and body lines.
   - Sequential heading matching in TOC order, with support for 2-line wrapped headings.
   - Sections 1 to 9 and Appendix A kept; Appendices B, C, D dropped by section label.
   - Footnote numbers stripped; unique pairs logged to artifacts/logs.

3. NIST SP 800-30 Rev 1:
   - Three header lines stripped on every page (Special Publication 800-30...,
     underscores, page label).
   - Body starts at 'CHAPTER 1  PAGE 1' (physical page 10); front matter dropped.
   - Body ends before first page with label starting 'APPENDIX'; all appendices dropped.
   - Uppercase TOC parsed to title-case section labels ('Chapter 1: Introduction', '1.1 Purpose and Applicability').
   - Sequential heading matching in TOC order.
   - Drop-cap repair for chapter initial paragraphs.
   - Footnote numbers stripped; unique pairs logged to artifacts/logs.

MITRE ATT&CK HTML cleaning:
---------------------------
- Removes page-end navigation block starting at standalone '×' / 'Core ATT&CK Objects'.
- Removes 'Version Permalink Live Version', 'Contributors:' line, and info symbol ⓘ.
- Drops the References section completely (bibliography entries).
- Section label before first real heading is 'Description' (not Sub-techniques).
- ID, Tactic, Platforms, and Impact Type kept as plain text in Description.

Post-processing & Merging:
--------------------------
- Drop chunks under 30 tokens containing no complete sentence.
- A chunk under 100 tokens may be merged ONLY:
  (a) with another chunk of the same section, or
  (b) when it is a parent heading's intro text and next chunk is first child section
      (e.g. 2.3 into 2.3.1, 3.1 into 3.1.1). Merged chunk uses the child's label.
- No other merges (1.3 stays separate from 1.2, 4.1.2 separate from 4.2).
- MITRE chunks are never merged across sections.
- Each chunk contains a 'sections' list of section labels it contains, and
  'section' is the label of the section the chunk text starts in.
"""

import json
import random
import re
import unicodedata
import uuid
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ids_rag.config import get_project_root, load_config

# ---------------------------------------------------------------------------
# Lazy tokenizer singleton (loaded once from config, never the full model)
# ---------------------------------------------------------------------------

_tokenizer: Any = None


def _get_tokenizer() -> Any:
    """Return a cached tokenizer instance for token-count estimation."""
    global _tokenizer
    if _tokenizer is None:
        import transformers
        from transformers import AutoTokenizer

        transformers.logging.set_verbosity_error()
        config = load_config()
        model_name = config.get("rag", {}).get(
            "embed_model", "sentence-transformers/all-mpnet-base-v2"
        )
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using the embedding tokenizer."""
    tok = _get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False, verbose=False))


# ---------------------------------------------------------------------------
# Heading Key Normalization
# ---------------------------------------------------------------------------

_DASH_VARIANTS_RE = re.compile(r"[\u00ad\u2010-\u2015\u2212]")
_LEADING_SEC_NUM_DOT_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.*)$")
_DASH_SPACES_RE = re.compile(r"\s*-\s*")


def normalize_heading_key(s: str) -> str:
    """Normalize heading text for robust TOC and body matching.

    - unicodedata.normalize("NFKC", s)
    - replace dash variants (U+2010 to U+2015, U+2212, U+00AD, and U+2011) with '-'
    - collapse whitespace (including U+00A0) to single spaces
    - casefold
    - remove period directly after leading section number ('1. introduction' -> '1 introduction')
    - remove spaces around a dash ('appendix a - glossary' -> 'appendix a-glossary')
    """
    s = unicodedata.normalize("NFKC", s)
    s = _DASH_VARIANTS_RE.sub("-", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.casefold()
    m = _LEADING_SEC_NUM_DOT_RE.match(s)
    if m:
        s = f"{m.group(1)} {m.group(2)}"
    s = _DASH_SPACES_RE.sub("-", s)
    return s


# ---------------------------------------------------------------------------
# Footnote stripping (applies to ALL PDFs)
# ---------------------------------------------------------------------------

# Glued footnote numbers: word ending in at least 3 letters (possibly capitalized),
# followed by optional punctuation [.,;:], followed by 1 or 2 digits, at word boundary.
_FOOTNOTE_RE = re.compile(r"\b([A-Za-z]?[a-z]{3,}[.,;:]?)\d{1,2}(?=\s|$)")


def remove_footnotes(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Remove glued footnote numbers from words (e.g. 'incidents.5' -> 'incidents.').

    Leaves untouched: 'IPv6', 'sha256', 'SP 800-61', '802.11', 'Section 3.2', 'T1498.001'.
    Returns (cleaned_text, list_of_(before, after)_samples).
    """
    samples: list[tuple[str, str]] = []

    def _repl(m: re.Match) -> str:
        full = m.group(0)
        replacement = m.group(1)
        samples.append((full, replacement))
        return replacement

    cleaned = _FOOTNOTE_RE.sub(_repl, text)
    return cleaned, samples


# ---------------------------------------------------------------------------
# Title-casing helper for uppercase headings (SP 800-30)
# ---------------------------------------------------------------------------

_LOWER_WORDS: frozenset[str] = frozenset(
    {"and", "of", "the", "for", "to", "in", "on", "a", "an"}
)
_UPPER_WORDS: frozenset[str] = frozenset({"nist", "it", "sp"})


def title_case_heading(title: str) -> str:
    """Convert uppercase titles to title case.

    Lowercase for: and, of, the, for, to, in, on, a, an (except as first word).
    Keep uppercase: NIST, IT, SP.
    """
    words = title.strip().split()
    if not words:
        return ""
    result: list[str] = []
    for i, w in enumerate(words):
        m = re.match(r"^([A-Za-z0-9]+)(.*)$", w)
        if m:
            core, punct = m.group(1), m.group(2)
            core_lower = core.lower()
            if core_lower in _UPPER_WORDS:
                core_cased = core_lower.upper()
            elif i > 0 and core_lower in _LOWER_WORDS:
                core_cased = core_lower
            else:
                core_cased = core_lower.capitalize()
            result.append(core_cased + punct)
        else:
            result.append(w)
    return " ".join(result)


# ---------------------------------------------------------------------------
# Drop-cap repair helper (SP 800-30)
# ---------------------------------------------------------------------------

def repair_drop_caps(lines: list[str]) -> tuple[list[str], int, int]:
    """Repair drop caps on a page.

    Rule: If a page contains a line consisting of a single uppercase letter
    and the first body paragraph after the heading block starts with a lowercase letter,
    prepend that letter to the paragraph and remove the stray line.

    Returns (repaired_lines, repairs_done_count, repairs_not_done_count).
    """
    single_upper_indices = [
        i for i, ln in enumerate(lines) if re.match(r"^[A-Z]$", ln.strip())
    ]

    candidate_body_idx = None
    for i, ln in enumerate(lines):
        ln_s = ln.strip()
        if i in single_upper_indices:
            continue
        # Body line candidate: at least 3 words and starts with lowercase
        if re.match(r"^[a-z]", ln_s) and len(ln_s.split()) >= 3:
            candidate_body_idx = i
            break

    if candidate_body_idx is not None and len(single_upper_indices) == 1:
        u_idx = single_upper_indices[0]
        letter = lines[u_idx].strip()
        repaired = list(lines)
        repaired[candidate_body_idx] = letter + repaired[candidate_body_idx].strip()
        repaired.pop(u_idx)
        return repaired, 1, 0

    if candidate_body_idx is not None:
        return lines, 0, 1

    return lines, 0, 0


# ---------------------------------------------------------------------------
# HTML parser – visible text only
# ---------------------------------------------------------------------------

_SKIP_TAGS: frozenset[str] = frozenset(
    {
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "head",
        "noscript",
        "iframe",
        "button",
        "svg",
        "form",
        "template",
        "aside",
    }
)

_BLOCK_TAGS: frozenset[str] = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "li",
        "tr",
        "td",
        "th",
        "blockquote",
        "pre",
        "br",
        "hr",
    }
)

_HEADING_TAGS: frozenset[str] = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


class _VisibleTextParser(HTMLParser):
    """Extract visible text and headings from HTML, discarding JS/CSS/nav/footer/header."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_stack: list[str] = []
        self.segments: list[dict[str, Any]] = []
        self._current_heading: str = ""
        self._current_level: int = 0
        self._in_heading: bool = False
        self._pending_text: list[str] = []

    def _flush_pending(self) -> None:
        raw = " ".join(self._pending_text).strip()
        raw = re.sub(r"\s+", " ", raw)
        if raw:
            self.segments.append(
                {"is_heading": False, "heading_level": 0, "text": raw}
            )
        self._pending_text = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "") or ""
        role = attrs_dict.get("role", "") or ""
        if (
            tag_lower in _SKIP_TAGS
            or "breadcrumb" in classes.lower()
            or role.lower() == "navigation"
        ):
            self._skip_stack.append(tag_lower)
            return

        if self._skip_stack:
            return

        if tag_lower in _HEADING_TAGS:
            self._flush_pending()
            self._in_heading = True
            self._current_heading = ""
            self._current_level = int(tag_lower[1])
        elif tag_lower in _BLOCK_TAGS:
            self._flush_pending()

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if self._skip_stack:
            if tag_lower == self._skip_stack[-1]:
                self._skip_stack.pop()
            elif tag_lower in self._skip_stack:
                while self._skip_stack and self._skip_stack[-1] != tag_lower:
                    self._skip_stack.pop()
                if self._skip_stack:
                    self._skip_stack.pop()
            return

        if tag_lower in _HEADING_TAGS:
            heading_text = re.sub(r"\s+", " ", self._current_heading).strip()
            if heading_text:
                self.segments.append(
                    {
                        "is_heading": True,
                        "heading_level": self._current_level,
                        "text": heading_text,
                    }
                )
            self._in_heading = False
            self._current_heading = ""
            self._current_level = 0
        elif tag_lower in _BLOCK_TAGS:
            self._flush_pending()

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        if self._in_heading:
            self._current_heading += data
        else:
            stripped = data.strip()
            if stripped:
                self._pending_text.append(stripped)

    def close(self) -> None:
        super().close()
        self._flush_pending()


def _parse_html(raw_html: str) -> list[dict[str, Any]]:
    """Return list of segment dicts from raw HTML string."""
    parser = _VisibleTextParser()
    parser.feed(raw_html)
    parser.close()
    return parser.segments


# ---------------------------------------------------------------------------
# MITRE HTML cleaner
# ---------------------------------------------------------------------------

_REAL_MITRE_HEADINGS: frozenset[str] = frozenset(
    {"procedure examples", "mitigations", "detection strategy"}
)


def clean_mitre_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean extracted segments from MITRE ATT&CK HTML pages."""
    cleaned: list[dict[str, Any]] = []
    in_references = False
    in_nav_end = False
    seen_first_real_heading = False

    for s in segments:
        txt = s["text"].strip()
        if not txt:
            continue

        # Page-end navigation block
        if txt in ("×", "x", "\xd7") or "Core ATT&CK Objects" in txt:
            in_nav_end = True
            break
        if in_nav_end:
            break

        # Drop References section
        if s["is_heading"] and txt.lower() == "references":
            in_references = True
            continue
        if in_references:
            if s["is_heading"] and txt.lower() != "references":
                in_references = False
            else:
                continue

        # Remove metadata boilerplate lines and symbols
        if txt in (
            "Version Permalink",
            "Live Version",
            "Version Permalink Live Version",
            "ⓘ",
            "\u24d8",
        ):
            continue
        if txt.startswith("Contributors:"):
            continue

        # Strip inline symbol and metadata strings
        clean_txt = txt.replace("ⓘ", "").replace("\u24d8", "")
        clean_txt = clean_txt.replace("Version Permalink", "").replace("Live Version", "").strip()
        clean_txt = re.sub(r"\s+", " ", clean_txt).strip()
        if not clean_txt:
            continue

        # Section labeling
        if s["is_heading"]:
            if txt.lower() in _REAL_MITRE_HEADINGS:
                seen_first_real_heading = True
                cleaned.append({"is_heading": True, "heading_level": s["heading_level"], "text": txt})
            else:
                if not seen_first_real_heading:
                    if not any(x["is_heading"] and x["text"] == "Description" for x in cleaned):
                        cleaned.append({"is_heading": True, "heading_level": 1, "text": "Description"})
                    cleaned.append({"is_heading": False, "heading_level": 0, "text": clean_txt})
                else:
                    cleaned.append({"is_heading": True, "heading_level": s["heading_level"], "text": txt})
        else:
            if not seen_first_real_heading and not any(x["is_heading"] and x["text"] == "Description" for x in cleaned):
                cleaned.append({"is_heading": True, "heading_level": 1, "text": "Description"})
            cleaned.append({"is_heading": False, "heading_level": 0, "text": clean_txt})

    return cleaned


# ---------------------------------------------------------------------------
# Common PDF line detection helpers
# ---------------------------------------------------------------------------

_DOT_LEADER_RE = re.compile(r"\.{4,}|(?:\.\s*){4,}")
_PAGE_NUMBER_RE = re.compile(r"^\s*[-\u2013\u2014]?\s*\d{1,4}\s*[-\u2013\u2014]?\s*$")

_NIST_HEADING_LINE_RE = re.compile(
    r"^("
    r"(?:\d{1,2}(?:\.\d+)*\.?)"
    r"|Executive Summary"
    r"|Appendix\s+[A-H](?:[—\-–\s].*)?"
    r")"
    r"(?:\s+(.*))?$"
)

_TOC_LINE_RE = re.compile(
    r"^(.+?)"
    r"\s+(?:\.{4,}|(?:\.\s*){4,})"
    r".*?(\d{1,4})\s*$",
    re.MULTILINE,
)

_SEC_LABEL_RE = re.compile(
    r"^("
    r"(?:\d{1,2}(?:\.\d+)*\.?)"
    r"|Executive Summary"
    r"|Appendix\s+[A-H](?:\s*[—\-–]\s*\S+(?:\s+\S+){0,8})?"
    r")\s*(.*)",
    re.DOTALL,
)

_SKIP_PAGE_PHRASES: tuple[str, ...] = (
    "Table of Contents",
    "List of Appendices",
    "List of Tables",
    "List of Figures",
)


def is_dot_leader_line(line: str) -> bool:
    """Return True if *line* contains dot leaders (common in TOCs)."""
    return bool(_DOT_LEADER_RE.search(line))


def _normalise_sec_num(raw: str) -> str:
    """Normalise section numbers (e.g. '3.2.' -> '3.2')."""
    s = raw.strip()
    if re.match(r"\d+\.\d", s):
        s = s.rstrip(".")
    return s


def detect_heading(
    line: str, toc: dict[str, str] | None = None
) -> tuple[bool, int, str]:
    """Detect if *line* is a section heading in NIST SP 800-61 style."""
    line_s = line.strip()
    if not line_s:
        return False, 0, ""

    if line_s == "Executive Summary":
        return True, 1, "Executive Summary"

    m = _NIST_HEADING_LINE_RE.match(line_s)
    if not m:
        return False, 0, ""

    prefix = m.group(1).strip()
    rest = (m.group(2) or "").strip()

    # Appendix headings
    if prefix.startswith("Appendix"):
        if toc:
            app_id = prefix.split()[0] + " " + prefix.split()[1][:1]
            for k, v in toc.items():
                if k.startswith(app_id):
                    return True, 1, v
        return True, 1, line_s

    # Numbered heading: e.g. "3.2.4 Incident Analysis", "3. Handling an Incident"
    if rest and rest[0].isupper() and not rest.endswith("."):
        num = _normalise_sec_num(prefix)
        if toc:
            toc_title = toc.get(num) or toc.get(num + ".")
            if toc_title and rest.lower().startswith(toc_title.lower()[:12]):
                level = len(num.rstrip(".").split("."))
                return True, level, f"{num} {toc_title}"
            return False, 0, ""

        if len(line_s) < 100:
            level = len(num.rstrip(".").split("."))
            clean_title = rest.rstrip(".")
            return True, level, f"{num} {clean_title}"

    return False, 0, ""


def _extract_toc(pages_raw: list[str]) -> dict[str, str]:
    """Parse TOC pages for SP 800-61 into section number -> title."""
    toc: dict[str, str] = {}
    for page_text in pages_raw:
        if not _DOT_LEADER_RE.search(page_text):
            continue
        for line in page_text.splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            m = _TOC_LINE_RE.match(line_s)
            if not m:
                continue
            pre_dots = m.group(1).strip().rstrip(". ")
            m2 = _SEC_LABEL_RE.match(pre_dots)
            if not m2:
                continue
            sec_num = m2.group(1).strip()
            title = m2.group(2).strip().rstrip(". ")
            sec_num = re.sub(r"\s*[—–]\s*", "—", sec_num)
            if sec_num == "Executive Summary":
                toc["Executive Summary"] = "Executive Summary"
            elif sec_num.startswith("Appendix"):
                display = sec_num if not title else f"{sec_num} {title}"
                toc[sec_num] = display
            else:
                if title:
                    toc[sec_num] = title
    return toc


# ---------------------------------------------------------------------------
# SP 800-94 Line & Header Helpers
# ---------------------------------------------------------------------------

_SP800_94_HEADER_PREFIX = "GUIDE TO INTRUSION DETECTION AND PREVENTION SYSTEM"
_SP800_94_PAGE_LABEL_RE = re.compile(r"^(ES|[A-D]|\d{1,2})-\d{1,3}$")
_BLANK_PAGE_NOTICE = "this page has been left blank intentionally."


def is_sp800_94_header(line: str) -> bool:
    """Return True if line matches the running header for SP 800-94."""
    return line.strip().upper().startswith(_SP800_94_HEADER_PREFIX)


def is_sp800_94_page_label(line: str) -> bool:
    """Return True if line matches the page label pattern for SP 800-94 (e.g. 'ES-4', '1-3')."""
    return bool(_SP800_94_PAGE_LABEL_RE.match(line.strip()))


def _format_sp800_94_label(title_part: str) -> str:
    """Format canonical display label for SP 800-94."""
    title_part = re.sub(r"\s+", " ", title_part).strip()
    if title_part.lower() == "executive summary":
        return "Executive Summary"
    if title_part.lower().startswith("appendix"):
        m = re.match(r"^Appendix\s+([A-D])\s*[—–\-]\s*(.*)$", title_part, re.IGNORECASE)
        if m:
            app_letter = m.group(1).upper()
            app_title = m.group(2).strip()
            return f"Appendix {app_letter}—{app_title}"
        return title_part

    m_num = re.match(r"^(\d+(?:\.\d+)*)(\.?)\s+(.*)$", title_part)
    if m_num:
        sec_num = m_num.group(1)
        rest = m_num.group(3).strip()
        if "." not in sec_num:
            return f"{sec_num}. {rest}"
        else:
            return f"{sec_num} {rest}"
    return title_part


def _extract_toc_sp800_94(pages_raw: list[str]) -> list[str]:
    """Extract canonical TOC entries for SP 800-94.

    Strips dot leaders and trailing page fragments (e.g. '...2-4', '...ES-1').
    Returns an ordered list of canonical section labels.
    """
    entries: list[str] = []
    seen_keys: set[str] = set()

    for p_text in pages_raw:
        if not _DOT_LEADER_RE.search(p_text):
            continue
        lines = p_text.splitlines()
        for idx, line in enumerate(lines):
            line_s = line.strip()
            if not _DOT_LEADER_RE.search(line_s):
                continue
            parts = _DOT_LEADER_RE.split(line_s)
            if not parts:
                continue
            title_part = parts[0].strip().rstrip(". ")
            if not title_part:
                continue

            # Check if this line is continuation of a wrapped TOC line
            if not (title_part.lower().startswith("executive") or
                    title_part.lower().startswith("appendix") or
                    re.match(r"^\d", title_part)):
                prev_idx = idx - 1
                while prev_idx >= 0 and not lines[prev_idx].strip():
                    prev_idx -= 1
                if prev_idx >= 0 and not _DOT_LEADER_RE.search(lines[prev_idx]):
                    prev_txt = lines[prev_idx].strip()
                    if (prev_txt.lower().startswith("executive") or
                        prev_txt.lower().startswith("appendix") or
                        re.match(r"^\d", prev_txt)):
                        title_part = prev_txt + " " + title_part

            if not (title_part.lower().startswith("executive") or
                    title_part.lower().startswith("appendix") or
                    re.match(r"^\d", title_part)):
                continue

            canonical = _format_sp800_94_label(title_part)
            key = normalize_heading_key(canonical)
            if key not in seen_keys:
                seen_keys.add(key)
                entries.append(canonical)

    return entries


# ---------------------------------------------------------------------------
# SP 800-30 Rev 1 Line & Header Helpers
# ---------------------------------------------------------------------------

_SP800_30_PAGE_LABEL_RE = re.compile(
    r"^(PAGE\s+[ivxlc]+|CHAPTER\s+\d+\s+PAGE\s+\d+|APPENDIX\s+[A-Z]\s+PAGE\s+[A-Z]-\d+)$",
    re.IGNORECASE,
)
_SP800_30_UNDERSCORES_RE = re.compile(r"^_{5,}$")
_SP800_30_HEADER_START = "Special Publication 800-30"


def is_sp800_30_header(line: str) -> bool:
    """Return True if line starts with 'Special Publication 800-30'."""
    return line.strip().startswith(_SP800_30_HEADER_START)


def is_sp800_30_page_label(line: str) -> bool:
    """Return True if line matches SP 800-30 page label pattern."""
    return bool(_SP800_30_PAGE_LABEL_RE.match(line.strip()))


_WORD_TO_NUM: dict[str, str] = {"ONE": "1", "TWO": "2", "THREE": "3"}


def _extract_toc_sp800_30(pages_raw: list[str]) -> list[tuple[str, str, Any]]:
    """Extract canonical TOC entries for SP 800-30 Rev 1 chapters 1 to 3 and sections."""
    entries: list[tuple[str, str, Any]] = []

    toc_text = ""
    for p_text in pages_raw:
        if _DOT_LEADER_RE.search(p_text) and "CHAPTER ONE" in p_text:
            toc_text = p_text
            break

    if not toc_text:
        return entries

    for line in toc_text.splitlines():
        line_s = line.strip()
        if not _DOT_LEADER_RE.search(line_s):
            continue
        parts = _DOT_LEADER_RE.split(line_s)
        if not parts:
            continue
        raw_title = parts[0].strip().rstrip(". ")
        raw_norm = re.sub(r"\s+", " ", raw_title)

        m_ch = re.match(r"^CHAPTER\s+(ONE|TWO|THREE)\s+(.+)$", raw_norm, re.IGNORECASE)
        if m_ch:
            word_num = m_ch.group(1).upper()
            ch_num = _WORD_TO_NUM.get(word_num, word_num)
            ch_title = title_case_heading(m_ch.group(2))
            canonical = f"Chapter {ch_num}: {ch_title}"
            spec = (f"CHAPTER {word_num}", m_ch.group(2).strip().upper())
            entries.append((canonical, "chapter", spec))
            continue

        m_sec = re.match(r"^([1-3]\.\d+(?:\.\d+)*)\s+(.+)$", raw_norm)
        if m_sec:
            sec_num = m_sec.group(1)
            sec_title = title_case_heading(m_sec.group(2))
            canonical = f"{sec_num} {sec_title}"
            spec = f"{sec_num} {re.sub(r's+', ' ', m_sec.group(2).strip().upper())}"
            entries.append((canonical, "section", spec))

    return entries


# ---------------------------------------------------------------------------
# PDF Loaders per Profile
# ---------------------------------------------------------------------------

PDF_LOAD_STATS: dict[str, dict[str, Any]] = {}


def _load_pdf_sp800_61(path: Path) -> list[dict[str, Any]]:
    """Loader for NIST SP 800-61 Rev 2."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages_raw: list[str] = [page.extract_text() or "" for page in reader.pages]

    toc = _extract_toc(pages_raw)

    first_lines: list[str] = []
    last_lines: list[str] = []
    for text in pages_raw:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        first_lines.append(lines[0] if lines else "")
        last_lines.append(lines[-1] if lines else "")

    c1 = Counter(first_lines)
    c2 = Counter(last_lines)
    repeated = {ln for ln, c in (c1 | c2).items() if c >= 3 and ln}

    segments: list[dict[str, Any]] = []
    dropped_pages_by_rule: Counter[str] = Counter()
    stripped_lines_by_rule: Counter[str] = Counter()
    footnote_samples: list[tuple[str, str]] = []

    seen_executive_summary = False
    in_dropped_section = False
    in_appendix_d = False
    matched_toc: set[str] = set()

    for p_idx, page_text in enumerate(pages_raw):
        if _DOT_LEADER_RE.search(page_text) and any(
            phrase in page_text for phrase in _SKIP_PAGE_PHRASES
        ):
            dropped_pages_by_rule["toc_lof_lot"] += 1
            continue

        if "Appendix D—Acronyms" in page_text or ("Appendix D" in page_text and "Selected acronyms" in page_text):
            in_appendix_d = True
        if in_appendix_d:
            if "Appendix E—Resources" in page_text or "Appendix E" in page_text:
                in_appendix_d = False
            else:
                dropped_pages_by_rule["acronyms_appendix_d"] += 1
                continue

        if "Appendix H—Change Log" in page_text or ("Appendix H" in page_text and "Change Log" in page_text):
            dropped_pages_by_rule["change_log_appendix_h"] += 1
            continue

        cur_para: list[str] = []

        def flush() -> None:
            if cur_para and not in_dropped_section:
                p_text = re.sub(r"\s+", " ", " ".join(cur_para)).strip()
                if p_text:
                    p_text_clean, fn_s = remove_footnotes(p_text)
                    footnote_samples.extend(fn_s)
                    segments.append({"is_heading": False, "heading_level": 0, "text": p_text_clean})
            cur_para.clear()

        for line in page_text.splitlines():
            line_s = line.strip()
            if not line_s:
                flush()
                continue
            if _PAGE_NUMBER_RE.match(line_s):
                stripped_lines_by_rule["page_numbers"] += 1
                continue
            if line_s in repeated:
                stripped_lines_by_rule["repeated_headers_footers"] += 1
                continue
            if is_dot_leader_line(line_s):
                stripped_lines_by_rule["dot_leader_lines"] += 1
                continue

            is_h, lvl, label = detect_heading(line_s, toc=toc)
            if is_h:
                if label == "Executive Summary":
                    seen_executive_summary = True

                if not seen_executive_summary:
                    dropped_pages_by_rule["front_matter_lines"] += 1
                    continue

                flush()

                if label == "1.1 Authority":
                    in_dropped_section = True
                    continue
                else:
                    in_dropped_section = False

                matched_toc.add(label)
                segments.append({"is_heading": True, "heading_level": lvl, "text": label})
            else:
                if not seen_executive_summary:
                    continue
                if in_dropped_section:
                    continue
                cur_para.append(line_s)

        flush()

    total_kept = len(pages_raw) - sum(
        v for k, v in dropped_pages_by_rule.items() if not k.endswith("_lines")
    )

    pair_counts = Counter(footnote_samples)
    log_dir = get_project_root() / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: (-x[1], x[0]))
    log_content = "\n".join(f"{b} -> {a} ({cnt})" for (b, a), cnt in sorted_pairs)
    (log_dir / f"footnote_subs_{path.stem}.txt").write_text(log_content, encoding="utf-8")
    (log_dir / f"footnote_subs_{path.name}.txt").write_text(log_content, encoding="utf-8")

    # Drop Appendix D and Appendix H on purpose
    dropped_on_purpose = [
        v for k, v in toc.items()
        if k.startswith("Appendix D") or k.startswith("Appendix H")
    ]
    not_matched = [
        v for k, v in toc.items()
        if v not in matched_toc and v not in dropped_on_purpose
    ]

    PDF_LOAD_STATS[path.name] = {
        "filename": path.name,
        "total_pages": len(pages_raw),
        "pages_kept": total_kept,
        "pages_dropped_by_rule": dict(dropped_pages_by_rule),
        "toc_entries_parsed": len(toc),
        "toc_entries_matched": len(matched_toc),
        "toc_entries_not_matched": not_matched,
        "toc_entries_dropped_on_purpose": dropped_on_purpose,
        "stripped_lines_by_rule": dict(stripped_lines_by_rule),
        "footnote_substitutions": len(footnote_samples),
        "footnote_unique_pairs": len(pair_counts),
        "footnote_samples": footnote_samples[:10],
        "drop_cap_repairs_done": 0,
        "drop_cap_repairs_not_done": 0,
    }

    return segments


def _load_pdf_sp800_94(path: Path) -> list[dict[str, Any]]:
    """Loader for NIST SP 800-94 (127 pages)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages_raw: list[str] = [page.extract_text() or "" for page in reader.pages]

    toc_entries = _extract_toc_sp800_94(pages_raw)
    toc_keys = [normalize_heading_key(e) for e in toc_entries]

    dropped_pages_by_rule: Counter[str] = Counter()
    stripped_lines_by_rule: Counter[str] = Counter()
    footnote_samples: list[tuple[str, str]] = []

    last_toc_page_idx = 0
    for idx, p_text in enumerate(pages_raw):
        if _DOT_LEADER_RE.search(p_text):
            last_toc_page_idx = idx

    body_start_idx = 0
    for idx in range(last_toc_page_idx + 1, len(pages_raw)):
        lines = [ln.strip() for ln in pages_raw[idx].splitlines() if ln.strip()]
        content_lines = [
            ln for ln in lines
            if not is_sp800_94_header(ln)
            and not is_sp800_94_page_label(ln)
            and ln.lower() != _BLANK_PAGE_NOTICE
        ]
        if content_lines and normalize_heading_key(content_lines[0]) == "executive summary":
            body_start_idx = idx
            break

    dropped_pages_by_rule["front_matter_before_exec_summary"] = body_start_idx

    segments: list[dict[str, Any]] = []
    matched_toc: list[str] = []
    toc_idx = 0
    in_dropped_section = False
    in_dropped_appendix = False

    for idx in range(body_start_idx, len(pages_raw)):
        page_text = pages_raw[idx]
        lines = page_text.splitlines()

        cur_para: list[str] = []

        def flush() -> None:
            if cur_para and not in_dropped_section and not in_dropped_appendix:
                p_text = re.sub(r"\s+", " ", " ".join(cur_para)).strip()
                if p_text:
                    p_text_clean, fn_s = remove_footnotes(p_text)
                    footnote_samples.extend(fn_s)
                    segments.append({"is_heading": False, "heading_level": 0, "text": p_text_clean})
            cur_para.clear()

        line_i = 0
        while line_i < len(lines):
            line_s = lines[line_i].strip()
            if not line_s:
                flush()
                line_i += 1
                continue

            if is_sp800_94_header(line_s):
                stripped_lines_by_rule["running_header"] += 1
                line_i += 1
                continue
            if is_sp800_94_page_label(line_s):
                stripped_lines_by_rule["page_label"] += 1
                line_i += 1
                continue
            if line_s.lower() == _BLANK_PAGE_NOTICE:
                stripped_lines_by_rule["blank_page_notice"] += 1
                line_i += 1
                continue
            if is_dot_leader_line(line_s):
                stripped_lines_by_rule["dot_leader_lines"] += 1
                line_i += 1
                continue

            is_h = False
            heading_label = ""
            lines_consumed = 1

            if toc_idx < len(toc_keys):
                target_key = toc_keys[toc_idx]
                cand_key = normalize_heading_key(line_s)

                if cand_key == target_key:
                    is_h = True
                    heading_label = toc_entries[toc_idx]
                    lines_consumed = 1
                elif target_key.startswith(cand_key):
                    # Check if following non-empty line completes wrapped heading
                    next_i = line_i + 1
                    while next_i < len(lines):
                        next_s = lines[next_i].strip()
                        if not next_s:
                            next_i += 1
                            continue
                        if is_sp800_94_header(next_s) or is_sp800_94_page_label(next_s):
                            next_i += 1
                            continue
                        break
                    if next_i < len(lines):
                        next_s = lines[next_i].strip()
                        joined_key = normalize_heading_key(line_s + " " + next_s)
                        if joined_key == target_key:
                            is_h = True
                            heading_label = toc_entries[toc_idx]
                            lines_consumed = (next_i - line_i) + 1

            if is_h:
                flush()
                matched_toc.append(heading_label)
                toc_idx += 1

                # Drop section "1.1 Authority" by label (statutory boilerplate)
                if normalize_heading_key(heading_label) == "1.1 authority" or heading_label.startswith("1.1 Authority"):
                    in_dropped_section = True
                    in_dropped_appendix = False
                # Drop Appendix B, C, D completely by section label
                elif (
                    heading_label.startswith("Appendix B")
                    or heading_label.startswith("Appendix C")
                    or heading_label.startswith("Appendix D")
                ):
                    in_dropped_appendix = True
                    in_dropped_section = False
                else:
                    in_dropped_section = False
                    in_dropped_appendix = False
                    m_num = re.match(r"^(\d+(?:\.\d+)*)", heading_label)
                    lvl = len(m_num.group(1).split(".")) if m_num else 1
                    segments.append({"is_heading": True, "heading_level": lvl, "text": heading_label})

                line_i += lines_consumed
            else:
                if not in_dropped_section and not in_dropped_appendix:
                    cur_para.append(line_s)
                line_i += 1

        flush()

    total_kept = len(pages_raw) - sum(dropped_pages_by_rule.values())

    pair_counts = Counter(footnote_samples)
    log_dir = get_project_root() / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: (-x[1], x[0]))
    log_content = "\n".join(f"{b} -> {a} ({cnt})" for (b, a), cnt in sorted_pairs)
    (log_dir / f"footnote_subs_{path.stem}.txt").write_text(log_content, encoding="utf-8")
    (log_dir / f"footnote_subs_{path.name}.txt").write_text(log_content, encoding="utf-8")

    not_matched = [e for e in toc_entries if e not in matched_toc]

    PDF_LOAD_STATS[path.name] = {
        "filename": path.name,
        "total_pages": len(pages_raw),
        "pages_kept": total_kept,
        "pages_dropped_by_rule": dict(dropped_pages_by_rule),
        "toc_entries_parsed": len(toc_entries),
        "toc_entries_matched": len(matched_toc),
        "toc_entries_not_matched": not_matched,
        "toc_entries_dropped_on_purpose": [],
        "stripped_lines_by_rule": dict(stripped_lines_by_rule),
        "footnote_substitutions": len(footnote_samples),
        "footnote_unique_pairs": len(pair_counts),
        "footnote_samples": footnote_samples[:10],
        "drop_cap_repairs_done": 0,
        "drop_cap_repairs_not_done": 0,
    }

    return segments


def _load_pdf_sp800_30(path: Path) -> list[dict[str, Any]]:
    """Loader for NIST SP 800-30 Rev 1 (95 pages)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages_raw: list[str] = [page.extract_text() or "" for page in reader.pages]

    toc_entries = _extract_toc_sp800_30(pages_raw)

    dropped_pages_by_rule: Counter[str] = Counter()
    stripped_lines_by_rule: Counter[str] = Counter()
    footnote_samples: list[tuple[str, str]] = []
    drop_cap_done = 0
    drop_cap_not_done = 0

    body_start_idx = None
    body_end_idx = None

    for idx, p_text in enumerate(pages_raw):
        lines = [ln.strip() for ln in p_text.splitlines() if ln.strip()]
        for ln in lines:
            if re.match(r"^CHAPTER\s+1\s+PAGE\s+1$", ln, re.IGNORECASE):
                if body_start_idx is None:
                    body_start_idx = idx
            if re.match(r"^APPENDIX\s+[A-Z]\s+PAGE\s+", ln, re.IGNORECASE):
                if body_end_idx is None:
                    body_end_idx = idx

    if body_start_idx is None:
        body_start_idx = 9
    if body_end_idx is None:
        body_end_idx = len(pages_raw)

    dropped_pages_by_rule["front_matter_before_ch1_p1"] = body_start_idx
    dropped_pages_by_rule["appendices_dropped"] = len(pages_raw) - body_end_idx

    segments: list[dict[str, Any]] = []
    matched_toc: set[str] = set()
    toc_idx = 0

    for idx in range(body_start_idx, body_end_idx):
        lines = pages_raw[idx].splitlines()

        clean_lines: list[str] = []
        for ln in lines:
            ln_s = ln.strip()
            if is_sp800_30_header(ln_s):
                stripped_lines_by_rule["header_special_pub_line"] += 1
                continue
            if _SP800_30_UNDERSCORES_RE.match(ln_s):
                stripped_lines_by_rule["header_underscore_line"] += 1
                continue
            if is_sp800_30_page_label(ln_s):
                stripped_lines_by_rule["header_page_label"] += 1
                continue
            clean_lines.append(ln)

        repaired_lines, done, not_done = repair_drop_caps(clean_lines)
        drop_cap_done += done
        drop_cap_not_done += not_done

        cur_para: list[str] = []

        def flush() -> None:
            if cur_para:
                p_text = re.sub(r"\s+", " ", " ".join(cur_para)).strip()
                if p_text:
                    p_text_clean, fn_s = remove_footnotes(p_text)
                    footnote_samples.extend(fn_s)
                    segments.append({"is_heading": False, "heading_level": 0, "text": p_text_clean})
            cur_para.clear()

        line_i = 0
        while line_i < len(repaired_lines):
            line = repaired_lines[line_i]
            line_s = line.strip()
            if not line_s:
                flush()
                line_i += 1
                continue

            if is_dot_leader_line(line_s):
                stripped_lines_by_rule["dot_leader_lines"] += 1
                line_i += 1
                continue

            is_h = False
            heading_label = ""
            heading_lvl = 1
            advance_step = 1

            if toc_idx < len(toc_entries):
                target_label, match_type, search_spec = toc_entries[toc_idx]
                norm_upper = re.sub(r"\s+", " ", line_s).upper()

                if match_type == "chapter":
                    ch_line1, ch_line2 = search_spec
                    if norm_upper == ch_line1:
                        if line_i + 1 < len(repaired_lines):
                            next_upper = re.sub(r"\s+", " ", repaired_lines[line_i + 1].strip()).upper()
                            if next_upper == ch_line2 or ch_line2 in next_upper:
                                is_h = True
                                heading_label = target_label
                                heading_lvl = 1
                                advance_step = 2
                                matched_toc.add(target_label)
                                toc_idx += 1
                elif match_type == "section":
                    if norm_upper == search_spec or norm_upper.startswith(search_spec):
                        is_h = True
                        heading_label = target_label
                        m_num = re.match(r"^(\d+(?:\.\d+)*)", target_label)
                        heading_lvl = len(m_num.group(1).split(".")) if m_num else 2
                        matched_toc.add(target_label)
                        toc_idx += 1

            if is_h:
                flush()
                segments.append({"is_heading": True, "heading_level": heading_lvl, "text": heading_label})
                line_i += advance_step
            else:
                cur_para.append(line_s)
                line_i += 1

        flush()

    total_kept = body_end_idx - body_start_idx

    pair_counts = Counter(footnote_samples)
    log_dir = get_project_root() / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: (-x[1], x[0]))
    log_content = "\n".join(f"{b} -> {a} ({cnt})" for (b, a), cnt in sorted_pairs)
    (log_dir / f"footnote_subs_{path.stem}.txt").write_text(log_content, encoding="utf-8")
    (log_dir / f"footnote_subs_{path.name}.txt").write_text(log_content, encoding="utf-8")

    not_matched = [lbl for lbl, _, _ in toc_entries if lbl not in matched_toc]

    PDF_LOAD_STATS[path.name] = {
        "filename": path.name,
        "total_pages": len(pages_raw),
        "pages_kept": total_kept,
        "pages_dropped_by_rule": dict(dropped_pages_by_rule),
        "toc_entries_parsed": len(toc_entries),
        "toc_entries_matched": len(matched_toc),
        "toc_entries_not_matched": not_matched,
        "toc_entries_dropped_on_purpose": [],
        "stripped_lines_by_rule": dict(stripped_lines_by_rule),
        "footnote_substitutions": len(footnote_samples),
        "footnote_unique_pairs": len(pair_counts),
        "footnote_samples": footnote_samples[:10],
        "drop_cap_repairs_done": drop_cap_done,
        "drop_cap_repairs_not_done": drop_cap_not_done,
    }

    return segments


def _load_pdf(path: Path) -> list[dict[str, Any]]:
    """Dispatch PDF loading to the appropriate PDF profile based on file name."""
    name = path.name
    if "800-94" in name:
        return _load_pdf_sp800_94(path)
    elif "800-30" in name:
        return _load_pdf_sp800_30(path)
    else:
        return _load_pdf_sp800_61(path)


# ---------------------------------------------------------------------------
# Plain-text / Markdown loader
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


def _load_text(path: Path) -> list[dict[str, Any]]:
    """Load a .txt or .md file into segments; Markdown headings become heading segments."""
    text = path.read_text(encoding="utf-8", errors="replace")
    segments: list[dict[str, Any]] = []
    current_para: list[str] = []

    def _flush() -> None:
        raw = re.sub(r"\s+", " ", " ".join(current_para)).strip()
        if raw:
            segments.append({"is_heading": False, "heading_level": 0, "text": raw})
        current_para.clear()

    for line in text.splitlines():
        m = _MD_HEADING_RE.match(line)
        if m:
            _flush()
            level = len(m.group(1))
            heading = m.group(2).strip()
            if heading:
                segments.append(
                    {"is_heading": True, "heading_level": level, "text": heading}
                )
        elif line.strip() == "":
            _flush()
        else:
            current_para.append(line.strip())

    _flush()
    return segments


# ---------------------------------------------------------------------------
# Source metadata helpers
# ---------------------------------------------------------------------------

def _derive_source_type(filename: str) -> str:
    """Return 'MITRE' or 'NIST' based on the file name prefix."""
    name_upper = filename.upper()
    if name_upper.startswith("MITRE"):
        return "MITRE"
    if name_upper.startswith("NIST"):
        return "NIST"
    return "OTHER"


_TECHNIQUE_RE = re.compile(r"(T\d{4}(?:[._]\d{3})?)", re.IGNORECASE)
_SP_RE = re.compile(r"SP[_\-\s]*([\d\-]+(?:[_\-]Rev\d+)?)", re.IGNORECASE)


def _derive_source_ref(filename: str, source_type: str) -> str:
    """Extract technique ID (MITRE) or SP number (NIST) from the file name."""
    stem = Path(filename).stem
    if source_type == "MITRE":
        m = _TECHNIQUE_RE.search(stem)
        if m:
            return m.group(1).upper().replace("_", ".")
        return stem
    if source_type == "NIST":
        m = _SP_RE.search(stem)
        if m:
            return "SP " + m.group(1).replace("_", " ")
        return stem
    return stem


# ---------------------------------------------------------------------------
# Fine-grained sentence & piece splitter
# ---------------------------------------------------------------------------

_SENT_END_RE = re.compile(r"(?<=[.!?])\s+")


def _split_text_strictly(text: str, max_tokens: int) -> list[str]:
    """Split text into sentences or sub-sentence phrases strictly <= max_tokens."""
    raw_sents = [s.strip() for s in _SENT_END_RE.split(text.strip()) if s.strip()]
    if not raw_sents:
        raw_sents = [text.strip()]

    final_pieces: list[str] = []
    for s in raw_sents:
        if count_tokens(s) <= max_tokens:
            final_pieces.append(s)
        else:
            sub_parts = re.split(r"(?<=[,;:\-])\s+", s)
            for sub in sub_parts:
                sub = sub.strip()
                if not sub:
                    continue
                if count_tokens(sub) <= max_tokens:
                    final_pieces.append(sub)
                else:
                    words = sub.split()
                    cur_w: list[str] = []
                    for w in words:
                        cand = " ".join(cur_w + [w])
                        if count_tokens(cand) > max_tokens:
                            if cur_w:
                                final_pieces.append(" ".join(cur_w))
                                cur_w = [w]
                            else:
                                final_pieces.append(w[: max_tokens * 2])
                                cur_w = []
                        else:
                            cur_w.append(w)
                    if cur_w:
                        final_pieces.append(" ".join(cur_w))
    return final_pieces


# ---------------------------------------------------------------------------
# Section hierarchy & merge helpers
# ---------------------------------------------------------------------------

def is_parent_and_first_child(parent_sec: str, child_sec: str) -> bool:
    """Return True if child_sec is the first child section of parent_sec (e.g. 2.3 and 2.3.1)."""
    m_parent = re.match(r"^(\d+(?:\.\d+)*)", parent_sec)
    m_child = re.match(r"^(\d+(?:\.\d+)*)", child_sec)
    if m_parent and m_child:
        p_num = m_parent.group(1).rstrip(".")
        c_num = m_child.group(1).rstrip(".")
        return c_num == f"{p_num}.1"
    return False


def has_complete_sentence(text: str) -> bool:
    """Return True if *text* contains at least one grammatically terminated sentence."""
    stripped = text.strip()
    if not re.search(r"[.!?](\s|$)", stripped):
        return False
    sents = re.split(r"(?<=[.!?])\s+", stripped)
    for s in sents:
        if len(s.split()) >= 3 and re.search(r"[.!?]$", s.strip()):
            return True
    return False


def postprocess_short_chunks(
    chunks: list[dict[str, Any]],
    max_tokens: int = 350,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[dict[str, Any], dict[str, Any], int]],
    list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]],
]:
    """Apply short chunk merge and drop rules."""
    dropped_under_30: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for c in chunks:
        if c["token_count"] < 30 and not has_complete_sentence(c["text"]):
            dropped_under_30.append(c)
        else:
            filtered.append(dict(c))

    merged: list[dict[str, Any]] = list(filtered)
    cannot_merge: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    merges_done: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]] = []

    i = 0
    while i < len(merged):
        c = merged[i]
        if c["token_count"] < 100:
            sec = c["section"]
            is_mitre = (c.get("source_type") == "MITRE")
            target_idx: int | None = None
            is_following = True
            is_parent_child = False

            if i + 1 < len(merged) and merged[i + 1]["section"] == sec:
                target_idx = i + 1
                is_following = True
            elif i - 1 >= 0 and merged[i - 1]["section"] == sec:
                target_idx = i - 1
                is_following = False
            elif not is_mitre and i + 1 < len(merged) and is_parent_and_first_child(sec, merged[i + 1]["section"]):
                target_idx = i + 1
                is_following = True
                is_parent_child = True

            if target_idx is not None:
                target = merged[target_idx]
                if is_following:
                    merged_text = (c["text"] + " " + target["text"]).strip()
                    cs = c.get("char_start", -1)
                    ce = target.get("char_end", -1)
                else:
                    merged_text = (target["text"] + " " + c["text"]).strip()
                    cs = target.get("char_start", -1)
                    ce = c.get("char_end", -1)

                merged_tc = count_tokens(merged_text)
                if merged_tc <= max_tokens:
                    if is_parent_child:
                        final_sec = target["section"]
                        final_sections = [c["section"], target["section"]]
                    else:
                        final_sec = sec
                        final_sections = list(dict.fromkeys(c.get("sections", [sec]) + target.get("sections", [target["section"]])))

                    new_chunk = {
                        "chunk_id": target["chunk_id"] if target["token_count"] >= c["token_count"] else c["chunk_id"],
                        "source_name": c["source_name"],
                        "source_type": c["source_type"],
                        "source_ref": c["source_ref"],
                        "section": final_sec,
                        "sections": final_sections,
                        "text": merged_text,
                        "token_count": merged_tc,
                        "char_start": cs,
                        "char_end": ce,
                    }
                    merges_done.append((c, target, new_chunk, "parent-child" if is_parent_child else "same-section"))
                    if is_following:
                        merged[i] = new_chunk
                        del merged[target_idx]
                        continue
                    else:
                        merged[target_idx] = new_chunk
                        del merged[i]
                        i -= 1
                        continue
                else:
                    cannot_merge.append((c, target, merged_tc))
        i += 1

    return merged, dropped_under_30, cannot_merge, merges_done


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

def _build_chunks(
    segments: list[dict[str, Any]],
    source_name: str,
    source_type: str,
    source_ref: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    front_matter_label: str = "Description",
) -> list[dict[str, Any]]:
    """Convert segments into overlapping token-window chunks grouped by section."""
    chunks: list[dict[str, Any]] = []

    flat_parts = [s["text"] for s in segments if not s["is_heading"]]
    flat_full_text = " ".join(flat_parts)

    sections: list[tuple[str, list[str]]] = []
    cur_sec = front_matter_label
    cur_texts: list[str] = []

    for seg in segments:
        if seg["is_heading"]:
            if cur_texts:
                sections.append((cur_sec, cur_texts))
                cur_texts = []
            cur_sec = seg["text"]
        else:
            cur_texts.append(seg["text"])
    if cur_texts:
        sections.append((cur_sec, cur_texts))

    for sec_heading, texts in sections:
        pieces: list[str] = []
        for t in texts:
            pieces.extend(_split_text_strictly(t, max_tokens))

        pending: list[str] = []
        for piece in pieces:
            cand = " ".join(pending + [piece])
            if count_tokens(cand) <= max_tokens:
                pending.append(piece)
            else:
                if pending:
                    body = re.sub(r"\s+", " ", " ".join(pending)).strip()
                    tc = count_tokens(body)
                    try:
                        cs = flat_full_text.index(body[:60])
                    except ValueError:
                        cs = -1
                    ce = cs + len(body) if cs >= 0 else -1

                    chunks.append(
                        {
                            "chunk_id": str(uuid.uuid4()),
                            "source_name": source_name,
                            "source_type": source_type,
                            "source_ref": source_ref,
                            "section": sec_heading,
                            "sections": [sec_heading],
                            "text": body,
                            "token_count": tc,
                            "char_start": cs,
                            "char_end": ce,
                        }
                    )

                    overlap: list[str] = []
                    for item in reversed(pending):
                        cand_overlap = [item] + overlap
                        if (
                            count_tokens(" ".join(cand_overlap)) <= overlap_tokens
                            and count_tokens(" ".join(cand_overlap + [piece])) <= max_tokens
                        ):
                            overlap.insert(0, item)
                        else:
                            break
                    pending = overlap + [piece]
                else:
                    pending = [piece]

        if pending:
            body = re.sub(r"\s+", " ", " ".join(pending)).strip()
            tc = count_tokens(body)
            try:
                cs = flat_full_text.index(body[:60])
            except ValueError:
                cs = -1
            ce = cs + len(body) if cs >= 0 else -1

            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "source_name": source_name,
                    "source_type": source_type,
                    "source_ref": source_ref,
                    "section": sec_heading,
                    "sections": [sec_heading],
                    "text": body,
                    "token_count": tc,
                    "char_start": cs,
                    "char_end": ce,
                }
            )

    merged: list[dict[str, Any]] = []
    for c in chunks:
        if (
            merged
            and c["token_count"] < min_tokens
            and merged[-1]["source_name"] == c["source_name"]
            and merged[-1]["section"] == c["section"]
        ):
            cand = re.sub(r"\s+", " ", merged[-1]["text"] + " " + c["text"]).strip()
            cand_tc = count_tokens(cand)
            if cand_tc <= max_tokens:
                merged[-1]["text"] = cand
                merged[-1]["token_count"] = cand_tc
                merged[-1]["char_end"] = c["char_end"]
                continue
        merged.append(c)

    for c in merged:
        c["text"] = c["text"].strip()

    return merged


# ---------------------------------------------------------------------------
# Document loader dispatcher
# ---------------------------------------------------------------------------

def load_document(path: Path) -> list[dict[str, Any]]:
    """Load *path* into a list of raw segments based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in (".html", ".htm"):
        raw = path.read_text(encoding="utf-8", errors="replace")
        segs = _parse_html(raw)
        if _derive_source_type(path.name) == "MITRE":
            segs = clean_mitre_segments(segs)
        return segs
    if suffix in (".txt", ".md"):
        return _load_text(path)
    raise ValueError(f"Unsupported file type: {path.suffix!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_knowledge_base(
    kb_dir: Path | None = None,
    output_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Process all supported documents in *kb_dir* and return all chunks."""
    if config is None:
        config = load_config()

    root = get_project_root()
    rag_cfg = config.get("rag", {})

    if kb_dir is None:
        kb_dir = root / config.get("paths", {}).get("kb", "data/kb")
    if output_path is None:
        output_path = kb_dir / "chunks.jsonl"

    min_tokens: int = int(rag_cfg.get("chunk_min_tokens", 200))
    max_tokens: int = int(rag_cfg.get("chunk_max_tokens", 350))
    overlap_tokens: int = int(rag_cfg.get("chunk_overlap", 200))

    supported_exts = {".pdf", ".html", ".htm", ".txt", ".md"}
    doc_paths = sorted(
        p
        for p in kb_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in supported_exts
        and p.name != "chunks.jsonl"
    )

    all_raw_chunks: list[dict[str, Any]] = []

    for doc_path in doc_paths:
        source_name = doc_path.name
        source_type = _derive_source_type(source_name)
        source_ref = _derive_source_ref(source_name, source_type)

        segments = load_document(doc_path)
        front_label = "Description" if source_type == "MITRE" else "Executive Summary"
        doc_chunks = _build_chunks(
            segments=segments,
            source_name=source_name,
            source_type=source_type,
            source_ref=source_ref,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            front_matter_label=front_label,
        )
        all_raw_chunks.extend(doc_chunks)

    final_chunks, dropped_30, cannot_merge, merges_done = postprocess_short_chunks(
        all_raw_chunks, max_tokens=max_tokens
    )

    chunk_knowledge_base._dropped_30 = dropped_30  # type: ignore[attr-defined]
    chunk_knowledge_base._cannot_merge = cannot_merge  # type: ignore[attr-defined]
    chunk_knowledge_base._merges_done = merges_done  # type: ignore[attr-defined]

    output_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in final_chunks),
        encoding="utf-8",
    )

    return final_chunks


# ---------------------------------------------------------------------------
# Coverage metrics
# ---------------------------------------------------------------------------

def interval_union_length(intervals: list[tuple[int, int]]) -> int:
    """Return the total length of the union of intervals [(start, end), ...]."""
    valid = [(s, e) for s, e in intervals if s >= 0 and e >= s]
    if not valid:
        return 0
    valid.sort()
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = valid[0]
    for s, e in valid[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return sum(e - s for s, e in merged)


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def write_chunks_preview(
    all_chunks: list[dict[str, Any]],
    output_path: Path,
    seed: int = 42,
) -> None:
    """Write artifacts/chunks_preview.txt."""
    rng = random.Random(seed)

    mitre_chunks = [c for c in all_chunks if c["source_type"] == "MITRE"]
    nist_chunks = [c for c in all_chunks if c["source_type"] == "NIST"]

    sample_nist = rng.sample(nist_chunks, min(15, len(nist_chunks)))
    shortest = sorted(all_chunks, key=lambda c: c["token_count"])[:10]

    def _fmt_chunk(c: dict[str, Any], idx: int | None = None) -> str:
        prefix = f"[{idx}] " if idx is not None else ""
        sections_str = json.dumps(c.get("sections", [c["section"]]), ensure_ascii=False)
        return (
            f"{prefix}chunk_id   : {c['chunk_id']}\n"
            f"  source     : {c['source_name']} ({c['source_type']} / {c['source_ref']})\n"
            f"  section    : {c['section']!r}\n"
            f"  sections   : {sections_str}\n"
            f"  token_count: {c['token_count']}\n"
            f"  char_span  : [{c.get('char_start', -1)}, {c.get('char_end', -1)}]\n"
            f"  text       :\n{c['text']}"
        )

    lines: list[str] = [
        "=" * 80,
        f"CHUNKS PREVIEW QUALITY REPORT (Total Chunks: {len(all_chunks)})",
        "=" * 80,
        "",
        f"--- PART 1: ALL MITRE CHUNKS ({len(mitre_chunks)} chunks) ---",
        "",
    ]

    for i, c in enumerate(mitre_chunks):
        lines.append(_fmt_chunk(c, i))
        lines.append("-" * 40)

    lines.extend([
        "",
        f"--- PART 2: 15 RANDOMLY SAMPLED NIST CHUNKS (seed={seed}) ---",
        "",
    ])

    for i, c in enumerate(sample_nist):
        lines.append(_fmt_chunk(c, i))
        lines.append("-" * 40)

    lines.extend([
        "",
        "--- PART 3: 10 SHORTEST CHUNKS OVERALL ---",
        "",
    ])

    for i, c in enumerate(shortest):
        lines.append(_fmt_chunk(c, i))
        lines.append("-" * 40)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_report(all_chunks: list[dict[str, Any]], doc_paths: list[Path]) -> None:
    """Print summary statistics and detailed per-PDF reports."""
    if not all_chunks:
        print("No chunks produced.")
        return

    per_doc: Counter[str] = Counter(c["source_name"] for c in all_chunks)
    token_counts = [c["token_count"] for c in all_chunks]
    total_tokens = sum(token_counts)

    print(f"\nDocuments processed : {len(doc_paths)}")
    print(f"Total chunks        : {len(all_chunks)}")
    print(f"Token count  min    : {min(token_counts)}")
    print(f"Token count  mean   : {total_tokens / len(token_counts):.1f}")
    print(f"Token count  max    : {max(token_counts)}")
    print("\nChunks per document:")
    for name, count in sorted(per_doc.items()):
        print(f"  {name}: {count}")

    mitre_chunks = [c for c in all_chunks if c["source_type"] == "MITRE"]
    if mitre_chunks:
        print(f"\n--- Chunks per MITRE Section ({len(mitre_chunks)} total) ---")
        mitre_sec: Counter[str] = Counter(f"{c['source_name']} | {c['section']}" for c in mitre_chunks)
        for sec, cnt in sorted(mitre_sec.items()):
            print(f"  {cnt:2d}  {sec}")

    nist_chunks = [c for c in all_chunks if c["source_type"] == "NIST"]
    if nist_chunks:
        nist_sections: Counter[str] = Counter(f"{c['source_name']} | {c['section']}" for c in nist_chunks)
        print(f"\nNIST distinct (document, section) pairs: {len(nist_sections)}")
        print("Chunks per NIST section (top 20):")
        for sec, cnt in sorted(nist_sections.items(), key=lambda x: -x[1])[:20]:
            print(f"  {cnt:3d}  {sec}")

    print("\n" + "=" * 80)
    print("DETAILED PER-PDF PROFILE REPORTS")
    print("=" * 80)

    for pdf_name, stats in sorted(PDF_LOAD_STATS.items()):
        parsed = stats.get("toc_entries_parsed", 0)
        matched = stats.get("toc_entries_matched", 0)
        not_matched = stats.get("toc_entries_not_matched", [])
        dropped_purpose = stats.get("toc_entries_dropped_on_purpose", [])

        # Assertion: parsed = matched + unmatched (+ dropped on purpose for SP 800-61)
        assert parsed == matched + len(not_matched) + len(dropped_purpose), (
            f"TOC check failed for {pdf_name}: parsed ({parsed}) != matched ({matched}) + "
            f"unmatched ({len(not_matched)}) + dropped_on_purpose ({len(dropped_purpose)})"
        )

        print(f"\n--- PDF Report: {pdf_name} ---")
        print(f"  Total pages in document       : {stats.get('total_pages')}")
        print(f"  Pages kept                    : {stats.get('pages_kept')}")
        print("  Pages dropped by rule:")
        for r_name, cnt in stats.get("pages_dropped_by_rule", {}).items():
            print(f"    - {r_name}: {cnt} pages")
        print(f"  TOC entries parsed            : {parsed}")
        print(f"  TOC entries matched in body   : {matched}")
        if dropped_purpose:
            print(f"  TOC entries dropped on purpose: {len(dropped_purpose)}")
            for dp in dropped_purpose:
                print(f"    - {dp}")
        print(f"  TOC entries NOT matched in body: {len(not_matched)}")
        for nm in not_matched:
            print(f"    - {nm}")
        print("  Header / footer / label lines stripped:")
        for l_rule, l_cnt in stats.get("stripped_lines_by_rule", {}).items():
            print(f"    - {l_rule}: {l_cnt} lines")
        print(f"  Footnote substitutions total  : {stats.get('footnote_substitutions')}")
        print(f"  Footnote unique pairs count   : {stats.get('footnote_unique_pairs')}")
        fn_samples = stats.get("footnote_samples", [])
        if fn_samples:
            print("  Footnote substitution samples (up to 10):")
            for b, a in fn_samples:
                print(f"    - {b!r} -> {a!r}")
        print("  Drop-cap repairs:")
        print(f"    - Done     : {stats.get('drop_cap_repairs_done')}")
        print(f"    - Not done : {stats.get('drop_cap_repairs_not_done')}")

    dropped_30 = getattr(chunk_knowledge_base, "_dropped_30", [])
    merges_done = getattr(chunk_knowledge_base, "_merges_done", [])

    print("\n--- Short Chunks Post-Processing Report ---")
    print(f"Chunks dropped (< 30 tokens, no complete sentence): {len(dropped_30)}")
    for d in dropped_30:
        print(f"  - [{d['token_count']} tok] ({d['source_name']} / {d['section']}): {d['text']!r}")

    print(f"Merges performed: {len(merges_done)}")
    for sc, nb, mc, kind in merges_done:
        print(f"  - [{kind}] Merged {sc['token_count']} tok ('{sc['section']}') into neighbour {nb['token_count']} tok ('{nb['section']}') -> {mc['token_count']} tok (kept section: '{mc['section']}', sections: {mc['sections']})")

    under_150 = sum(1 for tc in token_counts if tc < 150)
    print(f"\nChunks under 150 tokens : {under_150} ({100 * under_150 / len(token_counts):.1f}%)")

    doc_chunk_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in all_chunks:
        doc_chunk_map[c["source_name"]].append(c)

    all_sum_len = 0
    all_union_len = 0

    print("\n--- Coverage Ratio and Average Copies Report ---")
    for doc_name, d_chunks in sorted(doc_chunk_map.items()):
        sum_len = sum(len(c["text"]) for c in d_chunks)
        intervals = [(c["char_start"], c["char_end"]) for c in d_chunks]
        union_len = interval_union_length(intervals)
        cov_ratio = union_len / sum_len if sum_len > 0 else 0.0
        avg_copies = sum_len / union_len if union_len > 0 else 0.0
        all_sum_len += sum_len
        all_union_len += union_len
        print(f"  {doc_name}:")
        print(f"    Coverage ratio : {cov_ratio:.4f}")
        print(f"    Average copies : {avg_copies:.4f}")

    overall_cov = all_union_len / all_sum_len if all_sum_len > 0 else 0.0
    overall_copies = all_sum_len / all_union_len if all_union_len > 0 else 0.0
    print(f"  OVERALL:")
    print(f"    Coverage ratio : {overall_cov:.4f}")
    print(f"    Average copies : {overall_copies:.4f}")

    print("\n--- 10 Shortest Chunks in Full ---")
    shortest10 = sorted(all_chunks, key=lambda c: c["token_count"])[:10]
    for i, c in enumerate(shortest10):
        print(f"\n[{i}] chunk_id={c['chunk_id'][:8]}... | tokens={c['token_count']} | source={c['source_name']} | section={c['section']!r} | sections={c.get('sections')}")
        print(f"    {c['text']}")


def main() -> None:
    """CLI entry point: chunk all KB documents and print statistics."""
    config = load_config()
    root = get_project_root()
    kb_dir = root / config.get("paths", {}).get("kb", "data/kb")
    output_path = kb_dir / "chunks.jsonl"
    artifacts_dir = root / config.get("paths", {}).get("artifacts", "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    preview_path = artifacts_dir / "chunks_preview.txt"

    supported_exts = {".pdf", ".html", ".htm", ".txt", ".md"}
    doc_paths = sorted(
        p
        for p in kb_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in supported_exts
        and p.name != "chunks.jsonl"
    )

    print(f"Found {len(doc_paths)} document(s) in {kb_dir}:")
    for p in doc_paths:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name}  ({size_kb:.1f} KB)")

    print("\nChunking...")
    all_chunks = chunk_knowledge_base(
        kb_dir=kb_dir,
        output_path=output_path,
        config=config,
    )

    _cli_report(all_chunks, doc_paths)
    print(f"\nSaved {len(all_chunks)} chunks to {output_path}")

    write_chunks_preview(all_chunks, preview_path)
    print(f"Saved quality report to {preview_path}")


if __name__ == "__main__":
    main()
