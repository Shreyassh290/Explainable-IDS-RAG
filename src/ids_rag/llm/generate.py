"""Local LLM report generation and validation using Ollama.

Implements report generation and automated validation following arXiv 2605.17960:
- Verification of Ollama connectivity and installed model presence.
- Chat completion inference with strict configuration options and keep_alive.
- Automated validation of word count limits, required section headings,
  in-text citation bracket patterns, and IPv4 hallucination checks.
- Batch generation across test cases in RAG and vanilla modes.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import ollama

from ids_rag.config import get_project_root, load_config
from ids_rag.llm.prompt import build_prompt
from ids_rag.rag.retrieve import build_query, retrieve

# ---------------------------------------------------------------------------
# KB cache (loaded once per process for validate_report)
# ---------------------------------------------------------------------------

_KB_CACHE: dict[str, Any] | None = None  # None means not yet loaded


def _load_kb_cache() -> dict[str, Any]:
    """Load KB data from data/kb/chunks.jsonl exactly once.

    Returns a dict with:
        'attack_text': set of all T-id strings found in chunk text fields.
        'sp_numbers': set of "800-NN" strings extracted from source file names.
        'missing': True if the file did not exist.
    """
    global _KB_CACHE
    if _KB_CACHE is not None:
        return _KB_CACHE

    root = get_project_root()
    kb_path = root / "data" / "kb" / "chunks.jsonl"
    if not kb_path.exists():
        _KB_CACHE = {"attack_text": None, "sp_numbers": None, "missing": True}
        return _KB_CACHE

    attack_text: set[str] = set()
    sp_numbers: set[str] = set()
    sp_name_re = re.compile(r"800-(\d{1,3})")

    with open(kb_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for m in re.findall(r"\bT\d{4}(?:\.\d{3})?\b", row.get("text", "")):
                attack_text.add(m)
            for m in sp_name_re.findall(row.get("source_name", "")):
                sp_numbers.add(f"800-{m}")

    _KB_CACHE = {"attack_text": attack_text, "sp_numbers": sp_numbers, "missing": False}
    return _KB_CACHE


def _normalize_model_name(name: str) -> str:
    """Strip ':latest' suffix for canonical model name matching."""
    return name[:-7] if name.endswith(":latest") else name


def check_ollama(cfg: dict[str, Any], client: Any = None) -> list[str]:
    """Verify that Ollama server is accessible and the configured model is installed.

    Args:
        cfg: Configuration dictionary with 'llm' section.
        client: Optional injected Ollama Client instance for testing.

    Returns:
        List of installed model names.

    Raises:
        RuntimeError: If server is unreachable or configured model is missing.
    """
    llm_cfg = cfg.get("llm", {})
    host = llm_cfg.get("host", "http://localhost:11434")
    model_name = llm_cfg.get("model", "llama3:8b")
    timeout = llm_cfg.get("request_timeout_s", 600)

    if client is None:
        try:
            client = ollama.Client(host=host, timeout=timeout)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Ollama client for {host}. Ensure Ollama is running. Error: {e}"
            ) from e

    try:
        resp = client.list()
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Ollama server at {host}. Please start Ollama (e.g., 'ollama serve') before running. Error: {e}"
        ) from e

    models_raw = getattr(resp, "models", None)
    if models_raw is None and isinstance(resp, dict):
        models_raw = resp.get("models", [])
    if models_raw is None:
        models_raw = []

    installed_names: list[str] = []
    for m in models_raw:
        name = None
        if hasattr(m, "model") and m.model:
            name = m.model
        elif hasattr(m, "name") and m.name:
            name = m.name
        elif isinstance(m, dict):
            name = m.get("model") or m.get("name")
        elif isinstance(m, str):
            name = m
        if name:
            installed_names.append(str(name))

    norm_target = _normalize_model_name(model_name)
    installed_norms = [_normalize_model_name(n) for n in installed_names]

    if norm_target not in installed_norms:
        raise RuntimeError(
            f"Configured model '{model_name}' is not among installed models: {installed_names}. "
            f"Please run: ollama pull {model_name}"
        )

    return installed_names


def generate_report(
    prompt: dict[str, Any],
    cfg: dict[str, Any],
    client: Any = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Generate incident response report using Ollama chat completion API.

    Args:
        prompt: Prompt dict from build_prompt ('system', 'user', 'mode', 'sources').
        cfg: Configuration dictionary with 'llm' section.
        client: Optional injected Ollama Client instance for testing.
        seed_override: If provided, overrides llm.seed in the options for this call.

    Returns:
        Dict containing text, model, options, prompt_words, timings, tokens_per_second,
        and seed_used.

    Raises:
        ValueError: If prompt['mode'] is not 'rag' or 'vanilla'.
    """
    mode = prompt.get("mode")
    if mode not in ("rag", "vanilla"):
        raise ValueError(f"Invalid mode '{mode}' in prompt dict. Expected 'rag' or 'vanilla'.")

    llm_cfg = cfg.get("llm", {})
    model = str(llm_cfg.get("model", "llama3:8b"))
    base_seed = int(llm_cfg.get("seed", 42))
    seed_used = seed_override if seed_override is not None else base_seed
    options = {
        "temperature": float(llm_cfg.get("temperature", 0.2)),
        "seed": seed_used,
        "num_ctx": int(llm_cfg.get("num_ctx", 8192)),
        "num_predict": int(llm_cfg.get("num_predict", 1100)),
    }
    if "repeat_penalty" in llm_cfg and llm_cfg["repeat_penalty"] is not None:
        options["repeat_penalty"] = float(llm_cfg["repeat_penalty"])
    if "top_p" in llm_cfg and llm_cfg["top_p"] is not None:
        options["top_p"] = float(llm_cfg["top_p"])
    keep_alive = llm_cfg.get("keep_alive", "15m")

    if client is None:
        host = llm_cfg.get("host", "http://localhost:11434")
        timeout = llm_cfg.get("request_timeout_s", 600)
        client = ollama.Client(host=host, timeout=timeout)

    messages = [
        {"role": "system", "content": prompt.get("system", "")},
        {"role": "user", "content": prompt.get("user", "")},
    ]

    resp = client.chat(
        model=model,
        messages=messages,
        options=options,
        keep_alive=keep_alive,
    )

    if hasattr(resp, "message") and hasattr(resp.message, "content"):
        text = resp.message.content or ""
    elif isinstance(resp, dict):
        msg = resp.get("message", {})
        text = msg.get("content", "") if isinstance(msg, dict) else str(msg)
    else:
        text = str(resp)

    def _get_val(obj: Any, key: str, default: Any = 0) -> Any:
        if hasattr(obj, key):
            val = getattr(obj, key)
            return val if val is not None else default
        if isinstance(obj, dict):
            val = obj.get(key)
            return val if val is not None else default
        return default

    prompt_eval_count = int(_get_val(resp, "prompt_eval_count", 0))
    prompt_eval_sec = float(_get_val(resp, "prompt_eval_duration", 0)) / 1e9
    eval_count = int(_get_val(resp, "eval_count", 0))
    eval_sec = float(_get_val(resp, "eval_duration", 0)) / 1e9
    total_sec = float(_get_val(resp, "total_duration", 0)) / 1e9
    load_sec = float(_get_val(resp, "load_duration", 0)) / 1e9

    tokens_per_second = eval_count / eval_sec if eval_sec > 0 else 0.0
    prompt_words = len(prompt.get("system", "").split()) + len(prompt.get("user", "").split())
    resp_model = str(_get_val(resp, "model", model))

    return {
        "text": text,
        "model": resp_model,
        "options": options,
        "seed_used": seed_used,
        "prompt_words": prompt_words,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_sec,
        "eval_count": eval_count,
        "eval_duration": eval_sec,
        "total_duration": total_sec,
        "load_duration": load_sec,
        "tokens_per_second": tokens_per_second,
    }


def _extract_ipv4s(text: str) -> set[str]:
    """Extract valid IPv4 addresses from text."""
    matches = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    valid = set()
    for m in matches:
        octets = m.split(".")
        if all(0 <= int(o) <= 255 for o in octets):
            valid.add(m)
    return valid


# Compile once: citation range pattern (hyphen or en-dash)
_RANGE_RE = re.compile(r"\[\s*\d+\s*[-\u2013]\s*\d+\s*\]")
# Pattern that identifies a copied source header line:
# after stripping leading markdown symbols and bullet characters,
# matches lines of the form [N] NIST ... or [N] MITRE ...
_COPIED_HDR_LINE_RE = re.compile(
    r"^[#*_>\-\s]*\[(\d+)\]\s+(?:NIST|MITRE)\b",
    re.IGNORECASE,
)
# headings_any: strip leading markdown, optional "N." or "N)" prefix, trailing ":" or "*"
_HEADING_ANY_STRIP_RE = re.compile(r"^[#*_>\s]+")


def _check_heading_any(line: str, heading_name: str) -> bool:
    """Return True if line, after stripping markdown/numbering, matches heading_name."""
    # Strip leading markdown symbols and spaces
    stripped = _HEADING_ANY_STRIP_RE.sub("", line).strip()
    # Strip optional leading number prefix: "1." or "1)" or "1 "
    stripped = re.sub(r"^\d+[\.\)]\s*", "", stripped)
    # Strip trailing colons and asterisks
    stripped = stripped.rstrip(":*").strip()
    # Case-insensitive equality or "starts with heading_name followed by colon"
    lower = stripped.lower()
    target = heading_name.lower()
    return lower == target or lower.startswith(target + ":")


def validate_report(
    text: str,
    n_sources: int,
    mode: str,
    cfg: dict[str, Any],
    prompt_user_text: str = "",
    kb_path: Path | None = None,
) -> dict[str, Any]:
    """Validate generated report against constraints specified in arXiv 2605.17960.

    Args:
        text: Generated report text.
        n_sources: Number of available sources in prompt.
        mode: 'rag' or 'vanilla'.
        cfg: Configuration dictionary.
        prompt_user_text: Optional prompt user text to verify IPv4 references.
        kb_path: Override path to chunks.jsonl (used in tests; None uses default).

    Returns:
        Dict with word_count, headings_found, headings_any, headings_any_count,
        words_in_range, cited, cited_in_copied_headers, copied_source_headers,
        invalid_citations, citations_in_vanilla, unsupported_ips,
        placeholder_citations, citation_ranges, has_reference_list,
        attack_ids_not_in_kb, sp_not_in_kb.
    """
    llm_cfg = cfg.get("llm", {}) if cfg else {}
    min_words = int(llm_cfg.get("min_words", 400))
    max_words = int(llm_cfg.get("max_words", 500))

    # 1. Word count
    words = text.split()
    word_count = len(words)
    words_in_range = bool(min_words <= word_count <= max_words)

    # 2. Five required headings — numbered check (headings_found, unchanged)
    heading_defs = [
        (1, "Rationale"),
        (2, "Key Indicators"),
        (3, "Confidence Assessment"),
        (4, "Threat Assessment"),
        (5, "Recommendations"),
    ]
    headings_found: dict[str, bool] = {}
    for i, h_name in heading_defs:
        pattern = rf"^\W*{i}\.\s*{re.escape(h_name)}"
        headings_found[h_name] = bool(re.search(pattern, text, re.IGNORECASE | re.MULTILINE))

    # 2b. headings_any — unnumbered or numbered match per line
    headings_any: dict[str, bool] = {h_name: False for _, h_name in heading_defs}
    for line in text.splitlines():
        for _, h_name in heading_defs:
            if not headings_any[h_name] and _check_heading_any(line, h_name):
                headings_any[h_name] = True
    headings_any_count: int = sum(1 for v in headings_any.values() if v)

    # 3. Identify copied source header lines and collect their citation numbers
    lines = text.splitlines()
    copied_source_headers: int = 0
    cited_in_copied_headers_set: set[int] = set()
    non_header_lines: list[str] = []
    for line in lines:
        m = _COPIED_HDR_LINE_RE.match(line)
        if m:
            copied_source_headers += 1
            cited_in_copied_headers_set.add(int(m.group(1)))
        else:
            non_header_lines.append(line)
    cited_in_copied_headers: list[int] = sorted(cited_in_copied_headers_set)
    # Remaining text without copied source headers (used for citation extraction)
    text_without_headers = "\n".join(non_header_lines)

    # 4. Citation ranges (extracted from text_without_headers)
    citation_ranges: list[str] = _RANGE_RE.findall(text_without_headers)
    text_no_ranges = _RANGE_RE.sub("", text_without_headers)

    # 5. Placeholder citations: literal "[n]" occurrences in full text
    placeholder_citations: int = text.count("[n]")

    # 6. Citations in brackets: [n], [n, m], [n][m] — from text with headers and ranges removed
    cited_set: set[int] = set()
    for block in re.findall(r"\[[\d\s,]+\]", text_no_ranges):
        for num_str in re.findall(r"\d+", block):
            cited_set.add(int(num_str))
    cited = sorted(cited_set)

    # 7. Citation validity
    if mode == "vanilla":
        citations_in_vanilla = list(cited)
        invalid_citations = list(cited)
    else:
        citations_in_vanilla = []
        invalid_citations = [n for n in cited if n < 1 or n > n_sources]

    # 8. Unsupported IPv4 addresses (checked against full text)
    text_ips = _extract_ipv4s(text)
    prompt_ips = _extract_ipv4s(prompt_user_text) if prompt_user_text else set()
    unsupported_ips = sorted(text_ips - prompt_ips)

    # 9. Reference list detection: a line starting (after optional markdown symbols) with
    #    "References" or "Sources" (case-insensitive). Copied source headers also trigger this.
    ref_line_re = re.compile(r"^[#*_>\s]*(?:References|Sources)\b", re.IGNORECASE | re.MULTILINE)
    has_reference_list: bool = bool(ref_line_re.search(text)) or copied_source_headers > 0

    # 10. KB-based checks (attack IDs and SP numbers, checked against full text)
    if kb_path is not None:
        kb_attack_text: set[str] | None = set()
        kb_sp_numbers: set[str] | None = set()
        sp_name_re = re.compile(r"800-(\d{1,3})")
        try:
            with open(kb_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    for m_id in re.findall(r"\bT\d{4}(?:\.\d{3})?\b", row.get("text", "")):
                        kb_attack_text.add(m_id)  # type: ignore[union-attr]
                    for m_sp in sp_name_re.findall(row.get("source_name", "")):
                        kb_sp_numbers.add(f"800-{m_sp}")  # type: ignore[union-attr]
        except (OSError, json.JSONDecodeError):
            kb_attack_text = None
            kb_sp_numbers = None
    else:
        cache = _load_kb_cache()
        if cache.get("missing"):
            kb_attack_text = None
            kb_sp_numbers = None
        else:
            kb_attack_text = cache["attack_text"]
            kb_sp_numbers = cache["sp_numbers"]

    text_attack_ids: list[str] = sorted(set(re.findall(r"\bT\d{4}(?:\.\d{3})?\b", text)))
    if kb_attack_text is None:
        attack_ids_not_in_kb: list[str] | None = None
        kb_attack_ids_named: list[str] | None = None
    else:
        attack_ids_not_in_kb = sorted(aid for aid in text_attack_ids if aid not in kb_attack_text)
        kb_attack_ids_named = sorted(aid for aid in text_attack_ids if aid in kb_attack_text)

    sp_re = re.compile(r"(?:SP|Special Publication)\s*800-(\d{1,3})")
    text_sp_numbers = sorted({f"800-{m}" for m in sp_re.findall(text)})
    if kb_sp_numbers is None:
        sp_not_in_kb: list[str] | None = None
        kb_sp_named: list[str] | None = None
    else:
        sp_not_in_kb = sorted(sp for sp in text_sp_numbers if sp not in kb_sp_numbers)
        kb_sp_named = sorted(sp for sp in text_sp_numbers if sp in kb_sp_numbers)

    return {
        "word_count": word_count,
        "headings_found": headings_found,
        "headings_any": headings_any,
        "headings_any_count": headings_any_count,
        "words_in_range": words_in_range,
        "cited": cited,
        "cited_in_copied_headers": cited_in_copied_headers,
        "copied_source_headers": copied_source_headers,
        "invalid_citations": invalid_citations,
        "citations_in_vanilla": citations_in_vanilla,
        "unsupported_ips": unsupported_ips,
        "placeholder_citations": placeholder_citations,
        "citation_ranges": citation_ranges,
        "has_reference_list": has_reference_list,
        "attack_ids_not_in_kb": attack_ids_not_in_kb,
        "kb_attack_ids_named": kb_attack_ids_named,
        "sp_not_in_kb": sp_not_in_kb,
        "kb_sp_named": kb_sp_named,
    }


def run_case(
    case_name: str,
    explanation: dict[str, Any],
    chunks: list[dict[str, Any]] | None,
    cfg: dict[str, Any],
    client: Any = None,
    repeat: int = 1,
    vary_seed: bool = False,
    out_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build prompts, generate reports in both RAG and vanilla modes, save files and validate.

    For repeat > 1, run 1's text is written to the plain file; subsequent runs go to
    <out_dir>/runs/<case>_<mode>_r<k>.txt.
    With vary_seed=True, run k uses seed = llm.seed + (k - 1).

    Returns:
        (rag_validation_dict, vanilla_validation_dict) for run 1.
    """
    if out_dir is None:
        root = get_project_root()
        out_dir = root / "artifacts" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    if repeat > 1:
        runs_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = cfg.get("llm", {})
    base_seed = int(llm_cfg.get("seed", 42))

    first_rag_val: dict[str, Any] = {}
    first_van_val: dict[str, Any] = {}
    first_rag_gen: dict[str, Any] = {}
    first_van_gen: dict[str, Any] = {}
    first_rag_prompt: dict[str, Any] = {}
    first_van_prompt: dict[str, Any] = {}

    rag_prompt = build_prompt(explanation, chunks, mode="rag", cfg=cfg)
    vanilla_prompt = build_prompt(explanation, chunks, mode="vanilla", cfg=cfg)

    for run_idx in range(1, repeat + 1):
        seed_override = (base_seed + run_idx - 1) if vary_seed else None
        for mode, prompt in (("rag", rag_prompt), ("vanilla", vanilla_prompt)):
            gen = generate_report(prompt, cfg, client=client, seed_override=seed_override)
            n_src = len(prompt["sources"]) if mode == "rag" else 0
            val = validate_report(
                text=gen["text"],
                n_sources=n_src,
                mode=mode,
                cfg=cfg,
                prompt_user_text=prompt["user"],
            )

            if run_idx == 1:
                with open(out_dir / f"{case_name}_{mode}.txt", "w", encoding="utf-8") as f:
                    f.write(gen["text"])
                if mode == "rag":
                    first_rag_val = val
                    first_rag_gen = gen
                    first_rag_prompt = prompt
                else:
                    first_van_val = val
                    first_van_gen = gen
                    first_van_prompt = prompt
            else:
                run_file = runs_dir / f"{case_name}_{mode}_r{run_idx}.txt"
                with open(run_file, "w", encoding="utf-8") as f:
                    f.write(gen["text"])

    # Save complete JSON summary (using run-1 data; record seed_used)
    summary = {
        "case": case_name,
        "model": first_rag_gen["model"],
        "options": first_rag_gen["options"],
        "seed_used": first_rag_gen["seed_used"],
        "rag": {
            "prompt": first_rag_prompt,
            "response": first_rag_gen["text"],
            "validation": first_rag_val,
            "seed_used": first_rag_gen["seed_used"],
            "timings": {
                "prompt_eval_count": first_rag_gen["prompt_eval_count"],
                "prompt_eval_duration": first_rag_gen["prompt_eval_duration"],
                "eval_count": first_rag_gen["eval_count"],
                "eval_duration": first_rag_gen["eval_duration"],
                "total_duration": first_rag_gen["total_duration"],
                "load_duration": first_rag_gen["load_duration"],
                "tokens_per_second": first_rag_gen["tokens_per_second"],
            },
        },
        "vanilla": {
            "prompt": first_van_prompt,
            "response": first_van_gen["text"],
            "validation": first_van_val,
            "seed_used": first_van_gen["seed_used"],
            "timings": {
                "prompt_eval_count": first_van_gen["prompt_eval_count"],
                "prompt_eval_duration": first_van_gen["prompt_eval_duration"],
                "eval_count": first_van_gen["eval_count"],
                "eval_duration": first_van_gen["eval_duration"],
                "total_duration": first_van_gen["total_duration"],
                "load_duration": first_van_gen["load_duration"],
                "tokens_per_second": first_van_gen["tokens_per_second"],
            },
        },
    }
    with open(out_dir / f"{case_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return first_rag_val, first_van_val


def main() -> None:
    """CLI entry point for checking Ollama server and running report generation."""
    parser = argparse.ArgumentParser(description="LLM incident response report generation runner.")
    parser.add_argument("--check", action="store_true", help="Verify Ollama server and configured model.")
    parser.add_argument("--examples", action="store_true", help="Run report generation for sample explanations.")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="Number of times to generate each case/mode (default: 1).")
    parser.add_argument("--vary-seed", action="store_true",
                        help="With --repeat N, run k uses seed = llm.seed + (k-1).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for generated reports (default: artifacts/reports).")
    args = parser.parse_args()

    cfg = load_config()

    if args.check:
        models = check_ollama(cfg)
        print("Ollama connection verified. Installed models:")
        for m in models:
            print(f"  - {m}")
        return

    if args.examples:
        repeat = max(1, args.repeat)
        vary_seed: bool = bool(args.vary_seed)
        llm_cfg = cfg.get("llm", {})
        base_seed = int(llm_cfg.get("seed", 42))

        # Print seeds that will be used
        seeds_used = [base_seed + k for k in range(repeat)] if vary_seed else [base_seed] * repeat
        print(f"[CONFIG] Seeds for runs 1..{repeat}: {seeds_used}", flush=True)

        # Verify Ollama installation
        check_ollama(cfg)

        root = get_project_root()
        sample_path = root / "artifacts" / "sample_explanations.json"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample explanations not found at: {sample_path}")

        with open(sample_path, "r", encoding="utf-8") as f:
            samples: list[dict[str, Any]] = json.load(f)

        case_map: dict[str, dict[str, Any]] = {}
        for s in samples:
            c = s.get("case", "")
            if c:
                case_map[c] = s

        target_order = ["correct_Benign", "correct_DoS", "correct_DDoS", "misclassified"]
        ordered_cases: list[tuple[str, dict[str, Any]]] = []
        for name in target_order:
            if name in case_map:
                ordered_cases.append((name, case_map[name]))
        for s in samples:
            c = s.get("case", "")
            if c not in [name for name, _ in ordered_cases]:
                ordered_cases.append((c or f"case_{len(ordered_cases)}", s))

        # Table header: renamed Headings → H-num, added H-any and CopiedHdr
        header = (
            f"{'Case':<18} {'Mode':<8} {'Words':<6} {'InRange':<8} {'H-num':<7} {'H-any':<7} "
            f"{'CopiedHdr':<10} {'Cited':<12} {'Invalid':<8} {'Ranges':<7} "
            f"{'Placeholder':<12} {'BadIDs':<7} {'Named':<7} {'RefList':<8} {'Tok/s':<7} {'Seconds':<7}"
        )
        sep = "-" * len(header)
        print(header)
        print(sep)

        has_failure = False

        out_dir = Path(args.out_dir) if args.out_dir else (root / "artifacts" / "reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        runs_dir = out_dir / "runs"
        if repeat > 1:
            runs_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate stats per mode across all cases and all runs
        agg: dict[str, dict[str, int]] = {
            "rag": {
                "total_reports": 0,
                "all_num_headings": 0,
                "all_any_headings": 0,
                "copied_hdr": 0,
                "range_cit": 0,
                "bad_attack_id": 0,
                "bad_sp": 0,
                "at_least_one_citation": 0,
                "named_kb_id_or_sp": 0,
                "no_citation_and_no_kb_id": 0,
            },
            "vanilla": {
                "total_reports": 0,
                "all_num_headings": 0,
                "all_any_headings": 0,
                "copied_hdr": 0,
                "range_cit": 0,
                "bad_attack_id": 0,
                "bad_sp": 0,
                "at_least_one_citation": 0,
                "named_kb_id_or_sp": 0,
                "no_citation_and_no_kb_id": 0,
            },
        }

        for case_name, exp in ordered_cases:
            query = build_query(exp)
            chunks = retrieve(query, cfg)

            gen_results: dict[str, list[dict[str, Any]]] = {"rag": [], "vanilla": []}
            val_results: dict[str, list[dict[str, Any]]] = {"rag": [], "vanilla": []}
            prompt_dicts: dict[str, dict[str, Any]] = {}

            try:
                for mode in ("rag", "vanilla"):
                    prompt_dicts[mode] = build_prompt(exp, chunks, mode=mode, cfg=cfg)
            except Exception as e:
                print(f"ERROR: Prompt build failed for {case_name}: {e}", file=sys.stderr, flush=True)
                has_failure = True
                continue

            for run_idx in range(1, repeat + 1):
                seed_override = (base_seed + run_idx - 1) if vary_seed else None
                for mode in ("rag", "vanilla"):
                    p = prompt_dicts[mode]
                    seed_msg = f" seed={seed_override}" if seed_override is not None else ""
                    print(
                        f"[PROGRESS] Generating {case_name} ({mode}) run {run_idx}/{repeat}{seed_msg}...",
                        flush=True,
                    )
                    try:
                        gen = generate_report(p, cfg, seed_override=seed_override)
                        n_src = len(p["sources"]) if mode == "rag" else 0
                        val = validate_report(
                            text=gen["text"],
                            n_sources=n_src,
                            mode=mode,
                            cfg=cfg,
                            prompt_user_text=p["user"],
                        )

                        gen_results[mode].append(gen)
                        val_results[mode].append(val)

                        # Accumulate aggregate stats across all runs
                        agg[mode]["total_reports"] += 1
                        if all(val["headings_found"].values()):
                            agg[mode]["all_num_headings"] += 1
                        if val["headings_any_count"] == 5:
                            agg[mode]["all_any_headings"] += 1
                        if val["copied_source_headers"] > 0:
                            agg[mode]["copied_hdr"] += 1
                        if val["citation_ranges"]:
                            agg[mode]["range_cit"] += 1
                        if val["attack_ids_not_in_kb"]:
                            agg[mode]["bad_attack_id"] += 1
                        if val["sp_not_in_kb"]:
                            agg[mode]["bad_sp"] += 1

                        has_cit = len(val["cited"]) > 0
                        has_named_id = (
                            (len(val["kb_attack_ids_named"]) > 0 if val["kb_attack_ids_named"] is not None else False)
                            or (len(val["kb_sp_named"]) > 0 if val["kb_sp_named"] is not None else False)
                        )
                        if has_cit:
                            agg[mode]["at_least_one_citation"] += 1
                        if has_named_id:
                            agg[mode]["named_kb_id_or_sp"] += 1
                        if not has_cit and not has_named_id:
                            agg[mode]["no_citation_and_no_kb_id"] += 1

                        if run_idx == 1:
                            h_num = sum(1 for v in val["headings_found"].values() if v)
                            h_any = val["headings_any_count"]
                            copied = val["copied_source_headers"]
                            cited_str = str(val["cited"])
                            inv_str = str(val["invalid_citations"])
                            ranges_count = len(val["citation_ranges"])
                            placeholder_count = val["placeholder_citations"]
                            bad_ids = val["attack_ids_not_in_kb"]
                            bad_ids_str = str(len(bad_ids)) if bad_ids is not None else "n/a"
                            n_named = (len(val["kb_attack_ids_named"]) if val["kb_attack_ids_named"] is not None else 0) + (len(val["kb_sp_named"]) if val["kb_sp_named"] is not None else 0)
                            named_str = str(n_named) if val["kb_attack_ids_named"] is not None else "n/a"
                            ref_list_str = str(val["has_reference_list"])
                            row = (
                                f"{case_name:<18} {mode:<8} {val['word_count']:<6} "
                                f"{str(val['words_in_range']):<8} {f'{h_num}/5':<7} {f'{h_any}/5':<7} "
                                f"{copied:<10} {cited_str:<12} {inv_str:<8} {ranges_count:<7} "
                                f"{placeholder_count:<12} {bad_ids_str:<7} {named_str:<7} {ref_list_str:<8} "
                                f"{gen['tokens_per_second']:<7.1f} {gen['total_duration']:<7.2f}"
                            )
                            print(row, flush=True)

                            with open(out_dir / f"{case_name}_{mode}.txt", "w", encoding="utf-8") as f:
                                f.write(gen["text"])
                        else:
                            run_file = runs_dir / f"{case_name}_{mode}_r{run_idx}.txt"
                            with open(run_file, "w", encoding="utf-8") as f:
                                f.write(gen["text"])

                    except Exception as e:
                        print(
                            f"ERROR: Generation failed for {case_name} ({mode}) run {run_idx}: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                        has_failure = True

            # Print per-case/mode repeat statistics
            if repeat > 1:
                for mode in ("rag", "vanilla"):
                    runs = val_results[mode]
                    if not runs:
                        continue
                    word_counts = [v["word_count"] for v in runs]
                    all_headings_count = sum(1 for v in runs if all(v["headings_found"].values()))
                    wc_min = min(word_counts)
                    wc_mean = sum(word_counts) / len(word_counts)
                    wc_max = max(word_counts)
                    print(
                        f"  REPEAT STATS {case_name} ({mode}): "
                        f"words min={wc_min} mean={wc_mean:.1f} max={wc_max} "
                        f"all-headings={all_headings_count}/{repeat}",
                        flush=True,
                    )

            # Save JSON report (run-1 data)
            if gen_results["rag"] and gen_results["vanilla"]:
                rag_gen = gen_results["rag"][0]
                van_gen = gen_results["vanilla"][0]
                rag_val = val_results["rag"][0]
                van_val = val_results["vanilla"][0]
                summary = {
                    "case": case_name,
                    "model": rag_gen["model"],
                    "options": rag_gen["options"],
                    "seed_used": rag_gen["seed_used"],
                    "rag": {
                        "prompt": prompt_dicts["rag"],
                        "response": rag_gen["text"],
                        "validation": rag_val,
                        "seed_used": rag_gen["seed_used"],
                        "timings": {
                            "prompt_eval_count": rag_gen["prompt_eval_count"],
                            "prompt_eval_duration": rag_gen["prompt_eval_duration"],
                            "eval_count": rag_gen["eval_count"],
                            "eval_duration": rag_gen["eval_duration"],
                            "total_duration": rag_gen["total_duration"],
                            "load_duration": rag_gen["load_duration"],
                            "tokens_per_second": rag_gen["tokens_per_second"],
                        },
                    },
                    "vanilla": {
                        "prompt": prompt_dicts["vanilla"],
                        "response": van_gen["text"],
                        "validation": van_val,
                        "seed_used": van_gen["seed_used"],
                        "timings": {
                            "prompt_eval_count": van_gen["prompt_eval_count"],
                            "prompt_eval_duration": van_gen["prompt_eval_duration"],
                            "eval_count": van_gen["eval_count"],
                            "eval_duration": van_gen["eval_duration"],
                            "total_duration": van_gen["total_duration"],
                            "load_duration": van_gen["load_duration"],
                            "tokens_per_second": van_gen["tokens_per_second"],
                        },
                    },
                }
                with open(out_dir / f"{case_name}.json", "w", encoding="utf-8") as jf:
                    json.dump(summary, jf, indent=2)

        # Print aggregate block per mode
        print(f"\n{'='*60}", flush=True)
        print("AGGREGATE (across all cases and runs):", flush=True)
        for mode in ("rag", "vanilla"):
            a = agg[mode]
            tot = a["total_reports"]
            print(
                f"  [{mode}] total-reports={tot} "
                f"all-5-numbered-headings={a['all_num_headings']}/{tot} "
                f"all-5-any-headings={a['all_any_headings']}/{tot} "
                f"copied-src-hdr={a['copied_hdr']}/{tot} "
                f"range-cit={a['range_cit']}/{tot} "
                f"bad-attack-id={a['bad_attack_id']}/{tot} "
                f"bad-sp={a['bad_sp']}/{tot} "
                f"at-least-one-citation={a['at_least_one_citation']}/{tot} "
                f"named-kb-id-or-sp={a['named_kb_id_or_sp']}/{tot} "
                f"no-citation-no-kb-id={a['no_citation_and_no_kb_id']}/{tot}",
                flush=True,
            )
        print("=" * 60, flush=True)

        if has_failure:
            sys.exit(1)


if __name__ == "__main__":
    main()
