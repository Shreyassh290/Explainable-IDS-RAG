"""Unit tests for local LLM generation, Ollama checking, and report validation (tests/test_generate.py).

Tests run entirely in-memory with injected fake clients:
1. generate_report passes options and keep_alive from config and returns model text unchanged.
2. check_ollama raises RuntimeError with 'ollama pull' when model is missing, and supports both
   response structures and missing ':latest' suffixes.
3. validate_report counts words, finds markdown-formatted headings, extracts bracketed citations,
   flags out-of-range citations in RAG / any citations in vanilla, catches unsupported IPv4s,
   counts placeholder "[n]", extracts citation ranges, detects reference lists, checks KB IDs.
4. run_case saves exact model text files and structured JSON report summary.
5. Calling generate_report with an invalid mode raises ValueError.
6. validate_report new fields: placeholders, ranges, ref list, KB IDs.
7. --repeat 2 writes the plain file from run 1 and runs/ files for run 2.
8. headings_any detects unnumbered headings; headings_found (numbered) stays strict.
9. copied_source_headers strips header lines from citation counting; has_reference_list is True.
10. --vary-seed passes correct seeds to generate_report and records them in JSON.
"""

import json
from pathlib import Path
from typing import Any
import pytest

from ids_rag.llm.generate import check_ollama, generate_report, run_case, validate_report


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChatResponse:
    def __init__(
        self,
        content: str,
        model: str = "llama3:8b",
        prompt_eval_count: int = 150,
        prompt_eval_duration: int = 500_000_000,
        eval_count: int = 200,
        eval_duration: int = 2_000_000_000,
        total_duration: int = 2_600_000_000,
        load_duration: int = 100_000_000,
    ):
        self.message = FakeMessage(content)
        self.model = model
        self.prompt_eval_count = prompt_eval_count
        self.prompt_eval_duration = prompt_eval_duration
        self.eval_count = eval_count
        self.eval_duration = eval_duration
        self.total_duration = total_duration
        self.load_duration = load_duration


class FakeClient:
    def __init__(self, chat_response: Any = None, list_response: Any = None):
        self.chat_response = chat_response or FakeChatResponse("Synthetic report content.")
        self.list_response = list_response or {"models": [{"name": "llama3:8b:latest"}]}
        self.last_chat_args: dict[str, Any] = {}

    def chat(self, **kwargs: Any) -> Any:
        self.last_chat_args = kwargs
        return self.chat_response

    def list(self) -> Any:
        return self.list_response


# ---------------------------------------------------------------------------
# Test 1: generate_report passes options and returns text unchanged
# ---------------------------------------------------------------------------


def test_generate_report_passes_options_and_returns_text_unchanged() -> None:
    cfg = {
        "llm": {
            "model": "llama3:8b",
            "temperature": 0.2,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 1100,
            "keep_alive": "15m",
            "request_timeout_s": 600,
        }
    }
    fake_text = "This is exact un-edited output text from model."
    fake_client = FakeClient(chat_response=FakeChatResponse(fake_text))

    prompt = {
        "mode": "rag",
        "system": "System instructions.",
        "user": "User prompt content.",
    }

    result = generate_report(prompt, cfg, client=fake_client)

    assert result["text"] == fake_text

    called_opts = fake_client.last_chat_args["options"]
    assert called_opts["temperature"] == 0.2
    assert called_opts["seed"] == 42
    assert called_opts["num_ctx"] == 8192
    assert called_opts["num_predict"] == 1100
    assert fake_client.last_chat_args["keep_alive"] == "15m"
    assert fake_client.last_chat_args["model"] == "llama3:8b"

    assert result["prompt_eval_duration"] == 0.5
    assert result["eval_duration"] == 2.0
    assert result["total_duration"] == 2.6
    assert result["load_duration"] == 0.1
    assert result["tokens_per_second"] == 100.0

    # seed_used must be recorded in result
    assert result["seed_used"] == 42


# ---------------------------------------------------------------------------
# Test 2: check_ollama raises on missing model and supports both response shapes
# ---------------------------------------------------------------------------


def test_check_ollama_model_missing_and_response_shapes() -> None:
    cfg = {"llm": {"model": "llama3:8b", "host": "http://localhost:11434"}}

    fake_missing = FakeClient(list_response={"models": [{"name": "mistral:7b"}]})
    with pytest.raises(RuntimeError, match="ollama pull llama3:8b"):
        check_ollama(cfg, client=fake_missing)

    fake_dict = FakeClient(list_response={"models": [{"name": "llama3:8b"}]})
    models = check_ollama(cfg, client=fake_dict)
    assert "llama3:8b" in models

    class ModelItem:
        def __init__(self, m: str):
            self.model = m

    class ListRespObj:
        def __init__(self, items: list[Any]):
            self.models = items

    fake_obj = FakeClient(list_response=ListRespObj([ModelItem("llama3:8b")]))
    models_obj = check_ollama(cfg, client=fake_obj)
    assert "llama3:8b" in models_obj

    fake_latest = FakeClient(list_response={"models": [{"name": "llama3:8b:latest"}]})
    assert "llama3:8b:latest" in check_ollama(cfg, client=fake_latest)

    cfg_latest = {"llm": {"model": "llama3:8b:latest"}}
    assert "llama3:8b" in check_ollama(cfg_latest, client=fake_dict)


# ---------------------------------------------------------------------------
# Test 3: validate_report headings, citations, and IP extraction (existing)
# ---------------------------------------------------------------------------


def test_validate_report_headings_citations_and_ips() -> None:
    cfg = {"llm": {"min_words": 10, "max_words": 100}}

    prompt_user_text = "Incident on host with IP 192.168.1.100 communicating with 10.0.0.1."

    text = (
        "1. Rationale\n"
        "Normal traffic was observed with 192.168.1.100.\n\n"
        "**2. Key Indicators**\n"
        "High volume of connections observed [1] and [2, 3].\n\n"
        "## 3. Confidence Assessment\n"
        "Assessment supported by references [4][5] and [7].\n\n"
        "### 4. Threat Assessment\n"
        "Threat level is high due to communication with external IP 172.16.50.4.\n"
    )

    val_rag = validate_report(text, n_sources=5, mode="rag", cfg=cfg, prompt_user_text=prompt_user_text)

    assert val_rag["word_count"] > 10
    assert val_rag["words_in_range"] is True

    assert val_rag["headings_found"]["Rationale"] is True
    assert val_rag["headings_found"]["Key Indicators"] is True
    assert val_rag["headings_found"]["Confidence Assessment"] is True
    assert val_rag["headings_found"]["Threat Assessment"] is True
    assert val_rag["headings_found"]["Recommendations"] is False

    assert val_rag["cited"] == [1, 2, 3, 4, 5, 7]
    assert val_rag["invalid_citations"] == [7]
    assert val_rag["citations_in_vanilla"] == []

    assert "172.16.50.4" in val_rag["unsupported_ips"]
    assert "192.168.1.100" not in val_rag["unsupported_ips"]

    val_vanilla = validate_report(text, n_sources=0, mode="vanilla", cfg=cfg, prompt_user_text=prompt_user_text)
    assert val_vanilla["cited"] == [1, 2, 3, 4, 5, 7]
    assert val_vanilla["citations_in_vanilla"] == [1, 2, 3, 4, 5, 7]
    assert val_vanilla["invalid_citations"] == [1, 2, 3, 4, 5, 7]


# ---------------------------------------------------------------------------
# Test 4: run_case saves exact texts and json summary
# ---------------------------------------------------------------------------


def test_run_case_saves_exact_texts_and_json(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr("ids_rag.llm.generate.get_project_root", lambda: tmp_path)

    cfg = {
        "llm": {
            "model": "llama3:8b",
            "temperature": 0.2,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 1100,
            "keep_alive": "15m",
            "min_words": 10,
            "max_words": 500,
            "tier_actions": {"VERY_HIGH": "Very High: automated"},
            "suppress_near_average_implications": False,
        }
    }

    explanation = {
        "predicted_class": "DoS",
        "confidence": 0.99,
        "tier": "VERY_HIGH",
        "features": [],
        "metadata": {"Flow ID": "test-flow"},
    }

    chunks = [
        {
            "chunk_id": "c1",
            "text": "Sample chunk text.",
            "source_name": "NIST_SP_800-61_Rev2.pdf",
            "source_type": "NIST",
            "source_ref": "SP 800-61 Rev2",
            "section": "3.2 Incident Handling",
            "is_fallback": False,
        }
    ]

    rag_text = "Exact model output text for RAG mode."
    vanilla_text = "Exact model output text for Vanilla mode."

    class ModeClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, **kwargs: Any) -> Any:
            self.calls += 1
            txt = rag_text if self.calls % 2 == 1 else vanilla_text
            return FakeChatResponse(txt)

    client = ModeClient()
    rag_val, van_val = run_case("test_case_1", explanation, chunks, cfg, client=client)

    reports_dir = tmp_path / "artifacts" / "reports"
    rag_file = reports_dir / "test_case_1_rag.txt"
    van_file = reports_dir / "test_case_1_vanilla.txt"
    json_file = reports_dir / "test_case_1.json"

    assert rag_file.exists()
    assert rag_file.read_text(encoding="utf-8") == rag_text

    assert van_file.exists()
    assert van_file.read_text(encoding="utf-8") == vanilla_text

    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["case"] == "test_case_1"
    assert "rag" in data
    assert "vanilla" in data
    assert data["rag"]["validation"]["word_count"] == len(rag_text.split())
    assert data["vanilla"]["validation"]["word_count"] == len(vanilla_text.split())


# ---------------------------------------------------------------------------
# Test 5: generate_report with invalid mode raises ValueError
# ---------------------------------------------------------------------------


def test_generate_report_invalid_mode_raises_value_error() -> None:
    cfg = {"llm": {"model": "llama3:8b"}}
    fake_client = FakeClient()
    invalid_prompt = {
        "mode": "unsupported_mode",
        "system": "System instructions.",
        "user": "User prompt.",
    }
    with pytest.raises(ValueError, match="Invalid mode"):
        generate_report(invalid_prompt, cfg, client=fake_client)


# ---------------------------------------------------------------------------
# Test 6: validate_report new fields — placeholders, ranges, ref list, KB IDs
# ---------------------------------------------------------------------------


def test_validate_report_new_fields(tmp_path: Path) -> None:
    cfg = {"llm": {"min_words": 1, "max_words": 10000}}

    kb_file = tmp_path / "chunks.jsonl"
    kb_rows = [
        {"text": "Network flooding technique T1498 is documented here.", "source_name": "NIST_SP_800-61_Rev2.pdf"},
        {"text": "Some other general guidance text.", "source_name": "NIST_SP_800-94.pdf"},
    ]
    with open(kb_file, "w", encoding="utf-8") as f:
        for row in kb_rows:
            f.write(json.dumps(row) + "\n")

    text = (
        "The attack T1498 and T1498.001 was confirmed [n] but T1489 was not in KB. "
        "Reference [1-5] and [2\u20134] are ranges. "
        "SP 800-61 is covered; SP 800-99 is not.\n"
        "References:\n"
        "- Some source."
    )

    val = validate_report(
        text=text,
        n_sources=10,
        mode="rag",
        cfg=cfg,
        prompt_user_text="",
        kb_path=kb_file,
    )

    assert val["placeholder_citations"] == 1

    assert len(val["citation_ranges"]) == 2
    ranges_joined = " ".join(val["citation_ranges"])
    assert "1" in ranges_joined and "5" in ranges_joined
    assert "2" in ranges_joined and "4" in ranges_joined

    # Ranges excluded from cited; "[n]" is not a numeric citation
    assert val["cited"] == []

    assert val["has_reference_list"] is True

    # An ID present in stub KB goes to kb_attack_ids_named; absent goes to attack_ids_not_in_kb
    assert val["attack_ids_not_in_kb"] is not None
    assert "T1489" in val["attack_ids_not_in_kb"]
    assert "T1498.001" in val["attack_ids_not_in_kb"]

    assert val["kb_attack_ids_named"] is not None
    assert "T1498" in val["kb_attack_ids_named"]

    assert val["sp_not_in_kb"] is not None
    assert "800-99" in val["sp_not_in_kb"]

    assert val["kb_sp_named"] is not None
    assert "800-61" in val["kb_sp_named"]


# ---------------------------------------------------------------------------
# Test 7: --repeat 2 writes plain file from run 1 and runs/ files for run 2
# ---------------------------------------------------------------------------


def test_run_case_repeat_writes_runs_files(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr("ids_rag.llm.generate.get_project_root", lambda: tmp_path)

    cfg = {
        "llm": {
            "model": "llama3:8b",
            "temperature": 0.2,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 1100,
            "keep_alive": "15m",
            "min_words": 1,
            "max_words": 10000,
            "tier_actions": {"VERY_HIGH": "Very High: automated"},
            "suppress_near_average_implications": False,
        }
    }

    explanation = {
        "predicted_class": "DoS",
        "confidence": 0.99,
        "tier": "VERY_HIGH",
        "features": [],
        "metadata": {},
    }

    call_counter = {"n": 0}
    texts = ["run1-rag", "run1-vanilla", "run2-rag", "run2-vanilla"]

    class RepeatClient:
        def chat(self, **kwargs: Any) -> Any:
            idx = call_counter["n"]
            call_counter["n"] += 1
            return FakeChatResponse(texts[idx] if idx < len(texts) else "extra")

    run_case("mycase", explanation, chunks=None, cfg=cfg, client=RepeatClient(), repeat=2)

    reports_dir = tmp_path / "artifacts" / "reports"
    runs_dir = reports_dir / "runs"

    assert (reports_dir / "mycase_rag.txt").read_text(encoding="utf-8") == "run1-rag"
    assert (reports_dir / "mycase_vanilla.txt").read_text(encoding="utf-8") == "run1-vanilla"
    assert (runs_dir / "mycase_rag_r2.txt").read_text(encoding="utf-8") == "run2-rag"
    assert (runs_dir / "mycase_vanilla_r2.txt").read_text(encoding="utf-8") == "run2-vanilla"
    assert (reports_dir / "mycase.json").exists()


# ---------------------------------------------------------------------------
# Test 8: headings_any detects unnumbered / variously formatted headings
# ---------------------------------------------------------------------------


def test_headings_any_detects_unnumbered_headings() -> None:
    cfg = {"llm": {"min_words": 1, "max_words": 10000}}

    # Text where:
    #   "**Rationale**"              -> headings_any True, headings_found False
    #   "## 2) Key Indicators:"      -> headings_any True, headings_found False (uses ")" not ".")
    #   "3. Confidence Assessment"   -> BOTH True (numbered, exact)
    #   "Threat Assessment:"         -> headings_any True (trailing colon), headings_found False
    #   (no Recommendations)         -> both False
    text = (
        "**Rationale**\n"
        "Some rationale text here.\n\n"
        "## 2) Key Indicators:\n"
        "Some indicator text.\n\n"
        "3. Confidence Assessment\n"
        "Confidence text.\n\n"
        "Threat Assessment:\n"
        "Threat text.\n"
    )

    val = validate_report(text, n_sources=0, mode="vanilla", cfg=cfg)

    # headings_any
    assert val["headings_any"]["Rationale"] is True
    assert val["headings_any"]["Key Indicators"] is True
    assert val["headings_any"]["Confidence Assessment"] is True
    assert val["headings_any"]["Threat Assessment"] is True
    assert val["headings_any"]["Recommendations"] is False
    assert val["headings_any_count"] == 4

    # headings_found (numbered) — only "3. Confidence Assessment" matches
    assert val["headings_found"]["Rationale"] is False
    assert val["headings_found"]["Key Indicators"] is False
    assert val["headings_found"]["Confidence Assessment"] is True
    assert val["headings_found"]["Threat Assessment"] is False
    assert val["headings_found"]["Recommendations"] is False


# ---------------------------------------------------------------------------
# Test 9a: copied source headers are stripped from citation counting
# ---------------------------------------------------------------------------


def test_copied_source_headers_stripped_from_citations() -> None:
    cfg = {"llm": {"min_words": 1, "max_words": 10000}}

    # Text with copied source headers at the end
    text = (
        "The attack was mitigated using standard procedures.\n"
        "[1] MITRE ATT&CK T1498.001, section Description\n"
        "- [2] NIST SP 800-61 Rev2, section 3.6\n"
    )

    val = validate_report(text, n_sources=5, mode="rag", cfg=cfg)

    # copied_source_headers: 2 lines matched
    assert val["copied_source_headers"] == 2
    # cited_in_copied_headers: [1, 2]
    assert val["cited_in_copied_headers"] == [1, 2]
    # cited: 1 and 2 came from copied headers, not from inline citations → empty
    assert val["cited"] == []
    # has_reference_list must be True when copied headers > 0
    assert val["has_reference_list"] is True


# ---------------------------------------------------------------------------
# Test 9b: inline citation kept when same number also appears in copied header
# ---------------------------------------------------------------------------


def test_inline_citation_kept_when_also_in_copied_header() -> None:
    cfg = {"llm": {"min_words": 1, "max_words": 10000}}

    # [2] appears inline in a sentence AND in a copied header line
    text = (
        "According to the report [2], the attack was confirmed.\n"
        "[1] MITRE ATT&CK T1498.001, section Description\n"
        "- [2] NIST SP 800-61 Rev2, section 3.6\n"
    )

    val = validate_report(text, n_sources=5, mode="rag", cfg=cfg)

    assert val["copied_source_headers"] == 2
    assert val["cited_in_copied_headers"] == [1, 2]
    # [2] is in the remaining (non-header) text as an inline citation → cited = [2]
    assert val["cited"] == [2]
    assert val["has_reference_list"] is True


# ---------------------------------------------------------------------------
# Test 10: --vary-seed passes correct seeds and records them in JSON
# ---------------------------------------------------------------------------


def test_vary_seed_passes_correct_seeds_and_records_in_json(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr("ids_rag.llm.generate.get_project_root", lambda: tmp_path)

    cfg = {
        "llm": {
            "model": "llama3:8b",
            "temperature": 0.2,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 1100,
            "keep_alive": "15m",
            "min_words": 1,
            "max_words": 10000,
            "tier_actions": {"VERY_HIGH": "Very High: automated"},
            "suppress_near_average_implications": False,
        }
    }

    explanation = {
        "predicted_class": "DoS",
        "confidence": 0.99,
        "tier": "VERY_HIGH",
        "features": [],
        "metadata": {},
    }

    # Capture seeds passed to chat options
    seeds_seen: list[int] = []

    class SeedCapturingClient:
        def chat(self, **kwargs: Any) -> Any:
            seeds_seen.append(kwargs["options"]["seed"])
            return FakeChatResponse(f"text-seed-{kwargs['options']['seed']}")

    # With vary_seed=True, repeat=2: run 1 → seed 42, run 2 → seed 43
    run_case(
        "seedcase",
        explanation,
        chunks=None,
        cfg=cfg,
        client=SeedCapturingClient(),
        repeat=2,
        vary_seed=True,
    )

    # 4 calls: run1-rag(42), run1-vanilla(42), run2-rag(43), run2-vanilla(43)
    assert seeds_seen == [42, 42, 43, 43]

    # JSON records seed_used for run 1 (seed 42)
    json_data = json.loads((tmp_path / "artifacts" / "reports" / "seedcase.json").read_text())
    assert json_data["seed_used"] == 42
    assert json_data["rag"]["seed_used"] == 42
    assert json_data["vanilla"]["seed_used"] == 42


def test_vary_seed_false_all_runs_use_base_seed(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr("ids_rag.llm.generate.get_project_root", lambda: tmp_path)

    cfg = {
        "llm": {
            "model": "llama3:8b",
            "temperature": 0.2,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 1100,
            "keep_alive": "15m",
            "min_words": 1,
            "max_words": 10000,
            "tier_actions": {"VERY_HIGH": "Very High: automated"},
            "suppress_near_average_implications": False,
        }
    }

    explanation = {
        "predicted_class": "DoS",
        "confidence": 0.99,
        "tier": "VERY_HIGH",
        "features": [],
        "metadata": {},
    }

    seeds_seen: list[int] = []

    class SeedCapturingClient:
        def chat(self, **kwargs: Any) -> Any:
            seeds_seen.append(kwargs["options"]["seed"])
            return FakeChatResponse("text")

    # vary_seed=False (default): all runs use seed 42
    run_case(
        "seedcase2",
        explanation,
        chunks=None,
        cfg=cfg,
        client=SeedCapturingClient(),
        repeat=2,
        vary_seed=False,
    )

    # All 4 calls use seed 42
    assert all(s == 42 for s in seeds_seen)
    assert len(seeds_seen) == 4


# ---------------------------------------------------------------------------
# Test 11: aggregate block counts all runs (2 cases, repeat 2 -> 4 runs)
# ---------------------------------------------------------------------------


def test_aggregate_counts_all_runs(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    """Verify that aggregate block counts over all runs of a mode (every case and every seed)."""
    monkeypatch.setattr("ids_rag.llm.generate.get_project_root", lambda: tmp_path)

    art_dir = tmp_path / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        {
            "case": "case1",
            "predicted_class": "DoS",
            "confidence": 0.99,
            "tier": "VERY_HIGH",
            "features": [],
            "metadata": {},
        },
        {
            "case": "case2",
            "predicted_class": "Benign",
            "confidence": 0.95,
            "tier": "HIGH",
            "features": [],
            "metadata": {},
        },
    ]
    with open(art_dir / "sample_explanations.json", "w", encoding="utf-8") as f:
        json.dump(samples, f)

    monkeypatch.setattr("ids_rag.llm.generate.check_ollama", lambda cfg: ["llama3:8b"])
    monkeypatch.setattr("ids_rag.llm.generate.retrieve", lambda q, cfg: [])

    def fake_generate(prompt: Any, cfg: Any, client: Any = None, seed_override: Any = None) -> dict[str, Any]:
        return {
            "text": (
                "1. Rationale\nReport.\n"
                "2. Key Indicators\nReport.\n"
                "3. Confidence Assessment\nReport.\n"
                "4. Threat Assessment\nReport.\n"
                "5. Recommendations\nReport."
            ),
            "model": "llama3:8b",
            "options": {"seed": seed_override or 42},
            "seed_used": seed_override or 42,
            "prompt_words": 10,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 0.1,
            "eval_count": 20,
            "eval_duration": 0.2,
            "total_duration": 0.3,
            "load_duration": 0.05,
            "tokens_per_second": 100.0,
        }

    monkeypatch.setattr("ids_rag.llm.generate.generate_report", fake_generate)

    from ids_rag.llm.generate import main
    monkeypatch.setattr("sys.argv", ["generate", "--examples", "--repeat", "2"])

    main()

    captured = capsys.readouterr().out
    assert "[rag] total-reports=4" in captured
    assert "[vanilla] total-reports=4" in captured

