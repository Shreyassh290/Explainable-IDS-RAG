"""End-to-end incident response report generation pipeline.

Executes Stage 6 and Stage 7 of the architecture:
- Loads ensemble models, scaler, feature names, knowledge base index, cross-encoder,
  and evaluation datasets once into a reusable context.
- For a given test flow index:
  1. Computes ensemble prediction (predicted class, confidence, confidence tier).
  2. Generates top-k feature attribution explanations using explain_flow.
  3. Formulates a retrieval query without raw identifiers and retrieves KB chunks.
  4. Builds structured prompts for RAG and/or vanilla modes.
  5. Generates incident response reports via local LLM (Ollama).
  6. Validates generated reports against citation, formatting, and content constraints.
  7. Emits structured outputs to artifacts/reports/flows/ (or custom --out-dir).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ids_rag.config import get_project_root, load_config
from ids_rag.explain.shap_topk import explain_flow
from ids_rag.llm.generate import check_ollama, generate_report, validate_report
from ids_rag.llm.prompt import build_prompt
from ids_rag.models.ensemble import BinaryDNNEnsemble
from ids_rag.rag.index import KBIndex, compute_file_sha256, load_index
from ids_rag.rag.retrieve import build_query, get_cached_cross_encoder, retrieve


def get_ollama_version() -> str:
    """Run 'ollama --version' with 10s timeout, returning stdout text or 'UNKNOWN'."""
    try:
        proc = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout.strip()
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def write_summary_csv(
    csv_path: Path | str,
    flow_summaries: list[dict[str, Any]],
    out_dir: Path | str,
    modes: list[str],
) -> None:
    """Write summary CSV of all evaluated flows and modes."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir_path = Path(out_dir)

    columns = [
        "index",
        "true_class",
        "predicted_class",
        "is_correct",
        "confidence",
        "tier",
        "mode",
        "words",
        "words_in_range",
        "headings_numbered_count",
        "headings_any_count",
        "cited",
        "citation_ranges_count",
        "copied_source_headers",
        "attack_ids_not_in_kb",
        "kb_attack_ids_named",
        "sp_not_in_kb",
        "generate_seconds",
        "seed_used",
        "report_file",
    ]

    rows: list[dict[str, Any]] = []
    for summary in flow_summaries:
        idx = summary["index"]
        true_cls = summary.get("true_class", summary.get("ground_truth", ""))
        pred_cls = summary.get("predicted_class", "")
        is_corr = summary.get("is_correct", False)
        conf = summary.get("confidence", 0.0)
        tier = summary.get("tier", "")
        gen_sec = summary.get("timings", {}).get("generate", 0.0)
        seed = summary.get("seed_used", summary.get("options", {}).get("seed", 42))

        # Check which modes are present in this summary
        available_modes = [m for m in modes if m in summary.get("validation", {})]
        if not available_modes:
            available_modes = modes

        for m in available_modes:
            val = summary.get("validation", {}).get(m, {})
            h_num = sum(1 for v in val.get("headings_found", {}).values() if v)
            h_any = val.get("headings_any_count", 0)
            cited_str = ";".join(str(c) for c in val.get("cited", []))
            ranges_cnt = len(val.get("citation_ranges", []))
            copied_hdr = val.get("copied_source_headers", 0)

            bad_attacks = ";".join(val.get("attack_ids_not_in_kb") or [])
            named_attacks = ";".join(val.get("kb_attack_ids_named") or [])
            bad_sps = ";".join(val.get("sp_not_in_kb") or [])

            rep_file = str(out_dir_path / f"flow_{idx}_{m}.txt")

            rows.append(
                {
                    "index": idx,
                    "true_class": true_cls,
                    "predicted_class": pred_cls,
                    "is_correct": is_corr,
                    "confidence": conf,
                    "tier": tier,
                    "mode": m,
                    "words": val.get("word_count", 0),
                    "words_in_range": val.get("words_in_range", False),
                    "headings_numbered_count": h_num,
                    "headings_any_count": h_any,
                    "cited": cited_str,
                    "citation_ranges_count": ranges_cnt,
                    "copied_source_headers": copied_hdr,
                    "attack_ids_not_in_kb": bad_attacks,
                    "kb_attack_ids_named": named_attacks,
                    "sp_not_in_kb": bad_sps,
                    "generate_seconds": gen_sec,
                    "seed_used": seed,
                    "report_file": rep_file,
                }
            )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class PipelineContext:
    """Container for models, index, scaler, and datasets loaded once across runs."""

    def __init__(
        self,
        ensemble: BinaryDNNEnsemble | Any,
        scaler: Any,
        feature_names: list[str],
        kb: KBIndex | Any,
        cross_encoder: Any,
        df_test: pd.DataFrame,
        df_meta: pd.DataFrame | None,
        df_preds: pd.DataFrame | None,
        chunks_sha256: str = "",
    ) -> None:
        self.ensemble = ensemble
        self.scaler = scaler
        self.feature_names = feature_names
        self.kb = kb
        self.cross_encoder = cross_encoder
        self.df_test = df_test
        self.df_meta = df_meta
        self.df_preds = df_preds
        self.chunks_sha256 = chunks_sha256

    @classmethod
    def load(cls, cfg: dict[str, Any] | None = None) -> "PipelineContext":
        """Load ensemble, scaler, feature names, KB index, and datasets from disk.

        Performs stale index verification via load_index.
        """
        if cfg is None:
            cfg = load_config()

        root = get_project_root()
        paths_cfg = cfg.get("paths", {})
        models_dir = root / paths_cfg.get("models", "models")
        processed_dir = root / paths_cfg.get("processed", "data/processed")
        artifacts_dir = root / paths_cfg.get("artifacts", "artifacts")
        kb_path = root / paths_cfg.get("kb", "data/kb") / "chunks.jsonl"

        # 1. Load KB index (performs SHA-256 stale check against chunks.jsonl)
        kb = load_index(cfg)
        cross_encoder = get_cached_cross_encoder(cfg)

        # 2. Load ensemble and scaler
        ensemble = BinaryDNNEnsemble.load(models_dir=models_dir, use_temperature=False)
        scaler = joblib.load(models_dir / "scaler.joblib")
        feature_names = ensemble.feature_names

        # 3. Load test data, metadata, and test predictions
        test_path = processed_dir / "test.parquet"
        meta_path = processed_dir / "test_meta.parquet"
        preds_path = artifacts_dir / "test_predictions.parquet"

        if not test_path.exists():
            raise FileNotFoundError(f"Test dataset not found at: {test_path}")
        df_test = pd.read_parquet(test_path)

        df_meta = pd.read_parquet(meta_path) if meta_path.exists() else None
        df_preds = pd.read_parquet(preds_path) if preds_path.exists() else None

        chunks_sha256 = compute_file_sha256(kb_path) if kb_path.exists() else ""

        return cls(
            ensemble=ensemble,
            scaler=scaler,
            feature_names=feature_names,
            kb=kb,
            cross_encoder=cross_encoder,
            df_test=df_test,
            df_meta=df_meta,
            df_preds=df_preds,
            chunks_sha256=chunks_sha256,
        )


def validate_raw_features(
    raw_features: dict[str, Any] | "pd.Series",
    feature_names: list[str],
    scaler: Any,
) -> np.ndarray:
    """Validate raw (unscaled) feature input and return a scaled float32 array.

    Args:
        raw_features: Mapping of feature name to value (dict or pandas Series).
            CSV headers are stripped of surrounding whitespace before lookup.
        feature_names: Ordered list of expected feature names.
        scaler: A fitted sklearn-compatible scaler with a ``transform`` method.

    Returns:
        1-D float32 numpy array of shape ``(len(feature_names),)`` after scaling.

    Raises:
        ValueError: If any feature is missing, non-numeric, NaN, or Inf.
    """
    # Normalise to dict; strip whitespace from string keys
    if hasattr(raw_features, "to_dict"):
        raw_dict: dict[str, Any] = raw_features.to_dict()
    else:
        raw_dict = dict(raw_features)
    raw_dict = {str(k).strip(): v for k, v in raw_dict.items()}

    # Check for missing features
    missing = [fn for fn in feature_names if fn not in raw_dict]
    if missing:
        raise ValueError(
            f"Missing required features: {missing}. "
            f"Expected all of: {feature_names}"
        )

    # Convert to float; name any non-numeric feature
    values: list[float] = []
    for fn in feature_names:
        try:
            values.append(float(raw_dict[fn]))
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature '{fn}' has a non-numeric value: {raw_dict[fn]!r}. "
                "All features must be numeric."
            )

    arr = np.array(values, dtype=np.float64)

    # Reject NaN or Inf (the training pipeline also dropped such rows)
    if not np.all(np.isfinite(arr)):
        bad = [feature_names[i] for i, v in enumerate(arr) if not np.isfinite(v)]
        raise ValueError(
            f"Features contain NaN or Inf values: {bad}. "
            "The training pipeline dropped rows with NaN/Inf; "
            "such values are not supported at inference time."
        )

    df_features = pd.DataFrame([arr], columns=feature_names)
    scaled = scaler.transform(df_features)
    if hasattr(scaled, "to_numpy"):
        scaled = scaled.to_numpy()
    scaled = np.asarray(scaled)[0]
    return scaled.astype(np.float32)


def _run_core(
    x_scaled: np.ndarray,
    meta_row: "pd.Series | dict[str, Any] | None",
    ctx: PipelineContext,
    cfg: dict[str, Any],
    index_or_id: "int | str",
    mode_list: list[str],
    ground_truth: str = "UNKNOWN",
    seed: int | None = None,
    generate: bool = True,
    client: Any = None,
    out_dir: "Path | str | None" = None,
    ollama_version: str | None = None,
) -> dict[str, Any]:
    """Internal shared core: predict → explain → retrieve → prompt → generate → validate → write.

    Args:
        x_scaled: 1-D float32 numpy array of already-scaled feature values.
        meta_row: Optional flow metadata (pandas Series, dict, or None).
        ctx: Preloaded PipelineContext instance.
        cfg: System configuration dictionary.
        index_or_id: Integer index or string identifier used in output filenames
            and the ``"index"`` JSON key.
        mode_list: Normalised list of generation modes, e.g. ``["rag"]``.
        ground_truth: Ground-truth class string for console display only.
        seed: Optional random seed override for generation.
        generate: If False, stops after prompt construction (writes prompt files).
        client: Optional injected Ollama Client instance for testing.
        out_dir: Directory for reports and JSON.
        ollama_version: Recorded Ollama version string.

    Returns:
        Summary dict containing all run metadata, explanation, query, chunks,
        prompts, reports, validations, and timings.
    """
    t0_total = time.perf_counter()

    # 4. Ensemble prediction
    t0_ens = time.perf_counter()
    pred_labels, confidences, tiers = ctx.ensemble.predict(x_scaled.reshape(1, -1))
    pred_label = int(pred_labels[0])
    confidence = float(confidences[0])
    tier = str(tiers[0])
    predicted_class = ctx.ensemble.label_map.get(pred_label, str(pred_label))
    t_ensemble = time.perf_counter() - t0_ens

    is_correct = bool(predicted_class == ground_truth)

    # 5. Top-k feature attribution (ground truth is NOT passed)
    t0_exp = time.perf_counter()
    explain_cfg = cfg.get("explain", {})
    method = explain_cfg.get("method", "gradient")
    top_k = int(explain_cfg.get("top_k", 5))

    explanation = explain_flow(
        x_scaled=x_scaled,
        ensemble=ctx.ensemble,
        k=top_k,
        scaler=ctx.scaler,
        meta_row=meta_row,
        feature_names=ctx.feature_names,
        method=method,
    )
    t_explain = time.perf_counter() - t0_exp

    # 6. Build retrieval query and retrieve chunks
    t0_ret = time.perf_counter()
    query = build_query(explanation)
    chunks = retrieve(query, cfg=cfg, kb=ctx.kb, cross_encoder=ctx.cross_encoder)
    t_retrieve = time.perf_counter() - t0_ret

    # 7. Build prompts for requested modes
    t0_prompt = time.perf_counter()
    prompts_dict: dict[str, dict[str, Any]] = {}
    for m in mode_list:
        prompts_dict[m] = build_prompt(explanation, chunks, mode=m, cfg=cfg)
    t_prompt = time.perf_counter() - t0_prompt

    # 8. Generation and automated report validation
    reports_dict: dict[str, str] = {}
    validations_dict: dict[str, dict[str, Any]] = {}
    gen_results_dict: dict[str, dict[str, Any]] = {}

    t0_gen = time.perf_counter()
    t_validate = 0.0

    if generate:
        for m in mode_list:
            gen = generate_report(
                prompt=prompts_dict[m],
                cfg=cfg,
                client=client,
                seed_override=seed,
            )
            gen_results_dict[m] = gen
            reports_dict[m] = gen["text"]

            t0_v = time.perf_counter()
            n_src = len(prompts_dict[m]["sources"]) if m == "rag" else 0
            val = validate_report(
                text=gen["text"],
                n_sources=n_src,
                mode=m,
                cfg=cfg,
                prompt_user_text=prompts_dict[m]["user"],
            )
            t_validate += time.perf_counter() - t0_v
            validations_dict[m] = val
    t_generate = time.perf_counter() - t0_gen - t_validate

    t_total = time.perf_counter() - t0_total

    timings = {
        "ensemble": round(t_ensemble, 4),
        "explain": round(t_explain, 4),
        "retrieve": round(t_retrieve, 4),
        "prompt": round(t_prompt, 4),
        "generate": round(t_generate, 4),
        "validate": round(t_validate, 4),
        "total": round(t_total, 4),
    }

    # 9. Format chunk records for JSON export
    chunk_records: list[dict[str, Any]] = []
    for c in chunks:
        chunk_records.append(
            {
                "chunk_id": c.get("chunk_id"),
                "source": c.get("source_name") or c.get("source_ref", ""),
                "section": c.get("section", ""),
                "bm25_rank": c.get("bm25_rank"),
                "dense_rank": c.get("dense_rank"),
                "ce_score": c.get("ce_score"),
                "is_fallback": c.get("is_fallback", False),
            }
        )

    # 10. Console output
    print(f"\nFlow Index: {index_or_id}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Tier: {tier}")
    print(f"Ground truth (for evaluation only, not used by the pipeline): {ground_truth}")
    print(f"Prediction correct: {is_correct}")

    print("\nTop Features:")
    for rank, f in enumerate(explanation.get("features", [])[:5], 1):
        print(f"  {rank}. {f['name']} = {f['original_value']} ({f['direction']}, importance: {f.get('importance', 0):.6f})")

    print("\nRetrieved Sources:")
    for c in chunks:
        src = c.get("source_name") or c.get("source_ref", "")
        ce_val = c.get("ce_score")
        ce_str = f"{ce_val:.4f}" if ce_val is not None else "n/a"
        print(f"  Rank {c.get('rank', '-')}: CE={ce_str} | {src} | Section: {c.get('section', '')}")

    if generate:
        for m in mode_list:
            print(f"\n--- Report ({m}) ---")
            print(reports_dict[m])
            print(f"\n--- Validation Summary ({m}) ---")
            v = validations_dict[m]
            h_count = sum(1 for val in v.get("headings_found", {}).values() if val)
            print(f"  Words: {v.get('word_count')} (in range: {v.get('words_in_range')})")
            print(f"  Numbered headings: {h_count}/5")
            print(f"  Any headings: {v.get('headings_any_count', 0)}/5")
            print(f"  Cited: {v.get('cited')} | Invalid: {v.get('invalid_citations')}")

    print("\nTimings (seconds):")
    for step_name, sec in timings.items():
        print(f"  {step_name}: {sec:.4f}s")

    # 11. Write artifacts
    root = get_project_root()
    if out_dir is None:
        flows_dir = root / "artifacts" / "reports" / "flows"
    else:
        flows_dir = Path(out_dir)
    flows_dir.mkdir(parents=True, exist_ok=True)

    if generate:
        for m in mode_list:
            report_file = flows_dir / f"flow_{index_or_id}_{m}.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(reports_dict[m])
    else:
        for m in mode_list:
            prompt_file = flows_dir / f"flow_{index_or_id}_{m}_prompt.txt"
            p = prompts_dict[m]
            sep = "=" * 40
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(f"{p['system']}\n{sep}\n{p['user']}\n")

    llm_cfg = cfg.get("llm", {})
    if generate and gen_results_dict:
        first_gen = gen_results_dict[mode_list[0]]
        model = first_gen["model"]
        options = first_gen["options"]
        seed_used = first_gen["seed_used"]
    else:
        model = str(llm_cfg.get("model", "llama3:8b"))
        base_seed = int(llm_cfg.get("seed", 42))
        used_s = seed if seed is not None else base_seed
        options = {
            "temperature": float(llm_cfg.get("temperature", 0.2)),
            "seed": used_s,
            "num_ctx": int(llm_cfg.get("num_ctx", 8192)),
            "num_predict": int(llm_cfg.get("num_predict", 1100)),
        }
        seed_used = used_s

    chunks_sha256 = ctx.chunks_sha256
    if not chunks_sha256:
        kb_path = root / cfg.get("paths", {}).get("kb", "data/kb") / "chunks.jsonl"
        if kb_path.exists():
            chunks_sha256 = compute_file_sha256(kb_path)

    if ollama_version is None:
        ollama_version = get_ollama_version()

    summary = {
        "index": index_or_id,
        "ground_truth": ground_truth,
        "true_class": ground_truth,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "tier": tier,
        "is_correct": is_correct,
        "explanation": explanation,
        "query": query,
        "retrieval_query": query,
        "chunks": chunk_records,
        "retrieved_chunks": chunk_records,
        "prompts": prompts_dict,
        "reports": reports_dict,
        "report_texts": reports_dict,
        "validation": validations_dict,
        "validation_dicts": validations_dict,
        "model": model,
        "ollama_model": model,
        "options": options,
        "options_used": options,
        "seed_used": seed_used,
        "timings": timings,
        "config": {
            "llm": cfg.get("llm", {}),
            "rag": cfg.get("rag", {}),
        },
        "llm_config": cfg.get("llm", {}),
        "rag_config": cfg.get("rag", {}),
        "chunks_sha256": chunks_sha256,
        "ollama_version": ollama_version,
    }

    json_file = flows_dir / f"flow_{index_or_id}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_pipeline(
    index: int,
    ctx: PipelineContext,
    cfg: dict[str, Any],
    modes: list[str] | str = "rag",
    seed: int | None = None,
    generate: bool = True,
    client: Any = None,
    out_dir: Path | str | None = None,
    ollama_version: str | None = None,
) -> dict[str, Any]:
    """Execute end-to-end inference and report generation for a single flow index.

    Args:
        index: Row index in the test dataset.
        ctx: Preloaded PipelineContext instance.
        cfg: System configuration dictionary.
        modes: 'rag', 'vanilla', or 'both' (or list thereof).
        seed: Optional random seed override for generation.
        generate: If False, stops after prompt construction (writes prompt files).
        client: Optional injected Ollama Client instance for testing.
        out_dir: Directory for reports and JSON (default artifacts/reports/flows).
        ollama_version: Recorded Ollama version string.

    Returns:
        Dict containing all run metadata, explanation, query, chunks, prompts,
        reports, validations, and timings.

    Raises:
        IndexError: If index is out of bounds for the test dataset.
        ValueError: If an unrecognized mode is specified.
    """
    # 1. Normalize modes
    if isinstance(modes, str):
        if modes == "both":
            mode_list = ["rag", "vanilla"]
        else:
            mode_list = [modes]
    else:
        mode_list = list(modes)

    for m in mode_list:
        if m not in ("rag", "vanilla"):
            raise ValueError(f"Invalid mode '{m}'. Allowed modes are 'rag', 'vanilla', or 'both'.")

    # 2. Bounds check row index
    total_rows = len(ctx.df_test)
    if index < 0 or index >= total_rows:
        raise IndexError(
            f"Index {index} is out of range. Valid range is 0 to {total_rows - 1}."
        )

    # 3. Extract test row and flow metadata
    row_series = ctx.df_test.iloc[index]
    meta_series = (
        ctx.df_meta.iloc[index]
        if ctx.df_meta is not None and len(ctx.df_meta) > index
        else None
    )

    # Ground truth label for evaluation display only (never passed to pipeline components)
    ground_truth = "UNKNOWN"
    if ctx.df_preds is not None and len(ctx.df_preds) > index:
        preds_row = ctx.df_preds.iloc[index]
        if "true_class" in preds_row:
            ground_truth = str(preds_row["true_class"])
        elif "true_label" in preds_row:
            lbl = preds_row["true_label"]
            ground_truth = ctx.ensemble.label_map.get(int(lbl), str(lbl))
    elif "label" in row_series:
        lbl = row_series["label"]
        ground_truth = ctx.ensemble.label_map.get(int(lbl), str(lbl))
    elif "Label" in row_series:
        lbl = row_series["Label"]
        ground_truth = ctx.ensemble.label_map.get(int(lbl), str(lbl))

    # Features tensor in canonical feature_names order
    x_features = row_series[ctx.feature_names].to_numpy(dtype=np.float32)

    return _run_core(
        x_scaled=x_features,
        meta_row=meta_series,
        ctx=ctx,
        cfg=cfg,
        index_or_id=index,
        mode_list=mode_list,
        ground_truth=ground_truth,
        seed=seed,
        generate=generate,
        client=client,
        out_dir=out_dir,
        ollama_version=ollama_version,
    )


def run_flow(
    raw_features: "dict[str, Any] | pd.Series",
    ctx: PipelineContext,
    cfg: dict[str, Any],
    meta: "dict[str, Any] | None" = None,
    flow_id: "int | str" = "custom",
    modes: "list[str] | str" = "rag",
    seed: int | None = None,
    generate: bool = True,
    client: Any = None,
    out_dir: "Path | str | None" = None,
    ollama_version: str | None = None,
) -> dict[str, Any]:
    """Accept raw (unscaled) feature values, validate, scale, and run the full pipeline.

    This is the public entry point for callers that hold raw, unscaled feature
    values (e.g. a CSV row before StandardScaler is applied). It validates the
    input, applies ``ctx.scaler``, and then delegates to ``_run_core``.

    Args:
        raw_features: Mapping of feature name → value. Dict or pandas Series.
            String keys are stripped of surrounding whitespace before lookup.
            Extra columns are silently ignored.
        ctx: Preloaded PipelineContext instance (must include a fitted scaler).
        cfg: System configuration dictionary.
        meta: Optional dict of flow metadata (IP addresses, ports, timestamp).
            Passed verbatim to ``explain_flow`` as ``meta_row``.
            Pass ``None`` to omit metadata; null fields appear in the prompt.
        flow_id: Identifier used in output filenames and the ``"index"`` JSON key.
        modes: 'rag', 'vanilla', or 'both' (or list thereof).
        seed: Optional random seed override for generation.
        generate: If False, stops after prompt construction (writes prompt files).
        client: Optional injected Ollama Client instance for testing.
        out_dir: Directory for reports and JSON.
        ollama_version: Recorded Ollama version string.

    Returns:
        Summary dict identical in structure to the one returned by ``run_pipeline``.

    Raises:
        ValueError: If any feature is missing, non-numeric, NaN, or Inf.
        ValueError: If an unrecognized mode is specified.
    """
    # Normalize modes first so we surface mode errors before touching features
    if isinstance(modes, str):
        if modes == "both":
            mode_list = ["rag", "vanilla"]
        else:
            mode_list = [modes]
    else:
        mode_list = list(modes)

    for m in mode_list:
        if m not in ("rag", "vanilla"):
            raise ValueError(f"Invalid mode '{m}'. Allowed modes are 'rag', 'vanilla', or 'both'.")

    x_scaled = validate_raw_features(raw_features, ctx.feature_names, ctx.scaler)

    return _run_core(
        x_scaled=x_scaled,
        meta_row=meta,
        ctx=ctx,
        cfg=cfg,
        index_or_id=flow_id,
        mode_list=mode_list,
        ground_truth="UNKNOWN",
        seed=seed,
        generate=generate,
        client=client,
        out_dir=out_dir,
        ollama_version=ollama_version,
    )


def main() -> None:
    """CLI entrypoint for running the pipeline on single or multiple flow indices."""
    parser = argparse.ArgumentParser(description="End-to-end incident response inference pipeline.")

    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--index", type=int, default=None, help="Row index of flow to evaluate.")
    target_group.add_argument("--indices", type=str, default=None, help="Comma-separated row indices to evaluate.")
    target_group.add_argument("--indices-file", type=str, default=None, help="JSON file containing list of indices.")

    parser.add_argument("--mode", type=str, default="rag", choices=["rag", "vanilla", "both"], help="Generation mode.")
    parser.add_argument("--no-generate", action="store_true", help="Stop after retrieval and prompt, write prompt files.")
    parser.add_argument("--seed", type=int, default=None, help="Override llm.seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/reports/flows",
        help="Directory to save report and JSON artifacts.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generating flow if its JSON artifact already exists in out-dir.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Write summary CSV of all evaluated flows and modes.",
    )

    args = parser.parse_args()

    if args.index is None and args.indices is None and args.indices_file is None:
        parser.error("One of --index, --indices, or --indices-file must be provided.")

    indices: list[int] = []
    if args.indices_file is not None:
        file_path = Path(args.indices_file)
        if not file_path.exists():
            parser.error(f"Indices file not found: {args.indices_file}")
        with open(file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        if isinstance(file_data, dict):
            indices = [int(x) for x in file_data.get("indices", [])]
        elif isinstance(file_data, list):
            indices = [int(x) for x in file_data]
        else:
            parser.error(f"Invalid format in indices file: {args.indices_file}")
    elif args.indices is not None:
        indices.extend(int(x.strip()) for x in args.indices.split(",") if x.strip())
    elif args.index is not None:
        indices.append(args.index)

    cfg = load_config()

    # Step 1: Record and print ollama --version once at the start
    ollama_ver = get_ollama_version()
    print(f"Ollama version: {ollama_ver}")

    # Check Ollama model before heavy work (unless --no-generate)
    if not args.no_generate:
        try:
            check_ollama(cfg)
        except Exception as e:
            print(f"Error checking Ollama: {e}", file=sys.stderr)
            sys.exit(1)

    # Load context once (validates index freshness)
    try:
        ctx = PipelineContext.load(cfg)
    except Exception as e:
        print(f"Error initializing pipeline context: {e}", file=sys.stderr)
        sys.exit(1)

    root = get_project_root()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    flows_run = 0
    flows_skipped = 0
    flows_failed = 0
    correct_predictions = 0
    flow_summaries: list[dict[str, Any]] = []

    loop_start = time.perf_counter()
    n_flows = len(indices)
    has_failure = False

    for k, idx in enumerate(indices, 1):
        print(f"flow {k}/{n_flows} index {idx}")

        json_path = out_dir / f"flow_{idx}.json"
        if args.skip_existing and json_path.exists():
            print(f"Flow {idx} skipped: {json_path} already exists.")
            flows_skipped += 1
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    cached_summary = json.load(f)
                flow_summaries.append(cached_summary)
                if cached_summary.get("is_correct"):
                    correct_predictions += 1
            except Exception as e:
                print(f"Warning: Failed to read existing JSON {json_path}: {e}", file=sys.stderr)
            continue

        try:
            summary = run_pipeline(
                index=idx,
                ctx=ctx,
                cfg=cfg,
                modes=args.mode,
                seed=args.seed,
                generate=not args.no_generate,
                out_dir=out_dir,
                ollama_version=ollama_ver,
            )
            flow_summaries.append(summary)
            flows_run += 1
            if summary.get("is_correct"):
                correct_predictions += 1
        except Exception as e:
            print(f"Error running pipeline for index {idx}: {e}", file=sys.stderr)
            flows_failed += 1
            has_failure = True

    elapsed_min = (time.perf_counter() - loop_start) / 60.0
    print(
        f"\nSummary: flows run: {flows_run}, flows skipped: {flows_skipped}, "
        f"flows failed: {flows_failed}, correct predictions: {correct_predictions}, "
        f"time: {elapsed_min:.2f} min"
    )

    if args.summary_csv and flow_summaries:
        norm_modes = ["rag", "vanilla"] if args.mode == "both" else [args.mode]
        csv_out_path = Path(args.summary_csv)
        if not csv_out_path.is_absolute():
            csv_out_path = root / csv_out_path
        write_summary_csv(csv_out_path, flow_summaries, out_dir=out_dir, modes=norm_modes)

    if has_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
