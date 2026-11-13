"""V2 End-to-end incident response report generation pipeline for extra attack classes.

Supports the three new attack classes:
- Bot (Class ID 3)
- FTP-BruteForce (Class ID 4) [INDICATIVE ONLY - N=53]
- SSH-Bruteforce (Class ID 5)

Uses:
- Trained v2 binary DNN models from models_v2/ (never models/).
- Frozen baseline scaler from models/scaler.joblib (read-only transform/inverse_transform).
- V2 merged knowledge base (742 base chunks + 9 v2 chunks) from data/v2/kb/index/.
- Extended query formulation and retrieval from ids_rag.v2.retrieve_v2.
- Existing prompt building, report generation, and report validation.
- Output artifacts saved to artifacts/v2/reports/flows/ (never artifacts/reports/flows/).
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
import torch

from ids_rag.config import get_project_root
from ids_rag.explain.shap_topk import explain_flow
from ids_rag.llm.generate import check_ollama, generate_report, validate_report
from ids_rag.llm.prompt import build_prompt
from ids_rag.models.dnn import BinaryDNN
from ids_rag.models.ensemble import BinaryDNNEnsemble, assign_tiers
from ids_rag.pipeline import get_ollama_version, validate_raw_features
from ids_rag.rag.retrieve import get_cached_cross_encoder
from ids_rag.v2.prepare_extra import load_v2_config
from ids_rag.v2.retrieve_v2 import build_query_v2, get_cached_v2_kb, retrieve_v2

# Frozen directories that must never be written to
FORBIDDEN_OUTPUT_DIRS = [
    "models",
    "data/processed",
    "data/kb",
    "backup",
    "artifacts/eval",
    "artifacts/reports",
    "artifacts/fresh_eval",
    "artifacts/tuning_experiments",
]


def check_v2_output_safety(out_path: Path | str) -> None:
    """Validate that an output path does not target frozen baseline directories."""
    resolved = Path(out_path).resolve()
    root = get_project_root().resolve()

    resolved_str = str(resolved).lower().replace("\\", "/")
    root_str = str(root).lower().replace("\\", "/")

    # Whitelisted V2 directories that are always allowed
    v2_allowed = [
        f"{root_str}/artifacts/v2",
        f"{root_str}/artifacts/demo_runs",
        f"{root_str}/artifacts/demo_runs_v2",
        f"{root_str}/artifacts/demo_cache",
        f"{root_str}/data/v2",
        f"{root_str}/models_v2",
        f"{root_str}/app",
    ]

    for allowed in v2_allowed:
        if resolved_str == allowed or resolved_str.startswith(f"{allowed}/"):
            return

    # Check against forbidden directories
    for forbidden in FORBIDDEN_OUTPUT_DIRS:
        forbidden_str = f"{root_str}/{forbidden.lower()}"
        if (
            resolved_str == forbidden_str
            or resolved_str.startswith(f"{forbidden_str}/")
            or forbidden_str.startswith(f"{resolved_str}/")
        ):
            raise ValueError(
                f"Safety check failed: Output path '{out_path}' targets frozen "
                f"directory '{forbidden}'. All v2 outputs must target artifacts/v2/ or data/v2/."
            )


def load_v2_ensemble(
    models_dir: Path | str | None = None,
    label_map_path: Path | str | None = None,
    feature_names_path: Path | str | None = None,
    device: torch.device | None = None,
) -> BinaryDNNEnsemble:
    """Load the v2 binary DNN models from models_v2/ into a BinaryDNNEnsemble."""
    root = get_project_root()
    cfg = load_v2_config()

    if models_dir is None:
        models_dir = root / "models_v2"
    else:
        models_dir = Path(models_dir)

    if label_map_path is None:
        label_map_path = models_dir / "label_map_v2.json"
    else:
        label_map_path = Path(label_map_path)

    if feature_names_path is None:
        feature_names_path = models_dir / "feature_names_v2.json"
    else:
        feature_names_path = Path(feature_names_path)

    with open(label_map_path, "r", encoding="utf-8") as f:
        raw_map: dict[str, str] = json.load(f)
    label_map = {int(k): v for k, v in raw_map.items()}

    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names: list[str] = json.load(f)

    model_cfg = cfg.get("model", {})
    hidden = model_cfg.get("hidden", [128, 64, 32])
    dropout = model_cfg.get("dropout", 0.3)
    use_batchnorm = model_cfg.get("use_batchnorm", False)

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: dict[int, BinaryDNN] = {}

    for class_id, class_name in label_map.items():
        weights_path = models_dir / f"dnn_v2_{class_name}.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights for '{class_name}' not found at {weights_path}. "
                "Ensure Stage 2 training has been completed."
            )

        model = BinaryDNN(
            in_features=len(feature_names),
            hidden=hidden,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        state_dict = torch.load(weights_path, map_location=target_device)
        model.load_state_dict(state_dict)
        model.to(target_device)
        model.eval()
        models[class_id] = model

    ensemble_cfg = cfg.get("ensemble", {})
    tiers_cfg = ensemble_cfg.get("tiers", {"very_high": 0.95, "high": 0.70, "medium": 0.50})

    return BinaryDNNEnsemble(
        models=models,
        label_map=label_map,
        feature_names=feature_names,
        tiers_config=tiers_cfg,
        temperatures={name: 1.0 for name in label_map.values()},
        use_temperature=False,
        device=target_device,
    )


class PipelineV2Context:
    """Container for v2 models, index, scaler, and datasets loaded once across runs."""

    def __init__(
        self,
        ensemble: BinaryDNNEnsemble,
        scaler: Any,
        feature_names: list[str],
        kb: Any,
        cross_encoder: Any,
        df_test: pd.DataFrame | None = None,
        df_meta: pd.DataFrame | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.scaler = scaler
        self.feature_names = feature_names
        self.kb = kb
        self.cross_encoder = cross_encoder
        self.df_test = df_test
        self.df_meta = df_meta
        self.cfg = cfg or {}

    @classmethod
    def load(cls, cfg: dict[str, Any] | None = None) -> "PipelineV2Context":
        """Load v2 ensemble, scaler, feature names, KB index, and datasets from disk."""
        if cfg is None:
            cfg = load_v2_config()

        root = get_project_root()
        paths_cfg = cfg.get("paths", {})

        # Scaler is frozen baseline scaler (read-only)
        scaler_path = root / paths_cfg.get("scaler", "models/scaler.joblib")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
        scaler = joblib.load(scaler_path)

        # Load v2 ensemble
        ensemble = load_v2_ensemble()
        feature_names = ensemble.feature_names

        # Load v2 KB and cross-encoder
        kb = get_cached_v2_kb(cfg)
        cross_encoder = get_cached_cross_encoder(cfg)

        # Test dataset (optional for interactive inference)
        test_path = root / paths_cfg.get("processed", "data/v2/processed") / "test_extra.parquet"
        meta_path = root / paths_cfg.get("processed", "data/v2/processed") / "test_extra_meta.parquet"

        df_test = pd.read_parquet(test_path) if test_path.exists() else None
        df_meta = pd.read_parquet(meta_path) if meta_path.exists() else None

        return cls(
            ensemble=ensemble,
            scaler=scaler,
            feature_names=feature_names,
            kb=kb,
            cross_encoder=cross_encoder,
            df_test=df_test,
            df_meta=df_meta,
            cfg=cfg,
        )


def run_flow_v2(
    raw_features: dict[str, Any] | pd.Series,
    ctx: PipelineV2Context,
    cfg: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    flow_id: int | str = "v2_custom",
    modes: list[str] | str = "rag",
    seed: int | None = None,
    generate: bool = True,
    client: Any = None,
    out_dir: Path | str | None = None,
    ollama_version: str | None = None,
) -> dict[str, Any]:
    """Execute v2 inference and report generation for raw (unscaled) feature inputs.

    Args:
        raw_features: Mapping of feature name -> raw value.
        ctx: Preloaded PipelineV2Context instance.
        cfg: V2 configuration dictionary. If None, uses ctx.cfg or loads config_v2.yaml.
        meta: Optional dict of flow metadata (IP, port, timestamp).
        flow_id: Flow identifier used for output files and summary dict.
        modes: 'rag', 'vanilla', or 'both'.
        seed: Optional random seed override for report generation.
        generate: If False, prepares prompts and skips Ollama report generation.
        client: Optional mock/custom Ollama Client for testing.
        out_dir: Destination directory for artifacts (default artifacts/v2/reports/flows/).
        ollama_version: Ollama version string.

    Returns:
        Summary dictionary with predictions, feature attributions, retrieved chunks,
        prompts, generated reports, validations, and timings.
    """
    if cfg is None:
        cfg = ctx.cfg if ctx.cfg else load_v2_config()

    root = get_project_root()
    if out_dir is None:
        out_dir = root / "artifacts" / "v2" / "reports" / "flows"
    else:
        out_dir = Path(out_dir)

    # Validate output destination safety
    check_v2_output_safety(out_dir)

    # 1. Normalize modes
    if isinstance(modes, str):
        mode_list = ["rag", "vanilla"] if modes == "both" else [modes]
    else:
        mode_list = list(modes)

    for m in mode_list:
        if m not in ("rag", "vanilla"):
            raise ValueError(f"Invalid mode '{m}'. Allowed modes are 'rag', 'vanilla', or 'both'.")

    t0_total = time.perf_counter()

    # 2. Validate and scale features using frozen baseline scaler
    x_scaled = validate_raw_features(
        raw_features=raw_features,
        feature_names=ctx.feature_names,
        scaler=ctx.scaler,
    )

    # 3. Predict threat category with v2 ensemble
    t0_ens = time.perf_counter()
    pred_labels, confidences, tiers = ctx.ensemble.predict(x_scaled.reshape(1, -1))
    pred_label = int(pred_labels[0])
    confidence = float(confidences[0])
    tier = str(tiers[0])
    predicted_class = ctx.ensemble.label_map.get(pred_label, str(pred_label))
    t_ensemble = time.perf_counter() - t0_ens

    # FTP-BruteForce indicator warning check
    is_indicative = bool(predicted_class == "FTP-BruteForce")
    indicative_note = (
        "FTP-BruteForce model evaluation is INDICATIVE ONLY due to limited unique samples (N=53) in CICIDS2018."
        if is_indicative
        else None
    )

    # 4. Feature attribution explanation via explain_flow
    t0_exp = time.perf_counter()
    explain_cfg = cfg.get("explain", {})
    method = explain_cfg.get("method", "gradient")
    top_k = int(explain_cfg.get("top_k", 5))

    explanation = explain_flow(
        x_scaled=x_scaled,
        ensemble=ctx.ensemble,
        k=top_k,
        scaler=ctx.scaler,
        meta_row=meta,
        feature_names=ctx.feature_names,
        method=method,
    )
    t_explain = time.perf_counter() - t0_exp

    # 5. Formulate v2 query and retrieve chunks from merged v2 KB
    t0_ret = time.perf_counter()
    query = build_query_v2(explanation)
    chunks = retrieve_v2(query, cfg=cfg, kb=ctx.kb, cross_encoder=ctx.cross_encoder)
    t_retrieve = time.perf_counter() - t0_ret

    # 6. Build structured prompts
    t0_prompt = time.perf_counter()
    prompts_dict: dict[str, dict[str, Any]] = {}
    for m in mode_list:
        prompts_dict[m] = build_prompt(explanation, chunks, mode=m, cfg=cfg)
    t_prompt = time.perf_counter() - t0_prompt

    # 7. Generate reports via Ollama and validate
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

    # 8. Format chunk records for UI and JSON export
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

    # 9. Write outputs to out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if generate:
        for m in mode_list:
            report_file = out_dir / f"flow_{flow_id}_{m}.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(reports_dict[m])
    else:
        for m in mode_list:
            prompt_file = out_dir / f"flow_{flow_id}_{m}_prompt.txt"
            p = prompts_dict[m]
            sep = "=" * 40
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(f"{p['system']}\n{sep}\n{p['user']}\n")

    llm_cfg = cfg.get("llm", {})
    if generate and gen_results_dict:
        first_gen = gen_results_dict[mode_list[0]]
        model_name = first_gen["model"]
        options = first_gen["options"]
        seed_used = first_gen["seed_used"]
    else:
        model_name = str(llm_cfg.get("model", "llama3:8b"))
        base_seed = int(llm_cfg.get("seed", 42))
        used_s = seed if seed is not None else base_seed
        options = {
            "temperature": float(llm_cfg.get("temperature", 0.2)),
            "seed": used_s,
            "num_ctx": int(llm_cfg.get("num_ctx", 8192)),
            "num_predict": int(llm_cfg.get("num_predict", 1100)),
        }
        seed_used = used_s

    if ollama_version is None:
        ollama_version = get_ollama_version()

    summary: dict[str, Any] = {
        "index": flow_id,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "tier": tier,
        "is_indicative_only": is_indicative,
        "indicative_warning": indicative_note,
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
        "model": model_name,
        "ollama_model": model_name,
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
        "ollama_version": ollama_version,
        "pipeline_version": "v2",
    }

    json_file = out_dir / f"flow_{flow_id}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
