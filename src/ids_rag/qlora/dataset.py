"""QLoRA dataset flow sampling and raw generation pipeline.

Enforces strict isolation rules:
- Reads ONLY data/processed/train.parquet (refuses any test/eval data).
- Explicitly excludes any flow index listed in artifacts/fresh_eval/flows_final.json
  and artifacts/tuning_experiments/pilot_flows.json.
- Refuses execution if any overlap is detected.
- Generates tuned RAG prompts and local LLM reports using configured Ollama settings.
- Emits raw outputs to data/qlora/raw/ (resumable, never overwrites existing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ids_rag.config import get_project_root, load_config
from ids_rag.explain.shap_topk import explain_flow
from ids_rag.llm.generate import generate_report
from ids_rag.llm.prompt import build_prompt
from ids_rag.models.ensemble import BinaryDNNEnsemble
from ids_rag.rag.index import load_index
from ids_rag.rag.retrieve import build_query, get_cached_cross_encoder, retrieve


def load_forbidden_indices(root: Path | None = None) -> set[int]:
    """Load all flow indices that must never be included in the training dataset.

    Sources:
    - artifacts/fresh_eval/flows_final.json (exclude_rows, indices, indices_by_class)
    - artifacts/tuning_experiments/pilot_flows.json (flow_id)
    """
    if root is None:
        root = get_project_root()

    forbidden: set[int] = set()

    # 1. artifacts/fresh_eval/flows_final.json
    flows_final_path = root / "artifacts" / "fresh_eval" / "flows_final.json"
    if flows_final_path.exists():
        with open(flows_final_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx in data.get("exclude_rows", []):
                forbidden.add(int(idx))
            for idx in data.get("indices", []):
                forbidden.add(int(idx))
            for class_indices in data.get("indices_by_class", {}).values():
                for idx in class_indices:
                    forbidden.add(int(idx))

    # 2. artifacts/tuning_experiments/pilot_flows.json
    pilot_flows_path = root / "artifacts" / "tuning_experiments" / "pilot_flows.json"
    if pilot_flows_path.exists():
        with open(pilot_flows_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("pilot_flows", []):
                if "flow_id" in item:
                    forbidden.add(int(item["flow_id"]))

    return forbidden


def assert_safe_source_path(data_path: Path) -> None:
    """Ensure data is loaded strictly from data/processed/train.parquet."""
    resolved_name = data_path.name
    resolved_str = str(data_path.resolve()).replace("\\", "/")
    if resolved_name != "train.parquet" or "processed" not in resolved_str:
        raise RuntimeError(
            f"PROTOCOL VIOLATION: QLoRA dataset must be built ONLY from "
            f"data/processed/train.parquet. Attempted source: {data_path}"
        )


def sample_stratified_train_indices(
    df_train: pd.DataFrame,
    per_class: int,
    forbidden_indices: set[int],
    seed: int = 42,
    label_col: str = "label",
) -> list[int]:
    """Sample flow row indices strictly stratified by class, excluding forbidden indices.

    Refuses execution if any sampled index overlaps with forbidden indices.
    """
    rng = np.random.default_rng(seed)
    sampled_indices: list[int] = []

    classes = np.unique(df_train[label_col].values)
    for c in sorted(classes):
        matching_indices = np.where(df_train[label_col].values == c)[0]
        # Filter out any forbidden index
        allowed_indices = np.array([i for i in matching_indices if int(i) not in forbidden_indices])

        if len(allowed_indices) < per_class:
            raise ValueError(
                f"Class {c} has only {len(allowed_indices)} allowed candidate flows, "
                f"fewer than requested per_class={per_class}."
            )

        chosen = rng.choice(allowed_indices, size=per_class, replace=False)
        sampled_indices.extend(int(x) for x in chosen)

    # STRICT GUARD: Refuse to run if the sample overlaps forbidden files
    overlap = set(sampled_indices).intersection(forbidden_indices)
    if overlap:
        raise RuntimeError(
            f"PROTOCOL VIOLATION: Sampled flows overlap with forbidden evaluation/pilot files! "
            f"Overlapping indices: {sorted(overlap)}"
        )

    return sorted(sampled_indices)


def run_single_flow_raw(
    flow_idx: int,
    df_train: pd.DataFrame,
    df_meta: pd.DataFrame | None,
    ensemble: BinaryDNNEnsemble,
    scaler: Any,
    kb: Any,
    cross_encoder: Any,
    cfg: dict[str, Any],
    client: Any = None,
) -> dict[str, Any]:
    """Run pipeline up to prompt construction (tuned, RAG) and generate report via Ollama."""
    feature_cols = [c for c in df_train.columns if c != "label"]
    x_scaled = df_train.iloc[flow_idx][feature_cols].to_numpy(dtype=np.float32)

    meta_row = df_meta.iloc[flow_idx] if df_meta is not None else None
    true_label_val = int(df_train.iloc[flow_idx]["label"])
    true_class = ensemble.label_map.get(true_label_val, str(true_label_val))

    # 1. Feature attribution explanation
    explain_cfg = cfg.get("explain", {})
    method = explain_cfg.get("method", "gradient")
    top_k = int(explain_cfg.get("top_k", 5))

    explanation = explain_flow(
        x_scaled=x_scaled,
        ensemble=ensemble,
        k=top_k,
        scaler=scaler,
        meta_row=meta_row,
        feature_names=feature_cols,
        method=method,
    )

    # 2. Build retrieval query and retrieve chunks
    query = build_query(explanation)
    chunks = retrieve(query, cfg=cfg, kb=kb, cross_encoder=cross_encoder)

    # 3. Build prompt with tuned prompt version in RAG mode
    cfg_tuned = json.loads(json.dumps(cfg))  # deep copy
    cfg_tuned.setdefault("llm", {})["prompt_version"] = "tuned"

    prompt_dict = build_prompt(
        explanation=explanation,
        chunks=chunks,
        mode="rag",
        cfg=cfg_tuned,
    )

    # 4. Generate report via Ollama
    gen = generate_report(
        prompt=prompt_dict,
        cfg=cfg,
        client=client,
    )

    return {
        "flow_index": int(flow_idx),
        "true_class": true_class,
        "true_label": true_label_val,
        "predicted_class": explanation.get("predicted_class"),
        "confidence": explanation.get("confidence"),
        "tier": explanation.get("tier"),
        "explanation": explanation,
        "retrieved_chunks": chunks,
        "prompt": prompt_dict,
        "report": gen.get("text", ""),
        "gen_meta": gen,
    }

