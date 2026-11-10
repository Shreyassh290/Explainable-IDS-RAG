"""Incident response report prompt construction for local LLM generation.

Implements the prompt template specified in arXiv 2605.17960:
- Standardized cybersecurity analyst system prompt.
- Structured detection context with ensemble confidence tiers and SOC actions.
- Network flow metadata sanitization (missing fields mapped to 'not available').
- Key anomalous indicators with z-score position words and directional effects.
- Retrieved knowledge context (RAG mode with grounding rules vs. vanilla empty context).
- Five-section structured incident response report format.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ids_rag.config import get_project_root, load_config
from ids_rag.explain.features import describe_for_prompt
from ids_rag.rag.retrieve import build_query, retrieve


def _is_missing(val: Any) -> bool:
    """Return True if value is None, empty string, NaN, or non-finite."""
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, (float, np.floating)) and (math.isnan(val) or not math.isfinite(val)):
        return True
    try:
        if pd.isna(val):
            return True
    except Exception:
        pass
    return False


def _format_endpoint(ip: Any, port: Any) -> str:
    """Format ip:port endpoint, returning 'not available' if either or both is missing."""
    if _is_missing(ip) or _is_missing(port):
        return "not available"
    if isinstance(port, (float, np.floating)) and float(port).is_integer():
        port = int(port)
    return f"{ip}:{port}"


def format_value(v: float | int | Any) -> str:
    """Format numerical value with at most 4 significant digits and no scientific notation below 1e9."""
    if _is_missing(v):
        return "not available"
    try:
        val = float(v)
    except (ValueError, TypeError):
        return "not available"

    if not math.isfinite(val):
        return "not available"
    if val == 0.0:
        return "0"

    abs_val = abs(val)
    if abs_val >= 1e9:
        return f"{val:.4g}"

    magnitude = math.floor(math.log10(abs_val))
    decimals = 3 - magnitude
    if decimals <= 0:
        rounded = round(val, decimals)
        return str(int(rounded))
    else:
        rounded = round(val, decimals)
        s = f"{rounded:.{decimals}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s


def position_word(z: float, cfg: dict[str, Any] | None = None) -> str:
    """Return descriptive position word relative to training average based on z-score thresholds."""
    llm_cfg = (cfg or {}).get("llm", {})
    z_near = float(llm_cfg.get("z_near", 0.5))
    z_far = float(llm_cfg.get("z_far", 1.5))

    abs_z = abs(z)
    if abs_z < z_near:
        return "near the training average"
    elif abs_z < z_far:
        if z > 0:
            return "moderately above the training average"
        else:
            return "moderately below the training average"
    else:
        if z > 0:
            return "far above the training average"
        else:
            return "far below the training average"


def clean_chunk_text(text: str) -> str:
    """Clean chunk text: replace Unicode private-use characters with '- ', remove bracketed citation numbers, and collapse repeated spaces."""
    cleaned = re.sub(r"[\uE000-\uF8FF]", "- ", text)
    cleaned = re.sub(r"\s*\[\d{1,2}\]", "", cleaned)
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
    return cleaned.strip()


def build_prompt(
    explanation: dict[str, Any],
    chunks: list[dict[str, Any]] | None,
    mode: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build structured prompt for incident response report generation.

    Args:
        explanation: Stage 3 explanation dictionary with prediction, metadata, and top features.
        chunks: List of retrieved chunk dictionaries (used in RAG mode, ignored in vanilla mode).
        mode: Either 'rag' or 'vanilla'.
        cfg: Configuration dictionary (loads config.yaml if None).

    Returns:
        Dict with keys 'system', 'user', 'sources', 'mode'.

    Raises:
        ValueError: If mode is not 'rag' or 'vanilla'.
    """
    if mode not in ("rag", "vanilla"):
        raise ValueError(f"Invalid mode '{mode}'. Allowed modes are 'rag' or 'vanilla'.")

    if cfg is None:
        cfg = load_config()
    llm_cfg = cfg.get("llm", {})

    # 1. System Prompt (identical across modes)
    system_text = (
        "You are a senior cybersecurity analyst. Generate a structured incident response report based on the evidence provided. "
        "Cite NIST Special Publications and MITRE ATT&CK techniques explicitly where applicable."
    )

    # 2. Detection Context
    pred_class = str(explanation.get("predicted_class", "Benign"))
    confidence = float(explanation.get("confidence", 0.0))
    tier_raw = str(explanation.get("tier", "LOW"))
    tier_actions = llm_cfg.get("tier_actions", {})
    action_str = tier_actions.get(tier_raw, tier_actions.get(tier_raw.upper(), tier_raw))

    meta = explanation.get("metadata", {}) or {}
    dataset_name = meta.get("dataset", "CICIDS2018") if isinstance(meta, dict) else "CICIDS2018"
    if not dataset_name:
        dataset_name = "CICIDS2018"

    detection_context = (
        "[DETECTION CONTEXT]\n"
        f"  Attack class: {pred_class}\n"
        f"  Confidence: {confidence:.4f} ({action_str})\n"
        f"  Dataset: {dataset_name}"
    )

    # 3. Network Flow Metadata
    flow_id = "not available" if _is_missing(meta.get("Flow ID")) else str(meta["Flow ID"])
    timestamp = "not available" if _is_missing(meta.get("Timestamp")) else str(meta["Timestamp"])
    source_str = _format_endpoint(meta.get("Src IP"), meta.get("Src Port"))
    dest_str = _format_endpoint(meta.get("Dst IP"), meta.get("Dst Port"))

    proto_val = meta.get("protocol")
    if _is_missing(proto_val):
        proto_str = "not available"
    else:
        try:
            p_num = int(round(float(proto_val)))
            if p_num == 6:
                proto_str = "TCP (6)"
            elif p_num == 17:
                proto_str = "UDP (17)"
            else:
                proto_str = str(p_num)
        except Exception:
            proto_str = "not available"

    network_metadata = (
        "[NETWORK FLOW METADATA]\n"
        f"  Flow ID: {flow_id}\n"
        f"  Timestamp: {timestamp}\n"
        f"  Source: {source_str}\n"
        f"  Destination: {dest_str}\n"
        f"  Protocol: {proto_str}"
    )

    # 4. Key Anomalous Indicators
    features = explanation.get("features", [])
    k_features = len(features)
    indicators_lines = [
        "[KEY ANOMALOUS INDICATORS]",
        f"  Top-{k_features} features most influential in this decision (K = {k_features}):",
    ]
    suppress_near_avg = bool(llm_cfg.get("suppress_near_average_implications", False))
    z_near = float(llm_cfg.get("z_near", 0.5))
    for idx, f in enumerate(features, 1):
        feat_name = f.get("name", "")
        orig_val_str = format_value(f.get("original_value"))
        z_score = float(f.get("scaled_value", 0.0))
        pos_word = position_word(z_score, cfg)
        direction = f.get("direction", "raises")

        implication = f.get("implication", "")
        if not implication:
            try:
                implication = describe_for_prompt(feat_name).get("implication", "")
            except Exception:
                implication = ""

        indicators_lines.append(
            f"  Feature {idx}: {feat_name} = {orig_val_str} ({pos_word}; effect on the {pred_class} score: {direction})"
        )
        if suppress_near_avg and abs(z_score) < z_near:
            indicators_lines.append(
                "     -> This value is near the training average, so it does not by itself indicate an attack."
            )
        else:
            indicators_lines.append(f"     -> {implication}")

    indicators_lines.append(
        "  Use a feature's meaning only when the observed value fits it; if a value is near the training average, say that it does not by itself indicate an attack."
    )
    indicators_section = "\n".join(indicators_lines)

    # 5. Retrieved Knowledge Context
    if mode == "vanilla":
        retrieved_section = "[RETRIEVED KNOWLEDGE CONTEXT]"
        sources_list: list[dict[str, Any]] = []
    else:
        sources_list = []
        retrieved_lines = ["[RETRIEVED KNOWLEDGE CONTEXT]"]
        chunk_list = chunks or []
        for rank_idx, chunk in enumerate(chunk_list, 1):
            source_name = chunk.get("source_name", "")
            source_ref = chunk.get("source_ref", "")
            section = chunk.get("section", "")
            chunk_id = chunk.get("chunk_id", "")
            is_fallback = bool(chunk.get("is_fallback", False))

            if chunk.get("source_type") == "NIST" or source_name.startswith("NIST") or source_ref.startswith("SP"):
                display_name = f"NIST {source_ref}"
            elif chunk.get("source_type") == "MITRE" or source_name.startswith("T") or source_ref.startswith("T"):
                display_name = f"MITRE ATT&CK {source_ref}"
            else:
                display_name = source_ref or source_name

            if is_fallback:
                display_name += " (general guidance)"

            chunk_text_clean = clean_chunk_text(chunk.get("text", ""))

            retrieved_lines.append(f"[{rank_idx}] {display_name}, section {section}\n{chunk_text_clean}\n")

            sources_list.append(
                {
                    "n": rank_idx,
                    "source_name": source_name,
                    "source_ref": source_ref,
                    "section": section,
                    "chunk_id": chunk_id,
                    "is_fallback": is_fallback,
                }
            )

        grounding_rule = (
            "Cite a source by writing its number in square brackets after the statement it supports. "
            "Cite only the numbers listed above, one number per bracket, never a range, and do not add a reference list at the end. "
            "Use a source only if it supports the statement. "
            "If no listed source supports a recommendation, say so instead of inventing a citation."
        )
        retrieved_lines.append(grounding_rule)
        retrieved_section = "\n".join(retrieved_lines)

    # 6. Report Request
    min_words = int(llm_cfg.get("min_words", 400))
    max_words = int(llm_cfg.get("max_words", 500))

    report_lines = [
        "[REPORT REQUEST]",
        "  Generate a structured report with sections:",
        "  1. Rationale",
        "  2. Key Indicators",
        "  3. Confidence Assessment",
        "  4. Threat Assessment",
        "  5. Recommendations",
        f"  Write between {min_words} and {max_words} words in total, using the five section headings exactly as numbered above.",
    ]
    if pred_class == "Benign":
        report_lines.append(
            "  No attack was detected. Say so in the Rationale, describe which indicators are consistent with normal traffic, "
            "assess the confidence, state that the threat level is low, and recommend routine monitoring. Do not describe an attack."
        )
    report_section = "\n".join(report_lines)

    # 7. Assemble User Prompt
    user_text = f"{detection_context}\n\n{network_metadata}\n\n{indicators_section}\n\n{retrieved_section}\n\n{report_section}"

    return {
        "system": system_text,
        "user": user_text,
        "sources": sources_list,
        "mode": mode,
    }


def main() -> None:
    """CLI runner to generate prompt artifacts from sample explanations."""
    parser = argparse.ArgumentParser(description="LLM incident response report prompt builder.")
    parser.add_argument("--examples", action="store_true", help="Generate example prompts for sample explanations.")
    args = parser.parse_args()

    if args.examples:
        cfg = load_config()
        root = get_project_root()
        sample_path = root / "artifacts" / "sample_explanations.json"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample explanations not found at: {sample_path}")

        with open(sample_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        out_dir = root / "artifacts" / "prompts"
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, exp in enumerate(samples):
            case_name = exp.get("case", f"case_{idx}")
            query = build_query(exp)
            chunks = retrieve(query, cfg)

            for mode in ("rag", "vanilla"):
                prompt_dict = build_prompt(exp, chunks, mode=mode, cfg=cfg)
                sep = "=" * 40
                file_content = f"{prompt_dict['system']}\n{sep}\n{prompt_dict['user']}\n"
                file_path = out_dir / f"{case_name}_{mode}.txt"
                with open(file_path, "w", encoding="utf-8") as out_f:
                    out_f.write(file_content)

                word_count = len(file_content.split())
                source_count = len(prompt_dict["sources"])
                print(f"{file_path} words={word_count} sources={source_count}")


if __name__ == "__main__":
    main()
