"""Analyst alert builder.

Converts an existing artifacts/reports/flows/flow_<N>.json (written by ids_rag.pipeline)
into a short analyst alert file artifacts/alerts/alert_flow_<N>.json.
Does not load models and does not call Ollama.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ids_rag.config import get_project_root, load_config
from ids_rag.eval.citation_check import drop_header_lines, extract_flow


def build_alert(
    flow: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    thr: float,
) -> dict[str, Any]:
    """Build a structured analyst alert dict from a stored pipeline flow.

    Pure function: no file or network access.

    Args:
        flow: Loaded flow_<N>.json dict as written by ids_rag.pipeline.
        chunks_by_id: Dict mapping chunk_id -> chunk record from data/kb/chunks.jsonl.
        thr: Word-overlap threshold below which a citation is flagged unverified.

    Returns:
        Ordered dict with alert_version, flow_index, detected, flow_metadata,
        top_features, mitigation_report, sources, citations, checks, provenance,
        and warnings. Excludes ground_truth, true_class, is_correct, prompts,
        config, options, timings, and vanilla report.

    Raises:
        ValueError: If flow["reports"]["rag"] or flow["validation"]["rag"] is missing.
    """
    reports = flow.get("reports") or {}
    if "rag" not in reports:
        raise ValueError(
            "flow[\"reports\"][\"rag\"] is missing. "
            "The flow JSON must contain a RAG mode report."
        )
    validations = flow.get("validation") or {}
    if "rag" not in validations:
        raise ValueError(
            "flow[\"validation\"][\"rag\"] is missing. "
            "The flow JSON must contain RAG mode validation flags."
        )

    report_text: str = reports["rag"]
    val_flags: dict[str, Any] = validations["rag"]

    # --- detected block ---
    explanation: dict[str, Any] = flow.get("explanation") or {}
    detected = {
        "predicted_class": flow.get("predicted_class", explanation.get("predicted_class")),
        "confidence": flow.get("confidence", explanation.get("confidence")),
        "tier": flow.get("tier", explanation.get("tier")),
    }

    # --- flow_metadata: exact copy of explanation["metadata"] ---
    flow_metadata: dict[str, Any] = explanation.get("metadata") or {}

    # --- top_features ---
    raw_features: list[dict[str, Any]] = explanation.get("features") or []
    top_features = [
        {
            "rank": rank,
            "name": f["name"],
            "original_value": f["original_value"],
            "direction": f.get("direction"),
            "importance": f.get("importance"),
            "description": f.get("description"),
            "implication": f.get("implication"),
        }
        for rank, f in enumerate(raw_features, 1)
    ]

    # --- mitigation_report ---
    word_count = len(report_text.split())
    mitigation_report = {
        "mode": "rag",
        "word_count": word_count,
        "text": report_text,
    }

    # --- sources ---
    retrieved_chunks: list[dict[str, Any]] = flow.get("retrieved_chunks") or []
    sources = [
        {
            "number": pos,
            "chunk_id": rc.get("chunk_id"),
            "source": rc.get("source"),
            "section": rc.get("section"),
            "ce_score": rc.get("ce_score"),
        }
        for pos, rc in enumerate(retrieved_chunks, 1)
    ]

    # --- citations: use extract_flow from citation_check on the cleaned report text ---
    cleaned_text, _dropped = drop_header_lines(report_text)
    citation_rows = extract_flow(cleaned_text, retrieved_chunks, chunks_by_id, thr)
    citations = [
        {
            "n": row["n"],
            "sentence": row["sentence"],
            "source": row["source"],
            "overlap": round(row["overlap"], 3) if row["overlap"] is not None else None,
            "out_of_range": row["out_of_range"],
            "flagged_unverified": row["flagged"],
        }
        for row in citation_rows
    ]

    # --- checks ---
    feature_names = [f["name"] for f in raw_features]
    text_lower = report_text.lower()
    features_named = [name for name in feature_names if name.lower() in text_lower]
    features_not_named = [name for name in feature_names if name.lower() not in text_lower]

    checks_validation_flags = {
        "invalid_citations": val_flags.get("invalid_citations"),
        "has_reference_list": val_flags.get("has_reference_list"),
        "unsupported_ips": val_flags.get("unsupported_ips"),
        "attack_ids_not_in_kb": val_flags.get("attack_ids_not_in_kb"),
        "sp_not_in_kb": val_flags.get("sp_not_in_kb"),
        "placeholder_citations": val_flags.get("placeholder_citations"),
    }
    checks = {
        "features_named_in_report": features_named,
        "features_not_named_in_report": features_not_named,
        "validation_flags": checks_validation_flags,
    }

    # --- provenance ---
    provenance = {
        "model": flow.get("model"),
        "seed_used": flow.get("seed_used"),
        "ollama_version": flow.get("ollama_version"),
        "chunks_sha256": flow.get("chunks_sha256"),
        "overlap_threshold": thr,
    }

    # --- warnings ---
    warnings: list[str] = [
        "Draft generated by a local language model. Review before acting.",
        "Generated from a stored evaluation flow, not live traffic.",
        "Citation flags are a word-overlap heuristic, not a verdict.",
    ]
    m = len(citation_rows)
    k = sum(1 for row in citation_rows if row["flagged"])
    if m == 0:
        warnings.append("Report contains no inline citations.")
    elif k > 0:
        warnings.append(f"{k} of {m} citation rows flagged unverified.")
    a = len(features_named)
    b = len(feature_names)
    if a < b:
        warnings.append(f"Report names {a} of {b} top features.")

    alert: dict[str, Any] = {
        "alert_version": 1,
        "flow_index": flow.get("index"),
        "detected": detected,
        "flow_metadata": flow_metadata,
        "top_features": top_features,
        "mitigation_report": mitigation_report,
        "sources": sources,
        "citations": citations,
        "checks": checks,
        "provenance": provenance,
        "warnings": warnings,
    }
    return alert


def write_alert(
    alert: dict[str, Any],
    out_path: Path,
    force: bool = False,
) -> None:
    """Write an alert dict to a JSON file, refusing to overwrite unless force is set.

    Args:
        alert: Alert dict built by build_alert.
        out_path: Destination path for the alert JSON.
        force: If False, print a refusal message to stderr and exit non-zero when the file exists.
    """
    if out_path.exists() and not force:
        print(
            f"Refusing to overwrite existing alert file: {out_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(alert, fh, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for building an analyst alert from a stored flow JSON.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description="Convert a stored pipeline flow JSON into a structured analyst alert."
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Flow index; reads artifacts/reports/flows/flow_<index>.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing alert file.",
    )
    args = parser.parse_args(argv)

    root = get_project_root()
    cfg = load_config()

    # Read threshold from config; raise a clear error if the key is missing.
    citation_check_cfg = cfg.get("citation_check")
    if citation_check_cfg is None or "overlap_threshold" not in citation_check_cfg:
        print(
            "Error: config.yaml must contain a 'citation_check.overlap_threshold' key.",
            file=sys.stderr,
        )
        sys.exit(1)
    thr = float(citation_check_cfg["overlap_threshold"])

    # Load chunks.jsonl into a dict keyed by chunk_id.
    chunks_file = root / "data" / "kb" / "chunks.jsonl"
    if not chunks_file.exists():
        print(f"Error: knowledge-base chunks file not found: {chunks_file}", file=sys.stderr)
        sys.exit(1)
    chunks_by_id: dict[str, dict[str, Any]] = {}
    with open(chunks_file, "r", encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            chunks_by_id[c["chunk_id"]] = c

    # Read the flow JSON.
    flow_path = root / "artifacts" / "reports" / "flows" / f"flow_{args.index}.json"
    if not flow_path.exists():
        print(f"Error: flow JSON not found: {flow_path}", file=sys.stderr)
        sys.exit(1)
    with open(flow_path, "r", encoding="utf-8") as fh:
        flow: dict[str, Any] = json.load(fh)

    alert = build_alert(flow, chunks_by_id, thr)

    out_path = root / "artifacts" / "alerts" / f"alert_flow_{args.index}.json"
    write_alert(alert, out_path, force=args.force)
    print(out_path)


if __name__ == "__main__":
    main()

