"""Full 30-Flow Benchmark Evaluation Runner (Experiment B v2: Tuned Prompt).

Executes generation and validation for all 30 flows from artifacts/eval/eval_flows.json:
- 10 Benign flows
- 10 DoS flows
- 10 DDoS flows
Generates both RAG and Vanilla modes (60 reports total).
Saves all outputs strictly to artifacts/tuning_experiments/eval_full_tuned/.
Preserves all baseline and previous experiment artifacts intact.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
import yaml

from ids_rag.config import get_project_root, load_config
from ids_rag.llm.generate import check_ollama, generate_report, validate_report
from ids_rag.llm.prompt import build_prompt


def load_kb_chunks_by_id(root: Path) -> dict[str, dict[str, Any]]:
    """Index chunks from data/kb/chunks.jsonl by chunk_id."""
    kb_path = root / "data" / "kb" / "chunks.jsonl"
    chunk_map: dict[str, dict[str, Any]] = {}
    if not kb_path.exists():
        print(f"Warning: {kb_path} does not exist.")
        return chunk_map

    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cid = row.get("chunk_id")
                if cid:
                    chunk_map[cid] = row
            except Exception:
                continue
    return chunk_map


def load_flow_data(
    flow_id: int, root: Path, chunk_map: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], str, float, str, bool]:
    """Load explanation, chunks, and metadata from original baseline flow file."""
    eval_flow_file = root / "artifacts" / "eval" / "flows" / f"flow_{flow_id}.json"
    if not eval_flow_file.exists():
        raise FileNotFoundError(f"Original eval flow file not found: {eval_flow_file}")

    with open(eval_flow_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    explanation = data["explanation"]
    true_class = data.get("true_class", explanation.get("predicted_class", "UNKNOWN"))
    confidence = float(data.get("confidence", explanation.get("confidence", 1.0)))
    tier = str(data.get("tier", explanation.get("tier", "VERY_HIGH")))
    is_correct = bool(data.get("is_correct", True))

    chunks_meta = data.get("retrieved_chunks") or data.get("chunks") or []
    full_chunks: list[dict[str, Any]] = []

    for cm in chunks_meta:
        cid = cm.get("chunk_id", "")
        base_chunk = chunk_map.get(cid, {})
        merged = {
            "chunk_id": cid,
            "text": base_chunk.get("text", cm.get("text", "")),
            "source_name": cm.get("source_name") or cm.get("source", base_chunk.get("source_name", "")),
            "source_type": base_chunk.get("source_type", "NIST" if "800-" in str(cm) else "MITRE"),
            "source_ref": cm.get("source_ref") or base_chunk.get("source_ref", ""),
            "section": cm.get("section", base_chunk.get("section", "")),
            "is_fallback": bool(cm.get("is_fallback", False)),
        }
        full_chunks.append(merged)

    return explanation, full_chunks, true_class, confidence, tier, is_correct


def clean_text(text: str) -> str:
    """Strip markdown and bracketed numbers for ROUGE/BLEU evaluation."""
    text = re.sub(r"\[\d{1,2}\]", "", text)
    text = re.sub(r"[#*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def run_full_eval(dry_run: bool = False, skip_existing: bool = False) -> None:
    root = get_project_root()
    cfg = load_config()

    # Confirm prompt_version is tuned
    if cfg.get("llm", {}).get("prompt_version") != "tuned":
        print("Enforcing prompt_version: tuned in active config")
        cfg["llm"]["prompt_version"] = "tuned"

    # Load 30 benchmark flows
    eval_flows_file = root / "artifacts" / "eval" / "eval_flows.json"
    with open(eval_flows_file, "r", encoding="utf-8") as f:
        eval_meta = json.load(f)

    indices = eval_meta.get("indices", [])
    indices_by_class = eval_meta.get("indices_by_class", {})

    print("=================================================================")
    print("STARTING FULL 30-FLOW BENCHMARK EVALUATION (EXPERIMENT B v2)")
    print(f"Total flows: {len(indices)}")
    print(f"Class breakdown: Benign={len(indices_by_class.get('Benign', []))}, "
          f"DoS={len(indices_by_class.get('DoS', []))}, "
          f"DDoS={len(indices_by_class.get('DDoS', []))}")
    print(f"Model: {cfg['llm'].get('model', 'llama3:8b')}")
    print(f"Temperature: {cfg['llm'].get('temperature')}")
    print(f"Seed: {cfg['llm'].get('seed')}")
    print(f"Prompt version: {cfg['llm'].get('prompt_version')}")
    print("=================================================================\n")

    if not dry_run:
        check_ollama(cfg)

    out_dir = root / "artifacts" / "tuning_experiments" / "eval_full_tuned"
    flows_out_dir = out_dir / "flows"
    flows_out_dir.mkdir(parents=True, exist_ok=True)

    # Save config used
    config_save_path = out_dir / "config_eval_full_tuned.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    chunk_map = load_kb_chunks_by_id(root)

    header = (
        f"{'Flow':<8} {'Class':<8} {'Mode':<8} {'Words':<6} {'InRange':<8} "
        f"{'H-num':<7} {'H-any':<7} {'CopiedHdr':<10} {'Cited':<10} {'RefList':<8} {'Time(s)':<7}"
    )
    print(header)
    print("-" * len(header))

    summary_rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    flows_completed = 0
    flows_failed = 0

    for idx in indices:
        explanation, chunks, true_class, confidence, tier, is_correct = load_flow_data(idx, root, chunk_map)

        for mode in ("rag", "vanilla"):
            txt_file = flows_out_dir / f"flow_{idx}_{mode}.txt"
            json_file = flows_out_dir / f"flow_{idx}_{mode}.json"

            if skip_existing and txt_file.exists() and json_file.exists():
                print(f"Skipping existing: flow_{idx}_{mode}")
                with open(json_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                val = cached["validation"]
                h_num = sum(1 for v in val["headings_found"].values() if v)
                h_any = val["headings_any_count"]
                copied = val["copied_source_headers"]
                cited_str = ";".join(str(c) for c in val["cited"])
                ref_list = val["has_reference_list"]
                elapsed = cached["timings"]["total_duration"]
                summary_rows.append({
                    "index": idx,
                    "true_class": true_class,
                    "predicted_class": explanation.get("predicted_class"),
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "tier": tier,
                    "mode": mode,
                    "words": val["word_count"],
                    "words_in_range": val["words_in_range"],
                    "headings_numbered_count": h_num,
                    "headings_any_count": h_any,
                    "cited": cited_str,
                    "citation_ranges_count": len(val.get("citation_ranges", [])),
                    "copied_source_headers": copied,
                    "has_reference_list": ref_list,
                    "attack_ids_not_in_kb": ";".join(val.get("attack_ids_not_in_kb") or []),
                    "kb_attack_ids_named": ";".join(val.get("kb_attack_ids_named") or []),
                    "sp_not_in_kb": ";".join(val.get("sp_not_in_kb") or []),
                    "generate_seconds": round(elapsed, 4),
                    "seed_used": cached.get("seed_used", 42),
                    "report_file": str(txt_file),
                })
                flows_completed += 1
                continue

            prompt_dict = build_prompt(explanation, chunks, mode=mode, cfg=cfg)

            if dry_run:
                p_words = len(prompt_dict["system"].split()) + len(prompt_dict["user"].split())
                print(f"[DRY RUN] Flow {idx} ({mode}): prompt words = {p_words}")
                continue

            try:
                t0 = time.perf_counter()
                gen = generate_report(prompt_dict, cfg)
                elapsed = time.perf_counter() - t0

                n_src = len(prompt_dict["sources"]) if mode == "rag" else 0
                val = validate_report(
                    text=gen["text"],
                    n_sources=n_src,
                    mode=mode,
                    cfg=cfg,
                    prompt_user_text=prompt_dict["user"],
                )

                # Save report text
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(gen["text"])

                # Save JSON context
                summary_dict = {
                    "flow_id": idx,
                    "true_class": true_class,
                    "predicted_class": explanation.get("predicted_class"),
                    "mode": mode,
                    "model": gen["model"],
                    "options": gen["options"],
                    "seed_used": gen["seed_used"],
                    "prompt": prompt_dict,
                    "response": gen["text"],
                    "validation": val,
                    "timings": {
                        "total_duration": gen["total_duration"],
                        "eval_count": gen["eval_count"],
                        "tokens_per_second": gen["tokens_per_second"],
                    },
                }
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(summary_dict, f, indent=2)

                h_num = sum(1 for v in val["headings_found"].values() if v)
                h_any = val["headings_any_count"]
                copied = val["copied_source_headers"]
                cited_str = ";".join(str(c) for c in val["cited"])
                ref_list = val["has_reference_list"]

                print(
                    f"{idx:<8} {true_class:<8} {mode:<8} {val['word_count']:<6} "
                    f"{str(val['words_in_range']):<8} {f'{h_num}/5':<7} {f'{h_any}/5':<7} "
                    f"{copied:<10} {cited_str:<10} {str(ref_list):<8} {elapsed:<7.2f}"
                )

                summary_rows.append({
                    "index": idx,
                    "true_class": true_class,
                    "predicted_class": explanation.get("predicted_class"),
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "tier": tier,
                    "mode": mode,
                    "words": val["word_count"],
                    "words_in_range": val["words_in_range"],
                    "headings_numbered_count": h_num,
                    "headings_any_count": h_any,
                    "cited": cited_str,
                    "citation_ranges_count": len(val.get("citation_ranges", [])),
                    "copied_source_headers": copied,
                    "has_reference_list": ref_list,
                    "has_reference_list": ref_list,
                    "attack_ids_not_in_kb": ";".join(val.get("attack_ids_not_in_kb") or []),
                    "kb_attack_ids_named": ";".join(val.get("kb_attack_ids_named") or []),
                    "sp_not_in_kb": ";".join(val.get("sp_not_in_kb") or []),
                    "generate_seconds": round(elapsed, 4),
                    "seed_used": gen["seed_used"],
                    "report_file": str(txt_file),
                })
                flows_completed += 1

            except Exception as e:
                print(f"ERROR on Flow {idx} ({mode}): {e}", file=sys.stderr)
                flows_failed += 1

    total_elapsed = time.perf_counter() - total_start
    print(f"\nGeneration complete: {flows_completed}/60 reports finished ({flows_failed} failures) in {total_elapsed/60.0:.2f} min.")

    if dry_run or not summary_rows:
        return

    # Write summary CSV matching baseline schema exactly
    summary_csv_path = out_dir / "flow_summary.csv"
    fieldnames = [
        "index", "true_class", "predicted_class", "is_correct", "confidence", "tier",
        "mode", "words", "words_in_range", "headings_numbered_count", "headings_any_count",
        "cited", "citation_ranges_count", "copied_source_headers", "has_reference_list",
        "attack_ids_not_in_kb", "kb_attack_ids_named", "sp_not_in_kb", "generate_seconds",
        "seed_used", "report_file"
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV written to: {summary_csv_path}")

    # Run ROUGE / BLEU evaluation
    print("\nComputing ROUGE and BLEU against reference gold texts...")
    ref_paths = {
        "Benign": root / "data" / "eval" / "reference_benign.txt",
        "DoS": root / "data" / "eval" / "reference_dos.txt",
        "DDoS": root / "data" / "eval" / "reference_ddos.txt",
    }
    refs = {c: clean_text(p.read_text(encoding="utf-8-sig")) for c, p in ref_paths.items()}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    quality_rows = []
    for r in summary_rows:
        txt_path = Path(r["report_file"])
        cand = clean_text(txt_path.read_text(encoding="utf-8-sig"))
        ref = refs[r["true_class"]]
        s = scorer.score(ref, cand)
        bleu = sacrebleu.sentence_bleu(cand, [ref]).score / 100.0
        quality_rows.append({
            "index": r["index"],
            "true_class": r["true_class"],
            "mode": r["mode"],
            "words_scored": len(cand.split()),
            "rouge1": round(s["rouge1"].fmeasure, 6),
            "rouge2": round(s["rouge2"].fmeasure, 6),
            "rougeL": round(s["rougeL"].fmeasure, 6),
            "bleu": round(bleu, 6),
        })

    quality_csv = out_dir / "report_quality_per_report.csv"
    with open(quality_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["index", "true_class", "mode", "words_scored", "rouge1", "rouge2", "rougeL", "bleu"])
        w.writeheader()
        w.writerows(quality_rows)
    print(f"Quality CSV written to: {quality_csv}")

    # Citation checking
    print("\nEvaluating citation overlap against source chunks...")
    thr = float(cfg.get("citation_check", {}).get("overlap_threshold", 0.5))
    citation_lines = []
    cite_matches = 0
    cite_total = 0

    for r in summary_rows:
        if r["mode"] != "rag" or not r["cited"]:
            continue
        idx = r["index"]
        # load context
        with open(flows_out_dir / f"flow_{idx}_rag.json", "r", encoding="utf-8") as f:
            jf = json.load(f)
        report_text = jf["response"]
        sources = jf["prompt"]["sources"]

        # Find sentences with citations
        for sent in re.split(r"(?<=[.!?])\s+", report_text):
            m = re.findall(r"\[(\d{1,2})\]", sent)
            if not m:
                continue
            clean_sent = re.sub(r"\[\d+\]", "", sent).lower()
            sent_words = set(re.findall(r"\b[a-z]{3,}\b", clean_sent))
            if not sent_words:
                continue
            for n_str in m:
                n = int(n_str)
                if 1 <= n <= len(sources):
                    cid = sources[n - 1]["chunk_id"]
                    chunk_text = chunk_map.get(cid, {}).get("text", "").lower()
                    chunk_words = set(re.findall(r"\b[a-z]{3,}\b", chunk_text))
                    overlap = len(sent_words & chunk_words) / len(sent_words) if sent_words else 0.0
                    cite_total += 1
                    verified = overlap >= thr
                    if verified:
                        cite_matches += 1
                    citation_lines.append(
                        f"Flow {idx} [{n}] chunk={cid[:8]} overlap={overlap:.3f} verified={verified}"
                    )

    citation_summary_file = out_dir / "citation_check_summary.txt"
    with open(citation_summary_file, "w", encoding="utf-8") as f:
        f.write(f"Overlap threshold: {thr}\n")
        f.write(f"Total inline citations: {cite_total}\n")
        f.write(f"Verified citations (overlap >= {thr}): {cite_matches}/{cite_total} "
                f"({100.0*cite_matches/cite_total:.1f}%)\n\n" if cite_total else "0 citations\n")
        f.write("\n".join(citation_lines))
    print(f"Citation summary written to: {citation_summary_file}")

    # Generate Aggregate Summary File
    agg_file = out_dir / "aggregate_summary.txt"
    with open(agg_file, "w", encoding="utf-8") as f:
        f.write("=== FULL 30-FLOW BENCHMARK EVALUATION (EXPERIMENT B v2) ===\n\n")
        for mode in ("rag", "vanilla"):
            rs = [r for r in summary_rows if r["mode"] == mode]
            words = [r["words"] for r in rs]
            h5 = sum(1 for r in rs if r["headings_numbered_count"] == 5)
            in_range = sum(1 for r in rs if r["words_in_range"])
            copied = sum(1 for r in rs if r["copied_source_headers"] > 0)
            reflist = sum(1 for r in rs if r["has_reference_list"])
            cites = [r["cited"] for r in rs if r["cited"]]
            f.write(f"MODE: {mode.upper()} (n={len(rs)})\n")
            f.write(f"  Words min/mean/max: {min(words)} / {np.mean(words):.1f} / {max(words)}\n")
            f.write(f"  Words in 300-600 range: {in_range}/{len(rs)} ({100.0*in_range/len(rs):.1f}%)\n")
            f.write(f"  5 Numbered Headings (exact): {h5}/{len(rs)} ({100.0*h5/len(rs):.1f}%)\n")
            f.write(f"  Copied Source Headers: {copied}/{len(rs)}\n")
            f.write(f"  Trailing Reference Lists: {reflist}/{len(rs)}\n")
            f.write(f"  Reports with inline citations: {len(cites)}/{len(rs)}\n")
            f.write(f"  Average generation time: {np.mean([r['generate_seconds'] for r in rs]):.2f}s\n\n")

            for cls in ("Benign", "DoS", "DDoS"):
                sub = [r for r in rs if r["true_class"] == cls]
                sub_w = [r["words"] for r in sub]
                f.write(f"  [{cls}] n={len(sub)}, mean_words={np.mean(sub_w):.1f}, "
                        f"5_headings={sum(1 for r in sub if r['headings_numbered_count']==5)}/{len(sub)}\n")
            f.write("\n" + "=" * 60 + "\n\n")

    print(f"Aggregate report written to: {agg_file}")
    print("\nEVALUATION COMPLETE SUCCESSFULLY.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    run_full_eval(dry_run=args.dry_run, skip_existing=args.skip_existing)

