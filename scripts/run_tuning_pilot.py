"""Runner script for pilot prompt and parameter tuning experiments.

Loads pilot flows from artifacts/eval/flows/flow_<id>.json, joins chunk texts
from data/kb/chunks.jsonl for exact parity, builds prompts under the experimental
configuration, generates reports via Ollama, validates outputs, and saves all
artifacts strictly into artifacts/tuning_experiments/<exp_name>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

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
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Load explanation and reconstruct retrieved chunk list for a given flow."""
    eval_flow_file = root / "artifacts" / "eval" / "flows" / f"flow_{flow_id}.json"
    if not eval_flow_file.exists():
        raise FileNotFoundError(f"Original eval flow file not found: {eval_flow_file}")

    with open(eval_flow_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    explanation = data["explanation"]
    true_class = data.get("true_class", explanation.get("predicted_class", "UNKNOWN"))

    # Reconstruct chunks with full text
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

    return explanation, full_chunks, true_class


def run_experiment(
    exp_name: str,
    cfg_path: Path,
    flow_ids: list[int],
    dry_run: bool = False,
) -> Path:
    """Run generation and validation for a list of flows under the given experiment config."""
    root = get_project_root()
    chunk_map = load_kb_chunks_by_id(root)

    # Load custom config merged on top of base config
    base_cfg = load_config()
    with open(cfg_path, "r", encoding="utf-8") as f:
        import yaml
        custom_cfg = yaml.safe_load(f)

    cfg = dict(base_cfg)
    if "llm" in custom_cfg:
        cfg["llm"] = {**base_cfg.get("llm", {}), **custom_cfg["llm"]}

    exp_dir = root / "artifacts" / "tuning_experiments" / exp_name
    flows_out_dir = exp_dir / "flows"
    flows_out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = exp_dir / f"summary_{exp_name}.csv"

    if not dry_run:
        print(f"Checking Ollama connectivity for model '{cfg['llm'].get('model', 'llama3:8b')}'...")
        check_ollama(cfg)

    print(f"\n=================================================================")
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"Config: {cfg_path}")
    print(f"Output directory: {flows_out_dir}")
    print(f"Flows to evaluate: {flow_ids}")
    print(f"Parameters: temp={cfg['llm'].get('temperature')}, repeat_penalty={cfg['llm'].get('repeat_penalty')}, prompt_version={cfg['llm'].get('prompt_version')}")
    print(f"=================================================================\n")

    header = (
        f"{'Flow':<8} {'Class':<8} {'Mode':<8} {'Words':<6} {'InRange':<8} "
        f"{'H-num':<7} {'H-any':<7} {'CopiedHdr':<10} {'Cited':<10} {'RefList':<8} {'Time(s)':<7}"
    )
    print(header)
    print("-" * len(header))

    summary_rows: list[dict[str, Any]] = []

    for flow_id in flow_ids:
        explanation, chunks, true_class = load_flow_data(flow_id, root, chunk_map)

        for mode in ("rag", "vanilla"):
            prompt_dict = build_prompt(explanation, chunks, mode=mode, cfg=cfg)

            if dry_run:
                # Dry run: only print prompt sizes
                p_words = len(prompt_dict["system"].split()) + len(prompt_dict["user"].split())
                print(f"[DRY RUN] Flow {flow_id} ({mode}): prompt words = {p_words}")
                continue

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

            # Write text report
            txt_file = flows_out_dir / f"flow_{flow_id}_{mode}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(gen["text"])

            # Write individual JSON summary
            summary_dict = {
                "flow_id": flow_id,
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
            with open(flows_out_dir / f"flow_{flow_id}_{mode}.json", "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, indent=2)

            h_num = sum(1 for v in val["headings_found"].values() if v)
            h_any = val["headings_any_count"]
            copied = val["copied_source_headers"]
            cited_str = ";".join(str(c) for c in val["cited"])
            ref_list = val["has_reference_list"]

            print(
                f"{flow_id:<8} {true_class:<8} {mode:<8} {val['word_count']:<6} "
                f"{str(val['words_in_range']):<8} {f'{h_num}/5':<7} {f'{h_any}/5':<7} "
                f"{copied:<10} {cited_str:<10} {str(ref_list):<8} {elapsed:<7.2f}"
            )

            summary_rows.append(
                {
                    "flow_id": flow_id,
                    "true_class": true_class,
                    "predicted_class": explanation.get("predicted_class"),
                    "mode": mode,
                    "words": val["word_count"],
                    "words_in_range": val["words_in_range"],
                    "headings_numbered_count": h_num,
                    "headings_any_count": h_any,
                    "cited": cited_str,
                    "citation_ranges_count": len(val["citation_ranges"]),
                    "copied_source_headers": copied,
                    "has_reference_list": ref_list,
                    "unsupported_ips": ";".join(val["unsupported_ips"]),
                    "attack_ids_not_in_kb": ";".join(val["attack_ids_not_in_kb"] or []),
                    "kb_attack_ids_named": ";".join(val["kb_attack_ids_named"] or []),
                    "sp_not_in_kb": ";".join(val["sp_not_in_kb"] or []),
                    "elapsed_s": round(elapsed, 2),
                    "tokens_per_second": round(gen["tokens_per_second"], 1),
                }
            )

    if not dry_run and summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary saved to: {summary_csv_path}")

    return exp_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot tuning experiment runner.")
    parser.add_argument("--exp", choices=["exp_B", "exp_B_v2", "exp_C"], default="exp_B_v2", help="Experiment to run.")
    parser.add_argument("--all-pilot", action="store_true", help="Run both exp_B and exp_C sequentially.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and print info without generating.")
    args = parser.parse_args()

    root = get_project_root()
    pilot_file = root / "artifacts" / "tuning_experiments" / "pilot_flows.json"
    with open(pilot_file, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    flow_ids = [item["flow_id"] for item in pilot_data["pilot_flows"]]

    experiments = ["exp_B", "exp_C"] if args.all_pilot else [args.exp]

    for exp in experiments:
        cfg_path = root / "artifacts" / "tuning_experiments" / exp / f"config_{exp}.yaml"
        run_experiment(exp, cfg_path, flow_ids, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

