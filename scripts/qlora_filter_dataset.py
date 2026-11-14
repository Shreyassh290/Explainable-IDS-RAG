"""Stage 1: Rejection sampling filter, JSONL dataset export, and token statistics.

Evaluates raw generated reports against strict quality and grounding criteria:
1. All five numbered headings: 1. Rationale, 2. Key Indicators, 3. Confidence Assessment,
   4. Threat Assessment, 5. Recommendations.
2. Word count strictly between 300 and 600 words.
3. Only valid citation numbers within [1, N_sources], no placeholder '[n]', no ranges '[1-3]'.
4. No copied source-header lines.
5. No unsupported claims about botnets, multiple hosts, flooding, or saturation.
6. Inline citation word overlap not below citation_check.overlap_threshold in config.yaml.

Outputs:
- Rejection reason counts and class breakdown.
- Chat-format JSONL exports:
    data/qlora/dataset_train.jsonl (90%)
    data/qlora/dataset_val.jsonl   (10%)
- Token length statistics and flagging of examples > 1,800 tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ids_rag.config import get_project_root, load_config
from ids_rag.qlora.filter import (
    estimate_token_length,
    evaluate_report_filters,
    load_tokenizer_if_available,
    split_and_save_dataset,
)


def load_kb_chunks_by_id(root: Path) -> dict[str, dict[str, Any]]:
    """Load chunks.jsonl into a dictionary keyed by chunk_id."""
    chunks_file = root / "data" / "kb" / "chunks.jsonl"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Knowledge-base chunks file missing: {chunks_file}")

    chunks_by_id: dict[str, dict[str, Any]] = {}
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                chunks_by_id[c["chunk_id"]] = c
    return chunks_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter raw generated reports and export QLoRA dataset.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/qlora/raw",
        help="Directory containing raw generated flow JSONs (default: data/qlora/raw).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/qlora",
        help="Directory to save filtered dataset JSONL files (default: data/qlora).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42).",
    )
    args = parser.parse_args()

    root = get_project_root()
    cfg = load_config()

    raw_dir = root / args.raw_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw candidate directory missing: {raw_dir}")

    # Thresholds from config.yaml
    citation_check_cfg = cfg.get("citation_check", {})
    overlap_threshold = float(citation_check_cfg.get("overlap_threshold", 0.5))
    llm_cfg = cfg.get("llm", {})
    min_words = int(llm_cfg.get("min_words", 300))
    max_words = int(llm_cfg.get("max_words", 600))

    print("=" * 80)
    print("STAGE 1: QLoRA REJECTION SAMPLING FILTER & DATASET COMPILATION")
    print(f"Citation overlap threshold: {overlap_threshold:.2f}")
    print(f"Allowed word count range:   [{min_words}, {max_words}]")
    print("=" * 80)

    # 1. Load KB chunks
    print("Loading knowledge base chunks...")
    chunks_by_id = load_kb_chunks_by_id(root)
    print(f"Loaded {len(chunks_by_id):,} chunks for citation verification.")

    # 2. Iterate through all raw candidate flow JSONs
    raw_files = sorted(raw_dir.glob("flow_*.json"))
    if not raw_files:
        print(f"No candidate flow JSON files found in {raw_dir}.")
        return

    print(f"\nEvaluating {len(raw_files)} candidate flow reports against rejection filters...")

    rejection_counts: Counter[str] = Counter()
    kept_by_class: Counter[str] = Counter()
    kept_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []

    for fpath in raw_files:
        with open(fpath, "r", encoding="utf-8") as f:
            item = json.load(f)

        flow_idx = item.get("flow_index")
        cls_name = item.get("true_class", "UNKNOWN")
        report_text = item.get("report", "")
        retrieved_chunks = item.get("retrieved_chunks", [])
        explanation = item.get("explanation", {})
        prompt_dict = item.get("prompt", {})

        is_accepted, reasons, details = evaluate_report_filters(
            report_text=report_text,
            retrieved_chunks=retrieved_chunks,
            chunks_by_id=chunks_by_id,
            overlap_threshold=overlap_threshold,
            min_words=min_words,
            max_words=max_words,
            explanation=explanation,
        )

        if is_accepted:
            kept_by_class[cls_name] += 1
            kept_records.append({
                "flow_index": flow_idx,
                "class": cls_name,
                "system_text": prompt_dict.get("system", ""),
                "user_text": prompt_dict.get("user", ""),
                "report_text": report_text,
                "word_count": details.get("word_count", len(report_text.split())),
                "citation_count": details.get("citation_count", 0),
            })
        else:
            for r in reasons:
                # Group by prefix before colon/parenthesis for clean summary
                category = r.split(":")[0].split(" ")[0].strip()
                rejection_counts[category] += 1
            rejected_records.append({
                "flow_index": flow_idx,
                "class": cls_name,
                "reasons": reasons,
            })

    # 3. Print Rejection Counts
    print("\n" + "=" * 80)
    print("REJECTION FILTER SUMMARY")
    print("=" * 80)
    print(f"Total candidate reports evaluated: {len(raw_files)}")
    print(f"Total reports accepted:            {len(kept_records)} ({len(kept_records)/len(raw_files)*100:.1f}%)")
    print(f"Total reports rejected:            {len(rejected_records)} ({len(rejected_records)/len(raw_files)*100:.1f}%)")

    print("\nRejection Reasons Breakdown (a single report may have multiple reasons):")
    if rejection_counts:
        for reason, count in rejection_counts.most_common():
            print(f"  - {reason:<35}: {count:>4} instances")
    else:
        print("  None. All reports met criteria.")

    # 4. Print Kept Count per Class
    print("\nKept Examples per Class:")
    for cls_name, count in sorted(kept_by_class.items()):
        print(f"  - Class '{cls_name:<10}': {count:>4} kept examples")

    if not kept_records:
        print("\n[WARNING] No reports met all filter criteria. Dataset JSONL files will be empty.")
        return

    # 5. Token Length Statistics (using Llama 3 tokenizer if available, else estimate)
    print("\n" + "=" * 80)
    print("TOKEN LENGTH ANALYSIS (SYSTEM + USER + ASSISTANT)")
    print("=" * 80)
    tokenizer = load_tokenizer_if_available()
    tok_type = "Llama 3 AutoTokenizer" if tokenizer is not None else "Character/Word Heuristic Estimate"
    print(f"Tokenizer method: {tok_type}")

    token_lengths: list[int] = []
    flagged_over_1800: list[tuple[int, str, int]] = []

    for rec in kept_records:
        full_text = f"{rec['system_text']}\n\n{rec['user_text']}\n\n{rec['report_text']}"
        tokens = estimate_token_length(full_text, tokenizer=tokenizer)
        rec["total_tokens"] = tokens
        token_lengths.append(tokens)

        if tokens > 1800:
            flagged_over_1800.append((rec["flow_index"], rec["class"], tokens))

    tok_arr = np.array(token_lengths)
    print(f"Min tokens:       {int(np.min(tok_arr)):,}")
    print(f"Median tokens:    {int(np.median(tok_arr)):,}")
    print(f"Mean tokens:      {float(np.mean(tok_arr)):,.1f}")
    print(f"90th percentile:  {int(np.percentile(tok_arr, 90)):,}")
    print(f"95th percentile:  {int(np.percentile(tok_arr, 95)):,}")
    print(f"Max tokens:       {int(np.max(tok_arr)):,}")

    if flagged_over_1800:
        print(f"\n[FLAG] {len(flagged_over_1800)} example(s) exceed 1,800 tokens:")
        for f_idx, f_cls, f_tok in flagged_over_1800:
            print(f"  * Flow {f_idx} ({f_cls}): {f_tok} tokens")
    else:
        print("\nAll kept examples are within the 1,800 token budget.")

    # 6. Split and export chat-format JSONL
    print("\n" + "=" * 80)
    print("EXPORTING CHAT-FORMAT JSONL SPLITS")
    print("=" * 80)
    train_file, val_file = split_and_save_dataset(
        kept_records=kept_records,
        out_dir=out_dir,
        val_fraction=0.10,
        seed=args.seed,
    )

    with open(train_file, "r", encoding="utf-8") as f:
        train_count = sum(1 for _ in f)
    with open(val_file, "r", encoding="utf-8") as f:
        val_count = sum(1 for _ in f)

    print(f"Saved training split:   {train_file} ({train_count} flows, ~90%)")
    print(f"Saved validation split: {val_file} ({val_count} flows, ~10%)")

    # 7. Save complete summary artifact
    summary = {
        "total_evaluated": len(raw_files),
        "total_accepted": len(kept_records),
        "total_rejected": len(rejected_records),
        "acceptance_rate": round(len(kept_records) / len(raw_files), 4),
        "rejection_reasons": dict(rejection_counts),
        "kept_per_class": dict(kept_by_class),
        "splits": {
            "train_count": train_count,
            "val_count": val_count,
            "train_file": str(train_file),
            "val_file": str(val_file),
        },
        "token_statistics": {
            "method": tok_type,
            "min": int(np.min(tok_arr)),
            "median": int(np.median(tok_arr)),
            "mean": round(float(np.mean(tok_arr)), 1),
            "p90": int(np.percentile(tok_arr, 90)),
            "p95": int(np.percentile(tok_arr, 95)),
            "max": int(np.max(tok_arr)),
            "flagged_over_1800_count": len(flagged_over_1800),
            "flagged_examples": [
                {"flow_index": f_idx, "class": f_cls, "tokens": f_tok}
                for f_idx, f_cls, f_tok in flagged_over_1800
            ],
        },
    }

    summary_file = out_dir / "filter_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to:  {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

