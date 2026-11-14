"""Export 50 random kept fine-tuning examples to a readable text file for manual review.

Usage:
  python scripts/qlora_sample_for_review.py [--count 50] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ids_rag.config import get_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Export random kept examples for manual review.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/qlora/dataset_train.jsonl",
        help="Path to JSONL dataset (default: data/qlora/dataset_train.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/qlora/sample_50_for_review.txt",
        help="Output text file path (default: data/qlora/sample_50_for_review.txt).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of samples to export (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    args = parser.parse_args()

    root = get_project_root()
    in_path = root / args.input
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        # Fall back to dataset_train.jsonl + dataset_val.jsonl combined if input not directly found
        train_p = root / "data" / "qlora" / "dataset_train.jsonl"
        val_p = root / "data" / "qlora" / "dataset_val.jsonl"
        candidates = []
        if train_p.exists():
            candidates.append(train_p)
        if val_p.exists():
            candidates.append(val_p)
        if not candidates:
            raise FileNotFoundError(
                f"Dataset file missing: {in_path}. Please run scripts/qlora_filter_dataset.py first."
            )
        input_files = candidates
    else:
        input_files = [in_path]

    records: list[dict] = []
    for f in input_files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if not records:
        print(f"No records found in {input_files}. Exiting.")
        return

    n_sample = min(args.count, len(records))
    rng = np.random.default_rng(args.seed)
    chosen_indices = sorted(rng.choice(len(records), size=n_sample, replace=False))

    print(f"Sampling {n_sample} examples out of {len(records)} total records (seed={args.seed})...")

    sep_double = "=" * 80
    sep_single = "-" * 80

    lines: list[str] = [
        sep_double,
        f"QLoRA FINE-TUNING DATASET: MANUAL REVIEW EXPORT ({n_sample} EXAMPLES)",
        f"Source: {[str(p.name) for p in input_files]} | Seed: {args.seed}",
        sep_double,
        "",
    ]

    for order_idx, r_idx in enumerate(chosen_indices, 1):
        rec = records[r_idx]
        flow_idx = rec.get("flow_index", "UNKNOWN")
        cls_name = rec.get("class", "UNKNOWN")

        messages = rec.get("messages", [])
        system_text = ""
        user_text = ""
        assistant_text = ""

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system_text = content
            elif role == "user":
                user_text = content
            elif role == "assistant":
                assistant_text = content

        words = len(assistant_text.split())

        lines.append(sep_double)
        lines.append(f"EXAMPLE {order_idx} / {n_sample} | Flow Index: {flow_idx} | Class: {cls_name} | Word Count: {words}")
        lines.append(sep_double)
        lines.append("")
        lines.append("[SYSTEM PROMPT]")
        lines.append(sep_single)
        lines.append(system_text.strip())
        lines.append("")
        lines.append("[USER PROMPT (EVIDENCE & RETRIEVED SOURCES)]")
        lines.append(sep_single)
        lines.append(user_text.strip())
        lines.append("")
        lines.append("[ASSISTANT (GENERATED INCIDENT RESPONSE REPORT)]")
        lines.append(sep_single)
        lines.append(assistant_text.strip())
        lines.append("")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(lines))

    print(f"Successfully exported {n_sample} review examples to: {out_path}")


if __name__ == "__main__":
    main()

