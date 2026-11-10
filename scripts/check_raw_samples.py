"""Verify predictions for rows in app/raw_samples.csv using run_flow.

Runs each raw row through run_flow(generate=False, out_dir=temp_dir) and prints:
- Source dataset label (from 'Label' column)
- Predicted class
- Confidence & Tier
- Match status (DoS/DDoS/Benign matching prediction)
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import load_config
from ids_rag.pipeline import PipelineContext, run_flow


def main() -> None:
    csv_path = repo_root / "app" / "raw_samples.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading raw samples from {csv_path}...", flush=True)
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = list(csv.DictReader(f))

    print(f"Loaded {len(reader)} rows from CSV.\n", flush=True)

    cfg = load_config()
    ctx = PipelineContext.load(cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_out = Path(tmp_dir)

        print("=== EVALUATION OF RAW SAMPLES ===")
        all_passed = True
        for i, row in enumerate(reader, 1):
            src_label = row.get("Label", row.get("label", "UNKNOWN")).strip()
            
            # Map metadata if present
            meta = {
                "src_ip": row.get("Src IP", row.get("SrcIP", "")),
                "dst_ip": row.get("Dst IP", row.get("DstIP", "")),
                "src_port": row.get("Src Port", row.get("SrcPort", "")),
                "dst_port": row.get("Dst Port", row.get("DstPort", "")),
                "protocol": row.get("Protocol", row.get("protocol", "")),
                "timestamp": row.get("Timestamp", row.get("timestamp", "")),
            }

            summary = run_flow(
                raw_features=row,
                ctx=ctx,
                cfg=cfg,
                meta=meta,
                flow_id=f"raw_sample_{i}",
                modes="rag",
                generate=False,
                out_dir=temp_out,
            )

            pred_class = summary["predicted_class"]
            conf = summary["confidence"]
            tier = summary["tier"]

            # Expected mapping
            expected_class = None
            if "ddos" in src_label.lower():
                expected_class = "DDoS"
            elif "dos" in src_label.lower():
                expected_class = "DoS"
            elif "benign" in src_label.lower():
                expected_class = "Benign"

            matched = (expected_class == pred_class)
            if not matched:
                all_passed = False

            status = "MATCH" if matched else "MISMATCH"
            print(f"Row {i} [{status}]: Source Label = '{src_label}'")
            print(f"       -> Predicted: {pred_class} (conf: {conf:.4f}, tier: {tier})")
            print(f"       -> Expected:  {expected_class}\n")

    print(f"Result: {'All rows predicted as expected' if all_passed else 'MISMATCH DETECTED'}")
    if not all_passed:
        sys.exit(2)


if __name__ == "__main__":
    main()

