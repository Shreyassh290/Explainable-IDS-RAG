"""Comparison script for pilot tuning experiments.

Reads baseline results for the pilot flows and compares against Experiment B
and Experiment C, producing comparison tables and qualitative diffs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ids_rag.config import get_project_root


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_comparison() -> str:
    root = get_project_root()
    tuning_dir = root / "artifacts" / "tuning_experiments"

    pilot_file = tuning_dir / "pilot_flows.json"
    if not pilot_file.exists():
        return "pilot_flows.json not found"

    with open(pilot_file, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)
    pilot_ids = {str(item["flow_id"]): item for item in pilot_data["pilot_flows"]}

    # Baseline flows from original eval
    baseline_csv = root / "artifacts" / "eval" / "flow_summary.csv"
    baseline_rows = [r for r in load_csv(baseline_csv) if str(r.get("index")) in pilot_ids]

    # Exp B
    exp_b_csv = tuning_dir / "exp_B" / "summary_exp_B.csv"
    exp_b_rows = load_csv(exp_b_csv)

    # Exp B v2 (Revised Grounding & Anti-Hallucination)
    exp_b2_csv = tuning_dir / "exp_B_v2" / "summary_exp_B_v2.csv"
    exp_b2_rows = load_csv(exp_b2_csv)

    # Exp C
    exp_c_csv = tuning_dir / "exp_C" / "summary_exp_C.csv"
    exp_c_rows = load_csv(exp_c_csv)

    lines: list[str] = [
        "# Pilot Tuning Experiments — Comparison Report",
        "",
        "This report compares the 7 representative pilot flows across Baseline, Experiment B (Initial Prompt Tuning), and Experiment B v2 (Revised Grounding & Anti-Hallucination).",
        "",
        "## Pilot Flow Set Overview",
        "",
        "| Flow ID | Class | Selection Reason |",
        "|---|---|---|",
    ]
    for fid, pinfo in pilot_ids.items():
        lines.append(f"| {fid} | {pinfo['class']} | {pinfo['reason']} |")

    lines.append("")
    lines.append("## Automated Metrics Comparison (RAG Mode)")
    lines.append("")
    lines.append("| Flow ID | Class | Metric | Baseline | Exp B (Initial) | Exp B v2 (Revised) | Target / Ideal |")
    lines.append("|---|---|---|---|---|---|---|")

    def find_row(rows: list[dict[str, Any]], fid: str, mode: str) -> dict[str, Any] | None:
        for r in rows:
            rfid = str(r.get("flow_id") or r.get("index"))
            if rfid == fid and r.get("mode") == mode:
                return r
        return None

    for fid in pilot_ids:
        pinfo = pilot_ids[fid]
        b_row = find_row(baseline_rows, fid, "rag") or {}
        b_exp = find_row(exp_b_rows, fid, "rag") or {}
        b2_exp = find_row(exp_b2_rows, fid, "rag") or {}

        b_words = b_row.get("words", "n/a")
        b_hnum = b_row.get("headings_numbered_count", "n/a")
        b_cited = b_row.get("cited", "none") or "none"
        b_reflist = "Yes" if str(b_row.get("copied_source_headers", "0")) != "0" else "No"

        eb_words = b_exp.get("words", "n/a")
        eb_hnum = b_exp.get("headings_numbered_count", "n/a")
        eb_cited = b_exp.get("cited", "none") or "none"
        eb_reflist = "Yes" if b_exp.get("has_reference_list") is True or str(b_exp.get("has_reference_list")).lower() == "true" else ("No" if "has_reference_list" in b_exp else "n/a")

        eb2_words = b2_exp.get("words", "pending")
        eb2_hnum = b2_exp.get("headings_numbered_count", "pending")
        eb2_cited = b2_exp.get("cited", "pending") or "none"
        eb2_reflist = "Yes" if b2_exp.get("has_reference_list") is True or str(b2_exp.get("has_reference_list")).lower() == "true" else ("No" if "has_reference_list" in b2_exp else "pending")

        lines.append(f"| {fid} | {pinfo['class']} | Words (300-600) | {b_words} | {eb_words} | {eb2_words} | 300–600 |")
        lines.append(f"| {fid} | {pinfo['class']} | Headings (5/5) | {b_hnum}/5 | {eb_hnum}/5 | {eb2_hnum}/5 | 5/5 |")
        lines.append(f"| {fid} | {pinfo['class']} | In-text citations | [{b_cited}] | [{eb_cited}] | [{eb2_cited}] | Grounded |")
        lines.append(f"| {fid} | {pinfo['class']} | Copied/Ref list | {b_reflist} | {eb_reflist} | {eb2_reflist} | No |")
        lines.append("|---|---|---|---|---|---|---|")

    output_text = "\n".join(lines)
    comp_file = tuning_dir / "comparison_table.md"
    with open(comp_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"Comparison report written to: {comp_file}")
    return output_text


if __name__ == "__main__":
    generate_comparison()

