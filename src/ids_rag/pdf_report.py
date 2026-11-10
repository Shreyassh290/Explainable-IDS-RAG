"""PDF incident response report generator using fpdf2.

Generates structured, offline PDF incident response reports
from pipeline summary dictionaries.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from fpdf import FPDF


UNICODE_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "--",
    "\u2022": "*",
    "\u2023": "*",
    "\u25e6": "*",
    "\u2192": "->",
    "\u2190": "<-",
    "\u2191": "^",
    "\u2193": "v",
    "\u2026": "...",
    "\u00a0": " ",
}


def sanitize_text(text: Any) -> str:
    """Replace non-Latin-1 characters with ASCII equivalents and chunk long words."""
    if text is None:
        return ""
    s = str(text)
    for orig, repl in UNICODE_REPLACEMENTS.items():
        s = s.replace(orig, repl)
    # Break unbroken tokens longer than 50 characters to prevent layout overflow
    tokens = s.split(" ")
    safe_tokens: list[str] = []
    for t in tokens:
        if len(t) > 50:
            chunks = [t[i : i + 50] for i in range(0, len(t), 50)]
            safe_tokens.append(" ".join(chunks))
        else:
            safe_tokens.append(t)
    s = " ".join(safe_tokens)
    return s.encode("latin-1", "replace").decode("latin-1")


def is_recommendations_heading(line: str) -> bool:
    """Check if a line is a Recommendations section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) >= 60:
        return False
    if stripped.startswith("- ") or stripped.startswith("* "):
        return False
    trimmed = stripped.lstrip("#*0123456789.: ")
    trimmed = trimmed.rstrip("*: ")
    return trimmed.lower().startswith("recommendation")


def extract_recommendation_steps(report_text: str) -> list[str]:
    """Parse recommended handling steps from the Recommendations section."""
    if not report_text:
        return []

    lines = report_text.splitlines()
    in_recs = False
    rec_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if not in_recs:
            if is_recommendations_heading(stripped):
                in_recs = True
            continue

        # Next major heading ends the section
        if stripped.startswith("#") or (
            stripped.startswith("**")
            and stripped.endswith("**")
            and not any(c in stripped for c in [":", "-", ";"])
        ):
            break
        rec_lines.append(stripped)

    if not rec_lines:
        return []

    steps: list[str] = []
    cite_re = re.compile(r"\s*\[\d+(?:[–\-,]\s*\d+)*\]")

    for raw in rec_lines:
        # Strip markdown bullets: -, *, +, or digits with dot/paren
        cleaned = re.sub(r"^[\s*\-+]+", "", raw)
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)
        # Strip bold/italics markers
        cleaned = cleaned.replace("**", "").replace("__", "")
        cleaned = re.sub(r"(?<!\w)\*|\*(?!\w)", "", cleaned)
        # Strip citation markers like [1], [2, 3], [4-5] along with preceding whitespace
        cleaned = cite_re.sub("", cleaned).strip()
        if cleaned:
            steps.append(cleaned)

    return steps


def build_pdf(summary: dict[str, Any], source_label: str = "") -> bytes:
    """Generate PDF report bytes from a pipeline summary dictionary.

    Args:
        summary: Result dict from run_flow / run_pipeline.
        source_label: Optional label for the data source (e.g. sample name).

    Returns:
        Raw PDF document bytes starting with b"%PDF".
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Title and Generation Timestamp
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(
        0,
        8,
        sanitize_text("Incident Response Report"),
        new_x="LMARGIN",
        new_y="NEXT",
        align="L",
    )

    pdf.set_font("Helvetica", "", 9)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.cell(0, 5, sanitize_text(f"Generated: {ts}"), new_x="LMARGIN", new_y="NEXT")

    # 2. Source line if provided
    if source_label:
        pdf.cell(
            0,
            5,
            sanitize_text(f"Source: {source_label}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(3)

    # 3. Classification Summary
    pred_class = sanitize_text(summary.get("predicted_class", "UNKNOWN"))
    conf = summary.get("confidence", 0.0)
    try:
        conf_val = float(conf)
    except (ValueError, TypeError):
        conf_val = 0.0
    tier = sanitize_text(summary.get("tier", "UNKNOWN"))

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(
        0,
        6,
        sanitize_text("1. Classification Summary"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        5,
        sanitize_text(
            f"Predicted Class: {pred_class}  |  Confidence: {conf_val:.4f}  |  Confidence Tier: {tier}"
        ),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(3)

    # 4. Top-5 Indicators Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(
        0,
        6,
        sanitize_text("2. Key Feature Attribution Indicators"),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    explanation = summary.get("explanation", {})
    features = explanation.get("features", []) if isinstance(explanation, dict) else []
    top_features = features[:5] if isinstance(features, list) else []

    pdf.set_font("Helvetica", "B", 8)
    col_w = [60, 40, 55, 35]
    pdf.cell(col_w[0], 6, sanitize_text("Feature Name"), border=1)
    pdf.cell(col_w[1], 6, sanitize_text("Observed Value"), border=1)
    pdf.cell(col_w[2], 6, sanitize_text("Impact Direction"), border=1)
    pdf.cell(
        col_w[3],
        6,
        sanitize_text("Importance"),
        border=1,
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.set_font("Helvetica", "", 8)
    if top_features:
        for f in top_features:
            fname = sanitize_text(f.get("name", ""))
            val = sanitize_text(f.get("original_value", ""))
            dir_str = (
                "Raises predicted score"
                if f.get("direction") == "raises"
                else "Lowers predicted score"
            )
            imp = f.get("importance", 0.0)
            try:
                imp_str = f"{float(imp):.6f}"
            except (ValueError, TypeError):
                imp_str = sanitize_text(imp)

            pdf.cell(col_w[0], 5, fname, border=1)
            pdf.cell(col_w[1], 5, val, border=1)
            pdf.cell(col_w[2], 5, sanitize_text(dir_str), border=1)
            pdf.cell(
                col_w[3],
                5,
                sanitize_text(imp_str),
                border=1,
                new_x="LMARGIN",
                new_y="NEXT",
            )
    else:
        pdf.cell(
            sum(col_w),
            5,
            sanitize_text("No feature attribution available."),
            border=1,
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(4)

    # Resolve report text (prefer 'rag' over 'vanilla')
    reports = summary.get("reports", {})
    report_text = ""
    if isinstance(reports, dict):
        if "rag" in reports and reports["rag"]:
            report_text = str(reports["rag"])
        elif "vanilla" in reports and reports["vanilla"]:
            report_text = str(reports["vanilla"])
        elif reports:
            report_text = str(next(iter(reports.values())))
    elif isinstance(reports, str):
        report_text = reports

    # 5. Recommended Handling Steps
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(
        0,
        6,
        sanitize_text("3. Recommended Handling Steps"),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    steps = extract_recommendation_steps(report_text)
    if steps:
        pdf.set_font("Helvetica", "", 9)
        for i, step in enumerate(steps, 1):
            pdf.multi_cell(
                0,
                4.5,
                sanitize_text(f"{i}. {step}"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.ln(1)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(
            0,
            4.5,
            sanitize_text(
                "Parsed recommendation steps not found. Showing full report text below:"
            ),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 8.5)
        clean_full = report_text.replace("**", "").replace("__", "")
        pdf.multi_cell(
            0,
            4,
            sanitize_text(clean_full if clean_full else "No report generated."),
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(3)

    # 6. Full Report
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(
        0,
        6,
        sanitize_text("4. Full Incident Report"),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    if report_text:
        for line in report_text.splitlines():
            line_str = line.strip()
            if not line_str:
                pdf.ln(2)
                continue
            is_heading = line_str.startswith("#") or (
                line_str.startswith("**")
                and line_str.endswith("**")
                and len(line_str) > 4
            )
            cleaned_line = line_str.lstrip("#").replace("**", "").replace("__", "").strip()
            if is_heading:
                pdf.set_font("Helvetica", "B", 9.5)
                pdf.multi_cell(
                    0,
                    5,
                    sanitize_text(cleaned_line),
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
            else:
                pdf.set_font("Helvetica", "", 8.5)
                pdf.multi_cell(
                    0,
                    4.5,
                    sanitize_text(cleaned_line),
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
    else:
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(
            0,
            5,
            sanitize_text("No report content available."),
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(3)

    # 7. Retrieved Sources
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(
        0,
        6,
        sanitize_text("5. Retrieved Sources (retrieved automatically, not independently verified)"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font("Helvetica", "", 8)

    chunks = summary.get("chunks", [])
    if isinstance(chunks, list) and chunks:
        for idx, c in enumerate(chunks, 1):
            src_doc = c.get("source", "Unknown")
            sec = c.get("section", "N/A")
            ce = c.get("ce_score")
            try:
                score_str = f"{float(ce):.4f}" if ce is not None else "0.0000"
            except (ValueError, TypeError):
                score_str = sanitize_text(ce)
            entry = f"[{idx}] {src_doc} | Section: {sec} | Score: {score_str}"
            pdf.multi_cell(0, 4, sanitize_text(entry), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(
            0,
            4,
            sanitize_text("No sources recorded."),
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(4)

    # 8. Disclaimer
    pdf.set_font("Helvetica", "I", 7.5)
    disclaimer_text = (
        "Reports are LLM-generated and must be reviewed by a security analyst. "
        "Models trained on CICIDS2018, 3 classes only. "
        "Retrieved sources are not verified to support each statement."
    )
    pdf.multi_cell(
        0,
        3.5,
        sanitize_text(disclaimer_text),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    return bytes(pdf.output())


