"""Citation support checker module.

Scores every inline citation in RAG reports by word overlap between the sentence
and the cited chunk. A citation is flagged 'unverified' when overlap < threshold
defined in config.yaml under citation_check.overlap_threshold.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from ids_rag.config import get_project_root, load_config

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CITE_RE = re.compile(r"\[(\d{1,2}(?:\s*[,;\-\u2013]\s*\d{1,2})*)\]")
FRAG_RE = re.compile(r"^(?:\[[\d\s,;\-\u2013]+\]|[\s.,;:!?])+$")


def expand(body: str) -> list[int]:
    """Expand citation body such as '1-3' or '1, 2' into individual source indices."""
    nums: list[int] = []
    for part in re.split(r"\s*[,;]\s*", body.strip()):
        m = re.fullmatch(r"(\d{1,2})\s*-\s*(\d{1,2})", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            nums.extend(range(a, b + 1) if a <= b else [a, b])
        elif part.isdigit():
            nums.append(int(part))
    return nums


def form_of(body: str) -> str:
    """Categorize citation form as range, list, or single."""
    if "-" in body:
        return "range"
    if "," in body or ";" in body:
        return "list"
    return "single"


def split_units(text: str) -> list[tuple[str, list[str]]]:
    """Split report text into line units and constituent sentence fragments."""
    units: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if units and units[-1][1] and FRAG_RE.match(line):
            units[-1][1][-1] += " " + line
            units[-1] = (units[-1][0] + " " + line, units[-1][1])
            continue
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if p.strip()]
        sents: list[str] = []
        for p in parts:
            if sents and FRAG_RE.match(p):
                sents[-1] += " " + p
            else:
                sents.append(p)
        units.append((line, sents))
    return units


STOP = {
    "that", "this", "with", "from", "have", "been", "will", "such", "which", "their",
    "there", "these", "those", "than", "then", "them", "they", "into", "also", "when",
    "where", "while", "were", "what", "your", "should", "would", "could", "about",
    "more", "most", "some", "other", "only", "both", "each", "does", "very", "like",
}


def clean_sentence(text: Any) -> str:
    """Remove citation tags and leading list markers from sentence."""
    text = re.sub(r"\[[\d\s,\-\u2013]+\]", " ", str(text))
    text = re.sub(r"^\s*(?:[*\-\u2022]|\d+[.)])\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    """Extract lowercased alphanumeric tokens of length >= 4 excluding STOP."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 4 and t not in STOP}


def overlap_score(sentence: str, chunk: str) -> float:
    """Compute word overlap score between sentence and chunk."""
    s = tokens(sentence)
    if not s:
        return 0.0
    return len(s & tokens(chunk)) / len(s)


HDR_RE = re.compile(r"^[#*_>\-\s]*\[(\d+)\]\s+(?:NIST|MITRE)\b", re.IGNORECASE)


def drop_header_lines(text: str) -> tuple[str, int]:
    """Drop copied source-header lines from model report."""
    keep: list[str] = []
    dropped = 0
    for line in text.splitlines():
        if HDR_RE.match(line):
            dropped += 1
        else:
            keep.append(line)
    return "\n".join(keep), dropped


def family(source: Any) -> str:
    """Classify source into NIST, MITRE, or OTHER."""
    u = str(source).upper()
    if u.startswith("MITRE"):
        return "MITRE"
    if u.startswith("NIST"):
        return "NIST"
    return "OTHER"


def extract_flow(
    report_text: str,
    retrieved: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    thr: float,
) -> list[dict[str, Any]]:
    """Extract citation rows from a report text and evaluate overlap against retrieved chunks."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for line, sents in split_units(report_text):
        for s in sents:
            cites: dict[int, str] = {}
            for m in CITE_RE.finditer(s):
                body = m.group(1).replace("\u2013", "-")
                for n in expand(body):
                    cites.setdefault(n, form_of(body))
            for n in sorted(cites):
                key = (s, n)
                if key in seen:
                    continue
                seen.add(key)
                item = retrieved[n - 1] if 1 <= n <= len(retrieved) else None
                c = chunks.get(item.get("chunk_id")) if item else None
                if c is None:
                    score, oor = None, True
                else:
                    score, oor = overlap_score(clean_sentence(s), c["text"]), False
                rows.append({
                    "n": n,
                    "form": cites[n],
                    "sentence": s,
                    "source": (item or {}).get("source", ""),
                    "family": family((item or {}).get("source", "")) if item else "NONE",
                    "overlap": score,
                    "out_of_range": oor,
                    "flagged": oor or (score < thr),
                })
    return rows


def self_check(
    rows: list[dict[str, Any]],
    thr: float,
    eval_dir: Path,
    out_fn: Any = print,
) -> None:
    """Compare checker output on audited flows against reference audit sheet."""
    out_fn("")
    out_fn("SELF-CHECK against the 36 audited rows (labels used here only)")
    paths = {
        "key": eval_dir / "audit_key_v2.csv",
        "sheet": eval_dir / "audit_sheet_v2_scored.csv",
    }
    missing_files = [str(p) for p in paths.values() if not p.exists()]
    if missing_files:
        out_fn("  skipped, missing: " + ", ".join(missing_files))
        return

    with open(paths["key"], encoding="utf-8-sig", newline="") as fh:
        code2idx = {r["flow_code"]: r["flow_index"] for r in csv.DictReader(fh)}
    with open(paths["sheet"], encoding="utf-8-sig", newline="") as fh:
        sheet = [r for r in csv.DictReader(fh) if r.get("kind") == "citation"]

    audited = set(code2idx.values())
    mine = {(r["flow_index"], r["sentence"], str(r["n"])): r for r in rows if r.get("flow_index") in audited}
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing: list[str] = []
    for r in sheet:
        k = (code2idx.get(r["flow_code"]), r["sentence"], r["item"])
        if k in mine:
            matched.append((r, mine[k]))
        else:
            missing.append(r["row_id"])

    out_fn(f"  audited flows: {len(audited)} (expected 10)")
    out_fn(f"  audit sheet citation rows: {len(sheet)} (expected 36)")
    out_fn(f"  checker rows in those flows: {len(mine)} (expected 36)")
    out_fn(f"  matched on (flow, sentence, number): {len(matched)}; sheet rows not found by checker: {missing}")
    out_fn(f"  checker rows with no sheet row: {len(mine) - len(matched)}")
    out_fn(f"  at threshold {thr}: citations kept (overlap >= threshold) by manual label")
    for lab in ("yes", "partial", "no"):
        tot = [m for r, m in matched if r.get("judgment", "").strip().lower() == lab]
        kept = [m for m in tot if not m["flagged"]]
        out_fn(f"    {lab:<8} kept {len(kept)} of {len(tot)}")


def run_citation_check(
    root: Path | None = None,
    out_dir: Path | str | None = None,
    run_self_check: bool = False,
    cfg: dict[str, Any] | None = None,
    reports_dir: Path | str | None = None,
    chunks_file: Path | str | None = None,
) -> dict[str, Any]:
    """Execute citation verification pipeline and produce report summaries."""
    if root is None:
        root = get_project_root()
    if cfg is None:
        cfg = load_config()

    thr = float(cfg["citation_check"]["overlap_threshold"])
    is_phase1 = reports_dir is None

    if not is_phase1:
        if out_dir is None:
            print("STOP: out_dir is required when reports_dir is specified", file=sys.stderr)
            sys.exit(1)
        reports_dir = Path(reports_dir)
        if not reports_dir.is_absolute():
            reports_dir = root / reports_dir
    else:
        if out_dir is None:
            out_dir = root / "artifacts" / "eval" / "citation_check"

    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_rows = out_dir / "citation_check_rows.csv"
    out_reports = out_dir / "citation_check_per_report.csv"
    out_summary = out_dir / "citation_check_summary.txt"

    for p in (out_rows, out_reports, out_summary):
        if p.exists():
            print(f"Refusing to overwrite existing file: {p}", file=sys.stderr)
            sys.exit(1)

    if chunks_file is None:
        kb_path = cfg.get("paths", {}).get("kb", "data/kb") if cfg else "data/kb"
        chunks_file = root / kb_path / "chunks.jsonl"
    else:
        chunks_file = Path(chunks_file)
        if not chunks_file.is_absolute():
            chunks_file = root / chunks_file

    if not chunks_file.exists():
        print(f"STOP: missing file: {chunks_file}", file=sys.stderr)
        sys.exit(1)

    chunks: dict[str, dict[str, Any]] = {}
    with open(chunks_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line_str = line.strip()
            if not line_str:
                continue
            c = json.loads(line_str)
            chunks[c["chunk_id"]] = c

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    all_rows: list[dict[str, Any]] = []
    per_report: list[dict[str, Any]] = []
    hdr_total = 0

    if is_phase1:
        eval_dir = root / "artifacts" / "eval"
        flow_summary_file = eval_dir / "flow_summary.csv"

        if not flow_summary_file.exists():
            print(f"STOP: missing file: {flow_summary_file}", file=sys.stderr)
            sys.exit(1)

        with open(flow_summary_file, "r", encoding="utf-8", newline="") as fh:
            summ = [r for r in csv.DictReader(fh) if r.get("mode") == "rag"]
        summ.sort(key=lambda r: int(r["index"]))

        out(f"RAG reports in flow_summary.csv: {len(summ)} (expected 30)")
        out(f"overlap threshold from config.yaml: {thr}")

        for r in summ:
            idx = r["index"]
            fj = eval_dir / "flows" / f"flow_{idx}.json"
            rp = Path(r["report_file"])
            if not rp.is_absolute():
                rp = root / rp
            if not fj.exists() or not rp.exists():
                print(f"STOP: missing file: {fj if not fj.exists() else rp}", file=sys.stderr)
                sys.exit(1)
            with open(fj, "r", encoding="utf-8") as fh:
                j = json.load(fh)
            retrieved = j.get("retrieved_chunks") or []
            body, n_hdr = drop_header_lines(rp.read_text(encoding="utf-8"))
            hdr_total += n_hdr
            rows = extract_flow(body, retrieved, chunks, thr)
            for x in rows:
                x["flow_index"] = idx
                x["true_class"] = r["true_class"]
                x["predicted_class"] = r["predicted_class"]
            all_rows.extend(rows)
            scored = [x["overlap"] for x in rows if x["overlap"] is not None]
            per_report.append({
                "flow_index": idx,
                "true_class": r["true_class"],
                "predicted_class": r["predicted_class"],
                "citations": len(rows),
                "flagged": sum(1 for x in rows if x["flagged"]),
                "out_of_range": sum(1 for x in rows if x["out_of_range"]),
                "non_single_form": any(x["form"] != "single" for x in rows),
                "mean_overlap": round(sum(scored) / len(scored), 4) if scored else "",
            })
    else:
        if not reports_dir.exists() or not reports_dir.is_dir():
            print(f"STOP: missing directory: {reports_dir}", file=sys.stderr)
            sys.exit(1)

        flow_ids_set: set[str] = set()
        for p in reports_dir.iterdir():
            m = re.match(r"^flow_(.+)_rag\.(?:json|txt)$", p.name)
            if m:
                flow_ids_set.add(m.group(1))

        if not flow_ids_set:
            print(f"STOP: missing flow files in: {reports_dir}", file=sys.stderr)
            sys.exit(1)

        def _sort_key(s: str) -> tuple[int, Any]:
            return (0, int(s)) if s.isdigit() else (1, s)

        flow_ids = sorted(flow_ids_set, key=_sort_key)

        out(f"RAG reports in {reports_dir}: {len(flow_ids)} (expected n/a)")
        out(f"overlap threshold from config.yaml: {thr}")

        for idx in flow_ids:
            fj = reports_dir / f"flow_{idx}_rag.json"
            if not fj.exists():
                print(f"STOP: missing file: {fj}", file=sys.stderr)
                sys.exit(1)
            with open(fj, "r", encoding="utf-8") as fh:
                j = json.load(fh)

            prompt_obj = j.get("prompt")
            sources = prompt_obj.get("sources") if isinstance(prompt_obj, dict) else None
            if sources is None or not isinstance(sources, list):
                print(f"STOP: missing sources list in {fj}", file=sys.stderr)
                sys.exit(1)

            report_text = j.get("response")
            if report_text is None:
                rp = reports_dir / f"flow_{idx}_rag.txt"
                if rp.exists():
                    report_text = rp.read_text(encoding="utf-8")
                else:
                    print(f"STOP: missing response in {fj}", file=sys.stderr)
                    sys.exit(1)

            true_class = j.get("true_class", "")
            predicted_class = j.get("predicted_class", "")

            retrieved = []
            for s in sources:
                item = dict(s)
                if "source" not in item:
                    item["source"] = item.get("source_name", item.get("source_ref", ""))
                retrieved.append(item)

            body, n_hdr = drop_header_lines(report_text)
            hdr_total += n_hdr
            rows = extract_flow(body, retrieved, chunks, thr)
            for x in rows:
                x["flow_index"] = idx
                x["true_class"] = true_class
                x["predicted_class"] = predicted_class
            all_rows.extend(rows)
            scored = [x["overlap"] for x in rows if x["overlap"] is not None]
            per_report.append({
                "flow_index": idx,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "citations": len(rows),
                "flagged": sum(1 for x in rows if x["flagged"]),
                "out_of_range": sum(1 for x in rows if x["out_of_range"]),
                "non_single_form": any(x["form"] != "single" for x in rows),
                "mean_overlap": round(sum(scored) / len(scored), 4) if scored else "",
            })

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_rows, "x", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow([
            "flow_index", "true_class", "predicted_class", "n", "form", "family", "source",
            "overlap", "out_of_range", "flagged", "sentence",
        ])
        for x in all_rows:
            w.writerow([
                x["flow_index"],
                x["true_class"],
                x["predicted_class"],
                x["n"],
                x["form"],
                x["family"],
                x["source"],
                "" if x["overlap"] is None else f"{x['overlap']:.4f}",
                x["out_of_range"],
                x["flagged"],
                x["sentence"],
            ])

    with open(out_reports, "x", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        cols = [
            "flow_index", "true_class", "predicted_class", "citations", "flagged",
            "out_of_range", "non_single_form", "mean_overlap",
        ]
        w.writerow(cols)
        for p in per_report:
            w.writerow([p[c] for c in cols])

    out("")
    out("per report (RAG): flow  class  citations  flagged  out_of_range  mean_overlap")
    for p in per_report:
        out(f"  {p['flow_index']:>6}  {p['true_class']:<7} {p['citations']:>3}  {p['flagged']:>3}  {p['out_of_range']:>3}  {p['mean_overlap']}")

    exp_hdr = "5" if is_phase1 else "n/a"
    out(f"copied source-header lines skipped (same rule as generate.py): {hdr_total} (expected {exp_hdr})")
    n = len(all_rows)
    fl = sum(1 for x in all_rows if x["flagged"])
    oor = sum(1 for x in all_rows if x["out_of_range"])
    all_scored = [x["overlap"] for x in all_rows if x["overlap"] is not None]
    out("")
    exp_at_least_one = "26" if is_phase1 else "n/a"
    exp_non_single = "5" if is_phase1 else "n/a"
    out(f"reports with at least one citation: {sum(1 for p in per_report if p['citations'])} of {len(per_report)} (expected {exp_at_least_one})")
    out(f"reports with a list or range citation: {sum(1 for p in per_report if p['non_single_form'])} (expected {exp_non_single})")
    out(f"unique citation rows: {n}; flagged unverified: {fl} ({(100 * fl / n) if n else 0:.1f}%); outside retrieved list: {oor}")
    if all_scored:
        out(f"overlap over scored citations: mean {sum(all_scored) / len(all_scored):.3f}, median {statistics.median(all_scored):.3f}")
    out("")
    out("by true class: citations / flagged")
    for cls in ("Benign", "DoS", "DDoS"):
        sub = [x for x in all_rows if x["true_class"] == cls]
        out(f"  {cls:<7} {len(sub):>3} / {sum(1 for x in sub if x['flagged'])}")
    out("by cited source family: citations / flagged")
    for fam in ("NIST", "MITRE", "OTHER", "NONE"):
        sub = [x for x in all_rows if x["family"] == fam]
        if sub:
            out(f"  {fam:<7} {len(sub):>3} / {sum(1 for x in sub if x['flagged'])}")

    if is_phase1 and run_self_check:
        self_check(all_rows, thr, eval_dir, out_fn=out)

    out("")
    out("Caveats: the flag is a word-overlap heuristic, not a verdict. It was informed by the 36 audited rows,")
    out("so those rows cannot test it. Overlap separated 'yes' from 'partial+no' (AUC 0.94) but not 'partial'")
    out("from 'no' (AUC 0.71). A flagged citation means 'not clearly supported by the chunk', nothing stronger.")

    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "all_rows": all_rows,
        "per_report": per_report,
        "hdr_total": hdr_total,
        "summary_text": "\n".join(lines),
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for citation support checker."""
    parser = argparse.ArgumentParser(description="Citation support checker.")
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Custom reports directory holding flow_<id>_rag.json and flow_<id>_rag.txt files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (required when --reports-dir is given, default: artifacts/eval/citation_check).",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run self-check against the 36 audited rows.",
    )
    args = parser.parse_args(argv)

    if args.reports_dir is not None and args.out_dir is None:
        print("STOP: --out-dir is required when --reports-dir is specified", file=sys.stderr)
        sys.exit(1)

    root = get_project_root()
    cfg = load_config()

    if args.out_dir is None:
        out_dir = root / "artifacts" / "eval" / "citation_check"
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir

    reports_dir = Path(args.reports_dir) if args.reports_dir is not None else None
    if reports_dir is not None and not reports_dir.is_absolute():
        reports_dir = root / reports_dir

    run_citation_check(
        root=root,
        out_dir=out_dir,
        run_self_check=args.self_check,
        cfg=cfg,
        reports_dir=reports_dir,
    )


if __name__ == "__main__":
    main()

