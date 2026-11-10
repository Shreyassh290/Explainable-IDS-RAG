"""Build the blind audit sheet v2 for the 30-flow RAG evaluation.

Read-only on every input. Writes three NEW files and refuses to overwrite:
  artifacts/eval/audit_sheet_v2.csv   the sheet to score (no class column)
  artifacts/eval/audit_key_v2.csv     flow code -> flow index and true class
  artifacts/eval/audit_rubric_v2.txt  the scoring rules, frozen before scoring
Flow selection is identical to the v1 script (random.seed(42), same sample calls).
"""
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("artifacts/eval")
OUT_SHEET = ROOT / "audit_sheet_v2.csv"
OUT_KEY = ROOT / "audit_key_v2.csv"
OUT_RUBRIC = ROOT / "audit_rubric_v2.txt"

NAMES = {
    "T1498": "Network Denial of Service",
    "T1498.001": "Network Denial of Service: Direct Network Flood",
    "T1499": "Endpoint Denial of Service",
}

CITE_RE = re.compile(r"\[(\d{1,2}(?:\s*[,;\-\u2013]\s*\d{1,2})*)\]")
ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
FRAG_RE = re.compile(r"^(?:\[[\d\s,;\-\u2013]+\]|[\s.,;:!?])+$")

RUBRIC = """AUDIT RUBRIC v2 (frozen before scoring)

Scope: 10 RAG reports (3 Benign, 4 DoS, 3 DDoS), chosen with seed 42.
Each row is one judgment. Fill only the judgment column (and notes if needed).

CITATION rows (kind = citation)
Read the sentence, then the FULL chunk text in the same row.
  yes      The chunk states what the knowledge-base part of the sentence claims.
  partial  The chunk supports part of the claim, or only the general principle
           while the flow-specific part (feature values, confidence, class)
           comes from the prompt and not from the chunk.
  no       The chunk supports none of the claim (sharing a topic is not
           support), or the cited number is outside the retrieved list
           (source_or_name starts with NO SUCH CHUNK).
A citation written as a list or range (form = list or range) gets one row per
number. Judge each number against its own chunk.

ATTACK ID rows (kind = attack_id)
  correct  The id exists and the technique the sentence attaches it to matches
           the technique name in source_or_name.
  wrong    Anything else: the id names a different technique than the sentence
           says, or the id is invented.
If source_or_name says NOT IN KB LIST, look the id up on attack.mitre.org,
judge correct or wrong on the official name, and write "not in KB" in notes.
Judge against the MITRE pages in the knowledge base, not from memory.

Rules
- Score all rows of one flow code together, then move to the next.
- Decide from the text in the row. Do not open audit_key_v2.csv until every
  row is scored.
- Do not edit a judgment after seeing any results. If you change your mind
  about a rule, write it in notes and tell the project lead. Do not re-score
  silently.
- Allowed values: yes, partial, no, correct, wrong. Leave a row blank only if
  you cannot judge it, and say why in notes.
"""


def expand(body):
    nums = []
    for part in re.split(r"\s*[,;]\s*", body.strip()):
        m = re.fullmatch(r"(\d{1,2})\s*-\s*(\d{1,2})", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            nums.extend(range(a, b + 1) if a <= b else [a, b])
        elif part.isdigit():
            nums.append(int(part))
    return nums


def form_of(body):
    if "-" in body:
        return "range"
    if "," in body or ";" in body:
        return "list"
    return "single"


def split_units(text):
    """Return a list of (full_line, [sentences]); bracket-only fragments are
    merged into the sentence before them."""
    units = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if units and units[-1][1] and FRAG_RE.match(line):
            units[-1][1][-1] += " " + line
            units[-1] = (units[-1][0] + " " + line, units[-1][1])
            continue
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if p.strip()]
        sents = []
        for p in parts:
            if sents and FRAG_RE.match(p):
                sents[-1] += " " + p
            else:
                sents.append(p)
        units.append((line, sents))
    return units


def main():
    for p in (OUT_SHEET, OUT_KEY, OUT_RUBRIC):
        if p.exists():
            print("Refusing to overwrite existing file:", p)
            sys.exit(1)

    random.seed(42)
    with open(ROOT / "flow_summary.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["mode"] == "rag"]
    by = {}
    for r in rows:
        by.setdefault(r["true_class"], []).append(r)
    pick = []
    for cls, k in (("Benign", 3), ("DoS", 4), ("DDoS", 3)):
        pick += random.sample(by[cls], k)

    chunks = {}
    with open("data/kb/chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c

    shuffled = pick[:]
    random.Random(42).shuffle(shuffled)

    out_rows = []
    key_rows = []
    stats = {"dup_dropped": 0, "out_of_range": 0, "bracket_only": 0}
    forms = {"single": 0, "list": 0, "range": 0}

    for pos, r in enumerate(shuffled, start=1):
        code = "F%02d" % pos
        idx = r["index"]
        key_rows.append([code, idx, r["true_class"], r["report_file"]])
        with open(ROOT / "flows" / ("flow_%s.json" % idx), encoding="utf-8") as fh:
            j = json.load(fh)
        ch = j.get("retrieved_chunks") or []
        retrieved = []
        for k, item in enumerate(ch, start=1):
            c = chunks.get(item.get("chunk_id"))
            retrieved.append("%d:%s" % (k, c["source_name"] if c else "UNKNOWN"))
        retrieved_sources = " ; ".join(retrieved)

        text = Path(r["report_file"]).read_text(encoding="utf-8")
        seen = set()
        for line, sents in split_units(text):
            for s in sents:
                if FRAG_RE.match(s):
                    stats["bracket_only"] += 1
                cites = {}
                for m in CITE_RE.finditer(s):
                    body = m.group(1).replace("\u2013", "-")
                    for n in expand(body):
                        cites.setdefault(n, form_of(body))
                for n in sorted(cites):
                    key = ("citation", s, n)
                    if key in seen:
                        stats["dup_dropped"] += 1
                        continue
                    seen.add(key)
                    c = None
                    if 1 <= n <= len(ch):
                        c = chunks.get(ch[n - 1].get("chunk_id"))
                    if c is None:
                        stats["out_of_range"] += 1
                    forms[cites[n]] += 1
                    out_rows.append([
                        code, "citation", s, line, str(n), cites[n],
                        c["chunk_id"] if c else "",
                        c["source_name"] if c else "NO SUCH CHUNK (number outside retrieved list)",
                        c["section"] if c else "",
                        c["text"] if c else "",
                        retrieved_sources, "", "",
                    ])
                for t in sorted(set(ID_RE.findall(s))):
                    key = ("attack_id", s, t)
                    if key in seen:
                        stats["dup_dropped"] += 1
                        continue
                    seen.add(key)
                    out_rows.append([
                        code, "attack_id", s, line, t, "", "",
                        NAMES.get(t, "NOT IN KB LIST - look up on attack.mitre.org"),
                        "", "", retrieved_sources, "", "",
                    ])

    header = ["row_id", "flow_code", "kind", "sentence", "full_line", "item", "form",
              "chunk_id", "source_or_name", "section", "chunk_text",
              "retrieved_sources", "judgment", "notes"]
    with open(OUT_SHEET, "x", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, row in enumerate(out_rows, start=1):
            w.writerow(["R%03d" % i] + row)
    with open(OUT_KEY, "x", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["flow_code", "flow_index", "true_class", "report_file"])
        w.writerows(key_rows)
    with open(OUT_RUBRIC, "x", encoding="utf-8") as fh:
        fh.write(RUBRIC)

    n_cit = sum(1 for x in out_rows if x[1] == "citation")
    n_id = sum(1 for x in out_rows if x[1] == "attack_id")
    per_flow = {}
    for x in out_rows:
        per_flow[x[0]] = per_flow.get(x[0], 0) + 1
    longest = max([len(x[9]) for x in out_rows] or [0])
    print("flows picked (v1 order):", [(r["index"], r["true_class"]) for r in pick])
    print("rows written:", len(out_rows), "->", OUT_SHEET)
    print("citation rows:", n_cit, " attack_id rows:", n_id)
    print("citation forms:", forms)
    print("duplicate rows dropped:", stats["dup_dropped"])
    print("sentences still bracket-only:", stats["bracket_only"])
    print("citation numbers outside retrieved list:", stats["out_of_range"])
    print("rows per flow code:", dict(sorted(per_flow.items())))
    print("longest chunk_text (chars):", longest)
    print("wrote", OUT_KEY, "and", OUT_RUBRIC)


main()