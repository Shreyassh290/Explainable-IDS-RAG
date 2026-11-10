"""Score the blind audit (v2). Read-only on all inputs.

Refuses to compute (and never opens the key file) until every row of the scored
file has a valid judgment. Optional second scorer: agreement and Cohen's kappa
on the rows both people judged.
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("artifacts/eval")
CIT = ("yes", "partial", "no")
IDS = ("correct", "wrong")
ORDER = ("Benign", "DoS", "DDoS")


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def frac(k, n):
    return "%d/%d (%.0f%%)" % (k, n, 100.0 * k / n) if n else "0/0"


def structure_problems(rows, orig):
    probs = []
    if len(rows) != len(orig):
        probs.append("row count %d, original has %d" % (len(rows), len(orig)))
    by = {r["row_id"]: r for r in orig}
    for r in rows:
        o = by.get(r.get("row_id"))
        if o is None:
            probs.append("unknown row_id %r" % r.get("row_id"))
            continue
        for col in ("flow_code", "kind", "item", "chunk_id", "sentence"):
            if norm(r.get(col)) != norm(o.get(col)):
                probs.append("%s: column %s was changed" % (r["row_id"], col))
    return probs


def judgments(rows):
    vals, bad, blank = {}, [], []
    for r in rows:
        j = norm(r.get("judgment")).lower()
        allowed = CIT if r["kind"] == "citation" else IDS
        if not j:
            blank.append(r["row_id"])
        elif j not in allowed:
            bad.append((r["row_id"], j))
        else:
            vals[r["row_id"]] = j
    return vals, bad, blank


def kappa(pairs, cats):
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for x, y in pairs if x == y) / n
    ca = Counter(x for x, _ in pairs)
    cb = Counter(y for _, y in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def fmtk(k):
    return "n/a (no variation in labels)" if k is None else "%.2f" % k


def main():
    ap = argparse.ArgumentParser(description="Score the blind audit (v2)")
    ap.add_argument("--scored", default=str(ROOT / "audit_sheet_v2_scored.csv"))
    ap.add_argument("--original", default=str(ROOT / "audit_sheet_v2.csv"))
    ap.add_argument("--key", default=str(ROOT / "audit_key_v2.csv"))
    ap.add_argument("--second", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_lines = []

    def p(*x):
        s = " ".join(str(i) for i in x)
        print(s)
        out_lines.append(s)

    orig = read(a.original)
    rows = read(a.scored)
    probs = structure_problems(rows, orig)
    if probs:
        print("The scored file does not match the original sheet. Nothing computed.")
        for x in probs[:15]:
            print("  ", x)
        print("Save the scored file as CSV UTF-8 and edit only judgment and notes.")
        sys.exit(1)
    vals, bad, blank = judgments(rows)
    if bad or blank:
        print("Not scoring: %d rows have no judgment, %d rows have an invalid value."
              % (len(blank), len(bad)))
        for rid, j in bad[:15]:
            print("   invalid:", rid, repr(j))
        if blank:
            print("   first blank rows:", blank[:10])
        print("The key file was not opened.")
        sys.exit(1)

    cls_of = {k["flow_code"]: k["true_class"] for k in read(a.key)}

    def table(kind, cats, title):
        p(title)
        for cls in ORDER + ("ALL",):
            rs = [r for r in rows if r["kind"] == kind
                  and (cls == "ALL" or cls_of.get(r["flow_code"]) == cls)]
            n = len(rs)
            c = Counter(vals[r["row_id"]] for r in rs)
            flows = len({r["flow_code"] for r in rs})
            p("  %-6s rows %2d (flows %d)  %s"
              % (cls, n, flows, "  ".join("%s %d" % (k, c[k]) for k in cats)))
            if kind == "citation":
                p("         yes: %s | yes or partial: %s"
                  % (frac(c["yes"], n), frac(c["yes"] + c["partial"], n)))
            else:
                p("         correct: %s" % frac(c["correct"], n))

    p("AUDIT RESULTS v2 (scorer 1, rows judged: %d)" % len(rows))
    p("=" * 64)
    table("citation", CIT, "CITATION ROWS (sentence vs the chunk it cites)")
    p("")
    p("CITATION ROWS BY SOURCE TYPE")
    for label in ("NIST", "MITRE"):
        rs = [r for r in rows if r["kind"] == "citation"
              and r["source_or_name"].startswith(label)]
        c = Counter(vals[r["row_id"]] for r in rs)
        n = len(rs)
        p("  %-6s rows %2d  yes %d  partial %d  no %d | yes: %s"
          % (label, n, c["yes"], c["partial"], c["no"], frac(c["yes"], n)))
    p("")
    table("attack_id", IDS, "ATTACK ID ROWS (id and technique name match)")
    p("")
    p("Notes: rows within a flow are not independent, and Benign flows cite the")
    p("same chunk. Report counts, not only percentages. Small sample.")

    if a.second:
        sec = read(a.second)
        sp = structure_problems(sec, orig)
        if sp:
            print("The second scorer's file does not match the original sheet.")
            for x in sp[:15]:
                print("  ", x)
            sys.exit(1)
        v2, bad2, _ = judgments(sec)
        if bad2:
            print("Invalid values in the second file:", bad2[:15])
            sys.exit(1)
        p("")
        p("=" * 64)
        p("AGREEMENT WITH SECOND SCORER (rows judged by both)")
        for kind, cats in (("citation", CIT), ("attack_id", IDS)):
            ids = [r["row_id"] for r in rows if r["kind"] == kind and r["row_id"] in v2]
            pairs = [(vals[i], v2[i]) for i in ids]
            n = len(pairs)
            agree = sum(1 for x, y in pairs if x == y)
            p("  %s: compared %d, agree %s" % (kind, n, frac(agree, n)))
            if kind == "citation" and n:
                coll = [("yes" if x == "yes" else "not_yes",
                         "yes" if y == "yes" else "not_yes") for x, y in pairs]
                p("    kappa (yes/partial/no): %s" % fmtk(kappa(pairs, CIT)))
                p("    kappa (yes vs not yes): %s" % fmtk(kappa(coll, ("yes", "not_yes"))))
                if n < 15:
                    p("    fewer than 15 rows compared: kappa is unstable")
            for i in ids:
                if vals[i] != v2[i]:
                    p("    disagree %s: scorer1=%s scorer2=%s" % (i, vals[i], v2[i]))
        p("  Do not change any judgment because of these disagreements.")

    if a.out:
        try:
            with open(a.out, "x", encoding="utf-8") as fh:
                fh.write("\n".join(out_lines) + "\n")
            print("wrote", a.out)
        except FileExistsError:
            print("Not written, file exists:", a.out)


main()