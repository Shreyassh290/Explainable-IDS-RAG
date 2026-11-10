import csv, json, re, sys, collections, statistics as st
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
root = Path("artifacts/eval")
rows = list(csv.DictReader(open(root / "flow_summary.csv", encoding="utf-8")))
out = open(root / "aggregate.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
def truthy(x): return str(x).strip().lower() == "true"
def pct(k, n): return f"{k}/{n} ({100*k/n:.0f}%)" if n else "n/a"
def num(x): return int(float(x)) if str(x).strip() else 0
def load(i): return json.load(open(root / "flows" / f"flow_{i}.json", encoding="utf-8"))
p("rows:", len(rows), "| flows:", len({r["index"] for r in rows}), "| correct flows:", len({r["index"] for r in rows if truthy(r["is_correct"])}))
for mode in ("rag", "vanilla"):
    rs = [r for r in rows if r["mode"] == mode]; n = len(rs)
    words = [num(r["words"]) for r in rs]
    cites = [[int(x) for x in r["cited"].split(";") if x.strip().isdigit()] for r in rs]
    texts = [Path(r["report_file"]).read_text(encoding="utf-8") for r in rs]
    p("=" * 64); p("MODE", mode, "| reports:", n)
    p("  words min/mean/max:", min(words), round(st.mean(words), 1), max(words))
    p("  in 300-600:", pct(sum(300 <= w <= 600 for w in words), n), "| in 400-500:", pct(sum(400 <= w <= 500 for w in words), n))
    p("  all 5 numbered headings:", pct(sum(num(r["headings_numbered_count"]) == 5 for r in rs), n), "| all 5 in any form:", pct(sum(num(r["headings_any_count"]) == 5 for r in rs), n))
    p("  >=1 inline citation:", pct(sum(1 for c in cites if c), n), "| mean distinct citations:", round(st.mean(len(set(c)) for c in cites), 2))
    p("  range citations:", pct(sum(num(r["citation_ranges_count"]) > 0 for r in rs), n), "| copied source headers:", pct(sum(num(r["copied_source_headers"]) > 0 for r in rs), n))
    p("  attack id not in KB:", pct(sum(bool(r["attack_ids_not_in_kb"]) for r in rs), n), "| KB attack id named:", pct(sum(bool(r["kb_attack_ids_named"]) for r in rs), n), "| SP not in KB:", pct(sum(bool(r["sp_not_in_kb"]) for r in rs), n))
    p("  names NIST in text:", pct(sum(bool(re.search(r"\bNIST\b", t)) for t in texts), n), "| names MITRE/ATT&CK/technique id in text:", pct(sum(bool(re.search(r"MITRE|ATT&CK|\bT1\d{3}", t)) for t in texts), n))
    if mode == "rag":
        nc = mc = 0
        for r, c in zip(rs, cites):
            ch = load(r["index"]).get("retrieved_chunks") or []
            srcs = [(ch[k - 1].get("source_name") or ch[k - 1].get("source") or "") for k in set(c) if 1 <= k <= len(ch)]
            nc += any(s.startswith("NIST") for s in srcs); mc += any(s.startswith("MITRE") for s in srcs)
        p("  reports citing a NIST source:", pct(nc, n), "| citing a MITRE source:", pct(mc, n))
    for cls in ("Benign", "DoS", "DDoS"):
        sub = [(w, c) for r, w, c in zip(rs, words, cites) if r["true_class"] == cls]
        if sub:
            p(f"  {cls}: n={len(sub)} words mean={st.mean(w for w, _ in sub):.0f} with citation={sum(1 for _, c in sub if c)}")
sets = collections.defaultdict(set)
for r in rows:
    if r["mode"] == "rag":
        ch = load(r["index"]).get("retrieved_chunks") or []
        sets[r["predicted_class"]].add(tuple(c.get("chunk_id") for c in ch if not c.get("is_fallback")))
p("=" * 64)
p("distinct top-5 chunk sets per predicted class (10 flows each):", {k: len(v) for k, v in sets.items()})
