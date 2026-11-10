import csv, json, random, re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
random.seed(42)
root = Path("artifacts/eval")
rows = [r for r in csv.DictReader(open(root / "flow_summary.csv", encoding="utf-8")) if r["mode"] == "rag"]
by = {}
for r in rows:
    by.setdefault(r["true_class"], []).append(r)
pick = []
for cls, k in (("Benign", 3), ("DoS", 4), ("DDoS", 3)):
    pick += random.sample(by[cls], k)
chunks = {}
for line in open("data/kb/chunks.jsonl", encoding="utf-8"):
    c = json.loads(line); chunks[c["chunk_id"]] = c
NAMES = {"T1498": "Network Denial of Service", "T1498.001": "Network Denial of Service: Direct Network Flood", "T1499": "Endpoint Denial of Service"}
f = open(root / "audit_sheet.csv", "w", newline="", encoding="utf-8-sig")
w = csv.writer(f)
w.writerow(["flow", "class", "kind", "sentence", "item", "source_or_name", "section", "source_excerpt", "supported_yes_partial_no", "notes"])
for r in pick:
    j = json.load(open(root / "flows" / f"flow_{r['index']}.json", encoding="utf-8"))
    ch = j.get("retrieved_chunks") or []
    text = Path(r["report_file"]).read_text(encoding="utf-8")
    for s in [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]:
        for n in sorted({int(x) for x in re.findall(r"\[(\d{1,2})\]", s)}):
            c = chunks.get(ch[n - 1].get("chunk_id")) if 1 <= n <= len(ch) else None
            w.writerow([r["index"], r["true_class"], "citation", s, n, c["source_name"] if c else "NO SUCH SOURCE", c["section"] if c else "", c["text"][:400] if c else "", "", ""])
        for t in sorted(set(re.findall(r"\bT\d{4}(?:\.\d{3})?\b", s))):
            w.writerow([r["index"], r["true_class"], "attack_id", s, t, NAMES.get(t, "VERIFY on attack.mitre.org"), "", "", "", ""])
f.close()
print("flows picked:", [(r["index"], r["true_class"]) for r in pick])
print("wrote", root / "audit_sheet.csv")
