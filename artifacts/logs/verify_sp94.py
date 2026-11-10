import json, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rows = [json.loads(l) for l in open("data/kb/chunks.jsonl", encoding="utf-8")]
sp = [r for r in rows if r["source_name"] == "NIST_SP_800-94.pdf"]
secs = list(dict.fromkeys(r["section"] for r in sp))
print("SP 800-94 chunks:", len(sp), "sections:", len(secs))
print("labels with a non-ASCII dash:", [ascii(s) for s in secs if re.search("[\u2010-\u2015\u2212]", s)])
print("first 12 labels:", [ascii(s) for s in secs[:12]])
print("last 6 labels:", [ascii(s) for s in secs[-6:]])
print("appendix B/C/D labels present:", [s for s in secs if re.match(r"Appendix [BCD]\b", s)])
idx = re.compile(r"\b\d{1,2}-\d{1,2}(?:,\s*\d{1,2}-\d{1,2}){2,}")
hits = [r for r in sp if idx.search(r["text"])]
print("chunks with index-like page lists:", len(hits))
for r in hits[:3]:
    m = idx.search(r["text"])
    print("   ", r["section"], "::", r["text"][max(0, m.start() - 60):m.end() + 30].replace("\n", " "))
print("largest sections:", collections.Counter(r["section"] for r in sp).most_common(8))
