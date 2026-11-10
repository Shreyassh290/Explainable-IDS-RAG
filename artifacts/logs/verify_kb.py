import json, re, sys, random, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
random.seed(42)
rows = [json.loads(l) for l in open("data/kb/chunks.jsonl", encoding="utf-8")]
out = open("artifacts/logs/verify_kb.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")
by = collections.defaultdict(list)
for r in rows:
    by[r["source_name"]].append(r)
p("total chunks:", len(rows), "| unique chunk_ids:", len({r["chunk_id"] for r in rows}))
NEW = ("NIST_SP_800-94.pdf", "NIST_SP_800-30_Rev1.pdf")
CHECKS = [
    ("running header SP800-94", r"GUIDE TO INTRUSION DETECTION AND PREVENTION SYSTEMS"),
    ("running header SP800-30", r"Special Publication 800-30"),
    ("underscore rule", r"_{8,}"),
    ("PAGE label", r"\bPAGE\s+(?:[ivxlc]+|[A-L]-\d+|\d+)\b"),
    ("CHAPTER n PAGE", r"CHAPTER\s+\d+\s+PAGE"),
    ("dot leaders", r"\.{4,}"),
    ("blank-page note", r"left blank intentionally"),
    ("TOC words", r"Table of Contents|List of Figures|List of Tables"),
    ("ES- or appendix page labels", r"\bES-\d+\b|\b[A-D]-\d{1,2}\b"),
]
FN = re.compile(r"\b[a-z]{3,}[.,;:]?\d{1,2}(?=\s|$)")
for name, rs in by.items():
    secs = list(dict.fromkeys(r["section"] for r in rs))
    toks = [r["token_count"] for r in rs]
    p("=" * 72)
    p(f"{name}: chunks={len(rs)} sections={len(secs)} tokens min/mean/max={min(toks)}/{sum(toks)/len(toks):.1f}/{max(toks)}")
    p("  chunks over 350 tokens:", sum(1 for t in toks if t > 350))
    if name in NEW:
        p("  sections in order:")
        for s in secs:
            p(f"    {sum(1 for r in rs if r['section'] == s):3d}  {s}")
    for label, pat in CHECKS:
        hits = [(r, m) for r in rs for m in [re.search(pat, r["text"])] if m]
        if hits:
            p(f"  CHECK {label}: {len(hits)} chunks")
            for r, m in hits[:2]:
                a = max(0, m.start() - 50)
                b = min(len(r["text"]), m.end() + 50)
                p("      ...", r["text"][a:b].replace("\n", " "), "...")
    fn = [(r, m) for r in rs for m in FN.finditer(r["text"])]
    p(f"  footnote-like glued tokens: {len(fn)}")
    for r, m in fn[:8]:
        a = max(0, m.start() - 30)
        b = min(len(r["text"]), m.end() + 20)
        p("      ...", r["text"][a:b], "...")
    if name in NEW:
        first = {}
        for r in rs:
            first.setdefault(r["section"], r)
        low = [s for s, r in first.items() if r["text"][:1].islower()]
        p(f"  sections whose first chunk starts lowercase: {len(low)} {low[:10]}")
        for r in random.sample(rs, min(3, len(rs))):
            p(f"  SAMPLE [{r['section']}] {r['token_count']} tok:")
            p("     ", r["text"][:420].replace("\n", " "))
