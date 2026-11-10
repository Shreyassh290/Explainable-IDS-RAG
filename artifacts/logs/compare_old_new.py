import json, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
def load(path):
    d = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        d[r["source_name"]].append(r)
    return d
old = load("backup/chunks_217.jsonl")
new = load("data/kb/chunks.jsonl")
out = open("artifacts/logs/compare_old_new.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")
W = re.compile(r"\S+")
for name in old:
    o, n = old[name], new.get(name, [])
    so = list(dict.fromkeys(r["section"] for r in o))
    sn = list(dict.fromkeys(r["section"] for r in n))
    p("=" * 60)
    p(name, "| old chunks", len(o), "| new chunks", len(n))
    p("  same section sequence:", so == sn, "| old sections", len(so), "| new sections", len(sn))
    if so != sn:
        p("  only in old:", [s for s in so if s not in sn][:15])
        p("  only in new:", [s for s in sn if s not in so][:15])
    if name.endswith(".html"):
        same = [(a["section"], a["text"]) for a in o] == [(b["section"], b["text"]) for b in n]
        p("  HTML identical (section and text):", same)
    else:
        co = collections.Counter(w for r in o for w in W.findall(r["text"]))
        cn = collections.Counter(w for r in n for w in W.findall(r["text"]))
        p("  words that lost occurrences (top 40):", (co - cn).most_common(40))
        p("  words that gained occurrences (top 25):", (cn - co).most_common(25))
for name in new:
    if name not in old:
        p("NEW DOCUMENT:", name, "chunks", len(new[name]))
