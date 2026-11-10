import csv, json, re, sys, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
root = Path("artifacts/eval")
out = open(root / "grounding_overlap.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
def toks(t):
    t = re.sub(r"[\ue000-\uf8ff]", " ", t); t = re.sub(r"\[\d{1,2}\]", " ", t)
    return re.findall(r"[a-z0-9]+", t.lower())
def grams(w, n): return {tuple(w[i:i+n]) for i in range(len(w) - n + 1)}
def seq(w, n): return [tuple(w[i:i+n]) for i in range(len(w) - n + 1)]
chunks = {}
for line in open("data/kb/chunks.jsonl", encoding="utf-8"):
    c = json.loads(line); chunks[c["chunk_id"]] = c["text"]
kb3, kb8 = set(), set()
for t in chunks.values():
    w = toks(t); kb3 |= grams(w, 3); kb8 |= grams(w, 8)
p("REFERENCE DOCUMENTS versus the whole knowledge base")
for cls in ("Benign", "DoS", "DDoS"):
    w = toks(Path(f"data/eval/reference_{cls.lower()}.txt").read_text(encoding="utf-8-sig"))
    tri = seq(w, 3)
    p(f"  {cls:7s} trigrams found in KB: {sum(g in kb3 for g in tri) / len(tri):.2f} | 8-word spans copied from KB: {sum(g in kb8 for g in seq(w, 8))}")
rows = list(csv.DictReader(open(root / "flow_summary.csv", encoding="utf-8")))
res = {}
for r in rows:
    j = json.load(open(root / "flows" / f"flow_{r['index']}.json", encoding="utf-8"))
    sw = toks(" ".join(chunks.get(c.get("chunk_id"), "") for c in (j.get("retrieved_chunks") or [])))
    s3, s8 = grams(sw, 3), grams(sw, 8)
    rw = toks(Path(r["report_file"]).read_text(encoding="utf-8-sig"))
    tri = seq(rw, 3)
    res[(r["index"], r["mode"])] = (sum(g in s3 for g in tri) / max(1, len(tri)), sum(g in s8 for g in seq(rw, 8)), r["true_class"])
p("REPORTS versus the chunks retrieved for the same flow")
idx = sorted({i for i, _ in res}, key=int)
for mode in ("vanilla", "rag"):
    v = [res[(i, mode)] for i in idx]
    p(f"  {mode:8s} mean trigram share {st.mean(x[0] for x in v):.3f} | reports with an 8-word copied span {sum(x[1] > 0 for x in v)}/{len(v)} | mean copied 8-word spans {st.mean(x[1] for x in v):.2f}")
a = [res[(i, "rag")][0] for i in idx]; b = [res[(i, "vanilla")][0] for i in idx]
try:
    pv = wilcoxon(a, b).pvalue
except ValueError:
    pv = 1.0
p(f"  paired difference in trigram share (rag - vanilla): {st.mean(x - y for x, y in zip(a, b)):+.3f} | wins {sum(x > y for x, y in zip(a, b))}/{len(idx)} | Wilcoxon p {pv:.3g}")
for cls in ("Benign", "DoS", "DDoS"):
    ii = [i for i in idx if res[(i, "rag")][2] == cls]
    p(f"  {cls:7s} trigram share vanilla {st.mean(res[(i, 'vanilla')][0] for i in ii):.3f} -> rag {st.mean(res[(i, 'rag')][0] for i in ii):.3f}")
