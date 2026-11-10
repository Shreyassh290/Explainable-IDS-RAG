import json, re, collections, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
W = re.compile(r"\S+")
def words(path, name):
    c = collections.Counter()
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r["source_name"] == name:
            c.update(W.findall(r["text"]))
    return c
old = words("backup/chunks_217.jsonl", "NIST_SP_800-61_Rev2.pdf")
new = words("data/kb/chunks.jsonl", "NIST_SP_800-61_Rev2.pdf")
gone = sorted(w for w in old if w not in new)
fn = re.compile(r"^[A-Za-z]*[a-z]{3,}[.,;:]?\d{1,2}$")
print("total words old/new:", sum(old.values()), sum(new.values()))
print("distinct words that vanished completely:", len(gone))
print("  of which look like glued footnotes:", sum(1 for w in gone if fn.match(w)))
print("  others:", [w for w in gone if not fn.match(w)][:40])
