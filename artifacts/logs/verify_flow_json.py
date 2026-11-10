import json, hashlib, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
j = json.load(open("artifacts/reports/flows/flow_0.json", encoding="utf-8"))
print("top-level keys:", sorted(j.keys()))
real = hashlib.sha256(open("data/kb/chunks.jsonl", "rb").read()).hexdigest()
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, path + "[%d]" % i)
    else:
        yield path, o
for p, v in walk(j):
    if isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v):
        print("sha-like field", p, "matches chunks.jsonl:", v == real)
print("prompt fields mentioning 'ground':", [p for p, v in walk(j) if isinstance(v, str) and "prompt" in p.lower() and "ground" in v.lower()])
print("seed fields:", [(p, v) for p, v in walk(j) if p.lower().endswith("seed") or p.lower().endswith("seed_used")])
print("ground-truth fields:", [(p, v) for p, v in walk(j) if "ground" in p.lower() or "true" in p.lower()][:6])
