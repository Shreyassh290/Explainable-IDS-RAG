import csv, re, sys, collections
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
NAMES = {"T1498": "Network Denial of Service", "T1498.001": "Network Denial of Service: Direct Network Flood", "T1499": "Endpoint Denial of Service", "T1489": "Service Stop"}
rows = list(csv.DictReader(open("artifacts/eval/flow_summary.csv", encoding="utf-8")))
pat = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b([^.\n]{0,80})")
cnt = collections.Counter(); ex = {}
for r in rows:
    t = Path(r["report_file"]).read_text(encoding="utf-8")
    for m in pat.finditer(t):
        k = (r["mode"], m.group(1)); cnt[k] += 1; ex.setdefault(k, m.group(0).strip())
for (mode, tid), n in sorted(cnt.items()):
    print(f"{mode:8s} {tid:10s} x{n:<3d} expected: {NAMES.get(tid, 'CHECK https://attack.mitre.org/techniques/' + tid.replace('.', '/') + '/')}")
    print("         example:", ex[(mode, tid)][:110])
