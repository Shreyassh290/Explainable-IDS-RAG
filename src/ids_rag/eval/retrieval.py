"""Retrieval evaluation on labelled queries: P@5, Recall@5, section recall, success@5, MRR, latency."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import time

import yaml

from ids_rag.config import get_project_root, load_config
from ids_rag.rag.retrieve import retrieve

SRC = {
    "T98": "MITRE_T1498.html", "T98a": "MITRE_T1498_001.html", "T99": "MITRE_T1499.html",
    "N61": "NIST_SP_800-61_Rev2.pdf", "N94": "NIST_SP_800-94.pdf", "N30": "NIST_SP_800-30_Rev1.pdf",
}
GROUPS = {
    "MITRE attack knowledge": list(range(1, 12)),
    "IDPS (SP 800-94)": [12, 13] + list(range(26, 32)),
    "Incident handling (SP 800-61)": list(range(14, 26)),
    "Risk (SP 800-30)": [32, 33],
}


def expand(label: str) -> tuple[str, str]:
    code, section = label.split("|", 1)
    return SRC[code], section


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/eval/retrieval_queries.yaml")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = get_project_root()
    cfg = load_config()
    items = yaml.safe_load(open(root / args.queries, encoding="utf-8"))["queries"]
    chunks = [json.loads(line) for line in open(root / "data/kb/chunks.jsonl", encoding="utf-8")]
    n_by: dict[tuple[str, str], int] = {}
    for c in chunks:
        key = (c["source_name"], c["section"])
        n_by[key] = n_by.get(key, 0) + 1
    bad = [(it["id"], lab) for it in items for lab in it["rel"] if expand(lab) not in n_by]
    if bad:
        print("LABELS NOT FOUND IN THE KB (fix the query file; nothing computed):")
        for b in bad:
            print("  ", b)
        return 1
    rows, lat = [], []
    for it in items:
        rel = {expand(lab) for lab in it["rel"]}
        n_rel = sum(n_by[k] for k in rel)
        t0 = time.perf_counter()
        res = [r for r in retrieve(it["q"], cfg=cfg) if not r.get("is_fallback")][: args.k]
        lat.append(time.perf_counter() - t0)
        flags = [(r["source_name"], r["section"]) in rel for r in res]
        hits = sum(flags)
        first = next((i + 1 for i, f in enumerate(flags) if f), None)
        found = {(r["source_name"], r["section"]) for r, f in zip(res, flags) if f}
        rows.append({
            "id": it["id"], "query": it["q"], "n_relevant_chunks": n_rel, "n_relevant_sections": len(rel),
            "hits_top5": hits, "p_at_5": hits / args.k, "ceiling_p_at_5": min(args.k, n_rel) / args.k,
            "recall_chunks_at_5": hits / n_rel, "section_recall_at_5": len(found) / len(rel),
            "success": int(hits > 0), "rr": (1 / first) if first else 0.0,
            "top5": " ; ".join(f"{'+' if f else '-'}{r['source_name'].split('.')[0]}|{r['section']}" for r, f in zip(res, flags)),
        })
    for r in rows:
        print(f"[{r['id']:2d}] hits={r['hits_top5']} P@5={r['p_at_5']:.2f} (max {r['ceiling_p_at_5']:.2f}) secRecall={r['section_recall_at_5']:.2f} RR={r['rr']:.2f} | {r['query']}")
        print("      ", r["top5"])
    def mean(key: str, subset: list[dict]) -> float:
        return st.mean(x[key] for x in subset)
    print("=" * 70)
    print("queries:", len(rows))
    for name, sub in [("ALL", rows)] + [(g, [r for r in rows if r["id"] in ids]) for g, ids in GROUPS.items()]:
        print(f"{name:32s} n={len(sub):2d} P@5={mean('p_at_5', sub):.3f} ceilingP@5={mean('ceiling_p_at_5', sub):.3f} "
              f"R@5(chunks)={mean('recall_chunks_at_5', sub):.3f} secRecall@5={mean('section_recall_at_5', sub):.3f} "
              f"success@5={mean('success', sub):.3f} MRR={mean('rr', sub):.3f}")
    steady = lat[1:] if len(lat) > 1 else lat
    print(f"latency seconds: first call {lat[0]:.2f} (includes model loading) | median of the rest {st.median(steady):.2f}")
    out = root / "artifacts" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "retrieval_eval.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump({"rows": rows, "latency_seconds": lat}, open(out / "retrieval_eval.json", "w", encoding="utf-8"), indent=2)
    print("wrote", out / "retrieval_eval.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
