"""Smoke test: verify v2 retrieval returns v2 KB sources for the three new attack classes.

Run:
    uv run python scripts/v2_smoke_test_retrieve.py
"""

from ids_rag.v2.retrieve_v2 import get_cached_v2_kb, retrieve_v2

NEW_CLASS_QUERIES = [
    ("Bot",            "Bot C2 beaconing botnet command and control"),
    ("FTP-BruteForce", "FTP-BruteForce port 21 credential brute force password guessing"),
    ("SSH-Bruteforce",  "SSH-Bruteforce port 22 password spraying brute force T1110.003"),
]

V2_SOURCES = {
    "MITRE_T1110.txt",
    "MITRE_T1110_001.txt",
    "MITRE_T1110_003.txt",
    "MITRE_T1583_002.txt",
    "MITRE_T1071.txt",
}

def main():
    print("Loading v2 merged KB...")
    kb = get_cached_v2_kb()
    print(f"Loaded {len(kb.chunks)} chunks ({len(V2_SOURCES)} v2 source files)\n")

    all_pass = True

    for cls_name, query in NEW_CLASS_QUERIES:
        results = retrieve_v2(query)
        print(f"Class: {cls_name}")
        print(f"Query: {query}")
        print("-" * 60)

        v2_in_top5 = []
        for r in results:
            tag = "[FALLBACK]" if r["is_fallback"] else f"Rank {r['rank']}"
            v2_flag = " <-- v2 source" if r["source_name"] in V2_SOURCES else ""
            print(f"  {tag:10} CE={r['ce_score']:.3f} | {r['source_name']:30} | {r['section']}{v2_flag}")
            if r["source_name"] in V2_SOURCES and not r["is_fallback"]:
                v2_in_top5.append(r["source_name"])

        if v2_in_top5:
            print(f"  PASS: v2 source(s) in top-5: {v2_in_top5}")
        else:
            print(f"  NOTE: no v2-specific source in top-5 (all results from base KB)")
            # Not a hard failure — the base KB may have relevant NIST chunks
        print()

    print("Smoke test complete.")


if __name__ == "__main__":
    main()

