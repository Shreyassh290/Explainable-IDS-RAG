"""Write data/kb/KB_MANIFEST.md from measured values (hashes, counts, document metadata)."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import html
import json
import re
from pathlib import Path

import yaml
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
KB = ROOT / "data" / "kb"
MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
         r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
DATE_RE = re.compile(r"\b" + MONTH + r"\.?\s+(?:\d{1,2},\s+)?\d{4}\b")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pdf_info(path: Path) -> dict:
    reader = PdfReader(str(path))
    cover, dates = [], []
    for i in range(min(3, len(reader.pages))):
        lines = [l.strip() for l in (reader.pages[i].extract_text() or "").splitlines() if l.strip()]
        cover.append(f"p{i + 1}: " + " | ".join(lines[:4])[:220])
        for m in DATE_RE.finditer(" ".join(lines)):
            if m.group(0) not in dates:
                dates.append(m.group(0))
    return {"pages": len(reader.pages), "cover": cover, "dates": dates[:8]}


def html_info(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    t = re.search(r"<title>(.*?)</title>", raw, flags=re.S | re.I)
    body = re.sub(r"<(script|style).*?</\1>", " ", raw, flags=re.S | re.I)
    body = html.unescape(re.sub(r"<[^>]+>", " ", body))
    body = re.sub(r"\s+", " ", body)
    pick = lambda pat: (re.search(pat, body).group(1) if re.search(pat, body) else "NOT FOUND")
    return {
        "title": re.sub(r"\s+", " ", html.unescape(t.group(1))).strip() if t else "NOT FOUND",
        "version": pick(r"Version:\s*([\d.]+)"),
        "created": pick(r"Created:\s*(\d{1,2} \w+ \d{4})"),
        "modified": pick(r"Last Modified:\s*(\d{1,2} \w+ \d{4})"),
    }


def main() -> None:
    chunks = [json.loads(l) for l in open(KB / "chunks.jsonl", encoding="utf-8")]
    per_doc = collections.defaultdict(list)
    for c in chunks:
        per_doc[c["source_name"]].append(c)
    meta = json.loads((KB / "index" / "index_meta.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    rag = cfg.get("rag", {})
    out = [
        "# Knowledge base manifest",
        "",
        f"Frozen on {dt.date.today().isoformat()} for the evaluation. Do not add, remove or edit documents.",
        "",
        f"- chunks.jsonl SHA-256: {sha256(KB / 'chunks.jsonl')}",
        f"- total chunks: {len(chunks)} | index chunk_count: {meta.get('chunk_count')} | "
        f"index embedding dim: {meta.get('embedding_dim')} | embed model: {meta.get('embed_model')}",
        f"- index_meta chunks_sha256: {meta.get('chunks_sha256')}",
        "- retrieval settings (config.yaml, rag section): "
        + ", ".join(f"{k}={rag[k]}" for k in sorted(rag)),
        "",
        "## Documents",
        "",
    ]
    files = sorted(p for p in KB.iterdir() if p.suffix.lower() in (".pdf", ".html"))
    for p in files:
        rows = per_doc.get(p.name, [])
        secs = list(dict.fromkeys(r["section"] for r in rows))
        out += [
            f"### {p.name}",
            f"- SHA-256: {sha256(p)}",
            f"- size: {p.stat().st_size} bytes | chunks: {len(rows)} | distinct sections: {len(secs)}",
        ]
        if p.suffix.lower() == ".pdf":
            info = pdf_info(p)
            out.append(f"- pages: {info['pages']} | dates found on pages 1-3: {', '.join(info['dates']) or 'NONE'}")
            out += [f"- first lines {c}" for c in info["cover"]]
        else:
            info = html_info(p)
            out.append(f"- page title: {info['title']}")
            out.append(f"- ATT&CK version: {info['version']} | created: {info['created']} | last modified: {info['modified']}")
        out.append("")
    out += [
        "## Content rules applied by chunking.py",
        "",
        "- SP 800-61 Rev 2: table of contents, front matter, section 1.1 Authority, Appendix D and Appendix H dropped.",
        "- SP 800-94: front matter, section 1.1 Authority, Appendix B, C and D dropped; Appendix A kept.",
        "- SP 800-30 Rev 1: only chapters 1 to 3 kept; front matter and all appendices dropped.",
        "- MITRE pages: navigation block and References section removed.",
        "- Footnote numbers glued to words removed from the three PDFs (lists in artifacts/logs/footnote_subs_*.txt).",
        "- Known limits: footnote text and figure text can remain inside chunks; PDF bullet glyphs are private-use characters.",
        "",
        "## Differences from the paper's knowledge base",
        "",
        "- Paper: 14 documents and 5,234 chunks. Here: the documents above and the chunk total above.",
        "- NIST SP 800-30 is an addition (the paper cites its risk principles for the Threat Assessment section).",
        "- SP 800-61 Rev 2 was withdrawn by NIST on 3 April 2025 (the PDF's own first page says so); it is used because the paper uses it.",
        "- Embeddings are local (all-mpnet-base-v2), not an OpenAI API.",
        "",
    ]
    (KB / "KB_MANIFEST.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote", KB / "KB_MANIFEST.md")


if __name__ == "__main__":
    main()
