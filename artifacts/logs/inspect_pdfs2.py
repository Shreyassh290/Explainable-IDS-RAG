import re, sys
from pypdf import PdfReader
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DOT = re.compile(r"\.{4,}|(?:\.\s){4,}")
out = open("artifacts/logs/pdf_structure2.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.write(s + "\n")
def lines_of(r, i):
    return [l.strip() for l in (r.pages[i].extract_text() or "").splitlines() if l.strip()]
def show(r, i, n=26):
    p(f"[p{i+1}]")
    for l in lines_of(r, i)[:n]:
        p("   ", l[:120])
r = PdfReader("data/kb/NIST_SP_800-94.pdf")
p("=== SP 800-94 ===", len(r.pages), "pages")
toc = []
for i in range(4, 8):
    toc += [l for l in lines_of(r, i) if DOT.search(l)]
p("TOC-like lines in p5-p8:", len(toc))
for l in toc:
    if l.lower().startswith(("appendix", "glossary", "index")) or re.match(r"^\d+\.\s", l):
        p("  ", l[:120])
for i in (17, 22):
    show(r, i)
app = [i for i in range(len(r.pages)) if any(re.match(r"^Appendix [A-Z]\b", l) for l in lines_of(r, i)[:4])]
p("pages whose first lines start with 'Appendix X':", [i + 1 for i in app][:20])
if app:
    show(r, app[0], 22)
show(r, len(r.pages) - 1, 12)
r = PdfReader("data/kb/NIST_SP_800-30_Rev1.pdf")
p("=== SP 800-30 ===", len(r.pages), "pages")
show(r, 0, 20)
show(r, 9, 30)
show(r, 12, 30)
show(r, 20, 30)
hdr = {}
for i in range(len(r.pages)):
    for l in lines_of(r, i)[:4]:
        m = re.match(r"^(CHAPTER \d+|APPENDIX [A-Z])\s+PAGE\s+\S+$", l)
        if m:
            hdr.setdefault(m.group(1), i + 1)
p("first page of each chapter/appendix header label:", hdr)
for key in ("APPENDIX D", "APPENDIX H"):
    if key in hdr:
        show(r, hdr[key] - 1, 28)
