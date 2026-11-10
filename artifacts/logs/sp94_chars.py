import re
from pypdf import PdfReader
r = PdfReader("data/kb/NIST_SP_800-94.pdf")
def lines(i):
    return [l.strip() for l in (r.pages[i].extract_text() or "").splitlines() if l.strip()]
KEY = re.compile(r"Introduction|Signature|Network-Based|Appendix A|Executive Summary|Anomaly", re.I)
DOT = re.compile(r"\.{4,}|(?:\.\s){4,}")
for i in range(4, 8):
    for l in lines(i):
        if DOT.search(l) and KEY.search(l):
            print("TOC  p%d" % (i + 1), ascii(l[:80]))
n = 0
for i in range(8, 127):
    for l in lines(i)[:14]:
        if KEY.search(l) and re.match(r"^(\d+\.|\d+\.\d+|Appendix|Executive)", l) and n < 14:
            print("BODY p%d" % (i + 1), ascii(l[:80]))
            n += 1
