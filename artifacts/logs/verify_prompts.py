import re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SEP = "=" * 40
out = open("artifacts/logs/verify_prompts.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")
files = sorted(Path("artifacts/prompts").glob("*.txt"))
p("files:", [f.name for f in files])
HEADS = ["1. Rationale", "2. Key Indicators", "3. Confidence Assessment", "4. Threat Assessment", "5. Recommendations"]
POS = re.compile(r"near the training average|(?:moderately|far) (?:above|below) the training average")
for f in files:
    text = f.read_text(encoding="utf-8")
    mode = "vanilla" if f.name.endswith("_vanilla.txt") else "rag"
    parts = text.split(SEP)
    p("=" * 70)
    p(f.name, "| mode:", mode, "| split parts:", len(parts), "| words:", len(text.split()), "| rough tokens:", int(len(text.split()) * 1.4))
    system = parts[0]
    user = parts[1] if len(parts) > 1 else ""
    p("  system sentence ok:", "senior cybersecurity analyst" in system and "Cite NIST Special Publications and MITRE ATT&CK" in system)
    p("  five headings present:", all(h in user for h in HEADS))
    p("  value-fit instruction present:", "only when the observed value fits it" in user)
    p("  private-use chars:", len(re.findall("[\ue000-\uf8ff]", text)))
    p("  suspicious tokens (None/nan/NaN/UNCERTAIN/median):", re.findall(r"\bNone\b|\bnan\b|\bNaN\b|UNCERTAIN|\bmedian\b", text))
    p("  'not available' count:", text.count("not available"))
    m = re.search(r"\[RETRIEVED KNOWLEDGE CONTEXT\](.*?)\[REPORT REQUEST\]", user, re.S)
    retr = m.group(1).strip() if m else None
    nums = [int(x.group(1)) for x in re.finditer(r"^\[(\d+)\] (?:NIST|MITRE)", user, re.M)]
    if mode == "vanilla":
        p("  retrieved field empty:", retr == "", "| source lines:", len(nums))
    else:
        p("  source numbers:", nums, "| consecutive from 1:", nums == list(range(1, len(nums) + 1)))
        p("  '(general guidance)' count:", user.count("(general guidance)"))
        p("  grounding rule present:", "Use a source only if it supports the statement" in user)
    p("  Benign paragraph present:", "No attack was detected" in user, "| name has Benign:", "Benign" in f.name)
    blk = re.search(r"\[NETWORK FLOW METADATA\](.*?)\[KEY ANOMALOUS INDICATORS\]", user, re.S)
    p("  --- metadata block ---")
    for line in (blk.group(1).strip().splitlines() if blk else ["MISSING"]):
        p("    ", line)
    ind = re.search(r"\[KEY ANOMALOUS INDICATORS\](.*?)\[RETRIEVED KNOWLEDGE CONTEXT\]", user, re.S)
    p("  --- indicators block ---")
    for line in (ind.group(1).strip().splitlines() if ind else ["MISSING"]):
        p("    ", line)
    p("  position words in indicators:", len(POS.findall(ind.group(1))) if ind else 0)
    if mode == "rag":
        p("  --- source header lines ---")
        for line in re.findall(r"^\[\d+\] (?:NIST|MITRE).*$", user, re.M):
            p("    ", line[:140])
