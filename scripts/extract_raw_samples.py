"""Extract 3 raw rows (one Benign, one DoS, one DDoS) from original CICIDS2018 CSVs.

Saves to app/raw_samples.csv with raw column names and prints the source file,
original row index, and label for each extracted row.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent

# Check data/raw locations
raw_dirs = [
    repo_root / "data" / "raw",
    repo_root.parent / "data" / "raw",
    repo_root.parent / "ids-rag-raw_copy_2026-11-10",
]

raw_dir: Path | None = None
for d in raw_dirs:
    if d.exists() and any(d.glob("*.csv")):
        raw_dir = d
        break

if raw_dir is None:
    print("ERROR: Could not find raw CICIDS2018 CSV directory.", file=sys.stderr)
    sys.exit(1)

print(f"Reading raw CSVs from: {raw_dir}\n")

# Targets to extract
targets = {
    "Benign": {
        "file": "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
        "match": lambda lbl: lbl.strip().lower() == "benign",
    },
    "DoS": {
        "file": "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
        "match": lambda lbl: "dos" in lbl.strip().lower(),
    },
    "DDoS": {
        "file": "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
        "match": lambda lbl: "ddos" in lbl.strip().lower(),
    },
}

extracted_rows: list[dict[str, str]] = []
fieldnames: list[str] = []
metadata_info: list[dict[str, str | int]] = []

for cls_name, info in targets.items():
    file_path = raw_dir / info["file"]
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist, scanning all CSVs for {cls_name}...", file=sys.stderr)
        candidate_files = list(raw_dir.glob("*.csv"))
    else:
        candidate_files = [file_path]

    found = False
    for csv_file in candidate_files:
        with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not fieldnames and reader.fieldnames:
                fieldnames = reader.fieldnames
            label_col = None
            for col in reader.fieldnames or []:
                if col.strip().lower() == "label":
                    label_col = col
                    break
            if not label_col:
                continue

            for row_idx, row in enumerate(reader, start=2):  # start at 2 (row 1 is header)
                raw_lbl = row.get(label_col, "")
                if info["match"](raw_lbl):
                    extracted_rows.append(row)
                    metadata_info.append({
                        "class": cls_name,
                        "source_file": csv_file.name,
                        "row_number": row_idx,
                        "label_in_file": raw_lbl.strip(),
                    })
                    found = True
                    break
        if found:
            break

    if not found:
        print(f"ERROR: Could not find a row for {cls_name}", file=sys.stderr)
        sys.exit(1)

out_dir = repo_root / "app"
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "raw_samples.csv"

# Preserve all unique columns across files (e.g. Tuesday contains IP/Flow ID columns)
all_fieldnames = list(dict.fromkeys(col for row in extracted_rows for col in row.keys()))

with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_fieldnames, restval="")
    writer.writeheader()
    writer.writerows(extracted_rows)

print(f"Successfully extracted {len(extracted_rows)} rows to: {out_csv}\n")
print("Source provenance:")
for meta in metadata_info:
    print(f"  [{meta['class']}] File: {meta['source_file']}, Row: {meta['row_number']}, Label: '{meta['label_in_file']}'")

