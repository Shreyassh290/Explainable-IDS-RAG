"""Extract 3 raw rows per class from original CICIDS2018 CSVs for V2 demonstration.

Targets 6 classes:
- Benign (3 rows)
- DoS (3 rows: SlowHTTPTest, Hulk, GoldenEye or any DoS)
- DDoS (3 rows: LOIC-HTTP, HOIC or any DDoS)
- Bot (3 rows)
- FTP-BruteForce (up to 3 rows; prints count found)
- SSH-Bruteforce (3 rows)

Saves strictly to app/raw_samples_v2.csv. Refuses to write to app/raw_samples.csv.
Skips rows with Infinity, -Infinity, or NaN in any of the 41 feature columns.
Includes a 'Provenance' column recording source file and 0-based data row index.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parent.parent

# Output and Frozen Safety Guards
out_dir = repo_root / "app"
out_csv = out_dir / "raw_samples_v2.csv"
frozen_csv = repo_root / "app" / "raw_samples.csv"

assert out_csv.resolve() != frozen_csv.resolve(), "CRITICAL: Refusing to overwrite app/raw_samples.csv"
assert out_csv.name == "raw_samples_v2.csv", "CRITICAL: Output must be app/raw_samples_v2.csv"

# Load expected 41 feature names
feature_names_path = repo_root / "models_v2" / "feature_names_v2.json"
if not feature_names_path.exists():
    feature_names_path = repo_root / "models" / "feature_names.json"

with open(feature_names_path, "r", encoding="utf-8") as f:
    FEATURE_NAMES: list[str] = json.load(f)

# Locate raw CICIDS2018 CSV directory
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

TARGET_CLASSES = {
    "Benign": {
        "file": "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
        "match": lambda lbl: lbl.strip().lower() == "benign",
        "target_count": 3,
    },
    "DoS": {
        "file": "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
        "match": lambda lbl: any(sub in lbl.strip().lower() for sub in ["dos", "slowhttptest", "hulk", "goldeneye"]),
        "target_count": 3,
    },
    "DDoS": {
        "file": "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
        "match": lambda lbl: any(sub in lbl.strip().lower() for sub in ["ddos", "loic-http", "hoic"]),
        "target_count": 3,
    },
    "Bot": {
        "file": "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
        "match": lambda lbl: lbl.strip().lower() == "bot",
        "target_count": 3,
    },
    "FTP-BruteForce": {
        "file": "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
        "match": lambda lbl: "ftp" in lbl.strip().lower(),
        "target_count": 3,
    },
    "SSH-Bruteforce": {
        "file": "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
        "match": lambda lbl: "ssh" in lbl.strip().lower(),
        "target_count": 3,
    },
}


def is_valid_numeric_row(row: dict[str, str], feature_cols: list[str]) -> bool:
    """Check that all required feature columns exist and contain valid finite numbers."""
    for col in feature_cols:
        val = row.get(col)
        if val is None or val == "":
            return False
        try:
            fval = float(val)
            if math.isinf(fval) or math.isnan(fval):
                return False
        except (ValueError, TypeError):
            return False
    return True


def extract_demo_samples(rng_seed: int = 42) -> list[dict[str, Any]]:
    """Extract up to 3 valid rows per target class with reproducibility."""
    rng = random.Random(rng_seed)
    all_extracted: list[dict[str, Any]] = []

    for cls_name, info in TARGET_CLASSES.items():
        file_path = raw_dir / info["file"]
        if not file_path.exists():
            candidate_files = list(raw_dir.glob("*.csv"))
        else:
            candidate_files = [file_path]

        cls_candidates: list[dict[str, Any]] = []

        for csv_file in candidate_files:
            try:
                with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        continue

                    label_col = None
                    for col in reader.fieldnames:
                        if col.strip().lower() == "label":
                            label_col = col
                            break
                    if not label_col:
                        continue

                    # 0-based data row index
                    for row_idx, row in enumerate(reader):
                        raw_lbl = row.get(label_col, "")
                        if info["match"](raw_lbl):
                            # Clean whitespace in keys
                            cleaned_row = {k.strip(): v.strip() for k, v in row.items() if k}
                            if is_valid_numeric_row(cleaned_row, FEATURE_NAMES):
                                provenance = f"{csv_file.name}:{row_idx}"
                                cls_candidates.append({
                                    "row_data": cleaned_row,
                                    "class": cls_name,
                                    "label": raw_lbl.strip(),
                                    "provenance": provenance,
                                })
                                # Stop collecting if we have accumulated enough candidates for this class
                                if len(cls_candidates) >= 100:
                                    break
            except Exception as e:
                print(f"Warning: error reading {csv_file.name}: {e}", file=sys.stderr)
                continue

            if len(cls_candidates) >= info["target_count"] * 10:
                break

        count_found = len(cls_candidates)
        target = info["target_count"]
        if cls_name == "FTP-BruteForce":
            print(f"FTP-BruteForce candidate rows found: {count_found} (target: {target})")

        if count_found == 0:
            print(f"ERROR: Could not find any valid row for class '{cls_name}'.", file=sys.stderr)
            sys.exit(1)

        # Select reproducibly
        selected_count = min(target, count_found)
        selected = rng.sample(cls_candidates, selected_count)

        for item in selected:
            # Build output row: 41 features + Label + Provenance
            out_row: dict[str, Any] = {feat: item["row_data"][feat] for feat in FEATURE_NAMES}
            out_row["Label"] = item["label"]
            out_row["Provenance"] = item["provenance"]
            all_extracted.append(out_row)

        print(f"Selected {selected_count} rows for [{cls_name}]")

    return all_extracted


def main() -> None:
    # Explicit safety checks
    assert out_csv.resolve() != frozen_csv.resolve(), "Safety check failed: cannot overwrite app/raw_samples.csv"
    assert out_csv.name == "raw_samples_v2.csv", "Safety check failed: target filename mismatch"

    samples = extract_demo_samples(rng_seed=42)

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(FEATURE_NAMES) + ["Label", "Provenance"]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    print(f"\nSuccessfully wrote {len(samples)} demo rows to: {out_csv}")
    print(f"Verified output is NOT app/raw_samples.csv.")


if __name__ == "__main__":
    main()

