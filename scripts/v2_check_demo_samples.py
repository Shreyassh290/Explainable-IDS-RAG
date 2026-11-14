"""Verify V2 demo samples by running 6-class neural ensemble predictions.

Loads app/raw_samples_v2.csv and evaluates each sample flow through the
unified 6-class ensemble (Benign, DoS, DDoS, Bot, FTP-BruteForce, SSH-Bruteforce).
Skips LLM report generation.

Outputs:
- Formatted console table: Source Label | Predicted Class | Confidence | Tier | Match?
- Overall summary counts (matches vs mismatches, confidence tier routing).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.models.ensemble import BinaryDNNEnsemble
from ids_rag.pipeline import validate_raw_features
from ids_rag.v2.pipeline_v2 import PipelineV2Context
from ids_rag.v2.prepare_extra import load_v2_config


def build_six_class_ensemble(ctx: PipelineV2Context) -> BinaryDNNEnsemble:
    """Build a unified 6-class ensemble combining baseline and extra binary DNNs."""
    if len(ctx.ensemble.models) >= 6:
        return ctx.ensemble

    models_dir = repo_root / "models"
    base_ensemble = BinaryDNNEnsemble.load(models_dir=models_dir, use_temperature=False)

    combined_models: dict[int, Any] = {}
    combined_label_map: dict[int, str] = {}

    # 0: Benign, 1: DoS, 2: DDoS
    for cid, m in base_ensemble.models.items():
        combined_models[cid] = m
        combined_label_map[cid] = base_ensemble.label_map[cid]

    # 3: Bot, 4: FTP-BruteForce, 5: SSH-Bruteforce
    for cid, m in ctx.ensemble.models.items():
        combined_models[cid] = m
        combined_label_map[cid] = ctx.ensemble.label_map[cid]

    return BinaryDNNEnsemble(
        models=combined_models,
        label_map=combined_label_map,
        feature_names=ctx.feature_names,
        tiers_config=ctx.ensemble.tiers_config,
        temperatures={name: 1.0 for name in combined_label_map.values()},
        use_temperature=False,
        device=ctx.ensemble.device,
    )


def normalize_source_label(lbl: str) -> str:
    """Map raw dataset label strings to canonical class names."""
    low = lbl.strip().lower()
    if low == "benign":
        return "Benign"
    # Check DDoS before DoS because "ddos" contains the substring "dos"
    if "ddos" in low or "loic" in low or "hoic" in low:
        return "DDoS"
    if "dos" in low or "slowhttp" in low or "hulk" in low or "goldeneye" in low:
        return "DoS"
    if "bot" in low:
        return "Bot"
    if "ftp" in low:
        return "FTP-BruteForce"
    if "ssh" in low:
        return "SSH-Bruteforce"
    return lbl


def main() -> None:
    csv_path = repo_root / "app" / "raw_samples_v2.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} does not exist. Run scripts/v2_extract_demo_samples.py first.", file=sys.stderr)
        sys.exit(1)

    print("Initializing V2 pipeline context and 6-class neural ensemble...")
    cfg = load_v2_config()
    ctx = PipelineV2Context.load(cfg)
    ensemble = build_six_class_ensemble(ctx)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} samples from: {csv_path}\n")

    # Table Header
    header = f"{'#':<3} | {'Source Label':<26} | {'Predicted Class':<18} | {'Conf':<7} | {'Tier':<9} | {'Match?':<6} | {'Provenance'}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    matches = 0
    tier_counts: dict[str, int] = {"VERY_HIGH": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for idx, row in enumerate(rows, start=1):
        raw_label = row.get("Label", "")
        norm_source = normalize_source_label(raw_label)
        provenance = row.get("Provenance", "")

        # Extract only features for prediction (ignore Label and Provenance)
        feature_dict = {
            k: v for k, v in row.items()
            if k.lower() not in ("label", "provenance")
        }

        # Scale features using baseline scaler
        x_scaled = validate_raw_features(
            raw_features=feature_dict,
            feature_names=ctx.feature_names,
            scaler=ctx.scaler,
        )

        # Ensemble prediction (no LLM call)
        pred_labels, confidences, tiers = ensemble.predict(x_scaled.reshape(1, -1))
        pred_id = int(pred_labels[0])
        pred_class = ensemble.label_map.get(pred_id, str(pred_id))
        conf = float(confidences[0])
        tier = tiers[0]

        is_match = (norm_source == pred_class)
        if is_match:
            matches += 1
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        match_str = "YES" if is_match else "NO"
        print(
            f"{idx:<3} | {raw_label:<26} | {pred_class:<18} | {conf:<7.4f} | {tier:<9} | {match_str:<6} | {provenance}"
        )

    print(separator)
    print("\n======================= SUMMARY =======================")
    print(f"Total Demo Flows:         {len(rows)}")
    print(f"Matching Predictions:     {matches}/{len(rows)} ({matches/len(rows)*100:.1f}%)")
    print(f"Confidence Tier Breakdown: {tier_counts}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()

