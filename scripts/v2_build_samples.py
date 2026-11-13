"""Extract sample flows from data/v2/processed/test_extra.parquet into app/samples_v2.json.

Creates one preset benchmark sample for each of the three extra attack classes:
- Bot (Class ID 3)
- FTP-BruteForce (Class ID 4) [INDICATIVE ONLY - N=53]
- SSH-Bruteforce (Class ID 5)

Uses the frozen scaler (transform/inverse_transform only, read-only) to convert
scaled values back to raw feature units for interactive input and display.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root
from ids_rag.v2.prepare_extra import load_v2_config


def build_samples_v2() -> list[dict[str, Any]]:
    """Build preset sample dictionary for V2 extra attack classes."""
    root = get_project_root()
    cfg = load_v2_config()
    paths_cfg = cfg.get("paths", {})

    test_path = root / paths_cfg.get("processed", "data/v2/processed") / "test_extra.parquet"
    meta_path = root / paths_cfg.get("processed", "data/v2/processed") / "test_extra_meta.parquet"
    label_map_path = root / "models_v2" / "label_map_v2.json"
    feature_names_path = root / "models_v2" / "feature_names_v2.json"
    scaler_path = root / paths_cfg.get("scaler", "models/scaler.joblib")

    if not test_path.exists():
        raise FileNotFoundError(f"V2 test data not found at {test_path}")

    df_test = pd.read_parquet(test_path)
    df_meta = pd.read_parquet(meta_path) if meta_path.exists() else None

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names: list[str] = json.load(f)

    scaler = joblib.load(scaler_path)

    descriptions = {
        "Bot": "Botnet command-and-control communication and automated scanning",
        "FTP-BruteForce": "FTP credential dictionary brute-force attack [INDICATIVE ONLY - N=53]",
        "SSH-Bruteforce": "SSH automated credential attack and dictionary password guessing",
    }

    samples_data: list[dict[str, Any]] = []

    for class_id, class_name in sorted(label_map.items()):
        # Find rows belonging to this class
        class_rows = df_test[df_test["label"] == class_id]
        if class_rows.empty:
            continue

        sample_idx = int(class_rows.index[0])
        row_series = class_rows.iloc[0]

        x_scaled = row_series[feature_names].to_numpy(dtype=np.float32)
        x_raw = scaler.inverse_transform(x_scaled.reshape(1, -1))[0]
        raw_features = {k: round(float(v), 6) for k, v in zip(feature_names, x_raw)}

        meta_dict: dict[str, Any] = {}
        if df_meta is not None and sample_idx in df_meta.index:
            meta_row = df_meta.loc[sample_idx]
            for col in ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Protocol"]:
                if col in meta_row:
                    val = meta_row[col]
                    meta_dict[col] = None if pd.isna(val) else str(val)

        display_name = (
            f"{class_name} Sample (Row {sample_idx})"
            if class_name != "FTP-BruteForce"
            else f"{class_name} Sample (Row {sample_idx}) [INDICATIVE ONLY]"
        )

        samples_data.append({
            "name": display_name,
            "index": sample_idx,
            "dataset_label": class_name,
            "class_id": class_id,
            "description": descriptions.get(class_name, ""),
            "features": raw_features,
            "metadata": meta_dict,
            "is_indicative_only": bool(class_name == "FTP-BruteForce"),
        })

    out_file = root / "app" / "samples_v2.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(samples_data, f, indent=2)

    print(f"Saved {len(samples_data)} v2 samples to {out_file}:")
    for s in samples_data:
        print(f"  - {s['name']}: label={s['dataset_label']}, features={len(s['features'])}")

    return samples_data


if __name__ == "__main__":
    build_samples_v2()

