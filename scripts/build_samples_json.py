"""Build app/samples.json for the chosen sample indices (Benign=4, DDoS=0, DoS=3).

Saves raw unscaled feature values and metadata with ground truth shown ONLY as
"dataset label" per user constraint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import load_config
from ids_rag.pipeline import PipelineContext


def main() -> None:
    cfg = load_config()
    ctx = PipelineContext.load(cfg)

    chosen = [
        {"class": "Benign", "index": 4, "description": "Normal network traffic pattern"},
        {"class": "DDoS", "index": 0, "description": "High-volume distributed denial-of-service attack"},
        {"class": "DoS", "index": 3, "description": "Denial-of-service volumetric disruption"},
    ]

    samples_data: list[dict] = []
    for item in chosen:
        idx = item["index"]
        cls = item["class"]
        row_series = ctx.df_test.iloc[idx]
        x_scaled = row_series[ctx.feature_names].to_numpy(dtype=np.float32)
        x_raw = ctx.scaler.inverse_transform(x_scaled.reshape(1, -1))[0]
        raw_features = {k: round(float(v), 6) for k, v in zip(ctx.feature_names, x_raw)}

        meta_dict = {}
        if ctx.df_meta is not None and len(ctx.df_meta) > idx:
            meta_row = ctx.df_meta.iloc[idx]
            for col in ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "protocol"]:
                if col in meta_row:
                    val = meta_row[col]
                    meta_dict[col] = None if pd.isna(val) else str(val)

        samples_data.append({
            "name": f"{cls} Sample (Index {idx})",
            "index": idx,
            "dataset_label": cls,
            "description": item["description"],
            "features": raw_features,
            "metadata": meta_dict,
        })

    out_file = repo_root / "app" / "samples.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(samples_data, f, indent=2)

    print(f"Saved {len(samples_data)} samples to {out_file}:")
    for s in samples_data:
        print(f"  - {s['name']}: dataset label = '{s['dataset_label']}', {len(s['features'])} features")


if __name__ == "__main__":
    main()

