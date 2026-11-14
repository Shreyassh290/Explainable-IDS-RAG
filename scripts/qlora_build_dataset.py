"""Stage 1: Sample training flows and generate raw QLoRA candidate dataset.

Strict Isolation and Protocol Guarantees:
- Reads ONLY data/processed/train.parquet and train_meta.parquet.
- Excludes any flow index in artifacts/fresh_eval/flows_final.json or
  artifacts/tuning_experiments/pilot_flows.json.
- Refuses to execute if any sampled flow index overlaps with excluded files.
- Executes pipeline up to prompt construction using tuned prompt and RAG mode.
- Generates reports using configured Ollama settings.
- Emits one JSON file per flow in data/qlora/raw/ (resumable, never overwrites).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ids_rag.config import get_project_root, load_config
from ids_rag.models.ensemble import BinaryDNNEnsemble
from ids_rag.qlora.dataset import (
    assert_safe_source_path,
    load_forbidden_indices,
    run_single_flow_raw,
    sample_stratified_train_indices,
)
from ids_rag.rag.index import load_index
from ids_rag.rag.retrieve import get_cached_cross_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw candidate reports for QLoRA fine-tuning.")
    parser.add_argument(
        "--per-class",
        type=int,
        default=60,
        help="Number of flows to sample per class from train.parquet (default: 60, total 180 flows).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42).",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/qlora/raw",
        help="Directory to save raw generated flow JSONs (default: data/qlora/raw).",
    )
    args = parser.parse_args()

    root = get_project_root()
    cfg = load_config()

    train_data_path = root / "data" / "processed" / "train.parquet"
    train_meta_path = root / "data" / "processed" / "train_meta.parquet"
    raw_dir = root / args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 1: QLoRA CANDIDATE DATASET GENERATION")
    print("=" * 80)

    # 1. Strict Source Data Guard
    print(f"Verifying source dataset: {train_data_path}...")
    assert_safe_source_path(train_data_path)
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training dataset missing: {train_data_path}")

    # 2. Load and verify forbidden indices
    print("Loading forbidden indices from evaluation and pilot files...")
    forbidden = load_forbidden_indices(root)
    print(f"Total forbidden flow indices: {len(forbidden):,}")

    # 3. Read train.parquet
    print("Reading data/processed/train.parquet...")
    df_train = pd.read_parquet(train_data_path)
    df_meta = pd.read_parquet(train_meta_path) if train_meta_path.exists() else None
    print(f"Loaded {len(df_train):,} training flows.")

    # 4. Stratified sampling with strict overlap check
    print(f"Sampling {args.per_class} flows per class with seed={args.seed}...")
    sampled_indices = sample_stratified_train_indices(
        df_train=df_train,
        per_class=args.per_class,
        forbidden_indices=forbidden,
        seed=args.seed,
    )
    print(f"Successfully sampled {len(sampled_indices)} flows ({args.per_class} per class).")

    # 5. Load model ensemble, scaler, and KB index
    print("Loading ensemble models, scaler, and knowledge base index...")
    models_dir = root / "models"
    ensemble = BinaryDNNEnsemble.load(models_dir=models_dir, use_temperature=False)
    scaler = joblib.load(models_dir / "scaler.joblib")
    kb = load_index(cfg)
    cross_encoder = get_cached_cross_encoder(cfg)

    # 6. Generate reports for each flow (resumable)
    print("\nStarting generation pipeline (tuned prompt, RAG mode, Ollama inference)...")
    generated_count = 0
    skipped_count = 0

    t0_start = time.time()
    for idx_num, flow_idx in enumerate(sampled_indices, 1):
        out_file = raw_dir / f"flow_{flow_idx}.json"
        if out_file.exists():
            skipped_count += 1
            print(f"[{idx_num}/{len(sampled_indices)}] Flow {flow_idx}: already exists, skipping.")
            continue

        print(f"[{idx_num}/{len(sampled_indices)}] Processing Flow {flow_idx}...", flush=True)
        t_flow = time.time()
        try:
            flow_result = run_single_flow_raw(
                flow_idx=flow_idx,
                df_train=df_train,
                df_meta=df_meta,
                ensemble=ensemble,
                scaler=scaler,
                kb=kb,
                cross_encoder=cross_encoder,
                cfg=cfg,
            )
            elapsed = time.time() - t_flow

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(flow_result, f, indent=2, ensure_ascii=False)

            words = len(flow_result["report"].split())
            print(
                f"    -> Done in {elapsed:.1f}s | Predicted: {flow_result['predicted_class']} | "
                f"Words: {words} | Saved to: {out_file.name}",
                flush=True,
            )
            generated_count += 1

        except Exception as e:
            print(f"    ERROR generating flow {flow_idx}: {e}", file=sys.stderr, flush=True)

    total_time = time.time() - t0_start
    print("\n" + "=" * 80)
    print("GENERATION RUN SUMMARY")
    print("=" * 80)
    print(f"Total sampled flows:    {len(sampled_indices)}")
    print(f"Newly generated:        {generated_count}")
    print(f"Skipped (pre-existing): {skipped_count}")
    print(f"Total raw files:        {len(list(raw_dir.glob('flow_*.json')))}")
    print(f"Elapsed time:           {total_time:.1f}s")
    print(f"Raw output directory:   {raw_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

