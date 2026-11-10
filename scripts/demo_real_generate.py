"""One real run_flow test with generate=True against local Ollama (llama3:8b).

Writes to a system temporary directory (never artifacts/reports/flows/).
Prints predicted class, confidence, top features, retrieved sources,
generated report, validation results, and execution timings.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import load_config
from ids_rag.llm.generate import check_ollama
from ids_rag.pipeline import PipelineContext, run_flow


def main() -> None:
    print("Checking Ollama status...", flush=True)
    cfg = load_config()
    try:
        models = check_ollama(cfg)
        print(f"Ollama connected. Available models: {models}", flush=True)
    except Exception as e:
        print(f"ERROR: Ollama check failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Loading PipelineContext...", flush=True)
    ctx = PipelineContext.load(cfg)

    # Use index 0 as the test row
    test_idx = 0
    row_series = ctx.df_test.iloc[test_idx]
    x_scaled = row_series[ctx.feature_names].to_numpy(dtype=np.float32)
    x_raw = ctx.scaler.inverse_transform(x_scaled.reshape(1, -1))[0]
    raw_dict = dict(zip(ctx.feature_names, x_raw))

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_out = Path(tmp_dir)
        print(f"Executing run_flow (mode='rag', generate=True) to temporary out_dir: {temp_out}\n", flush=True)
        t0 = time.perf_counter()
        summary = run_flow(
            raw_features=raw_dict,
            ctx=ctx,
            cfg=cfg,
            flow_id="live_demo_test",
            modes="rag",
            generate=True,
            out_dir=temp_out,
        )
        elapsed = time.perf_counter() - t0

    print("\n================ LIVE RUN COMPLETE ================")
    print(f"Predicted class: {summary['predicted_class']}")
    print(f"Confidence: {summary['confidence']:.4f} ({summary['tier']})")
    print(f"Total wall time: {elapsed:.2f}s")
    print(f"Timings: {summary['timings']}")
    print("\nRetrieved Chunks:")
    for i, c in enumerate(summary.get("chunks", []), 1):
        print(f"  {i}. {c.get('source')} | Section: {c.get('section')} | CE: {c.get('ce_score')}")

    print("\n--- Generated Report (rag) ---")
    print(summary.get("reports", {}).get("rag", ""))

    print("\n--- Validation ---")
    val = summary.get("validation", {}).get("rag", {})
    print(f"Word count: {val.get('word_count')} (in range: {val.get('words_in_range')})")
    print(f"Headings found: {val.get('headings_found')}")
    print(f"Citations: cited={val.get('cited')}, invalid={val.get('invalid_citations')}")
    print("====================================================")


if __name__ == "__main__":
    main()

