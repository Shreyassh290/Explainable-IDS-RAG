"""Evaluation flow selector for benchmark evaluation.

Selects balanced, deterministic evaluation flow indices across all classes:
- Loads class order from data/processed/label_map.json.
- Reads only the label column of data/processed/test.parquet.
- Excludes configured development/prompt tuning rows.
- Draws per_class samples without replacement using a single RNG instance.
- Saves reproducible selection with dataset SHA-256 and metadata to artifacts/eval/eval_flows.json.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ids_rag.config import get_project_root, load_config
from ids_rag.rag.index import compute_file_sha256


def select_flows(
    cfg: dict[str, Any] | None = None,
    test_path: Path | str | None = None,
    label_map_path: Path | str | None = None,
    out_path: Path | str | None = None,
    seed: int | None = None,
    per_class: int | None = None,
    exclude_rows: list[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Select a deterministic evaluation sample of flows from test.parquet.

    Args:
        cfg: Configuration dictionary (loads config.yaml if None).
        test_path: Path to test.parquet (default from config).
        label_map_path: Path to label_map.json (default from config).
        out_path: Path to output JSON file (default artifacts/eval/eval_flows.json).
        seed: Random seed for drawing candidates (default from config eval.seed).
        per_class: Number of samples per class (default from config eval.per_class).
        exclude_rows: Row indices to exclude (default from config eval.exclude_rows).
        force: If True, overwrite existing out_path file; otherwise raises FileExistsError.

    Returns:
        Dict containing selection metadata, indices_by_class, and total indices.

    Raises:
        FileExistsError: If out_path exists and force is False.
        FileNotFoundError: If test_path or label_map_path is missing.
        ValueError: If any class has fewer candidates than per_class.
    """
    if cfg is None:
        cfg = load_config()

    root = get_project_root()
    eval_cfg = cfg.get("eval", {})
    paths_cfg = cfg.get("paths", {})

    if seed is None:
        seed = int(eval_cfg.get("seed", 42))
    if per_class is None:
        per_class = int(eval_cfg.get("per_class", 10))
    if exclude_rows is None:
        exclude_rows = [int(x) for x in eval_cfg.get("exclude_rows", [])]

    if test_path is None:
        test_path = root / paths_cfg.get("processed", "data/processed") / "test.parquet"
    else:
        test_path = Path(test_path)

    if label_map_path is None:
        label_map_path = root / paths_cfg.get("processed", "data/processed") / "label_map.json"
    else:
        label_map_path = Path(label_map_path)

    if out_path is None:
        out_path = root / eval_cfg.get("flows_file", "artifacts/eval/eval_flows.json")
    else:
        out_path = Path(out_path)

    # 1. Enforce overwrite protection
    if out_path.exists() and not force:
        raise FileExistsError(
            f"Evaluation flows file already exists at '{out_path}'. Use --force to overwrite."
        )

    # 2. Check input files exist
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found at: {label_map_path}")

    # 3. Load label map class order (sorted by numeric class ID)
    with open(label_map_path, "r", encoding="utf-8") as f:
        raw_map: dict[str, str] = json.load(f)
    ordered_classes = [(int(k), v) for k, v in sorted(raw_map.items(), key=lambda item: int(item[0]))]

    # 4. Load only the label column of test.parquet
    parquet_file = pq.ParquetFile(test_path)
    label_col = "label" if "label" in parquet_file.schema.names else "Label"
    df_labels = pd.read_parquet(test_path, columns=[label_col])
    labels = df_labels[label_col].to_numpy()

    # 5. Draw per_class rows without replacement using one generator across classes in order
    exclude_set = set(exclude_rows)
    rng = np.random.default_rng(seed)

    indices_by_class: dict[str, list[int]] = {}
    all_indices: list[int] = []
    candidate_counts: dict[str, int] = {}

    for class_id, class_name in ordered_classes:
        # Match candidate rows by class ID or class name string
        match_mask = (labels == class_id) | (labels == class_name)
        matched_indices = np.where(match_mask)[0]
        candidates = np.array([idx for idx in matched_indices if idx not in exclude_set], dtype=np.int64)
        candidate_counts[class_name] = int(len(candidates))

        if len(candidates) < per_class:
            raise ValueError(
                f"Class '{class_name}' (ID {class_id}) has only {len(candidates)} candidates "
                f"after exclusions, which is fewer than required per_class={per_class}."
            )

        drawn = rng.choice(candidates, size=per_class, replace=False)
        drawn_sorted = sorted(int(x) for x in drawn)
        indices_by_class[class_name] = drawn_sorted
        all_indices.extend(drawn_sorted)

    # 6. Checksum and timestamp
    test_sha256 = compute_file_sha256(test_path)
    created_at = datetime.now(timezone.utc).isoformat()

    result_data = {
        "seed": seed,
        "per_class": per_class,
        "exclude_rows": exclude_rows,
        "indices_by_class": indices_by_class,
        "indices": all_indices,
        "candidate_counts": candidate_counts,
        "test_sha256": test_sha256,
        "created_at": created_at,
    }

    # 7. Write evaluation flows file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    return result_data


def main() -> None:
    """CLI entrypoint for selecting evaluation flows."""
    parser = argparse.ArgumentParser(description="Select deterministic evaluation flow indices.")
    parser.add_argument("--test-path", type=str, default=None, help="Path to test.parquet.")
    parser.add_argument("--label-map-path", type=str, default=None, help="Path to label_map.json.")
    parser.add_argument("--out-path", type=str, default=None, help="Path to output eval_flows.json.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for candidate sampling.")
    parser.add_argument("--per-class", type=int, default=None, help="Number of flows to select per class.")
    parser.add_argument(
        "--exclude-rows",
        type=str,
        default=None,
        help="Comma-separated list of row indices to exclude.",
    )
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing evaluation flows file.")
    args = parser.parse_args()

    cfg = load_config()

    exclude_rows: list[int] | None = None
    if args.exclude_rows is not None:
        exclude_rows = [int(x.strip()) for x in args.exclude_rows.split(",") if x.strip()]

    try:
        data = select_flows(
            cfg=cfg,
            test_path=args.test_path,
            label_map_path=args.label_map_path,
            out_path=args.out_path,
            seed=args.seed,
            per_class=args.per_class,
            exclude_rows=exclude_rows,
            force=args.force,
        )
    except FileExistsError as e:
        print(f"Error: {e}")
        return

    out_file = args.out_path or cfg.get("eval", {}).get("flows_file", "artifacts/eval/eval_flows.json")
    print("Selected evaluation flow indices per class:")
    for c_name, c_indices in data["indices_by_class"].items():
        print(f"  {c_name} (n={len(c_indices)}): {c_indices}")
    print(f"Total evaluation flows: {len(data['indices'])}")
    print(f"Saved evaluation flows file to: {out_file}")


if __name__ == "__main__":
    main()

