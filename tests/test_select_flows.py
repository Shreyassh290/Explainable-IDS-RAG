"""Unit tests for evaluation flow selector (ids_rag.eval.select_flows).

Tests flow selection on synthetic datasets with mock configs:
1. select_flows respects row exclusions.
2. Draws exactly per_class samples per class in label_map order.
3. Deterministic: identical selection for identical seed.
4. Randomness: different selection for different seed.
5. Fails with ValueError if candidates in any class are fewer than per_class.
6. Refuses overwrite with FileExistsError if output exists and force is False.
7. Successfully overwrites if force is True.
8. CLI main() functions as expected.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ids_rag.eval.select_flows import main, select_flows


@pytest.fixture
def synthetic_eval_data(tmp_path: Path) -> dict[str, Any]:
    """Create synthetic test.parquet and label_map.json."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 3 classes: 0: Benign, 1: DoS, 2: DDoS (15 rows each, total 45 rows)
    label_map = {"0": "Benign", "1": "DoS", "2": "DDoS"}
    label_map_file = data_dir / "label_map.json"
    with open(label_map_file, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    labels = [0] * 15 + [1] * 15 + [2] * 15
    df = pd.DataFrame(
        {
            "feature1": np.arange(45, dtype=np.float32),
            "label": labels,
        }
    )
    test_file = data_dir / "test.parquet"
    df.to_parquet(test_file)

    out_file = tmp_path / "artifacts" / "eval" / "eval_flows.json"

    cfg = {
        "eval": {
            "seed": 42,
            "per_class": 10,
            "exclude_rows": [0, 1, 15, 30],
            "flows_file": str(out_file),
        },
        "paths": {
            "processed": str(data_dir),
        },
    }

    return {
        "cfg": cfg,
        "test_file": test_file,
        "label_map_file": label_map_file,
        "out_file": out_file,
        "exclude_rows": [0, 1, 15, 30],
    }


def test_select_flows_synthetic_dataset(synthetic_eval_data: dict[str, Any]) -> None:
    """Test exclusions, balance, determinism, candidate checks, and overwrite protection."""
    test_file = synthetic_eval_data["test_file"]
    label_map_file = synthetic_eval_data["label_map_file"]
    out_file = synthetic_eval_data["out_file"]
    exclude_rows = synthetic_eval_data["exclude_rows"]

    # 1. First run: select 10 per class with exclusions
    result = select_flows(
        cfg=synthetic_eval_data["cfg"],
        test_path=test_file,
        label_map_path=label_map_file,
        out_path=out_file,
        seed=42,
        per_class=10,
        exclude_rows=exclude_rows,
        force=False,
    )

    # Exclusions respected
    for idx in result["indices"]:
        assert idx not in exclude_rows

    # 10 per class
    assert len(result["indices_by_class"]["Benign"]) == 10
    assert len(result["indices_by_class"]["DoS"]) == 10
    assert len(result["indices_by_class"]["DDoS"]) == 10
    assert len(result["indices"]) == 30

    # Output file was written and matches returned dict
    assert out_file.exists()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["indices"] == result["indices"]
    assert "test_sha256" in saved
    assert "created_at" in saved

    # 2. Refuses overwrite without --force
    with pytest.raises(FileExistsError, match="already exists"):
        select_flows(
            cfg=synthetic_eval_data["cfg"],
            test_path=test_file,
            label_map_path=label_map_file,
            out_path=out_file,
            seed=42,
            per_class=10,
            exclude_rows=exclude_rows,
            force=False,
        )

    # 3. Determinism: identical seed produces identical indices
    result_same = select_flows(
        cfg=synthetic_eval_data["cfg"],
        test_path=test_file,
        label_map_path=label_map_file,
        out_path=out_file,
        seed=42,
        per_class=10,
        exclude_rows=exclude_rows,
        force=True,
    )
    assert result_same["indices"] == result["indices"]
    assert result_same["indices_by_class"] == result["indices_by_class"]

    # 4. Different seed produces different indices
    result_diff = select_flows(
        cfg=synthetic_eval_data["cfg"],
        test_path=test_file,
        label_map_path=label_map_file,
        out_path=out_file,
        seed=99,
        per_class=10,
        exclude_rows=exclude_rows,
        force=True,
    )
    assert result_diff["indices"] != result["indices"]

    # 5. Fails if class has fewer candidates than per_class
    # Benign has 15 rows total, minus 2 excluded rows = 13 candidates.
    # Requesting per_class=14 must raise ValueError.
    with pytest.raises(ValueError, match="fewer than required per_class"):
        select_flows(
            cfg=synthetic_eval_data["cfg"],
            test_path=test_file,
            label_map_path=label_map_file,
            out_path=out_file,
            seed=42,
            per_class=14,
            exclude_rows=exclude_rows,
            force=True,
        )


def test_select_flows_cli(synthetic_eval_data: dict[str, Any], monkeypatch: Any) -> None:
    """Test CLI runner with command-line arguments."""
    test_file = synthetic_eval_data["test_file"]
    label_map_file = synthetic_eval_data["label_map_file"]
    out_file = synthetic_eval_data["out_file"]

    # Delete out_file if it exists from previous test
    if out_file.exists():
        out_file.unlink()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_flows",
            "--test-path",
            str(test_file),
            "--label-map-path",
            str(label_map_file),
            "--out-path",
            str(out_file),
            "--seed",
            "42",
            "--per-class",
            "5",
            "--exclude-rows",
            "0,1,2",
        ],
    )

    main()

    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["per_class"] == 5
    assert len(data["indices"]) == 15
    assert not any(idx in [0, 1, 2] for idx in data["indices"])

