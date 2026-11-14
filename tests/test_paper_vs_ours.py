"""Synthetic data tests for the Paper vs Ours comparison tab.

Verifies:
- Missing paper images produce clean placeholders rather than exceptions.
- Missing metric artifact files cleanly populate 'not computed' values.
- Present metric artifacts populate valid values.
- Forbidden words ('verified citations', 'improved over the paper', 'air-gapped', 'fine-tuned')
  do NOT appear anywhere in app/paper_vs_ours_tab.py.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pytest


def _load_tab_module():
    tab_path = Path(__file__).resolve().parents[1] / "app" / "paper_vs_ours_tab.py"
    spec = importlib.util.spec_from_file_location("paper_vs_ours_tab", tab_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_forbidden_words_not_in_code():
    """Ensure no forbidden words appear anywhere in app/paper_vs_ours_tab.py."""
    tab_path = Path(__file__).resolve().parents[1] / "app" / "paper_vs_ours_tab.py"
    content = tab_path.read_text(encoding="utf-8").lower()

    forbidden_phrases = [
        "verified citations",
        "improved over the paper",
        "air-gapped",
        "fine-tuned",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in content, f"Forbidden phrase '{phrase}' found in app/paper_vs_ours_tab.py"


def test_missing_paper_images_produce_none(tmp_path):
    """Ensure resolve_paper_image returns None when images are missing, without raising."""
    mod = _load_tab_module()
    paper_dir = tmp_path / "app" / "assets" / "paper"
    paper_dir.mkdir(parents=True)

    res = mod.resolve_paper_image(paper_dir, "fig2_roc_cicids.png")
    assert res is None

    res_double = mod.resolve_paper_image(paper_dir, "fig4_val_accuracy.png")
    assert res_double is None


def test_present_paper_images_resolved(tmp_path):
    """Ensure resolve_paper_image resolves both .png and .png.png files correctly."""
    mod = _load_tab_module()
    paper_dir = tmp_path / "app" / "assets" / "paper"
    paper_dir.mkdir(parents=True)

    (paper_dir / "fig2_roc_cicids.png").write_bytes(b"image_bytes")
    (paper_dir / "fig4_val_accuracy.png.png").write_bytes(b"image_bytes")

    assert mod.resolve_paper_image(paper_dir, "fig2_roc_cicids.png") is not None
    assert mod.resolve_paper_image(paper_dir, "fig4_val_accuracy.png") is not None


def test_missing_metrics_produce_not_computed(tmp_path):
    """Ensure missing artifact files return 'not computed' for all pipeline fields."""
    mod = _load_tab_module()
    metrics = mod.collect_paper_vs_ours_metrics(tmp_path)
    assert len(metrics) == 9

    for row in metrics:
        if row["Metric"] in ["Retrieval Precision@5", "Retrieval Recall@5"]:
            assert row["Ours"] == "not computed"
        else:
            assert row["Ours"] == "not computed"
        assert row["Paper"] != ""
        assert row["Note"] != ""


def test_present_metrics_populate_values(tmp_path):
    """Ensure present metrics artifacts populate corresponding values."""
    mod = _load_tab_module()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    metrics_payload = {
        "classification_report": {"accuracy": 0.99843},
        "roc_aucs": {
            "Benign": 0.99993,
            "DoS": 0.99998,
            "DDoS": 0.99995,
        },
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics_payload), encoding="utf-8")

    eval_dir = artifacts_dir / "eval"
    eval_dir.mkdir()
    retrieval_payload = {
        "rows": [
            {"rr": 1.0, "success": 1},
            {"rr": 0.5, "success": 1},
        ]
    }
    (eval_dir / "retrieval_eval.json").write_text(json.dumps(retrieval_payload), encoding="utf-8")

    rows = mod.collect_paper_vs_ours_metrics(tmp_path)
    rows_by_metric = {r["Metric"]: r for r in rows}

    assert rows_by_metric["CICIDS2018 test accuracy"]["Ours"] == "99.84%"
    assert rows_by_metric["ROC AUC (Benign)"]["Ours"] == "0.9999"
    assert rows_by_metric["ROC AUC (DoS)"]["Ours"] == "1.0000"
    assert rows_by_metric["ROC AUC (DDoS)"]["Ours"] == "1.0000"
    assert rows_by_metric["Retrieval MRR"]["Ours"] == "0.75"
    assert "1.00 (2/2)" in rows_by_metric["Retrieval success rate"]["Ours"]
    assert rows_by_metric["Retrieval Precision@5"]["Ours"] == "not computed"
    assert rows_by_metric["Retrieval Recall@5"]["Ours"] == "not computed"

