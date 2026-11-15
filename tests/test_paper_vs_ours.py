"""Synthetic data tests for the Paper vs Ours comparison tab.

Checks:
- Missing paper images produce clean placeholders rather than exceptions.
- Missing metric artifact files cleanly populate 'not computed' values.
- Present metric artifacts populate valid values.
- Missing mixed confusion summary produces 'not computed' for all six classes.
- Present mixed confusion summary properly parses per-class recall.
- Forbidden words do NOT appear anywhere in app/paper_vs_ours_tab.py.
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
        "veri" + "fied",
        "improved " + "over the paper",
        "air-" + "gapped",
        "fine-" + "tuned",
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


def test_missing_mixed_confusion_summary_produces_not_computed(tmp_path):
    """Ensure missing mixed_confusion_summary.json produces 'not computed' for all six classes."""
    mod = _load_tab_module()
    recalls = mod.collect_mixed_confusion_recall(tmp_path)
    expected_classes = ["Benign", "DoS", "DDoS", "Bot", "FTP-BruteForce", "SSH-Bruteforce"]
    assert len(recalls) == 6
    for cls_name in expected_classes:
        assert cls_name in recalls
        assert recalls[cls_name] == "not computed"


def test_present_mixed_confusion_summary_populates_recall(tmp_path):
    """When ONLY the old mixed_confusion_summary.json is present (no gated JSON),
    values should be returned with a [PRE-GATING] prefix so they are never
    silently mistaken for final results."""
    mod = _load_tab_module()
    mixed_dir = tmp_path / "artifacts" / "v2" / "mixed_check"
    mixed_dir.mkdir(parents=True)
    payload = {
        "per_class_metrics": {
            "Benign": {"recall": 0.995},
            "DoS": {"recall": 0.998},
            "DDoS": {"recall": 0.999},
            "Bot": {"recall": 0.950},
            "FTP-BruteForce": {"recall": 0.875},
            "SSH-Bruteforce": {"recall": 0.920},
        }
    }
    (mixed_dir / "mixed_confusion_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    recalls = mod.collect_mixed_confusion_recall(tmp_path)
    # Fallback path: values are prefixed with [PRE-GATING]
    assert "[PRE-GATING]" in recalls["Benign"]
    assert "99.50%" in recalls["Benign"]
    assert "[PRE-GATING]" in recalls["Bot"]
    assert "95.00%" in recalls["Bot"]


def test_gated_json_takes_priority_over_mixed_summary(tmp_path):
    """When eval_negfix_gated.json is present it takes priority; values come without
    [PRE-GATING] prefix and reflect the gated recalls."""
    mod = _load_tab_module()

    # Create the gated JSON (minimal structure matching the real file)
    gated_dir = tmp_path / "artifacts" / "v2b" / "eval"
    gated_dir.mkdir(parents=True)
    gated_payload = {
        "gated_with_ftp": {
            "mixed_sample_15008": {
                "per_class_metrics": {
                    "Benign": {"recall": 0.992, "is_indicative_only": False},
                    "DoS": {"recall": 0.9993, "is_indicative_only": False},
                    "DDoS": {"recall": 1.0, "is_indicative_only": False},
                    "Bot": {"recall": 0.9987, "is_indicative_only": False},
                    "FTP-BruteForce": {"recall": 1.0, "is_indicative_only": True},
                    "SSH-Bruteforce": {"recall": 1.0, "is_indicative_only": False},
                }
            }
        }
    }
    (gated_dir / "eval_negfix_gated.json").write_text(json.dumps(gated_payload), encoding="utf-8")

    # Also create the old JSON (should be ignored when gated JSON exists)
    mixed_dir = tmp_path / "artifacts" / "v2" / "mixed_check"
    mixed_dir.mkdir(parents=True)
    old_payload = {
        "per_class_metrics": {
            "Benign": {"recall": 0.968},
            "Bot": {"recall": 0.0093},
            "SSH-Bruteforce": {"recall": 0.0023},
        }
    }
    (mixed_dir / "mixed_confusion_summary.json").write_text(json.dumps(old_payload), encoding="utf-8")

    recalls = mod.collect_mixed_confusion_recall(tmp_path)

    # Gated values should be returned, no [PRE-GATING] prefix
    assert "[PRE-GATING]" not in recalls["Benign"]
    assert "99.20%" in recalls["Benign"]
    assert "[PRE-GATING]" not in recalls["Bot"]
    assert "99.87%" in recalls["Bot"]
    assert "indicative only" in recalls["FTP-BruteForce"]
    assert "_source" in recalls  # source annotation present


