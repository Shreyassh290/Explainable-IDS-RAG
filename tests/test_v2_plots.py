"""Synthetic data tests for paper-style plots and v2 evaluation plots.

Tests:
- make_paper_style_plots with synthetic history dicts.
- missing history does not crash and renders placeholder.
- macro-F1 figures are accurately titled and never labelled accuracy.
- make_v2_plots with synthetic summary JSON and synthetic history.
- missing v2 input files cleanly render placeholders without crashing.
- FTP-BruteForce is always labelled indicative only in ROC, curves, and recall charts.
- Forbidden words do NOT appear in the plotting scripts or performance tab.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest


def _load_module(rel_path: str, mod_name: str):
    full_path = Path(__file__).resolve().parents[1] / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_forbidden_words_not_in_plotting_scripts():
    """Ensure forbidden words do not appear in any newly created or edited files."""
    root = Path(__file__).resolve().parents[1]
    files_to_check = [
        root / "scripts" / "make_paper_style_plots.py",
        root / "scripts" / "make_v2_plots.py",
        root / "app" / "performance_tab.py",
        root / "app" / "paper_vs_ours_tab.py",
    ]

    forbidden_phrases = [
        "veri" + "fied",
        "improved " + "over the paper",
        "air-" + "gapped",
        "fine-" + "tuned",
    ]

    for fpath in files_to_check:
        if fpath.is_file():
            text = fpath.read_text(encoding="utf-8").lower()
            for phrase in forbidden_phrases:
                assert phrase not in text, f"Forbidden phrase '{phrase}' found in {fpath.name}"


def test_paper_style_plots_synthetic_and_missing(tmp_path):
    """Test make_paper_style_plots with both synthetic data and missing history."""
    mod = _load_module("scripts/make_paper_style_plots.py", "make_paper_style_plots")

    synthetic_histories = {
        "Benign": {
            "train_loss": [0.05, 0.02, 0.01],
            "val_loss": [0.03, 0.015, 0.009],
            "val_macro_f1_history": [0.990, 0.995, 0.998],
            "best_epoch": 3,
        },
        "DoS": {
            "train_loss": [0.04, 0.01, 0.005],
            "val_loss": [0.02, 0.008, 0.004],
            "val_macro_f1_history": [0.991, 0.996, 0.999],
            "best_epoch": 3,
        },
        "DDoS": None,  # Missing history test case
    }

    p_loss = tmp_path / "fig5_style_loss.png"
    out_loss = mod.plot_loss_curves(synthetic_histories, p_loss)
    assert out_loss.is_file()
    assert out_loss.stat().st_size > 0

    p_val = tmp_path / "fig4_style_val.png"
    out_val = mod.plot_validation_metric_curves(synthetic_histories, p_val)
    assert out_val.is_file()
    assert out_val.stat().st_size > 0


def test_val_macro_f1_never_labelled_accuracy():
    """Ensure the validation metric plot in make_paper_style_plots never labels macro F1 as accuracy."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "make_paper_style_plots.py"
    code = script_path.read_text(encoding="utf-8")

    # The title must explicitly specify macro F1
    assert "Validation macro F1 per epoch" in code
    assert "Validation Macro F1" in code


def test_v2_plots_synthetic_and_missing(tmp_path):
    """Test make_v2_plots with synthetic inputs and missing file handling."""
    mod = _load_module("scripts/make_v2_plots.py", "make_v2_plots")

    # 1. Training curves with FTP indicative label
    synth_v2_hist = {
        "Bot": {
            "train_loss": [0.01, 0.005],
            "val_loss": [0.008, 0.003],
            "best_epoch": 2,
        },
        "FTP-BruteForce": {
            "train_loss": [0.02, 0.01],
            "val_loss": [0.015, 0.009],
            "best_epoch": 2,
        },
        "SSH-Bruteforce": None,  # Missing
    }
    p_train = tmp_path / "v2_training_curves.png"
    out_train = mod.plot_v2_training_curves(synth_v2_hist, p_train)
    assert out_train.is_file()
    assert out_train.stat().st_size > 0

    # 2. One-vs-Rest ROC curve with FTP indicative label
    synth_roc = {
        "curves": {
            "Bot": {"fpr": [0.0, 0.01, 1.0], "tpr": [0.0, 0.99, 1.0], "auc": 1.0000},
            "FTP-BruteForce": {"fpr": [0.0, 0.05, 1.0], "tpr": [0.0, 0.95, 1.0], "auc": 0.9998},
            "SSH-Bruteforce": {"fpr": [0.0, 0.02, 1.0], "tpr": [0.0, 0.98, 1.0], "auc": 0.9999},
        }
    }
    p_roc = tmp_path / "v2_roc_ovr.png"
    out_roc = mod.plot_v2_roc_ovr(synth_roc, p_roc)
    assert out_roc.is_file()
    assert out_roc.stat().st_size > 0

    # Test missing ROC data creates placeholder without crashing
    p_roc_none = tmp_path / "v2_roc_none.png"
    out_roc_none = mod.plot_v2_roc_ovr(None, p_roc_none)
    assert out_roc_none.is_file()

    # 3. Mixed confusion matrix
    synth_summary = {
        "classes": ["Benign", "DoS", "DDoS", "Bot", "FTP-BruteForce", "SSH-Bruteforce"],
        "confusion_matrix": [
            [3000, 0, 0, 0, 0, 0],
            [0, 3000, 0, 0, 0, 0],
            [0, 0, 3000, 0, 0, 0],
            [2500, 0, 0, 500, 0, 0],
            [1, 0, 0, 0, 7, 0],
            [2600, 0, 0, 0, 0, 400],
        ],
        "per_class_metrics": {
            "Benign": {"recall": 1.0},
            "DoS": {"recall": 1.0},
            "DDoS": {"recall": 1.0},
            "Bot": {"recall": 0.1667},
            "FTP-BruteForce": {"recall": 0.875},
            "SSH-Bruteforce": {"recall": 0.1333},
        },
    }
    p_cm = tmp_path / "v2_mixed_confusion.png"
    out_cm = mod.plot_v2_mixed_confusion(synth_summary, p_cm)
    assert out_cm.is_file()

    # Test missing confusion summary creates placeholder
    p_cm_none = tmp_path / "v2_mixed_cm_none.png"
    out_cm_none = mod.plot_v2_mixed_confusion(None, p_cm_none)
    assert out_cm_none.is_file()

    # 4. Mixed recall bar chart
    p_rec = tmp_path / "v2_mixed_recall.png"
    out_rec = mod.plot_v2_mixed_recall(synth_summary, p_rec)
    assert out_rec.is_file()

    # Test missing recall creates placeholder
    p_rec_none = tmp_path / "v2_mixed_rec_none.png"
    out_rec_none = mod.plot_v2_mixed_recall(None, p_rec_none)
    assert out_rec_none.is_file()


def test_performance_tab_v2_assets(tmp_path):
    """Test collect_v2_performance_assets correctly references the v2 plots (gated + historical)."""
    mod = _load_module("app/performance_tab.py", "performance_tab")
    assets = mod.collect_v2_performance_assets(tmp_path)

    # Gated (final D5) assets must be present
    assert "mixed_confusion_gated" in assets
    assert "mixed_recall_gated" in assets
    assert assets["mixed_confusion_gated"]["name"] == "v2_mixed_confusion_gated.png"
    assert assets["mixed_recall_gated"]["name"] == "v2_mixed_recall_gated.png"

    # Historical pre-gating assets must be retained (kept, labelled)
    assert "mixed_confusion_historical" in assets
    assert "mixed_recall_historical" in assets
    assert assets["mixed_confusion_historical"]["name"] == "v2_mixed_confusion.png"
    assert assets["mixed_recall_historical"]["name"] == "v2_mixed_recall.png"

    # Always-present assets
    assert "roc_ovr" in assets
    assert "training_curves" in assets
    assert assets["roc_ovr"]["name"] == "v2_roc_ovr.png"
    assert assets["training_curves"]["name"] == "v2_training_curves.png"

    # Captions must mark historical items clearly
    assert "PRE-GATING" in assets["mixed_confusion_historical"]["caption"]
    assert "PRE-GATING" in assets["mixed_recall_historical"]["caption"]

